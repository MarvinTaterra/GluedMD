"""
03_production_glued.py — 10-walker OPES production run on KOR using GLUED.

Walkers share a single OPES bias via GLUED's multi-walker GPU array mechanism
(all contexts run in the same process on one GPU — no MPI required).

Stopping criterion:
  Each walker drives its own OPESConvergenceReporter with the dimensionless
  ``rct_relative`` criterion (|Δrct|/max(|rct|, kT) < tol over a sliding
  window of consecutive checks). Default: tol=0.01, min_consecutive_passes=3,
  min_kernels=50 — robust against early-run noise. After all walkers
  converge, sampling continues for POST_CONV_STEPS more steps so the
  reweighted PMF has a well-defined tail.

  Hard cap: 25 000 000 steps (50 ns) per walker.

Note: rct is now PLUMED-style ``kT·log(<w>)`` (c(t) reweighting indicator).
This is a dimensionful kJ/mol quantity but small in magnitude (typically a
few kJ/mol), so the relative criterion is much more robust than an absolute
0.1 kJ/mol heuristic.

Outputs (in output/production/walkerN/):
  COLVAR.dat       — A100, opes.bias, opes.rct, opes.zed every 500 steps
  traj.dcd         — coordinates every 2 500 steps (5 ps)
  convergence.log  — full reporter trace + convergence/done events

Usage:
  python 03_production_glued.py [--input-dir PATH] [--equil-dir PATH]
                                 [--n-walkers N] [--output-dir PATH]
"""

import argparse, os, sys, time
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# ---------------------------------------------------------------------------
# A100 CV definition
# ---------------------------------------------------------------------------
A100_PAIRS  = [(379,4484),(809,1412),(1496,1940),(3232,3569),(3979,4169)]
A100_COEFFS = [-144.3, -76.2, 91.1, -63.2, -52.2]
A100_CONST  =  278.88

# Production settings
TEMP            = 303.15
DT_PS           = 0.002
OPES_SIGMA      = 2.0         # A100 score units (range ≈ 0–100); tune after pilot
OPES_GAMMA      = 20.28       # biasfactor = BARRIER/kT = 50/2.479 ≈ 20.18 → round up
OPES_PACE       = 500
OPES_MAX_KER    = 200_000
COLVAR_STRIDE   = 500         # every 1 ps
DCD_STRIDE      = 2_500       # every 5 ps
CHECK_INTERVAL  = 10_000      # convergence check every 20 ps
POST_CONV_STEPS = 500_000     # 1 ns of post-convergence sampling
MAX_STEPS       = 25_000_000  # 50 ns hard cap per walker

# OPESConvergenceReporter settings (PLUMED-style dimensionless rct criterion)
CONV_CRITERION  = "rct_relative"
CONV_TOL        = 0.01        # |Δrct|/max(|rct|,kT) — dimensionless
CONV_MIN_PASSES = 3           # consecutive passing checks before declaring converged
CONV_MIN_KER    = 50          # don't start checking until ≥ N kernels deposited

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",
                   default=os.environ.get("INPUT_DIR", ""))
    p.add_argument("--equil-dir",    default="output/equil")
    p.add_argument("--output-dir",   default="output/production")
    p.add_argument("--n-walkers",    type=int, default=10)
    p.add_argument("--seeds-dir",
                   help="Optional directory containing walker{i}_seed.chk "
                        "files produced by seed_walkers.py. When set, each "
                        "walker loads its own seed for stratified starting "
                        "positions; otherwise all walkers start from "
                        "equil_final.chk (legacy behaviour).")
    return p.parse_args()


def find_input_dir(hint):
    for c in [hint,
              os.path.join(os.path.dirname(__file__), "input"),
              "/data/charmm-gui-7823302191/openmm",
              "/mnt/c/Users/Marvin/Desktop/Glued Benchmark/charmm-gui-7823302191/openmm"]:
        if c and os.path.isfile(os.path.join(c, "step5_input.psf")):
            return os.path.abspath(c)
    raise FileNotFoundError("step5_input.psf not found — pass --input-dir")


def load_charmm_params(input_dir):
    toppar = os.path.join(input_dir, "toppar.str")
    base   = os.path.dirname(os.path.abspath(toppar))
    files  = [os.path.normpath(os.path.join(base, l.strip()))
              for l in open(toppar)
              if l.strip() and not l.startswith("!")]
    return app.CharmmParameterSet(*[f for f in files if os.path.exists(f)])


def make_production_system(psf, params):
    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        switchDistance=1.0 * unit.nanometers,
        constraints=app.HBonds,
        ewaldErrorTolerance=0.0005,
    )
    system.addForce(mm.MonteCarloMembraneBarostat(
        1.0 * unit.bar, 0.0 * unit.bar * unit.nanometers,
        TEMP * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFree, 100,
    ))
    return system


def add_a100_opes(force, walker_idx):
    """Add A100 CV + OPES bias to a glued.Force. Returns (a100_cv_idx, bias_idx)."""
    d = [force.add_distance(list(pair)) for pair in A100_PAIRS]
    expr = " + ".join(f"({c})*cv{i}" for i, c in enumerate(A100_COEFFS))
    expr += f" + {A100_CONST}"
    a100 = force.add_expression(expr, d)
    bias = force.add_opes([a100], sigma=OPES_SIGMA, gamma=OPES_GAMMA,
                          pace=OPES_PACE, max_kernels=OPES_MAX_KER,
                          mode='metad')   # 'explore' is the alternative
    return a100, bias


# ---------------------------------------------------------------------------
# COLVAR writer (plain text, 1 file per walker)
# ---------------------------------------------------------------------------

class COLVARWriter:
    def __init__(self, path, force, a100_idx, bias_idx):
        self._f     = open(path, "w")
        self._force = force
        self._aidx  = a100_idx
        self._bidx  = bias_idx
        print("#! FIELDS time A100 opes.bias opes.rct opes.zed", file=self._f, flush=True)

    def write(self, context, time_ps):
        cvs     = self._force.getLastCVValues(context)
        metrics = self._force.getOPESMetrics(context, self._bidx)
        a100    = cvs[self._aidx]
        bias    = metrics[0]  # Note: getLastCVValues returns cv values; bias comes from energy
        # getOPESMetrics returns [zed, rct, nker, neff]
        zed, rct = metrics[0], metrics[1]
        # bias energy from getState
        state = context.getState(getEnergy=True, groups={context.getSystem().getNumForces()-1})
        bias_e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        print(f" {time_ps:14.5f}  {a100:14.6f}  {bias_e:14.6f}  {rct:14.6f}  {zed:14.6f}",
              file=self._f, flush=True)

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Convergence tracking via OPESConvergenceReporter
#
# The reporter expects an OpenMM-Simulation-shaped object with `.context`
# and `.currentStep`. The benchmark runs raw `Context` + `LangevinMiddle-
# Integrator` instances (no Simulation), so we wrap each walker's context
# with a tiny adapter and call `reporter.report(sim, None)` manually after
# each batch of steps.
# ---------------------------------------------------------------------------

from OPESConvergenceReporter import OPESConvergenceReporter   # noqa: E402


class _WalkerSim:
    """Minimal openmm.Simulation surface for OPESConvergenceReporter."""
    def __init__(self, ctx):
        self.context     = ctx
        self.currentStep = 0
        self.reporters   = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    input_dir  = find_input_dir(args.input_dir)
    equil_dir  = args.equil_dir
    prod_dir   = args.output_dir
    n_walkers  = args.n_walkers
    os.makedirs(prod_dir, exist_ok=True)

    chk_path = os.path.join(equil_dir, "equil_final.chk")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(
            f"Equilibration checkpoint not found: {chk_path}\n"
            "Run 01_minimize_equilibrate.py first."
        )

    # Resolve per-walker seed checkpoints. With --seeds-dir set, each walker
    # loads its own walker{i}_seed.chk (strategy-2 stratified seeding). Any
    # missing slot falls back to equil_final.chk — useful when seeding only a
    # subset of the production walkers.
    def _seed_path_for(w):
        if args.seeds_dir:
            cand = os.path.join(args.seeds_dir, f"walker{w}_seed.chk")
            if os.path.exists(cand):
                return cand
        return chk_path
    per_walker_chk = [_seed_path_for(w) for w in range(n_walkers)]
    using_seeds = any(p != chk_path for p in per_walker_chk)

    # Best platform
    platform = None
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = mm.Platform.getPlatformByName(name)
            break
        except mm.OpenMMException:
            pass
    print(f"Platform  : {platform.getName()}")
    print(f"Walkers   : {n_walkers}")
    print("Seeds     : "
          + ("stratified ("
             + ", ".join(os.path.basename(p) for p in per_walker_chk) + ")"
             if using_seeds else "identical (equil_final.chk for all)"))
    print(f"Max steps : {MAX_STEPS:,} ({MAX_STEPS * DT_PS / 1000:.0f} ns)")
    print(f"Conv      : criterion={CONV_CRITERION}  tol={CONV_TOL}  "
          f"min_passes={CONV_MIN_PASSES}  min_kernels={CONV_MIN_KER}")
    print(f"Post-conv : {POST_CONV_STEPS:,} steps "
          f"({POST_CONV_STEPS * DT_PS / 1000:.1f} ns) per walker")

    try:
        import glued
    except ImportError:
        print("ERROR: GLUED not installed. Build and install the plugin first.")
        sys.exit(1)

    print("\nLoading topology + parameters ...")
    psf    = app.CharmmPsfFile(os.path.join(input_dir, "step5_input.psf"))
    params = load_charmm_params(input_dir)
    # CharmmPsfFile needs setBox() before createSystem can use PME. The PDB
    # has no CRYST1 line, so reuse the equilibration script's parser which
    # reads box dimensions from CHARMM-GUI's step3_size.str. The actual run-
    # time box will be overwritten by ctx.loadCheckpoint() below.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    _equil = import_module("01_minimize_equilibrate")
    _equil._set_psf_box(psf, input_dir)
    # Pre-read each walker's seed once. Multiple walkers will share the same
    # bytes object when they fall back to equil_final.chk, so this allocates
    # only as many distinct buffers as there are distinct seeds.
    _chk_cache = {}
    chk_bytes_per_walker = []
    for p in per_walker_chk:
        if p not in _chk_cache:
            with open(p, "rb") as f:
                _chk_cache[p] = f.read()
        chk_bytes_per_walker.append(_chk_cache[p])

    # ── Build all walker contexts ────────────────────────────────────────────
    walkers = []
    for w in range(n_walkers):
        wdir = os.path.join(prod_dir, f"walker{w}")
        os.makedirs(wdir, exist_ok=True)

        system     = make_production_system(psf, params)
        force      = glued.Force(pbc=True, temperature=TEMP)
        a100_idx, bias_idx = add_a100_opes(force, w)
        system.addForce(force)

        integrator = mm.LangevinMiddleIntegrator(
            TEMP * unit.kelvin, 1.0 / unit.picoseconds, DT_PS * unit.picoseconds)
        ctx = mm.Context(system, integrator, platform)
        ctx.loadCheckpoint(chk_bytes_per_walker[w])

        # DCD reporter
        dcd = app.DCDFile(
            open(os.path.join(wdir, "traj.dcd"), "wb"),
            psf.topology, DT_PS * unit.picoseconds,
        )

        colvar = COLVARWriter(os.path.join(wdir, "COLVAR.dat"), force, a100_idx, bias_idx)
        logf   = open(os.path.join(wdir, "convergence.log"), "w")
        sim    = _WalkerSim(ctx)
        reporter = OPESConvergenceReporter(
            force, bias_idx=bias_idx,
            criterion=CONV_CRITERION,
            tol=CONV_TOL,
            check_interval=CHECK_INTERVAL,
            min_consecutive_passes=CONV_MIN_PASSES,
            min_kernels=CONV_MIN_KER,
            post_convergence_steps=POST_CONV_STEPS,
            file=logf, verbose=True,
        )

        walkers.append(dict(ctx=ctx, integ=integrator, force=force,
                            a100_idx=a100_idx, bias_idx=bias_idx,
                            dcd=dcd, colvar=colvar, sim=sim,
                            reporter=reporter, logf=logf, w=w))

    # Wire multi-walker shared bias (walker 0 is primary)
    primary_ptrs = walkers[0]["force"].getMultiWalkerPtrs(
        walkers[0]["ctx"], walkers[0]["bias_idx"]
    )
    for wd in walkers[1:]:
        wd["force"].setMultiWalkerPtrs(wd["ctx"], wd["bias_idx"], primary_ptrs)
    print(f"\nMulti-walker bias wired (primary = walker 0, {n_walkers-1} secondaries).")

    # Print initial A100 values to confirm indices are correct
    print("\nInitial A100 values (should be ~0–30 for active-state 6B73 + agonist):")
    for wd in walkers[:3]:
        # Warm up 2 steps to initialise cvValuesReady
        wd["integ"].step(2)
        cv = wd["force"].getLastCVValues(wd["ctx"])
        print(f"  Walker {wd['w']}: A100 = {cv[wd['a100_idx']]:.2f}")

    # ── Production loop ──────────────────────────────────────────────────────
    print(f"\nStarting production run — stepping {CHECK_INTERVAL:,} steps per batch ...\n")
    steps_done = 0
    t_start    = time.time()

    while steps_done < MAX_STEPS:
        batch = min(CHECK_INTERVAL, MAX_STEPS - steps_done)

        # Step all walkers
        for wd in walkers:
            if not wd["reporter"].done:
                wd["integ"].step(batch)

        steps_done += batch
        time_ps     = steps_done * DT_PS

        # Log COLVAR/DCD and run reporter for each walker
        all_done = True
        for wd in walkers:
            if wd["reporter"].done:
                continue
            all_done = False
            ctx = wd["ctx"]

            # COLVAR output
            if steps_done % COLVAR_STRIDE == 0:
                wd["colvar"].write(ctx, time_ps)

            # DCD output
            if steps_done % DCD_STRIDE == 0:
                state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                wd["dcd"].writeModel(state.getPositions())

            # Convergence check via OPESConvergenceReporter (auto-handles
            # rct_relative criterion, multi-pass confirmation, post-conv
            # window, and warm-up gates).
            wd["sim"].currentStep = steps_done
            wd["reporter"].report(wd["sim"], None)

        elapsed = time.time() - t_start
        n_conv  = sum(1 for wd in walkers if wd["reporter"].converged)
        n_done  = sum(1 for wd in walkers if wd["reporter"].done)
        print(f"  step {steps_done:>10,}  ({time_ps/1000:.3f} ns)"
              f"  converged: {n_conv}/{n_walkers}"
              f"  done: {n_done}/{n_walkers}"
              f"  elapsed: {elapsed/3600:.2f}h")

        if all_done:
            print("\nAll walkers completed post-convergence sampling — stopping.")
            break

    # Cleanup
    for wd in walkers:
        wd["colvar"].close()
        wd["logf"].close()

    # Write per-walker summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'Walker':<8} {'Conv step':>12} {'Conv time (ns)':>16} {'Done':>6}")
    for wd in walkers:
        r  = wd["reporter"]
        cs = r.converged_at_step or steps_done
        print(f"  {wd['w']:<6d}  {cs:>12,}  {cs * DT_PS / 1000:>16.3f}  "
              f"{'yes' if r.done else 'no':>6}")

    total_time = time.time() - t_start
    total_ns   = n_walkers * steps_done * DT_PS / 1000
    print(f"\nTotal sampling: {total_ns:.1f} ns  |  Wall time: {total_time/3600:.2f}h")
    print(f"Output: {prod_dir}/walker*/")


if __name__ == "__main__":
    main()
