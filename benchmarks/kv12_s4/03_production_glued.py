"""
03_production_glued.py — multi-walker OPES Explore production on Kv1.2 S4.

Reproduces the S4 voltage-sensor gating motion observed by Bjelkmar et al.
(PLoS Comput Biol 5:e1000289, 2009) with OPES Explore instead of an
applied electric field. CV: S4 Cα center-of-mass Z-displacement relative
to the VSD reference frame (S1/S2/S3 anchor Cα's of the same subunit) —
the same coordinate Bjelkmar plots in his Figure 4.

Stopping criterion:
  Each walker drives its own OPESConvergenceReporter with the
  ``rct_relative`` criterion (|Δrct|/max(|rct|, kT) < tol). After all
  walkers converge, sampling continues for POST_CONV_STEPS more steps
  so the reweighted PMF has a well-defined tail.

  Hard cap: 25 000 000 steps × dt = 50 ns per walker (at dt=2 fs).

Outputs (in output/production/walkerN/):
  COLVAR.dat       — time, S4-Z (nm), opes.bias, opes.rct, opes.zed
  traj.dcd         — coordinates every 5 ps
  convergence.log  — reporter trace + convergence/done events
"""

import argparse, os, sys, time
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# ---------------------------------------------------------------------------
# CV definition (chain A = first PROA segment after CHARMM-GUI processing)
# ---------------------------------------------------------------------------
# S4 gating arginines: R293, R296, R299, R305  (Bjelkmar R294/297/300/306)
# VSD anchors:        S176, C229, I260  (S1/S2/S3 helix midpoints)
# All indices below are 0-based (OpenMM convention) and were verified by
# parsing step5_input.pdb for the first PROA segment.
S4_CA  = [2153, 2213, 2275, 2390]
VSD_CA = [263, 1073, 1584]

# Production settings
TEMP            = 310.0       # K — Bjelkmar's setpoint
DT_PS           = 0.002       # 2 fs without HMR; bump to 0.004 with HMR (todo)
OPES_MODE       = 'explore'   # EXPLORE: barrier unknown, plain KDE
OPES_GAMMA      = 10.0        # biasfactor; spec recommends 10 for ~12 kT barriers
OPES_SIGMA      = 0.05        # nm — Bjelkmar S4 equilibrium fluctuation ~0.5 Å
OPES_SIGMA_MIN  = 0.005       # nm — allow narrow features
OPES_PACE       = 500
OPES_MAX_KER    = 10_000
COLVAR_STRIDE   = 500         # 1 ps
DCD_STRIDE      = 2_500       # 5 ps
CHECK_INTERVAL  = 10_000      # convergence check every 20 ps
POST_CONV_STEPS = 500_000     # 1 ns of post-conv sampling
MAX_STEPS       = 25_000_000  # 50 ns hard cap per walker

# OPESConvergenceReporter
CONV_CRITERION  = "rct_relative"   # robust across variants; switch to
                                   # 'neff_rate' for EXPLORE if rct is noisy
CONV_TOL        = 0.01
CONV_MIN_PASSES = 3
CONV_MIN_KER    = 50

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",
                   default=os.environ.get("INPUT_DIR", ""))
    p.add_argument("--equil-dir",    default="output/equil")
    p.add_argument("--output-dir",   default="output/production")
    p.add_argument("--n-walkers",    type=int, default=4)
    p.add_argument("--seeds-dir",
                   help="Optional directory of walker{i}_seed.chk files from "
                        "seed_walkers.py for stratified starting positions.")
    return p.parse_args()


def find_input_dir(hint):
    candidates = [
        hint,
        os.path.join(os.path.dirname(__file__), "input"),
        "/data/charmm-gui-7851356233/openmm",
        "/mnt/c/Users/Marvin/Desktop/Glued Benchmark/KV1.2/charmm-gui-7851356233/openmm",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "step5_input.psf")):
            return os.path.abspath(c)
    raise FileNotFoundError("step5_input.psf not found — pass --input-dir")


def load_charmm_params(input_dir):
    toppar = os.path.join(input_dir, "toppar.str")
    base   = os.path.dirname(os.path.abspath(toppar))
    files  = [os.path.normpath(os.path.join(base, ln.strip()))
              for ln in open(toppar)
              if ln.strip() and not ln.startswith("!")]
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


def add_s4_z_cv(force):
    """Add the S4 Z-displacement CV (S4 Cα-COM z minus VSD anchor-COM z) to
    a glued.Force. Returns the CV index of the resulting scalar.

    Implementation: GLUED has add_position(atom, component=2) for the z-
    coordinate of a single atom but no native group-COM-z primitive. Since
    Cα masses are all 12.011 amu, the COM-z of a Cα group is exactly the
    arithmetic mean — easy to express via add_expression. We register one
    add_position CV per S4 Cα and per VSD Cα, then combine.
    """
    zs = [force.add_position(idx, component=2) for idx in S4_CA]
    zv = [force.add_position(idx, component=2) for idx in VSD_CA]
    # cv0..cv3 = S4 Cα Z; cv4..cv6 = VSD Cα Z
    expr = ("(" + " + ".join(f"cv{i}" for i in range(len(S4_CA))) + f")/{len(S4_CA)}.0"
            " - "
            "(" + " + ".join(f"cv{i+len(S4_CA)}" for i in range(len(VSD_CA))) + f")/{len(VSD_CA)}.0")
    return force.add_expression(expr, zs + zv)


def add_s4_opes(force):
    """Build the S4 CV + OPES bias on a fresh glued.Force.
    Returns (cv_idx, bias_idx)."""
    cv = add_s4_z_cv(force)
    bias = force.add_opes(
        [cv], sigma=OPES_SIGMA, gamma=OPES_GAMMA,
        sigma_min=OPES_SIGMA_MIN, pace=OPES_PACE,
        max_kernels=OPES_MAX_KER, mode=OPES_MODE,
    )
    return cv, bias


# ---------------------------------------------------------------------------
# COLVAR writer
# ---------------------------------------------------------------------------

class COLVARWriter:
    def __init__(self, path, force, cv_idx, bias_idx):
        self._f     = open(path, "w")
        self._force = force
        self._cidx  = cv_idx
        self._bidx  = bias_idx
        print("#! FIELDS time s4_z opes.bias opes.rct opes.zed",
              file=self._f, flush=True)

    def write(self, context, time_ps):
        cvs     = self._force.getLastCVValues(context)
        metrics = self._force.getOPESMetrics(context, self._bidx)
        s4_z    = cvs[self._cidx]
        state   = context.getState(getEnergy=True,
                                   groups={context.getSystem().getNumForces() - 1})
        bias_e  = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        zed, rct = metrics[0], metrics[1]
        print(f" {time_ps:14.5f}  {s4_z:14.6f}  {bias_e:14.6f}  {rct:14.6f}  {zed:14.6f}",
              file=self._f, flush=True)

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Convergence tracking via OPESConvergenceReporter
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
    args      = parse_args()
    input_dir = find_input_dir(args.input_dir)
    equil_dir = args.equil_dir
    prod_dir  = args.output_dir
    n_walkers = args.n_walkers
    os.makedirs(prod_dir, exist_ok=True)

    chk_path = os.path.join(equil_dir, "equil_final.chk")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(
            f"Equilibration checkpoint not found: {chk_path}\n"
            "Run 01_minimize_equilibrate.py first."
        )

    def _seed_path_for(w):
        if args.seeds_dir:
            cand = os.path.join(args.seeds_dir, f"walker{w}_seed.chk")
            if os.path.exists(cand):
                return cand
        return chk_path
    per_walker_chk = [_seed_path_for(w) for w in range(n_walkers)]
    using_seeds = any(p != chk_path for p in per_walker_chk)

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
    print(f"Mode      : OPES_{OPES_MODE.upper()}  γ={OPES_GAMMA}  "
          f"σ={OPES_SIGMA} nm  pace={OPES_PACE}")
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
    # Reuse the equilibration script's CHARMM-GUI box parser.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    _equil = import_module("01_minimize_equilibrate")
    _equil._set_psf_box(psf, input_dir)

    _chk_cache = {}
    chk_bytes_per_walker = []
    for p in per_walker_chk:
        if p not in _chk_cache:
            with open(p, "rb") as f:
                _chk_cache[p] = f.read()
        chk_bytes_per_walker.append(_chk_cache[p])

    walkers = []
    for w in range(n_walkers):
        wdir = os.path.join(prod_dir, f"walker{w}")
        os.makedirs(wdir, exist_ok=True)

        system  = make_production_system(psf, params)
        force   = glued.Force(pbc=True, temperature=TEMP)
        cv_idx, bias_idx = add_s4_opes(force)
        system.addForce(force)

        integ = mm.LangevinMiddleIntegrator(
            TEMP * unit.kelvin, 1.0 / unit.picoseconds, DT_PS * unit.picoseconds)
        ctx = mm.Context(system, integ, platform)
        ctx.loadCheckpoint(chk_bytes_per_walker[w])

        dcd = app.DCDFile(
            open(os.path.join(wdir, "traj.dcd"), "wb"),
            psf.topology, DT_PS * unit.picoseconds,
        )
        colvar = COLVARWriter(os.path.join(wdir, "COLVAR.dat"), force, cv_idx, bias_idx)
        logf   = open(os.path.join(wdir, "convergence.log"), "w")
        sim    = _WalkerSim(ctx)
        reporter = OPESConvergenceReporter(
            force, bias_idx=bias_idx,
            criterion=CONV_CRITERION, tol=CONV_TOL,
            check_interval=CHECK_INTERVAL,
            min_consecutive_passes=CONV_MIN_PASSES,
            min_kernels=CONV_MIN_KER,
            post_convergence_steps=POST_CONV_STEPS,
            file=logf, verbose=True,
        )

        walkers.append(dict(ctx=ctx, integ=integ, force=force,
                            cv_idx=cv_idx, bias_idx=bias_idx,
                            dcd=dcd, colvar=colvar, sim=sim,
                            reporter=reporter, logf=logf, w=w))

    # Multi-walker B2 wiring.
    primary = walkers[0]["force"].getMultiWalkerPtrs(
        walkers[0]["ctx"], walkers[0]["bias_idx"])
    for wd in walkers[1:]:
        wd["force"].setMultiWalkerPtrs(wd["ctx"], wd["bias_idx"], primary)
    print(f"\nMulti-walker bias wired (primary=0, {n_walkers-1} secondaries).")

    # Sanity print
    print("\nInitial S4 Z-displacement (chain-A, equilibrated):")
    for wd in walkers[:min(4, n_walkers)]:
        wd["integ"].step(2)
        cv = wd["force"].getLastCVValues(wd["ctx"])
        print(f"  Walker {wd['w']}: s4_z = {cv[wd['cv_idx']]:.3f} nm")

    print(f"\nStarting production — stepping {CHECK_INTERVAL:,} steps per batch ...\n")
    steps_done = 0
    t_start    = time.time()
    while steps_done < MAX_STEPS:
        batch = min(CHECK_INTERVAL, MAX_STEPS - steps_done)
        for wd in walkers:
            if not wd["reporter"].done:
                wd["integ"].step(batch)
        steps_done += batch
        time_ps = steps_done * DT_PS

        all_done = True
        for wd in walkers:
            if wd["reporter"].done:
                continue
            all_done = False
            ctx = wd["ctx"]
            if steps_done % COLVAR_STRIDE == 0:
                wd["colvar"].write(ctx, time_ps)
            if steps_done % DCD_STRIDE == 0:
                state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                wd["dcd"].writeModel(state.getPositions())
            wd["sim"].currentStep = steps_done
            wd["reporter"].report(wd["sim"], None)

        elapsed = time.time() - t_start
        n_conv = sum(1 for wd in walkers if wd["reporter"].converged)
        n_done = sum(1 for wd in walkers if wd["reporter"].done)
        print(f"  step {steps_done:>10,}  ({time_ps/1000:.3f} ns)"
              f"  converged: {n_conv}/{n_walkers}"
              f"  done: {n_done}/{n_walkers}"
              f"  elapsed: {elapsed/3600:.2f}h")

        if all_done:
            print("\nAll walkers reached post-convergence — stopping.")
            break

    for wd in walkers:
        wd["colvar"].close()
        wd["logf"].close()

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
