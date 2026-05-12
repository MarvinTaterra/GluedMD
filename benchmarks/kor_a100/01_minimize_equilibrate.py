"""
01_minimize_equilibrate.py — CHARMM-GUI 6-stage equilibration for the KOR benchmark.

Reproduces the CHARMM-GUI OpenMM equilibration protocol exactly:
  Stage 6.1: 5 000-step minimization + 125 ps NVT, dt=1 fs, heavy restraints
  Stage 6.2: 125 ps NVT, dt=1 fs, medium restraints
  Stage 6.3: 125 ps NPT (membrane barostat), dt=1 fs, medium restraints
  Stage 6.4: 500 ps NPT, dt=2 fs, light restraints
  Stage 6.5: 500 ps NPT, dt=2 fs, very light restraints
  Stage 6.6: 500 ps NPT, dt=2 fs, backbone-only restraints

Total: ~1.875 ns.  Each stage saves a checkpoint so the run is restartable.

Usage:
  python 01_minimize_equilibrate.py [--input-dir PATH] [--output-dir PATH] [--skip-if-done]
"""

import argparse, math, os, shutil, sys, time
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# ---------------------------------------------------------------------------
# CLI / path resolution
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",
                   default=os.environ.get("INPUT_DIR", ""),
                   help="Path to CHARMM-GUI openmm/ directory")
    p.add_argument("--output-dir", default="output/equil")
    p.add_argument("--skip-if-done", action="store_true",
                   help="Exit immediately if equil_final.chk already exists")
    return p.parse_args()


def find_input_dir(hint):
    candidates = [
        hint,
        os.path.join(os.path.dirname(__file__), "input"),
        "/data/charmm-gui-7823302191/openmm",
        "/mnt/c/Users/Marvin/Desktop/Glued Benchmark/charmm-gui-7823302191/openmm",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "step5_input.psf")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "step5_input.psf not found. Pass --input-dir or set INPUT_DIR."
    )


def best_platform():
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except mm.OpenMMException:
            pass
    raise RuntimeError("No OpenMM platform available")

# ---------------------------------------------------------------------------
# CHARMM parameter loading
# ---------------------------------------------------------------------------

def load_charmm_params(input_dir):
    toppar = os.path.join(input_dir, "toppar.str")
    base   = os.path.dirname(os.path.abspath(toppar))
    files, missing = [], []
    with open(toppar) as f:
        for line in f:
            fname = line.strip()
            if fname and not fname.startswith("!") and not fname.startswith("#"):
                path = os.path.normpath(os.path.join(base, fname))
                if os.path.exists(path):
                    files.append(path)
                else:
                    missing.append(path)
    if missing:
        print(f"  [warn] {len(missing)} parameter files not found (first: {missing[0]})")
    print(f"  Loaded {len(files)} CHARMM parameter files")
    return app.CharmmParameterSet(*files)

# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------

def _read_box_from_charmm_gui(input_dir):
    """Return (a, b, c, α, β, γ) from CHARMM-GUI's step3_size.str.

    CHARMM-GUI generated PDBs typically lack a CRYST1 record; the canonical
    box dimensions live in `<jobdir>/step3_size.str` (the openmm/ directory
    is a child of the jobdir, so we walk up one level).
    """
    candidates = [
        os.path.join(input_dir, "..", "step3_size.str"),
        os.path.join(input_dir, "step3_size.str"),
    ]
    str_path = next((p for p in candidates if os.path.exists(p)), None)
    if str_path is None:
        raise FileNotFoundError(
            "step3_size.str not found near {}; cannot infer box "
            "dimensions for this CHARMM-GUI system.".format(input_dir))

    vals = {}
    with open(str_path) as f:
        for line in f:
            tokens = line.strip().split()
            # Expected form:  SET <KEY> = <VALUE>
            if len(tokens) >= 4 and tokens[0].upper() == "SET" and tokens[2] == "=":
                vals[tokens[1].upper()] = tokens[3]
    a = float(vals["A"]) / 10.0   # Å → nm
    b = float(vals["B"]) / 10.0
    c = float(vals["C"]) / 10.0
    alpha = float(vals["ALPHA"])
    beta  = float(vals["BETA"])
    gamma = float(vals["GAMMA"])
    return a, b, c, alpha, beta, gamma


def _set_psf_box(psf, input_dir):
    """Apply box dimensions from `step3_size.str` to a CharmmPsfFile.

    `psf.setBox(a, b, c, α, β, γ)` is what `createSystem` actually consults;
    setting `topology.setPeriodicBoxVectors` alone is NOT enough.
    """
    a, b, c, alpha, beta, gamma = _read_box_from_charmm_gui(input_dir)
    psf.setBox(a * unit.nanometer, b * unit.nanometer, c * unit.nanometer,
               alpha * unit.degrees, beta * unit.degrees, gamma * unit.degrees)
    return a, b, c, alpha, beta, gamma


def build_system(psf, params, barostat=False, temp=303.15):
    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        switchDistance=1.0 * unit.nanometers,
        constraints=app.HBonds,
        ewaldErrorTolerance=0.0005,
    )
    if barostat:
        system.addForce(mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            0.0 * unit.bar * unit.nanometers,
            temp * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFree,
            100,
        ))
    return system

# ---------------------------------------------------------------------------
# Restraints
# ---------------------------------------------------------------------------

def add_restraints(system, positions, input_dir,
                   fc_bb, fc_sc, fc_lpos, fc_ldih):
    prot_file  = os.path.join(input_dir, "restraints", "prot_pos.txt")
    lipid_file = os.path.join(input_dir, "restraints", "lipid_pos.txt")
    dihe_file  = os.path.join(input_dir, "restraints", "dihe.txt")

    if fc_bb > 0 or fc_sc > 0:
        bb_f = mm.CustomExternalForce("0.5*k_bb*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        bb_f.addGlobalParameter("k_bb", fc_bb)
        bb_f.addPerParticleParameter("x0")
        bb_f.addPerParticleParameter("y0")
        bb_f.addPerParticleParameter("z0")
        sc_f = mm.CustomExternalForce("0.5*k_sc*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        sc_f.addGlobalParameter("k_sc", fc_sc)
        sc_f.addPerParticleParameter("x0")
        sc_f.addPerParticleParameter("y0")
        sc_f.addPerParticleParameter("z0")
        with open(prot_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                idx = int(parts[0])
                pos = positions[idx].value_in_unit(unit.nanometers)
                (bb_f if parts[1] == "BB" else sc_f).addParticle(idx, pos)
        if fc_bb > 0:
            system.addForce(bb_f)
        if fc_sc > 0:
            system.addForce(sc_f)

    if fc_lpos > 0:
        lp_f = mm.CustomExternalForce("0.5*k_lpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        lp_f.addGlobalParameter("k_lpos", fc_lpos)
        lp_f.addPerParticleParameter("x0")
        lp_f.addPerParticleParameter("y0")
        lp_f.addPerParticleParameter("z0")
        with open(lipid_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    idx = int(line)
                    pos = positions[idx].value_in_unit(unit.nanometers)
                    lp_f.addParticle(idx, pos)
        system.addForce(lp_f)

    if fc_ldih > 0:
        dh_f = mm.CustomTorsionForce(
            "0.5*k_ldih*min(dtheta,2*pi-dtheta)^2;"
            "dtheta=abs(theta-theta0);"
            "pi=3.14159265358979"
        )
        dh_f.addGlobalParameter("k_ldih", fc_ldih)
        dh_f.addPerTorsionParameter("theta0")
        with open(dihe_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                a1, a2, a3, a4 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                theta0 = float(parts[4]) * math.pi / 180.0
                dh_f.addTorsion(a1, a2, a3, a4, [theta0])
        system.addForce(dh_f)

# ---------------------------------------------------------------------------
# Single stage runner
# ---------------------------------------------------------------------------

def run_stage(label, psf, params, positions, box_vectors, input_dir,
              nstep, dt_ps, gen_vel, barostat,
              fc_bb, fc_sc, fc_lpos, fc_ldih,
              mini_steps, prev_chk, out_chk, platform):

    system = build_system(psf, params, barostat=barostat)
    add_restraints(system, positions, input_dir, fc_bb, fc_sc, fc_lpos, fc_ldih)

    integrator = mm.LangevinMiddleIntegrator(
        303.15 * unit.kelvin,
        1.0 / unit.picoseconds,
        dt_ps * unit.picoseconds,
    )
    ctx = mm.Context(system, integrator, platform)

    if prev_chk and os.path.exists(prev_chk):
        with open(prev_chk, "rb") as fh:
            ctx.loadCheckpoint(fh.read())
        print(f"  Resumed from: {prev_chk}")
    else:
        ctx.setPositions(positions)
        ctx.setPeriodicBoxVectors(*box_vectors)
        if gen_vel:
            ctx.setVelocitiesToTemperature(303.15 * unit.kelvin)

    if mini_steps > 0:
        print(f"  Minimizing ({mini_steps} steps) ...")
        mm.LocalEnergyMinimizer.minimize(ctx, 100.0, mini_steps)

    print(f"  Running {nstep:,} steps ({nstep * dt_ps / 1000:.3f} ns) ...")
    t0 = time.time()
    integrator.step(nstep)
    elapsed = time.time() - t0
    print(f"  {elapsed:.1f}s  ({nstep / elapsed:.0f} steps/s)")

    with open(out_chk, "wb") as fh:
        fh.write(ctx.createCheckpoint())

    state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
    return state.getPositions(), state.getPeriodicBoxVectors()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# (label, nstep, dt_ps, barostat, mini_steps, gen_vel, fc_bb, fc_sc, fc_lpos, fc_ldih)
STAGES = [
    ("6.1 NVT",  125_000, 0.001, False, 5_000, True,  4000., 2000., 1000., 1000.),
    ("6.2 NVT",  125_000, 0.001, False,     0, False, 2000., 1000.,  400.,  400.),
    ("6.3 NPT",  125_000, 0.001, True,      0, False, 1000.,  500.,  400.,  200.),
    ("6.4 NPT",  250_000, 0.002, True,      0, False,  500.,  200.,  200.,  200.),
    ("6.5 NPT",  250_000, 0.002, True,      0, False,  200.,   50.,   40.,  100.),
    ("6.6 NPT",  250_000, 0.002, True,      0, False,   50.,    0.,    0.,    0.),
]


def main():
    args       = parse_args()
    input_dir  = find_input_dir(args.input_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    final_chk = os.path.join(output_dir, "equil_final.chk")
    final_pdb = os.path.join(output_dir, "equil_final.pdb")

    if args.skip_if_done and os.path.exists(final_chk):
        print(f"Equilibration already done ({final_chk}) — skipping.")
        return

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    plat = best_platform()
    print(f"Platform  : {plat.getName()}")

    print("\nLoading topology + parameters ...")
    psf    = app.CharmmPsfFile(os.path.join(input_dir, "step5_input.psf"))
    pdb    = app.PDBFile(os.path.join(input_dir, "step5_input.pdb"))
    params = load_charmm_params(input_dir)

    positions   = pdb.positions
    # CHARMM-GUI's step5_input.pdb has no CRYST1 record; pull box from
    # ../step3_size.str. CharmmPsfFile.createSystem requires periodicity
    # set via setBox(a, b, c, α, β, γ) — topology.setPeriodicBoxVectors is
    # not enough (createSystem doesn't consult it).
    a, b, c, alpha, beta, gamma = _set_psf_box(psf, input_dir)
    print(f"  Box: {a:.3f} × {b:.3f} × {c:.3f} nm, "
          f"angles {alpha:.1f}/{beta:.1f}/{gamma:.1f}°")
    box_vectors = psf.topology.getPeriodicBoxVectors()

    prev_chk = None
    for i, (label, nstep, dt, baro, mini, gen_vel,
            fc_bb, fc_sc, fc_lpos, fc_ldih) in enumerate(STAGES, start=1):

        stage_chk = os.path.join(output_dir, f"stage{i}.chk")
        print(f"\n── Stage {label} ──────────────────────────────────────────")

        if os.path.exists(stage_chk):
            print(f"  Checkpoint exists — skipping: {stage_chk}")
            prev_chk = stage_chk
            continue

        positions, box_vectors = run_stage(
            label, psf, params, positions, box_vectors, input_dir,
            nstep=nstep, dt_ps=dt, gen_vel=gen_vel, barostat=baro,
            fc_bb=fc_bb, fc_sc=fc_sc, fc_lpos=fc_lpos, fc_ldih=fc_ldih,
            mini_steps=mini, prev_chk=prev_chk,
            out_chk=stage_chk, platform=plat,
        )
        prev_chk = stage_chk

    shutil.copy(prev_chk, final_chk)
    app.PDBFile.writeFile(psf.topology, positions,
                          open(final_pdb, "w"), keepIds=True)

    print("\nEquilibration complete.")
    print(f"  Checkpoint : {final_chk}")
    print(f"  Final PDB  : {final_pdb}")


if __name__ == "__main__":
    main()
