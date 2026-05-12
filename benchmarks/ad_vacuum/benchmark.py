"""Alanine dipeptide in vacuum — OPES Metad on φ.

Smallest case in the suite: 22 atoms, no PME, no constraints, dt=2 fs.
Establishes the "floor" cost of GLUED — what does the bias machinery
add to a system tiny enough that the bias kernel dominates wall time?

The CV is φ = dihedral(N1, CA, C, N2) computed natively by GLUED
(CV_DIHEDRAL). The bias is OPES_METAD with σ=0.35 rad, γ=10. There are
no PyTorch CVs in this case — it's the analytical-CV baseline.
"""

import argparse
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import best_platform_cuda_or_skip, run_timed, write_result


CASE         = "ad_vacuum"
TEMPERATURE  = 300.0
DT_PS        = 0.002
WARMUP_STEPS = 2_000
TIMED_STEPS  = 50_000   # ADP is so cheap on GPU we need many steps for stable timing


def _find_phi_dihedral(topology):
    """Locate a backbone N-CA-C-N dihedral spanning the two ALA residues.

    The shipped ``adp.pdb`` is heavy-atom-only; after ``addHydrogens`` the
    atom serial numbering is FF-determined, so we have to find the dihedral
    by atom names rather than hard-coding indices.
    """
    residues = list(topology.residues())
    if len(residues) < 2:
        raise RuntimeError(f"ADP PDB needs ≥2 residues, has {len(residues)}.")
    r1, r2 = residues[0], residues[1]
    def _atom(res, name):
        for a in res.atoms():
            if a.name == name:
                return a.index
        raise KeyError(f"{name} not in residue {res.name}{res.index}")
    return [_atom(r1, "C"), _atom(r2, "N"), _atom(r2, "CA"), _atom(r2, "C")]


def build_context(pdb_path: str):
    import glued

    pdb = app.PDBFile(pdb_path)
    ff  = app.ForceField("amber14-all.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    # GLUED: a backbone dihedral CV + OPES_METAD bias on it.
    force = glued.Force(temperature=TEMPERATURE)
    phi_atoms = _find_phi_dihedral(modeller.topology)
    phi = force.add_dihedral(phi_atoms)
    force.add_opes([phi], sigma=0.35, gamma=10.0, pace=500,
                   max_kernels=100_000, mode="metad")
    system.addForce(force)

    integ = mm.LangevinMiddleIntegrator(
        TEMPERATURE * unit.kelvin, 1.0 / unit.picoseconds,
        DT_PS * unit.picoseconds)
    ctx = mm.Context(system, integ, best_platform_cuda_or_skip())
    ctx.setPositions(modeller.positions)
    ctx.setVelocitiesToTemperature(TEMPERATURE)
    return ctx, system.getNumParticles()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="benchmarks/results")
    ap.add_argument("--pdb", default=os.path.join(os.path.dirname(__file__), "adp.pdb"))
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    ap.add_argument("--steps",  type=int, default=TIMED_STEPS)
    args = ap.parse_args()

    print(f"=== {CASE} ===")
    ctx, n_atoms = build_context(args.pdb)
    result = run_timed(ctx, dt_ps=DT_PS,
                       warmup_steps=args.warmup, timed_steps=args.steps)
    result["n_atoms"] = n_atoms
    result["cv_kind"] = "CV_DIHEDRAL"
    result["bias_kind"] = "OPES_METAD"
    path = write_result(args.output_dir, CASE, result)
    print(f"  atoms={n_atoms}  steps/s={result['steps_per_sec']:.0f}  "
          f"ns/day={result['ns_per_day']:.1f}")
    print(f"  → {path}")


if __name__ == "__main__":
    main()
