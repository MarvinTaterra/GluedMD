"""Alanine dipeptide in TIP3P water — OPES Metad on φ.

ADP in a small explicit-water box (~2500 atoms with PME). Same CV +
bias as ad_vacuum, so the difference vs ad_vacuum measures the cost
the surrounding water adds to a step (PME, bond constraints, more
atoms in the force scatter).

No shipped solvated coordinates — at startup we run ``Modeller`` to
solvate the vacuum ADP in a ~2.5 nm TIP3P cube. That makes the
benchmark fully self-contained without bloating the repo with a
solvated PDB.
"""

import argparse
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import best_platform_cuda_or_skip, run_timed, write_result


CASE         = "ad_water"
TEMPERATURE  = 300.0
DT_PS        = 0.002
WARMUP_STEPS = 2_000
TIMED_STEPS  = 20_000
BOX_PADDING  = 1.0   # nm


def _find_phi_dihedral(topology):
    residues = [r for r in topology.residues() if r.name in ("ALA", "ACE", "NME")]
    if len(residues) < 2:
        raise RuntimeError(f"Expected ≥2 protein residues, found {len(residues)}")
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
    ff  = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, model="tip3p",
                        padding=BOX_PADDING * unit.nanometer)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )

    force = glued.Force(pbc=True, temperature=TEMPERATURE)
    phi = force.add_dihedral(_find_phi_dihedral(modeller.topology))
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
    ap.add_argument("--pdb",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "ad_vacuum", "adp.pdb"))
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
