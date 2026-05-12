"""Kv1.2 + S4 Z-displacement CV — OPES Explore benchmark.

167 800-atom voltage-gated K⁺ channel (Kv1.2 chimera 2R9R + 3:1 POPC:POPG +
TIP3P + 150 mM KCl). The CV is the Z-coordinate of the S4 helix Cα COM
minus the VSD reference frame Cα COM, built from 7 ``add_position`` CVs
(one per Cα, z-component only) + ``add_expression`` — no PyTorch involved.

This is the suite's "large analytical CV" case; the per-step cost is
dominated by the 167k-atom integrator + PME, not the CV.

Setup:
  Requires ``input_dir`` pointing at the KV1.2 CHARMM-GUI ``openmm/`` and
  an ``equil_final.chk`` from ``benchmarks/kv12_s4/01_minimize_equilibrate.py``.
"""

import argparse
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import best_platform_cuda_or_skip, run_timed, write_result

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
prod  = import_module("03_production_glued")
equil = import_module("01_minimize_equilibrate")


CASE         = "kv12_s4"
WARMUP_STEPS = 500    # 167k atoms — warmup is more expensive
TIMED_STEPS  = 2_000


def build_context(input_dir: str, equil_chk: str):
    import glued

    psf    = app.CharmmPsfFile(os.path.join(input_dir, "step5_input.psf"))
    params = prod.load_charmm_params(input_dir)
    equil._set_psf_box(psf, input_dir)
    system = prod.make_production_system(psf, params)

    force = glued.Force(pbc=True, temperature=prod.TEMP)
    cv_idx, bias_idx = prod.add_s4_opes(force)
    system.addForce(force)

    integ = mm.LangevinMiddleIntegrator(
        prod.TEMP * unit.kelvin, 1.0 / unit.picoseconds,
        prod.DT_PS * unit.picoseconds)
    ctx = mm.Context(system, integ, best_platform_cuda_or_skip())
    with open(equil_chk, "rb") as f:
        ctx.loadCheckpoint(f.read())
    integ.step(2)
    return ctx, system.getNumParticles()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", ""))
    ap.add_argument("--equil-dir",
                    default=os.environ.get("EQUIL_DIR",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "equil")))
    ap.add_argument("--output-dir", default="benchmarks/results")
    ap.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    ap.add_argument("--steps",  type=int, default=TIMED_STEPS)
    args = ap.parse_args()

    print(f"=== {CASE} ===")
    try:
        input_dir = prod.find_input_dir(args.input_dir)
    except FileNotFoundError as e:
        sys.exit(f"[skip] {CASE}: {e}")
    equil_chk = os.path.join(args.equil_dir, "equil_final.chk")
    if not os.path.exists(equil_chk):
        sys.exit(f"[skip] {CASE}: {equil_chk} not found — run "
                 "benchmarks/kv12_s4/01_minimize_equilibrate.py first.")

    ctx, n_atoms = build_context(input_dir, equil_chk)
    result = run_timed(ctx, dt_ps=prod.DT_PS,
                       warmup_steps=args.warmup, timed_steps=args.steps)
    result["n_atoms"]   = n_atoms
    result["cv_kind"]   = "S4 Z-COM diff (7×position + expression)"
    result["bias_kind"] = "OPES_EXPLORE"
    path = write_result(args.output_dir, CASE, result)
    print(f"  atoms={n_atoms}  steps/s={result['steps_per_sec']:.0f}  "
          f"ns/day={result['ns_per_day']:.1f}")
    print(f"  → {path}")


if __name__ == "__main__":
    main()
