"""KOR + small TorchScript CV (10 Cα atoms) — OPES Explore benchmark.

Same 71 873-atom system as ``kor_a100``, but the bias-driving CV is now a
TorchScript model (`deepcv.pt`, ~21 KB, 2 017 parameters) wrapping all
45 pairwise Cα–Cα distances among the 10 atoms that define A100. GLUED
runs the model's forward + autograd backward on the **same CUDA stream**
as the integrator — no CPU↔GPU traffic per step (see
``platforms/common/src/CommonGluedKernels.cpp`` L4476–4528 for the
zero-copy ``torch::from_blob`` + ``CUDAStreamGuard`` wiring).

This case measures the per-step cost of a *cheap* PyTorch CV.
``kor_deepcv_large`` measures the cost when the model is 21 M parameters.
"""

import argparse
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import best_platform_cuda_or_skip, run_timed, write_result

# Reuse the KOR system builder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "kor_a100"))
from importlib import import_module
prod  = import_module("03_production_glued")
equil = import_module("01_minimize_equilibrate")


CASE         = "kor_deepcv_small"
WARMUP_STEPS = 1_000
TIMED_STEPS  = 5_000

# Atoms the model expects (in this order). Matches deep_cv/01_gather_training_data.py.
DEEP_CV_ATOMS = sorted({a for pair in prod.A100_PAIRS for a in pair})


def build_context(input_dir: str, equil_chk: str, model_path: str):
    import glued

    psf    = app.CharmmPsfFile(os.path.join(input_dir, "step5_input.psf"))
    params = prod.load_charmm_params(input_dir)
    equil._set_psf_box(psf, input_dir)
    system = prod.make_production_system(psf, params)

    force = glued.Force(pbc=True, temperature=prod.TEMP)
    cv_idx = force.addPyTorchCV(os.path.abspath(model_path), DEEP_CV_ATOMS, [])
    # OPES on the deep CV. Bias-evaluation cost is the same as on A100;
    # what changes is the CV evaluation (PyTorch forward+backward).
    force.add_opes([cv_idx], sigma=0.1, gamma=10.0, pace=500,
                   max_kernels=10_000, mode="explore")
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
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "kor_a100", "output", "equil")))
    ap.add_argument("--model",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "deepcv.pt"))
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
                 "benchmarks/kor_a100/01_minimize_equilibrate.py first.")
    if not os.path.exists(args.model):
        sys.exit(f"[skip] {CASE}: {args.model} not found.")

    ctx, n_atoms = build_context(input_dir, equil_chk, args.model)
    result = run_timed(ctx, dt_ps=prod.DT_PS,
                       warmup_steps=args.warmup, timed_steps=args.steps)
    result["n_atoms"]      = n_atoms
    result["cv_kind"]      = "CV_PYTORCH (10-atom, ~2k params)"
    result["bias_kind"]    = "OPES_EXPLORE"
    result["model_path"]   = args.model
    result["model_atoms"]  = len(DEEP_CV_ATOMS)
    path = write_result(args.output_dir, CASE, result)
    print(f"  atoms={n_atoms}  steps/s={result['steps_per_sec']:.0f}  "
          f"ns/day={result['ns_per_day']:.1f}")
    print(f"  → {path}")


if __name__ == "__main__":
    main()
