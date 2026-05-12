"""KOR + large TorchScript CV (285 protein Cα, ~21M params) — OPES Explore.

Same 71 873-atom KOR system, but the CV is an all-Cα MLP (40 470 pairwise
distances → 512 → 256 → 128 → 1, ~21M parameters, 80.6 MiB on disk). The
model is too large to ship with the repo; ``download_model.py`` fetches
it from an external source on demand.

This case measures the per-step cost when the PyTorch model dominates
the GPU work — useful for sizing the "PyTorch CV overhead curve" against
the cheap ``kor_deepcv_small`` (~2k params) and the analytic ``kor_a100``
(no torch model).
"""

import argparse
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import best_platform_cuda_or_skip, run_timed, write_result

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "kor_a100"))
from importlib import import_module
prod  = import_module("03_production_glued")
equil = import_module("01_minimize_equilibrate")


CASE         = "kor_deepcv_large"
WARMUP_STEPS = 200    # large model: warmup is slow, smaller default
TIMED_STEPS  = 1_000  # the model is the dominant cost; fewer steps still gives stable timing


def build_context(input_dir: str, equil_chk: str, model_path: str):
    import glued
    import torch

    # The large model bakes its 285 global atom indices into a TorchScript
    # buffer; read them back so we don't drift from the training script.
    atom_indices = torch.jit.load(model_path, map_location="cpu") \
                        .atom_indices.tolist()
    if len(atom_indices) < 50:
        raise RuntimeError(f"Suspicious atom_indices length {len(atom_indices)} "
                           f"— is {model_path} the all-Cα model?")

    psf    = app.CharmmPsfFile(os.path.join(input_dir, "step5_input.psf"))
    params = prod.load_charmm_params(input_dir)
    equil._set_psf_box(psf, input_dir)
    system = prod.make_production_system(psf, params)

    force = glued.Force(pbc=True, temperature=prod.TEMP)
    cv_idx = force.addPyTorchCV(os.path.abspath(model_path), atom_indices, [])
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
    return ctx, system.getNumParticles(), len(atom_indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", ""))
    ap.add_argument("--equil-dir",
                    default=os.environ.get("EQUIL_DIR",
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "kor_a100", "output", "equil")))
    ap.add_argument("--model",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "deepcv_allca.pt"))
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
        sys.exit(f"[skip] {CASE}: {args.model} not found — too large to "
                 "ship with the repo. See README.md for download instructions.")

    ctx, n_atoms, n_model_atoms = build_context(input_dir, equil_chk, args.model)
    result = run_timed(ctx, dt_ps=prod.DT_PS,
                       warmup_steps=args.warmup, timed_steps=args.steps)
    result["n_atoms"]      = n_atoms
    result["cv_kind"]      = f"CV_PYTORCH ({n_model_atoms}-atom, ~21M params)"
    result["bias_kind"]    = "OPES_EXPLORE"
    result["model_path"]   = args.model
    result["model_atoms"]  = n_model_atoms
    path = write_result(args.output_dir, CASE, result)
    print(f"  atoms={n_atoms}  model_atoms={n_model_atoms}  "
          f"steps/s={result['steps_per_sec']:.0f}  ns/day={result['ns_per_day']:.1f}")
    print(f"  → {path}")


if __name__ == "__main__":
    main()
