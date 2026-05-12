"""Shared timing primitives for the GLUED benchmark suite.

Design constraints
------------------
1. **Zero CPU↔GPU round-trips in the timed loop.** The timed phase is a
   single ``integrator.step(N)`` call — no Python in the inner loop, no
   ``getState(getPositions=True)``, no DCD writes, no
   ``getLastCVValues``. Each step does CV eval, bias eval, chain-rule
   force scatter, and integrator update entirely on the GPU. Even
   PyTorch CVs (CV_PYTORCH) execute their forward + autograd backward
   on the GPU via libtorch on the CUDA platform — see
   ``platforms/cuda/src/CudaCalcGluedForceKernel.cpp`` for the wiring.

2. **The trailing sync is the only CPU-visible op.** After ``step(N)``
   we call ``getState(getEnergy=False, getPositions=False, ...)`` with
   every flag off so OpenMM only flushes the CUDA queue; nothing is
   downloaded.

3. **Warmup is included.** First ``step()`` compiles NVRTC kernels
   (eval + scatter + integrator) which dominates wall time at small N.
   ``run_timed()`` runs ``warmup_steps`` first that are discarded.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Callable


# ---------------------------------------------------------------------------
# Benchmark primitive
# ---------------------------------------------------------------------------

def run_timed(
    context,
    *,
    dt_ps: float,
    warmup_steps: int = 1000,
    timed_steps: int = 5000,
) -> dict:
    """Run a GPU benchmark and return throughput numbers.

    The context's integrator is stepped ``warmup_steps`` (untimed) then
    ``timed_steps`` (timed). The timed phase is a single ``step()`` call,
    guaranteeing zero CPU↔GPU traffic inside the measurement window. A
    final no-data ``getState`` synchronizes the CUDA stream so wall-time
    captures every queued kernel.
    """
    integrator = context.getIntegrator()

    # Warmup — compile kernels, populate caches.
    integrator.step(warmup_steps)

    # Stream sync before timing.
    context.getState()

    t0 = time.perf_counter()
    integrator.step(timed_steps)
    context.getState()       # stream sync — no buffer download
    elapsed = time.perf_counter() - t0

    steps_per_sec = timed_steps / elapsed
    ns_per_day    = steps_per_sec * dt_ps * 86_400 / 1_000.0
    return {
        "warmup_steps":  int(warmup_steps),
        "timed_steps":   int(timed_steps),
        "elapsed_s":     float(elapsed),
        "steps_per_sec": float(steps_per_sec),
        "ns_per_day":    float(ns_per_day),
        "dt_ps":         float(dt_ps),
    }


def best_platform_cuda_or_skip():
    """Return the CUDA platform, or raise SystemExit if unavailable.

    GLUED's full performance + PyTorch CV path require CUDA. Other
    platforms run but produce numbers that aren't comparable.
    """
    import openmm as mm
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        raise SystemExit("[skip] CUDA platform not available — benchmarks "
                         "are meaningful only on the CUDA backend.")


# ---------------------------------------------------------------------------
# Environment reporting
# ---------------------------------------------------------------------------

def gpu_info() -> dict | None:
    """Return {'name': ..., 'memory_mb': ...} for GPU 0 (nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
        return {"name": name, "memory_mb": int(mem)}
    except Exception:
        return None


def write_result(out_dir: str, name: str, result: dict, extra: dict | None = None):
    """Write a result JSON record (overwrites)."""
    os.makedirs(out_dir, exist_ok=True)
    record = dict(result)
    record["benchmark"] = name
    record["gpu"]       = gpu_info()
    if extra:
        record.update(extra)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
