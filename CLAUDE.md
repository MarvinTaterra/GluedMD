# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GLUED** (GPU-accelerated Library for Unified Exploration Dynamics) is a GPU-resident enhanced sampling plugin for OpenMM, providing PLUMED-equivalent functionality (collective variables, metadynamics, OPES, etc.) with all CV evaluation, bias evaluation, and chain-rule force scatter running on the GPU — eliminating the CPU↔GPU round-trip cost of the openmm-plumed plugin.

The full design specification is in `GLUED_DESIGN_PLAN.md`. Read it before writing any code.

## Architecture

### Core design decisions (locked)

- **Pattern B (two-pass):** CV kernels fill `cvValues` and `jacobianGrads`; a separate scatter kernel applies chain rule to atom forces. Not a fused-per-CV kernel.
- **Per-kind kernel dispatch (Pattern b):** one kernel launch per CV kind present in the system, not a switch-in-one-kernel or per-CV-thread approach.
- **Double for bias storage always:** `cvValues`, `cvBiasGradients`, and all bias grids are `double` regardless of platform precision (float drifts visibly over millions of steps).
- **Bias deposition in `updateState()`, not `execute()`:** `execute()` may fire multiple times per step. The `lastStepIndex` guard in `updateState()` prevents duplicate deposition.
- **Direct `ForceImpl` extension, not `CustomCPPForceImpl`:** the whole point is no CPU↔GPU round-trip.

### Key data layout

Force buffer: field-major with stride `paddedNumAtoms`. Index for atom `a`, component `c`: `forceBuffer[a + c * paddedNumAtoms]`.

Fixed-point convention: `long long` at scale `0x100000000LL` (2³²). Use `realToFixedPoint()` from OpenMM's `CommonKernelSources.h`.

GPU struct (`GpuPlan`):
- `cvValues` / `cvBiasGradients` — double arrays, length `numCVs`
- `cvAtomOffsets` / `cvAtoms` — prefix-sum + flat list of user-space atom indices
- `jacobianAtomIdx` / `jacobianGrads` / `jacobianCvIdx` — Jacobian entries filled by CV kernels

### Atom reordering

GPU-internal atom order differs from user-visible order. Always convert via `atomIndexArray` (user→GPU) in every scatter. Even symmetric CVs need this — the *forces* go to the wrong atoms if skipped.

### Platform hierarchy

```
CalcGluedForceKernel (abstract)
  └── ReferenceCalcGluedForceKernel  (CPU reference)
  └── CommonCalcGluedForceKernel     (shared GPU logic)
        ├── CudaCalcGluedForceKernel
        ├── OpenCLCalcGluedForceKernel
        └── HIPCalcGluedForceKernel
```

GPU-platform-specific optimizations go in platform files, never in the common layer.

### PBC conventions

Triclinic minimum-image order is strict: `dr -= boxZ * round(dr.z/boxZ.z)`, then `boxY`, then `boxX`. Reordering breaks triclinic correctness. Box vectors must be re-fetched every evaluation (barostats change them). Copy the formula verbatim from OpenMM's `utilities.cu`.

## Build system (WSL / Linux)

**Primary development environment: WSL2 (Ubuntu).** The repo lives at
`/mnt/c/Users/Marvin/Desktop/Plumed2GPU` in WSL. For faster compile times,
clone or copy to WSL's native filesystem (e.g. `~/glued/`).

### One-time WSL environment setup

```bash
conda create -n openmm_env -c conda-forge \
    openmm cmake ninja swig \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev cxx-compiler
conda activate openmm_env
```

### Configure and build

```bash
# With openmm_env active — OPENMM_DIR auto-detected from $CONDA_PREFIX
cmake -S . -B build -G Ninja
cmake --build build
cmake --install build   # installs .so files into $CONDA_PREFIX/lib/plugins
```

To target a specific platform only:
```bash
cmake --build build --target OpenMMGluedReference
```

### Run tests

```bash
# Smoke test (Stages 1.2-1.4 acceptance)
python tests/test_api_smoke.py

# Via ctest:
cd build && ctest --output-on-failure
```

### Key CMake variables

| Variable | Default | Notes |
|---|---|---|
| `OPENMM_DIR` | `$CONDA_PREFIX` | Auto-detected; override with `-DOPENMM_DIR=...` |
| `GLUED_BUILD_PYTHON_WRAPPERS` | ON if swig+python found | Set OFF to skip SWIG step |

CMake include rule (frequently missed): both `${OPENMM_DIR}/include` **and**
`${OPENMM_DIR}/include/openmm/common` must be on the include path — the second
contains `ComputeContext.h`, `ComputeArray.h`, and all common-layer GPU headers.

### Reference source trees (for reading, not building)

Located as siblings of the repo on the Windows side:
- `C:\Users\Marvin\Desktop\openmm` (tag 8.4.0)
- `C:\Users\Marvin\Desktop\openmm-plumed`
- `C:\Users\Marvin\Desktop\plumed2`

From WSL these are at `/mnt/c/Users/Marvin/Desktop/{openmm,openmm-plumed,plumed2}`.

### Plugin registration

`registerKernelFactories()` is the plugin entry point called by
`Platform::loadPluginsFromDirectory`. Kernel factories use this export —
**not** `__attribute__((constructor))` (that is for serialization proxies only).

## Testing strategy

Every CV kind requires:
1. CPU reference cross-check (GPU vs host, 1000 random configs)
2. Numerical derivative test (perturb positions by 1e-5, compare to emitted Jacobian)
3. Cross-validation against PLUMED via openmm-plumed

Every bias kind requires cross-validation against PLUMED's equivalent (`RESTRAINT`, `METAD`, `OPES_METAD`, etc.).

Test pyramid target (see Appendix B of design plan): ~200 unit tests (per-commit), ~30 PLUMED agreement tests (nightly), ~10 perf regression benchmarks (nightly).

## Development stages

Stages are dependency-ordered. Each must pass its acceptance tests before proceeding:
- **Section 1** — repo skeleton, public API stub, Reference platform, CUDA/OpenCL/HIP empty kernels, CI
- **Section 2** — GPU position reads/force writes, PBC delta vector
- **Section 3** — CV layer: distance, angle, dihedral, COM-distance, Rg, coordination number, RMSD, path CVs, expression CVs, ML CVs
- **Section 4** — chain-rule scatter kernel, energy reduction
- **Section 5** — bias methods: harmonic, moving restraint, well-tempered METAD, PBMETAD, OPES, external bias, ABMD
- **Section 6** — bias state serialization, XML serialization, COLVAR output, live CV query
- **Section 7** — multi-GPU, replica exchange, MPI (defer until 1–6 are stable)
- **Section 8** — profiling, documentation, conda packaging, published benchmark validation

## Anti-patterns to refuse

1. Using `CustomCPPForceImpl` for the GPU path — defeats the purpose.
2. Wrapping PLUMED instead of reimplementing CVs natively.
3. Float bias grid storage — always double.
4. Skipping cross-validation against PLUMED.
5. Omitting user→GPU atom index conversion in scatter kernels, even for symmetric CVs.
6. Adding CUDA-specific code to the common kernel layer.
7. Depositing bias in `execute()` instead of `updateState()`.
