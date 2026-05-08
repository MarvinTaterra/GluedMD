# GLUED Architecture Notes

Source references: `C:/Users/Marvin/Desktop/openmm` (tag 8.4.0),
`C:/Users/Marvin/Desktop/openmm-plumed`, `C:/Users/Marvin/Desktop/plumed2`.

---

## 1 — How `Force::createImpl()` becomes a per-step kernel call

**Sources:** `openmmapi/include/openmm/internal/ForceImpl.h`,
`openmmapi/src/ContextImpl.cpp` L54–188 (constructor), L300–313
(`calcForcesAndEnergy`), L329–334 (`updateContextState`).

When `mm.Context(system, integrator, platform)` is constructed,
`ContextImpl::ContextImpl()` loops over every `Force` in the `System`:

```cpp
// ContextImpl.cpp L116–120
for (int i = 0; i < system.getNumForces(); ++i) {
    forceImpls.push_back(system.getForce(i).createImpl());
    ...
}
// L179–181
for (size_t i = 0; i < forceImpls.size(); ++i)
    forceImpls[i]->initialize(*this);
```

`Force::createImpl()` is the user-defined factory; `GluedForce::createImpl()`
returns a `new GluedForceImpl(owner)`.  `ForceImpl::initialize()` then creates
the platform kernel:

```cpp
// GluedForceImpl.cpp
kernel_ = context.getPlatform().createKernel(
    CalcGluedForceKernel::Name(), context);
kernel_.getAs<CalcGluedForceKernel>().initialize(system, owner_);
```

Each integrator step calls two hooks on all `ForceImpl` objects:

- **`updateContextState()`** — called *once* at the top of each step (before
  integration).  This is where bias deposition must happen.
- **`calcForcesAndEnergy()`** — called potentially multiple times per step (during
  minimization, constraint iteration, force-group recalculation).  This is where
  CV evaluation and force scatter happen.

The `lastStepIndex_` guard in `GluedForceImpl::updateContextState()` ensures
deposition fires exactly once per step count.

---

## 2 — How GPU positions flow in and forces flow out

**Sources:** `platforms/common/include/openmm/common/ComputeContext.h`,
`platforms/cuda/include/openmm/cuda/CudaContext.h`,
`platforms/common/src/CommonKernels.cpp` (RMSDForceKernel, ~L4400;
CustomBondForceKernel, ~L583).

Positions live in a `ComputeArray` named `posq` (position + charge, packed as
`float4` in single precision, `double4` in double precision):

```cpp
ComputeArray& posq = cc.getPosq();   // float4* or double4*
```

Forces are accumulated in a flat `long long` force buffer — field-major layout:

```
forceBuffer[atom]                    // Fx of atom
forceBuffer[atom + paddedNumAtoms]   // Fy of atom
forceBuffer[atom + 2*paddedNumAtoms] // Fz of atom
```

`cc.getPaddedNumAtoms()` is the stride (always a multiple of 32 for CUDA, 64 for
HIP) and differs from `cc.getNumAtoms()`.  Kernels write to the buffer via
atomic-add using the fixed-point scale (see §3).

The ComputeContext passed to `CommonCalcGluedForceKernel` (stored as `cc_`)
provides these arrays without any CPU involvement.  This is the architectural
difference from openmm-plumed's `CustomCPPForceImpl`, which downloads positions to
CPU then uploads forces back.

---

## 3 — Fixed-point force convention (`realToFixedPoint`, scale 2³²)

**Sources:** `platforms/common/include/openmm/common/CommonKernelSources.h`,
search for `realToFixedPoint`.

OpenMM accumulates forces in 64-bit integers (`long long`) to avoid race conditions
in parallel GPU reductions.  The conversion factor is `0x100000000LL = 2^32`:

```cuda
// Kernel pseudocode
long long fx = (long long)(force_x * 0x100000000LL);
atomicAdd(&forceBuffer[gpuAtom],                    fx);
atomicAdd(&forceBuffer[gpuAtom + paddedNumAtoms],   fy);
atomicAdd(&forceBuffer[gpuAtom + 2*paddedNumAtoms], fz);
```

Using the wrong scale produces forces off by 2^32 — silent in some float regimes
(roundoff absorbs it) but catastrophic in double precision.  `atomicAdd` on
`long long` requires CUDA compute capability ≥ 6.0.

---

## 4 — The atom-reorder issue and `getAtomIndexArray()`

**Sources:** `CudaContext.h` (`getAtomIndexArray()`),
`CommonKernels.cpp` `CommonCalcRMSDForceKernel` for the canonical pattern.

OpenMM reorders atoms internally for memory-access efficiency.  GPU atom index `g`
corresponds to user atom index `atomIndexArray[g]`.  The reverse map (user→GPU)
can be derived by iterating the array, or by using the per-atom sorted permutation.

CV kernels receive user-visible atom indices from `cvAtoms[]`.  Before writing to
the force buffer they must look up the GPU index:

```cuda
int gpuAtom = atomIndexArray[userAtom];
atomicAdd(&forceBuffer[gpuAtom], ...);
```

Symmetric CVs (e.g. distance) give the same *value* regardless of reordering, but
*forces* go to the wrong atoms if the lookup is omitted.  Every scatter must do it.

---

## 5 — NVRTC source assembly via `ComputeContext`

**Sources:** `platforms/common/include/openmm/common/ComputeContext.h`,
`platforms/common/src/ExpressionUtilities.cpp`,
`platforms/cuda/include/openmm/cuda/CudaContext.h`.

OpenMM compiles CUDA/OpenCL kernel source at runtime using NVRTC (CUDA) or
the OpenCL runtime compiler.  The workflow is:

1. Build a kernel source string (CUDA `.cu` text) in C++.
2. Call `cc.compileProgram(source, defines)` which invokes NVRTC.
3. Get a `ComputeKernel` handle: `cc.getKernel(program, "kernelFunctionName")`.
4. Set arguments and enqueue: `kernel.setArg(0, buffer); cc.executeKernel(kernel, numAtoms, blockSize)`.

The source string can include `#define` blocks built via `cc.getExpressionUtilities()`
which translates Lepton expressions to CUDA source.  NVRTC can see all of OpenMM's
utility headers via include paths embedded in the `ComputeContext`.

For GLUED: CV kernel source is a `.cu` file in `platforms/common/src/kernels/`
that is read at initialization time, potentially with per-system `#define` injections
(e.g. `#define NUM_DISTANCE_CVS 3`), then compiled via NVRTC on first `execute()`.

---

## 6 — Why openmm-plumed pays CPU↔GPU traffic, and how GLUED avoids it

**Sources:** `openmm-plumed/openmmapi/include/internal/PlumedForceImpl.h`,
`openmm-plumed/openmmapi/src/PlumedForceImpl.cpp` L121–152
(`computeForce`), `openmm/openmmapi/include/openmm/internal/CustomCPPForceImpl.h`.

`PlumedForceImpl` inherits `CustomCPPForceImpl`.  `CustomCPPForceImpl::calcForcesAndEnergy`
downloads the full position vector from GPU to CPU, calls the user's
`computeForce(positions, forces)` on CPU, then uploads the resulting force array
back to GPU.  From `openmm-plumed/openmmapi/src/PlumedForceImpl.cpp`:

```cpp
// computeForce() receives CPU-side positions / forces
plumed_cmd(plumedmain, "setPositions", &pos[0][0]);
plumed_cmd(plumedmain, "setForces",    &forces[0][0]);
plumed_cmd(plumedmain, "performCalcNoUpdate", NULL);
```

For a 100k-atom system at fp64 this is 100k × 3 × 8 bytes = 2.4 MB per direction
per step.  At 1000 steps/second that is ~4.8 GB/s sustained PCIe bandwidth —
saturating a PCIe 4.0 ×16 link by itself.

GLUED avoids this by inheriting `ForceImpl` directly and placing all
computation — CV kernels, bias kernels, chain-rule scatter — inside OpenMM's GPU
kernel pipeline.  The only CPU↔GPU traffic is optional: periodic CV value logging
(a few bytes every N steps) and bias state checkpointing (once per simulation).

**Corollary:** GLUED's `GluedForceImpl::calcForcesAndEnergy` calls
`kernel_.getAs<CalcGluedForceKernel>().execute(context, ...)` which dispatches
directly to GPU kernels compiled via NVRTC, with no position download.

---

*This document was produced as the Stage 0 / Section 0 deliverable before Stage 1.1
begins.  Human review required at this boundary per the design plan.*
