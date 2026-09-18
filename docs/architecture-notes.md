# GLUED Architecture Notes

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
  integration).  This is where bias history is committed: deposits, running
  statistics, auxiliary coordinates.  It sets OpenMM's `forcesInvalid` flag
  when the bias potential changed so integrators that cache forces
  (`CustomIntegrator`) recompute them.
- **`calcForcesAndEnergy()`** — called potentially multiple times per step (during
  minimization, constraint iteration, force-group recalculation, `getState()`).
  This is where CV evaluation and force scatter happen.  It never modifies
  bias history, so energy queries and rejected trial moves are side-effect free.

The `lastStepIndex_` guard in `GluedForceImpl::updateContextState()` ensures
history is committed exactly once per step count.  See §7 for the resulting
execution contract.

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
holds user atom `atomIndexArray[g]`; the array maps *GPU slot → user atom*, so the
lookup a CV kernel needs (user atom → GPU slot) is its inverse.

GLUED keeps that inverse on the device.  When OpenMM signals a reorder
(`getAtomsWereReordered()`), `rebuildGpuAtomIndices()` downloads the index array,
inverts it on the host and re-uploads the per-CV GPU atom lists that the CV and
scatter kernels index directly:

```cuda
int gpuAtom = cvGpuAtoms[i];          // already translated on reorder
atomicAdd(&forceBuffer[gpuAtom], ...);
```

Symmetric CVs (e.g. distance) give the same *value* regardless of reordering, but
*forces* go to the wrong atoms if the translation is stale.  Reorders are rare
(OpenMM sorts every few hundred steps), so the download is cheap in practice.

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

For GLUED: every kernel source is an embedded raw string in
`platforms/common/src/CommonGluedKernels.cpp` (CV kernels, bias kernels, scatter),
compiled with `cc.compileProgram()` during `initialize()`.  Per-bias constants that
change the generated code (e.g. the OPES periodic-CV mask) are passed as defines.

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

GLUED avoids this by inheriting `ForceImpl` directly and placing the per-step
computation — CV kernels, bias kernels, chain-rule scatter — inside OpenMM's GPU
kernel pipeline.  Positions and forces never leave the device.  The remaining host
traffic is small and mostly infrequent:

- the energy CV (`CV_ENERGY`) reduces the inner Context's energy on the host and
  uploads one scalar per evaluation, as OpenMM's own `CustomCVForce` does;
- an OPES deposit reads one integer back to detect a full kernel table;
- an atom reorder downloads the index array (§4);
- CV logging, OPES diagnostics and checkpoints download on request.

**Corollary:** GLUED's `GluedForceImpl::calcForcesAndEnergy` calls
`kernel_.getAs<CalcGluedForceKernel>().execute(context, ...)` which dispatches
directly to GPU kernels compiled via NVRTC, with no position download.

---

## 7 — Execution, checkpoint and multi-walker contracts

**Evaluation vs. history.** `calcForcesAndEnergy()` evaluates CVs and biases and
never commits history.  `updateContextState()` re-evaluates CVs and biases at the
current coordinates on the steps where something is committed (a deposit is due,
or a bias such as ABMD, EDS, extended-Lagrangian or adaptive OPES accumulates
every step) and then commits.  Consequences:

- Extra `getState()` calls, minimization and rejected trial configurations leave
  adaptive biases untouched.
- Deposits use the coordinates of the step they are attributed to.
- `CustomIntegrator` users must call `addUpdateContextState()` once per step.
- Rewinding a trajectory does not rewind bias history; restore a bias checkpoint
  alongside the OpenMM checkpoint.
- Device counters (deposits, adaptive samples) are 32-bit; a run stops with an
  error before they can wrap (about 2.1 billion steps).

**Checkpoints.** `getBiasState()` produces a version-2 blob: a hash of the
force configuration, the last committed step, and every mutable field of every
bias, including the OPES normalization, weight sums, deposit count and learned
bandwidths needed to continue exactly.  `setBiasState()` requires the same
version, an identical configuration (the OPES kernel capacity may be raised),
and exactly the expected number of bytes; a rejected blob leaves the previous
state in place.  Pass the Context explicitly when several Contexts share a
System.  Version-1 blobs cannot be restored: they lack the continuation fields.

**Multiple walkers.** Walkers on one device do not share device arrays.
`MultiWalkerPool` moves the selected bias between Contexts through checkpoints:
MetaD/PBMetaD grids merge their increments every `sync_interval` steps (hills
are additive), while OPES-family biases hand one state from walker to walker one
step at a time, because a compressed kernel table cannot be merged after the
fact.  Unselected biases stay private to each walker.

**Replica exchange.** Both drivers evaluate all four cross energies with each
replica's own Hamiltonian, so every bias is part of the criterion and nothing
needs isolating in a force group.  An isotropic `MonteCarloBarostat` adds the
pressure–volume term; other barostats are rejected.

**Precision and platforms.** Bias tables and checkpoints are double; several
geometric Jacobians and parameter tables are float even in a double Context.
Only CUDA and OpenCL run CVs; the Reference plugin accepts empty forces only.
