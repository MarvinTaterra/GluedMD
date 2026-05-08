# GLUED

**GLUED** is a GPU-resident enhanced sampling plugin for OpenMM. It provides PLUMED-equivalent collective variables (CVs) and bias methods with all CV evaluation, bias evaluation, and chain-rule force scatter running natively inside OpenMM's GPU kernel infrastructure — eliminating the CPU↔GPU round-trip cost of the `openmm-plumed` plugin.

## Why GLUED?

The standard `openmm-plumed` plugin works by: pulling positions from GPU → CPU, calling PLUMED, then pushing forces back to GPU every step. At GPU speeds this transfer can cost more time than the actual MD integration. GLUED reimplements the CV and bias layer directly as NVRTC-compiled GPU kernels so that positions never leave the device between steps.

## Feature overview

| Category | What's implemented |
|---|---|
| **Collective variables** | Distance, Angle, Dihedral, COM-Distance, Radius of Gyration, Coordination Number, RMSD, Distance-RMSD, Contact Map, Path (s/z), Position, Plane, Projection, Dipole, Volume, Cell, Ring Puckering, Secondary Structure, PCA, eRMSD, Expression (algebraic), PyTorch (TorchScript ML) |
| **Bias methods** | Harmonic restraint, Moving restraint, Well-tempered MetaD, PBMetaD, OPES, OPES-Expanded, External grid, ABMD, Linear coupling, Upper/Lower walls, Extended-Lagrangian (AFED), EDS, MaxEnt |
| **Utilities** | COLVAR file reporter, Bias state checkpoint/restore, Replica exchange (H-REUS and T-REMD), Multi-walker MetaD/OPES (shared GPU arrays, no CPU merge) |
| **Multi-GPU** | Single system across GPUs (OpenMM native), N replicas one-per-GPU, N walkers across G GPUs with intra-GPU pointer sharing + cross-GPU additive grid merge |
| **Platforms** | Reference (CPU), CUDA (GPU). OpenCL and HIP stubs present. |

## Quick start

```python
import openmm as mm
import openmm.app as app
import glued                          # Pythonic wrapper (recommended)

# 1. Create the force — PBC and temperature set once, used by all biases
force = glued.Force(pbc=True, temperature=300.0)

# 2. Register CVs — plain Python lists, no mm.vectori() needed
phi = force.add_dihedral([4, 6,  8, 14])
psi = force.add_dihedral([6, 8, 14, 16])

# 3. Add a bias — keyword arguments, temperature inherited from Force
force.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500)

# 4. Build simulation as normal
system.addForce(force)
integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
simulation = app.Simulation(topology, system, integ,
                             mm.Platform.getPlatformByName("CUDA"))

# 5. Log CVs to a COLVAR file
from COLVARReporter import COLVARReporter
simulation.reporters.append(
    COLVARReporter('colvar.dat', 500, force, cvNames=['phi', 'psi'])
)

simulation.step(5_000_000)
```

The `glued` module is a thin wrapper around `gluedplugin.GluedForce`; the raw `gluedplugin` API remains fully available for advanced use.

## Documentation

- [Installation](installation.md) — build from source (WSL2 / Linux)
- [Collective Variables](collective_variables.md) — all 22 CV types with parameter tables
- [Bias Methods](bias_methods.md) — all 14 bias types with parameter tables
- [Python API Reference](python_api.md) — `glued.Force`, `GluedForce`, `COLVARReporter`, `ReplicaExchange`, `MultiGPUManager`
- [Examples](examples.md) — practical recipes for MetaD, OPES, REUS, checkpointing, multi-walker
- [Multi-GPU Guide](multi_gpu.md) — all three multi-GPU scenarios with code examples
- [Developer Guide](developer_guide.md) — code architecture, adding new CVs and biases
- [Architecture Notes](architecture-notes.md) — deep dive into OpenMM internals and why GLUED avoids the CPU↔GPU round-trip
