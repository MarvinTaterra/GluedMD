# GLUED
**GPU-Accelerated Library for Unified Exploration Dynamics**

🌐 **Website:** [gluedmd.com](https://www.gluedmd.com/)

GLUED is a GPU-resident enhanced sampling plugin for OpenMM. Collective variables, biases, and chain-rule force scatter all run natively inside OpenMM's GPU kernel infrastructure — no CPU↔GPU round-trip per step.
## Features
- **Collective variables** — distance, angle, dihedral, COM-distance, gyration, coordination, RMSD, DRMSD, contact map, dipole, path CVs, ring puckering, secondary structure, expression CVs, PyTorch CVs
- **Bias methods** — harmonic / moving restraints, walls, well-tempered metadynamics, PBMETAD, OPES (METAD + EXPANDED), ABMD, EDS, MaxEnt, external bias, extended-Lagrangian (AFED)
- **Multi-walker** support on a single GPU and across multiple GPUs
- **Bias state serialization** for restartable runs
## Quick start
```bash
# In a conda environment with OpenMM installed
cmake -S . -B build -G Ninja \
    -DOPENMM_DIR="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
python tests/test_api_smoke.py
```
See [`docs/installation.md`](docs/installation.md) for full build instructions including CUDA/OpenCL/HIP backends.
## Documentation
- [Overview](docs/index.md)
- [Installation](docs/installation.md)
- [Collective variables](docs/collective_variables.md)
- [Bias methods](docs/bias_methods.md)
- [Python API](docs/python_api.md)
- [Examples](docs/examples.md)
- [Multi-GPU](docs/multi_gpu.md)
- [Architecture notes](docs/architecture-notes.md)
## License
GLUED is released under the **GLUED Academic License v1.1** — free for academic and non-profit use; commercial use is reserved. See [`LICENSE`](LICENSE) for the full text.
For commercial licensing inquiries, please contact the author.
