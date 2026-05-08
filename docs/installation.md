# Installation

GLUED is built from source using CMake and conda. All three major operating systems are supported, though available GPU platforms differ by OS.

## Platform support

| Platform | Reference (CPU) | OpenCL | CUDA |
|---|---|---|---|
| Linux + NVIDIA GPU | ✓ | ✓ | ✓ |
| Linux + AMD GPU | ✓ | ✓ | — |
| Windows + NVIDIA GPU | ✓ | ✓ | ✓ |
| Windows + AMD/Intel GPU | ✓ | ✓ | — |
| macOS Intel | ✓ | ✓ | — |
| macOS Apple Silicon (M1/M2/M3) | ✓ | — | — |

CUDA on macOS is not supported by Apple (dropped in 2019). OpenCL on Apple Silicon is not available. The Reference platform works on every OS and is fully functional for development and small systems.

---

## Linux

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| CMake | ≥ 3.17 | |
| Ninja | any | Faster than Make for incremental builds |
| SWIG | ≥ 4.0 | Python wrapper generation |
| Python | ≥ 3.9 | |
| OpenMM | 8.x | From conda-forge |
| CUDA toolkit | ≥ 11.0 | Only needed for the CUDA platform |
| libtorch | ≥ 2.0 | **Optional** — only needed for `CV_PYTORCH` |

### One-time environment setup

```bash
conda create -n openmm_env -c conda-forge \
    openmm cmake ninja swig \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev cxx-compiler
conda activate openmm_env
```

To build without CUDA (Reference + OpenCL only), omit the `cuda-*` packages.

### Build

```bash
git clone <repo-url>
cd glued

cmake -S . -B build -G Ninja
cmake --build build
cmake --install build
```

`cmake --install` copies the platform plugins into `$CONDA_PREFIX/lib/plugins/` and installs `glued.py`, `gluedplugin.py`, `COLVARReporter.py`, and `ReplicaExchange.py` into site-packages.

CMake auto-detects `OPENMM_DIR` from `$CONDA_PREFIX`. CMake auto-detects CUDA and OpenCL; platforms are only built when the relevant toolkit is found.

### WSL2 note

Building from `/mnt/c/...` (the Windows filesystem) is slow due to cross-filesystem I/O. Copy the source to WSL's native filesystem for faster builds:

```bash
cp -r /mnt/c/Users/<you>/Desktop/glued ~/glued
cd ~/glued
cmake -S . -B build -G Ninja && cmake --build build
```

The Windows copy can still be used for editing — just build from the WSL copy.

---

## Windows

### Prerequisites

Install [Miniforge](https://github.com/conda-forge/miniforge/releases/latest) (or Miniconda). The CUDA toolkit for compilation comes from conda-forge — no separate NVIDIA installer is required for building. For runtime GPU execution you need an NVIDIA driver (≥ 450.x for CUDA 11.0).

### One-time environment setup

Open an **Anaconda Prompt** (or any terminal where `conda` is available):

```bat
conda create -n openmm_env -c conda-forge ^
    openmm cmake ninja swig ^
    cuda-nvcc cuda-cudart-dev cxx-compiler
conda activate openmm_env
```

### Build

```bat
git clone <repo-url>
cd glued

cmake -S . -B build -G Ninja -DOPENMM_DIR="%CONDA_PREFIX%\Library"
cmake --build build
cmake --install build
```

The `-DOPENMM_DIR="%CONDA_PREFIX%\Library"` override is required on Windows because conda installs OpenMM headers and libraries under the `Library\` subdirectory rather than at the prefix root.

### Known issue: NVRTC version mismatch

If you see `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` at runtime, the NVRTC bundled with conda's OpenMM package is newer than your host driver. Fix by preloading the system NVRTC before running any Python script:

**Linux/WSL2:**
```bash
export LD_PRELOAD=/usr/local/cuda/lib64/libnvrtc.so.XX:/usr/local/cuda/lib64/libnvrtc-builtins.so.XX
```

**Windows** (PowerShell — replace `XX` with your CUDA version):
```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.Y\bin;$env:PATH"
```

The permanent fix is to update your NVIDIA driver to one that supports the NVRTC version bundled with your OpenMM build (run `nvcc --version` and compare to the driver's CUDA version in `nvidia-smi`).

---

## macOS

### Prerequisites

```bash
conda create -n openmm_env -c conda-forge \
    openmm cmake ninja swig
conda activate openmm_env
```

No CUDA or OpenCL packages are needed. On Apple Silicon (M1/M2/M3) only the Reference platform builds; on Intel Macs the OpenCL platform is also available via Apple's built-in OpenCL runtime.

### Build

```bash
git clone <repo-url>
cd glued

cmake -S . -B build -G Ninja
cmake --build build
cmake --install build
```

CMake detects that no CUDA toolkit is present and skips the CUDA platform automatically.

---

## Common CMake options

| Option | Default | Effect |
|---|---|---|
| `-DOPENMM_DIR=<path>` | `$CONDA_PREFIX` (Linux/macOS) or `%CONDA_PREFIX%\Library` (Windows) | Override OpenMM installation path |
| `-DGLUED_BUILD_PYTHON_WRAPPERS=OFF` | ON | Skip SWIG wrapper generation |

### Build a single platform target

```bash
cmake --build build --target OpenMMGluedReference   # CPU reference, no CUDA required
cmake --build build --target OpenMMGluedCUDA        # CUDA platform only
cmake --build build --target OpenMMGluedOpenCL      # OpenCL platform only
```

---

## Running the tests

```bash
# Smoke test — verifies the plugin loads on every available platform
python tests/test_api_smoke.py

# Full pytest suite
python -m pytest tests/ -q

# Single test file
python -m pytest tests/test_md_enhanced_sampling.py -v

# Single test by name
python -m pytest tests/test_md_enhanced_sampling.py::test_metad_deposits -v
```

Tests that require a CUDA or OpenCL GPU are automatically skipped when that platform is unavailable. A Reference-only build still passes all non-GPU tests.

---

## Verifying the install

```python
import glued
import openmm as mm

f = glued.Force()
print("Available platforms:", [mm.Platform.getPlatform(i).getName()
                                for i in range(mm.Platform.getNumPlatforms())])
```

Expected output on a Linux machine with an NVIDIA GPU:
```
Available platforms: ['Reference', 'CPU', 'CUDA', 'OpenCL']
```

On macOS Apple Silicon:
```
Available platforms: ['Reference', 'CPU']
```

The low-level SWIG module is also directly importable:

```python
import gluedplugin as gp   # raw API — all GluedForce methods available
```

---

## Reference source trees (development machine only)

| Path | Contents |
|---|---|
| `/mnt/c/Users/Marvin/Desktop/openmm` | OpenMM 8.4.0 source |
| `/mnt/c/Users/Marvin/Desktop/openmm-plumed` | openmm-plumed plugin |
| `/mnt/c/Users/Marvin/Desktop/plumed2` | PLUMED2 source |
