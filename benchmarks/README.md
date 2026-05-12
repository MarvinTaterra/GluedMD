# GLUED benchmark suite

Six representative cases spanning system size (22 → 167 800 atoms) and CV
kind (analytical → tiny TorchScript → large TorchScript). Every case runs
**only GLUED** — no PLUMED comparisons, no head-to-head against any other
plugin. The purpose of the suite is to measure GLUED's per-step throughput
across the workload mix users actually run.

## Design constraints

1. **Zero CPU↔GPU round-trips inside the timed loop.** The timed phase is a
   single `integrator.step(N)` call. No `getState(getPositions=…)`, no
   COLVAR write, no `getLastCVValues`, no DCD output. Even PyTorch CVs run
   their forward + autograd backward on the same CUDA stream as the
   integrator — `torch::from_blob` wraps the GPU position buffer
   zero-copy, `c10::cuda::CUDAStreamGuard` keeps everything in stream.
   See `platforms/common/src/CommonGluedKernels.cpp` L4476–4528 for the
   wiring.
2. **CUDA-only.** Non-CUDA platforms exist (Reference, OpenCL) but the
   PyTorch-CV fast path is CUDA-specific. `_common.best_platform_cuda_or_skip()`
   raises a clean skip rather than producing meaningless numbers on
   Reference.
3. **Self-contained where possible.** `ad_vacuum` ships its PDB. `ad_water`
   solvates the same PDB at startup with `Modeller`. The KOR/KV1.2 cases
   need user-supplied CHARMM-GUI data — they print a skip line, not an
   error, when that data isn't present.

## Cases

| Case | Atoms | CV | Bias | Notes |
|---|---:|---|---|---|
| `ad_vacuum` | 22 | `CV_DIHEDRAL` φ | `OPES_METAD` | Self-contained (ships `adp.pdb`). Floor cost of GLUED. |
| `ad_water` | ~2 500 | `CV_DIHEDRAL` φ | `OPES_METAD` | Solvates ADP in TIP3P at startup. PME cost over `ad_vacuum`. |
| `kor_a100` | 71 873 | A100 score (5×distance + expression) | `OPES_METAD` | Requires KOR CHARMM-GUI + equil checkpoint. |
| `kor_deepcv_small` | 71 873 | `CV_PYTORCH` (10 atoms, ~2k params) | `OPES_EXPLORE` | Ships the small `.pt`. PyTorch overhead with cheap model. |
| `kor_deepcv_large` | 71 873 | `CV_PYTORCH` (285 atoms, ~21M params) | `OPES_EXPLORE` | Model not shipped (80 MB). See case README. |
| `kv12_s4` | 167 800 | S4 Z-COM diff (7×position + expression) | `OPES_EXPLORE` | Largest analytical-CV case. |

## Running the suite

```bash
# Activate openmm_env first
python benchmarks/run_suite.py
```

Each case is run as its own Python subprocess so they don't interfere.
Missing-data cases print a `[skip]` and the suite continues. Output is a
per-case JSON in `benchmarks/results/<case>.json` plus an aggregated
`benchmarks/results/suite.json`.

Filtering:

```bash
python benchmarks/run_suite.py --only ad_vacuum,ad_water
python benchmarks/run_suite.py --skip kor_deepcv_large
```

Per-case (e.g. for debugging or rerunning with different step counts):

```bash
python benchmarks/ad_vacuum/benchmark.py --steps 100000
python benchmarks/kor_a100/benchmark.py --input-dir "$CHARMM_GUI" --steps 10000
```

## Data the suite needs

Light data ships with the repo:

```
benchmarks/ad_vacuum/adp.pdb           ~3 KB
benchmarks/kor_deepcv_small/deepcv.pt  ~21 KB
```

Heavy data the user must provide locally:

| Case | What you need | Where benchmark.py looks |
|---|---|---|
| `kor_a100` | KOR CHARMM-GUI `openmm/` + `equil_final.chk` | `--input-dir` / `--equil-dir` or `$INPUT_DIR` / `$EQUIL_DIR` |
| `kor_deepcv_small` | same as `kor_a100` | same |
| `kor_deepcv_large` | same as `kor_a100` + 80 MB `deepcv_allca.pt` | place model at `benchmarks/kor_deepcv_large/deepcv_allca.pt` |
| `kv12_s4` | Kv1.2 CHARMM-GUI `openmm/` + `equil_final.chk` | `--input-dir` / `--equil-dir` or `$INPUT_DIR` / `$EQUIL_DIR` |

Equilibrations are produced by the corresponding `01_minimize_equilibrate.py`
inside each case directory.

## What "0 round-trips" means in practice

Inside `run_timed()` in `_common.py`:

```python
integrator.step(warmup_steps)   # untimed
context.getState()              # stream sync (no buffer download)

t0 = time.perf_counter()
integrator.step(timed_steps)    # the timed call
context.getState()              # stream sync — flushes pending GPU work
elapsed = time.perf_counter() - t0
```

`context.getState()` with no flags is a stream sync that doesn't copy any
buffer back to the host. The CV+bias machinery, PyTorch model forward+
backward (for the deep CV cases), and chain-rule force scatter all run on
the GPU between the two `step()` boundaries.

## Regression tracking

`check_regression.py` compares fresh `results/*.json` against
`baselines/*.json`:

```bash
python benchmarks/check_regression.py results/ baselines/ --threshold 0.90
```

Update baselines after a deliberate improvement:

```bash
cp results/<case>.json baselines/<gpu>_<case>.json
```

## Layout

```
benchmarks/
├── README.md                       this file
├── _common.py                      timing + result-writing helpers
├── run_suite.py                    top-level runner
├── check_regression.py             compares results/ to baselines/
├── baselines/                      per-GPU per-case reference numbers
├── results/                        per-run output (created by run_suite.py)
├── ad_vacuum/    benchmark.py + adp.pdb
├── ad_water/     benchmark.py
├── kor_a100/     benchmark.py + 01_minimize_equilibrate.py + 03_production_glued.py + …
├── kor_deepcv_small/  benchmark.py + deepcv.pt
├── kor_deepcv_large/  benchmark.py + (download required, see local README)
└── kv12_s4/      benchmark.py + 01_minimize_equilibrate.py + 03_production_glued.py + …
```
