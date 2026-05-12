# `kor_a100` — KOR + hand-crafted A100 score

71 873-atom κ-opioid-receptor + POPC + TIP3P (PDB 6B73). Measures GLUED
throughput on a realistic membrane-protein system with an analytical CV
(A100 score = 5 weighted Cα–Cα distances + constant) and OPES_METAD.

## Files in this case

| File | Role |
|---|---|
| `benchmark.py` | The benchmark — runs `_common.run_timed` after building the system |
| `01_minimize_equilibrate.py` | Produces `output/equil/equil_final.chk` (~45 min wall on a 2070S). Must run once before `benchmark.py`. |
| `03_production_glued.py` | Imported by `benchmark.py` for atom indices + system builder. Not executed by the benchmark itself. |

## Data this case needs

The benchmark expects:
- A CHARMM-GUI `openmm/` directory for KOR (PDB 6B73, `step5_input.{psf,pdb}` etc.)
- An `equil_final.chk` in `output/equil/`

Neither is shipped with the repo (CHARMM-GUI files are GB-sized, the
checkpoint is 8.5 MB and not portable). Run the equilibration once on
your machine:

```bash
python benchmarks/kor_a100/01_minimize_equilibrate.py \
    --input-dir /path/to/charmm-gui-7823302191/openmm \
    --output-dir benchmarks/kor_a100/output/equil
```

Then:

```bash
python benchmarks/kor_a100/benchmark.py \
    --input-dir /path/to/charmm-gui-7823302191/openmm
```

Or via the suite (which forwards `INPUT_DIR` / `EQUIL_DIR` from env):

```bash
INPUT_DIR=/path/to/charmm-gui-7823302191/openmm \
    python benchmarks/run_suite.py --only kor_a100
```
