# `kv12_s4` — Kv1.2 + S4 Z-displacement CV

167 800-atom voltage-gated potassium channel (Kv1.2 chimera 2R9R + 3:1
POPC:POPG + TIP3P + 150 mM KCl). The largest case in the suite by atom
count. CV is the Z-COM of the S4 helix Cα atoms minus the Z-COM of the
S1/S2/S3 anchor Cα atoms (built from 7 `add_position` CVs + one
`add_expression`). Bias: OPES_EXPLORE.

## Files

| File | Role |
|---|---|
| `benchmark.py` | The benchmark itself |
| `01_minimize_equilibrate.py` | Produces `output/equil/equil_final.chk`. ~3 hours wall on a 2070S — KV1.2 is 2.3× the size of KOR. |
| `03_production_glued.py` | Imported by `benchmark.py` for atom indices + system builder. |

## Equilibration timestep note

The original CHARMM-GUI 6-stage protocol switches dt from 0.001 to 0.002 ps
between stages 6.3 and 6.4 *while* softening the lipid restraints. For
this larger POPC:POPG system that double change blows up with NaN forces
at the start of stage 6.4. The shipped script keeps dt=0.001 ps for
stages 6.4–6.6 (twice the step count to maintain the same physical
duration), which is stable across the protocol.

## Data this case needs

- A CHARMM-GUI `openmm/` directory for Kv1.2 (PDB 2R9R chimera)
- `output/equil/equil_final.chk` produced by `01_minimize_equilibrate.py`

Run the equilibration once:

```bash
python benchmarks/kv12_s4/01_minimize_equilibrate.py \
    --input-dir /path/to/charmm-gui-7851356233/openmm \
    --output-dir benchmarks/kv12_s4/output/equil
```

Then either:

```bash
python benchmarks/kv12_s4/benchmark.py \
    --input-dir /path/to/charmm-gui-7851356233/openmm
```

or via the suite:

```bash
INPUT_DIR=/path/to/charmm-gui-7851356233/openmm \
    python benchmarks/run_suite.py --only kv12_s4
```
