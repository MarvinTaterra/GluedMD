# Python API Reference

## glued.Force  *(recommended)*

`glued.Force` is the recommended entry point. It inherits from `GluedForce` and adds:

- **Plain Python lists** everywhere — no `mm.vectori()` / `mm.vectord()` wrappers.
- **Named CV methods** — `add_distance`, `add_dihedral`, `add_rmsd`, etc.
- **Named bias methods** — `add_opes`, `add_metad`, `add_harmonic`, etc.
- **Temperature set once** on construction and reused by all temperature-dependent biases.

```python
import glued

force = glued.Force(pbc=True, temperature=300.0, group=1)
phi = force.add_dihedral([4, 6,  8, 14])
psi = force.add_dihedral([6, 8, 14, 16])
force.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500)
system.addForce(force)
```

### Constructor

```python
glued.Force(*, pbc=False, temperature=None, group=None)
```

| Parameter | Description |
|---|---|
| `pbc` | Enable periodic boundary conditions (default `False`) |
| `temperature` | System temperature in K — used by OPES, MetaD, EDS unless overridden per-call |
| `group` | OpenMM force group (0–31); leave `None` for the default group |

### CV methods

| Method | CV type |
|---|---|
| `add_distance(atoms)` | `CV_DISTANCE` |
| `add_angle(atoms)` | `CV_ANGLE` |
| `add_dihedral(atoms)` | `CV_DIHEDRAL` |
| `add_com_distance(group_a, group_b)` | `CV_COM_DISTANCE` |
| `add_gyration(atoms)` | `CV_GYRATION` |
| `add_coordination(group_a, group_b, r0, n=6, m=12, d0=0)` | `CV_COORDINATION` |
| `add_rmsd(atoms, reference_positions)` | `CV_RMSD` |
| `add_drmsd(atom_pairs, ref_distances)` | `CV_DRMSD` |
| `add_contact_map(pairs, *, r0, n=6, m=12, w=1, ref=0)` | `CV_CONTACTMAP` |
| `add_path(atoms, frames, lambda_)` | `CV_PATH` — returns index of *s*; *z* is at index+1 |
| `add_position(atom, component)` | `CV_POSITION` — component 0=x, 1=y, 2=z |
| `add_plane_distance(plane_atoms, query_atom)` | `CV_PLANE` |
| `add_projection(atom_a, atom_b, axis_a, axis_b)` | `CV_PROJECTION` |
| `add_dipole(atoms, component=0)` | `CV_DIPOLE` |
| `add_volume()` | `CV_VOLUME` |
| `add_cell(component)` | `CV_CELL` |
| `add_puckering(ring_atoms, component)` | `CV_PUCKERING` |
| `add_secondary_structure(atoms, subtype, r0=0.08)` | `CV_SECONDARY_STRUCTURE` — subtype: 0=α, 1=anti-β, 2=para-β; r0: switching cutoff (nm) |
| `add_pca(atoms, mean_positions, eigenvector)` | `CV_PCA` — projection onto a PC vector; mean_positions and eigenvector are (N,3) arrays in nm |
| `add_ermsd(atoms_per_residue, reference_positions, cutoff=2.4)` | `CV_ERMSD` — Bottaro eRMSD for RNA; atoms_per_residue is a list of [P, C4', N1/N9] triplets |
| `add_expression(expr, input_cvs)` | `CV_EXPRESSION` — Lepton algebraic string |
| `addPyTorchCV(model_path, atoms, params)` | `CV_PYTORCH` — TorchScript model; CUDA + libtorch required |

### Bias methods

| Method | Bias type |
|---|---|
| `add_harmonic(cvs, kappa, at, *, periodic=False)` | `BIAS_HARMONIC` |
| `add_upper_wall(cv, at, kappa, n=2, eps=1.0)` | `BIAS_UPPER_WALL` |
| `add_lower_wall(cv, at, kappa, n=2, eps=1.0)` | `BIAS_LOWER_WALL` |
| `add_linear(cvs, k)` | `BIAS_LINEAR` |
| `add_abmd(cvs, kappa, to)` | `BIAS_ABMD` |
| `add_moving_restraint(cvs, schedule)` | `BIAS_MOVING_RESTRAINT` — `schedule` = list of `(step, kappa, at)` |
| `add_opes(cvs, sigma, *, gamma=10, pace=500, temperature=None, sigma_min=None, max_kernels=100000)` | `BIAS_OPES` |
| `add_metad(cvs, sigma, height, pace, *, grid_min, grid_max, bins=100, periodic=False, gamma=None, temperature=None)` | `BIAS_METAD` |
| `add_eds(cvs, target, max_range=None, *, temperature=None)` | `BIAS_EDS` |

All raw `GluedForce` methods (`addCollectiveVariable`, `addBias`, enum constants, …) remain available on `glued.Force` instances.

---

## gluedplugin.GluedForce  *(low-level)*

The central class. Inherits from `openmm.Force` and is added to an OpenMM `System` in the normal way. `glued.Force` inherits from this class.

```python
import gluedplugin as gp
force = gp.GluedForce()
```

### Constructor

```python
GluedForce()
```

Creates an empty force with no CVs or biases.

---

### CV registration

#### `addCollectiveVariable(type, atoms, parameters) → int`

Register a CV. Returns the CV value index (used as input to `addBias`).  
`CV_PATH` returns the index of `s`; `z` is at `index + 1`.

See [Collective Variables](collective_variables.md) for per-type parameter details.

#### `addExpressionCV(expression, inputCVIndices) → int`

Register an algebraic CV built from existing CVs.

```python
ratio = force.addExpressionCV("cv0 / cv1", [d1, d2])
```

#### `addPyTorchCV(torchScriptPath, atomIndices, parameters) → int`

Register a TorchScript ML model as a CV (CUDA + libtorch required).

```python
ml_cv = force.addPyTorchCV("/path/model.pt", [0, 1, 2, 5], [])
```

#### `getNumCollectiveVariables() → int`

Total number of CV output values (PATH counts as 2).

#### `getNumCollectiveVariableSpecs() → int`

Number of `addCollectiveVariable` / `addExpressionCV` / `addPyTorchCV` calls.

#### `getCollectiveVariableParameters(idx) → (type, [atoms], [params])`

Python-friendly introspection of CV registration `idx`.

---

### Bias registration

#### `addBias(type, cvIndices, parameters, integerParameters) → int`

Register a bias. Returns the bias index (0-based, in registration order).  
See [Bias Methods](bias_methods.md) for per-type parameter details.

#### `getNumBiases() → int`

#### `getBiasParameters(idx) → (type, [cvIndices], [params], [intParams])`

Python-friendly introspection of bias registration `idx`.

---

### Configuration

#### `setUsesPeriodicBoundaryConditions(yes: bool)`
#### `usesPeriodicBoundaryConditions() → bool`

Controls whether PBC minimum-image is applied in CV kernels. Must be set to `True` for any system with a periodic box.

#### `setTemperature(kelvin: float)`
#### `getTemperature() → float`

Optional convenience — some biases read this; alternatively pass `kT` explicitly in parameters.

#### `setForceGroup(group: int)`

Assign this force to a non-default force group (0–31). Useful for T-REMD so that the bias energy can be isolated:

```python
force.setForceGroup(1)
# Then in T-REMD:
mm_only_energy = ctx.getState(getEnergy=True).getPotentialEnergy() \
               - ctx.getState(getEnergy=True, groups=1<<1).getPotentialEnergy()
```

---

### Runtime queries

#### `getCurrentCVValues(context) → [float]`

Return the CV values from the most recently completed force evaluation.

```python
values = force.getCurrentCVValues(context)
phi = values[phi_idx]
```

#### `getLastCVValues(context) → [float]`

Equivalent Python-level alias.

#### `getOPESMetrics(context, biasIndex) → [zed, rct, nker, neff]`

Convergence diagnostics for the `biasIndex`-th OPES bias:

| Return value | Meaning |
|---|---|
| `zed` | exp(log Z) — normalization estimate |
| `rct` | kT·log Z — c(t) indicator (flattens at convergence) |
| `nker` | number of compressed kernels |
| `neff` | effective sample size |

---

### Bias state (checkpoint / restore)

#### `getBiasState() → bytes`

Serialize all stateful bias data (MetaD grids, OPES kernels, ABMD floors, EDS λ, ExtLag s, MaxEnt λ) to a bytes object.

```python
blob = force.getBiasState()
with open("checkpoint.bin", "wb") as f:
    f.write(blob)
```

The binary format starts with a magic header (`GLUED`) and a version integer.

#### `setBiasState(blob: bytes)`

Restore a previously saved bias state. The current bias configuration (number and type of biases) must match what was saved.

```python
with open("checkpoint.bin", "rb") as f:
    force.setBiasState(f.read())
```

---

### XML serialization

`GluedForce` is registered with OpenMM's serialization framework. Save and reload a full simulation state:

```python
import openmm.app as app

# Save
xml = mm.XmlSerializer.serialize(system)
with open("system.xml", "w") as f:
    f.write(xml)

# Load
with open("system.xml") as f:
    system2 = mm.XmlSerializer.deserialize(f.read())

# Recover the GluedForce from the deserialized system
for i in range(system2.getNumForces()):
    raw_force = system2.getForce(i)
    gsp_force = gp.GluedForce.cast(raw_force)
    if gsp_force is not None:
        force = gsp_force
        break
```

---

### Multi-walker shared GPU arrays

These methods allow multiple OpenMM Contexts on the same GPU to share a single bias grid (MetaD) or kernel list (OPES). All deposits are atomic on the GPU — no CPU merge step, no periodic synchronization.

> Requires CUDA platform.

#### `getMultiWalkerPtrs(context, biasIdx) → [long long]`

Get raw device pointers for the primary walker's bias GPU arrays.

- `BIAS_METAD`: returns `[grid_ptr]`
- `BIAS_OPES`: returns `[centers, sigmas, logweights, numKernels, numAllocated]`

#### `setMultiWalkerPtrs(context, biasIdx, ptrs)`

Wire this (secondary) walker's bias kernels to use the primary's GPU arrays. Must be called **after** the secondary's Context is created.

```python
# Primary deposits into its own grid
ptrs = f_primary.getMultiWalkerPtrs(ctx_primary, bias_idx=0)

# Secondary reads from the same grid
f_secondary.setMultiWalkerPtrs(ctx_secondary, bias_idx=0, ptrs=ptrs)
```

See [Examples — Multi-walker MetaD](examples.md#multi-walker-metadynamics) for a full recipe.

---

### Helper statics

#### `GluedForce.barrierToGamma(barrier_kJ, kT_kJ) → float`

Convert a PLUMED-style `BARRIER` (kJ/mol) to the biasfactor γ used by `addBias`.  
`γ = BARRIER / kT`

#### `GluedForce.kTFromTemperature(T_kelvin) → float`

`kT = R * T` in kJ/mol. R = 8.314462618 × 10⁻³ kJ/mol/K.

```python
kT  = gp.GluedForce.kTFromTemperature(300.0)   # → 2.479 kJ/mol
gam = gp.GluedForce.barrierToGamma(30.0, kT)   # → 12.1
```

---

---

## OPESConvergenceReporter

Automatically stops an OPES simulation when the free-energy estimate has converged, avoiding unnecessary compute beyond what is needed.

```python
from OPESConvergenceReporter import OPESConvergenceReporter
```

### Convergence criteria

The reporter supports three convergence criteria, selectable via `criterion=`:

| `criterion` | Signal | Tol units | Notes |
|---|---|---|---|
| `'rct_relative'` *(default)* | `rct / max(|rct|, kT)` | dimensionless (~0.01) | Scale-invariant; robust across OPES variants |
| `'neff_rate'`               | `neff / step`           | dimensionless (~0.02) | Variant-agnostic; **recommended for `mode='explore'`** where rct is non-stationary by design |
| `'rct_absolute'`            | `rct`                   | kJ/mol (~0.05)        | Original behavior; absolute drift in kJ/mol |

The reporter declares convergence only after `min_consecutive_passes` checks
(default 3) consecutively pass `|Δsignal| < tol`. The test does not start
until at least `min_kernels` kernels have been deposited and `min_steps` MD
steps have elapsed (warm-up gates). After convergence is detected the
simulation runs for `post_convergence_steps` more steps before stopping.

### Constructor

```python
OPESConvergenceReporter(
    force, bias_idx=0, *,
    criterion='rct_relative', tol=0.01,
    check_interval=1000,
    min_consecutive_passes=3,
    min_kernels=50, min_steps=0,
    post_convergence_steps=50_000,
    file=None, verbose=True,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `force` | GluedForce | — | The force containing the OPES bias |
| `bias_idx` | int | 0 | Index of the OPES bias (0 for the first/only bias) |
| `criterion` | str | `'rct_relative'` | One of `'rct_relative'`, `'neff_rate'`, `'rct_absolute'` |
| `tol` | float | 0.01 | Tolerance — units depend on `criterion` (see table above) |
| `check_interval` | int | 1000 | Steps between successive checks |
| `min_consecutive_passes` | int | 3 | Consecutive passing checks required before declaring convergence |
| `min_kernels` | int | 50 | Don't start checking until at least this many kernels deposited |
| `min_steps` | int | 0 | Don't start checking before this MD step count |
| `post_convergence_steps` | int | 50 000 | Additional steps after convergence is detected |
| `file` | str, file, or None | None | Log output destination; defaults to stdout |
| `verbose` | bool | True | Print one line per check; set False for events only |

### Properties

| Property | Type | Description |
|---|---|---|
| `converged` | bool | True once the rct criterion has been satisfied |
| `done` | bool | True once post-convergence sampling is also complete |
| `converged_at_step` | int or None | MD step index where convergence was first detected |

### Methods

#### `run(simulation, max_steps)`

Attach the reporter, run in `check_interval`-sized batches until `done` or `max_steps` is reached, then detach. This is the primary usage pattern.

#### `force_converge(simulation)`

Manually mark the run as converged — useful for interactive sessions where external evidence (e.g. visual inspection of the FES) shows the run is good enough. Triggers the `post_convergence_steps` window normally.

#### `describeNextReport(simulation)` / `report(simulation, state)`

Standard OpenMM reporter interface — attach manually if you need finer loop control.

### Picking a criterion

For most users **`'rct_relative'` (default)** with `tol=0.01` is the right answer — it's dimensionless, robust to noise, and works for all OPES variants.

For **OPES_METAD_EXPLORE** (`force.add_opes(..., mode='explore')`) prefer **`'neff_rate'`**: the EXPLORE bias is non-stationary by design, so rct never plateaus the way it does in standard METAD. `neff/step` does stabilize once the bias has equilibrated.

Use `'rct_absolute'` only when porting from existing scripts that depended on the old kJ/mol-based behavior.

### Usage

```python
from OPESConvergenceReporter import OPESConvergenceReporter
import glued

force = glued.Force(pbc=True, temperature=300.0)
phi = force.add_dihedral(PHI_ATOMS)
force.add_opes([phi], sigma=0.05, gamma=10.0, pace=500)
system.addForce(force)

simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)

reporter = OPESConvergenceReporter(
    force,
    tol=0.05,                    # kJ/mol — tighten for better convergence
    check_interval=2000,
    post_convergence_steps=100_000,
    file='convergence.log',
    verbose=True,
)
reporter.run(simulation, max_steps=10_000_000)

print(f"Converged at step {reporter.converged_at_step}")
print(f"Done at step    {simulation.currentStep}")
```

**Manual loop** (when you need other reporters or checkpointing to also fire):

```python
simulation.reporters.append(reporter)
simulation.reporters.append(COLVARReporter('colvar.dat', 500, force))

while not reporter.done and simulation.currentStep < 10_000_000:
    simulation.step(reporter._interval)   # step one check window at a time
```

### Verbose log format

```
step          0  rct=    0.0000 kJ/mol  |Δrct|=      --- kJ/mol  nker=     0  neff=     0.0
step       2000  rct=    1.2341 kJ/mol  |Δrct|=   1.2341 kJ/mol  nker=     4  neff=    12.3
step       4000  rct=    1.8802 kJ/mol  |Δrct|=   0.6461 kJ/mol  nker=     8  neff=    24.7
...
[OPESConvergence] Converged at step 128000: rct=4.2317 kJ/mol, |Δrct|=0.0031 < tol=0.0500. Running 100,000 more steps.
...
[OPESConvergence] Post-convergence run complete at step 228000. Total post-convergence steps: 100,000.
```

---

---

## COLVARReporter

Writes CV values to a PLUMED-compatible `COLVAR` file at regular intervals.

```python
from COLVARReporter import COLVARReporter
```

### Constructor

```python
COLVARReporter(file, reportInterval, force, cvNames=None, append=False)
```

| Parameter | Type | Description |
|---|---|---|
| `file` | str or file | Output path or open file object |
| `reportInterval` | int | Steps between output lines |
| `force` | GluedForce | The force whose CVs to log |
| `cvNames` | list[str] or None | Column headers; defaults to `cv0`, `cv1`, … |
| `append` | bool | If True, appends to existing file (skips header) |

### Output format

```
#! FIELDS time phi psi
  0.00000    -1.23456     2.34567
  1.00000    -1.24512     2.31201
```

Time is in picoseconds. CV values are in native OpenMM units (nm for distances, rad for angles).

### Usage

```python
force = gp.GluedForce()
# ... add CVs and biases ...
system.addForce(force)

simulation = app.Simulation(...)
simulation.reporters.append(
    COLVARReporter('colvar.dat', 500, force, cvNames=['phi', 'psi'])
)
simulation.step(1_000_000)
```

---

---

## ReplicaExchange

Drives replica exchange between N OpenMM Contexts. Supports H-REUS (Hamiltonian Replica Exchange Umbrella Sampling) and T-REMD (Temperature Replica Exchange MD).

```python
from ReplicaExchange import ReplicaExchange
```

### Constructor

```python
ReplicaExchange(replicas, mode="H-REUS", *,
                kT=None, temperatures=None,
                bias_force_group=None,
                scheme="neighbor", seed=None)
```

| Parameter | Description |
|---|---|
| `replicas` | `list[(Context, GluedForce)]` — one entry per replica, ordered along the exchange ladder |
| `mode` | `"H-REUS"` or `"T-REMD"` |
| `kT` | Thermal energy in kJ/mol — required for H-REUS |
| `temperatures` | List of temperatures in K, one per replica — required for T-REMD |
| `bias_force_group` | Force group index of GluedForce — used by T-REMD to isolate MM energy |
| `scheme` | `"neighbor"` (alternating even/odd pairs, standard) or `"all"` (all unique pairs) |
| `seed` | RNG seed for reproducibility |

### H-REUS criterion

All replicas run at the same temperature but with different bias parameters (e.g., different harmonic window centres). The Metropolis criterion is:

```
Δ = −β [ E(x_j | H_i) + E(x_i | H_j) − E(x_i | H_i) − E(x_j | H_j) ]
P_acc = min(1, exp(Δ))
```

The MM contributions cancel analytically, leaving only bias energy differences.

### T-REMD criterion

All replicas share the same Hamiltonian but run at different temperatures. Sugita-Okamoto criterion:

```
Δ = (β_i − β_j) * (U_MM(x_i) − U_MM(x_j))
P_acc = min(1, exp(Δ))
```

Velocities are rescaled by `√(T_new / T_old)` after each accepted swap.

### Methods

#### `run(n_cycles, steps_per_cycle=500)`

Run `n_cycles` exchange cycles. Each cycle: step all replicas for `steps_per_cycle` MD steps, then attempt swaps.

#### `acceptance_rate → float`

Overall fraction of accepted swap proposals across all cycles.

#### `pair_acceptance_rate(i, j) → float`

Acceptance rate for the specific (i, j) pair.

#### `sync_bias_state(source_idx, target_indices=None)`

Broadcast bias state from `source_idx` to other replicas. Used for multi-walker MetaD where all replicas should share a growing bias:

```python
re.sync_bias_state(0)   # broadcast replica 0's bias to all others
```

### H-REUS example

```python
from ReplicaExchange import ReplicaExchange
import glued

kT = glued.GluedForce.kTFromTemperature(298.0)   # 2.479 kJ/mol

targets = [-2.0, -1.0, 0.0, 1.0]   # window centres (rad)
replicas = []
for target in targets:
    f = glued.Force(pbc=True)
    cv = f.add_dihedral(PHI_ATOMS)
    f.add_harmonic(cv, kappa=200.0, at=target, periodic=True)
    system.addForce(f)
    ctx = mm.Context(system, mm.LangevinMiddleIntegrator(298, 1.0, 0.002), platform)
    ctx.setPositions(positions)
    replicas.append((ctx, f))

re = ReplicaExchange(replicas, mode="H-REUS", kT=kT, scheme="neighbor", seed=42)
re.run(n_cycles=500, steps_per_cycle=500)
print(f"Acceptance rate: {re.acceptance_rate:.1%}")
```

### T-REMD example

```python
temperatures = [300, 320, 340, 360]
replicas = []
for T in temperatures:
    f = glued.Force(pbc=True, group=1)   # group=1 isolates bias energy
    # ... add CVs and biases ...
    ctx = mm.Context(system, mm.LangevinMiddleIntegrator(T, 1.0, 0.002), platform)
    replicas.append((ctx, f))

re = ReplicaExchange(
    replicas, mode="T-REMD",
    temperatures=temperatures, bias_force_group=1, seed=99
)
re.run(n_cycles=200, steps_per_cycle=1000)
```
