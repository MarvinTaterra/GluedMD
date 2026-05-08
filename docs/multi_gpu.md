# Multi-GPU Guide

GLUED supports three multi-GPU usage patterns through the `MultiGPUManager` and `MultiWalkerPool` classes.

```python
from MultiGPUManager import MultiGPUManager, MultiWalkerPool
```

---

## Scenario A — One system across multiple GPUs

OpenMM distributes non-bonded work across multiple CUDA devices natively. GluedForce runs on the primary device without modification.

```python
import openmm as mm
from MultiGPUManager import MultiGPUManager

# Create a platform properties dict for devices 0 and 1
platform, props = MultiGPUManager.multi_device_platform(devices=[0, 1])

integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
ctx = mm.Context(system, integ, platform, props)
```

When to use: the system is large enough that non-bonded computation (PME) saturates a single GPU. OpenMM benchmarks show linear NB scaling up to 4 GPUs for systems above ~50 k atoms.

---

## Scenario B — One system per GPU (Replica Exchange)

Each simulation has its own OpenMM Context pinned to a specific GPU. `MultiGPUManager.build_replicas()` is a factory helper that passes the device index into each context constructor.

```python
from MultiGPUManager import MultiGPUManager
from ReplicaExchange import ReplicaExchange
import gluedplugin as gp

kT = gp.GluedForce.kTFromTemperature(300.0)

def make_replica(device_idx, target):
    """Build a single H-REUS window on the given GPU."""
    sys = build_system()
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(True)
    cv = f.addCollectiveVariable(gp.GluedForce.CV_DIHEDRAL, PHI_ATOMS, [])
    f.addBias(gp.GluedForce.BIAS_HARMONIC, [cv], [target, 200.0], [])
    sys.addForce(f)

    props = MultiGPUManager.cuda_properties(device_idx)
    integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
    ctx = mm.Context(sys, integ, mm.Platform.getPlatformByName("CUDA"), props)
    ctx.setPositions(start_positions)
    return ctx, f

targets = [-2.0, -1.0, 0.0, 1.0]   # window centres (rad)
replicas = MultiGPUManager.build_replicas(
    [lambda d, t=t: make_replica(d, t) for t in targets],
    devices=[0, 1, 2, 3]   # or [0, 0, 1, 1] if only 2 GPUs
)

re = ReplicaExchange(replicas, mode="H-REUS", kT=kT, seed=42)
re.run(n_cycles=500, steps_per_cycle=500)
print(f"Overall acceptance rate: {re.acceptance_rate:.1%}")
```

`MultiGPUManager.cuda_properties(device_idx)` returns `{"DeviceIndex": str(device_idx), "Precision": "mixed"}`. Pass additional properties as needed.

---

## Scenario C — Multiple walkers per GPU, shared bias across GPUs

This is the most powerful pattern: W walkers spread across G GPUs, where walkers within the same GPU group share their bias arrays in real time (GPU-atomic, no CPU round-trip), and bias state is periodically merged across GPU groups at the Python level.

```python
from MultiGPUManager import MultiGPUManager, MultiWalkerPool
import gluedplugin as gp

kT  = gp.GluedForce.kTFromTemperature(300.0)
N_WALKERS_PER_GPU = 4
DEVICES = [0, 1]   # GPU device indices

groups, forces = [], []
for g, dev in enumerate(DEVICES):
    ctxs, fors = [], []
    for w in range(N_WALKERS_PER_GPU):
        sys_ = build_system()
        f = gp.GluedForce()
        f.setUsesPeriodicBoundaryConditions(True)
        cv = f.addCollectiveVariable(gp.GluedForce.CV_DIHEDRAL, PHI_ATOMS, [])
        f.addBias(gp.GluedForce.BIAS_METAD, [cv],
                  [1.0, 0.35, 15.0, kT, -3.14159, 3.14159],
                  [500, 360, 1])
        sys_.addForce(f)

        props = MultiGPUManager.cuda_properties(dev)
        integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        ctx = mm.Context(sys_, integ,
                         mm.Platform.getPlatformByName("CUDA"), props)
        ctx.setPositions(start_positions)
        ctxs.append(ctx)
        fors.append(f)
    groups.append(ctxs)
    forces.append(fors)

pool = MultiWalkerPool(
    walker_groups=groups,
    force_groups=forces,
    bias_index=0,        # index of the MetaD bias to share
    sync_interval=100,   # steps between cross-GPU merges
    sync_mode="additive" # element-wise sum of MetaD grids
)
pool.run(n_steps=5_000_000)
```

### What happens under the hood

**Intra-GPU (within a group):** `MultiWalkerPool.__init__` calls `getMultiWalkerPtrs` on each group primary and `setMultiWalkerPtrs` on every secondary in the same group. From this point, all walkers in the group write their MetaD deposits atomically to the same GPU double-precision grid — at full GPU throughput with no synchronization overhead.

**Cross-GPU (between groups):** Every `sync_interval` steps `MultiWalkerPool` downloads the bias state from each group primary (one `getBiasState()` call per group), sums the MetaD grid arrays element-wise using `BiasStateMerger`, and uploads the merged state to every group primary via `setBiasState()`. This CPU round-trip is amortized over `sync_interval` steps; the cost is negligible when `sync_interval ≥ 50`.

### Sync modes

| `sync_mode` | MetaD behavior | OPES behavior |
|---|---|---|
| `"additive"` (default) | Element-wise grid sum | Copy from group with most kernels |
| `"broadcast"` | Copy group 0's grid to all others | Copy group 0's state to all others |

Use `"additive"` for true multi-walker sampling where every walker contributes hills to the shared landscape. Use `"broadcast"` when groups have independent bias targets (e.g. H-REUS windows) and you only want one-way propagation.

---

## Scenario C + Replica Exchange

`MultiWalkerPool` can simultaneously manage intra-GPU sharing AND drive H-REUS or T-REMD swaps between group primaries:

```python
pool = MultiWalkerPool(
    walker_groups=groups,
    force_groups=forces,
    bias_index=0,
    sync_interval=100,
    sync_mode="additive",
    re_mode="H-REUS",      # or "T-REMD"
    re_interval=500,        # steps between RE proposals
    kT=kT,
    seed=42
)
pool.run(n_steps=5_000_000)
print(f"RE acceptance rate: {pool.re_acceptance_rate:.1%}")
```

For T-REMD, provide `temperatures=[300, 320, 340, 360]` (one per group) instead of `kT`.

---

## Choosing `sync_interval`

The cross-GPU merge cost scales as `O(G × grid_size)` where G is the number of GPU groups and grid_size is the number of MetaD grid bins. For a 1-D grid with 360 bins and 4 GPUs, this is about 4 × 360 × 8 bytes = ~12 kB of transfers per merge. Even at `sync_interval=10` this is well below 1% overhead.

A practical guideline:

| MetaD pace | Recommended sync_interval |
|---|---|
| 500 steps | 500 (sync after every deposit) |
| 100 steps | 100–200 |
| 50 steps  | 50–100 |

Shorter intervals keep the cross-GPU grids more consistent (important when walkers are rapidly depositing in the same CV region); longer intervals reduce overhead.

---

## API reference

### `MultiGPUManager.multi_device_platform(devices) → (Platform, props)`

Returns a CUDA Platform and properties dict for Scenario A.

### `MultiGPUManager.build_replicas(factories, devices) → list`

Build `(context, force)` pairs for Scenario B. Each factory is called as `factory(device_idx) → (context, force)`.

### `MultiGPUManager.cuda_properties(device_idx, precision) → dict`

Return `{"DeviceIndex": ..., "Precision": ...}` for direct use with `mm.Context`.

### `MultiWalkerPool(walker_groups, force_groups, bias_index, sync_interval, sync_mode, re_mode, re_interval, kT, temperatures, bias_force_group, seed)`

See class docstring in `MultiGPUManager.py` for full parameter documentation.

### `MultiWalkerPool.run(n_steps)`

Run `n_steps` MD steps with automatic cross-GPU sync and optional RE.

### `MultiWalkerPool.re_acceptance_rate → float`

Overall RE acceptance rate across all group-pair swaps.

### `MultiWalkerPool.re_pair_acceptance_rate(i, j) → float`

RE acceptance rate for the (i, j) group pair.

---

## `BiasStateMerger`

Low-level utility for parsing and merging `getBiasState()` binary blobs. Can be used independently of `MultiWalkerPool` if you need custom merge logic.

```python
from MultiGPUManager import BiasStateMerger

blob_a = force_a.getBiasState()
blob_b = force_b.getBiasState()

merged = BiasStateMerger.merge_additive([blob_a, blob_b], force_a)

force_a.setBiasState(merged)
force_b.setBiasState(merged)
```
