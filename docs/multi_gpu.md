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

## Scenario C — Multiple walkers sharing one bias

W walkers spread across G GPUs all contribute to one bias. `MultiWalkerPool`
keeps the shared bias consistent by moving bias-state checkpoints between the
walkers' Contexts; groups on different devices step concurrently.

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

### How the shared bias is kept consistent

The pool shares only the bias selected by `bias_index`; every other bias keeps
its own per-walker history (so H-REUS windows can share a MetaD grid while
keeping separate restraints).

| Shared bias | Mechanism |
|---|---|
| MetaD, PBMetaD | Every `sync_interval` steps the hills each walker deposited since the previous merge are summed and written back to every walker. Hills are additive, so this is exact up to the merge latency. Groups run concurrently between merges. |
| OPES, OPES-expanded, multithermal | The walkers advance one MD step at a time in a fixed order, each starting from the state the previous walker left. A compressed kernel table with running statistics cannot be merged after the fact, so this is the only exact scheme; it is serial and costs a checkpoint transfer per walker step. Any positive `sync_interval` enables it. |

`sync_interval=0` leaves every walker's bias independent (use it for H-REUS
windows that only exchange configurations). `sync_mode="broadcast"` copies group
0's grid to everyone at each sync instead of merging. All walkers must start from
the same shared-bias state and define the shared bias identically. Advance the
walkers only through `pool.run()`.

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

For MetaD/PBMetaD, merge after every deposit or every few deposits: between
merges a walker does not see the others' newest hills. A merge downloads and
re-uploads each walker's grid (`8 × grid points` bytes per walker), which is
negligible for typical 1-D/2-D grids at intervals of a few hundred steps.

| MetaD pace | Recommended sync_interval |
|---|---|
| 500 steps | 500 (sync after every deposit) |
| 100 steps | 100–200 |
| 50 steps  | 50–100 |

For the OPES family the interval only switches sharing on or off; the state is
transferred on every walker step.

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

Run `n_steps` MD steps with automatic bias synchronization and optional RE.
Syncs and exchange attempts fall on absolute multiples of their intervals, so
repeated short calls follow the same schedule as one long call.

### `MultiWalkerPool.re_acceptance_rate → float`

Overall RE acceptance rate across all group-pair swaps.

### `MultiWalkerPool.re_pair_acceptance_rate(i, j) → float`

RE acceptance rate for the (i, j) group pair.

---

## `BiasStateMerger`

Low-level utility for parsing, repacking and merging `getBiasState()` blobs. Can
be used independently of `MultiWalkerPool` for custom MetaD merge logic.

```python
from MultiGPUManager import BiasStateMerger

blob_a = force_a.getBiasState(ctx_a)
blob_b = force_b.getBiasState(ctx_b)

merged = BiasStateMerger.merge_additive([blob_a, blob_b], force_a)   # sums all MetaD grids

force_a.setBiasState(merged, ctx_a)
force_b.setBiasState(merged, ctx_b)
```

`merge_additive` sums every MetaD/PBMetaD grid and copies all other sections
from the first blob. For a persistent shared grid use
`merge_additive_incremental(blobs, force, baselines)` with the baselines
returned by the previous call, so hills are not counted twice. `parse(blob,
force)` and `pack(state)` expose the sections for anything else.
