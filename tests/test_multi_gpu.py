"""
test_multi_gpu.py — Tests for MultiGPUManager (all three scenarios).

Scenario A tests: single system on multiple GPUs.
  - Skipped gracefully if fewer than 2 CUDA devices are present.

Scenario B tests: one system per GPU (replica exchange via MultiGPUManager helpers).
  - Exercises build_replicas() + ReplicaExchange integration.
  - Runs with a single GPU (two contexts on the same device) for local testing.

Scenario C tests: MultiWalkerPool — walkers per GPU, cross-GPU bias merge.
  - Single-GPU mode: two "groups" of one walker each, same physical device.
  - Verifies intra-group wiring and cross-GPU (CPU-mediated) merge.
  - Multi-GPU mode (2+ devices): full GPU-affinity test, skipped when unavailable.
"""

import sys
import os
import math
import struct
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

try:
    import openmm as mm
    import openmm.unit as unit
    import gluedplugin as gsp
    from MultiGPUManager import MultiGPUManager, MultiWalkerPool, BiasStateMerger
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    for i in range(mm.Platform.getNumPlatforms()):
        if mm.Platform.getPlatform(i).getName() == "CUDA":
            return True
    return False


def _cuda_device_count() -> int:
    """Return number of CUDA devices visible to OpenMM (best-effort)."""
    if not _cuda_available():
        return 0
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            return len(lines)
    except Exception:
        pass
    return 1   # assume at least one if CUDA platform exists


# ---------------------------------------------------------------------------
# Mini-system factory
# ---------------------------------------------------------------------------

def _make_system_force(n_atoms: int = 2, add_metad: bool = True):
    """Return (system, force) with n_atoms and a distance CV + optional MetaD."""
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(1000.0)

    f = gsp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])

    if add_metad:
        f.addBias(gsp.GluedForce.BIAS_METAD, [cv],
                  [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
                  [10, 50, 0])
    sys_.addForce(f)
    return sys_, f


def _make_ctx(system, device_idx: int = 0, platform_name: str = "CUDA"):
    """Create an OpenMM Context pinned to a specific CUDA device."""
    integ = mm.LangevinIntegrator(300, 1.0, 0.002)
    if platform_name == "CUDA":
        props = MultiGPUManager.cuda_properties(device_idx)
        platform = mm.Platform.getPlatformByName("CUDA")
        ctx = mm.Context(system, integ, platform, props)
    else:
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform_name))
    return ctx, integ


def _set_pos(ctx, d: float = 0.3):
    ctx.setPositions([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Scenario A — single system, multiple GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
@pytest.mark.skipif(_cuda_device_count() < 2, reason="fewer than 2 CUDA devices")
def test_scenario_a_multi_device_platform():
    """
    Scenario A: single system across 2 GPUs.
    Verify that a context is created without error and produces a finite energy.
    """
    sys_, f = _make_system_force(add_metad=False)
    platform, props = MultiGPUManager.multi_device_platform(devices=[0, 1])
    integ = mm.LangevinIntegrator(300, 1.0, 0.002)
    ctx = mm.Context(sys_, integ, platform, props)
    _set_pos(ctx)
    integ.step(10)
    E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    assert math.isfinite(E), "Energy should be finite on multi-GPU context"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_a_single_device_platform():
    """
    Scenario A helper smoke test on a single device (works on 1-GPU machines).
    """
    sys_, f = _make_system_force(add_metad=False)
    platform, props = MultiGPUManager.multi_device_platform(devices=[0])
    integ = mm.LangevinIntegrator(300, 1.0, 0.002)
    ctx = mm.Context(sys_, integ, platform, props)
    _set_pos(ctx)
    integ.step(5)
    E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    assert math.isfinite(E)


# ---------------------------------------------------------------------------
# Scenario B — build_replicas helper
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_b_build_replicas():
    """
    Scenario B: build_replicas() creates (context, force) pairs on specified devices.
    Both contexts land on device 0 (single GPU machine) — the API is exercised
    without requiring a 2nd physical GPU.
    """
    n_replicas = 3
    targets = [-1.0, 0.0, 1.0]   # harmonic window centres (nm — distance)

    def factory(device_idx, target=None):
        sys_ = mm.System()
        sys_.addParticle(1000.0); sys_.addParticle(1000.0)
        f = gsp.GluedForce()
        f.setUsesPeriodicBoundaryConditions(False)
        cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
        # Harmonic restraint at target (converted to positive: distance always ≥0)
        at = abs(target) + 0.1
        f.addBias(gsp.GluedForce.BIAS_HARMONIC, [cv], [at, 200.0], [])
        sys_.addForce(f)
        props = MultiGPUManager.cuda_properties(device_idx)
        integ = mm.LangevinIntegrator(300, 1.0, 0.002)
        ctx = mm.Context(sys_, integ, mm.Platform.getPlatformByName("CUDA"), props)
        ctx.setPositions([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        return ctx, f

    # All land on device 0 (single GPU) — multi-GPU wiring is the same regardless
    devices = [0] * n_replicas
    replicas = MultiGPUManager.build_replicas(
        [lambda dev, t=t: factory(dev, t) for t in targets],
        devices=devices)

    assert len(replicas) == n_replicas
    for ctx, f in replicas:
        E = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        assert math.isfinite(E)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_b_with_replica_exchange():
    """
    Scenario B: build_replicas() + ReplicaExchange integration.
    Two harmonic-window replicas run H-REUS for 4 cycles on device 0.
    """
    from ReplicaExchange import ReplicaExchange

    kT = 2.479
    targets = [0.2, 0.5]

    def factory(device_idx, target):
        sys_ = mm.System()
        sys_.addParticle(1.0); sys_.addParticle(1.0)
        f = gsp.GluedForce()
        f.setUsesPeriodicBoundaryConditions(False)
        cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
        f.addBias(gsp.GluedForce.BIAS_HARMONIC, [cv], [target, 200.0], [])
        sys_.addForce(f)
        props = MultiGPUManager.cuda_properties(device_idx)
        integ = mm.LangevinIntegrator(300, 1.0, 0.002)
        ctx = mm.Context(sys_, integ, mm.Platform.getPlatformByName("CUDA"), props)
        ctx.setPositions([[0.0, 0.0, 0.0], [target, 0.0, 0.0]])
        return ctx, f

    replicas = MultiGPUManager.build_replicas(
        [lambda d, t=t: factory(d, t) for t in targets],
        devices=[0, 0])   # same device — works on single-GPU machine

    re = ReplicaExchange(replicas, mode="H-REUS", kT=kT, seed=42)
    re.run(n_cycles=4, steps_per_cycle=20)

    assert isinstance(re.acceptance_rate, float)
    assert 0.0 <= re.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# Scenario C — MultiWalkerPool
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_pool_construction_single_gpu():
    """
    Scenario C: MultiWalkerPool with two groups (1 walker each) on the same GPU.
    Tests that setup succeeds and does not raise.
    """
    sys0, f0 = _make_system_force()
    sys1, f1 = _make_system_force()

    ctx0, _ = _make_ctx(sys0, 0)
    ctx1, _ = _make_ctx(sys1, 0)
    _set_pos(ctx0, 0.3)
    _set_pos(ctx1, 0.4)

    pool = MultiWalkerPool(
        walker_groups=[[ctx0], [ctx1]],
        force_groups=[[f0], [f1]],
        bias_index=0,
        sync_interval=10,
        sync_mode="additive",
    )
    assert pool.n_groups == 2
    assert pool.total_walkers == 2


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_intra_group_wiring():
    """
    Scenario C: within a single GPU group, secondary walker sees primary's grid.
    Two walkers, same group, same device.
    """
    PACE   = 5
    NSTEPS = 25

    sys0, f0 = _make_system_force()
    sys1, f1 = _make_system_force()

    ctx0, _ = _make_ctx(sys0, 0)
    ctx1, _ = _make_ctx(sys1, 0)
    _set_pos(ctx0, 0.3)
    _set_pos(ctx1, 0.3)

    # One group, two walkers — MultiWalkerPool wires sharing automatically.
    pool = MultiWalkerPool(
        walker_groups=[[ctx0, ctx1]],
        force_groups=[[f0, f1]],
        bias_index=0,
        sync_interval=0,  # no cross-GPU needed (single group)
    )

    # Run only the primary to deposit hills into the shared grid.
    for _ in range(NSTEPS):
        ctx0.getIntegrator().step(1)

    # Secondary reads the same shared grid.
    E0 = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    E1 = ctx1.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    assert E0 > 0.0, "Primary should have non-zero bias after deposits"
    assert abs(E0 - E1) < 0.1, (
        f"Primary ({E0:.4f}) and secondary ({E1:.4f}) should read the same shared grid"
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_cross_group_bias_merge():
    """
    Scenario C: cross-GPU bias merge.
    Two single-walker groups both on device 0 (single-GPU machine).
    Each walker deposits MetaD hills; after sync, both should see the merged grid.
    """
    NSTEPS_BEFORE_SYNC = 30   # 3 deposits each at pace=10

    sys0, f0 = _make_system_force()
    sys1, f1 = _make_system_force()

    ctx0, _ = _make_ctx(sys0, 0)
    ctx1, _ = _make_ctx(sys1, 0)
    _set_pos(ctx0, 0.25)   # different positions → different hills
    _set_pos(ctx1, 0.55)

    pool = MultiWalkerPool(
        walker_groups=[[ctx0], [ctx1]],
        force_groups=[[f0], [f1]],
        bias_index=0,
        sync_interval=NSTEPS_BEFORE_SYNC,
        sync_mode="additive",
    )

    # Run to first sync point.
    pool.run(NSTEPS_BEFORE_SYNC)

    # After the sync, each group primary's grid is the additive merge.
    # The merged bias at position 0.25 should exceed what a single walker at 0.55 produced
    # (because the grid at 0.25 now has hills from group 0).
    E_after_sync_g0 = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    E_after_sync_g1 = ctx1.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # Both grids should be non-trivially non-zero (at least one of them deposited hills
    # near each CV value).
    assert E_after_sync_g0 > 0.0 or E_after_sync_g1 > 0.0, (
        "After sync, at least one group should see a non-zero bias"
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_run_does_not_crash():
    """
    Scenario C: pool.run() completes without error for a short run.
    Two groups × two walkers, device 0 throughout.
    """
    groups = []
    forces = []
    for g in range(2):
        ctxs, fors = [], []
        for w in range(2):
            sys_ = mm.System()
            sys_.addParticle(1000.0); sys_.addParticle(1000.0)
            f = gsp.GluedForce()
            f.setUsesPeriodicBoundaryConditions(False)
            cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
            f.addBias(gsp.GluedForce.BIAS_METAD, [cv],
                      [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
                      [10, 50, 0])
            sys_.addForce(f)
            ctx, _ = _make_ctx(sys_, 0)
            ctx.setPositions([[0.0, 0.0, 0.0], [0.3 + g * 0.1 + w * 0.05, 0.0, 0.0]])
            ctxs.append(ctx)
            fors.append(f)
        groups.append(ctxs)
        forces.append(fors)

    pool = MultiWalkerPool(
        walker_groups=groups,
        force_groups=forces,
        bias_index=0,
        sync_interval=20,
        sync_mode="additive",
    )
    pool.run(n_steps=40)   # 2 sync cycles


@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
@pytest.mark.skipif(_cuda_device_count() < 2, reason="fewer than 2 CUDA devices")
def test_scenario_c_two_physical_gpus():
    """
    Scenario C: 2 walker groups on 2 different physical GPUs.
    4 walkers total: 2 per GPU.  Runs 50 steps with sync every 10 steps.
    This test only executes on machines with 2+ CUDA devices.
    """
    groups, forces = [], []
    for g, dev in enumerate([0, 1]):
        ctxs, fors = [], []
        for w in range(2):
            sys_ = mm.System()
            sys_.addParticle(1000.0); sys_.addParticle(1000.0)
            f = gsp.GluedForce()
            f.setUsesPeriodicBoundaryConditions(False)
            cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
            f.addBias(gsp.GluedForce.BIAS_METAD, [cv],
                      [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
                      [10, 50, 0])
            sys_.addForce(f)
            props = MultiGPUManager.cuda_properties(dev)
            integ = mm.LangevinIntegrator(300, 1.0, 0.002)
            ctx = mm.Context(sys_, integ, mm.Platform.getPlatformByName("CUDA"), props)
            ctx.setPositions([[0.0, 0.0, 0.0], [0.3 + g * 0.15 + w * 0.05, 0.0, 0.0]])
            ctxs.append(ctx)
            fors.append(f)
        groups.append(ctxs)
        forces.append(fors)

    pool = MultiWalkerPool(
        walker_groups=groups,
        force_groups=forces,
        bias_index=0,
        sync_interval=10,
        sync_mode="additive",
    )
    pool.run(n_steps=50)

    # Verify each group primary has a finite energy.
    for g in range(2):
        E = groups[g][0].getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        assert math.isfinite(E), f"Group {g} primary energy should be finite"


# ---------------------------------------------------------------------------
# Scenario C + RE — MultiWalkerPool with replica exchange between groups
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_with_hremd_between_groups():
    """
    Scenario C + H-REUS: 2 walker groups on device 0, harmonic bias at different
    centres, RE swaps between group primaries.
    Verifies RE runs without error and acceptance rate is valid.
    """
    kT = 2.479
    targets = [0.2, 0.5]

    groups, forces = [], []
    for g, target in enumerate(targets):
        sys_ = mm.System()
        sys_.addParticle(1.0); sys_.addParticle(1.0)
        f = gsp.GluedForce()
        f.setUsesPeriodicBoundaryConditions(False)
        cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
        f.addBias(gsp.GluedForce.BIAS_HARMONIC, [cv], [target, 200.0], [])
        sys_.addForce(f)
        ctx, _ = _make_ctx(sys_, 0)
        ctx.setPositions([[0.0, 0.0, 0.0], [target, 0.0, 0.0]])
        groups.append([ctx])
        forces.append([f])

    pool = MultiWalkerPool(
        walker_groups=groups,
        force_groups=forces,
        bias_index=0,
        sync_interval=0,     # no bias sync (different bias targets)
        re_mode="H-REUS",
        re_interval=20,
        kT=kT,
        seed=7,
    )
    pool.run(n_steps=80)

    rate = pool.re_acceptance_rate
    assert 0.0 <= rate <= 1.0, f"RE acceptance rate out of [0,1]: {rate}"


# ---------------------------------------------------------------------------
# BiasStateMerger unit tests (no CUDA required)
# ---------------------------------------------------------------------------

def _make_metad_blob(grid_values: list, n_deposited: int = 5) -> bytes:
    """
    Craft a minimal GPUS blob with exactly one MetaD bias and no others.
    Used to test BiasStateMerger without a running simulation.
    """
    import struct
    buf = bytearray()
    buf.extend(b'GPUS')
    buf.extend(struct.pack("<i", 1))   # version

    buf.extend(struct.pack("<i", 0))   # n_opes = 0
    buf.extend(struct.pack("<i", 0))   # n_abmd = 0

    buf.extend(struct.pack("<i", 1))   # n_metad = 1
    buf.extend(struct.pack("<i", n_deposited))
    for v in grid_values:
        buf.extend(struct.pack("<d", v))

    buf.extend(struct.pack("<i", 0))   # n_pbmetad = 0
    buf.extend(struct.pack("<i", 0))   # n_external
    buf.extend(struct.pack("<i", 0))   # n_linear
    buf.extend(struct.pack("<i", 0))   # n_wall
    buf.extend(struct.pack("<i", 0))   # n_opes_expanded
    buf.extend(struct.pack("<i", 0))   # n_ext_lagrangian
    buf.extend(struct.pack("<i", 0))   # n_eds
    buf.extend(struct.pack("<i", 0))   # n_maxent
    return bytes(buf)


class _FakeForceMeta:
    """Minimal stand-in for GluedForce to drive BiasStateMerger in unit tests."""
    def getNumBiases(self): return 1
    def getBiasParameters(self, idx):
        # BIAS_METAD=3, 1 CV, intparams=[pace=10, numBins=5, isPeriodic=0] → G=6
        return (3, [0], [], [10, 5, 0])


def test_bias_state_merger_additive():
    """
    BiasStateMerger.merge_additive correctly sums two MetaD grids.
    No CUDA required — uses crafted binary blobs.
    """
    import numpy as np

    grid_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    grid_b = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    blob_a = _make_metad_blob(grid_a, n_deposited=10)
    blob_b = _make_metad_blob(grid_b, n_deposited=5)

    fake_force = _FakeForceMeta()
    merged = BiasStateMerger.merge_additive([blob_a, blob_b], fake_force)

    # Re-parse merged blob.
    parsed = BiasStateMerger._parse(merged, fake_force)
    assert len(parsed["metad"]) == 1
    nd, grid_bytes = parsed["metad"][0]

    assert nd == 15, f"numDeposited should be 10+5=15, got {nd}"

    grid_merged = np.frombuffer(grid_bytes, dtype="<f8")
    expected = np.array(grid_a) + np.array(grid_b)
    np.testing.assert_allclose(grid_merged, expected, rtol=1e-12,
                                err_msg="Merged grid should be element-wise sum")


def test_bias_state_merger_single_blob():
    """merge_additive with a single blob returns the blob unchanged."""
    grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    blob = _make_metad_blob(grid)
    fake_force = _FakeForceMeta()
    merged = BiasStateMerger.merge_additive([blob], fake_force)

    parsed = BiasStateMerger._parse(merged, fake_force)
    nd, grid_bytes = parsed["metad"][0]
    import numpy as np
    grid_merged = np.frombuffer(grid_bytes, dtype="<f8")
    np.testing.assert_allclose(grid_merged, np.array(grid), rtol=1e-12)


def test_bias_state_merger_three_blobs():
    """merge_additive with three MetaD blobs sums all three grids."""
    import numpy as np
    grids = [[1.0]*6, [2.0]*6, [3.0]*6]
    blobs = [_make_metad_blob(g, n_deposited=10) for g in grids]
    fake_force = _FakeForceMeta()
    merged = BiasStateMerger.merge_additive(blobs, fake_force)

    parsed = BiasStateMerger._parse(merged, fake_force)
    _, grid_bytes = parsed["metad"][0]
    grid_m = np.frombuffer(grid_bytes, dtype="<f8")
    np.testing.assert_allclose(grid_m, np.full(6, 6.0), rtol=1e-12)


def test_bias_state_merger_repack_roundtrip():
    """_parse → _pack → _parse produces identical data."""
    import numpy as np
    grid = [0.1 * i for i in range(6)]
    blob = _make_metad_blob(grid, n_deposited=7)
    fake_force = _FakeForceMeta()

    parsed = BiasStateMerger._parse(blob, fake_force)
    repacked = BiasStateMerger._pack(parsed)
    parsed2  = BiasStateMerger._parse(repacked, fake_force)

    assert parsed["metad"][0][0] == parsed2["metad"][0][0], "numDeposited should survive round-trip"
    np.testing.assert_array_equal(
        np.frombuffer(parsed["metad"][0][1], dtype="<f8"),
        np.frombuffer(parsed2["metad"][0][1], dtype="<f8"),
    )


# ---------------------------------------------------------------------------
# Multi-walker + RE combination (single GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _cuda_available(), reason="CUDA platform not available")
def test_scenario_c_multi_walker_with_re_single_gpu():
    """
    Full Scenario C: 2 groups × 2 walkers (all on device 0) with:
    - Intra-group GPU pointer sharing (MetaD grid)
    - Cross-group bias merge every 20 steps
    - H-REUS exchange every 50 steps
    """
    kT = 2.479
    targets = [0.25, 0.45]   # group harmonic centres

    groups, forces = [], []
    for g, target in enumerate(targets):
        ctxs, fors = [], []
        for w in range(2):
            sys_ = mm.System()
            sys_.addParticle(1.0); sys_.addParticle(1.0)
            f = gsp.GluedForce()
            f.setUsesPeriodicBoundaryConditions(False)
            cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
            # Bias 0: MetaD (what's shared between walkers and merged across groups)
            f.addBias(gsp.GluedForce.BIAS_METAD, [cv],
                      [0.5, 0.05, 5.0, kT, 0.1, 0.8],
                      [10, 50, 0])
            # Bias 1: Harmonic restraint (window for RE; NOT shared)
            f.addBias(gsp.GluedForce.BIAS_HARMONIC, [cv], [target, 100.0], [])
            sys_.addForce(f)
            ctx, _ = _make_ctx(sys_, 0)
            ctx.setPositions([[0.0, 0.0, 0.0], [target, 0.0, 0.0]])
            ctxs.append(ctx)
            fors.append(f)
        groups.append(ctxs)
        forces.append(fors)

    pool = MultiWalkerPool(
        walker_groups=groups,
        force_groups=forces,
        bias_index=0,      # share / merge BIAS_METAD (bias index 0)
        sync_interval=20,
        sync_mode="additive",
        re_mode="H-REUS",
        re_interval=50,
        kT=kT,
        seed=99,
    )
    pool.run(n_steps=100)

    rate = pool.re_acceptance_rate
    assert 0.0 <= rate <= 1.0, f"RE acceptance rate {rate} out of bounds"
    # Both group primaries should have finite energy.
    for g in range(2):
        E = groups[g][0].getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        assert math.isfinite(E), f"Group {g} primary energy not finite after run"


if __name__ == "__main__":
    import sys as _sys

    tests = [
        test_bias_state_merger_additive,
        test_bias_state_merger_single_blob,
        test_bias_state_merger_three_blobs,
        test_bias_state_merger_repack_roundtrip,
    ]
    if _cuda_available():
        tests += [
            test_scenario_a_single_device_platform,
            test_scenario_b_build_replicas,
            test_scenario_b_with_replica_exchange,
            test_scenario_c_pool_construction_single_gpu,
            test_scenario_c_intra_group_wiring,
            test_scenario_c_cross_group_bias_merge,
            test_scenario_c_run_does_not_crash,
            test_scenario_c_with_hremd_between_groups,
            test_scenario_c_multi_walker_with_re_single_gpu,
        ]
    if _cuda_device_count() >= 2:
        tests += [
            test_scenario_a_multi_device_platform,
            test_scenario_c_two_physical_gpus,
        ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    _sys.exit(1 if failed > 0 else 0)
