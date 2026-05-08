"""
test_multiwalker.py — Tests for multiwalker B2 (shared GPU allocation with atomic deposits).

Two walkers share a single bias grid (MetaD) or kernel list (OPES) via raw CUDA
device pointers. All deposits from both walkers go to the same GPU arrays atomically.
This test only runs on the CUDA platform.
"""

import sys
import os
import math
import pytest

# Allow running directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import openmm as mm
    import openmm.unit as unit
    import gluedplugin as gsp
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


def has_cuda():
    """Return True if CUDA platform is available."""
    for i in range(mm.Platform.getNumPlatforms()):
        if mm.Platform.getPlatform(i).getName() == "CUDA":
            return True
    return False


def make_simple_system_and_force(n_atoms=2):
    """Create a minimal 2-atom system with a distance CV."""
    sys = mm.System()
    for _ in range(n_atoms):
        sys.addParticle(1000.0)  # large mass so atoms barely move

    force = gsp.GluedForce()
    force.setUsesPeriodicBoundaryConditions(False)
    return sys, force


def make_context(system, force, platform_name="CUDA"):
    """Create an OpenMM Context with the given force on the given platform."""
    integrator = mm.LangevinIntegrator(300, 1.0, 0.002)
    platform = mm.Platform.getPlatformByName(platform_name)
    ctx = mm.Context(system, integrator, platform)
    return ctx, integrator


def set_positions_distance(ctx, d=0.3):
    """Set positions so atom 0 is at origin, atom 1 is at (d, 0, 0)."""
    ctx.setPositions([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# MetaD multiwalker tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_metad_multiwalker_ptr_roundtrip():
    """
    Basic smoke test: create two MetaD walkers, get ptrs from primary,
    set ptrs on secondary — no exception should occur.
    """
    sys0, f0 = make_simple_system_and_force()
    cv_idx = f0.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    # MetaD: height=1.0, sigma=0.05, gamma=5.0, kT=2.479, pace=10,
    #        numBins=50, origin=0.1, max=0.8
    f0.addBias(gsp.GluedForce.BIAS_METAD, [cv_idx],
               [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
               [10, 50, 0])
    sys0.addForce(f0)

    sys1, f1 = make_simple_system_and_force()
    cv_idx1 = f1.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f1.addBias(gsp.GluedForce.BIAS_METAD, [cv_idx1],
               [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
               [10, 50, 0])
    sys1.addForce(f1)

    ctx0, _ = make_context(sys0, f0)
    ctx1, _ = make_context(sys1, f1)

    set_positions_distance(ctx0, 0.3)
    set_positions_distance(ctx1, 0.35)

    # Get ptrs from primary
    ptrs = f0.getMultiWalkerPtrs(ctx0, 0)
    assert len(ptrs) == 1, f"MetaD should return 1 ptr, got {len(ptrs)}"
    assert ptrs[0] != 0, "Device pointer should be non-zero"

    # Set ptrs on secondary
    f1.setMultiWalkerPtrs(ctx1, 0, ptrs)


@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_metad_multiwalker_shared_deposition():
    """
    Two MetaD walkers share the same grid. After N steps from both walkers,
    the bias at the primary's CV value should be larger than after N steps
    from a single walker only, because both walkers deposit into the same grid.
    """
    N_STEPS = 50
    PACE = 5
    d0 = 0.30  # CV value for walker 0
    d1 = 0.30  # Same CV value for both to maximize overlap

    # --- Single walker baseline ---
    sys_single, f_single = make_simple_system_and_force()
    cv_s = f_single.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f_single.addBias(gsp.GluedForce.BIAS_METAD, [cv_s],
                     [1.0, 0.05, 1.0, 2.479, 0.1, 0.8],  # gamma=1 => flat height
                     [PACE, 50, 0])
    sys_single.addForce(f_single)
    ctx_s, integ_s = make_context(sys_single, f_single)
    set_positions_distance(ctx_s, d0)

    # Run single walker
    for _ in range(N_STEPS):
        integ_s.step(1)

    single_energy = ctx_s.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # --- Two-walker shared grid ---
    sys0, f0 = make_simple_system_and_force()
    cv0 = f0.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f0.addBias(gsp.GluedForce.BIAS_METAD, [cv0],
               [1.0, 0.05, 1.0, 2.479, 0.1, 0.8],
               [PACE, 50, 0])
    sys0.addForce(f0)

    sys1, f1 = make_simple_system_and_force()
    cv1 = f1.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f1.addBias(gsp.GluedForce.BIAS_METAD, [cv1],
               [1.0, 0.05, 1.0, 2.479, 0.1, 0.8],
               [PACE, 50, 0])
    sys1.addForce(f1)

    ctx0, integ0 = make_context(sys0, f0)
    ctx1, integ1 = make_context(sys1, f1)

    set_positions_distance(ctx0, d0)
    set_positions_distance(ctx1, d1)

    # Share the grid: secondary uses primary's grid
    ptrs = f0.getMultiWalkerPtrs(ctx0, 0)
    f1.setMultiWalkerPtrs(ctx1, 0, ptrs)

    # Run both walkers
    for _ in range(N_STEPS):
        integ0.step(1)
        integ1.step(1)

    # Both walkers read the same shared grid — energy at the same CV value
    # should be larger than single walker (roughly 2x the deposits went in).
    shared_energy = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # Shared grid had 2x as many deposits as the single walker → higher bias energy
    # Use a lenient threshold: shared should be at least 1.3x the single-walker value.
    # (Not exactly 2x because the bias itself feeds back into the height calculation
    # and positions may drift slightly.)
    print(f"Single walker energy: {single_energy:.4f} kJ/mol")
    print(f"Two-walker shared energy: {shared_energy:.4f} kJ/mol")
    assert shared_energy > single_energy * 1.3, (
        f"Shared MetaD bias ({shared_energy:.4f}) should be > 1.3x "
        f"single walker ({single_energy:.4f})"
    )


# ---------------------------------------------------------------------------
# OPES multiwalker tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_opes_multiwalker_ptr_roundtrip():
    """
    Smoke test: OPES primary returns 5 non-zero ptrs; secondary accepts them.
    """
    sys0, f0 = make_simple_system_and_force()
    cv0 = f0.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    # OPES params: [kT, gamma, sigma0, sigmaMin]; int params: [variant, pace, maxKernels]
    f0.addBias(gsp.GluedForce.BIAS_OPES, [cv0],
               [2.479, 10.0, 0.05, 0.005],
               [0, 20, 10000])
    sys0.addForce(f0)

    sys1, f1 = make_simple_system_and_force()
    cv1 = f1.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f1.addBias(gsp.GluedForce.BIAS_OPES, [cv1],
               [2.479, 10.0, 0.05, 0.005],
               [0, 20, 10000])
    sys1.addForce(f1)

    ctx0, _ = make_context(sys0, f0)
    ctx1, _ = make_context(sys1, f1)

    set_positions_distance(ctx0, 0.3)
    set_positions_distance(ctx1, 0.35)

    ptrs = f0.getMultiWalkerPtrs(ctx0, 0)
    assert len(ptrs) == 5, f"OPES should return 5 ptrs, got {len(ptrs)}: {ptrs}"
    for i, p in enumerate(ptrs):
        assert p != 0, f"OPES device pointer[{i}] should be non-zero"

    f1.setMultiWalkerPtrs(ctx1, 0, ptrs)


@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_opes_multiwalker_kernel_count():
    """
    With two walkers sharing OPES kernel lists, the committed kernel count
    after N steps should be larger than the single-walker count, because the
    walkers explore different CV regions and their kernels don't all merge.

    Walker 0 stays at d=0.25, walker 1 stays at d=0.45 — the two clusters
    are ~4 sigma apart (sigma0=0.05) so OPES compression keeps them separate.
    The single-walker baseline uses only d=0.25.
    """
    N_STEPS = 200
    PACE = 20
    D_WALKER0 = 0.25   # primary walker CV position
    D_WALKER1 = 0.45   # secondary walker CV position — well separated from walker 0

    # --- Single walker (at d=0.25 only) ---
    sys_s, f_s = make_simple_system_and_force()
    cv_s = f_s.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    # OPES params: [kT, gamma, sigma0, sigmaMin]; int params: [variant=0, pace, maxKernels]
    f_s.addBias(gsp.GluedForce.BIAS_OPES, [cv_s],
                [2.479, 10.0, 0.05, 0.005],
                [0, PACE, 10000])
    sys_s.addForce(f_s)
    ctx_s, integ_s = make_context(sys_s, f_s)
    set_positions_distance(ctx_s, D_WALKER0)

    for _ in range(N_STEPS):
        integ_s.step(1)

    single_metrics = f_s.getOPESMetrics(ctx_s, 0)
    single_nker = int(single_metrics[2])

    # --- Two walkers at different CV positions ---
    sys0, f0 = make_simple_system_and_force()
    cv0 = f0.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f0.addBias(gsp.GluedForce.BIAS_OPES, [cv0],
               [2.479, 10.0, 0.05, 0.005],
               [0, PACE, 10000])
    sys0.addForce(f0)

    sys1, f1 = make_simple_system_and_force()
    cv1 = f1.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f1.addBias(gsp.GluedForce.BIAS_OPES, [cv1],
               [2.479, 10.0, 0.05, 0.005],
               [0, PACE, 10000])
    sys1.addForce(f1)

    ctx0, integ0 = make_context(sys0, f0)
    ctx1, integ1 = make_context(sys1, f1)

    # Place walkers at well-separated positions so kernels form two distinct clusters
    set_positions_distance(ctx0, D_WALKER0)
    set_positions_distance(ctx1, D_WALKER1)

    # Share OPES arrays: secondary uses primary's kernel list
    ptrs = f0.getMultiWalkerPtrs(ctx0, 0)
    f1.setMultiWalkerPtrs(ctx1, 0, ptrs)

    for _ in range(N_STEPS):
        integ0.step(1)
        integ1.step(1)

    shared_metrics = f0.getOPESMetrics(ctx0, 0)
    shared_nker = int(shared_metrics[2])

    print(f"Single walker OPES kernel count (d={D_WALKER0}): {single_nker}")
    print(f"Two-walker shared OPES kernel count (d={D_WALKER0} + d={D_WALKER1}): {shared_nker}")

    # The shared kernel list covers two separated CV regions, so it must have
    # significantly more kernels than a single walker exploring only one region.
    # Use a lenient threshold: shared must have at least 1.5x the single-walker count.
    assert shared_nker >= int(single_nker * 1.5), (
        f"Shared OPES kernel count ({shared_nker}) should be >= 1.5x "
        f"single-walker count ({single_nker}) — two walkers at distinct CV values "
        f"({D_WALKER0} vs {D_WALKER1}) should populate two separate kernel clusters"
    )


@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_metad_single_walker_unaffected():
    """
    Sanity check: a single walker with no multiwalker setup still works normally.
    Energy should be 0.0 at step 0 (before any deposits).
    """
    sys, f = make_simple_system_and_force()
    cv = f.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f.addBias(gsp.GluedForce.BIAS_METAD, [cv],
              [1.0, 0.05, 5.0, 2.479, 0.1, 0.8],
              [10, 50, 0])
    sys.addForce(f)

    ctx, integ = make_context(sys, f)
    set_positions_distance(ctx, 0.3)

    # Before any step: bias energy should be ~0 (no deposits yet)
    energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    assert abs(energy) < 1e-6, f"Expected ~0 bias at step 0, got {energy}"

    # Step and check it runs without error
    integ.step(20)
    energy_after = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    print(f"MetaD energy after 20 steps: {energy_after:.4f} kJ/mol")
    # Just verify it's a finite number
    assert math.isfinite(energy_after), "MetaD energy should be finite after 20 steps"


@pytest.mark.skipif(not has_cuda(), reason="CUDA platform not available")
def test_metad_multiwalker_secondary_reads_shared_grid():
    """
    After primary deposits, secondary should read the shared grid and see non-zero bias.
    This tests that the evalKernel arg redirect works correctly.
    """
    PACE = 5
    N_STEPS = 25  # deposit 5 times

    sys0, f0 = make_simple_system_and_force()
    cv0 = f0.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f0.addBias(gsp.GluedForce.BIAS_METAD, [cv0],
               [1.0, 0.05, 1.0, 2.479, 0.1, 0.8],
               [PACE, 50, 0])
    sys0.addForce(f0)

    sys1, f1 = make_simple_system_and_force()
    cv1 = f1.addCollectiveVariable(gsp.GluedForce.CV_DISTANCE, [0, 1], [])
    f1.addBias(gsp.GluedForce.BIAS_METAD, [cv1],
               [1.0, 0.05, 1.0, 2.479, 0.1, 0.8],
               [PACE, 50, 0])
    sys1.addForce(f1)

    ctx0, integ0 = make_context(sys0, f0)
    ctx1, integ1 = make_context(sys1, f1)

    set_positions_distance(ctx0, 0.3)
    set_positions_distance(ctx1, 0.3)

    # Set up sharing
    ptrs = f0.getMultiWalkerPtrs(ctx0, 0)
    f1.setMultiWalkerPtrs(ctx1, 0, ptrs)

    # Run only the primary walker (deposits into shared grid)
    for _ in range(N_STEPS):
        integ0.step(1)

    # Secondary evaluates: should see non-zero bias from primary's deposits
    # Force energy evaluation on secondary
    energy1 = ctx1.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # Primary's bias at its own position
    energy0 = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    print(f"Primary bias (read own grid): {energy0:.4f} kJ/mol")
    print(f"Secondary bias (read shared grid): {energy1:.4f} kJ/mol")

    # Both should be reading the same grid — energies at the same position should match
    assert abs(energy0 - energy1) < 0.1, (
        f"Primary ({energy0:.4f}) and secondary ({energy1:.4f}) should read same grid"
    )
    assert energy0 > 0.0, "Bias energy should be non-zero after deposits"


if __name__ == "__main__":
    print("Running multiwalker tests...")

    tests = [
        test_metad_single_walker_unaffected,
        test_metad_multiwalker_ptr_roundtrip,
        test_opes_multiwalker_ptr_roundtrip,
        test_metad_multiwalker_secondary_reads_shared_grid,
        test_metad_multiwalker_shared_deposition,
        test_opes_multiwalker_kernel_count,
    ]

    if not has_cuda():
        print("CUDA platform not available — skipping all multiwalker tests.")
        sys.exit(0)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
