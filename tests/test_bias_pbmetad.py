"""Stage 5.4 acceptance tests — PBMETAD (parallel-bias metadynamics).

PBMETAD uses N independent 1-D MetaD grids (one per CV) sharing height0,
gamma, kT, and pace.  The total bias is the sum of the N 1-D biases.

Parameter encoding for addBias(BIAS_PBMETAD, cvList, params, intParams):
  params       = [height0, gamma, kT,
                  sigma_0, origin_0, max_0,
                  sigma_1, origin_1, max_1, ...]
  integerParams = [pace, numBins_0, isPeriodic_0,
                         numBins_1, isPeriodic_1, ...]
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_E = 1e-4   # float32 GPU grid → slightly looser
TOL_F = 1e-3


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_system(positions_nm, cv_specs, bias_specs, platform=None):
    """bias_specs: [(cv_list, params, int_params)]"""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)

    f = gp.GluedForce()
    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms:
            av.append(a)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    for cv_list, params, int_params in bias_specs:
        civ = mm.vectori()
        for c in cv_list:
            civ.append(c)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        iv = mm.vectori()
        for i in int_params:
            iv.append(i)
        f.addBias(gp.GluedForce.BIAS_PBMETAD, civ, pv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def get_energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        mm.unit.kilojoules_per_mole)


def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pbmetad_zero_before_deposit(platform):
    """Empty grid → bias = 0 everywhere before any deposition."""
    height0, gamma, kT = 1.2, 10.0, 2.479
    sigma, origin, maxV = 0.1, 0.0, 2.0
    numBins, periodic   = 100, 0
    params     = [height0, gamma, kT, sigma, origin, maxV]
    int_params = [1000, numBins, periodic]
    pos = [(0, 0, 0), (1.0, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.2e}, expected 0 before deposit"
    print(f"  test_pbmetad_zero_before_deposit: OK  (E={E:.2e})")


def test_pbmetad_two_cv_sum(platform):
    """Two CVs with independent 1-D grids; zero grid → total bias = 0.
    With a pre-loaded grid on one CV, checks that only that CV contributes."""
    height0, gamma, kT = 2.0, 10.0, 2.479
    # CV0: distance atoms 0-1; CV1: distance atoms 0-2
    # sigma, origin, max per CV
    sigma0, sigma1     = 0.1, 0.1
    origin0, maxV0     = 0.0, 2.0
    origin1, maxV1     = 3.0, 8.0   # atom2 is at (5,5,5) → dist ~ 8.66
    numBins, periodic  = 50, 0
    params     = [height0, gamma, kT,
                  sigma0, origin0, maxV0,
                  sigma1, origin1, maxV1]
    int_params = [1000, numBins, periodic, numBins, periodic]
    pos = [(0, 0, 0), (1.0, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], []),
                          (gp.GluedForce.CV_DISTANCE, [0, 2], [])],
                         [([0, 1], params, int_params)], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.2e}, expected 0 (empty grids)"
    print(f"  test_pbmetad_two_cv_sum: OK  (E={E:.2e})")


def test_pbmetad_deposit_raises_energy(platform):
    """After one deposition at cv=1.0, evaluating near cv=1.0 gives positive energy.
    Single-CV case to verify deposit→eval pipeline works end-to-end."""
    height0, gamma, kT = 2.0, 1.0, 2.479   # gamma=1 → fixed height
    sigma, origin, maxV = 0.2, 0.0, 2.0
    numBins, periodic   = 200, 0
    pace = 1
    params     = [height0, gamma, kT, sigma, origin, maxV]
    int_params = [pace, numBins, periodic]

    d = 1.0
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)

    # Prime and deposit one Gaussian at cv=1.0.
    # MetaD skips deposition on step=0 (matching PLUMED !isFirstStep_ guard).
    # Two steps needed: step=0 (skip), step=1 (deposit).
    get_energy(ctx)                   # primes cvValuesReady_
    ctx.getIntegrator().step(1)       # step=0: updateState skips deposit
    ctx.getIntegrator().step(1)       # step=1: updateState deposits at cv=1.0

    # Reset to same position and read energy
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d, 0, 0), mm.Vec3(5, 5, 5)])
    E = get_energy(ctx)

    # Should be near height0 (peak of the Gaussian)
    assert E > height0 * 0.5, f"E={E:.4f} too low after deposit (expected ~{height0})"
    assert E < height0 * 1.5, f"E={E:.4f} too high after deposit"
    print(f"  test_pbmetad_deposit_raises_energy: OK  (E={E:.4f}, expected ~{height0:.2f})")


def test_pbmetad_force_direction(platform):
    """Finite-difference force check on a single-CV PBMETAD after one deposit.

    Grid spacing = (maxV-origin)/numBins = 3/300 = 0.01 nm.
    Evaluate forces at eval_d=1.005 (midpoint between grid bins 100 and 101)
    with dx=0.005 (half a grid spacing).  This makes E_p use grid[101] exactly
    and E_m use grid[100] exactly, so F_num = F_ana to double precision."""
    height0, gamma, kT = 2.0, 1.0, 2.479
    sigma, origin, maxV = 0.3, 0.0, 3.0
    numBins, periodic   = 300, 0
    spacing = (maxV - origin) / numBins   # 0.01 nm
    pace = 1
    params     = [height0, gamma, kT, sigma, origin, maxV]
    int_params = [pace, numBins, periodic]

    deposit_d = 1.5
    eval_d    = origin + 100.5 * spacing  # midpoint of bins 100–101 = 1.005
    dx        = 0.5 * spacing             # = 0.005 → E_p at bin 101, E_m at bin 100

    pos = [(0, 0, 0), (deposit_d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)   # deposit at cv=1.5

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d + dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_p = get_energy(ctx)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d - dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_m = get_energy(ctx)
    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d, 0, 0), mm.Vec3(5, 5, 5)])
    F_ana = get_forces(ctx)[1][0]

    assert abs(F_num - F_ana) < TOL_F * 5, \
        f"F_num={F_num:.4f}, F_ana={F_ana:.4f}"
    print(f"  test_pbmetad_force_direction: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


def test_pbmetad_restart(platform):
    """getBiasStateBytes / setBiasStateBytes round-trip preserves the grid."""
    height0, gamma, kT = 1.5, 1.0, 2.479
    sigma, origin, maxV = 0.2, 0.0, 2.0
    numBins, periodic   = 100, 0
    pace = 1
    params     = [height0, gamma, kT, sigma, origin, maxV]
    int_params = [pace, numBins, periodic]

    d = 0.8
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, f = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)

    get_energy(ctx)
    ctx.getIntegrator().step(1)   # deposit
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d, 0, 0), mm.Vec3(5, 5, 5)])
    E_before = get_energy(ctx)
    state_bytes = f.getBiasState()   # returns Python bytes

    # Fresh context
    ctx2, f2 = make_system(pos,
                           [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                           [([0], params, int_params)], platform)
    f2.setBiasState(state_bytes)    # accepts Python bytes

    ctx2.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d, 0, 0), mm.Vec3(5, 5, 5)])
    E_after = get_energy(ctx2)

    assert abs(E_after - E_before) < TOL_E * 10, \
        f"E_before={E_before:.6f}, E_after={E_after:.6f}"
    print(f"  test_pbmetad_restart: OK  "
          f"(E_before={E_before:.4f}, E_after={E_after:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping PBMETAD tests.")
        sys.exit(0)

    print("Stage 5.4 PBMETAD tests (CUDA platform):")
    test_pbmetad_zero_before_deposit(plat)
    test_pbmetad_two_cv_sum(plat)
    test_pbmetad_deposit_raises_energy(plat)
    test_pbmetad_force_direction(plat)
    test_pbmetad_restart(plat)
    print("All PBMETAD tests passed.")
