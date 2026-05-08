"""Stage 5.6 acceptance tests — BIAS_EXTERNAL (precomputed grid bias).

Parameter encoding for addBias(BIAS_EXTERNAL, cvList, params, intParams):
  params       = [origin_0, ..., origin_{D-1},
                  max_0,    ..., max_{D-1},
                  grid_val_0, ..., grid_val_{totalGridPoints-1}]
  integerParams = [numBins_0, ..., numBins_{D-1},
                   isPeriodic_0, ..., isPeriodic_{D-1}]

Grid layout:
  totalGridPoints = product over d of (numBins_d + (isPeriodic_d ? 0 : 1))
  flatIdx = sum_d(bin_d * strides[d])  with strides[0]=1, strides[d]=strides[d-1]*actualPoints[d-1]
  position of bin i along dim d: origin_d + i * spacing_d
"""
import sys
import math
import struct
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_E = 1e-4
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
        for a in atoms: av.append(a)
        pv = mm.vectord()
        for p in params: pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    for cv_list, params, int_params in bias_specs:
        civ = mm.vectori()
        for c in cv_list: civ.append(c)
        pv = mm.vectord()
        for p in params: pv.append(p)
        iv = mm.vectori()
        for i in int_params: iv.append(i)
        f.addBias(gp.GluedForce.BIAS_EXTERNAL, civ, pv, iv)

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


def gaussian_grid_1d(height, center, sigma, origin, maxV, numBins):
    """Return grid values for a single Gaussian on a 1-D non-periodic grid."""
    actualPoints = numBins + 1
    spacing = (maxV - origin) / numBins
    return [height * math.exp(-((origin + i * spacing - center) ** 2) / (2 * sigma ** 2))
            for i in range(actualPoints)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_external_zero_grid(platform):
    """All-zero grid → energy = 0 everywhere."""
    origin, maxV, numBins = 0.0, 2.0, 100
    actualPoints = numBins + 1
    gridValues = [0.0] * actualPoints
    params = [origin, maxV] + gridValues
    int_params = [numBins, 0]
    pos = [(0, 0, 0), (1.0, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.2e}, expected 0"
    print(f"  test_external_zero_grid: OK  (E={E:.2e})")


def test_external_gaussian_peak(platform):
    """Grid with a single Gaussian; evaluate at peak → energy ≈ height."""
    height, sigma = 2.0, 0.2
    origin, maxV, numBins = 0.0, 2.0, 200
    center = 1.0
    gridValues = gaussian_grid_1d(height, center, sigma, origin, maxV, numBins)
    params = [origin, maxV] + gridValues
    int_params = [numBins, 0]
    pos = [(0, 0, 0), (center, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)
    E = get_energy(ctx)
    assert abs(E - height) < height * 0.01, f"E={E:.4f}, expected ~{height}"
    print(f"  test_external_gaussian_peak: OK  (E={E:.4f}, expected ~{height})")


def test_external_force_direction(platform):
    """FD force check on a 1-D external Gaussian bias.

    Evaluate at alpha=0.5 (midpoint between bin 100 and bin 101) with
    dx=0.5*spacing, so E_p=grid[101] and E_m=grid[100] exactly — giving
    F_num = F_ana to machine precision on the discretized grid.
    """
    height, sigma = 2.0, 0.3
    origin, maxV, numBins = 0.0, 3.0, 300
    spacing = (maxV - origin) / numBins
    center = 1.5
    gridValues = gaussian_grid_1d(height, center, sigma, origin, maxV, numBins)
    params = [origin, maxV] + gridValues
    int_params = [numBins, 0]

    eval_d = origin + 100.5 * spacing  # midpoint of bins 100–101 = 1.005
    dx     = 0.5 * spacing             # = 0.005

    pos = [(0, 0, 0), (eval_d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], params, int_params)], platform)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d + dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_p = get_energy(ctx)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d - dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_m = get_energy(ctx)
    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d, 0, 0), mm.Vec3(5, 5, 5)])
    F_ana = get_forces(ctx)[1][0]

    assert abs(F_num - F_ana) < TOL_F * 5, \
        f"F_num={F_num:.4f}, F_ana={F_ana:.4f}"
    print(f"  test_external_force_direction: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


def test_external_two_cv_sum(platform):
    """Two independent 1-D external biases (one per CV); total bias = sum.

    Two separate addBias calls, each D=1.  Both grids peaked at their
    evaluation point, so E_total ≈ height0 + height1.
    """
    height0, height1 = 2.0, 3.0
    sigma0, sigma1   = 0.2, 0.15
    origin, maxV, numBins = 0.0, 2.0, 100
    center0, center1 = 1.0, 0.8

    grid0 = gaussian_grid_1d(height0, center0, sigma0, origin, maxV, numBins)
    grid1 = gaussian_grid_1d(height1, center1, sigma1, origin, maxV, numBins)

    params0 = [origin, maxV] + grid0
    params1 = [origin, maxV] + grid1
    int_params = [numBins, 0]

    # CV0: distance atoms 0-1 = center0  →  CV1: distance atoms 0-2 = center1
    pos = [(0, 0, 0), (center0, 0, 0), (center1, 0, 0)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], []),
                          (gp.GluedForce.CV_DISTANCE, [0, 2], [])],
                         [([0], params0, int_params),
                          ([1], params1, int_params)], platform)
    E = get_energy(ctx)
    E_expected = height0 + height1
    assert abs(E - E_expected) < E_expected * 0.01, \
        f"E={E:.4f}, expected ~{E_expected:.4f}"
    print(f"  test_external_two_cv_sum: OK  (E={E:.4f}, expected ~{E_expected:.4f})")


def test_external_matches_metad_after_deposit(platform):
    """METAD bias after one deposit == external bias with the same grid.

    After one fixed-height MetaD deposition at cv=1.0 (gamma=1, pace=1),
    the MetaD grid contains exactly one Gaussian of height=height0.
    We build the same grid analytically and verify that the external bias
    gives the same energy at cv=0.8.
    """
    height0, sigma = 2.0, 0.2
    origin, maxV, numBins = 0.0, 2.0, 200
    spacing = (maxV - origin) / numBins
    deposit_d = 1.0
    eval_d    = 0.8

    # ---- MetaD context ----
    sys_m = mm.System()
    for _ in range(3): sys_m.addParticle(1.0)
    f_m = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    f_m.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    civ = mm.vectori(); civ.append(0)
    # MetaD params: [height0, sigma, gamma, kT, origin, maxV]
    # intParams:    [pace, numBins, isPeriodic]
    pv = mm.vectord()
    for x in [height0, sigma, 1.0, 2.479, origin, maxV]: pv.append(x)
    iv = mm.vectori()
    for x in [1, numBins, 0]: iv.append(x)
    f_m.addBias(gp.GluedForce.BIAS_METAD, civ, pv, iv)
    sys_m.addForce(f_m)
    integ_m = mm.VerletIntegrator(0.001)
    ctx_m = mm.Context(sys_m, integ_m, platform)
    ctx_m.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(deposit_d, 0, 0), mm.Vec3(5, 5, 5)])
    get_energy(ctx_m)                # prime cvValuesReady
    ctx_m.getIntegrator().step(1)    # step=0: updateState skips deposit (PLUMED !isFirstStep_)
    ctx_m.getIntegrator().step(1)    # step=1: deposits Gaussian at cv=deposit_d

    ctx_m.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(eval_d, 0, 0), mm.Vec3(5, 5, 5)])
    E_metad = get_energy(ctx_m)

    # ---- External bias context with same grid ----
    gridValues = gaussian_grid_1d(height0, deposit_d, sigma, origin, maxV, numBins)
    params = [origin, maxV] + gridValues
    int_params = [numBins, 0]
    pos = [(0, 0, 0), (eval_d, 0, 0), (5, 5, 5)]
    ctx_ext, _ = make_system(pos,
                             [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                             [([0], params, int_params)], platform)
    E_ext = get_energy(ctx_ext)

    assert abs(E_ext - E_metad) < TOL_E * 20, \
        f"E_ext={E_ext:.6f}, E_metad={E_metad:.6f}"
    print(f"  test_external_matches_metad_after_deposit: OK  "
          f"(E_ext={E_ext:.4f}, E_metad={E_metad:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping External bias tests.")
        sys.exit(0)

    print("Stage 5.6 External bias tests (CUDA platform):")
    test_external_zero_grid(plat)
    test_external_gaussian_peak(plat)
    test_external_force_direction(plat)
    test_external_two_cv_sum(plat)
    test_external_matches_metad_after_deposit(plat)
    print("All External bias tests passed.")
