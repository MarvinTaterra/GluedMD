"""Stage 3.13 — CV_PLANE acceptance tests.

CV = n_hat[component]  where  n = (b-a) × (c-a),  n_hat = n / |n|.
component: 0=x, 1=y, 2=z.

Tests: CV value correctness and numerical gradient.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV  = 1e-5
TOL_JAC = 1e-3


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _cross(u, v):
    return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]


def _norm(n):
    l = math.sqrt(sum(x*x for x in n))
    return [x/l for x in n], l


def _plane_cv(positions, a, b, c, comp):
    u = [positions[b][k] - positions[a][k] for k in range(3)]
    v = [positions[c][k] - positions[a][k] for k in range(3)]
    n = _cross(u, v)
    n_hat, _ = _norm(n)
    return n_hat[comp]


def _make_ctx(positions, a_idx, b_idx, c_idx, comp, platform):
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori(); av.append(a_idx); av.append(b_idx); av.append(c_idx)
    pv = mm.vectord(); pv.append(comp)
    f.addCollectiveVariable(gp.GluedForce.CV_PLANE, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


def test_plane_xy(platform):
    """Three atoms in XY plane → normal is along Z → n_hat[2] = ±1."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] + [[0.0]*3]
    ctx, f = _make_ctx(positions, 0, 1, 2, comp=2, platform=platform)
    cvs = f.getLastCVValues(ctx)
    assert abs(abs(cvs[0]) - 1.0) < TOL_CV, f"expected |cv|=1, got {cvs[0]}"
    print(f"  test_plane_xy: OK  (cv={cvs[0]:.6f})")


def test_plane_value(platform):
    """CV value matches Python cross-product reference."""
    rng = random.Random(42)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(5)]
    for comp in range(3):
        ctx, f = _make_ctx(positions, 0, 1, 2, comp=comp, platform=platform)
        cvs = f.getLastCVValues(ctx)
        expected = _plane_cv(positions, 0, 1, 2, comp)
        assert abs(cvs[0] - expected) < TOL_CV, \
            f"comp={comp}: expected {expected:.6f}, got {cvs[0]:.6f}"
    print("  test_plane_value: OK")


def test_plane_unit_normal(platform):
    """Components of the normal vector satisfy x²+y²+z² = 1."""
    rng = random.Random(7)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(4)]
    cvVals = []
    for comp in range(3):
        ctx, f = _make_ctx(positions, 0, 1, 2, comp=comp, platform=platform)
        cvVals.append(f.getLastCVValues(ctx)[0])
    norm_sq = sum(v**2 for v in cvVals)
    assert abs(norm_sq - 1.0) < 1e-4, f"expected |n_hat|²=1, got {norm_sq:.6f}"
    print(f"  test_plane_unit_normal: OK  (norm²={norm_sq:.6f})")


def test_plane_numerical_gradient(platform):
    """Finite-difference check of plane Jacobian for each component."""
    rng = random.Random(13)
    positions = [[rng.uniform(0.1, 0.9) for _ in range(3)] for _ in range(5)]
    h = 1e-4

    for comp in range(3):
        n_atoms = len(positions)
        sys = mm.System()
        for _ in range(n_atoms): sys.addParticle(12.0)
        f_cv = gp.GluedForce()
        f_cv.setUsesPeriodicBoundaryConditions(False)
        av = mm.vectori(); av.append(0); av.append(1); av.append(2)
        pv = mm.vectord(); pv.append(comp)
        f_cv.addCollectiveVariable(gp.GluedForce.CV_PLANE, av, pv)
        f_cv.setTestBiasGradients(mm.vectord([1.0]))
        sys.addForce(f_cv)
        ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

        max_err = 0.0
        unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
        for ai in [0, 1, 2]:
            for ci in range(3):
                pos_p = [list(p) for p in positions]
                pos_m = [list(p) for p in positions]
                pos_p[ai][ci] += h
                pos_m[ai][ci] -= h
                fd = (_plane_cv(pos_p, 0, 1, 2, comp) -
                       _plane_cv(pos_m, 0, 1, 2, comp)) / (2*h)
                ctx.setPositions(positions)
                state = ctx.getState(getForces=True)
                raw = state.getForces(asNumpy=False)
                f_anal = -raw[ai][ci].value_in_unit(unit)
                err = abs(fd - f_anal)
                if err > max_err:
                    max_err = err

        assert max_err < TOL_JAC, \
            f"comp={comp}: max gradient error {max_err:.2e} > {TOL_JAC}"
    print(f"  test_plane_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_PLANE tests.")
        sys.exit(0)
    print("Stage 3.13 — CV_PLANE tests (CUDA platform):")
    test_plane_xy(plat)
    test_plane_value(plat)
    test_plane_unit_normal(plat)
    test_plane_numerical_gradient(plat)
    print("All CV_PLANE tests passed.")
