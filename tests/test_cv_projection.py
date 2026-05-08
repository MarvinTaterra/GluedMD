"""Stage 3.14 — CV_PROJECTION acceptance tests.

CV = dot(r_b - r_a, d_hat)
where d_hat is a user-supplied direction (normalized in buildPlan).

Tests: CV value, force direction/magnitude, non-unit input direction.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV = 1e-5
TOL_F  = 1e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _proj_cv(positions, a, b, d):
    dr = [positions[b][k] - positions[a][k] for k in range(3)]
    dlen = math.sqrt(sum(x*x for x in d))
    d_hat = [x/dlen for x in d]
    return sum(dr[k]*d_hat[k] for k in range(3))


def _make_ctx(positions, a_idx, b_idx, direction, platform):
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori(); av.append(a_idx); av.append(b_idx)
    pv = mm.vectord()
    for d in direction: pv.append(d)
    f.addCollectiveVariable(gp.GluedForce.CV_PROJECTION, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


def test_projection_x_axis(platform):
    """CV along x-axis = x displacement."""
    positions = [[0.0]*3, [0.7, 0.3, 0.2]] + [[0.0]*3]*2
    ctx, f = _make_ctx(positions, 0, 1, [1.0, 0.0, 0.0], platform)
    cvs = f.getLastCVValues(ctx)
    assert abs(cvs[0] - 0.7) < TOL_CV, f"expected 0.7, got {cvs[0]}"
    print(f"  test_projection_x_axis: OK  (cv={cvs[0]:.6f})")


def test_projection_value(platform):
    """CV matches dot(dr, d_hat) for arbitrary direction."""
    rng = random.Random(99)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(4)]
    direction = [0.6, 0.8, 0.0]
    ctx, f = _make_ctx(positions, 0, 2, direction, platform)
    cvs = f.getLastCVValues(ctx)
    expected = _proj_cv(positions, 0, 2, direction)
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_projection_value: OK  (cv={cvs[0]:.6f}, ref={expected:.6f})")


def test_projection_nonnormalized_dir(platform):
    """Non-unit input direction is normalized by buildPlan."""
    positions = [[0.0]*3, [0.5, 0.5, 0.0]] + [[0.0]*3]*2
    # d=[3,4,0] → d_hat=[0.6,0.8,0] → cv = 0.5*0.6 + 0.5*0.8 = 0.7
    ctx, f = _make_ctx(positions, 0, 1, [3.0, 4.0, 0.0], platform)
    cvs = f.getLastCVValues(ctx)
    expected = 0.5*0.6 + 0.5*0.8
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.4f}, got {cvs[0]:.4f}"
    print(f"  test_projection_nonnormalized_dir: OK  (cv={cvs[0]:.4f})")


def test_projection_force(platform):
    """Force on atom b = -d_hat, force on atom a = +d_hat (bias grad=1)."""
    positions = [[0.0]*3, [0.5, 0.0, 0.0]] + [[0.0]*3]*2
    direction = [0.0, 1.0, 0.0]  # d_hat = y
    ctx, f = _make_ctx(positions, 0, 1, direction, platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)
    fa = [raw[0][c].value_in_unit(unit) for c in range(3)]
    fb = [raw[1][c].value_in_unit(unit) for c in range(3)]
    # F = -dV/dr = -bias_grad * dCV/dr; dCV/dr_a = -d_hat → F_a = d_hat
    # dCV/dr_b = +d_hat → F_b = -d_hat
    assert abs(fa[1] - 1.0) < TOL_F, f"F_a_y should be +1, got {fa[1]:.4f}"
    assert abs(fb[1] + 1.0) < TOL_F, f"F_b_y should be -1, got {fb[1]:.4f}"
    print(f"  test_projection_force: OK  (F_a_y={fa[1]:.4f}, F_b_y={fb[1]:.4f})")


def test_projection_numerical_gradient(platform):
    """Finite-difference check of projection Jacobian."""
    rng = random.Random(77)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(4)]
    direction = [0.4, 0.5, 0.7]
    h = 1e-4

    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori(); av.append(0); av.append(1)
    pv = mm.vectord()
    for d in direction: pv.append(d)
    f_cv.addCollectiveVariable(gp.GluedForce.CV_PROJECTION, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f_cv)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in range(n_atoms):
        for ci in range(3):
            pos_p = [list(p) for p in positions]
            pos_m = [list(p) for p in positions]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            fd = (_proj_cv(pos_p, 0, 1, direction) -
                   _proj_cv(pos_m, 0, 1, direction)) / (2*h)
            ctx.setPositions(positions)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err

    assert max_err < TOL_F, f"max gradient error {max_err:.2e} > {TOL_F}"
    print(f"  test_projection_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_PROJECTION tests.")
        sys.exit(0)
    print("Stage 3.14 — CV_PROJECTION tests (CUDA platform):")
    test_projection_x_axis(plat)
    test_projection_value(plat)
    test_projection_nonnormalized_dir(plat)
    test_projection_force(plat)
    test_projection_numerical_gradient(plat)
    print("All CV_PROJECTION tests passed.")
