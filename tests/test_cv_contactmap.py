"""Stage 3.12 — CV_CONTACTMAP acceptance tests.

Contact map: CV = sum_p w_p * s(d_p; r0, nn, mm)
Switching function: s = (1 - (d/r0)^nn) / (1 - (d/r0)^mm)

Tests value accuracy and numerical gradient for the chain-rule scatter.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV  = 1e-5
TOL_JAC = 5e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _switch(d, r0, nn, mm_exp):
    """Python reference for the rational switching function."""
    x = d / r0
    if abs(1.0 - x**mm_exp) < 1e-8:
        return nn / mm_exp
    return (1.0 - x**nn) / (1.0 - x**mm_exp)


def _cm_cpu(positions, pairs, params_list):
    """Reference contact-map value."""
    val = 0.0
    for (a, b), (r0, nn, mm_exp, w) in zip(pairs, params_list):
        dx = positions[b][0] - positions[a][0]
        dy = positions[b][1] - positions[a][1]
        dz = positions[b][2] - positions[a][2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        val += w * _switch(d, r0, nn, mm_exp)
    return val


def _make_ctx(positions, pairs, params_list, platform, periodic=False):
    n = len(positions)
    sys = mm.System()
    for _ in range(n): sys.addParticle(12.0)
    if periodic:
        sys.setDefaultPeriodicBoxVectors([5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0])

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(periodic)

    av = mm.vectori()
    for a, b in pairs:
        av.append(a); av.append(b)
    pv = mm.vectord()
    for r0, nn, mm_exp, w in params_list:
        pv.append(r0); pv.append(float(nn))
        pv.append(float(mm_exp)); pv.append(w)

    f.addCollectiveVariable(gp.GluedForce.CV_CONTACTMAP, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f)

    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)   # trigger first CV evaluation
    return ctx, f


def test_contact_single_pair_far(platform):
    """At d >> r0, switching function ≈ 0 (contact not formed)."""
    positions = [[0.0]*3, [5.0, 0.0, 0.0]]
    ctx, f = _make_ctx(positions, [(0, 1)], [(0.5, 6, 12, 1.0)], platform)
    cvs = f.getLastCVValues(ctx)
    assert cvs[0] < 1e-4, f"expected ~0, got {cvs[0]}"
    print(f"  test_contact_single_pair_far: OK  (cv={cvs[0]:.2e})")


def test_contact_single_pair_near(platform):
    """At d << r0, switching function ≈ 1 (contact formed)."""
    positions = [[0.0]*3, [0.01, 0.0, 0.0]]
    ctx, f = _make_ctx(positions, [(0, 1)], [(0.5, 6, 12, 1.0)], platform)
    cvs = f.getLastCVValues(ctx)
    assert abs(cvs[0] - 1.0) < 0.01, f"expected ~1, got {cvs[0]}"
    print(f"  test_contact_single_pair_near: OK  (cv={cvs[0]:.6f})")


def test_contact_three_pairs(platform):
    """Three-pair contact map matches Python reference."""
    rng = random.Random(13)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(6)]
    pairs = [(0, 1), (2, 3), (4, 5)]
    params = [(0.5, 6, 12, 1.0), (0.4, 8, 16, 0.5), (0.6, 6, 12, 2.0)]

    ctx, f = _make_ctx(positions, pairs, params, platform)
    cvs = f.getLastCVValues(ctx)
    expected = _cm_cpu(positions, pairs, params)
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_contact_three_pairs: OK  (cv={cvs[0]:.6f}, ref={expected:.6f})")


def test_contact_weight(platform):
    """Weight scales CV value linearly."""
    positions = [[0.0]*3, [0.2, 0.0, 0.0]]  # d=0.2 << r0=0.5 → s≈1
    ctx1, f1 = _make_ctx(positions, [(0, 1)], [(0.5, 6, 12, 1.0)], platform)
    ctx2, f2 = _make_ctx(positions, [(0, 1)], [(0.5, 6, 12, 3.0)], platform)
    cv1 = f1.getLastCVValues(ctx1)[0]
    cv2 = f2.getLastCVValues(ctx2)[0]
    ratio = cv2 / cv1
    assert abs(ratio - 3.0) < 0.01, f"weight scaling failed: ratio={ratio:.4f}"
    print(f"  test_contact_weight: OK  (ratio={ratio:.4f})")


def test_contact_numerical_gradient(platform):
    """Finite-difference check of contactmap Jacobian."""
    rng = random.Random(99)
    positions = [[rng.uniform(0.0, 0.5) for _ in range(3)] for _ in range(4)]
    pairs = [(0, 1), (2, 3), (0, 3)]
    params = [(0.5, 6, 12, 1.0), (0.4, 8, 16, 0.5), (0.6, 6, 12, 1.5)]

    n = len(positions)
    sys = mm.System()
    for _ in range(n): sys.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a, b in pairs: av.append(a); av.append(b)
    pv = mm.vectord()
    for r0, nn, mm_exp, w in params:
        pv.append(r0); pv.append(float(nn))
        pv.append(float(mm_exp)); pv.append(w)
    f_cv.addCollectiveVariable(gp.GluedForce.CV_CONTACTMAP, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f_cv)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

    h = 1e-4
    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in range(n):
        for ci in range(3):
            pos_p = [list(p) for p in positions]
            pos_m = [list(p) for p in positions]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            fd = (_cm_cpu(pos_p, pairs, params) - _cm_cpu(pos_m, pairs, params)) / (2*h)

            ctx.setPositions(positions)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err

    assert max_err < TOL_JAC, f"max gradient error {max_err:.2e} > {TOL_JAC}"
    print(f"  test_contact_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_CONTACTMAP tests.")
        sys.exit(0)
    print("Stage 3.12 — CV_CONTACTMAP tests (CUDA platform):")
    test_contact_single_pair_far(plat)
    test_contact_single_pair_near(plat)
    test_contact_three_pairs(plat)
    test_contact_weight(plat)
    test_contact_numerical_gradient(plat)
    print("All CV_CONTACTMAP tests passed.")
