"""Stage 3.7 acceptance tests — expression CVs."""
import sys, math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL = 1e-4

def get_cuda_platform():
    try: return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException: return None

def make_ctx(positions_nm, cv_specs, bias_gradients, platform):
    """cv_specs: list of (type_or_'expr', args, params_or_expr_str).
    For expr: args = inputCVIndices (list), params_or_expr_str = expression string.
    bias_gradients: list of floats, one per CV value.
    """
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n): sys_.addParticle(1.0)
    f = gp.GluedForce()
    for spec in cv_specs:
        if spec[0] == 'expr':
            _, inputCVs, exprStr = spec
            iv = mm.vectori()
            for c in inputCVs: iv.append(c)
            f.addExpressionCV(exprStr, iv)
        else:
            cvtype, atoms, params = spec
            av = mm.vectori()
            for a in atoms: av.append(a)
            pv = mm.vectord()
            for p in params: pv.append(p)
            f.addCollectiveVariable(cvtype, av, pv)
    gv = mm.vectord()
    for g in bias_gradients: gv.append(g)
    f.setTestBiasGradients(gv)
    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f

def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]

def test_expr_identity(platform):
    """expr = cv0 should behave identically to using the distance CV directly."""
    pos = [(0,0,0),(1.0,0,0),(5,5,5)]
    # Direct distance CV: grad=1.0 → F1.x = -1.0 * dDist/dr1.x = -1.0
    cv_specs_direct = [(gp.GluedForce.CV_DISTANCE, [0,1], [])]
    ctx_d, _ = make_ctx(pos, cv_specs_direct, [1.0], platform)
    F_direct = get_forces(ctx_d)[1][0]

    # Expression CV wrapping the same distance
    cv_specs_expr = [
        (gp.GluedForce.CV_DISTANCE, [0,1], []),   # cv0
        ('expr', [0], 'cv0'),                           # cv1 = cv0
    ]
    ctx_e, _ = make_ctx(pos, cv_specs_expr, [0.0, 1.0], platform)
    F_expr = get_forces(ctx_e)[1][0]
    assert abs(F_expr - F_direct) < TOL, f"F_expr={F_expr:.6f}, F_direct={F_direct:.6f}"
    print(f"  test_expr_identity: OK  (F={F_expr:.4f})")

def test_expr_sin_cos(platform):
    """expr = sin(cv0), verify value and analytical force."""
    dist = 1.2  # cv0 = dist(0,1)
    pos = [(0,0,0),(dist,0,0),(5,5,5)]
    cv_specs = [
        (gp.GluedForce.CV_DISTANCE, [0,1], []),   # cv0 = dist
        ('expr', [0], 'sin(cv0)'),                      # cv1 = sin(dist)
    ]
    ctx, _ = make_ctx(pos, cv_specs, [0.0, 1.0], platform)
    forces = get_forces(ctx)
    # dexpr/dcv0 = cos(cv0) = cos(dist)
    # dDist/dr1.x = +1 (distance along x)
    # F1.x = -dV/dr1.x = -(grad_on_expr * dexpr/dcv0 * dDist/dr1.x) = -1.0*cos(dist)*1.0
    expected_F1x = -math.cos(dist)
    assert abs(forces[1][0] - expected_F1x) < TOL, \
        f"F1.x={forces[1][0]:.6f}, expected {expected_F1x:.6f}"
    print(f"  test_expr_sin_cos: OK  (F1.x={forces[1][0]:.4f}, expected {expected_F1x:.4f})")

def test_expr_two_inputs(platform):
    """expr = cv0 + 2*cv1: grad splits proportionally."""
    d0, d1 = 1.0, 2.0
    pos = [(0,0,0),(d0,0,0),(0,d1,0),(5,5,5)]
    cv_specs = [
        (gp.GluedForce.CV_DISTANCE, [0,1], []),   # cv0
        (gp.GluedForce.CV_DISTANCE, [0,2], []),   # cv1
        ('expr', [0,1], 'cv0 + 2*cv1'),               # cv2
    ]
    ctx, _ = make_ctx(pos, cv_specs, [0.0, 0.0, 1.0], platform)
    forces = get_forces(ctx)
    # dexpr/dcv0=1, dexpr/dcv1=2
    # F1.x = -(1.0 * 1 * 1) = -1.0  (cv0=dist along x)
    # F2.y = -(1.0 * 2 * 1) = -2.0  (cv1=dist along y)
    assert abs(forces[1][0] - (-1.0)) < TOL, f"F1.x={forces[1][0]:.6f}"
    assert abs(forces[2][1] - (-2.0)) < TOL, f"F2.y={forces[2][1]:.6f}"
    print(f"  test_expr_two_inputs: OK  (F1.x={forces[1][0]:.4f}, F2.y={forces[2][1]:.4f})")

def test_expr_chained(platform):
    """Chained expression: cv0=dist, cv1=sin(cv0), cv2=cv0*cv0 (where cv0 inside cv2 refers to cv1).
    d(cv2)/dr1.x = 2*sin(cv0)*cos(cv0)*1 = sin(2*dist)."""
    dist = 0.8
    pos = [(0,0,0),(dist,0,0),(5,5,5)]
    cv_specs = [
        (gp.GluedForce.CV_DISTANCE, [0,1], []),   # cv0 (spec 0)
        ('expr', [0], 'sin(cv0)'),                      # cv1 (spec 1) = sin(cv0)
        ('expr', [1], 'cv0*cv0'),                       # cv2 (spec 2) = cv1^2
    ]
    ctx, _ = make_ctx(pos, cv_specs, [0.0, 0.0, 1.0], platform)
    forces = get_forces(ctx)
    # d(cv2)/dr1.x = d(cv1^2)/dcv1 * d(sin(cv0))/dcv0 * dDist/dr1.x
    #              = 2*cv1 * cos(cv0) * 1 = 2*sin(dist)*cos(dist) = sin(2*dist)
    expected = -math.sin(2*dist)
    assert abs(forces[1][0] - expected) < TOL * 10, \
        f"F1.x={forces[1][0]:.6f}, expected {expected:.6f}"
    print(f"  test_expr_chained: OK  (F1.x={forces[1][0]:.4f}, expected {expected:.4f})")

if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA not available — skip"); sys.exit(0)
    print("Stage 3.7 expression CV tests (CUDA):")
    test_expr_identity(plat)
    test_expr_sin_cos(plat)
    test_expr_two_inputs(plat)
    test_expr_chained(plat)
    print("All expression CV tests passed.")
