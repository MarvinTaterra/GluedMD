"""Stage 3.2 acceptance tests — angle and dihedral CV evaluation + force scatter."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-5   # rad
TOL_FORCE = 1e-4   # kJ/mol/nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_context(positions_nm, cv_specs, bias_gradients=None, platform=None):
    """Build a Context from a list of (cv_type, atom_indices) tuples."""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)

    f = gp.GluedForce()
    for cv_type, atoms in cv_specs:
        av = mm.vectori()
        for a in atoms:
            av.append(a)
        pv = mm.vectord()
        f.addCollectiveVariable(cv_type, av, pv)

    if bias_gradients is not None:
        gv = mm.vectord()
        for g in bias_gradients:
            gv.append(g)
        f.setTestBiasGradients(gv)

    sys_.addForce(f)
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def eval_cv(ctx, f):
    ctx.getState(getForces=True)
    return list(f.getLastCVValues(ctx))


def get_forces(ctx):
    state = ctx.getState(getForces=True)
    raw   = state.getForces(asNumpy=False)
    unit  = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


# ---------------------------------------------------------------------------
# Angle tests
# ---------------------------------------------------------------------------

def test_angle_right(platform):
    """90° angle: A=(1,0,0), B=(0,0,0), C=(0,1,0) → π/2."""
    pos = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_ANGLE, [0, 1, 2])],
                          platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - math.pi/2) < TOL_CV, f"angle={cvs[0]:.8f}, expected π/2"
    print(f"  test_angle_right: OK  (θ={cvs[0]:.6f} rad, expected {math.pi/2:.6f})")


def test_angle_60(platform):
    """60° angle: A=(1,0,0), B=(0,0,0), C=(0.5, √3/2, 0)."""
    c = math.sqrt(3.0) / 2.0
    pos = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, c, 0.0), (0.5, 0.5, 0.5)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_ANGLE, [0, 1, 2])],
                          platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - math.pi/3) < TOL_CV, f"angle={cvs[0]:.8f}, expected π/3"
    print(f"  test_angle_60: OK  (θ={cvs[0]:.6f} rad, expected {math.pi/3:.6f})")


def test_angle_force_scatter(platform):
    """90° angle with bias gradient = 1.0; verify analytical force on atom A.
    dr1=(1,0,0), dr2=(0,1,0); jac_A = -(n2 - 0*n1)/r1 = (0,-1,0).
    force_A = -1.0 * (0,-1,0) = (0,+1,0)."""
    pos = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_ANGLE, [0, 1, 2])],
                          bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    # atom A: force = (0, +1, 0)
    assert abs(forces[0][0])        < TOL_FORCE, f"A fx={forces[0][0]:.6f}"
    assert abs(forces[0][1] - 1.0)  < TOL_FORCE, f"A fy={forces[0][1]:.6f}"
    assert abs(forces[0][2])        < TOL_FORCE, f"A fz={forces[0][2]:.6f}"
    # force sum must be zero
    for c in range(3):
        s = sum(forces[i][c] for i in range(3))
        assert abs(s) < TOL_FORCE, f"force sum component {c} = {s:.6f}"
    print(f"  test_angle_force_scatter: OK  "
          f"(fA=({forces[0][0]:.3f},{forces[0][1]:.3f},{forces[0][2]:.3f}))")


def test_angle_numerical_derivative(platform):
    """Finite-difference check of angle Jacobian on atom A x-component."""
    dx = 1e-4
    base = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    pos_p = list(base); pos_p[0] = (1.0 + dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, [(gp.GluedForce.CV_ANGLE, [0, 1, 2])],
                               platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (1.0 - dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, [(gp.GluedForce.CV_ANGLE, [0, 1, 2])],
                               platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    # Analytical: jac_A_x = -(n2.x - cosT*n1.x)/(r1*sinT)
    #   at 90° angle: cosT=0, n1=(1,0,0), n2=(0,1,0), r1=1, sinT=1
    #   → jac_A_x = -(0 - 0)/1 = 0
    jac_ana = 0.0
    assert abs(jac_num - jac_ana) < 5e-4, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_angle_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


# ---------------------------------------------------------------------------
# Dihedral tests
# ---------------------------------------------------------------------------

def test_dihedral_zero(platform):
    """Dihedral = 0: A=(1,0,0), B=(0,0,0), C=(0,0,1), D=(1,0,1)."""
    pos = [(1.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0), (1.0,0.0,1.0),
           (0.5,0.5,0.5)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_DIHEDRAL, [0, 1, 2, 3])],
                          platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0]) < TOL_CV, f"dihedral={cvs[0]:.8f}, expected 0"
    print(f"  test_dihedral_zero: OK  (φ={cvs[0]:.8f} rad)")


def test_dihedral_pi(platform):
    """Dihedral = π: A=(1,0,0), B=(0,0,0), C=(0,0,1), D=(-1,0,1)."""
    pos = [(1.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0), (-1.0,0.0,1.0),
           (0.5,0.5,0.5)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_DIHEDRAL, [0, 1, 2, 3])],
                          platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(abs(cvs[0]) - math.pi) < TOL_CV, \
        f"dihedral={cvs[0]:.8f}, expected ±π"
    print(f"  test_dihedral_pi: OK  (φ={cvs[0]:.8f} rad)")


def test_dihedral_90(platform):
    """Dihedral = π/2: A=(1,0,0), B=(0,0,0), C=(0,0,1), D=(0,1,1)."""
    pos = [(1.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0), (0.0,1.0,1.0),
           (0.5,0.5,0.5)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_DIHEDRAL, [0, 1, 2, 3])],
                          platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - math.pi/2) < TOL_CV, \
        f"dihedral={cvs[0]:.8f}, expected π/2"
    print(f"  test_dihedral_90: OK  (φ={cvs[0]:.8f} rad)")


def test_dihedral_numerical_derivative(platform):
    """Finite-difference check on atom A x-component."""
    dx = 1e-4
    # 90° dihedral configuration
    base = [(1.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0), (0.0,1.0,1.0),
            (0.5,0.5,0.5)]
    specs = [(gp.GluedForce.CV_DIHEDRAL, [0, 1, 2, 3])]

    pos_p = list(base); pos_p[0] = (1.0+dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (1.0-dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    # Analytical: b1=(-1,0,0), b2=(0,0,1), b3=(0,1,0)
    # t = b1×b2 = (-1,0,0)×(0,0,1) = (0*1-0*0, 0*0-(-1)*1, (-1)*0-0*0) = (0,1,0)
    # d = 1; |t|^2 = 1; aT = 1
    # jac_A = -aT * t = (0,-1,0); jac_A_x = 0
    jac_ana = 0.0
    assert abs(jac_num - jac_ana) < 5e-4, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_dihedral_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_dihedral_force_scatter(platform):
    """0° dihedral with bias gradient = 1.0; verify force sum = 0."""
    pos = [(1.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,1.0), (1.0,0.0,1.0),
           (0.5,0.5,0.5)]
    ctx, f = make_context(pos, [(gp.GluedForce.CV_DIHEDRAL, [0, 1, 2, 3])],
                          bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(4))
        assert abs(s) < TOL_FORCE, f"force sum component {c} = {s:.6f}"
    print("  test_dihedral_force_scatter: OK  (force sum ≈ 0)")


def test_mixed_cvs(platform):
    """One angle + one dihedral in the same Force — verify both CV values."""
    # angle: atoms 0,1,2 → 90°; dihedral: atoms 0,1,3,4 → 90°
    pos = [
        (1.0, 0.0, 0.0),   # 0
        (0.0, 0.0, 0.0),   # 1  (angle vertex / dihedral bond start)
        (0.0, 1.0, 0.0),   # 2  (angle end)
        (0.0, 0.0, 1.0),   # 3  (dihedral bond end)
        (0.0, 1.0, 1.0),   # 4  (dihedral last atom → φ = π/2)
    ]
    specs = [
        (gp.GluedForce.CV_ANGLE,    [0, 1, 2]),
        (gp.GluedForce.CV_DIHEDRAL, [0, 1, 3, 4]),
    ]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2
    assert abs(cvs[0] - math.pi/2) < TOL_CV, f"angle CV={cvs[0]:.6f}"
    assert abs(cvs[1] - math.pi/2) < TOL_CV, f"dihedral CV={cvs[1]:.6f}"
    print(f"  test_mixed_cvs: OK  "
          f"(angle={cvs[0]:.5f}, dihedral={cvs[1]:.5f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping angle/dihedral CV tests.")
        sys.exit(0)

    print("Stage 3.2 angle + dihedral CV tests (CUDA platform):")
    test_angle_right(plat)
    test_angle_60(plat)
    test_angle_force_scatter(plat)
    test_angle_numerical_derivative(plat)
    test_dihedral_zero(plat)
    test_dihedral_pi(plat)
    test_dihedral_90(plat)
    test_dihedral_numerical_derivative(plat)
    test_dihedral_force_scatter(plat)
    test_mixed_cvs(plat)
    print("All angle/dihedral CV tests passed.")
