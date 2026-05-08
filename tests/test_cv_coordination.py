"""Stage 3.4 acceptance tests — coordination number CV."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-4   # dimensionless
TOL_FORCE = 1e-3   # kJ/mol/nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_context(positions_nm, cv_specs, bias_gradients=None, platform=None):
    """cv_specs: list of (cv_type, atoms_list, params_list)."""
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
    raw = state.getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


# ---------------------------------------------------------------------------
# Coordination number tests
# ---------------------------------------------------------------------------

def test_coord_exact_value(platform):
    """1A + 1B at r=r0/3=0.5 nm, n=1 m=2: CN = 1/(1+x) = 3/4 = 0.75."""
    # atoms[0]=nA=1, atoms[1]=atom0 (A), atoms[2]=atom1 (B)
    # params=[r0=1.5, n=1, m=2]
    pos = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1],
              [1.5, 1.0, 2.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.75) < TOL_CV, f"CN={cvs[0]:.8f}, expected 0.75"
    print(f"  test_coord_exact_value: OK  (CN={cvs[0]:.6f})")


def test_coord_zero(platform):
    """Atoms beyond r0 still contribute: f(x=2,n=6,m=12) = (1-64)/(1-4096) = 63/4095."""
    pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1],
              [1.0, 6.0, 12.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    expected = (1.0 - 2.0**6) / (1.0 - 2.0**12)  # ≈ 0.01538
    assert abs(cvs[0] - expected) < TOL_CV * 10, \
        f"CN={cvs[0]:.8f}, expected {expected:.8f}"
    print(f"  test_coord_zero: OK  (CN={cvs[0]:.8f}, expected {expected:.8f})")


def test_coord_two_pairs(platform):
    """1A + 2B: both pairs contribute per PLUMED rational switch.
    r0=1.0, n=1, m=2: f(x=0.5) = (1-0.5)/(1-0.25) = 2/3; f(x=2.0) = (1-2)/(1-4) = 1/3.
    CN = 2/3 + 1/3 = 1."""
    pos = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1, 2],   # nA=1, atom0=A, atoms1,2=B
              [1.0, 1.0, 2.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    f05 = (1.0 - 0.5**1) / (1.0 - 0.5**2)   # 2/3
    f20 = (1.0 - 2.0**1) / (1.0 - 2.0**2)   # 1/3
    expected = f05 + f20                       # = 1.0
    assert abs(cvs[0] - expected) < TOL_CV, f"CN={cvs[0]:.8f}, expected {expected:.6f}"
    print(f"  test_coord_two_pairs: OK  (CN={cvs[0]:.6f}, expected {expected:.6f})")


def test_coord_force_sum(platform):
    """Force sum over coordination atoms must be zero (Newton's 3rd law)."""
    pos = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1],
              [1.5, 1.0, 2.0])]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(2))
        assert abs(s) < TOL_FORCE, f"coord force sum [{c}] = {s:.6f}"
    print("  test_coord_force_sum: OK  (Newton's 3rd law)")


def test_coord_analytical_force(platform):
    """Analytical force on atom A.x for 1A+1B with n=1, m=2, r=0.5, r0=1.5.
    jac_A.x = df/dr * (r_A.x - r_B.x)/r = (-3/8) * (-1) = +3/8.
    Force_A.x = -(dU/dCV) * jac_A.x = -1.0 * 3/8 = -3/8 = -0.375."""
    pos = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1],
              [1.5, 1.0, 2.0])]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    assert abs(forces[0][0] - (-0.375)) < TOL_FORCE, \
        f"F_A.x={forces[0][0]:.6f}, expected -0.375"
    assert abs(forces[0][1]) < TOL_FORCE, f"F_A.y={forces[0][1]:.6f}"
    assert abs(forces[0][2]) < TOL_FORCE, f"F_A.z={forces[0][2]:.6f}"
    print(f"  test_coord_analytical_force: OK  "
          f"(F_A=({forces[0][0]:.4f},{forces[0][1]:.4f},{forces[0][2]:.4f}))")


def test_coord_numerical_derivative(platform):
    """Finite-difference Jacobian check on atom A x-component.
    At r=0.5, r0=1.5, n=1, m=2: analytical jac = 0.375."""
    dx = 1e-3
    base = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [1, 0, 1],
              [1.5, 1.0, 2.0])]

    pos_p = list(base); pos_p[0] = ( dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (-dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    jac_ana = 0.375
    assert abs(jac_num - jac_ana) < 2e-3, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_coord_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_coord_2a_2b(platform):
    """2A + 2B: symmetric config, 4 pairs each at r=sqrt(2)*0.5 nm.
    r0=1.5, n=1, m=2: x=sqrt(2)*0.5/1.5, CN = 4 * 1/(1+x)."""
    r = math.sqrt(2.0) * 0.5
    x = r / 1.5
    expected_cn = 4.0 / (1.0 + x)
    # atom 0,1 in A; atom 2,3 in B
    pos = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
           (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COORDINATION,
              [2, 0, 1, 2, 3],   # nA=2, A={0,1}, B={2,3}
              [1.5, 1.0, 2.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - expected_cn) < 2e-3, \
        f"CN={cvs[0]:.6f}, expected {expected_cn:.6f}"
    print(f"  test_coord_2a_2b: OK  (CN={cvs[0]:.6f}, expected {expected_cn:.6f})")


def test_coord_mixed_with_distance(platform):
    """Coordination + distance CV in the same Force."""
    # atom 0=(0,0,0), atom 1=(0.5,0,0), atom 2=(2,0,0) dummy
    # CV0: coord 1A+1B, r=0.5, r0=1.5, n=1,m=2 → CN=0.75
    # CV1: distance atoms 0-1 → d=0.5 nm
    pos = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (2.0, 0.0, 0.0)]
    specs = [
        (gp.GluedForce.CV_COORDINATION, [1, 0, 1], [1.5, 1.0, 2.0]),
        (gp.GluedForce.CV_DISTANCE,     [0, 1],    []),
    ]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2
    assert abs(cvs[0] - 0.75) < TOL_CV, f"CN={cvs[0]:.6f}"
    assert abs(cvs[1] - 0.5)  < TOL_CV, f"dist={cvs[1]:.6f}"
    print(f"  test_coord_mixed_with_distance: OK  "
          f"(CN={cvs[0]:.4f}, dist={cvs[1]:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping coordination CV tests.")
        sys.exit(0)

    print("Stage 3.4 coordination number CV tests (CUDA platform):")
    test_coord_exact_value(plat)
    test_coord_zero(plat)
    test_coord_two_pairs(plat)
    test_coord_force_sum(plat)
    test_coord_analytical_force(plat)
    test_coord_numerical_derivative(plat)
    test_coord_2a_2b(plat)
    test_coord_mixed_with_distance(plat)
    print("All coordination CV tests passed.")
