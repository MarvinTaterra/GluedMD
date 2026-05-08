"""Stage 3.1 + 4.1 acceptance tests — distance CV evaluation and chain-rule scatter."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-5   # nm — CV value tolerance
TOL_FORCE = 1e-4   # kJ/mol/nm — force tolerance (float Jacobian × double bias grad)


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_cv_context(positions_nm, cv_atom_pairs, bias_gradients=None,
                    pbc=False, box_nm=None, platform=None):
    """Build a Context with distance CVs.  All parameters fixed before Context creation."""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)
    if pbc and box_nm is not None:
        sys_.setDefaultPeriodicBoxVectors(
            mm.Vec3(box_nm, 0, 0), mm.Vec3(0, box_nm, 0), mm.Vec3(0, 0, box_nm))

    f = gp.GluedForce()
    if pbc:
        f.setUsesPeriodicBoundaryConditions(True)
    for a_idx, b_idx in cv_atom_pairs:
        atoms_v = gp.vectori()
        atoms_v.append(a_idx)
        atoms_v.append(b_idx)
        params_v = gp.vectord()
        f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, atoms_v, params_v)
    if bias_gradients:
        grads_v = gp.vectord()
        for g in bias_gradients:
            grads_v.append(g)
        f.setTestBiasGradients(grads_v)

    sys_.addForce(f)
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def get_forces(ctx):
    state = ctx.getState(getForces=True)
    raw   = state.getForces(asNumpy=False)
    unit  = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def eval_cv(ctx, f):
    """Trigger a force evaluation and return CV values."""
    ctx.getState(getForces=True)
    return list(f.getLastCVValues(ctx))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_distance_value(platform):
    """Single distance CV: atoms 0 and 1, expected d = 0.5 nm."""
    # r01 = sqrt(0.3^2 + 0.4^2) = 0.5 nm
    positions = [(0.0, 0.0, 0.0), (0.3, 0.4, 0.0), (0.5, 0.5, 0.5)]
    ctx, f = make_cv_context(positions, [(0, 1)], platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 1
    assert abs(cvs[0] - 0.5) < TOL_CV, f"CV={cvs[0]:.8f}, expected 0.5"
    print(f"  test_distance_value: OK  (d={cvs[0]:.6f} nm, expected 0.5)")


def test_force_scatter(platform):
    """One distance CV along x, bias gradient = 1.0.
    Expected: force_0 = (+1, 0, 0), force_1 = (-1, 0, 0) kJ/mol/nm."""
    # dr = (1,0,0), d = 1 nm → jac_0 = (-1,0,0), jac_1 = (+1,0,0)
    # force = -(dU/dCV)·jac → f_0 = -(1.0)(-1,0,0) = (+1,0,0)
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    ctx, f = make_cv_context(positions, [(0, 1)], bias_gradients=[1.0],
                             platform=platform)
    forces = get_forces(ctx)
    fx0, fy0, fz0 = forces[0]
    fx1, fy1, fz1 = forces[1]
    assert abs(fx0 -  1.0) < TOL_FORCE, f"atom 0 fx={fx0:.6f}, expected +1.0"
    assert abs(fy0)        < TOL_FORCE
    assert abs(fz0)        < TOL_FORCE
    assert abs(fx1 - -1.0) < TOL_FORCE, f"atom 1 fx={fx1:.6f}, expected -1.0"
    assert abs(fy1)        < TOL_FORCE
    assert abs(fz1)        < TOL_FORCE
    print(f"  test_force_scatter: OK  (f0=({fx0:.3f},{fy0:.3f},{fz0:.3f}), "
          f"f1=({fx1:.3f},{fy1:.3f},{fz1:.3f}))")


def test_numerical_derivative(platform):
    """Verify Jacobian via finite difference on atom 0 x-component."""
    dx = 1e-4  # nm
    base = [(0.0, 0.0, 0.0), (0.3, 0.4, 0.0), (0.7, 0.7, 0.7)]
    d_ref = math.sqrt(0.09 + 0.16)  # 0.5 nm

    pos_p = list(base); pos_p[0] = (dx, 0.0, 0.0)
    ctx_p, f_p = make_cv_context(pos_p, [(0, 1)], platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (-dx, 0.0, 0.0)
    ctx_m, f_m = make_cv_context(pos_m, [(0, 1)], platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2 * dx)
    jac_ana = -(0.3 - 0.0) / d_ref   # -0.6
    assert abs(jac_num - jac_ana) < 5e-4, \
        f"Jacobian: numerical={jac_num:.6f}, analytical={jac_ana:.6f}"
    print(f"  test_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_pbc_distance(platform):
    """Distance CV with PBC: atoms at 0.0 and 0.99 in 1 nm box → d = 0.01 nm."""
    positions = [(0.0, 0.0, 0.0), (0.99, 0.0, 0.0), (0.5, 0.5, 0.5)]
    ctx, f = make_cv_context(positions, [(0, 1)], pbc=True, box_nm=1.0,
                             platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.01) < TOL_CV, f"PBC d={cvs[0]:.8f}, expected 0.01"
    print(f"  test_pbc_distance: OK  (d={cvs[0]:.8f} nm)")


def test_multiple_cvs(platform):
    """Two distance CVs: verify both values independently."""
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    ctx, f = make_cv_context(positions, [(0, 1), (0, 2)], platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2
    assert abs(cvs[0] - 1.0) < TOL_CV, f"CV0={cvs[0]:.6f}, expected 1.0"
    assert abs(cvs[1] - 2.0) < TOL_CV, f"CV1={cvs[1]:.6f}, expected 2.0"
    print(f"  test_multiple_cvs: OK  (d01={cvs[0]:.4f}, d02={cvs[1]:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping distance CV tests.")
        sys.exit(0)

    print("Stage 3.1 + 4.1 distance CV tests (CUDA platform):")
    test_distance_value(plat)
    test_force_scatter(plat)
    test_numerical_derivative(plat)
    test_pbc_distance(plat)
    test_multiple_cvs(plat)
    print("All distance CV tests passed.")
