"""Stage 3.5 acceptance tests — no-fit (TYPE=SIMPLE) RMSD CV."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-4   # nm
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
# RMSD tests
# ---------------------------------------------------------------------------

def test_rmsd_single_atom(platform):
    """1 atom displaced by 0.3 nm along x from reference (0,0,0). RMSD = 0.3."""
    pos = [(0.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    # params = [ref_x0, ref_y0, ref_z0] for atom 0
    specs = [(gp.GluedForce.CV_RMSD,
              [0],
              [0.0, 0.0, 0.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.3) < TOL_CV, f"RMSD={cvs[0]:.8f}, expected 0.3"
    print(f"  test_rmsd_single_atom: OK  (RMSD={cvs[0]:.6f} nm)")


def test_rmsd_two_atoms_equal_displacement(platform):
    """2 atoms both displaced by 0.3 nm along x. RMSD = 0.3."""
    pos = [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_RMSD,
              [0, 1],
              [0.0, 0.0, 0.0,   # ref for atom 0
               0.0, 0.0, 0.0])] # ref for atom 1
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    # dr0 = (0.3,0,0), dr1 = (-0.3,0,0); RMSD = sqrt((0.09+0.09)/2) = 0.3
    assert abs(cvs[0] - 0.3) < TOL_CV, f"RMSD={cvs[0]:.8f}, expected 0.3"
    print(f"  test_rmsd_two_atoms_equal: OK  (RMSD={cvs[0]:.6f} nm)")


def test_rmsd_unequal_displacements(platform):
    """2 atoms with displacements 0 and 0.6 nm. RMSD = sqrt(0.18) = 0.3*sqrt(2)."""
    pos = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (5.0, 5.0, 5.0)]
    refs = [0.0, 0.0, 0.0,   # ref for atom 0 (no displacement)
            0.0, 0.0, 0.0]   # ref for atom 1 → dr = 0.6
    specs = [(gp.GluedForce.CV_RMSD, [0, 1], refs)]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    expected = math.sqrt(0.36 / 2.0)   # sqrt(0.18) ≈ 0.4243
    assert abs(cvs[0] - expected) < TOL_CV, \
        f"RMSD={cvs[0]:.8f}, expected {expected:.8f}"
    print(f"  test_rmsd_unequal_displacements: OK  "
          f"(RMSD={cvs[0]:.6f}, expected {expected:.6f})")


def test_rmsd_3d_displacement(platform):
    """1 atom displaced by (0.1, 0.2, 0.2) from ref (0,0,0): RMSD = 0.3 nm."""
    pos = [(0.1, 0.2, 0.2), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_RMSD, [0], [0.0, 0.0, 0.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    expected = math.sqrt(0.01 + 0.04 + 0.04)  # sqrt(0.09) = 0.3
    assert abs(cvs[0] - expected) < TOL_CV, \
        f"RMSD={cvs[0]:.8f}, expected {expected:.8f}"
    print(f"  test_rmsd_3d_displacement: OK  (RMSD={cvs[0]:.6f})")


def test_rmsd_analytical_force(platform):
    """Force on atom 0 with 1-atom RMSD, bias_gradient=1.0.
    jac_0.x = dr.x/(N*RMSD) = 0.3/(1*0.3) = 1.0; Force_0.x = -1.0."""
    pos = [(0.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_RMSD, [0], [0.0, 0.0, 0.0])]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    assert abs(forces[0][0] - (-1.0)) < TOL_FORCE, \
        f"F0.x={forces[0][0]:.6f}, expected -1.0"
    assert abs(forces[0][1]) < TOL_FORCE, f"F0.y={forces[0][1]:.6f}"
    assert abs(forces[0][2]) < TOL_FORCE, f"F0.z={forces[0][2]:.6f}"
    print(f"  test_rmsd_analytical_force: OK  "
          f"(F0=({forces[0][0]:.4f},{forces[0][1]:.4f},{forces[0][2]:.4f}))")


def test_rmsd_two_atom_force(platform):
    """2-atom RMSD at equal symmetric displacements.
    jac_0.x = 0.3/(2*0.3) = 0.5, jac_1.x = -0.3/(2*0.3) = -0.5.
    With bias_gradient=1.0: F0.x=-0.5, F1.x=+0.5."""
    pos = [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    refs = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
    specs = [(gp.GluedForce.CV_RMSD, [0, 1], refs)]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    assert abs(forces[0][0] - (-0.5)) < TOL_FORCE, \
        f"F0.x={forces[0][0]:.6f}, expected -0.5"
    assert abs(forces[1][0] - (0.5)) < TOL_FORCE, \
        f"F1.x={forces[1][0]:.6f}, expected +0.5"
    print(f"  test_rmsd_two_atom_force: OK  "
          f"(F0.x={forces[0][0]:.4f}, F1.x={forces[1][0]:.4f})")


def test_rmsd_numerical_derivative(platform):
    """FD check on atom 0 x-component.
    2-atom case: RMSD=0.3, jac_0.x = 0.3/(2*0.3) = 0.5."""
    dx = 1e-4
    base = [(0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    refs = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
    specs = [(gp.GluedForce.CV_RMSD, [0, 1], refs)]

    pos_p = list(base); pos_p[0] = (0.3 + dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (0.3 - dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    jac_ana = 0.5
    assert abs(jac_num - jac_ana) < 1e-3, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_rmsd_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_rmsd_non_zero_reference(platform):
    """Reference not at origin. atom 0 at (1.3,0,0), ref at (1.0,0,0) → dr=0.3."""
    pos = [(1.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_RMSD, [0], [1.0, 0.0, 0.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.3) < TOL_CV, f"RMSD={cvs[0]:.8f}, expected 0.3"
    print(f"  test_rmsd_non_zero_reference: OK  (RMSD={cvs[0]:.6f})")


def test_rmsd_mixed_with_distance(platform):
    """RMSD + distance CV in the same Force."""
    # atom 0 at (0.3,0,0), ref (0,0,0) → RMSD=0.3
    # atom 0 to atom 1=(1.3,0,0) → dist=1.0
    pos = [(0.3, 0.0, 0.0), (1.3, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [
        (gp.GluedForce.CV_RMSD,     [0],    [0.0, 0.0, 0.0]),
        (gp.GluedForce.CV_DISTANCE, [0, 1], []),
    ]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2
    assert abs(cvs[0] - 0.3) < TOL_CV, f"RMSD={cvs[0]:.6f}"
    assert abs(cvs[1] - 1.0) < TOL_CV, f"dist={cvs[1]:.6f}"
    print(f"  test_rmsd_mixed_with_distance: OK  "
          f"(RMSD={cvs[0]:.4f}, dist={cvs[1]:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping RMSD CV tests.")
        sys.exit(0)

    print("Stage 3.5 RMSD CV tests (CUDA platform):")
    test_rmsd_single_atom(plat)
    test_rmsd_two_atoms_equal_displacement(plat)
    test_rmsd_unequal_displacements(plat)
    test_rmsd_3d_displacement(plat)
    test_rmsd_analytical_force(plat)
    test_rmsd_two_atom_force(plat)
    test_rmsd_numerical_derivative(plat)
    test_rmsd_non_zero_reference(plat)
    test_rmsd_mixed_with_distance(plat)
    print("All RMSD CV tests passed.")
