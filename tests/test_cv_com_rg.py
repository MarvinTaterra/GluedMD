"""Stage 3.3 acceptance tests — COM-distance and gyration (Rg) CVs."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-5   # nm or nm
TOL_FORCE = 1e-4   # kJ/mol/nm


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
# COM-distance tests
# ---------------------------------------------------------------------------

def test_com_distance_single_atom_groups(platform):
    """1-atom groups: COM-distance equals simple distance."""
    # atoms[0]=n_group1=1, atoms[1]=0 (g1), atoms[2]=1 (g2)
    # params = masses [1.0, 1.0]
    pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    specs = [(gp.GluedForce.CV_COM_DISTANCE,
              [1, 0, 1],        # n_g1=1, atom0, atom1
              [1.0, 1.0])]      # masses
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.0) < TOL_CV, f"COM dist={cvs[0]:.8f}, expected 1.0"
    print(f"  test_com_distance_single_atom_groups: OK  (d={cvs[0]:.6f} nm)")


def test_com_distance_two_atom_group(platform):
    """Group 1 has 2 equal-mass atoms; COM is their midpoint."""
    # g1 = atoms 0,1 (COM at (0.5, 0, 0)), g2 = atom 2 (at (2, 0, 0))
    # Expected distance = 1.5 nm
    pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COM_DISTANCE,
              [2, 0, 1, 2],        # n_g1=2, atoms 0,1 (g1), atom 2 (g2)
              [1.0, 1.0, 1.0])]    # masses
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.5) < TOL_CV, f"COM dist={cvs[0]:.8f}, expected 1.5"
    print(f"  test_com_distance_two_atom_group: OK  (d={cvs[0]:.6f} nm)")


def test_com_distance_weighted_com(platform):
    """Unequal masses shift the COM. g1: atoms 0(m=1) and 1(m=3) → COM at 0.75."""
    # pos: atom0=(0,0,0), atom1=(1,0,0) → COM_g1 = (0.75, 0, 0)
    # pos: atom2=(2,0,0) → COM_g2 = (2, 0, 0)
    # expected distance = 1.25 nm
    pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COM_DISTANCE,
              [2, 0, 1, 2],
              [1.0, 3.0, 1.0])]   # masses: m0=1, m1=3, m2=1
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.25) < TOL_CV, f"COM dist={cvs[0]:.8f}, expected 1.25"
    print(f"  test_com_distance_weighted_com: OK  (d={cvs[0]:.6f} nm)")


def test_com_distance_force_sum(platform):
    """Force sum over all COM-distance atoms must be zero."""
    pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COM_DISTANCE,
              [2, 0, 1, 2],
              [1.0, 1.0, 1.0])]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(3))
        assert abs(s) < TOL_FORCE, f"COM-dist force sum [{c}] = {s:.6f}"
    print("  test_com_distance_force_sum: OK  (Newton's 3rd law)")


def test_com_distance_numerical_derivative(platform):
    """Finite-difference check on atom 0 x-component.
    Uses dx=1e-3 because GPU positions are float32: at dist~1.5 nm the
    rounding error amplifies to ~1.7e-3 with dx=1e-4."""
    dx = 1e-3
    base = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_COM_DISTANCE,
              [2, 0, 1, 2],
              [1.0, 1.0, 1.0])]

    pos_p = list(base); pos_p[0] = (dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (-dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    # COM_g1 = (atom0 + atom1)/2; dDist/d(atom0.x) = -(m0/M1) * dr.x/dist
    # At base: dr=(1.5,0,0), dist=1.5, m0/M1=0.5 → jac = -0.5
    jac_ana = -0.5
    assert abs(jac_num - jac_ana) < 1e-3, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_com_distance_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


# ---------------------------------------------------------------------------
# Gyration (Rg) tests
# ---------------------------------------------------------------------------

def test_rg_symmetric(platform):
    """4 atoms at (±1, 0, 0) and (0, ±1, 0) with equal masses — Rg = 1 nm."""
    pos = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
           (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    specs = [(gp.GluedForce.CV_GYRATION,
              [0, 1, 2, 3],
              [1.0, 1.0, 1.0, 1.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.0) < TOL_CV, f"Rg={cvs[0]:.8f}, expected 1.0"
    print(f"  test_rg_symmetric: OK  (Rg={cvs[0]:.6f} nm)")


def test_rg_two_atoms(platform):
    """2 equal-mass atoms 2 nm apart — Rg = 1 nm."""
    pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_GYRATION,
              [0, 1],
              [1.0, 1.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.0) < TOL_CV, f"Rg={cvs[0]:.8f}, expected 1.0"
    print(f"  test_rg_two_atoms: OK  (Rg={cvs[0]:.6f} nm)")


def test_rg_weighted(platform):
    """Unequal masses: heavy atom 0 (m=9) at origin, light atom 1 (m=1) at 10 nm.
    COM = 1 nm; Rg = sqrt(9/10*1^2 + 1/10*9^2) = sqrt(0.9+8.1) = 3.0 nm."""
    pos = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_GYRATION,
              [0, 1],
              [9.0, 1.0])]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 3.0) < TOL_CV, f"Rg={cvs[0]:.8f}, expected 3.0"
    print(f"  test_rg_weighted: OK  (Rg={cvs[0]:.6f} nm)")


def test_rg_force_sum(platform):
    """Force sum over all Rg atoms must be zero."""
    pos = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
           (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    specs = [(gp.GluedForce.CV_GYRATION,
              [0, 1, 2, 3],
              [1.0, 1.0, 1.0, 1.0])]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(4))
        assert abs(s) < TOL_FORCE, f"Rg force sum [{c}] = {s:.6f}"
    print("  test_rg_force_sum: OK  (Newton's 3rd law)")


def test_rg_numerical_derivative(platform):
    """Finite-difference check on atom 0 x-component for 2-atom Rg."""
    dx = 1e-3
    base = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [(gp.GluedForce.CV_GYRATION, [0, 1], [1.0, 1.0])]

    pos_p = list(base); pos_p[0] = (dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    cv_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (-dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    cv_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (cv_p - cv_m) / (2.0 * dx)
    # Equal masses, atom0~(0,0,0), atom1=(2,0,0): COM=1, Rg=1
    # jac_0.x = (0.5/1) * (0-1)/1 = -0.5
    jac_ana = -0.5
    assert abs(jac_num - jac_ana) < 1e-3, \
        f"jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f}"
    print(f"  test_rg_numerical_derivative: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_mixed_com_rg(platform):
    """COM-distance and Rg in the same Force."""
    # Atoms: 0=(0,0,0), 1=(2,0,0), 2=(4,0,0), dummy 3=(5,5,5)
    # COM-distance: g1={0}, g2={1} → d=2 nm
    # Rg of {0,1,2}: COM=(2,0,0); Rg=sqrt((4+0+4)/3)=sqrt(8/3)
    pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    specs = [
        (gp.GluedForce.CV_COM_DISTANCE, [1, 0, 1], [1.0, 1.0]),
        (gp.GluedForce.CV_GYRATION,     [0, 1, 2], [1.0, 1.0, 1.0]),
    ]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2
    assert abs(cvs[0] - 2.0) < TOL_CV, f"COM dist={cvs[0]:.6f}"
    expected_rg = math.sqrt(8.0 / 3.0)
    assert abs(cvs[1] - expected_rg) < TOL_CV, f"Rg={cvs[1]:.6f}, expected {expected_rg:.6f}"
    print(f"  test_mixed_com_rg: OK  (COM-d={cvs[0]:.4f}, Rg={cvs[1]:.6f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping COM-distance/Rg CV tests.")
        sys.exit(0)

    print("Stage 3.3 COM-distance + gyration CV tests (CUDA platform):")
    test_com_distance_single_atom_groups(plat)
    test_com_distance_two_atom_group(plat)
    test_com_distance_weighted_com(plat)
    test_com_distance_force_sum(plat)
    test_com_distance_numerical_derivative(plat)
    test_rg_symmetric(plat)
    test_rg_two_atoms(plat)
    test_rg_weighted(plat)
    test_rg_force_sum(plat)
    test_rg_numerical_derivative(plat)
    test_mixed_com_rg(plat)
    print("All COM-distance/Rg CV tests passed.")
