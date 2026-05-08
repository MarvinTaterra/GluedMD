"""Stage 3.17 — CV_DIPOLE acceptance tests.

CV = μ_k  (k=0→x, 1→y, 2→z)  or  CV = |μ|  (k=3)
μ = Σ_i q_i * r_i

API:
  atoms  = [a0, a1, ..., aN-1]
  params = [q0, q1, ..., qN-1, component]
  component: 0=x, 1=y, 2=z, 3=|μ|

Tests: value correctness, force (Jacobian) direction, numerical gradient.
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


def _dipole(positions, atoms, charges):
    """Reference dipole vector."""
    mu = [0.0, 0.0, 0.0]
    for a, q in zip(atoms, charges):
        for k in range(3):
            mu[k] += q * positions[a][k]
    return mu


def _make_ctx(positions, atom_list, charges, comp, platform):
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atom_list: av.append(a)
    pv = mm.vectord()
    for q in charges: pv.append(q)
    pv.append(comp)
    f.addCollectiveVariable(gp.GluedForce.CV_DIPOLE, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


def test_dipole_two_opposite_charges(platform):
    """Two opposite charges at known positions → analytical dipole."""
    # q=+1 at (0.5,0,0) and q=-1 at (0,0,0) → μ.x = 1*0.5 + (-1)*0 = 0.5
    positions = [[0.5, 0.2, 0.1], [0.0, 0.0, 0.0]] + [[0.0]*3]*2
    charges = [1.0, -1.0]
    for comp, expected in [(0, 0.5), (1, 0.2), (2, 0.1)]:
        ctx, f = _make_ctx(positions, [0, 1], charges, comp, platform)
        cvs = f.getLastCVValues(ctx)
        assert abs(cvs[0] - expected) < TOL_CV, \
            f"comp={comp}: expected {expected:.4f}, got {cvs[0]:.6f}"
    print("  test_dipole_two_opposite_charges: OK")


def test_dipole_x_component(platform):
    """μ.x = Σ q_i * x_i matches reference."""
    rng = random.Random(11)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(6)]
    atoms = [0, 1, 3, 5]
    charges = [0.5, -0.3, 0.8, -0.2]
    ctx, f = _make_ctx(positions, atoms, charges, comp=0, platform=platform)
    cvs = f.getLastCVValues(ctx)
    mu = _dipole(positions, atoms, charges)
    assert abs(cvs[0] - mu[0]) < TOL_CV, f"expected {mu[0]:.6f}, got {cvs[0]:.6f}"
    print(f"  test_dipole_x_component: OK  (cv={cvs[0]:.6f}, ref={mu[0]:.6f})")


def test_dipole_magnitude(platform):
    """Component=3 returns |μ| = sqrt(μ.x²+μ.y²+μ.z²)."""
    rng = random.Random(22)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(5)]
    atoms = [0, 1, 2, 3]
    charges = [1.0, -0.5, 0.3, -0.8]
    ctx, f = _make_ctx(positions, atoms, charges, comp=3, platform=platform)
    cvs = f.getLastCVValues(ctx)
    mu = _dipole(positions, atoms, charges)
    expected = math.sqrt(sum(v**2 for v in mu))
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_dipole_magnitude: OK  (|μ|={cvs[0]:.6f}, ref={expected:.6f})")


def test_dipole_force_component(platform):
    """Force on atom a for component CV: F_a[k] = -q_a (bias_grad=1)."""
    positions = [[0.3, 0.2, 0.1], [0.1, 0.4, 0.5]] + [[0.0]*3]*2
    charges = [2.0, -1.0]
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    # component=0 (x): dCV/dr_a_x = q_a, so F_a_x = -q_a
    ctx, f = _make_ctx(positions, [0, 1], charges, comp=0, platform=platform)
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    fx0 = raw[0][0].value_in_unit(unit)
    fx1 = raw[1][0].value_in_unit(unit)
    assert abs(fx0 - (-charges[0])) < TOL_F, f"F_a0_x: expected {-charges[0]}, got {fx0}"
    assert abs(fx1 - (-charges[1])) < TOL_F, f"F_a1_x: expected {-charges[1]}, got {fx1}"
    # y and z forces should be zero for x-component CV
    fy0 = raw[0][1].value_in_unit(unit)
    assert abs(fy0) < TOL_F, f"F_a0_y should be 0, got {fy0}"
    print(f"  test_dipole_force_component: OK  (F0x={fx0:.4f}, F1x={fx1:.4f})")


def test_dipole_numerical_gradient(platform):
    """Finite-difference check for all components and |μ|."""
    rng = random.Random(55)
    positions = [[rng.uniform(0.1, 0.9) for _ in range(3)] for _ in range(5)]
    atoms = [0, 1, 2, 3]
    charges = [1.2, -0.7, 0.4, -0.9]
    h = 1e-4

    for comp in range(4):
        n_atoms = len(positions)
        sys = mm.System()
        for _ in range(n_atoms): sys.addParticle(12.0)
        f_cv = gp.GluedForce()
        f_cv.setUsesPeriodicBoundaryConditions(False)
        av = mm.vectori()
        for a in atoms: av.append(a)
        pv = mm.vectord()
        for q in charges: pv.append(q)
        pv.append(comp)
        f_cv.addCollectiveVariable(gp.GluedForce.CV_DIPOLE, av, pv)
        f_cv.setTestBiasGradients(mm.vectord([1.0]))
        sys.addForce(f_cv)
        ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

        def cv_ref(pos):
            mu = _dipole(pos, atoms, charges)
            if comp == 0: return mu[0]
            if comp == 1: return mu[1]
            if comp == 2: return mu[2]
            return math.sqrt(sum(v**2 for v in mu))

        max_err = 0.0
        unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
        for ai in atoms:
            for ci in range(3):
                pos_p = [list(p) for p in positions]
                pos_m = [list(p) for p in positions]
                pos_p[ai][ci] += h
                pos_m[ai][ci] -= h
                fd = (cv_ref(pos_p) - cv_ref(pos_m)) / (2*h)
                ctx.setPositions(positions)
                state = ctx.getState(getForces=True)
                raw = state.getForces(asNumpy=False)
                f_anal = -raw[ai][ci].value_in_unit(unit)
                err = abs(fd - f_anal)
                if err > max_err:
                    max_err = err

        assert max_err < TOL_F, f"comp={comp}: max gradient error {max_err:.2e} > {TOL_F}"
    print(f"  test_dipole_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_DIPOLE tests.")
        sys.exit(0)
    print("Stage 3.17 — CV_DIPOLE tests (CUDA platform):")
    test_dipole_two_opposite_charges(plat)
    test_dipole_x_component(plat)
    test_dipole_magnitude(plat)
    test_dipole_force_component(plat)
    test_dipole_numerical_gradient(plat)
    print("All CV_DIPOLE tests passed.")
