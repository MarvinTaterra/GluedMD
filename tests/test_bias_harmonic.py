"""Stage 5.1 acceptance tests — harmonic restraint bias."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_E = 1e-5
TOL_F = 1e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_system(positions_nm, cv_specs, bias_specs, platform=None):
    """cv_specs: [(cv_type, atoms, params)], bias_specs: [(cv_index, k, s0)]."""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)

    f = gp.GluedForce()
    cv_idx = {}
    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms:
            av.append(a)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        cv_idx[len(cv_idx)] = f.addCollectiveVariable(cv_type, av, pv)

    # Each bias_spec: (list_of_cv_indices, [k0, s0_0, k1, s0_1, ...])
    for cv_list, params in bias_specs:
        civ = mm.vectori()
        for c in cv_list:
            civ.append(c)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        iv = mm.vectori()  # no integer params for harmonic
        f.addBias(gp.GluedForce.BIAS_HARMONIC, civ, pv, iv)

    sys_.addForce(f)
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def get_energy(ctx):
    state = ctx.getState(getEnergy=True, getForces=True)
    return state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)


def get_forces(ctx):
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_harmonic_energy_basic(platform):
    """V = k*(s-s0)^2/2 for a distance CV at known separation.
    k=100, s0=0.5, atoms at 0 and 1.0nm → s=1.0, V=100*(1.0-0.5)^2/2=12.5 kJ/mol."""
    k, s0, dist = 100.0, 0.5, 1.0
    pos = [(0.0, 0.0, 0.0), (dist, 0.0, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    bias_specs = [([0], [k, s0])]
    ctx, f = make_system(pos, cv_specs, bias_specs, platform)
    E = get_energy(ctx)
    expected = 0.5 * k * (dist - s0)**2
    assert abs(E - expected) < TOL_E, f"E={E:.8f}, expected {expected:.8f}"
    print(f"  test_harmonic_energy_basic: OK  (E={E:.6f} kJ/mol)")


def test_harmonic_energy_at_minimum(platform):
    """Atom exactly at s0 → E = 0."""
    k, s0 = 200.0, 1.0
    pos = [(0.0, 0.0, 0.0), (s0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    bias_specs = [([0], [k, s0])]
    ctx, f = make_system(pos, cv_specs, bias_specs, platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.8f}, expected 0"
    print(f"  test_harmonic_energy_at_minimum: OK  (E={E:.2e})")


def test_harmonic_force_direction(platform):
    """Force on atom 1 must point from atom 1 toward s0 when s > s0.
    k=100, s0=0.5, s=1.0nm → F_x on atom 1 = -k*(s-s0) = -50 kJ/mol/nm (toward atom 0)."""
    k, s0, dist = 100.0, 0.5, 1.0
    pos = [(0.0, 0.0, 0.0), (dist, 0.0, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    bias_specs = [([0], [k, s0])]
    ctx, f = make_system(pos, cv_specs, bias_specs, platform)
    forces = get_forces(ctx)
    # dV/ds = k*(s-s0) = 50; dS/dr_1 = +x_hat → F_1.x = -dV/ds = -50
    expected_fx1 = -k * (dist - s0)
    assert abs(forces[1][0] - expected_fx1) < TOL_F, \
        f"F1.x={forces[1][0]:.6f}, expected {expected_fx1:.6f}"
    # Newton's 3rd law: F0 = -F1
    assert abs(forces[0][0] + forces[1][0]) < TOL_F, \
        f"Newton 3rd law violated: F0.x={forces[0][0]:.6f}, F1.x={forces[1][0]:.6f}"
    print(f"  test_harmonic_force_direction: OK  (F1.x={forces[1][0]:.4f})")


def test_harmonic_negative_displacement(platform):
    """s < s0 → force pushes away from s0 (restoring)."""
    k, s0, dist = 100.0, 2.0, 1.0
    pos = [(0.0, 0.0, 0.0), (dist, 0.0, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    bias_specs = [([0], [k, s0])]
    ctx, f = make_system(pos, cv_specs, bias_specs, platform)
    E = get_energy(ctx)
    forces = get_forces(ctx)
    expected_E = 0.5 * k * (dist - s0)**2
    expected_fx1 = -k * (dist - s0)  # positive: pushes atom 1 away from atom 0
    assert abs(E - expected_E) < TOL_E, f"E={E:.6f}, expected {expected_E:.6f}"
    assert abs(forces[1][0] - expected_fx1) < TOL_F, \
        f"F1.x={forces[1][0]:.6f}, expected {expected_fx1:.6f}"
    print(f"  test_harmonic_negative_displacement: OK  (E={E:.4f}, F1.x={forces[1][0]:.4f})")


def test_harmonic_two_cvs(platform):
    """Two independent distance CVs each with their own harmonic bias.
    Total E = V1 + V2."""
    k1, s0_1 = 100.0, 0.5
    k2, s0_2 = 50.0, 1.5
    d1, d2 = 1.0, 2.0
    pos = [(0.0, 0.0, 0.0), (d1, 0.0, 0.0), (0.0, d2, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [
        (gp.GluedForce.CV_DISTANCE, [0, 1], []),
        (gp.GluedForce.CV_DISTANCE, [0, 2], []),
    ]
    # Both CVs in one bias, two sets of [k, s0]
    bias_specs = [([0, 1], [k1, s0_1, k2, s0_2])]
    ctx, f = make_system(pos, cv_specs, bias_specs, platform)
    E = get_energy(ctx)
    expected = 0.5*k1*(d1-s0_1)**2 + 0.5*k2*(d2-s0_2)**2
    assert abs(E - expected) < TOL_E * 10, f"E={E:.6f}, expected {expected:.6f}"
    print(f"  test_harmonic_two_cvs: OK  (E={E:.6f} vs {expected:.6f})")


def test_harmonic_numerical_derivative(platform):
    """Finite-difference check: F_x ≈ -(V(+dx) - V(-dx)) / (2*dx).

    dx=1e-2 is used (not 1e-3) because CUDA stores positions as float32.
    The float quantisation error at ~1.5 nm is ~ulp(1.5) ≈ 2e-7 nm, which
    is amplified by 1/dx in the central-difference formula; dx=1e-2 keeps
    the amplified error well below 1e-4 kJ/mol/nm.
    All three evaluations use the same context (setPositions) to avoid
    context-initialisation differences.
    """
    dx = 1e-2
    k, s0 = 200.0, 1.0
    base_pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    bias_specs = [([0], [k, s0])]
    ctx, _ = make_system(base_pos, cv_specs, bias_specs, platform)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1.5 + dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_p = get_energy(ctx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1.5 - dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_m = get_energy(ctx)

    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1.5, 0, 0), mm.Vec3(5, 5, 5)])
    F_ana = get_forces(ctx)[1][0]

    assert abs(F_num - F_ana) < TOL_F * 10, \
        f"F_num={F_num:.6f}, F_ana={F_ana:.6f}"
    print(f"  test_harmonic_numerical_derivative: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping harmonic bias tests.")
        sys.exit(0)

    print("Stage 5.1 harmonic bias tests (CUDA platform):")
    test_harmonic_energy_basic(plat)
    test_harmonic_energy_at_minimum(plat)
    test_harmonic_force_direction(plat)
    test_harmonic_negative_displacement(plat)
    test_harmonic_two_cvs(plat)
    test_harmonic_numerical_derivative(plat)
    print("All harmonic bias tests passed.")
