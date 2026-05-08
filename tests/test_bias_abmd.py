"""Stage 5.7 acceptance tests — ABMD (Adiabatic Bias MD) ratchet bias.

Physics (PLUMED-compatible formula, Marchi & Ballone 1999):
  rho(t)    = (CV(t) - TO)^2         (squared distance from target)
  rhoMin(t) = min over tau <= t of rho(tau)   (running minimum)

  V = 0.5 * kappa * (rho - rhoMin)^2   when rho > rhoMin
  V = 0                                  when rho <= rhoMin

  Force on CV: -dV/d(CV) = -2 * kappa * (rho - rhoMin) * (CV - TO)

  On the first getState() call, rhoMin is initialized to rho(CV_initial).
  Subsequent calls with rho < rhoMin update rhoMin (moving closer to TO).
  Calls with rho > rhoMin return non-zero bias.

Parameters: [kappa_0, TO_0, kappa_1, TO_1, ...]
integerParameters: [] (empty)
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

TOL_E = 1e-5
TOL_F = 1e-4


def make_system(positions_nm, cv_specs, bias_specs, platform=None):
    """bias_specs: [(cv_list, [kappa_0, TO_0, ...])]"""
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

    for cv_list, params in bias_specs:
        civ = mm.vectori()
        for c in cv_list:
            civ.append(c)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        iv = mm.vectori()
        f.addBias(gp.GluedForce.BIAS_ABMD, civ, pv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def get_energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        mm.unit.kilojoules_per_mole)


def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_abmd_first_call_no_bias(platform):
    """First getState() always sets rhoMin; V = 0."""
    k, TO = 100.0, 1.0
    d = 0.5  # cv = 0.5, rho = (0.5-1.0)^2 = 0.25; first call → rhoMin = 0.25, V = 0
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.8f}, expected 0 on first call"
    print(f"  test_abmd_first_call_no_bias: OK  (E={E:.2e})")


def test_abmd_at_target_no_bias(platform):
    """CV exactly at TO: rho = 0 = rhoMin → V = 0 always."""
    k, TO = 100.0, 1.0
    pos = [(0, 0, 0), (TO, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.2e}, expected 0"
    print(f"  test_abmd_at_target_no_bias: OK  (E={E:.2e})")


def test_abmd_approach_then_retreat(platform):
    """rhoMin is updated when CV moves closer to TO; bias when CV retreats.

    Flow:
      1. cv=0.8 → rho=(0.8-1.0)^2=0.04 → rhoMin=0.04, V=0
      2. cv=0.9 → rho=(0.9-1.0)^2=0.01 < rhoMin=0.04 → rhoMin=0.01, V=0
      3. cv=0.5 → rho=(0.5-1.0)^2=0.25 > rhoMin=0.01 → V=0.5*k*(0.25-0.01)^2
    """
    k, TO = 100.0, 1.0
    # Step 1: prime rhoMin at cv=0.8
    pos_1 = [(0, 0, 0), (0.8, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos_1,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    E1 = get_energy(ctx)
    assert abs(E1) < TOL_E

    # Step 2: move closer (cv=0.9) → rhoMin updates
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.9, 0, 0), mm.Vec3(5, 5, 5)])
    E2 = get_energy(ctx)
    assert abs(E2) < TOL_E, f"Moving closer should give V=0, got {E2:.6f}"

    # Step 3: retreat to cv=0.5 → rho=0.25 > rhoMin=0.01 → bias
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0), mm.Vec3(5, 5, 5)])
    E3 = get_energy(ctx)
    rhoMin = (0.9 - 1.0) ** 2  # 0.01
    rho = (0.5 - 1.0) ** 2     # 0.25
    expected = 0.5 * k * (rho - rhoMin) ** 2
    assert abs(E3 - expected) < TOL_E, f"E={E3:.8f}, expected {expected:.8f}"
    print(f"  test_abmd_approach_then_retreat: OK  (E3={E3:.6f})")


def test_abmd_force_toward_target(platform):
    """Force points toward TO when CV has retreated from closest approach.

    cv=0.9 (prime rhoMin), then cv=0.5 → force on atom1 must point in +x (toward TO=1.0).
    """
    k, TO = 100.0, 1.0
    pos_prime = [(0, 0, 0), (0.9, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos_prime,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    get_energy(ctx)  # prime rhoMin

    cv_far = 0.5
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_far, 0, 0), mm.Vec3(5, 5, 5)])
    forces = get_forces(ctx)

    rhoMin = (0.9 - TO) ** 2   # 0.01
    rho = (cv_far - TO) ** 2    # 0.25
    cvDist = cv_far - TO         # -0.5
    # dV/dcv = 2*k*(rho-rhoMin)*cvDist → force = -dV/dcv = -2k*(rho-rhoMin)*cvDist
    expected_fx1 = -2.0 * k * (rho - rhoMin) * cvDist  # positive → toward TO
    assert expected_fx1 > 0, "Expected positive force (toward TO)"
    assert abs(forces[1][0] - expected_fx1) < TOL_F, \
        f"F1.x={forces[1][0]:.6f}, expected {expected_fx1:.6f}"
    assert abs(forces[0][0] + forces[1][0]) < TOL_F, "Newton 3rd law"
    print(f"  test_abmd_force_toward_target: OK  (F1.x={forces[1][0]:.4f})")


def test_abmd_rhomin_only_decreases(platform):
    """rhoMin only decreases (CV moves closer to TO); retreating keeps rhoMin fixed.

    Flow:
      1. cv=0.8 → rhoMin=0.04
      2. cv=0.5 → rho=0.25 > rhoMin → NO update to rhoMin (stays 0.04)
      3. cv=0.5 again → same rhoMin=0.04 → V = 0.5*100*(0.25-0.04)^2 (same as step 2)
    """
    k, TO = 100.0, 1.0
    pos_0 = [(0, 0, 0), (0.8, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos_0,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    get_energy(ctx)  # prime rhoMin = (0.8-1.0)^2 = 0.04

    pos_far = [(0, 0, 0), (0.5, 0, 0), (5, 5, 5)]
    ctx.setPositions([mm.Vec3(*p) for p in pos_far])
    E2 = get_energy(ctx)
    ctx.setPositions([mm.Vec3(*p) for p in pos_far])
    E3 = get_energy(ctx)
    assert abs(E2 - E3) < TOL_E, \
        f"rhoMin should not change when retreating; E2={E2:.6f}, E3={E3:.6f}"
    print(f"  test_abmd_rhomin_only_decreases: OK  (E={E2:.6f})")


def test_abmd_numerical_derivative(platform):
    """Finite-difference force check at a position where bias is active."""
    dx = 1e-3
    k, TO = 200.0, 1.5

    # Prime rhoMin at cv=1.4 (close to TO=1.5)
    pos_prime = [(0, 0, 0), (1.4, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos_prime,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], [k, TO])], platform)
    get_energy(ctx)  # rhoMin = (1.4-1.5)^2 = 0.01

    # Evaluate at cv=0.8 (farther from TO)
    base_d = 0.8
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d + dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_p = get_energy(ctx)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d - dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_m = get_energy(ctx)
    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d, 0, 0), mm.Vec3(5, 5, 5)])
    F_ana = get_forces(ctx)[1][0]

    # ABMD is degree-4 in cv → finite-diff error is O(dx²·V'''), allow 2e-3
    assert abs(F_num - F_ana) < TOL_F * 20, \
        f"F_num={F_num:.6f}, F_ana={F_ana:.6f}"
    print(f"  test_abmd_numerical_derivative: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

    # Attempt to use the Reference platform for CPU tests
    try:
        plat = mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        plat = mm.Platform.getPlatformByName("Reference")

    print("Stage 5.7 ABMD tests:")
    test_abmd_first_call_no_bias(plat)
    test_abmd_at_target_no_bias(plat)
    test_abmd_approach_then_retreat(plat)
    test_abmd_force_toward_target(plat)
    test_abmd_rhomin_only_decreases(plat)
    test_abmd_numerical_derivative(plat)
    print("All ABMD tests passed.")
