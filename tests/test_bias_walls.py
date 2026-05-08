"""Stage 5.8 — BIAS_UPPER_WALL / BIAS_LOWER_WALL acceptance tests.

V = kappa * max(0, delta)^n * exp(eps * delta)
  where delta = s - at  (UPPER_WALL)  or  at - s  (LOWER_WALL)

Tests: energy value, force direction and magnitude, zero outside wall.
"""

import sys, math
import openmm as mm
import gluedplugin as gp

TOL_E = 1e-5   # kJ/mol
TOL_F = 1e-4   # kJ/mol/nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _wall_energy(s, at, kappa, eps, n, wall_type):
    """Reference wall energy."""
    delta = (s - at) if wall_type == 0 else (at - s)
    if delta <= 0.0:
        return 0.0
    return kappa * delta**n * math.exp(eps * delta)


def _wall_grad(s, at, kappa, eps, n, wall_type):
    """dV/ds for the wall bias."""
    delta = (s - at) if wall_type == 0 else (at - s)
    if delta <= 0.0:
        return 0.0
    dpow = n * delta**(n - 1.0)
    powered = delta**n
    dV_ddelta = kappa * (dpow + eps * powered) * math.exp(eps * delta)
    sign = 1.0 if wall_type == 0 else -1.0
    return sign * dV_ddelta


def _make_ctx(positions, cv_atoms, at, kappa, eps, n, wall_type, platform):
    """Build a Context with one position CV and one wall bias."""
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    # Single x-position CV
    av = mm.vectori(); av.append(cv_atoms[0])
    pv = mm.vectord(); pv.append(0)   # x component
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)

    # Wall bias: params = [at, kappa, eps, n]
    cvi = mm.vectori(); cvi.append(0)
    bp = mm.vectord(); bp.append(at); bp.append(kappa); bp.append(eps); bp.append(n)
    iv = mm.vectori()
    btype = gp.GluedForce.BIAS_UPPER_WALL if wall_type == 0 else gp.GluedForce.BIAS_LOWER_WALL
    f.addBias(btype, cvi, bp, iv)

    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    return ctx, f


def test_upper_wall_outside(platform):
    """UPPER_WALL: no energy when s < at."""
    positions = [[0.2, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=0.5, kappa=100.0, eps=0.0, n=2, wall_type=0, platform=platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    assert abs(E) < TOL_E, f"expected E=0 outside wall, got {E}"
    print(f"  test_upper_wall_outside: OK  (E={E:.2e})")


def test_upper_wall_inside(platform):
    """UPPER_WALL: energy matches formula when s > at."""
    at, kappa, eps, n = 0.5, 100.0, 0.0, 2
    s = 0.7   # delta = 0.2
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=0, platform=platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    expected = _wall_energy(s, at, kappa, eps, n, wall_type=0)
    assert abs(E - expected) < TOL_E, f"expected {expected:.6f}, got {E:.6f}"
    print(f"  test_upper_wall_inside: OK  (E={E:.6f}, ref={expected:.6f})")


def test_upper_wall_force(platform):
    """UPPER_WALL: force = -dV/ds acts to push atom back below at."""
    at, kappa, eps, n = 0.5, 100.0, 0.0, 2
    s = 0.6   # delta = 0.1
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=0, platform=platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    fx = state.getForces(asNumpy=False)[0][0].value_in_unit(unit)
    dV_ds = _wall_grad(s, at, kappa, eps, n, wall_type=0)
    expected_fx = -dV_ds
    assert abs(fx - expected_fx) < TOL_F, f"expected Fx={expected_fx:.4f}, got {fx:.4f}"
    assert fx < 0, "UPPER_WALL force should be negative (push atom left)"
    print(f"  test_upper_wall_force: OK  (Fx={fx:.4f}, ref={expected_fx:.4f})")


def test_lower_wall_outside(platform):
    """LOWER_WALL: no energy when s > at."""
    at, kappa, eps, n = 0.3, 100.0, 0.0, 2
    s = 0.5
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=1, platform=platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    assert abs(E) < TOL_E, f"expected E=0 outside wall, got {E}"
    print(f"  test_lower_wall_outside: OK  (E={E:.2e})")


def test_lower_wall_inside(platform):
    """LOWER_WALL: energy matches formula when s < at."""
    at, kappa, eps, n = 0.5, 80.0, 0.0, 2
    s = 0.3   # delta = 0.2
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=1, platform=platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    expected = _wall_energy(s, at, kappa, eps, n, wall_type=1)
    assert abs(E - expected) < TOL_E, f"expected {expected:.6f}, got {E:.6f}"
    print(f"  test_lower_wall_inside: OK  (E={E:.6f}, ref={expected:.6f})")


def test_lower_wall_force(platform):
    """LOWER_WALL: force = -dV/ds acts to push atom above at."""
    at, kappa, eps, n = 0.5, 100.0, 0.0, 2
    s = 0.4   # delta = 0.1
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=1, platform=platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    fx = state.getForces(asNumpy=False)[0][0].value_in_unit(unit)
    dV_ds = _wall_grad(s, at, kappa, eps, n, wall_type=1)
    expected_fx = -dV_ds
    assert abs(fx - expected_fx) < TOL_F, f"expected Fx={expected_fx:.4f}, got {fx:.4f}"
    assert fx > 0, "LOWER_WALL force should be positive (push atom right)"
    print(f"  test_lower_wall_force: OK  (Fx={fx:.4f}, ref={expected_fx:.4f})")


def test_wall_with_exp(platform):
    """Wall with nonzero eps: V = kappa * delta^n * exp(eps*delta)."""
    at, kappa, eps, n = 0.5, 50.0, 2.0, 2
    s = 0.8
    positions = [[s, 0, 0]] + [[0.0]*3]*3
    ctx, f = _make_ctx(positions, [0], at=at, kappa=kappa, eps=eps, n=n, wall_type=0, platform=platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    expected = _wall_energy(s, at, kappa, eps, n, wall_type=0)
    assert abs(E - expected) < TOL_E, f"expected {expected:.4f}, got {E:.4f}"
    print(f"  test_wall_with_exp: OK  (E={E:.4f}, ref={expected:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping wall bias tests.")
        sys.exit(0)
    print("Stage 5.8 — BIAS_UPPER_WALL / BIAS_LOWER_WALL tests (CUDA platform):")
    test_upper_wall_outside(plat)
    test_upper_wall_inside(plat)
    test_upper_wall_force(plat)
    test_lower_wall_outside(plat)
    test_lower_wall_inside(plat)
    test_lower_wall_force(plat)
    test_wall_with_exp(plat)
    print("All wall bias tests passed.")
