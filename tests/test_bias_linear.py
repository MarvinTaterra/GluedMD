"""Stage 5.9 — BIAS_LINEAR acceptance tests.

V = sum_d k_d * cv_d
dV/dcv_d = k_d  (constant, so force = -k_d * dCV/dr)

Tests energy value, force direction/magnitude, multi-CV coupling.
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


def _make_ctx_1d(s_val, k, platform, component=0):
    """1-D linear bias: CV = x-position of atom 0, bias = k * CV."""
    n_atoms = 2
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    av = mm.vectori(); av.append(0)
    pv = mm.vectord(); pv.append(component)
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)

    cvi = mm.vectori(); cvi.append(0)
    bp  = mm.vectord(); bp.append(k)
    iv  = mm.vectori()
    f.addBias(gp.GluedForce.BIAS_LINEAR, cvi, bp, iv)

    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    positions = [[s_val, 0, 0]] + [[0.0]*3]
    ctx.setPositions(positions)
    return ctx, f


def test_linear_energy(platform):
    """V = k * s matches formula."""
    k, s = 5.0, 0.8
    ctx, _ = _make_ctx_1d(s, k, platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    expected = k * s
    assert abs(E - expected) < TOL_E, f"expected {expected:.6f}, got {E:.6f}"
    print(f"  test_linear_energy: OK  (E={E:.6f}, ref={expected:.6f})")


def test_linear_negative_k(platform):
    """Negative k: V = k*s < 0 for s > 0."""
    k, s = -3.0, 0.5
    ctx, _ = _make_ctx_1d(s, k, platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    expected = k * s
    assert abs(E - expected) < TOL_E, f"expected {expected:.6f}, got {E:.6f}"
    print(f"  test_linear_negative_k: OK  (E={E:.6f})")


def test_linear_force_x(platform):
    """Force on atom = -k (pushes in -x for positive k)."""
    k, s = 7.0, 0.5
    ctx, _ = _make_ctx_1d(s, k, platform, component=0)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    fx = state.getForces(asNumpy=False)[0][0].value_in_unit(unit)
    expected = -k   # F = -dV/dr = -k * (dCV/dr) = -k * 1
    assert abs(fx - expected) < TOL_F, f"expected Fx={expected:.4f}, got {fx:.4f}"
    print(f"  test_linear_force_x: OK  (Fx={fx:.4f})")


def test_linear_force_z(platform):
    """CV = z-position: force is in z-direction."""
    k = 4.0
    n_atoms = 2
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori(); av.append(0)
    pv = mm.vectord(); pv.append(2)   # z component
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)
    cvi = mm.vectori(); cvi.append(0)
    bp  = mm.vectord(); bp.append(k)
    iv  = mm.vectori()
    f.addBias(gp.GluedForce.BIAS_LINEAR, cvi, bp, iv)
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions([[0.0, 0.0, 0.5]] + [[0.0]*3])
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)[0]
    forces = [raw[c].value_in_unit(unit) for c in range(3)]
    assert abs(forces[0]) < TOL_F, f"Fx should be ~0, got {forces[0]}"
    assert abs(forces[1]) < TOL_F, f"Fy should be ~0, got {forces[1]}"
    assert abs(forces[2] - (-k)) < TOL_F, f"Fz should be {-k}, got {forces[2]}"
    print(f"  test_linear_force_z: OK  (Fz={forces[2]:.4f})")


def test_linear_two_cvs(platform):
    """Two-CV linear bias: V = k1*cv1 + k2*cv2."""
    n_atoms = 2
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    # CV0 = x of atom 0; CV1 = y of atom 0
    av0 = mm.vectori(); av0.append(0)
    pv0 = mm.vectord(); pv0.append(0)   # x
    av1 = mm.vectori(); av1.append(0)
    pv1 = mm.vectord(); pv1.append(1)   # y
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av0, pv0)
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av1, pv1)

    k1, k2 = 3.0, -2.0
    cvi = mm.vectori(); cvi.append(0); cvi.append(1)
    bp  = mm.vectord(); bp.append(k1); bp.append(k2)
    iv  = mm.vectori()
    f.addBias(gp.GluedForce.BIAS_LINEAR, cvi, bp, iv)

    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    x, y = 0.6, 0.4
    ctx.setPositions([[x, y, 0.0]] + [[0.0]*3])

    state = ctx.getState(getEnergy=True, getForces=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)[0]
    forces = [raw[c].value_in_unit(unit) for c in range(3)]

    assert abs(E - (k1*x + k2*y)) < TOL_E, "E mismatch"
    assert abs(forces[0] - (-k1)) < TOL_F, "Fx mismatch"
    assert abs(forces[1] - (-k2)) < TOL_F, "Fy mismatch"
    print(f"  test_linear_two_cvs: OK  (E={E:.4f}, Fx={forces[0]:.4f}, Fy={forces[1]:.4f})")


def test_linear_zero_energy_at_origin(platform):
    """V = k*0 = 0 when CV = 0."""
    ctx, _ = _make_ctx_1d(0.0, 10.0, platform)
    state = ctx.getState(getEnergy=True)
    E = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)
    assert abs(E) < TOL_E, f"expected E=0, got {E}"
    print("  test_linear_zero_energy_at_origin: OK")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping linear bias tests.")
        sys.exit(0)
    print("Stage 5.9 — BIAS_LINEAR tests (CUDA platform):")
    test_linear_energy(plat)
    test_linear_negative_k(plat)
    test_linear_force_x(plat)
    test_linear_force_z(plat)
    test_linear_two_cvs(plat)
    test_linear_zero_energy_at_origin(plat)
    print("All linear bias tests passed.")
