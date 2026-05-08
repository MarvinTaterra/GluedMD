"""Stage 3.10 — CV_POSITION acceptance tests.

Tests single-atom Cartesian position CVs (x, y, z components) against
known positions, and verifies the chain-rule force output numerically.
"""

import sys, math
import openmm as mm
import gluedplugin as gp

TOL = 1e-5     # nm
FTOL = 1e-4    # kJ/mol/nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def eval_cv(ctx, f):
    ctx.getState(getForces=True)
    return list(f.getLastCVValues(ctx))


def make_ctx(positions, atom, component, bias_grad=1.0, platform=None):
    n = len(positions)
    sys = mm.System()
    for _ in range(n): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori(); av.append(atom)
    pv = mm.vectord(); pv.append(float(component))
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)
    grads = mm.vectord(); grads.append(bias_grad)
    f.setTestBiasGradients(grads)
    sys.addForce(f)
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys, integrator, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions])
    return ctx, f


def test_position_x(platform):
    """CV_POSITION component=0 returns the x coordinate of the atom."""
    positions = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1.7, 0.8, 0.9], [0.0, 0.0, 0.0]]
    ctx, f = make_ctx(positions, atom=2, component=0, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 1.7) < TOL, f"expected x=1.7, got {cvs[0]}"
    print(f"  test_position_x: OK  (cv={cvs[0]:.6f})")


def test_position_y(platform):
    """CV_POSITION component=1 returns the y coordinate."""
    positions = [[0.1, 0.5, 0.3]] + [[0.0]*3]*3
    ctx, f = make_ctx(positions, atom=0, component=1, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.5) < TOL, f"expected y=0.5, got {cvs[0]}"
    print(f"  test_position_y: OK  (cv={cvs[0]:.6f})")


def test_position_z(platform):
    """CV_POSITION component=2 returns the z coordinate."""
    positions = [[0.1, 0.2, 0.77]] + [[0.0]*3]*3
    ctx, f = make_ctx(positions, atom=0, component=2, platform=platform)
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.77) < TOL, f"expected z=0.77, got {cvs[0]}"
    print(f"  test_position_z: OK  (cv={cvs[0]:.6f})")


def test_position_force_gradient(platform):
    """Chain-rule force for CV_POSITION x: F_x on the atom = -dU/dCV."""
    positions = [[0.5, 0.0, 0.0]] + [[0.0]*3]*3
    dU_dCV = 5.0
    ctx, f = make_ctx(positions, atom=0, component=0, bias_grad=dU_dCV, platform=platform)
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    fx = raw[0][0].value_in_unit(unit)
    expected = -dU_dCV
    assert abs(fx - expected) < FTOL, f"expected Fx={expected}, got {fx}"
    print(f"  test_position_force_gradient: OK  (Fx={fx:.6f})")


def test_multiple_position_cvs(platform):
    """Two position CVs on different atoms and components are independent."""
    n = 4
    sys = mm.System()
    for _ in range(n): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    av0 = mm.vectori(); av0.append(0)
    pv0 = mm.vectord(); pv0.append(0)    # atom 0, x
    av1 = mm.vectori(); av1.append(2)
    pv1 = mm.vectord(); pv1.append(2)    # atom 2, z
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av0, pv0)
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av1, pv1)
    grads = mm.vectord(); grads.append(0.0); grads.append(0.0)
    f.setTestBiasGradients(grads)

    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    positions = [[0.3, 0.1, 0.2], [0.0]*3, [0.5, 0.6, 0.9], [0.0]*3]
    ctx.setPositions([mm.Vec3(*p) for p in positions])
    cvs = eval_cv(ctx, f)
    assert abs(cvs[0] - 0.3) < TOL, f"cv0 expected 0.3 got {cvs[0]}"
    assert abs(cvs[1] - 0.9) < TOL, f"cv1 expected 0.9 got {cvs[1]}"
    print(f"  test_multiple_position_cvs: OK  (cv0={cvs[0]:.4f}, cv1={cvs[1]:.4f})")


def test_position_force_zero_on_other_atoms(platform):
    """Force from position CV is zero on all atoms except the target."""
    positions = [[0.5, 0.1, 0.2]] + [[float(i)*0.1]*3 for i in range(1, 4)]
    dU_dCV = 3.0
    ctx, f = make_ctx(positions, atom=0, component=0, bias_grad=dU_dCV, platform=platform)
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in range(1, len(positions)):
        fvec = [raw[ai][c].value_in_unit(unit) for c in range(3)]
        mag = math.sqrt(sum(v**2 for v in fvec))
        assert mag < FTOL, f"expected zero force on atom {ai}, got {fvec}"
    print("  test_position_force_zero_on_other_atoms: OK")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_POSITION tests.")
        sys.exit(0)
    print("Stage 3.10 — CV_POSITION tests (CUDA platform):")
    test_position_x(plat)
    test_position_y(plat)
    test_position_z(plat)
    test_position_force_gradient(plat)
    test_multiple_position_cvs(plat)
    test_position_force_zero_on_other_atoms(plat)
    print("All CV_POSITION tests passed.")
