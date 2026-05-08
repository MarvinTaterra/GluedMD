"""Stage 5.2 acceptance tests — moving restraint (time-dependent harmonic) bias.

Schedule format passed to addBias:
  parameters = [step_0, k_0_cv0, at_0_cv0, ..., step_1, k_1_cv0, at_1_cv0, ...]
  integerParameters = [M]   (number of schedule entries)

Step tracking: updateState(context, step) sets lastKnownStep_ before returning
(even when it returns early due to !cvValuesReady_).  execute() reads
lastKnownStep_ and passes it to the GPU kernel.  This means:

  * Before any integ.step(): lastKnownStep_=0 → kernel sees step=0.
  * After integ.step(N): lastKnownStep_=N → subsequent getState() sees step=N.

The recommended test pattern for a specific step N (N small):
  integ.step(N)          # sets lastKnownStep_=N; atoms move slightly
  ctx.setPositions(...)  # reset to desired test positions
  get_energy(ctx)        # evaluates at test positions with step=N
"""
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
    """cv_specs: [(cv_type, atoms, params)]
    bias_specs: [(cv_list, schedule_params, [M])]"""
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

    for cv_list, sched_params, int_params in bias_specs:
        civ = mm.vectori()
        for c in cv_list:
            civ.append(c)
        pv = mm.vectord()
        for p in sched_params:
            pv.append(p)
        iv = mm.vectori()
        for i in int_params:
            iv.append(i)
        f.addBias(gp.GluedForce.BIAS_MOVING_RESTRAINT, civ, pv, iv)

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

def test_moving_restraint_single_anchor(platform):
    """Single-anchor schedule at step=0: V = 0.5*k*(cv-at)^2.
    k=100, at=0.5, cv=1.0 → V=12.5 kJ/mol (same as harmonic bias)."""
    k, at, d = 100.0, 0.5, 1.0
    sched = [0.0, k, at]           # [step_0, k_0, at_0]
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)
    E = get_energy(ctx)
    expected = 0.5 * k * (d - at)**2
    assert abs(E - expected) < TOL_E, f"E={E:.8f}, expected {expected:.8f}"
    print(f"  test_moving_restraint_single_anchor: OK  (E={E:.6f})")


def test_moving_restraint_at_minimum(platform):
    """cv exactly at at → E=0."""
    k, at = 200.0, 1.0
    sched = [0.0, k, at]
    pos = [(0, 0, 0), (at, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.2e}, expected 0"
    print(f"  test_moving_restraint_at_minimum: OK  (E={E:.2e})")


def test_moving_restraint_force_direction(platform):
    """F on atom1 must point toward at when cv > at.
    k=100, at=0.5, cv=1.0 → F1.x = -k*(cv-at) = -50 kJ/mol/nm."""
    k, at, d = 100.0, 0.5, 1.0
    sched = [0.0, k, at]
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)
    forces = get_forces(ctx)
    expected_fx1 = -k * (d - at)
    assert abs(forces[1][0] - expected_fx1) < TOL_F, \
        f"F1.x={forces[1][0]:.6f}, expected {expected_fx1:.6f}"
    assert abs(forces[0][0] + forces[1][0]) < TOL_F, "Newton 3rd law violated"
    print(f"  test_moving_restraint_force_direction: OK  (F1.x={forces[1][0]:.4f})")


def test_moving_restraint_interpolation_midpoint(platform):
    """Two anchors at steps 0 and 2; evaluate at step 1 (midpoint alpha=0.5).
    Anchor 0: k=0,   at=2.0
    Anchor 1: k=100, at=1.0
    At step 1: k=50, at=1.5.  cv=2.0 → V=0.5*50*(2.0-1.5)^2=6.25 kJ/mol."""
    sched = [0.0, 0.0, 2.0,    # [step_0, k_0, at_0]
             2.0, 100.0, 1.0]  # [step_1, k_1, at_1]
    d = 2.0
    k_mid, at_mid = 50.0, 1.5
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [2])], platform)
    # Advance to step 1 (updateState sets lastKnownStep_=1 before execute)
    ctx.getIntegrator().step(1)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d, 0, 0), mm.Vec3(5, 5, 5)])
    E = get_energy(ctx)
    expected = 0.5 * k_mid * (d - at_mid)**2
    assert abs(E - expected) < TOL_E, f"E={E:.8f}, expected {expected:.8f}"
    print(f"  test_moving_restraint_interpolation_midpoint: OK  (E={E:.6f})")


def test_moving_restraint_clamp_after_end(platform):
    """Step beyond last anchor clamps to last anchor's parameters.
    Single anchor at step=0: should still give same result at step=5."""
    k, at, d = 100.0, 0.5, 1.0
    sched = [0.0, k, at]
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)
    ctx.getIntegrator().step(5)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d, 0, 0), mm.Vec3(5, 5, 5)])
    E = get_energy(ctx)
    expected = 0.5 * k * (d - at)**2
    assert abs(E - expected) < TOL_E, f"E={E:.8f}, expected {expected:.8f}"
    print(f"  test_moving_restraint_clamp_after_end: OK  (E={E:.6f})")


def test_moving_restraint_clamp_before_start(platform):
    """Step before first anchor (step=0 with first anchor at step=5) clamps to anchor 0.
    Here lastKnownStep_=0 so it's already before schedule[0]=5."""
    k, at, d = 100.0, 0.5, 1.0
    sched = [5.0, k, at]          # anchor starts at step 5
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)
    # lastKnownStep_=0 < 5 → clamp to anchor 0 (the only anchor)
    E = get_energy(ctx)
    expected = 0.5 * k * (d - at)**2
    assert abs(E - expected) < TOL_E, f"E={E:.8f}, expected {expected:.8f}"
    print(f"  test_moving_restraint_clamp_before_start: OK  (E={E:.6f})")


def test_moving_restraint_numerical_derivative(platform):
    """Finite-difference check on force for single anchor.
    Uses dx=1e-2 to stay within float32 quantisation limits."""
    dx = 1e-2
    k, at = 200.0, 1.0
    base_d = 1.5
    sched = [0.0, k, at]
    base_pos = [(0, 0, 0), (base_d, 0, 0), (5, 5, 5)]
    ctx, _ = make_system(base_pos,
                         [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                         [([0], sched, [1])], platform)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d + dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_p = get_energy(ctx)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d - dx, 0, 0), mm.Vec3(5, 5, 5)])
    E_m = get_energy(ctx)
    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(base_d, 0, 0), mm.Vec3(5, 5, 5)])
    F_ana = get_forces(ctx)[1][0]

    assert abs(F_num - F_ana) < TOL_F * 10, \
        f"F_num={F_num:.6f}, F_ana={F_ana:.6f}"
    print(f"  test_moving_restraint_numerical_derivative: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping moving restraint tests.")
        sys.exit(0)

    print("Stage 5.2 moving restraint tests (CUDA platform):")
    test_moving_restraint_single_anchor(plat)
    test_moving_restraint_at_minimum(plat)
    test_moving_restraint_force_direction(plat)
    test_moving_restraint_interpolation_midpoint(plat)
    test_moving_restraint_clamp_after_end(plat)
    test_moving_restraint_clamp_before_start(plat)
    test_moving_restraint_numerical_derivative(plat)
    print("All moving restraint tests passed.")
