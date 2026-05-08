"""Stage 5.12 acceptance tests — BIAS_EDS (White-Voth AdaGrad).

Physics (PLUMED JCTC 2014 / White-Voth):
  Linear adaptive bias  V = -λ * CV
  dV/dCV = -λ   (bias gradient; force on atom = +λ * dCV/dr)

  λ updated every PACE steps via AdaGrad:
    - Welford online mean + SSD accumulated every step
    - Every PACE steps:
        step_size = 2*(mean - target) * ssd / (N-1) / kbt / scale
        accum    += step_size^2
        λ        -= step_size * max_range / sqrt(accum)
      then statistics reset to zero.

  On the FIRST update (accum=0 initially):
    sqrt(accum) = |step_size|  →  λ changes by ±max_range
    (maximum correction on the first update)

Parameters: [target, max_range_kJmol, kbt]  (3 doubles)
integerParameters: [pace]

NOTE: with constant CV, variance=0 → ssd=0 → step_size=0 → no λ update.
      Tests must drive the CV with a varying trajectory (setPositions each step).

TIMING: updateState(step) is called with step = getStepCount() BEFORE the
step is incremented.  So with pace=5:
  - During integ.step(1): step=0, doUpdate=(0%5=0)=1 → no-op (ssd=0, reset)
  - During integ.step(2..5): step=1..4, doUpdate=0, accumulate
  - During integ.step(6): step=5, doUpdate=(5%5=0)=1 → REAL update fires
Total: prime + 5 acc steps + 1 trigger = 7 operations to see first λ update.
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

TOL_E = 1e-3   # kJ/mol
TOL_F = 1e-2   # kJ/mol/nm

BOLTZMANN_kJ = 8.314462618e-3   # kJ/(mol·K)
T_K = 300.0
KBT = BOLTZMANN_kJ * T_K        # ~2.479 kJ/mol
MAX_RANGE = 25.0 * KBT           # ~62.0 kJ/mol (PLUMED RANGE=25 default)
PACE = 5


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def make_system(x0, target, max_range=MAX_RANGE, kbt=KBT, pace=PACE,
                atom_mass=1e6, platform=None):
    """1-atom system with CV_POSITION (x-component) and White-Voth EDS bias.

    bParams = [target, max_range, kbt]
    bIntParams = [pace]
    """
    sys_ = mm.System()
    sys_.addParticle(atom_mass)

    f = gp.GluedForce()
    av = mm.vectori(); av.append(0)
    pv = mm.vectord(); pv.append(0.0)   # component=0 (x)
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)

    civ = mm.vectori(); civ.append(0)
    bpv = mm.vectord()
    bpv.append(target)
    bpv.append(max_range)
    bpv.append(kbt)
    iv = mm.vectori(); iv.append(pace)
    f.addBias(gp.GluedForce.BIAS_EDS, civ, bpv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.0001)
    ctx = mm.Context(sys_, integ, platform) if platform else mm.Context(sys_, integ)
    ctx.setPositions([mm.Vec3(x0, 0, 0)])
    ctx.setVelocities([mm.Vec3(0, 0, 0)])
    return ctx, f


def get_energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        mm.unit.kilojoules_per_mole)


def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def _run_one_period(ctx, cv_acc, trigger_cv=None):
    """Prime, accumulate PACE samples from cv_acc, fire one update.

    Sequence:
      prime at cv_acc[0]              (sets cvValuesReady, step=0 no-op update)
      setPositions(cv_acc[i])+step(1) for each i in range(PACE)  [accumulate]
      setPositions(trigger_cv)+step(1)                            [fires update]

    After this call, λ has been updated once with the statistics from cv_acc.
    """
    if trigger_cv is None:
        trigger_cv = cv_acc[-1]
    ctx.setPositions([mm.Vec3(cv_acc[0], 0, 0)])
    get_energy(ctx)   # prime cvValuesReady_
    for d in cv_acc:
        ctx.setPositions([mm.Vec3(d, 0, 0)])
        ctx.getIntegrator().step(1)
    ctx.setPositions([mm.Vec3(trigger_cv, 0, 0)])
    ctx.getIntegrator().step(1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_lambda_initially(platform):
    """Before any steps, λ=0 → force=0, energy=0."""
    ctx, _ = make_system(x0=1.0, target=2.0, platform=platform)
    E = get_energy(ctx)
    forces = get_forces(ctx)
    assert abs(E) < TOL_E, f"Expected E=0 initially (λ=0), got E={E:.6f}"
    assert abs(forces[0][0]) < TOL_F, \
        f"Expected Fx=0 initially, got Fx={forces[0][0]:.6f}"
    print(f"  test_zero_lambda_initially: OK  (E={E:.2e}, Fx={forces[0][0]:.2e})")


def test_eds_force_direction_below_target(platform):
    """When CV < target, λ grows positive → force is positive (pushes toward target).

    The first AdaGrad update always applies ±max_range to λ.
    With CV below target: step_size < 0 → λ increases by max_range > 0.
    Force = +λ > 0.
    """
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]   # all below target, has variance
    trigger_cv = 0.4

    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx, cv_acc, trigger_cv)

    forces = get_forces(ctx)
    assert forces[0][0] > 0.0, \
        f"Expected Fx > 0 (λ>0, CV<target), got Fx={forces[0][0]:.6f}"
    print(f"  test_eds_force_direction_below_target: OK  (Fx={forces[0][0]:.6f} > 0)")


def test_eds_force_direction_above_target(platform):
    """When CV > target, λ grows negative → force is negative (pushes toward target)."""
    target = 0.5
    cv_acc = [1.7, 2.0, 2.3, 2.0, 1.7]   # all above target
    trigger_cv = 2.0

    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx, cv_acc, trigger_cv)

    forces = get_forces(ctx)
    assert forces[0][0] < 0.0, \
        f"Expected Fx < 0 (λ<0, CV>target), got Fx={forces[0][0]:.6f}"
    print(f"  test_eds_force_direction_above_target: OK  (Fx={forces[0][0]:.6f} < 0)")


def test_eds_energy_sign(platform):
    """Energy = -λ * CV.
    With λ > 0 and CV > 0: energy < 0."""
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]
    trigger_cv = 0.4

    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx, cv_acc, trigger_cv)

    E = get_energy(ctx)
    assert E < 0.0, f"Expected E < 0 (V=-λ*CV<0 when λ>0, CV>0), got E={E:.6f}"
    print(f"  test_eds_energy_sign: OK  (E={E:.6f} < 0)")


def test_eds_lambda_accumulates(platform):
    """|λ| grows monotonically over multiple update periods for fixed-direction offset."""
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]   # below target → λ positive and growing

    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)

    # Period 1: prime + acc + trigger
    _run_one_period(ctx, cv_acc, trigger_cv=0.4)
    F1 = get_forces(ctx)[0][0]

    # Period 2: additional period (no re-prime needed; keep stepping)
    for d in cv_acc:
        ctx.setPositions([mm.Vec3(d, 0, 0)])
        ctx.getIntegrator().step(1)
    ctx.setPositions([mm.Vec3(0.4, 0, 0)])
    ctx.getIntegrator().step(1)
    F2 = get_forces(ctx)[0][0]

    # Period 3
    for d in cv_acc:
        ctx.setPositions([mm.Vec3(d, 0, 0)])
        ctx.getIntegrator().step(1)
    ctx.setPositions([mm.Vec3(0.4, 0, 0)])
    ctx.getIntegrator().step(1)
    F3 = get_forces(ctx)[0][0]

    assert F2 > F1 > 0.0, \
        f"Expected F1<F2 both positive: F1={F1:.4f}, F2={F2:.4f}"
    assert F3 > F2, \
        f"Expected F2<F3: F2={F2:.4f}, F3={F3:.4f}"
    print(f"  test_eds_lambda_accumulates: OK  (F1={F1:.4f}, F2={F2:.4f}, F3={F3:.4f})")


def test_eds_pace_gating(platform):
    """λ only updates when step%pace==0.

    After fewer than PACE accumulation steps: λ=0 (no real update yet).
    After PACE acc steps + 1 trigger: λ=±max_range (first update fires).
    """
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]

    # Check: after only 4 acc steps (< pace=5), no real update yet → λ=0
    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    ctx.setPositions([mm.Vec3(cv_acc[0], 0, 0)])
    get_energy(ctx)   # prime
    for d in cv_acc[:4]:   # only 4 steps, not yet PACE
        ctx.setPositions([mm.Vec3(d, 0, 0)])
        ctx.getIntegrator().step(1)
    F_early = get_forces(ctx)[0][0]

    # Check: after full PACE + trigger → λ=max_range
    ctx2, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx2, cv_acc, trigger_cv=0.4)
    F_full = get_forces(ctx2)[0][0]

    assert abs(F_early) < TOL_F, \
        f"Expected F=0 after only 4 acc steps (no update), got F={F_early:.4f}"
    assert F_full > TOL_F, \
        f"Expected F>0 after full period, got F={F_full:.4f}"
    print(f"  test_eds_pace_gating: OK  (F_early={F_early:.4f}, F_full={F_full:.4f})")


def test_eds_energy_equals_neg_lambda_times_cv(platform):
    """Verify E = -λ * CV numerically.

    After one period with cv_acc below target:
      first update → λ = +max_range  (first AdaGrad update always ±max_range)
      E = -max_range * trigger_cv
    """
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]
    trigger_cv = 0.4

    ctx, _ = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx, cv_acc, trigger_cv)

    E = get_energy(ctx)
    # After first update: λ = max_range, E = -max_range * trigger_cv
    E_expected = -MAX_RANGE * trigger_cv
    assert abs(E - E_expected) < TOL_E * 10, \
        f"E={E:.4f}, E_expected={E_expected:.4f} (max_range={MAX_RANGE:.4f})"
    print(f"  test_eds_energy_equals_neg_lambda_times_cv: OK  "
          f"(E={E:.4f}, E_expected={E_expected:.4f})")


def test_eds_serialization(platform):
    """getBiasState / setBiasState round-trip preserves λ and accum."""
    target = 1.5
    cv_acc = [0.3, 0.4, 0.5, 0.4, 0.3]
    trigger_cv = 0.4

    ctx, force = make_system(x0=cv_acc[0], target=target, platform=platform)
    _run_one_period(ctx, cv_acc, trigger_cv)

    ctx.setPositions([mm.Vec3(trigger_cv, 0, 0)])
    F_before = get_forces(ctx)[0][0]

    state_bytes = force.getBiasState()

    ctx2, force2 = make_system(x0=cv_acc[0], target=target, platform=platform)
    get_energy(ctx2)
    force2.setBiasState(state_bytes)

    ctx2.setPositions([mm.Vec3(trigger_cv, 0, 0)])
    F_after = get_forces(ctx2)[0][0]

    assert abs(F_before - F_after) < TOL_F * 5, \
        f"Serialization mismatch: F_before={F_before:.6f}, F_after={F_after:.6f}"
    print(f"  test_eds_serialization: OK  (F_before={F_before:.6f}, F_after={F_after:.6f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping BIAS_EDS tests.")
        import sys; sys.exit(0)

    print("Stage 5.12 BIAS_EDS tests (CUDA platform):")
    test_zero_lambda_initially(plat)
    test_eds_force_direction_below_target(plat)
    test_eds_force_direction_above_target(plat)
    test_eds_energy_sign(plat)
    test_eds_lambda_accumulates(plat)
    test_eds_pace_gating(plat)
    test_eds_energy_equals_neg_lambda_times_cv(plat)
    test_eds_serialization(plat)
    print("All BIAS_EDS tests passed.")
