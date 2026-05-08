"""Stage 5.12 acceptance tests — BIAS_MAXENT (Lagrange-multiplier linear bias).

Algorithm (Cesari et al. JCTC 2016):
  V   = kT * Σᵢ leff_i * cv_i
  dV/dcv_i = kT * leff_i
  leff_i = convert_lambda(type, lambda_i)
    EQUAL:       leff = lambda
    INEQUAL_GT:  leff = min(lambda, 0)   (penalises only when lambda < 0)
    INEQUAL_LT:  leff = max(lambda, 0)

  Update (every PACE steps, step t=step/pace):
    xi_i = 0                                           (sigma=0)
    xi_i = -lambda_i * sigma^2                         (GAUSSIAN)
    xi_i = -lambda_i*sigma^2 / (1 - lambda_i^2*sigma^2/(alpha+1))  (LAPLACE)
    lambda_i += [kappa_i / (1 + t/tau_i)] * (cv_i + xi_i - at_i)

Parameter encoding for addBias(BIAS_MAXENT, cvList, params, intParams):
  params    = [kbt, sigma, alpha,  at_0, kappa_0, tau_0,  at_1, kappa_1, tau_1, ...]
  intParams = [pace, type, errorType]
"""
import sys, math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
KBT  = 2.479       # kJ/mol at ~300 K
TOL_E = 1e-4
TOL_F = 2e-3


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def _vd(*args):
    v = mm.vectord()
    for x in args: v.append(float(x))
    return v


def _vi(*args):
    v = mm.vectori()
    for x in args: v.append(int(x))
    return v


def make_ctx(pos_nm, cv_specs, bias_specs, platform=None):
    """cv_specs: [(cv_type, [atoms], [params])]
    bias_specs: [(cv_list, params_list, int_params_list)]"""
    n = len(pos_nm)
    sys_ = mm.System()
    for _ in range(n): sys_.addParticle(1.0)

    f = gp.GluedForce()
    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms: av.append(a)
        pv = mm.vectord()
        for p in params: pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    for cv_list, params, int_params in bias_specs:
        civ = _vi(*cv_list)
        pv  = _vd(*params)
        iv  = _vi(*int_params)
        f.addBias(gp.GluedForce.BIAS_MAXENT, civ, pv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(1e-6)   # tiny dt → negligible position change
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in pos_nm])
    return ctx, f


def get_energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        mm.unit.kilojoules_per_mole)


def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def reset_pos(ctx, pos_nm):
    ctx.setPositions([mm.Vec3(*p) for p in pos_nm])


# ---------------------------------------------------------------------------
# 1. Zero energy at lambda=0 (before any update)
# ---------------------------------------------------------------------------
def test_maxent_zero_energy(platform):
    """Energy must be zero when lambda=0 (initial state)."""
    pos = [(0, 0, 0), (1.0, 0, 0), (5, 5, 5)]
    # EQUAL, sigma=0, pace=1; at=0.5, kappa=0.1, tau=1e9
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, 0.5, 0.1, 1e9], [1, 0, 0])],
                      platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.6f}, expected 0"
    print(f"  test_maxent_zero_energy: OK  (E={E:.2e})")


# ---------------------------------------------------------------------------
# 2. Lambda update: EQUAL, no noise
#    Positions: d = 1.0 nm, target at = 0.5, kappa = 0.1, tau = 1e9 (no decay)
#    After prime (getEnergy) + 1 step:
#      update fires at step=1 with uc=1, lr = kappa/(1 + 1/tau) ≈ kappa = 0.1
#      lambda = 0.1 * (1.0 - 0.5) = 0.05
#    Then (reset pos, getEnergy):
#      E = kbt * 0.05 * 1.0 = 0.12395 kJ/mol
# ---------------------------------------------------------------------------
def test_maxent_lambda_update_equal(platform):
    """Lambda and energy match analytical prediction after one update."""
    d = 1.0
    at, kappa, tau = 0.5, 0.1, 1e9
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, kappa, tau], [pace, 0, 0])],
                      platform)
    get_energy(ctx)                         # prime cvValuesReady
    ctx.getIntegrator().step(1)            # step=1: lambda updated
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lr = kappa / (1.0 + 1.0 / tau)
    lam_expected = lr * (d - at)
    E_expected = KBT * lam_expected * d
    assert abs(E - E_expected) < TOL_E, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_lambda_update_equal: OK  "
          f"(E={E:.6f}, expected {E_expected:.6f})")


# ---------------------------------------------------------------------------
# 3. Force direction (EQUAL, no noise)
#    lambda > 0 (CV > target) → cvBiasGradients = -kbt*lambda
#    Force on atom1 x = cvBiasGradients * jac = (-kbt*lambda) * (+1) = -0.12395
#    Force on atom0 x = (-kbt*lambda) * (-1) = +0.12395   (Newton pair)
# ---------------------------------------------------------------------------
def test_maxent_force_direction(platform):
    """Forces are analytical and Newton's-3rd holds."""
    d = 1.0
    at, kappa, tau = 0.5, 0.1, 1e9
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, kappa, tau], [pace, 0, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)

    lr = kappa / (1.0 + 1.0 / tau)
    lam = lr * (d - at)    # +0.05
    # scatter: F = -(dV/dCV * jac) = -(kbt*lam * +1) = -kbt*lam (negative: pushes toward target)
    F_ana = -KBT * lam

    forces = get_forces(ctx)
    assert abs(forces[1][0] - F_ana) < TOL_F, \
        f"F_atom1_x={forces[1][0]:.6f}, expected {F_ana:.6f}"
    # Newton's 3rd: atom0 and atom1 sum to zero
    assert abs(forces[0][0] + forces[1][0]) < TOL_F, \
        f"Newton sum={forces[0][0]+forces[1][0]:.6f}"
    print(f"  test_maxent_force_direction: OK  "
          f"(F_atom1_x={forces[1][0]:.4f}, expected {F_ana:.4f})")


# ---------------------------------------------------------------------------
# 4. INEQUAL_GT: leff = min(lambda, 0)
#    CV = 1.0 nm > at = 0.5 nm → lambda += 0.1*(1.0-0.5) = +0.05 → leff = 0 → E = 0
# ---------------------------------------------------------------------------
def test_maxent_inequal_gt_satisfied(platform):
    """INEQUAL_GT: when CV already exceeds target, constraint is satisfied (E=0)."""
    d = 1.0
    at = 0.5
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, 0.1, 1e9], [pace, 1, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.6f}, expected 0 (constraint satisfied)"
    print(f"  test_maxent_inequal_gt_satisfied: OK  (E={E:.2e})")


# ---------------------------------------------------------------------------
# 5. INEQUAL_GT violated: CV < at → lambda < 0 → leff = lambda → E != 0
#    CV = 0.3, at = 0.8 → lambda += 0.1*(0.3-0.8) = -0.05 → leff = -0.05
#    E = kbt * (-0.05) * 0.3 = -0.037185 kJ/mol
# ---------------------------------------------------------------------------
def test_maxent_inequal_gt_violated(platform):
    """INEQUAL_GT: when CV is below target, bias is active."""
    d = 0.3
    at = 0.8
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, 0.1, 1e9], [pace, 1, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lam = 0.1 * (d - at)   # -0.05
    E_expected = KBT * lam * d  # negative
    assert abs(E - E_expected) < TOL_E, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_inequal_gt_violated: OK  "
          f"(E={E:.6f}, expected {E_expected:.6f})")


# ---------------------------------------------------------------------------
# 6. INEQUAL_LT: leff = max(lambda, 0)
#    CV = 0.3 nm < at = 0.8 nm → lambda += 0.1*(0.3-0.8) = -0.05 → leff = 0 → E = 0
# ---------------------------------------------------------------------------
def test_maxent_inequal_lt_satisfied(platform):
    """INEQUAL_LT: when CV is below target, constraint is satisfied (E=0)."""
    d = 0.3
    at = 0.8
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, 0.1, 1e9], [pace, 2, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"E={E:.6f}, expected 0 (constraint satisfied)"
    print(f"  test_maxent_inequal_lt_satisfied: OK  (E={E:.2e})")


# ---------------------------------------------------------------------------
# 7. INEQUAL_LT violated: CV > at → lambda > 0 → leff = lambda → E != 0
#    CV = 1.0, at = 0.5 → lambda = 0.1*(1.0-0.5) = +0.05 → leff = 0.05
#    E = kbt * 0.05 * 1.0 = 0.12395
# ---------------------------------------------------------------------------
def test_maxent_inequal_lt_violated(platform):
    """INEQUAL_LT: when CV exceeds target, bias is active."""
    d = 1.0
    at = 0.5
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, 0.1, 1e9], [pace, 2, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lam = 0.1 * (d - at)   # +0.05
    E_expected = KBT * lam * d
    assert abs(E - E_expected) < TOL_E, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_inequal_lt_violated: OK  "
          f"(E={E:.6f}, expected {E_expected:.6f})")


# ---------------------------------------------------------------------------
# 8. Gaussian noise: xi = -lambda * sigma^2
#    CV = 1.0, at = 0.5, sigma = 1.0, kappa = 0.1, tau = 1e9
#    First update (lambda_before = 0):
#      xi = -0 * 1.0^2 = 0   (no effect when lambda=0)
#      lambda = 0.1 * (1.0 + 0 - 0.5) = 0.05   (same as no-noise case)
#    → energy same as EQUAL no-noise case
# ---------------------------------------------------------------------------
def test_maxent_gaussian_noise_first_step(platform):
    """Gaussian noise has no effect on the first step (lambda_0 = 0)."""
    d = 1.0
    at, kappa, sigma, tau = 0.5, 0.1, 1.0, 1e9
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, sigma, 1.0, at, kappa, tau], [pace, 0, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lam = kappa * (d - at)   # xi=0 since lambda was 0
    E_expected = KBT * lam * d
    assert abs(E - E_expected) < TOL_E, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_gaussian_noise_first_step: OK  "
          f"(E={E:.6f}, expected {E_expected:.6f})")


# ---------------------------------------------------------------------------
# 9. Gaussian noise second step: xi_1 = -lambda_1 * sigma^2
#    After step 1: lambda_1 = 0.05 (from above)
#    Step 2 update: xi = -0.05 * 1.0 = -0.05
#    lambda_2 = 0.05 + kappa/(1+2/tau) * (cv + xi - at)
#             = 0.05 + 0.1 * (1.0 + (-0.05) - 0.5) = 0.05 + 0.1*0.45 = 0.05 + 0.045 = 0.095
#    E = kbt * 0.095 * 1.0 = 0.23551 kJ/mol
# ---------------------------------------------------------------------------
def test_maxent_gaussian_noise_second_step(platform):
    """Gaussian noise modifies the second update (xi = -lambda * sigma^2)."""
    d = 1.0
    at, kappa, sigma, tau = 0.5, 0.1, 1.0, 1e9
    pace = 1
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, sigma, 1.0, at, kappa, tau], [pace, 0, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)   # step=1: lam → 0.05
    ctx.getIntegrator().step(1)   # step=2: lam updated again
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lam1 = kappa * (d - at)                # 0.05
    xi2  = -lam1 * sigma * sigma           # -0.05
    lr2  = kappa / (1.0 + 2.0 / tau)      # ≈ kappa
    lam2 = lam1 + lr2 * (d + xi2 - at)    # 0.05 + 0.1*(0.45) = 0.095
    E_expected = KBT * lam2 * d
    assert abs(E - E_expected) < TOL_E * 5, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_gaussian_noise_second_step: OK  "
          f"(E={E:.6f}, expected {E_expected:.6f})")


# ---------------------------------------------------------------------------
# 10. Pace: verify update fires at step=0 (first after prime) then every PACE
#     pace=3: fires at step=0, skips step=1 and step=2, fires at step=3
# ---------------------------------------------------------------------------
def test_maxent_pace(platform):
    """Lambda updates at step=0 (first prime) then every PACE steps; skips others."""
    d = 1.0
    at, kappa, tau, pace = 0.5, 0.1, 1e9, 3
    pos = [(0, 0, 0), (d, 0, 0), (5, 5, 5)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], [])],
                      [([0], [KBT, 0.0, 1.0, at, kappa, tau], [pace, 0, 0])],
                      platform)

    get_energy(ctx)                    # prime cvValuesReady
    # step=0: 0%3==0 → fires (lam: 0→0.05)
    # step=1: skip; step=2: skip
    ctx.getIntegrator().step(2)       # drives steps 0 and 1
    reset_pos(ctx, pos)
    E_after_0 = get_energy(ctx)       # lambda = 0.05 after step=0 update
    lam1 = kappa * (d - at)           # 0.05
    assert abs(E_after_0 - KBT * lam1 * d) < TOL_E, \
        f"E_after_0={E_after_0:.6f}, expected {KBT*lam1*d:.6f}"

    ctx.getIntegrator().step(1)       # step=2: 2%3≠0 → no update
    reset_pos(ctx, pos)
    E_no_update = get_energy(ctx)
    assert abs(E_no_update - KBT * lam1 * d) < TOL_E, \
        f"E unchanged={E_no_update:.6f}, expected {KBT*lam1*d:.6f}"

    ctx.getIntegrator().step(1)       # step=3: 3%3==0 → fires (lam: 0.05→0.1)
    reset_pos(ctx, pos)
    E_after_3 = get_energy(ctx)
    # uc=3/3=1, lr=kappa/(1+1/tau)≈kappa; lambda += kappa*(cv-at)
    lam2 = lam1 + kappa * (d - at)    # 0.1
    assert abs(E_after_3 - KBT * lam2 * d) < TOL_E, \
        f"E_after_3={E_after_3:.6f}, expected {KBT*lam2*d:.6f}"

    print(f"  test_maxent_pace: OK  "
          f"(E@0={E_after_0:.4f}, E@2={E_no_update:.4f}, E@3={E_after_3:.4f})")


# ---------------------------------------------------------------------------
# 11. Multi-CV: two distance CVs, both targeted
# ---------------------------------------------------------------------------
def test_maxent_multi_cv(platform):
    """Two CVs with independent lambda updates."""
    d0, d1 = 1.0, 0.5
    at0, kappa0, tau0 = 0.3, 0.1, 1e9
    at1, kappa1, tau1 = 0.8, 0.2, 1e9
    pace = 1
    # CV0: dist atoms 0-1 = d0; CV1: dist atoms 0-2 = d1
    pos = [(0, 0, 0), (d0, 0, 0), (d1, 0, 0)]
    ctx, _ = make_ctx(pos,
                      [(gp.GluedForce.CV_DISTANCE, [0, 1], []),
                       (gp.GluedForce.CV_DISTANCE, [0, 2], [])],
                      [([0, 1],
                        [KBT, 0.0, 1.0,
                         at0, kappa0, tau0,
                         at1, kappa1, tau1],
                        [pace, 0, 0])],
                      platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)
    reset_pos(ctx, pos)
    E = get_energy(ctx)

    lam0 = kappa0 * (d0 - at0)
    lam1 = kappa1 * (d1 - at1)
    E_expected = KBT * (lam0 * d0 + lam1 * d1)
    assert abs(E - E_expected) < TOL_E, \
        f"E={E:.6f}, expected {E_expected:.6f}"
    print(f"  test_maxent_multi_cv: OK  (E={E:.6f}, expected {E_expected:.6f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping MaxEnt bias tests.")
        sys.exit(0)

    print("Stage 5.12 MaxEnt bias tests (CUDA platform):")
    test_maxent_zero_energy(plat)
    test_maxent_lambda_update_equal(plat)
    test_maxent_force_direction(plat)
    test_maxent_inequal_gt_satisfied(plat)
    test_maxent_inequal_gt_violated(plat)
    test_maxent_inequal_lt_satisfied(plat)
    test_maxent_inequal_lt_violated(plat)
    test_maxent_gaussian_noise_first_step(plat)
    test_maxent_gaussian_noise_second_step(plat)
    test_maxent_pace(plat)
    test_maxent_multi_cv(plat)
    print("All MaxEnt bias tests passed.")
