"""Stage 5.2 acceptance tests — OPES bias (all variants).

Execution-order notes:
  • updateContextState() (→ updateState) is called from VerletIntegrator::step(),
    BEFORE calcForcesAndEnergy() within that same step.
  • context.getState() calls calcForcesAndEnergy() only — it does NOT call
    updateContextState(). So get_energy() never triggers a deposition.
  • Therefore the first deposition requires two integ.step(1) calls:
      1. integ.step(1)  → updateState(0) skipped (cvValuesReady=false);
                          execute() fills cvValues → cvValuesReady=true. step 0→1
      2. integ.step(1)  → updateState(1) fires (1 ≠ lastUpdateStep −1) → deposits.
                          execute() evaluates bias with numKernels=1. step 1→2

OPES formulas (Invernizzi & Parrinello 2020, J. Phys. Chem. Lett. 11:2093):
  evaluateKernel(s,k) = h_k * (G_k(s) − ε)  if norm2 < cutoff2, else 0
  prob_unnorm = Σ_k evaluateKernel(s,k)
  pz          = prob_unnorm * nker / sum_uprob          (nker = # stored kernels)
  V(s)        = invGF · kT · log(pz + ε)
  ε           = exp(−1/(invGF·(1−invGF)))  = exp(−γ²/(γ−1))
  invGF       = (γ−1)/γ   (= biasfactor−1)/biasfactor, γ=biasfactor)
  barrier     = kT/( 1−invGF ) = kT·γ
  before any kernels: V = −barrier = −kT·γ

  Silverman sigma (1st deposit):
    sentinels:  sum_w = exp(−γ),  sum_w2 = exp(−2γ)
    deposit 1:  sum_w += exp(V_before/kT) = exp(−γ) →  sum_w = 2·exp(−γ)
    neff        = (1+sum_w)²/(1+sum_w2)  ≈ 1  (both terms ≈ 0)
    sig_dep     = sigma0·(neff·(D+2)/4)^{−1/(D+4)}  with D=1

  Height correction: logWeight_stored = V_before/kT + log(sigma0/sig_dep)
  sum_uprob   = Σ_{j,k} h_k_stored·(G(c_j,c_k) − ε)  (pairwise, recomputed)
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_E = 1e-4
TOL_F = 2e-3

_MASS_AMU = 1000.0   # large mass → atoms barely move in one step
_DT_PS    = 0.0001


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_system(positions_nm, cv_specs, bias_specs, platform=None):
    """bias_specs: [(cvIndices, params_double, intParams_int, type)]."""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(_MASS_AMU)

    f = gp.GluedForce()
    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms: av.append(a)
        pv = mm.vectord()
        for p in params: pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    for cv_list, params, intparams, btype in bias_specs:
        civ = mm.vectori()
        for c in cv_list: civ.append(c)
        pv = mm.vectord()
        for p in params: pv.append(p)
        iv = mm.vectori()
        for i in intparams: iv.append(i)
        f.addBias(btype, civ, pv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(_DT_PS)
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


def deposit_one_kernel(ctx, s_dep):
    """Deposit exactly one OPES kernel at approximately s_dep.

    After return: 1 kernel in table, context at step=2, lastUpdateStep=1.
    Subsequent get_energy() / get_forces() calls (no integ.step) will evaluate
    using this kernel without triggering further deposition because getState()
    calls calcForcesAndEnergy() only — it does NOT call updateContextState().

    Mechanism:
      Step 0→1: updateContextState(0) skipped (cvValuesReady=false).
                execute() fills cvValues(s_dep) → cvValuesReady=true.
      Step 1→2: updateContextState(1) deposits from cvValues(s_dep).
                execute() evaluates bias with numKernels=1.
    """
    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
    ctx.getIntegrator().step(1)                       # step 0→1; cvValuesReady=true
    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
    ctx.getIntegrator().step(1)                       # step 1→2; deposits at s_dep


# ---------------------------------------------------------------------------
# Reference OPES formulas (Python, Invernizzi & Parrinello 2020 J. Phys. Chem. Lett. 11:2093)
# ---------------------------------------------------------------------------

def opes_invGF(gamma):
    return (gamma - 1.0) / gamma

def opes_eps(gamma):
    """epsilon (Invernizzi & Parrinello 2020): exp(-1/(invGF*(1-invGF))) = exp(-gamma^2/(gamma-1))."""
    igf = opes_invGF(gamma)
    denom = igf * (1.0 - igf)
    if denom < 1e-30:
        return 0.0
    return math.exp(-1.0 / denom)

def opes_silverman_sigma(sigma0, gamma, D=1):
    """Silverman-adapted sigma for the very first deposit.

    At first deposit:
      - V_before = -kT*gamma  →  weight w = exp(-gamma)
      - Sentinels were uploaded as {exp(-gamma), exp(-2*gamma)}
      - After deposit:  sum_w = 2*exp(-gamma),  sum_w2 = 2*exp(-2*gamma)
      - neff = (1+sum_w)^2/(1+sum_w2) ≈ 1  (exp(-gamma) is tiny)
    """
    sentinel = math.exp(-gamma)
    sw  = 2.0 * sentinel   # sentinel + first deposit weight
    sw2 = 2.0 * sentinel**2
    neff = (1.0 + sw)**2 / (1.0 + sw2)
    rescale = (neff * (D + 2) / 4.0) ** (-1.0 / (D + 4.0))
    return sigma0 * rescale

def opes_V_1kernel(s_val, s_dep, sigma_dep, gamma, kT, variant=0):
    """OPES V(s) reference (Invernizzi & Parrinello 2020) for exactly 1 stored kernel.

    variant=0 (well-tempered): host uses invGF=(gamma-1)/gamma.
      eps_eval = eps_dep = exp(-gamma^2/(gamma-1)).
      h_stored = exp(-gamma) * (sigma0/sig_dep) [height correction].
      pz = max(G-eps_eval,0) / (1-eps_dep) [h_stored cancels].
      V = invGF*kT*log(pz+eps_eval).

    variant=1 (FIXED_SIGMA/EXPLORE): host forces invGF=1.0, so:
      eps_eval = 0 (barrier=0, V_before_kernels=0).
      eps_dep = exp(-gamma) (from cutoff2=2*gamma/invGF=2*gamma).
      h_stored = exp(0) = 1 (V_before=0, no height correction since sig=sigma0).
      pz = G / (1-eps_dep).
      V = kT*log(pz).
    """
    if variant == 1:
        igf      = 1.0
        eps_eval = 0.0
        cutoff2  = 2.0 * gamma            # = 2*gamma/invGF with invGF=1
        eps_dep  = math.exp(-0.5 * cutoff2)  # exp(-gamma)
    else:
        igf = opes_invGF(gamma)
        if igf <= 0.0:
            return 0.0
        eps_eval = opes_eps(gamma)
        cutoff2  = 2.0 * gamma / igf
        eps_dep  = math.exp(-0.5 * cutoff2)  # equals eps_eval for variant=0

    norm2 = ((s_val - s_dep) / sigma_dep)**2
    if norm2 >= cutoff2:
        eff_G = 0.0
    else:
        G     = math.exp(-0.5 * norm2)
        eff_G = max(G - eps_eval, 0.0)   # eval kernel subtracts eps_eval (0 for variant=1)

    # sum_uprob from deposit kernel: h_stored*(G_self - eps_dep); h_stored cancels with numer.
    G_self_dep = max(1.0 - eps_dep, 1e-300)
    pz = eff_G / G_self_dep
    return igf * kT * math.log(pz + eps_eval)

def opes_dVds_1kernel(s_val, s_dep, sigma_dep, gamma, kT):
    """Analytical dV/ds for 1 kernel.

    From the kernel code:
      dV/ds = dVfactor * dAccum
            = invGF*kT*(pz/(pz+eps))/prob_unnorm  *  h_stored*G*(-(s-c)/sig^2)
    With h_stored cancelling in prob_unnorm:
      dV/ds = invGF*kT*(pz/(pz+eps)) * G/(G-eps) * (-(s-c)/sig^2)

    Positive when s < s_dep (approaching kernel); negative when s > s_dep.
    Force F = -dV/ds is positive when s > s_dep (pushes away from kernel).
    """
    igf  = opes_invGF(gamma)
    eps  = opes_eps(gamma)
    if igf <= 0.0:
        return 0.0
    cutoff2 = 2.0 * gamma / igf
    norm2   = ((s_val - s_dep) / sigma_dep)**2
    if norm2 >= cutoff2:
        return 0.0
    G = math.exp(-0.5 * norm2)
    if G <= eps:
        return 0.0
    pz = (G - eps) / max(1.0 - eps, 1e-300)
    dVfactor_scaled = igf * kT * (pz / (pz + eps)) * G / (G - eps)
    return dVfactor_scaled * (-(s_val - s_dep) / sigma_dep**2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_opes_no_kernel_yet(platform):
    """Before any deposition (first getState, step=0): bias = −kT·γ (= −barrier)."""
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.2, 0.01
    pos = [(0,0,0),(1.0,0,0),(5,5,5)]
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system(pos, cv_specs, bias_specs, platform)
    E = get_energy(ctx)
    barrier = kT * gamma  # kT / (1 - invGF) = kT * gamma
    assert abs(E - (-barrier)) < 1e-3, \
        f"E={E:.6e}, expected -barrier={-barrier:.6e} (= -kT*gamma)"
    print(f"  test_opes_no_kernel_yet: OK  (E={E:.4f}, -barrier={-barrier:.4f})")


def test_opes_single_kernel_energy_at_center(platform):
    """After 1 deposition at s_dep, evaluate at s_dep.

    At the kernel center: pz = 1  →  V = invGF·kT·log(1 + ε) ≈ 0.
    Exact value: invGF*kT*log(1+eps).
    """
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.2, 0.01
    s_dep = 1.0
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    deposit_one_kernel(ctx, s_dep)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
    E_gpu = get_energy(ctx)

    # At kernel center pz = 1 exactly (h cancels); V = invGF*kT*log(1+eps).
    igf = opes_invGF(gamma)
    eps = opes_eps(gamma)
    V_ref = igf * kT * math.log(1.0 + eps)

    assert abs(E_gpu - V_ref) < TOL_E, \
        f"E_gpu={E_gpu:.6f}, V_ref={V_ref:.6f}  (expected ≈0)"
    print(f"  test_opes_single_kernel_energy_at_center: OK  "
          f"(E={E_gpu:.6f}, ref={V_ref:.6f})")


def test_opes_single_kernel_energy_off_center(platform):
    """After 1 deposition at s_dep=1.0, evaluate at s_eval=1.5.

    The Silverman-adapted sigma for deposit 1 is computed from neff≈1.
    """
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.3, 0.01
    s_dep, s_eval = 1.0, 1.5
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    deposit_one_kernel(ctx, s_dep)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval,0,0), mm.Vec3(5,5,5)])
    E_gpu = get_energy(ctx)

    sig_dep = opes_silverman_sigma(sigma0, gamma)
    V_ref = opes_V_1kernel(s_eval, s_dep, sig_dep, gamma, kT)

    assert abs(E_gpu - V_ref) < TOL_E, \
        f"E_gpu={E_gpu:.6f}, V_ref={V_ref:.6f}  (sig_dep={sig_dep:.5f})"
    print(f"  test_opes_single_kernel_energy_off_center: OK  "
          f"(E={E_gpu:.6f}, ref={V_ref:.6f}, sig_dep={sig_dep:.5f})")


def test_opes_force_direction(platform):
    """At s > s_dep: dV/ds < 0 → F = -dV/ds > 0 (pushes atom away from kernel)."""
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.3, 0.01
    s_dep, s_eval = 1.0, 1.5
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)
    deposit_one_kernel(ctx, s_dep)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval,0,0), mm.Vec3(5,5,5)])
    forces = get_forces(ctx)

    # F1.x = -dV/ds * ds/d(r1.x); ds/dr1.x = +1 for distance along x axis.
    sig_dep = opes_silverman_sigma(sigma0, gamma)
    dVds = opes_dVds_1kernel(s_eval, s_dep, sig_dep, gamma, kT)
    F_ana = -dVds   # positive when s > s_dep (pushes away from kernel)

    assert abs(forces[1][0] - F_ana) < TOL_F, \
        f"F1.x={forces[1][0]:.6f}, expected {F_ana:.6f}  (sig_dep={sig_dep:.5f})"
    print(f"  test_opes_force_direction: OK  "
          f"(F1.x={forces[1][0]:.4f}, ana={F_ana:.4f})")


def test_opes_numerical_derivative(platform):
    """FD force check at s_eval using ±dx with same kernel table."""
    dx = 1e-3
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.3, 0.01
    s_dep, s_eval = 1.0, 1.5
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)
    deposit_one_kernel(ctx, s_dep)
    # get_energy() calls calcForcesAndEnergy only (no updateContextState) → no deposit

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval+dx,0,0), mm.Vec3(5,5,5)])
    E_p = get_energy(ctx)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval-dx,0,0), mm.Vec3(5,5,5)])
    E_m = get_energy(ctx)

    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval,0,0), mm.Vec3(5,5,5)])
    F_ana = get_forces(ctx)[1][0]

    assert abs(F_num - F_ana) < TOL_F * 5, \
        f"F_num={F_num:.6f}, F_ana={F_ana:.6f}"
    print(f"  test_opes_numerical_derivative: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


def test_opes_explore_variant(platform):
    """OPES variant=1 (FIXED_SIGMA): sigma = sigma0, no Silverman adaptation.

    Uses gamma=10 (not 1e9) to avoid float underflow in exp(-gamma) weights.
    For variant=1 the host forces invGF=1.0, giving: barrier=0, eps_eval=0,
    eps_dep=exp(-gamma). V_before_kernels = 0 (not -kT*gamma like variant=0).
    With variant=0 the stored sigma would be opes_silverman_sigma(sigma0,gamma).
    With variant=1 the stored sigma is sigma0 exactly.
    """
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.25, 0.01
    s_dep, s_eval = 1.0, 1.4
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [1, 1, 100000], gp.GluedForce.BIAS_OPES)]  # variant=1
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)
    deposit_one_kernel(ctx, s_dep)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_eval,0,0), mm.Vec3(5,5,5)])
    E_gpu = get_energy(ctx)

    # variant=1 → invGF=1 (host-forced), eps_eval=0, eps_dep=exp(-gamma), sigma=sigma0.
    V_ref = opes_V_1kernel(s_eval, s_dep, sigma0, gamma, kT, variant=1)

    assert abs(E_gpu - V_ref) < TOL_E, \
        f"E_gpu={E_gpu:.6f}, V_ref={V_ref:.6f}  (sigma0={sigma0}, no Silverman)"
    print(f"  test_opes_explore_variant: OK  (E={E_gpu:.6f}, ref={V_ref:.6f})")


def test_opes_energy_sign(platform):
    """V at kernel center > V far away (OPES raises bias in visited regions toward 0)."""
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.3, 0.01
    s_dep = 1.0
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)
    deposit_one_kernel(ctx, s_dep)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
    V_center = get_energy(ctx)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep+3.0,0,0), mm.Vec3(5,5,5)])
    V_far = get_energy(ctx)

    # At center: pz=1, V≈0.  Far away: pz→0, V→-barrier=-kT*gamma<0.
    # So V_center > V_far.
    assert V_center > V_far, \
        f"V_center({V_center:.4f}) should be > V_far({V_far:.4f})"
    print(f"  test_opes_energy_sign: OK  "
          f"(V_center={V_center:.4f}, V_far={V_far:.4f})")


def test_opes_multiple_depositions(platform):
    """Run 4 integ.step() calls with pace=1 → should accumulate kernels; E finite."""
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.25, 0.01
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, _ = make_system([(0,0,0),(1.0,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    deposit_one_kernel(ctx, 1.0)   # 1 kernel; context at step=2
    # Additional depositions via integ.step (pace=1 → deposit every step)
    for _ in range(3):
        ctx.getIntegrator().step(1)

    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(1.0,0,0), mm.Vec3(5,5,5)])
    E = get_energy(ctx)
    assert math.isfinite(E), f"E={E} not finite after multiple depositions"
    print(f"  test_opes_multiple_depositions: OK  (E={E:.4f})")


def test_opes_metrics(platform):
    """getOPESMetrics returns [zed, rct, nker, neff] with sanity values.

    Verification:
      nker >= 1 after depositions (compression may reduce count below 5).
      zed  > 0 and finite.
      rct  finite and non-decreasing as bias grows.
      neff >= 1 after 2+ depositions (effective sample size).
    """
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.2, 0.01
    s_dep = 1.0
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    # pace=1 so every step after cvValuesReady deposits
    bias_specs = [([0], params, [0, 1, 100000], gp.GluedForce.BIAS_OPES)]
    ctx, force = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    def get_metrics():
        return force.getOPESMetrics(ctx, 0)

    # Before any deposition
    m0 = get_metrics()
    assert m0[2] == 0, f"nker should be 0 before deposition, got {m0[2]}"

    # Deposit first kernel
    deposit_one_kernel(ctx, s_dep)
    m1 = get_metrics()
    assert m1[2] == 1, f"nker should be 1 after deposit_one_kernel, got {m1[2]}"
    assert math.isfinite(m1[0]) and m1[0] > 0, f"zed={m1[0]} not positive/finite"
    assert math.isfinite(m1[1]), f"rct={m1[1]} not finite"

    # Deposit 4 more kernels via integ.step (pace=1, deposits every step)
    for i in range(4):
        ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
        ctx.getIntegrator().step(1)
    m5 = get_metrics()
    # Compression merges same-location deposits: nker may be 1 (all merged into 1).
    assert 1 <= m5[2] <= 5, f"nker should be in [1,5] after 5 depositions, got {m5[2]}"
    assert m5[3] >= 1.0, f"neff={m5[3]:.3f} should be >= 1"
    # rct = kT·log(sum_weights/counter) — c(t) reweighting indicator (PLUMED-style).
    # Should be finite and bounded (typically a few kJ/mol). Not strictly monotonic
    # but should stabilize at convergence; for this short test we just check
    # bounded magnitude.
    assert math.isfinite(m5[1]), f"rct={m5[1]} not finite"
    assert abs(m5[1]) < 100.0, f"rct={m5[1]:.4f} kJ/mol unreasonably large"

    # zed = sum_uprob/(KDEnorm·nker) — should be bounded near order 1.
    assert 0.0 < m5[0] < 100.0, f"zed={m5[0]:.4f} outside reasonable bounds"

    print(f"  test_opes_metrics: OK  "
          f"(nker={m5[2]:.0f}, zed={m5[0]:.4f}, rct={m5[1]:.4f} kJ/mol, neff={m5[3]:.2f})")


def test_opes_explore_basic(platform):
    """OPES_METAD_EXPLORE (variant=2): basic deposition + bounded metrics."""
    kT, gamma, sigma0, sigmaMin = 2.479, 10.0, 0.2, 0.01
    s_dep = 1.0
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, sigma0, sigmaMin]
    bias_specs = [([0], params, [2, 1, 100000], gp.GluedForce.BIAS_OPES)]  # variant=2
    ctx, force = make_system(
        [(0,0,0), (s_dep,0,0), (5,5,5)], cv_specs, bias_specs, platform)

    # Before any deposition
    m0 = force.getOPESMetrics(ctx, 0)
    assert m0[2] == 0, f"nker should be 0, got {m0[2]}"

    # Deposit 5 kernels with pace=1
    for _ in range(5):
        ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
        ctx.getIntegrator().step(1)

    m5 = force.getOPESMetrics(ctx, 0)
    assert 1 <= m5[2] <= 5, f"nker should be in [1,5], got {m5[2]}"
    assert math.isfinite(m5[0]) and m5[0] > 0, f"zed={m5[0]} invalid"
    assert math.isfinite(m5[1]), f"rct={m5[1]} not finite"

    E = get_energy(ctx)
    assert math.isfinite(E), f"E={E} not finite for EXPLORE"

    print(f"  test_opes_explore_basic: OK  "
          f"(nker={int(m5[2])}, zed={m5[0]:.4f}, rct={m5[1]:.4f}, neff={m5[3]:.2f})")


def test_opes_explore_rejects_inf_gamma(platform):
    """variant=2 must reject gamma=inf (PLUMED requires finite γ for EXPLORE)."""
    kT = 2.479
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, float('inf'), 0.2, 0.01]
    bias_specs = [([0], params, [2, 500, 100000], gp.GluedForce.BIAS_OPES)]
    raised = False
    try:
        ctx, force = make_system(
            [(0,0,0), (1,0,0), (5,5,5)], cv_specs, bias_specs, platform)
    except Exception as e:
        raised = True
        msg = str(e)
        assert ("EXPLORE" in msg) or ("gamma" in msg.lower()), \
            f"Unexpected error message: {msg}"
    assert raised, "Should have raised for gamma=inf in EXPLORE mode"
    print("  test_opes_explore_rejects_inf_gamma: OK")


def test_opes_explore_python_mode(platform):
    """Smoke-check the Python add_opes(mode='explore') ergonomic wrapper."""
    import openmm as mm_lib
    system = mm_lib.System()
    for _ in range(3):
        system.addParticle(1.0)
    f = gp.GluedForce()
    f.setTemperature(300.0)
    cv_dist = f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, [0, 1], [])
    # Mirror the Python helper's parameter assembly
    kT = 0.0083144626 * 300.0
    sigma0 = 0.2
    params = [kT, 10.0, sigma0, sigma0 * 0.05]
    int_params = [2, 1, 100000]   # variant=2
    f.addBias(gp.GluedForce.BIAS_OPES, [cv_dist], params, int_params)
    system.addForce(f)
    integ = mm_lib.LangevinIntegrator(300.0, 1.0, 0.001)
    ctx = mm_lib.Context(system, integ, platform)
    ctx.setPositions([mm_lib.Vec3(0,0,0), mm_lib.Vec3(1.0,0,0), mm_lib.Vec3(5,5,5)])
    integ.step(2)
    m = f.getOPESMetrics(ctx, 0)
    assert m[2] >= 1, f"EXPLORE should deposit, got nker={m[2]}"
    print(f"  test_opes_explore_python_mode: OK  (nker={int(m[2])})")


def test_opes_python_adaptive_sigma_wrapper(platform):
    """Python add_opes(sigma=None) → fully-adaptive σ; respects custom stride."""
    import openmm as mm_lib
    import glued as g

    system = mm_lib.System()
    for _ in range(3):
        system.addParticle(1.0)
    f = g.Force(temperature=300.0)
    cv = f.add_distance([0, 1])

    # sigma=None ⇒ adaptive; sigma_min defaults to 1e-3 in CV units.
    bias_idx = f.add_opes(cv, sigma=None, gamma=10.0, pace=2,
                          adaptive_sigma_stride=4)
    system.addForce(f)
    integ = mm_lib.LangevinIntegrator(300.0, 1.0, 0.001)
    ctx   = mm_lib.Context(system, integ, platform)
    ctx.setPositions([mm_lib.Vec3(0,0,0), mm_lib.Vec3(1.0,0,0), mm_lib.Vec3(5,5,5)])

    # First 4 steps populate the running variance — no deposition allowed.
    integ.step(4)
    m4 = f.getOPESMetrics(ctx, bias_idx)
    assert m4[2] == 0, f"adaptive warm-up: expected nker=0 at step 4, got {m4[2]}"

    # After warm-up, deposition becomes allowed and σ is set from running M2.
    integ.step(10)
    m14 = f.getOPESMetrics(ctx, bias_idx)
    assert m14[2] >= 1, f"after warm-up expected ≥1 deposit, got nker={m14[2]}"
    print(f"  test_opes_python_adaptive_sigma_wrapper: OK  (nker={int(m14[2])})")


def test_opes_python_adaptive_sigma_string_sentinel(platform):
    """sigma='adaptive' (case-insensitive) is also accepted."""
    import openmm as mm_lib
    import glued as g

    system = mm_lib.System()
    for _ in range(3):
        system.addParticle(1.0)
    f = g.Force(temperature=300.0)
    cv = f.add_distance([0, 1])
    bias_idx = f.add_opes(cv, sigma='ADAPTIVE', gamma=10.0, pace=1,
                          adaptive_sigma_stride=2)
    system.addForce(f)
    integ = mm_lib.LangevinIntegrator(300.0, 1.0, 0.001)
    ctx   = mm_lib.Context(system, integ, platform)
    ctx.setPositions([mm_lib.Vec3(0,0,0), mm_lib.Vec3(1.0,0,0), mm_lib.Vec3(5,5,5)])
    integ.step(5)
    m = f.getOPESMetrics(ctx, bias_idx)
    assert m[2] >= 1, f"sigma='ADAPTIVE' should accept and deposit, got {m[2]}"
    print(f"  test_opes_python_adaptive_sigma_string_sentinel: OK  (nker={int(m[2])})")


def test_opes_python_adaptive_rejects_fixed_uniform():
    """mode='fixed_uniform' must reject adaptive σ — incompatible combo."""
    import glued as g
    f = g.Force(temperature=300.0)
    f.add_distance([0, 1])
    try:
        f.add_opes(0, sigma=None, mode='fixed_uniform')
        assert False, "Should have raised for sigma=None + mode='fixed_uniform'"
    except ValueError as e:
        assert "fixed_uniform" in str(e) or "incompatible" in str(e).lower(), e
    print("  test_opes_python_adaptive_rejects_fixed_uniform: OK")


def test_opes_python_adaptive_rejects_stride_with_explicit_sigma():
    """adaptive_sigma_stride is only meaningful when sigma is adaptive."""
    import glued as g
    f = g.Force(temperature=300.0)
    f.add_distance([0, 1])
    try:
        f.add_opes(0, sigma=0.05, adaptive_sigma_stride=100)
        assert False, "Should have raised for explicit sigma + stride"
    except ValueError as e:
        assert "adaptive_sigma_stride" in str(e), e
    print("  test_opes_python_adaptive_rejects_stride_with_explicit_sigma: OK")


def test_opes_adaptive_blocks_early_deposition(platform):
    """sigma0=0.0 (adaptive mode) with adaptiveSigmaStride=5 via bIntParams[3].

    Deposition must be suppressed until nSamples >= adaptiveSigmaStride.
    The every-step welfordKernel increments nSamples on each execute() call.

    Step sequence (pace=1, adaptiveSigmaStride=5):
      step 0→1: updateState skipped (cvValuesReady=false); execute: nSamples→1
      step 1→2: updateState gatherDeposit (nSamples=1 < 5, skip); execute: nSamples→2
      step 2→3: …nSamples=2 < 5, skip…; nSamples→3
      step 3→4: …nSamples=3 < 5, skip…; nSamples→4
      step 4→5: …nSamples=4 < 5, skip…; nSamples→5
      step 5→6: updateState gatherDeposit (nSamples=5 >= 5, DEPOSIT); nker→1
    """
    kT, gamma, sigmaMin = 2.479, 10.0, 0.01
    STRIDE = 5
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    # sigma0=0.0 signals adaptive mode; bIntParams=[variant, pace, maxKernels, adaptiveSigmaStride]
    params = [kT, gamma, 0.0, sigmaMin]
    bias_specs = [([0], params, [0, 1, 100000, STRIDE], gp.GluedForce.BIAS_OPES)]
    ctx, force = make_system([(0,0,0),(1.0,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    # Run STRIDE steps — deposit must still be blocked (nSamples will just reach STRIDE
    # in execute() of the last step, but updateState fires first and sees nSamples-1).
    for _ in range(STRIDE):
        ctx.getIntegrator().step(1)

    nker_before = force.getOPESMetrics(ctx, 0)[2]
    assert nker_before == 0, \
        f"Expected 0 kernels after {STRIDE} steps (blocked by stride), got {nker_before}"

    # One more step: updateState now sees nSamples == STRIDE → deposit.
    ctx.getIntegrator().step(1)

    nker_after = force.getOPESMetrics(ctx, 0)[2]
    assert nker_after == 1, \
        f"Expected 1 kernel after {STRIDE+1} steps, got {nker_after}"

    print(f"  test_opes_adaptive_blocks_early_deposition: OK  "
          f"(nker@{STRIDE}steps={nker_before}, nker@{STRIDE+1}steps={nker_after})")


def test_opes_adaptive_sigma_from_variance(platform):
    """With adaptiveSigmaStride=3 and atoms fixed, Welford variance ≈ 0 so the
    deposit kernel uses the internal fallback sigma=1.0.  At the deposit center
    pz = 1 → V = invGF·kT·log(1+ε) ≈ 0, independent of sigma_fallback.

    This verifies that the adaptive path runs without NaN/inf errors and produces
    a physically sane bias energy at the kernel center.
    """
    kT, gamma, sigmaMin, STRIDE = 2.479, 10.0, 1e-6, 3
    s_dep = 1.0
    cv_specs = [(gp.GluedForce.CV_DISTANCE, [0, 1], [])]
    params = [kT, gamma, 0.0, sigmaMin]  # sigma0=0.0 → adaptive
    bias_specs = [([0], params, [0, 1, 100000, STRIDE], gp.GluedForce.BIAS_OPES)]

    # Large mass ensures atoms barely drift — Welford samples are all ≈ s_dep,
    # so variance ≈ 0 and the deposit kernel falls back to sigma_fallback = 1.0.
    ctx, force = make_system([(0,0,0),(s_dep,0,0),(5,5,5)], cv_specs, bias_specs, platform)

    # Run until just before the first deposit (STRIDE steps → blocked).
    for _ in range(STRIDE):
        ctx.getIntegrator().step(1)
    assert force.getOPESMetrics(ctx, 0)[2] == 0, "Should be 0 kernels before stride"

    # One more step triggers the first deposit.
    ctx.getIntegrator().step(1)
    assert force.getOPESMetrics(ctx, 0)[2] == 1, "Should be 1 kernel after stride+1 steps"

    # At kernel center: pz = 1 → V = invGF*kT*log(1+eps) ≈ 0, regardless of sigma.
    ctx.setPositions([mm.Vec3(0,0,0), mm.Vec3(s_dep,0,0), mm.Vec3(5,5,5)])
    E_gpu = get_energy(ctx)

    igf = opes_invGF(gamma)
    eps = opes_eps(gamma)
    V_ref = igf * kT * math.log(1.0 + eps)  # ≈ 0

    assert abs(E_gpu - V_ref) < TOL_E, \
        f"E_gpu={E_gpu:.6f}, V_ref={V_ref:.6f}  (expected ≈0 at kernel center)"
    print(f"  test_opes_adaptive_sigma_from_variance: OK  "
          f"(E_gpu={E_gpu:.6f}, V_ref={V_ref:.6f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping OPES bias tests.")
        sys.exit(0)

    print("Stage 5.2 OPES bias tests (CUDA platform):")
    test_opes_no_kernel_yet(plat)
    test_opes_single_kernel_energy_at_center(plat)
    test_opes_single_kernel_energy_off_center(plat)
    test_opes_force_direction(plat)
    test_opes_numerical_derivative(plat)
    test_opes_explore_variant(plat)
    test_opes_energy_sign(plat)
    test_opes_multiple_depositions(plat)
    test_opes_metrics(plat)
    test_opes_adaptive_blocks_early_deposition(plat)
    test_opes_adaptive_sigma_from_variance(plat)
    print("All OPES bias tests passed.")
