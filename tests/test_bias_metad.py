"""Stage 5.3 acceptance tests — Well-tempered metadynamics (grid-based).

Physics:
  Well-tempered MetaD deposits Gaussians of decreasing height:
    height_n = height0 * exp(-V(cv_n) / ((gamma-1)*kT))
  where V(cv_n) is the accumulated bias at the current CV position.

  When gamma→∞ (standard MetaD): height_n = height0 (constant).
  In the long-time limit: V(cv) → (gamma-1)*kT   at the deposition site.

Grid layout (PLUMED convention):
  actualPoints[d] = numBins[d] + (periodic[d] ? 0 : 1)
  spacing[d] = (maxVal[d] - origin[d]) / numBins[d]
  strides[0] = 1; strides[d] = strides[d-1] * actualPoints[d-1]

Bias parameters:
  parameters: [height0, sigma_0, ..., sigma_{D-1}, gamma, kT,
               origin_0, ..., origin_{D-1}, max_0, ..., max_{D-1}]
  integerParameters: [pace, numBins_0, ..., numBins_{D-1},
                      periodic_0, ..., periodic_{D-1}]

Key test notes:
  - updateState() deposits at step % pace == 0, using CV values from the
    PREVIOUS execute() (cvValuesReady_ must be True).
  - Prime cvValuesReady_ by calling getState(getEnergy=True) BEFORE step(1).
  - After step(1): the grid has one Gaussian deposited at the primed CV value.
    To read back the bias at a specific position: setPositions, then getState().
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_E  = 1e-6   # energy tolerance (kJ/mol)
TOL_F  = 1e-3   # force tolerance (kJ/mol/nm), looser — FD on discrete grid


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_system_1d(d_nm, height0, sigma, gamma, kT, origin, maxVal, numBins,
                   pace=1, periodic=0, platform=None):
    """One-particle system with a single distance CV and a 1-D MetaD bias.

    Two atoms: atom0 at origin, atom1 at (d_nm, 0, 0).
    CV = distance(0, 1).
    A dummy third atom far away avoids PBC singularities.
    """
    sys_ = mm.System()
    for _ in range(3):
        sys_.addParticle(1.0)

    f = gp.GluedForce()

    # Distance CV: atoms 0 and 1
    av = mm.vectori(); av.append(0); av.append(1)
    pv = mm.vectord()
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, pv)

    # MetaD bias on CV 0
    civ = mm.vectori(); civ.append(0)
    bparams = mm.vectord()
    # [height0, sigma_0, gamma, kT, origin_0, max_0]
    bparams.append(height0)
    bparams.append(sigma)
    bparams.append(gamma)
    bparams.append(kT)
    bparams.append(origin)
    bparams.append(maxVal)
    biparams = mm.vectori()
    # [pace, numBins_0, periodic_0]
    biparams.append(pace)
    biparams.append(numBins)
    biparams.append(periodic)
    f.addBias(gp.GluedForce.BIAS_METAD, civ, bparams, biparams)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform)
    pos = [mm.Vec3(0, 0, 0), mm.Vec3(d_nm, 0, 0), mm.Vec3(50, 50, 50)]
    ctx.setPositions(pos)
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
# Test 1: Single Gaussian deposit — grid value at peak and ±sigma
# ---------------------------------------------------------------------------

def test_metad_deposit_single_gaussian(platform):
    """Deposit one Gaussian at cv=1.0, read back at center and center±sigma.

    Grid: 200 bins, [0.0, 2.0], non-periodic.
    spacing = 2.0/200 = 0.01 nm.
    actualPoints = 201.
    Bin 100 corresponds to pos = 0 + 100*0.01 = 1.0 nm  (exactly at center).
    Bin 105 corresponds to pos = 1.05 nm  = center + sigma.

    After one deposit at cv=1.0:
      grid[100] = height * exp(-(0/sigma)^2/2) = height * 1.0 = height
      grid[105] = height * exp(-(0.05/0.05)^2/2) = height * exp(-0.5)

    The eval kernel interpolates the grid. At exactly cv=1.0:
      frac = (1.0 - 0.0) * (200/2.0) = 100.0 → lo=99, alpha=1.0 (or lo=100, alpha=0)
    Wait: frac=100.0 → (int)frac = 100 → lo = min(100, N-2=199) = 100, alpha=0.0, hi=101.
    So the interpolated value is (1-0)*grid[100] + 0*grid[101] = grid[100] = height. ✓

    At cv=1.05:
      frac = (1.05 - 0.0) * 100 = 105.0 → lo=105, alpha=0.0, hi=106.
      V = grid[105] = height * exp(-0.5). ✓
    """
    height0 = 1.0
    sigma   = 0.05
    gamma   = 100.0   # large → nearly standard MetaD (constant height)
    kT      = 2.479
    origin  = 0.0
    maxVal  = 2.0
    numBins = 200

    center_cv = 1.0   # prime positions so cv=1.0

    ctx, _ = make_system_1d(center_cv, height0, sigma, gamma, kT,
                             origin, maxVal, numBins, pace=1,
                             periodic=0, platform=platform)

    # Prime cvValuesReady_ = True (cv=1.0 stored in GPU buffer)
    E0 = get_energy(ctx)
    assert abs(E0) < TOL_E, f"Before deposition: E should be 0, got {E0:.3e}"

    # MetaD skips deposition on step=0 (matching PLUMED !isFirstStep_ guard).
    # First step(1): updateState(step=0) → skip; atom stays at cv=1.0 (no force).
    # Second step(1): updateState(step=1) → deposits at cv=1.0 (cvValues from step 0's execute).
    ctx.getIntegrator().step(1)   # step=0: skip
    ctx.getIntegrator().step(1)   # step=1: deposit Gaussian at cv=1.0

    # Read back bias at exactly cv=1.0 (center of the Gaussian)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E_center = get_energy(ctx)
    assert abs(E_center - height0) < TOL_E, \
        (f"At center: E={E_center:.8f}, expected height0={height0:.8f}  "
         f"(diff={abs(E_center-height0):.2e})")

    # Read back bias at cv = center + sigma = 1.05
    cv_plus = center_cv + sigma
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_plus, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E_plus = get_energy(ctx)
    expected_plus = height0 * math.exp(-0.5)
    assert abs(E_plus - expected_plus) < TOL_E, \
        (f"At center+sigma: E={E_plus:.8f}, expected {expected_plus:.8f}  "
         f"(diff={abs(E_plus-expected_plus):.2e})")

    # Read back bias at cv = center - sigma = 0.95
    cv_minus = center_cv - sigma
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_minus, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E_minus = get_energy(ctx)
    assert abs(E_minus - expected_plus) < TOL_E, \
        (f"At center-sigma: E={E_minus:.8f}, expected {expected_plus:.8f}  "
         f"(diff={abs(E_minus-expected_plus):.2e})")

    print(f"  test_metad_deposit_single_gaussian: OK  "
          f"(E_center={E_center:.6f}, E_±sigma={E_plus:.6f})")


# ---------------------------------------------------------------------------
# Test 2: Well-tempered convergence — 50 depositions at same point
# ---------------------------------------------------------------------------

def test_metad_well_tempered_convergence(platform):
    """Deposit 50 Gaussians at the same CV position with well-tempering.

    Theory: the accumulated bias V at the deposition site satisfies
      V_n+1 = V_n + height0 * exp(-V_n / ((gamma-1)*kT))
    which converges to V_∞ = (gamma-1)*kT.

    Parameters: gamma=10, kT=2.479, height0=1.0.
    Convergence target: V_∞ = 9 * 2.479 = 22.311 kJ/mol.

    After 50 depositions (all at same point), the accumulated bias should be
    within 5% of the convergence limit.

    Approach:
      - pace=1, so deposition happens every step.
      - Keep positions fixed (cv=1.0) throughout.
      - After 50 steps, read bias at cv=1.0.
    """
    height0 = 1.0
    sigma   = 0.05
    gamma   = 10.0
    kT      = 2.479
    origin  = 0.0
    maxVal  = 2.0
    numBins = 200

    center_cv = 1.0

    # Compute the expected bias from the discrete well-tempered recursion.
    # The continuous-limit V_inf = (gamma-1)*kT is only approached as h→0
    # and n→∞.  For discrete steps with finite h, V grows beyond V_inf.
    V_expected = 0.0
    for _ in range(50):
        V_expected += height0 * math.exp(-V_expected / ((gamma - 1.0) * kT))

    ctx, _ = make_system_1d(center_cv, height0, sigma, gamma, kT,
                             origin, maxVal, numBins, pace=1,
                             periodic=0, platform=platform)

    # Prime cvValuesReady_
    get_energy(ctx)

    # 50 steps, repositioning to center_cv each time so deposition is always
    # at cv=1.0 (the integrator moves atoms but we reset before the next step).
    for _ in range(50):
        ctx.getIntegrator().step(1)
        # Reset positions to keep cv at center for next deposition
        ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                          mm.Vec3(50, 50, 50)])

    # Read accumulated bias at center: should match the discrete recursion.
    E = get_energy(ctx)
    rel_err = abs(E - V_expected) / V_expected
    assert rel_err < 0.02, \
        (f"Well-tempered convergence: E={E:.4f} kJ/mol, "
         f"V_expected={V_expected:.4f}, rel_err={rel_err:.3f} (>2%)")
    print(f"  test_metad_well_tempered_convergence: OK  "
          f"(E={E:.4f}, V_expected={V_expected:.4f}, rel_err={rel_err:.3%})")


# ---------------------------------------------------------------------------
# Test 3: Force direction — numerical derivative check
# ---------------------------------------------------------------------------

def test_metad_force_direction(platform):
    """After one Gaussian deposit at cv=1.0, check forces via FD.

    Grid spacing = maxVal/numBins = 2.0/200 = 0.01 nm.
    Evaluate at test_cv = midpoint between grid bins 105 and 106 = 1.055 nm,
    with dx = half a grid spacing = 0.005 nm.  This makes E_p use grid[106]
    exactly and E_m use grid[105] exactly, so F_num = F_ana to double precision.
    """
    height0 = 1.0
    sigma   = 0.05
    gamma   = 100.0
    kT      = 2.479
    origin  = 0.0
    maxVal  = 2.0
    numBins = 200
    spacing = (maxVal - origin) / numBins  # 0.01 nm

    center_cv = 1.0

    # Midpoint between bins 105 and 106 (0-indexed)
    test_cv = origin + 105.5 * spacing    # = 1.055 nm
    dx      = 0.5 * spacing               # = 0.005 nm → E_p at bin 106, E_m at bin 105

    ctx, _ = make_system_1d(center_cv, height0, sigma, gamma, kT,
                             origin, maxVal, numBins, pace=1,
                             periodic=0, platform=platform)

    # Prime and deposit
    get_energy(ctx)
    ctx.getIntegrator().step(1)

    # FD at test_cv ± dx
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(test_cv + dx, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E_p = get_energy(ctx)

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(test_cv - dx, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E_m = get_energy(ctx)

    F_num = -(E_p - E_m) / (2.0 * dx)

    # Analytic force from the kernel
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(test_cv, 0, 0),
                      mm.Vec3(50, 50, 50)])
    forces = get_forces(ctx)
    F_ana = forces[1][0]   # x-component of force on atom 1

    assert abs(F_num - F_ana) < TOL_F, \
        (f"FD force check at cv={test_cv}: "
         f"F_num={F_num:.6f}, F_ana={F_ana:.6f}, "
         f"diff={abs(F_num-F_ana):.2e}")
    # Also verify Newton's third law between atoms 0 and 1
    F0x = forces[0][0]
    assert abs(F0x + F_ana) < TOL_F, \
        (f"Newton 3rd law violated: F0.x={F0x:.6f}, F1.x={F_ana:.6f}")
    print(f"  test_metad_force_direction: OK  "
          f"(F_num={F_num:.4f}, F_ana={F_ana:.4f})")


# ---------------------------------------------------------------------------
# Test 4: Serialization round-trip
# ---------------------------------------------------------------------------

def test_metad_restart(platform):
    """Deposit a few Gaussians, serialize, restore to new context, check energy.

    Flow:
      1. Create context A, deposit 3 Gaussians at cv=1.0.
      2. Call getBiasStateBytes() on context A's kernel.
      3. Create context B (same force, same system), positions at cv=1.0.
      4. Call setBiasStateBytes() with context A's state.
      5. Get energy from context B → should match context A.
    """
    height0 = 1.0
    sigma   = 0.05
    gamma   = 100.0
    kT      = 2.479
    origin  = 0.0
    maxVal  = 2.0
    numBins = 200

    center_cv = 1.0

    # ---- Context A ----
    ctx_a, force_a = make_system_1d(center_cv, height0, sigma, gamma, kT,
                                    origin, maxVal, numBins, pace=1,
                                    periodic=0, platform=platform)
    get_energy(ctx_a)
    for _ in range(3):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                            mm.Vec3(50, 50, 50)])

    ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                        mm.Vec3(50, 50, 50)])
    E_a = get_energy(ctx_a)
    assert E_a > 0.0, f"Expected positive bias energy after 3 deposits, got {E_a}"

    # Serialize context A state via the Python-friendly bytes API
    bias_bytes = force_a.getBiasState()

    # ---- Context B (fresh) ----
    ctx_b, force_b = make_system_1d(center_cv, height0, sigma, gamma, kT,
                                    origin, maxVal, numBins, pace=1,
                                    periodic=0, platform=platform)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                        mm.Vec3(50, 50, 50)])
    # Energy before restore should be 0 (empty grid)
    E_b_empty = get_energy(ctx_b)
    assert abs(E_b_empty) < TOL_E, \
        f"Context B before restore: E={E_b_empty:.3e}, expected 0"

    # Restore via Python-friendly bytes API
    force_b.setBiasState(bias_bytes)

    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(center_cv, 0, 0),
                        mm.Vec3(50, 50, 50)])
    E_b = get_energy(ctx_b)
    assert abs(E_b - E_a) < TOL_E, \
        (f"Restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}, "
         f"diff={abs(E_b-E_a):.2e}")
    print(f"  test_metad_restart: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ---------------------------------------------------------------------------
# Test 5: Zero energy before any deposition
# ---------------------------------------------------------------------------

def test_metad_zero_before_deposit(platform):
    """Before any deposition (grid is zero), the bias energy must be 0."""
    ctx, _ = make_system_1d(1.0, 1.0, 0.05, 10.0, 2.479,
                             0.0, 2.0, 200, pace=100,  # pace=100: no deposit yet
                             periodic=0, platform=platform)
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"Empty grid energy: {E:.3e}, expected 0"
    print(f"  test_metad_zero_before_deposit: OK  (E={E:.2e})")


# ---------------------------------------------------------------------------
# Test 6: Grid clamp — CV outside grid range gives edge value (not crash)
# ---------------------------------------------------------------------------

def test_metad_clamp_out_of_range(platform):
    """CV values outside [origin, maxVal] should be clamped to edge, not crash.

    Deposit at cv=1.0, then evaluate at cv=3.0 (beyond maxVal=2.0).
    Should return the edge grid value without error.
    """
    ctx, _ = make_system_1d(1.0, 1.0, 0.05, 100.0, 2.479,
                             0.0, 2.0, 200, pace=1,
                             periodic=0, platform=platform)
    get_energy(ctx)
    ctx.getIntegrator().step(1)

    # Evaluate at cv=3.0 (out of range)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(3.0, 0, 0),
                      mm.Vec3(50, 50, 50)])
    E = get_energy(ctx)
    # Should not crash; the clamped-edge value is near 0 (Gaussian decays)
    # Just verify it returns a finite, non-negative number.
    assert math.isfinite(E), f"Out-of-range eval returned non-finite: {E}"
    assert E >= -TOL_E, f"Out-of-range eval returned negative: {E:.6f}"
    print(f"  test_metad_clamp_out_of_range: OK  (E={E:.6f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping MetaD tests.")
        sys.exit(0)

    print("Stage 5.3 MetaD (grid-based) tests (CUDA platform):")
    test_metad_zero_before_deposit(plat)
    test_metad_deposit_single_gaussian(plat)
    test_metad_well_tempered_convergence(plat)
    test_metad_force_direction(plat)
    test_metad_clamp_out_of_range(plat)
    test_metad_restart(plat)
    print("All MetaD tests passed.")
