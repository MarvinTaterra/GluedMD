"""Stage 5.11 acceptance tests — BIAS_EXT_LAGRANGIAN (AFED coupling).

Physics:
  Coupling potential V = κ/2 * (CV - s)²
  dV/dCV = κ * (CV - s)   (applied as bias gradient → force on atoms via chain rule)

  Auxiliary coordinate s and momentum p are integrated on the CPU via velocity Verlet
  in updateState(). On the very first updateState() call, s is initialized to the
  current CV value.

Parameters: [kappa_0, mass_0, kappa_1, mass_1, ...]  (2*D floats)
integerParameters: [] (empty)

CV used in tests: CV_POSITION (atom 0, x-component).
  CV = x-position of atom 0.  dCV/dx = 1.  Force on atom 0 = -dV/dx = -κ*(CV-s).

Key test flow:
  updateState() is called only inside integ.step() — NOT by getState().
  So the pattern is:
    1. get_energy(ctx)         → primes cvValuesReady_=True; s still 0.
    2. integ.step(1)           → updateState() initializes s=CV at atom's current position.
    3. setPositions(x1)        → move atom; s stays at x0.
    4. get_energy / get_forces → evaluate with known (CV=x1, s=x0).

  After step(1), the atom position may have changed slightly due to Verlet integration.
  The coupling force (large kappa) drives the atom; use very small dt or zero-mass
  atom workaround. For tests, use very small kappa so force is negligible over 1 step.
"""
import sys
import math
import openmm as mm
import gluedplugin as gp

TOL_E = 1e-3   # kJ/mol — float32 Jacobian grads limit precision
TOL_F = 5e-3   # kJ/mol/nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def make_system(x0, kappa, mass_s, atom_mass=1.0, platform=None):
    """1-atom system with CV_POSITION (x-component) and EXT_LAGRANGIAN bias.

    Returns (ctx, force) with atom at x=x0 nm.
    atom_mass: OpenMM particle mass (amu). Use large mass for less displacement per step.
    """
    sys_ = mm.System()
    sys_.addParticle(atom_mass)

    f = gp.GluedForce()
    # CV_POSITION on atom 0, x-component (params[0]=0.0)
    av = mm.vectori(); av.append(0)
    pv = mm.vectord(); pv.append(0.0)   # component=0 (x)
    f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)

    # Extended Lagrangian bias on CV 0
    civ = mm.vectori(); civ.append(0)
    bpv = mm.vectord()
    bpv.append(kappa)    # kappa
    bpv.append(mass_s)   # mass of auxiliary coordinate
    iv = mm.vectori()    # empty int params
    f.addBias(gp.GluedForce.BIAS_EXT_LAGRANGIAN, civ, bpv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.0001)   # 0.1 fs — very small dt to minimize atom motion
    ctx = mm.Context(sys_, integ, platform) if platform else mm.Context(sys_, integ)
    ctx.setPositions([mm.Vec3(x0, 0, 0)])
    ctx.setVelocities([mm.Vec3(0, 0, 0)])
    return ctx, f


def prime_and_initialize_s(ctx, x0):
    """Prime cvValuesReady_ and initialize s to x0.

    Must be called with atom at x0. After this:
      - cvValuesReady_ = True
      - s = x0 (set by updateState() inside step(1))
      - Atom position is approximately x0 (may drift slightly due to forces).

    We then call setPositions to reset the atom exactly to x0 so we have a
    clean starting point.
    """
    get_energy(ctx)          # prime cvValuesReady_=True
    ctx.getIntegrator().step(1)  # updateState() initializes s=x0
    ctx.setPositions([mm.Vec3(x0, 0, 0)])  # reset atom to exact x0
    ctx.setVelocities([mm.Vec3(0, 0, 0)])


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

def test_zero_force_at_equilibrium(platform):
    """After initialization: s=CV=x0 → V=0, F=0.

    updateState() is called inside step(1), setting s=x0.
    Then with atom back at x0: CV=x0=s → V=0.
    """
    kappa = 50.0
    x0 = 1.0
    ctx, _ = make_system(x0, kappa, mass_s=1.0, platform=platform)
    prime_and_initialize_s(ctx, x0)

    # Now CV=x0=s → V=0, F=0
    E = get_energy(ctx)
    assert abs(E) < TOL_E, f"Expected E=0 at equilibrium, got E={E:.6f}"
    forces = get_forces(ctx)
    assert abs(forces[0][0]) < TOL_F, \
        f"Expected Fx=0 at equilibrium, got Fx={forces[0][0]:.6f}"
    print(f"  test_zero_force_at_equilibrium: OK  (E={E:.2e}, Fx={forces[0][0]:.2e})")


def test_coupling_energy(platform):
    """After initialization (s=x0), move atom to x1.
    V = κ/2*(x1-x0)²."""
    kappa = 100.0
    x0 = 0.5
    x1 = 1.0
    ctx, _ = make_system(x0, kappa, mass_s=1.0, platform=platform)
    prime_and_initialize_s(ctx, x0)

    # Move atom; s stays at x0 (no more step() calls, so updateState() not triggered).
    ctx.setPositions([mm.Vec3(x1, 0, 0)])
    E = get_energy(ctx)
    expected = 0.5 * kappa * (x1 - x0)**2
    assert abs(E - expected) < TOL_E, \
        f"E={E:.6f}, expected {expected:.6f}  (kappa={kappa}, x1-x0={x1-x0})"
    print(f"  test_coupling_energy: OK  (E={E:.6f}, expected={expected:.6f})")


def test_coupling_force(platform):
    """After initialization (s=x0), move atom to x1.
    Force on atom 0 along x: F = -κ*(x1-x0).
    (Positive displacement → negative force restoring toward s=x0.)"""
    kappa = 80.0
    x0 = 0.3
    x1 = 0.8   # displaced +0.5 from s
    ctx, _ = make_system(x0, kappa, mass_s=1.0, platform=platform)
    prime_and_initialize_s(ctx, x0)

    ctx.setPositions([mm.Vec3(x1, 0, 0)])
    forces = get_forces(ctx)
    expected_fx = -kappa * (x1 - x0)   # negative: pulls back toward s
    assert abs(forces[0][0] - expected_fx) < TOL_F, \
        f"Fx={forces[0][0]:.6f}, expected {expected_fx:.6f}"
    print(f"  test_coupling_force: OK  (Fx={forces[0][0]:.6f}, expected={expected_fx:.6f})")


def test_negative_displacement_force(platform):
    """Atom moved below s → positive restoring force (pushes back up)."""
    kappa = 60.0
    x0 = 1.0
    x1 = 0.4   # displaced -0.6 from s
    ctx, _ = make_system(x0, kappa, mass_s=1.0, platform=platform)
    prime_and_initialize_s(ctx, x0)

    ctx.setPositions([mm.Vec3(x1, 0, 0)])
    forces = get_forces(ctx)
    expected_fx = -kappa * (x1 - x0)   # positive: pushes toward s
    assert abs(forces[0][0] - expected_fx) < TOL_F, \
        f"Fx={forces[0][0]:.6f}, expected {expected_fx:.6f}"
    print(f"  test_negative_displacement_force: OK  "
          f"(Fx={forces[0][0]:.6f}, expected={expected_fx:.6f})")


def test_numerical_derivative(platform):
    """Finite-difference force check: F_num = -(E(x+dx)-E(x-dx))/(2*dx).

    s is frozen at x0 after initialization. No further step() calls, so
    updateState() is not triggered and s stays constant during FD evaluation.
    """
    kappa = 120.0
    x0 = 0.6
    x1 = 1.2   # working point
    dx = 1e-3  # nm

    ctx, _ = make_system(x0, kappa, mass_s=1.0, platform=platform)
    prime_and_initialize_s(ctx, x0)

    # All subsequent getState calls don't call updateState → s stays at x0.
    ctx.setPositions([mm.Vec3(x1 + dx, 0, 0)])
    E_p = get_energy(ctx)
    ctx.setPositions([mm.Vec3(x1 - dx, 0, 0)])
    E_m = get_energy(ctx)
    F_num = -(E_p - E_m) / (2.0 * dx)

    ctx.setPositions([mm.Vec3(x1, 0, 0)])
    F_ana = get_forces(ctx)[0][0]

    assert abs(F_num - F_ana) < TOL_F * 5, \
        f"F_num={F_num:.6f}, F_ana={F_ana:.6f}"
    print(f"  test_numerical_derivative: OK  (F_num={F_num:.4f}, F_ana={F_ana:.4f})")


def test_auxiliary_evolution(platform):
    """After initialization at x0, move atom to x_atom and run steps.
    The coupling force drives s toward x_atom; energy should decrease over time.

    Use a large atom mass so the atom barely moves (s chases it, not vice versa).
    """
    kappa = 200.0
    mass_s = 0.1   # light auxiliary → moves faster
    x0_init = 0.0   # s initialized here
    x_atom  = 1.0   # atom placed here after initialization

    # Large atom mass (1000 amu) so Verlet barely moves the atom
    ctx, _ = make_system(x0_init, kappa, mass_s, atom_mass=1000.0, platform=platform)
    prime_and_initialize_s(ctx, x0_init)

    # Move atom to x=1.0; s is still 0.0 (no step between initialization and setPositions)
    ctx.setPositions([mm.Vec3(x_atom, 0, 0)])
    ctx.setVelocities([mm.Vec3(0, 0, 0)])

    E_before = get_energy(ctx)
    expected_before = 0.5 * kappa * (x_atom - x0_init)**2  # 100.0 kJ/mol
    assert abs(E_before - expected_before) < TOL_E * 100, \
        f"E_before={E_before:.4f}, expected≈{expected_before:.4f}"

    # Run 50 steps with small dt=0.0001 fs — s evolves toward x_atom
    ctx.getIntegrator().step(50)

    E_after = get_energy(ctx)
    assert E_after < E_before, \
        (f"Expected energy to decrease as s moves toward CV, "
         f"got E_before={E_before:.4f}, E_after={E_after:.4f}")
    print(f"  test_auxiliary_evolution: OK  "
          f"(E_before={E_before:.2f}, E_after={E_after:.2f})")


def test_serialization(platform):
    """getBiasState / setBiasState round-trip preserves s and p."""
    kappa = 100.0
    x0 = 0.5
    ctx, force = make_system(x0, kappa, mass_s=1.0, atom_mass=1000.0, platform=platform)

    # Initialize s=x0, then move atom and run steps to build non-zero p
    prime_and_initialize_s(ctx, x0)
    ctx.setPositions([mm.Vec3(1.0, 0, 0)])
    ctx.setVelocities([mm.Vec3(0, 0, 0)])
    ctx.getIntegrator().step(10)

    # Record energy at a fixed position
    ctx.setPositions([mm.Vec3(0.8, 0, 0)])
    E_before = get_energy(ctx)

    # Serialize
    state_bytes = force.getBiasState()

    # Create fresh context, restore state
    ctx2, force2 = make_system(x0, kappa, mass_s=1.0, atom_mass=1000.0, platform=platform)
    prime_and_initialize_s(ctx2, x0)
    force2.setBiasState(state_bytes)

    ctx2.setPositions([mm.Vec3(0.8, 0, 0)])
    E_after = get_energy(ctx2)

    assert abs(E_before - E_after) < TOL_E * 10, \
        f"Serialization mismatch: E_before={E_before:.6f}, E_after={E_after:.6f}"
    print(f"  test_serialization: OK  (E_before={E_before:.6f}, E_after={E_after:.6f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping BIAS_EXT_LAGRANGIAN tests.")
        sys.exit(0)

    print("Stage 5.11 BIAS_EXT_LAGRANGIAN tests (CUDA platform):")
    test_zero_force_at_equilibrium(plat)
    test_coupling_energy(plat)
    test_coupling_force(plat)
    test_negative_displacement_force(plat)
    test_numerical_derivative(plat)
    test_auxiliary_evolution(plat)
    test_serialization(plat)
    print("All BIAS_EXT_LAGRANGIAN tests passed.")
