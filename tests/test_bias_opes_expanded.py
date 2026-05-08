"""Stage 5.10 — BIAS_OPES_EXPANDED acceptance tests.

V(x) = -kT * (log Σ_λ w_λ exp(-ecv_λ/kT) − logZ)
Bias gradient: dV/d(ecv_λ) = p_λ  (softmax probability)

Uses position CVs as ECVs so that forces are analytically known.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_E = 1e-5
TOL_F = 1e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _logQ(ecvs, logW, invKT):
    """Python reference: log Σ_λ exp(logW_λ - ecv_λ * invKT)."""
    vals = [logW[l] - ecvs[l] * invKT for l in range(len(ecvs))]
    mx = max(vals)
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


def _softmax(ecvs, logW, invKT):
    """Softmax probabilities p_λ."""
    lq = _logQ(ecvs, logW, invKT)
    return [math.exp(logW[l] - ecvs[l] * invKT - lq) for l in range(len(ecvs))]


def _make_ctx(positions, cv_atom_comp_pairs, kT, weights, logZ, platform):
    """Build a system where each ECV is a position CV (x/y/z of one atom).

    cv_atom_comp_pairs: list of (atom_index, component) tuples
    Returns (ctx, force, cv_indices)
    """
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms):
        sys.addParticle(12.0)

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    cv_indices = []
    for atom_idx, comp in cv_atom_comp_pairs:
        av = mm.vectori()
        av.append(atom_idx)
        pv = mm.vectord()
        pv.append(float(comp))
        cv_idx = f.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv)
        cv_indices.append(cv_idx)

    D = len(cv_indices)
    pv_bias = mm.vectord()
    pv_bias.append(kT)
    for w in weights:
        pv_bias.append(w)
    cv_idx_vec = mm.vectori()
    for ci in cv_indices:
        cv_idx_vec.append(ci)
    f.addBias(gp.GluedForce.BIAS_OPES_EXPANDED, cv_idx_vec, pv_bias,
              mm.vectori([500]))

    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    return ctx, f, cv_indices


def test_uniform_weights_bias_energy(platform):
    """Uniform weights: logZ=0, V = -kT*(log(D) - 0) for equal ECVs."""
    # 3 ECVs — positions of atoms 0,1,2 x-coordinate all equal to 1.0
    kT = 2.479
    positions = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] + [[0.0]*3]
    weights = [1.0, 1.0, 1.0]
    D = 3
    ctx, f, _ = _make_ctx(positions, [(0, 0), (1, 0), (2, 0)], kT, weights, 0.0, platform)

    state = ctx.getState(getEnergy=True)
    V_plugin = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)

    # logW = log(1/3) for each; ecvs = [1.0, 1.0, 1.0]
    invKT = 1.0 / kT
    logW = [math.log(1.0/D)] * D
    ecvs = [1.0, 1.0, 1.0]
    lq = _logQ(ecvs, logW, invKT)
    V_ref = -kT * (lq - 0.0)  # logZ=0 initially

    assert abs(V_plugin - V_ref) < TOL_E, \
        f"uniform bias energy: expected {V_ref:.6f}, got {V_plugin:.6f}"
    print(f"  test_uniform_weights_bias_energy: OK  (V={V_plugin:.6f}, ref={V_ref:.6f})")


def test_bias_energy_value(platform):
    """V matches Python reference for non-uniform weights."""
    rng = random.Random(17)
    kT = 2.479
    invKT = 1.0 / kT
    D = 4
    positions = [[rng.uniform(0.2, 0.8), 0.0, 0.0] for _ in range(D)] + [[0.0]*3]
    weights = [rng.uniform(0.5, 2.0) for _ in range(D)]
    ecvs = [positions[i][0] for i in range(D)]  # x-component of atom i

    ctx, f, _ = _make_ctx(positions, [(i, 0) for i in range(D)], kT, weights, 0.0, platform)
    state = ctx.getState(getEnergy=True)
    V_plugin = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)

    wSum = sum(weights)
    logW = [math.log(w / wSum) for w in weights]
    lq = _logQ(ecvs, logW, invKT)
    V_ref = -kT * (lq - 0.0)

    assert abs(V_plugin - V_ref) < TOL_E, \
        f"bias energy mismatch: expected {V_ref:.6f}, got {V_plugin:.6f}"
    print(f"  test_bias_energy_value: OK  (V={V_plugin:.6f}, ref={V_ref:.6f})")


def test_force_direction(platform):
    """Force on atom i x-component = -p_i (since d(ecv_i)/d(x_i) = 1.0)."""
    kT = 2.479
    invKT = 1.0 / kT
    D = 3
    positions = [[0.3, 0.0, 0.0], [0.7, 0.0, 0.0], [0.5, 0.0, 0.0]] + [[0.0]*3]
    weights = [1.0, 2.0, 0.5]

    ctx, f, _ = _make_ctx(positions, [(0, 0), (1, 0), (2, 0)], kT, weights, 0.0, platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)

    wSum = sum(weights)
    logW = [math.log(w / wSum) for w in weights]
    ecvs = [positions[i][0] for i in range(D)]
    probs = _softmax(ecvs, logW, invKT)

    for i in range(D):
        # Force on atom i x-component: -dV/d(x_i) = -dV/d(ecv_i)*d(ecv_i)/d(x_i) = -p_i
        got = raw[i][0].value_in_unit(unit)
        expected = -probs[i]
        assert abs(got - expected) < TOL_F, \
            f"atom {i} force: expected {expected:.6f}, got {got:.6f}"
    print(f"  test_force_direction: OK  (probs={[f'{p:.4f}' for p in probs]})")


def test_two_biases_independent(platform):
    """Two OPES_EXPANDED biases on separate CVs contribute independently."""
    kT = 2.479
    invKT = 1.0 / kT
    # Bias 0: ECVs from atoms 0 and 1 (x-component)
    # Bias 1: ECVs from atoms 2 and 3 (x-component)
    positions = [[0.4, 0.0, 0.0], [0.6, 0.0, 0.0],
                 [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]] + [[0.0]*3]

    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms):
        sys.addParticle(12.0)

    force = gp.GluedForce()
    force.setUsesPeriodicBoundaryConditions(False)

    cv_ids = []
    for atom_idx in range(4):
        av = mm.vectori(); av.append(atom_idx)
        pv = mm.vectord(); pv.append(0.0)  # x-component
        cv_ids.append(force.addCollectiveVariable(gp.GluedForce.CV_POSITION, av, pv))

    # Bias 0 on CVs 0,1; bias 1 on CVs 2,3
    for grp in [(0, 1), (2, 3)]:
        ci = mm.vectori()
        for g in grp: ci.append(cv_ids[g])
        pv = mm.vectord([kT, 1.0, 1.0])
        force.addBias(gp.GluedForce.BIAS_OPES_EXPANDED, ci, pv, mm.vectori([500]))

    sys.addForce(force)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True)
    V_plugin = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoules_per_mole)

    logW = [math.log(0.5), math.log(0.5)]
    V0 = -kT * (_logQ([positions[0][0], positions[1][0]], logW, invKT) - 0.0)
    V1 = -kT * (_logQ([positions[2][0], positions[3][0]], logW, invKT) - 0.0)
    V_ref = V0 + V1

    assert abs(V_plugin - V_ref) < TOL_E, \
        f"two biases: expected {V_ref:.6f}, got {V_plugin:.6f}"
    print(f"  test_two_biases_independent: OK  (V={V_plugin:.6f}, V0={V0:.4f}, V1={V1:.4f})")


def test_softmax_normalization(platform):
    """Sum of softmax probabilities must equal 1 (verified via forces)."""
    kT = 2.479
    invKT = 1.0 / kT
    rng = random.Random(77)
    D = 5
    positions = [[rng.uniform(0.1, 0.9), 0.0, 0.0] for _ in range(D)] + [[0.0]*3]
    weights = [rng.uniform(0.3, 3.0) for _ in range(D)]

    ctx, f, _ = _make_ctx(positions, [(i, 0) for i in range(D)], kT, weights, 0.0, platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)

    # Sum of forces on x-component = sum of p_λ = 1 (with negative sign from force convention)
    total = sum(-raw[i][0].value_in_unit(unit) for i in range(D))
    assert abs(total - 1.0) < TOL_F, f"sum(p_λ) should be 1.0, got {total:.6f}"
    print(f"  test_softmax_normalization: OK  (Σp_λ={total:.8f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping BIAS_OPES_EXPANDED tests.")
        sys.exit(0)
    print("Stage 5.10 — BIAS_OPES_EXPANDED tests (CUDA platform):")
    test_uniform_weights_bias_energy(plat)
    test_bias_energy_value(plat)
    test_force_direction(plat)
    test_two_biases_independent(plat)
    test_softmax_normalization(plat)
    print("All BIAS_OPES_EXPANDED tests passed.")
