"""Stage 2 (OPES multithermal): bias evaluation + ECV mapping acceptance tests.

The multithermal bias expands a single CV_ENERGY over an inverse-temperature ladder
{β_l}.  Before any ΔF learning (ΔF_l = 0), the bias is (expanded-ensemble form):

    diff_l = -(β_l - β0)·U
    V      = -kT0·(diffMax + log( Σ_l exp(diff_l - diffMax) / N ))
    dV/dU  =  kT0·Σ_l p_l·(β_l - β0),    p_l = softmax(diff)_l

The GLUED force on each atom is then -dV/dCV · dCV/dx = -dV/dU·(-F) = (dV/dU)·F, so the
total force (real + GLUED) is (1 + dV/dU)·F_ref.

Validates:
  1. value:    energy CV U == PE of the system without GLUED.
  2. bias:     V (= PE_glued - PE_ref) matches the hand-computed reference.
  3. dV/dU:    (F_glued - F_ref) == dV/dU · F_ref (force-factor correctness).
  4. N=1@T0:   a single reference-temperature state gives zero bias and zero added force.

(getState() triggers execute() but NOT updateContextState(), so ΔF stays 0 here — we
test the bias mapping at the flat-ΔF point; ΔF *learning* is exercised by stepping.)
"""
import sys, math
sys.path.insert(0, ".")
import openmm as mm
from openmm import unit
import gluedplugin as gp

PLATFORM = "CUDA"
N = 6
KB = 0.0083144621            # kJ/mol/K
KJ = unit.kilojoule_per_mole
KJNM = unit.kilojoule_per_mole / unit.nanometer
POS = [mm.Vec3(0.15 * i, 0.03 * ((-1) ** i), 0.01 * i) for i in range(N)]


def _add_forces(system):
    bond = mm.HarmonicBondForce()
    for i in range(N - 1):
        bond.addBond(i, i + 1, 0.15, 5000.0)
    system.addForce(bond)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for i in range(N):
        nb.addParticle(0.2 if i % 2 == 0 else -0.2, 0.30, 0.6)
    system.addForce(nb)


def _make_system():
    s = mm.System()
    for _ in range(N):
        s.addParticle(12.0)
    _add_forces(s)
    return s


def _ref_bias(U, betas, kT0):
    """Hand-computed V and dV/dU at ΔF=0."""
    beta0 = 1.0 / kT0
    diff = [-(b - beta0) * U for b in betas]
    dmax = max(diff)
    exps = [math.exp(d - dmax) for d in diff]
    s = sum(exps)
    V = -kT0 * (dmax + math.log(s / len(betas)))
    p = [e / s for e in exps]
    dVdU = kT0 * sum(p[l] * (betas[l] - beta0) for l in range(len(betas)))
    return V, dVdU


def _reference(platform):
    sref = _make_system()
    cref = mm.Context(sref, mm.VerletIntegrator(0.001), platform)
    cref.setPositions(POS)
    st = cref.getState(getEnergy=True, getForces=True)
    return st.getPotentialEnergy().value_in_unit(KJ), st.getForces()


def _glued(platform, betas, kT0, pace=500):
    s = _make_system()
    f = gp.GluedForce()
    eidx = f.addCollectiveVariable(gp.GluedForce.CV_ENERGY, mm.vectori(), mm.vectord())
    params = mm.vectord([kT0] + list(betas))
    f.addBias(gp.GluedForce.BIAS_OPES_MULTITHERMAL, mm.vectori([eidx]),
              params, mm.vectori([pace]))
    s.addForce(f)
    c = mm.Context(s, mm.VerletIntegrator(0.001), platform)
    c.setPositions(POS)
    st = c.getState(getEnergy=True, getForces=True)
    U = list(f.getLastCVValues(c))[eidx]
    PE = st.getPotentialEnergy().value_in_unit(KJ)
    F = st.getForces()
    return U, PE, F


def main(platform):
    PE_ref, F_ref = _reference(platform)
    T0 = 300.0
    kT0 = KB * T0

    # --- multithermal over a temperature ladder ---
    temps = [300.0, 400.0, 500.0, 600.0]
    betas = [1.0 / (KB * T) for T in temps]
    U, PE_g, F_g = _glued(platform, betas, kT0)
    V_ref, dVdU_ref = _ref_bias(U, betas, kT0)

    # 1. value
    assert abs(U - PE_ref) < max(1e-2, 1e-4 * abs(PE_ref)), f"U={U}, PE_ref={PE_ref}"
    print(f"  test_value: OK  (U={U:.4f}, PE_ref={PE_ref:.4f})")

    # 2. bias V
    V_g = PE_g - PE_ref
    assert abs(V_g - V_ref) < max(1e-2, 1e-3 * abs(V_ref)), f"V_glued={V_g:.4f}, V_ref={V_ref:.4f}"
    print(f"  test_bias_value: OK  (V={V_g:.4f}, ref={V_ref:.4f}, dV/dU_ref={dVdU_ref:.5f})")

    # 3. force factor: (F_glued - F_ref) == dV/dU * F_ref
    max_err = 0.0
    fmax = max(abs(F_ref[a][c].value_in_unit(KJNM)) for a in range(N) for c in range(3))
    for a in range(N):
        for comp in range(3):
            fr = F_ref[a][comp].value_in_unit(KJNM)
            fg = F_g[a][comp].value_in_unit(KJNM)
            max_err = max(max_err, abs((fg - fr) - dVdU_ref * fr))
    assert max_err < max(5.0, 0.01 * fmax), f"force-factor mismatch (max {max_err:.3f})"
    print(f"  test_force_factor: OK  (max |ΔF - dV/dU·F| {max_err:.3f} kJ/mol/nm)")

    # 4. N=1 at the reference temperature -> zero bias, zero added force
    beta0 = 1.0 / kT0
    U1, PE_1, F_1 = _glued(platform, [beta0], kT0)
    assert abs((PE_1 - PE_ref)) < max(1e-3, 1e-5 * abs(PE_ref)), f"N=1 V={PE_1-PE_ref}"
    max_df = max(abs(F_1[a][c].value_in_unit(KJNM) - F_ref[a][c].value_in_unit(KJNM))
                 for a in range(N) for c in range(3))
    assert max_df < max(1.0, 1e-3 * fmax), f"N=1 added force {max_df}"
    print(f"  test_single_state_at_T0_no_bias: OK  (V={PE_1-PE_ref:.2e}, max added F {max_df:.2e})")

    # 5. ΔF learning runs: V at a FIXED config must change after stepping (ΔF updated),
    #    isolating the learning from the position-dependence of U. Stays finite throughout.
    s = _make_system()
    f = gp.GluedForce()
    eidx = f.addCollectiveVariable(gp.GluedForce.CV_ENERGY, mm.vectori(), mm.vectord())
    f.addBias(gp.GluedForce.BIAS_OPES_MULTITHERMAL, mm.vectori([eidx]),
              mm.vectord([kT0] + list(betas)), mm.vectori([5]))   # pace=5
    s.addForce(f)
    integ = mm.LangevinMiddleIntegrator(T0 * unit.kelvin, 1.0 / unit.picosecond,
                                        0.0005 * unit.picosecond)
    c = mm.Context(s, integ, platform)
    c.setPositions(POS)
    V_before = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) - PE_ref
    finite = True
    for _ in range(30):
        integ.step(2)
        e = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
        if not math.isfinite(e):
            finite = False; break
    c.setPositions(POS)   # back to the reference config; ΔF has now been learned
    V_after = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) - PE_ref
    assert finite, "energy went non-finite during multithermal dynamics"
    assert math.isfinite(V_after) and abs(V_after - V_before) > 1e-6, \
        f"ΔF learning did not change the bias at a fixed config (V_before={V_before:.4f}, V_after={V_after:.4f})"
    print(f"  test_deltaF_learning_runs: OK  (V@POS {V_before:.3f} -> {V_after:.3f} after learning)")

    # 6. ΔF serialization round-trip: the learned per-state ΔF (GPU-resident) must
    #    survive getBiasState()/setBiasState(). Learn ΔF in run A, snapshot the bias
    #    state, then restore it into a fresh run B and confirm the bias at the SAME
    #    config matches A (and differs from the flat-ΔF value, proving learning happened).
    def _mt_force():
        s = _make_system()
        f = gp.GluedForce()
        e = f.addCollectiveVariable(gp.GluedForce.CV_ENERGY, mm.vectori(), mm.vectord())
        f.addBias(gp.GluedForce.BIAS_OPES_MULTITHERMAL, mm.vectori([e]),
                  mm.vectord([kT0] + list(betas)), mm.vectori([5]))
        s.addForce(f)
        integ = mm.LangevinMiddleIntegrator(T0 * unit.kelvin, 1.0 / unit.picosecond,
                                            0.0005 * unit.picosecond)
        c = mm.Context(s, integ, platform)
        c.setPositions(POS)
        return f, c, integ

    fA, cA, iA = _mt_force()
    for _ in range(40):
        iA.step(2)
    cA.setPositions(POS)
    V_learned = cA.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) - PE_ref
    blob = fA.getBiasState()

    fB, cB, iB = _mt_force()
    V_fresh = cB.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) - PE_ref
    fB.setBiasState(blob)
    V_restored = cB.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) - PE_ref
    assert abs(V_fresh - V_learned) > 1e-6, \
        f"ΔF did not move during learning (fresh={V_fresh:.4f}, learned={V_learned:.4f})"
    assert abs(V_restored - V_learned) < 1e-4, \
        f"ΔF serialization round-trip mismatch (learned={V_learned:.4f}, restored={V_restored:.4f})"
    print(f"  test_deltaF_serialization: OK  (fresh {V_fresh:.3f}, learned {V_learned:.3f}, "
          f"restored {V_restored:.3f})")

    print("All multithermal bias tests passed.")


if __name__ == "__main__":
    try:
        plat = mm.Platform.getPlatformByName(PLATFORM)
    except Exception:
        print("CUDA platform not available — skipping multithermal tests.")
        sys.exit(0)
    print("Stage 2 — OPES multithermal bias tests (CUDA platform):")
    main(plat)
