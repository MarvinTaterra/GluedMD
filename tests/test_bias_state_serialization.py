"""Stage 6.1 — Bias state serialization acceptance tests.

Verifies getBiasState() / setBiasState() round-trips for every stateful bias:
  OPES           — kernels + logZ + nSamples + runningMean/M2
  ABMD           — maxCv ratchet position
  OPES_EXPANDED  — logZ + numUpdates
  MetaD (smoke)  — grid + numDeposited (full suite in test_bias_metad.py)
  EDS   (smoke)  — lambda + runAvg    (full suite in test_bias_eds.py)
  ExtLag (smoke) — s + p              (full suite in test_bias_ext_lagrangian.py)
  Combined       — all stateful biases in one GluedForce

Also verifies the binary header: magic 'GPUS' + little-endian int32 version=1.
"""
import sys, math, struct
import openmm as mm
from openmm.unit import kilojoules_per_mole
import gluedplugin as gp

# ── Platform ──────────────────────────────────────────────────────────────
def _get_platform():
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform found")

PLAT = _get_platform()
print(f"Platform: {PLAT.getName()}", flush=True)

# ── Shared constants ──────────────────────────────────────────────────────
_MASS = 1000.0   # large mass → atoms barely move per step
_DT   = 0.0001   # ps
_KT   = 2.479    # kJ/mol at 300 K
TOL   = 1e-5     # kJ/mol energy tolerance

# ── Helpers ───────────────────────────────────────────────────────────────
def _vi(*args): v = mm.vectori(); [v.append(int(x)) for x in args]; return v
def _vd(*args): v = mm.vectord(); [v.append(float(x)) for x in args]; return v

def _make_1d_ctx(force, cv_val=1.0):
    """3-atom system: atom0 at origin, atom1 at cv_val on x-axis, atom2 as ghost."""
    sys = mm.System()
    for _ in range(3):
        sys.addParticle(_MASS)
    sys.addForce(force)
    ctx = mm.Context(sys, mm.VerletIntegrator(_DT), PLAT)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    return ctx

def _get_E(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

def _dist_cv(f):
    """Add distance(atom0, atom1) as CV; return cv_index."""
    return f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, _vi(0, 1), _vd())

def _pos_cv(f, atom):
    """Add x-position of atom as CV; return cv_index."""
    return f.addCollectiveVariable(gp.GluedForce.CV_POSITION, _vi(atom), _vd(0.0))

# ══════════════════════════════════════════════════════════════════════════
# 1. Header format
# ══════════════════════════════════════════════════════════════════════════

def test_magic_version_header():
    """getBiasState() blob starts with 'GPUS' + version=1."""
    f = gp.GluedForce()
    _dist_cv(f)
    f.addBias(gp.GluedForce.BIAS_HARMONIC, _vi(0), _vd(10.0, 1.0), _vi())
    ctx = _make_1d_ctx(f)
    _get_E(ctx)
    blob = f.getBiasState()
    assert len(blob) >= 8, f"blob too short: {len(blob)} bytes"
    assert blob[:4] == b'GPUS', f"wrong magic: {blob[:4]!r}"
    version = struct.unpack_from('<i', blob, 4)[0]
    assert version == 1, f"expected version=1, got {version}"
    print(f"  test_magic_version_header: OK  ({len(blob)} bytes, version={version})")


# ══════════════════════════════════════════════════════════════════════════
# 2. OPES round-trip
# ══════════════════════════════════════════════════════════════════════════

def _make_opes_force(pace=1, sigma0=0.1, sigmaMin=0.01, gamma=15.0):
    f = gp.GluedForce()
    idx = _dist_cv(f)
    f.addBias(gp.GluedForce.BIAS_OPES, _vi(idx),
              _vd(_KT, gamma, sigma0, sigmaMin), _vi(0, pace, 100000))
    return f

def test_opes_round_trip():
    """OPES: deposit 3 kernels, serialize, restore → same bias energy at same position."""
    cv_val = 0.5

    # ── Context A: deposit 3 kernels ──────────────────────────────────────
    f_a = _make_opes_force(pace=1)
    ctx_a = _make_1d_ctx(f_a, cv_val)
    # Two integrator steps trigger the first deposition (see test_bias_opes.py notes)
    for _ in range(6):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()
    assert blob[:4] == b'GPUS', "missing header after OPES run"

    # ── Context B: fresh, restore, compare ──────────────────────────────
    f_b = _make_opes_force(pace=1)
    ctx_b = _make_1d_ctx(f_b, cv_val)
    E_b_empty = _get_E(ctx_b)
    # PLUMED OPES returns -barrier when no kernels are deposited.
    # barrier = kT * gamma (Invernizzi 2020: biasfactor=gamma, invGF=(gamma-1)/gamma).
    # Our invGammaFactor = (gamma-1)/gamma, so barrier = kT / (1 - invGF) = kT * gamma.
    barrier_expected = _KT * 15.0   # _KT * gamma used in _make_opes_force
    assert abs(E_b_empty + barrier_expected) < barrier_expected * 0.01, \
        f"OPES before restore should be -barrier≈{-barrier_expected:.2f}, got {E_b_empty:.3e}"

    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"OPES restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}, diff={abs(E_b-E_a):.2e}"
    print(f"  test_opes_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 3. ABMD round-trip
# ══════════════════════════════════════════════════════════════════════════

def _make_abmd_force(kappa=200.0, initial_max=0.3):
    f = gp.GluedForce()
    idx = _dist_cv(f)
    f.addBias(gp.GluedForce.BIAS_ABMD, _vi(idx),
              _vd(kappa, initial_max), _vi())
    return f

def test_abmd_round_trip():
    """ABMD: advance maxCv by running steps, serialize, restore → same ratchet state."""
    # Start at cv=0.5, initial_max=0.3 → bias is active immediately (0.5 > 0.3)
    # After a few updateState calls, maxCv advances to ~0.5.
    kappa = 200.0
    init_max = 0.3
    cv_val = 0.5

    f_a = _make_abmd_force(kappa=kappa, initial_max=init_max)
    ctx_a = _make_1d_ctx(f_a, cv_val)
    for _ in range(10):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    # Restore to fresh context
    f_b = _make_abmd_force(kappa=kappa, initial_max=init_max)
    ctx_b = _make_1d_ctx(f_b, cv_val)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"ABMD restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}, diff={abs(E_b-E_a):.2e}"
    print(f"  test_abmd_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 4. OPES_EXPANDED round-trip
# ══════════════════════════════════════════════════════════════════════════

def _make_opes_exp_force(pace=2):
    """2 ECVs: x-position of atoms 1 and 2."""
    f = gp.GluedForce()
    ip0 = _pos_cv(f, 1)
    ip1 = _pos_cv(f, 2)
    f.addBias(gp.GluedForce.BIAS_OPES_EXPANDED, _vi(ip0, ip1),
              _vd(_KT, 1.0, 1.0), _vi(pace))
    return f

def test_opes_expanded_round_trip():
    """OPES_EXPANDED: logZ + numUpdates survive serialize/restore."""
    f_a = _make_opes_exp_force(pace=2)
    ctx_a = _make_1d_ctx(f_a, cv_val=0.4)
    for _ in range(10):
        ctx_a.getIntegrator().step(1)
    ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.4, 0, 0), mm.Vec3(0.6, 0, 0)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    f_b = _make_opes_exp_force(pace=2)
    ctx_b = _make_1d_ctx(f_b, cv_val=0.4)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.4, 0, 0), mm.Vec3(0.6, 0, 0)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"OPES_EXP restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}"
    print(f"  test_opes_expanded_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 5. MetaD smoke round-trip (grid)
# ══════════════════════════════════════════════════════════════════════════

def test_metad_smoke_round_trip():
    """MetaD grid survives serialize/restore (spot-check; full tests in test_bias_metad.py)."""
    cv_val = 1.0
    origin, maxv, nbins = 0.0, 2.0, 100
    f_a = gp.GluedForce()
    idx = _dist_cv(f_a)
    f_a.addBias(gp.GluedForce.BIAS_METAD, _vi(idx),
                _vd(1.0, 0.05, 100.0, _KT, origin, maxv), _vi(1, nbins, 0))
    ctx_a = _make_1d_ctx(f_a, cv_val)
    for _ in range(5):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    f_b = gp.GluedForce()
    idx2 = _dist_cv(f_b)
    f_b.addBias(gp.GluedForce.BIAS_METAD, _vi(idx2),
                _vd(1.0, 0.05, 100.0, _KT, origin, maxv), _vi(1, nbins, 0))
    ctx_b = _make_1d_ctx(f_b, cv_val)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"MetaD restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}"
    print(f"  test_metad_smoke_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 6. EDS smoke round-trip
# ══════════════════════════════════════════════════════════════════════════

def test_eds_smoke_round_trip():
    """EDS lambda survives serialize/restore."""
    cv_val = 1.0
    target = 0.8
    f_a = gp.GluedForce()
    idx = _dist_cv(f_a)
    f_a.addBias(gp.GluedForce.BIAS_EDS, _vi(idx),
                _vd(target, 10.0), _vi(1, 20))
    ctx_a = _make_1d_ctx(f_a, cv_val)
    for _ in range(30):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    f_b = gp.GluedForce()
    idx2 = _dist_cv(f_b)
    f_b.addBias(gp.GluedForce.BIAS_EDS, _vi(idx2),
                _vd(target, 10.0), _vi(1, 20))
    ctx_b = _make_1d_ctx(f_b, cv_val)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"EDS restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}"
    print(f"  test_eds_smoke_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 7. ExtLag smoke round-trip
# ══════════════════════════════════════════════════════════════════════════

def test_extlag_smoke_round_trip():
    """ExtLag s + p survive serialize/restore."""
    cv_val = 1.0
    f_a = gp.GluedForce()
    idx = _dist_cv(f_a)
    f_a.addBias(gp.GluedForce.BIAS_EXT_LAGRANGIAN, _vi(idx),
                _vd(100.0, 10.0), _vi())
    ctx_a = _make_1d_ctx(f_a, cv_val)
    for _ in range(15):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    f_b = gp.GluedForce()
    idx2 = _dist_cv(f_b)
    f_b.addBias(gp.GluedForce.BIAS_EXT_LAGRANGIAN, _vi(idx2),
                _vd(100.0, 10.0), _vi())
    ctx_b = _make_1d_ctx(f_b, cv_val)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"ExtLag restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}"
    print(f"  test_extlag_smoke_round_trip: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# 8. Stateless biases: serialize/restore must not crash
# ══════════════════════════════════════════════════════════════════════════

def test_stateless_biases_no_crash():
    """Harmonic, walls, linear: getBiasState/setBiasState are no-ops (no crash)."""
    for bias_type, params in [
        (gp.GluedForce.BIAS_HARMONIC,     [100.0, 1.0]),
        (gp.GluedForce.BIAS_UPPER_WALL,   [100.0, 1.5, 0.0, 2.0]),
        (gp.GluedForce.BIAS_LOWER_WALL,   [100.0, 0.5, 0.0, 2.0]),
        (gp.GluedForce.BIAS_LINEAR,       [5.0]),
    ]:
        f_a = gp.GluedForce()
        idx = _dist_cv(f_a)
        f_a.addBias(bias_type, _vi(idx), _vd(*params), _vi())
        ctx_a = _make_1d_ctx(f_a, cv_val=1.0)
        E_a = _get_E(ctx_a)
        blob = f_a.getBiasState()
        assert blob[:4] == b'GPUS', "missing header from stateless bias"

        f_b = gp.GluedForce()
        idx2 = _dist_cv(f_b)
        f_b.addBias(bias_type, _vi(idx2), _vd(*params), _vi())
        ctx_b = _make_1d_ctx(f_b, cv_val=1.0)
        f_b.setBiasState(blob)
        E_b = _get_E(ctx_b)
        assert abs(E_b - E_a) < TOL, \
            f"Stateless bias {bias_type} energy changed after restore: {E_a:.6f} vs {E_b:.6f}"
    print("  test_stateless_biases_no_crash: OK")


# ══════════════════════════════════════════════════════════════════════════
# 9. Combined: all stateful biases in one GluedForce
# ══════════════════════════════════════════════════════════════════════════

def test_combined_all_stateful():
    """All stateful biases together: serialize/restore preserves combined energy."""
    cv_val = 0.8
    origin, maxv, nbins = 0.0, 2.0, 50

    def _build():
        f = gp.GluedForce()
        # CV 0: distance(0,1)
        idist = _dist_cv(f)
        # CV 1: x-position of atom 1
        ipos  = _pos_cv(f, 1)
        # OPES on dist
        f.addBias(gp.GluedForce.BIAS_OPES, _vi(idist),
                  _vd(_KT, 15.0, 0.1, 0.01), _vi(0, 1, 100000))
        # ABMD on dist
        f.addBias(gp.GluedForce.BIAS_ABMD, _vi(idist),
                  _vd(200.0, 0.3), _vi())
        # MetaD on dist
        f.addBias(gp.GluedForce.BIAS_METAD, _vi(idist),
                  _vd(0.5, 0.05, 100.0, _KT, origin, maxv), _vi(1, nbins, 0))
        # OPES_EXPANDED on pos
        f.addBias(gp.GluedForce.BIAS_OPES_EXPANDED, _vi(ipos),
                  _vd(_KT, 1.0), _vi(2))
        return f

    f_a = _build()
    ctx_a = _make_1d_ctx(f_a, cv_val)
    for _ in range(8):
        ctx_a.getIntegrator().step(1)
        ctx_a.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_a = _get_E(ctx_a)
    blob = f_a.getBiasState()

    f_b = _build()
    ctx_b = _make_1d_ctx(f_b, cv_val)
    f_b.setBiasState(blob)
    ctx_b.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(cv_val, 0, 0), mm.Vec3(50, 50, 50)])
    E_b = _get_E(ctx_b)
    assert abs(E_b - E_a) < TOL, \
        f"Combined restart mismatch: E_a={E_a:.8f}, E_b={E_b:.8f}, diff={abs(E_b-E_a):.2e}"
    print(f"  test_combined_all_stateful: OK  (E_a={E_a:.6f}, E_b={E_b:.6f})")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Stage 6.1 — Bias state serialization tests:")
    test_magic_version_header()
    test_opes_round_trip()
    test_abmd_round_trip()
    test_opes_expanded_round_trip()
    test_metad_smoke_round_trip()
    test_eds_smoke_round_trip()
    test_extlag_smoke_round_trip()
    test_stateless_biases_no_crash()
    test_combined_all_stateful()
    print("All Stage 6.1 tests passed.")
