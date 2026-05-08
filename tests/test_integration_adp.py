"""Integration tests for all implemented CVs and biases on alanine dipeptide (ACE-ALA-NME).

System: CHARMM-GUI ADP in 29 Å cubic TIP3P box, CHARMM36m force field.
Each test builds a fresh GluedForce, attaches it to a deserialized copy of
the base CHARMM36m system, and evaluates one force computation — confirming that
every kernel path is exercised on a real, solvated MD system without crashing.

CV tests  verify the CV value is in a physically reasonable range.
Bias tests verify the potential energy is finite and non-NaN.

Atom indices (0-based, from step3_input.pdb):
  0  CAY  (ACE methyl C)       4  CY  (ACE carbonyl C) — phi atom 1
  5  OY   (ACE carbonyl O)     6  N   (ALA1 backbone N) — phi / psi
  8  CA   (ALA1 Cα)           14  C   (ALA1 backbone C) — phi / psi
 15  O    (ALA1 backbone O)   16  N   (ALA2 backbone N) — psi atom 4
 18  CA   (ALA2 Cα)           24  C   (ALA2 backbone C)
 26  NT   (NME N)             28  CAT (NME methyl C)

Phi (φ): CY(4)–N(6)–CA(8)–C(14)
Psi (ψ): N(6)–CA(8)–C(14)–N(16)
"""
import os, sys, math
import openmm as mm
from openmm.unit import *
import gluedplugin as gp

# ── Locate CHARMM-GUI openmm directory ────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ADP  = os.path.join(_REPO, "adp", "charmm-gui-7782503285", "openmm")

if not os.path.isdir(_ADP):
    print(f"SKIP: ADP CHARMM-GUI directory not found: {_ADP}")
    sys.exit(0)

# ── Import CHARMM-GUI utilities ────────────────────────────────────────────
sys.path.insert(0, _ADP)
_prev_cwd = os.getcwd()
os.chdir(_ADP)   # needed so toppar.str relative paths resolve

from omm_readparams import read_top, read_crd, read_params, read_box
from omm_vfswitch   import vfswitch

print("Building ADP system (CHARMM36m) …", flush=True)

_top    = read_top("step3_input.psf")
_crd    = read_crd("step3_input.crd")
_params = read_params("toppar.str")
_top    = read_box(_top, "sysinfo.dat")

_nbopts = dict(
    nonbondedMethod     = mm.app.PME,
    nonbondedCutoff     = 1.2 * nanometers,
    constraints         = mm.app.HBonds,
    ewaldErrorTolerance = 0.0005,
)
_base_sys = _top.createSystem(_params, **_nbopts)
_base_sys = vfswitch(_base_sys, _top, type("_I", (), {"r_on": 1.0, "r_off": 1.2})())
_SYS_XML  = mm.XmlSerializer.serialize(_base_sys)

os.chdir(_prev_cwd)

# ── Platform (CUDA > OpenCL > CPU) ────────────────────────────────────────
def _platform():
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform found")

_PLAT = _platform()
print(f"Platform: {_PLAT.getName()}", flush=True)

# ── Brief minimization from CRD positions ─────────────────────────────────
print("Minimizing (500 steps) …", flush=True)
_sys_min = mm.XmlSerializer.deserialize(_SYS_XML)
_integ   = mm.LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.001 * picoseconds)
_ctx_min = mm.Context(_sys_min, _integ, _PLAT)
_ctx_min.setPositions(_crd.positions)
mm.LocalEnergyMinimizer.minimize(_ctx_min, tolerance=100.0, maxIterations=500)
_state0  = _ctx_min.getState(getPositions=True)
_MIN_POS = _state0.getPositions(asNumpy=True).value_in_unit(nanometers)  # raw nm floats
del _ctx_min, _integ, _sys_min
print("Minimization done.", flush=True)

# ── Atom indices ───────────────────────────────────────────────────────────
CY   = 4    # ACE carbonyl C
N1   = 6    # ALA1 backbone N
CA1  = 8    # ALA1 Cα
C1   = 14   # ALA1 backbone C
O1   = 15   # ALA1 backbone O
N2   = 16   # ALA2 backbone N
CA2  = 18   # ALA2 Cα
C2   = 24   # ALA2 backbone C
O2   = 25   # ALA2 backbone O
NT   = 26   # NME N

PHI = [CY, N1, CA1, C1]
PSI = [N1, CA1, C1, N2]

# Backbone heavy atoms for RMSD / PCA (ALA1 + ALA2 N,CA,C)
BB4 = [N1, CA1, C1, N2]

# ── Helpers ────────────────────────────────────────────────────────────────
def _v(lst):
    v = mm.vectori()
    for x in lst: v.append(int(x))
    return v

def _p(lst):
    v = mm.vectord()
    for x in lst: v.append(float(x))
    return v

def _dist(a, b):
    d = _MIN_POS[a] - _MIN_POS[b]
    return float((d[0]**2 + d[1]**2 + d[2]**2) ** 0.5)

def _make_ctx(force):
    """Fresh context (deserialized base system + force) at minimized positions."""
    sys = mm.XmlSerializer.deserialize(_SYS_XML)
    sys.addForce(force)
    integ = mm.LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.001 * picoseconds)
    ctx   = mm.Context(sys, integ, _PLAT)
    ctx.setPositions([mm.Vec3(*row) for row in _MIN_POS])
    return ctx

def _cv(ctx, f):
    ctx.getState(getEnergy=True)
    return list(f.getLastCVValues(ctx))

def _E(ctx):
    s = ctx.getState(getEnergy=True)
    E = s.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    assert math.isfinite(E), f"Non-finite energy: {E}"
    return E

def _add_cv(f, cv_type, atoms, params):
    return f.addCollectiveVariable(cv_type, _v(atoms), _p(params))

def _add_bias(f, bias_type, cv_list, params, int_params):
    f.addBias(bias_type, _v(cv_list), _p(params), _v(int_params))

# ══════════════════════════════════════════════════════════════════════════
# CV TESTS — each just evaluates the CV at minimized positions
# ══════════════════════════════════════════════════════════════════════════

def test_cv_distance():
    """N(6)-CA(8) covalent bond ≈ 0.13–0.16 nm."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_DISTANCE, [N1, CA1], [])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 0.12 < cv[0] < 0.17, f"N-CA distance = {cv[0]:.4f} nm (expected 0.13–0.16)"
    print(f"  cv_distance:    N-CA = {cv[0]:.4f} nm  ✓")


def test_cv_angle():
    """N(6)-CA(8)-C(14) angle ≈ 1.8–2.1 rad (103–120°)."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_ANGLE, [N1, CA1, C1], [])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 1.7 < cv[0] < 2.2, f"N-CA-C angle = {math.degrees(cv[0]):.1f}° (expected 103–120°)"
    print(f"  cv_angle:       N-CA-C = {math.degrees(cv[0]):.1f}°  ✓")


def test_cv_phi_dihedral():
    """Phi dihedral CY-N-CA-C in (-π, π)."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PHI, [])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert -math.pi < cv[0] < math.pi, f"phi = {math.degrees(cv[0]):.1f}° out of range"
    print(f"  cv_phi:         φ = {math.degrees(cv[0]):.1f}°  ✓")


def test_cv_psi_dihedral():
    """Psi dihedral N-CA-C-N in (-π, π)."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PSI, [])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert -math.pi < cv[0] < math.pi, f"psi = {math.degrees(cv[0]):.1f}° out of range"
    print(f"  cv_psi:         ψ = {math.degrees(cv[0]):.1f}°  ✓")


def test_cv_com_distance():
    """COM(ALA1 N,CA,C) vs COM(ALA2 N,CA,C) distance: finite, > 0."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    # atoms = [n_g1, g1_atoms..., g2_atoms...]
    # params = masses of all atoms in order
    _add_cv(f, gp.GluedForce.CV_COM_DISTANCE,
            [3, N1, CA1, C1, N2, CA2, C2],
            [14.007, 12.011, 12.011, 14.007, 12.011, 12.011])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 0.1 < cv[0] < 1.5, f"COM distance = {cv[0]:.4f} nm"
    print(f"  cv_com_distance: d = {cv[0]:.4f} nm  ✓")


def test_cv_rmsd():
    """RMSD of 4 backbone atoms from their minimized positions ≈ 0."""
    ref = []
    for a in BB4:
        ref += [_MIN_POS[a, 0], _MIN_POS[a, 1], _MIN_POS[a, 2]]
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_RMSD, BB4, ref)
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert cv[0] < 1e-3, f"RMSD from self = {cv[0]:.2e} nm (expected ~0)"
    print(f"  cv_rmsd:        RMSD = {cv[0]:.2e} nm  ✓")


def test_cv_coordination():
    """N(6)–CA(8) coordination with r0=0.2 nm: ≈ 0.8 (bond well inside r0)."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    # atoms: [n_A=1, atom_A, atom_B], params: [r0, n_nn, m_nn]
    _add_cv(f, gp.GluedForce.CV_COORDINATION,
            [1, N1, CA1], [0.20, 6.0, 12.0])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 0.5 < cv[0] <= 1.0, f"coordination N-CA = {cv[0]:.4f} (expected 0.5–1)"
    print(f"  cv_coordination: CN = {cv[0]:.4f}  ✓")


def test_cv_expression():
    """Expression CV: phi + psi (sum of two dihedrals), finite value."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    idx_phi = _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PHI, [])
    idx_psi = _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PSI, [])
    iv = mm.vectori(); iv.append(idx_phi); iv.append(idx_psi)
    f.addExpressionCV("cv0 + cv1", iv)
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert -2 * math.pi < cv[2] < 2 * math.pi, f"phi+psi = {cv[2]:.4f}"
    print(f"  cv_expression:  φ+ψ = {math.degrees(cv[2]):.1f}°  ✓")


def test_cv_position():
    """X-coordinate of CA1(8), finite value in box range."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_POSITION, [CA1], [0.0])   # component = x
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 0.0 < cv[0] < 2.9, f"CA1 x = {cv[0]:.4f} nm"
    print(f"  cv_position:    CA1.x = {cv[0]:.4f} nm  ✓")


def test_cv_drmsd():
    """DRMSD of N1-CA1 pair from their own minimized distance ≈ 0."""
    ref_d = _dist(N1, CA1)
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_DRMSD, [N1, CA1], [ref_d])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert cv[0] < 1e-3, f"DRMSD from self = {cv[0]:.2e} nm (expected ~0)"
    print(f"  cv_drmsd:       DRMSD = {cv[0]:.2e} nm  ✓")


def test_cv_contactmap():
    """Contact map between 3 backbone pairs; CV is finite, in [0, 3]."""
    pairs = [(N1, N2), (CA1, CA2), (C1, C2)]
    pair_params = [0.5, 6.0, 12.0, 1.0] * 3   # r0, nn, mm, weight per pair
    atom_list = [a for ab in pairs for a in ab]
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_CONTACTMAP, atom_list, pair_params)
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 0.0 <= cv[0] <= 3.0, f"contact map = {cv[0]:.4f} (expected 0–3)"
    print(f"  cv_contactmap:  CM = {cv[0]:.4f}  ✓")


def test_cv_plane():
    """Z-component of normal to N1-CA1-C1 plane, in [-1, 1]."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_PLANE, [N1, CA1, C1], [2.0])  # component = z
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert -1.01 < cv[0] < 1.01, f"plane z = {cv[0]:.4f}"
    print(f"  cv_plane:       nz = {cv[0]:.4f}  ✓")


def test_cv_projection():
    """Projection of CA1-N2 vector onto the x-axis, finite value."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_PROJECTION, [CA1, N2], [1.0, 0.0, 0.0])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert -2.9 < cv[0] < 2.9, f"projection = {cv[0]:.4f} nm"
    print(f"  cv_projection:  proj = {cv[0]:.4f} nm  ✓")


def test_cv_volume_cell():
    """Box volume ≈ 29³ Å³ = 24.4 nm³ (allow NPT fluctuations)."""
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_VOLUME, [], [])
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert 20.0 < cv[0] < 30.0, f"box volume = {cv[0]:.3f} nm³ (expected ~24.4)"
    print(f"  cv_volume_cell: V = {cv[0]:.3f} nm³  ✓")


def test_cv_dipole():
    """Dipole magnitude of ALA1 backbone atoms (CHARMM36m partial charges)."""
    # Approximate CHARMM36m charges for N,HN,CA,C,O of ALA
    atoms   = [N1, 7, CA1, C1, O1]       # N, HN, CA, C, O
    charges = [-0.47, 0.31, 0.07, 0.51, -0.51]
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_DIPOLE, atoms, charges + [3.0])  # component=3 → |μ|
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert math.isfinite(cv[0]) and cv[0] >= 0.0, f"dipole |μ| = {cv[0]}"
    print(f"  cv_dipole:      |μ| = {cv[0]:.4f} e·nm  ✓")


def test_cv_pca():
    """PCA component of backbone displacement along first eigenvector."""
    mean  = []
    eigv  = []
    for a in BB4:
        mean += [_MIN_POS[a, 0], _MIN_POS[a, 1], _MIN_POS[a, 2]]
        eigv += [1.0/len(BB4)**0.5, 0.0, 0.0]   # x-component only, normalised
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_PCA, BB4, mean + eigv)
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    assert math.isfinite(cv[0]), f"PCA value non-finite: {cv[0]}"
    assert abs(cv[0]) < 1e-4, f"PCA from reference should be ~0, got {cv[0]:.2e}"
    print(f"  cv_pca:         PC1 = {cv[0]:.2e} nm  ✓")


def test_cv_path():
    """Path CV (s, z) with 2 frames: at frame1 s ≈ 1, z finite.
    Frames are 1-indexed (matching PLUMED): s=1 → at frame1, s=2 → at frame2."""
    # Use CA1 as the single tracked atom; large lambda → sharp weighting
    p0 = [_MIN_POS[CA1, 0], _MIN_POS[CA1, 1], _MIN_POS[CA1, 2]]
    p1 = [p0[0] + 0.3, p0[1], p0[2]]   # frame 2 shifted 0.3 nm along x
    lam = 50.0   # large lambda: at frame1, w(frame2)=exp(-50*0.09)≈0.011 → s≈1.01
    params = [lam, 2.0] + p0 + p1      # [lambda, N_frames, frame1, frame2]
    f = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    _add_cv(f, gp.GluedForce.CV_PATH, [CA1], params)
    ctx = _make_ctx(f)
    cv = _cv(ctx, f)
    # with sharp lambda, s ≈ 1.0 when atom is at frame1 (1-indexed)
    assert 0.95 < cv[0] < 1.1, f"path s = {cv[0]:.4f} (expected ~1.0 near frame1)"
    assert math.isfinite(cv[1]),  f"path z non-finite: {cv[1]}"
    print(f"  cv_path:        s = {cv[0]:.4f}, z = {cv[1]:.4f}  ✓")


# ══════════════════════════════════════════════════════════════════════════
# BIAS TESTS — each adds a phi CV + one bias, evaluates energy once
# ══════════════════════════════════════════════════════════════════════════

def _phi_force(bias_type, bias_params, bias_int=[]):
    """Build force with phi CV and one bias."""
    f   = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    idx = _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PHI, [])
    _add_bias(f, bias_type, [idx], bias_params, bias_int)
    return f

def _phi_psi_force(bias_type, bias_params, bias_int=[]):
    """Build force with phi+psi CVs and one bias acting on both."""
    f    = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    iphi = _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PHI, [])
    ipsi = _add_cv(f, gp.GluedForce.CV_DIHEDRAL, PSI, [])
    _add_bias(f, bias_type, [iphi, ipsi], bias_params, bias_int)
    return f


def test_bias_harmonic():
    """Harmonic restraint on phi: V = k*(phi - phi0)^2/2, total E finite."""
    f = _phi_force(gp.GluedForce.BIAS_HARMONIC, [100.0, -1.0])
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_harmonic:         E = {E:.4f} kJ/mol  ✓")


def test_bias_upper_wall():
    """Upper wall on phi (at = π/2): total E finite."""
    f = _phi_force(gp.GluedForce.BIAS_UPPER_WALL,
                   [100.0, math.pi/2, 0.0, 2.0])  # kappa, at, eps, n
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_upper_wall:       E = {E:.4f} kJ/mol  ✓")


def test_bias_lower_wall():
    """Lower wall on phi (at = -π/2): total E finite."""
    f = _phi_force(gp.GluedForce.BIAS_LOWER_WALL,
                   [100.0, -math.pi/2, 0.0, 2.0])  # kappa, at, eps, n
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_lower_wall:       E = {E:.4f} kJ/mol  ✓")


def test_bias_linear():
    """Linear bias on phi: V = k * phi, E finite."""
    f = _phi_force(gp.GluedForce.BIAS_LINEAR, [5.0])   # slope = 5 kJ/mol/rad
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_linear:           E = {E:.4f} kJ/mol  ✓")


def test_bias_moving_restraint():
    """Moving restraint on phi: two schedule entries, total E finite."""
    # params = [step_0, k_0, at_0, step_1, k_1, at_1], intParams = [M=2]
    params = [0.0, 50.0, -1.0,   100.0, 50.0, -0.5]
    f = _phi_force(gp.GluedForce.BIAS_MOVING_RESTRAINT, params, [2])
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_moving_restraint: E = {E:.4f} kJ/mol  ✓")


def test_bias_metad():
    """Well-tempered metadynamics on phi: E finite (0 before first deposition)."""
    # params = [height, sigma, gamma, kT, origin, max]
    # intParams = [pace, numBins, periodic]
    kT = 2.479   # 300 K in kJ/mol
    params   = [1.0, 0.2, 15.0, kT, -math.pi, math.pi]
    intpar   = [10, 64, 1]
    f = _phi_force(gp.GluedForce.BIAS_METAD, params, intpar)
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"metad E non-finite: {E}"
    print(f"  bias_metad:            E = {E:.4f} kJ/mol  ✓")


def test_bias_pbmetad():
    """Parallel-bias metadynamics on phi+psi: E finite."""
    # params = [height, gamma, kT, sigma_0, origin_0, max_0, sigma_1, origin_1, max_1]
    # intParams = [pace, numBins_0, isPeriodic_0, numBins_1, isPeriodic_1]
    kT = 2.479
    params = [1.0, 15.0, kT,
              0.2, -math.pi, math.pi,
              0.2, -math.pi, math.pi]
    intpar = [10, 64, 1, 64, 1]
    f = _phi_psi_force(gp.GluedForce.BIAS_PBMETAD, params, intpar)
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"pbmetad E non-finite: {E}"
    print(f"  bias_pbmetad:          E = {E:.4f} kJ/mol  ✓")


def test_bias_opes():
    """OPES on phi: E = 0 before first deposition, kernel runs without crash."""
    # params = [kT, gamma, sigma0, sigmaMin]
    # intParams = [?, pace, maxKernels]
    kT = 2.479
    params = [kT, 15.0, 0.2, 0.01]
    intpar = [0, 10, 100000]
    f = _phi_force(gp.GluedForce.BIAS_OPES, params, intpar)
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"opes E non-finite: {E}"
    print(f"  bias_opes:             E = {E:.4f} kJ/mol  ✓")


def test_bias_opes_expanded():
    """OPES expanded ensemble: 3 replicas using x-coord of 3 backbone atoms as ECVs."""
    # Use position CVs (x-component) for N1, CA1, C1 — each as a separate ECV / replica
    f   = gp.GluedForce(); f.setUsesPeriodicBoundaryConditions(True)
    ip0 = _add_cv(f, gp.GluedForce.CV_POSITION, [N1],  [0.0])   # x of N1
    ip1 = _add_cv(f, gp.GluedForce.CV_POSITION, [CA1], [0.0])   # x of CA1
    ip2 = _add_cv(f, gp.GluedForce.CV_POSITION, [C1],  [0.0])   # x of C1
    kT = 2.479
    # params = [kT, w0, w1, w2] — equal unnormalized weights (must be > 0)
    params = [kT, 1.0, 1.0, 1.0]
    _add_bias(f, gp.GluedForce.BIAS_OPES_EXPANDED, [ip0, ip1, ip2], params, [500])
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"opes_expanded E non-finite: {E}"
    print(f"  bias_opes_expanded:    E = {E:.4f} kJ/mol  ✓")


def test_bias_abmd():
    """ABMD ratchet on phi with maxCv = π: total E finite."""
    # params = [kappa, initial_max]
    f = _phi_force(gp.GluedForce.BIAS_ABMD, [50.0, math.pi])
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_abmd:             E = {E:.4f} kJ/mol  ✓")


def test_bias_external():
    """External bias from a precomputed sinusoidal table on phi: E finite."""
    # 1D periodic grid: origin=-π, max=π, 64 bins
    numBins, origin, maxV = 64, -math.pi, math.pi
    spacing = (maxV - origin) / numBins
    grid = [math.sin(origin + i * spacing) * 2.0 for i in range(numBins)]  # periodic
    # params = [origin, max, grid_val_0, ..., grid_val_N-1]
    # intParams = [numBins, isPeriodic]
    params = [origin, maxV] + grid
    intpar = [numBins, 1]
    f = _phi_force(gp.GluedForce.BIAS_EXTERNAL, params, intpar)
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"external E non-finite: {E}"
    print(f"  bias_external:         E = {E:.4f} kJ/mol  ✓")


def test_bias_ext_lagrangian():
    """Extended Lagrangian (AFED) coupling to phi: total E finite."""
    # params = [kappa, mass_s]; s initialised to 0 before first updateState
    f = _phi_force(gp.GluedForce.BIAS_EXT_LAGRANGIAN, [50.0, 10.0])
    ctx = _make_ctx(f)
    E = _E(ctx)
    print(f"  bias_ext_lagrangian:   E = {E:.4f} kJ/mol  ✓")


def test_bias_eds():
    """EDS adaptive restraint on phi: λ=0 initially → E=0, kernel runs."""
    # params = [target, sigma]; intParams = [pace, tau]
    f = _phi_force(gp.GluedForce.BIAS_EDS, [-1.0, 10.0], [10, 50])
    ctx = _make_ctx(f)
    E = _E(ctx)
    assert math.isfinite(E), f"EDS E non-finite: {E}"
    print(f"  bias_eds:              E = {E:.4f} kJ/mol  ✓")


# ══════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════

_CV_TESTS = [
    test_cv_distance,
    test_cv_angle,
    test_cv_phi_dihedral,
    test_cv_psi_dihedral,
    test_cv_com_distance,
    test_cv_rmsd,
    test_cv_coordination,
    test_cv_expression,
    test_cv_position,
    test_cv_drmsd,
    test_cv_contactmap,
    test_cv_plane,
    test_cv_projection,
    test_cv_volume_cell,
    test_cv_dipole,
    test_cv_pca,
    test_cv_path,
]

_BIAS_TESTS = [
    test_bias_harmonic,
    test_bias_upper_wall,
    test_bias_lower_wall,
    test_bias_linear,
    test_bias_moving_restraint,
    test_bias_metad,
    test_bias_pbmetad,
    test_bias_opes,
    test_bias_opes_expanded,
    test_bias_abmd,
    test_bias_external,
    test_bias_ext_lagrangian,
    test_bias_eds,
]

if __name__ == "__main__":
    passed = failed = 0

    print("\n── CV tests ──────────────────────────────────────────────────────")
    for fn in _CV_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed += 1

    print("\n── Bias tests ────────────────────────────────────────────────────")
    for fn in _BIAS_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed += 1

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Integration tests: {passed}/{total} passed")
    if failed:
        sys.exit(1)
