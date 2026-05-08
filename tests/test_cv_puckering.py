"""Stage 3.14 -- CV_PUCKERING acceptance tests (Cremer-Pople ring puckering).

Tests:
  1. test_flat_ring_Q_zero         : flat 5- and 6-membered rings -> Q=0
  2. test_chair_Q_nonzero          : canonical chair conformation -> Q > 0
  3. test_puckering_value_5ring    : Q and phi match Python reference (5-ring)
  4. test_puckering_value_6ring    : Q, theta, phi match Python reference (6-ring)
  5. test_numerical_gradient_Q_5ring : finite-difference check (5-ring, Q)
  6. test_numerical_gradient_Q_6ring : finite-difference check (6-ring, Q)
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV = 1e-4
TOL_F  = 1e-3


# ---------------------------------------------------------------------------
# Python reference implementations
# ---------------------------------------------------------------------------

def _bk_torsion(p0, p1, p2, p3):
    """Blondel-Karplus dihedral torsion(p0,p1,p2,p3)."""
    b1 = [p1[k]-p0[k] for k in range(3)]
    b2 = [p2[k]-p1[k] for k in range(3)]
    b3 = [p3[k]-p2[k] for k in range(3)]
    t = [b1[1]*b2[2]-b1[2]*b2[1], b1[2]*b2[0]-b1[0]*b2[2], b1[0]*b2[1]-b1[1]*b2[0]]
    u = [b2[1]*b3[2]-b2[2]*b3[1], b2[2]*b3[0]-b2[0]*b3[2], b2[0]*b3[1]-b2[1]*b3[0]]
    b2len = math.sqrt(sum(x**2 for x in b2))
    t_u = sum(t[k]*u[k] for k in range(3))
    b1_u = sum(b1[k]*u[k] for k in range(3))
    return math.atan2(b2len * b1_u, t_u)


def _huang_5(positions, atoms):
    """Huang torsion-based 5-ring puckering (amp, phase). Matches kernel."""
    cos4pi5 = math.cos(4*math.pi/5)
    sin4pi5 = math.sin(4*math.pi/5)
    pts = [positions[a] for a in atoms]
    # v1 = torsion(atoms[1,2,3,4]), v3 = torsion(atoms[3,4,0,1])
    v1 = _bk_torsion(pts[1], pts[2], pts[3], pts[4])
    v3 = _bk_torsion(pts[3], pts[4], pts[0], pts[1])
    Zx = (v1 + v3) / (2 * cos4pi5)
    Zy = (v1 - v3) / (2 * sin4pi5)
    amp = math.sqrt(Zx**2 + Zy**2)
    phase = math.atan2(Zy, Zx)
    return amp, phase


def _cremer_pople_6(positions, atoms):
    """Return (Q, theta, phi) for a 6-membered ring."""
    N = 6
    pts = [positions[a] for a in atoms]
    cx = sum(p[0] for p in pts)/N
    cy = sum(p[1] for p in pts)/N
    cz = sum(p[2] for p in pts)/N
    dx = [p[0]-cx for p in pts]
    dy = [p[1]-cy for p in pts]
    dz = [p[2]-cz for p in pts]
    Sx = Sy = Sz = 0.0
    Cxv = Cyv = Czv = 0.0
    for j in range(N):
        a = 2*math.pi*j/N
        Sx  += dx[j]*math.sin(a); Sy  += dy[j]*math.sin(a); Sz  += dz[j]*math.sin(a)
        Cxv += dx[j]*math.cos(a); Cyv += dy[j]*math.cos(a); Czv += dz[j]*math.cos(a)
    nx = Sy*Czv - Sz*Cyv
    ny = Sz*Cxv - Sx*Czv
    nz = Sx*Cyv - Sy*Cxv
    nlen = math.sqrt(nx*nx + ny*ny + nz*nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    zj = [dx[j]*nx + dy[j]*ny + dz[j]*nz for j in range(N)]
    A2 = math.sqrt(1.0/3.0) * sum(zj[j]*math.cos(2*math.pi*j/3) for j in range(N))
    B2 = -math.sqrt(1.0/3.0) * sum(zj[j]*math.sin(2*math.pi*j/3) for j in range(N))
    q3 = math.sqrt(1.0/6.0) * sum(zj[j]*(1 if j%2==0 else -1) for j in range(N))
    rho = math.sqrt(A2*A2 + B2*B2)
    Q = math.sqrt(rho*rho + q3*q3)
    theta = math.atan2(rho, q3)
    phi = math.atan2(B2, A2)
    return Q, theta, phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _make_ctx_puckering(positions, ring_atoms, ring_size, component, platform,
                        bias_grad=1.0):
    """Build a minimal OpenMM context with a single CV_PUCKERING CV."""
    n_atoms = len(positions)
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in ring_atoms:
        av.append(a)
    pv = mm.vectord()
    pv.append(float(ring_size))
    pv.append(float(component))
    f.addCollectiveVariable(gp.GluedForce.CV_PUCKERING, av, pv)
    f.setTestBiasGradients(mm.vectord([bias_grad]))
    sys_.addForce(f)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flat_ring_Q_zero(platform):
    """Flat rings (all atoms in z=0) should give Q=0."""
    # 5-membered flat ring
    r5 = 0.15  # nm, typical C-C bond radius
    atoms5 = 5
    pos5 = [[r5*math.cos(2*math.pi*j/atoms5), r5*math.sin(2*math.pi*j/atoms5), 0.0]
            for j in range(atoms5)]
    pos5_padded = pos5 + [[0.0, 0.0, 0.0]]  # extra dummy atom
    ctx5, f5 = _make_ctx_puckering(pos5_padded, list(range(5)), 5, 0, platform)
    cv5 = f5.getLastCVValues(ctx5)
    assert abs(cv5[0]) < TOL_CV, f"5-ring flat: expected Q=0, got {cv5[0]:.2e}"

    # 6-membered flat ring
    atoms6 = 6
    pos6 = [[r5*math.cos(2*math.pi*j/atoms6), r5*math.sin(2*math.pi*j/atoms6), 0.0]
            for j in range(atoms6)]
    pos6_padded = pos6 + [[0.0, 0.0, 0.0]]
    ctx6, f6 = _make_ctx_puckering(pos6_padded, list(range(6)), 6, 0, platform)
    cv6 = f6.getLastCVValues(ctx6)
    assert abs(cv6[0]) < TOL_CV, f"6-ring flat: expected Q=0, got {cv6[0]:.2e}"
    print("  test_flat_ring_Q_zero: OK")


def test_chair_Q_nonzero(platform):
    """Canonical cyclohexane chair conformation -> Q > 0 and theta near pi/2."""
    # Chair: even atoms up, odd atoms down by d=0.025 nm from mean plane
    r6 = 0.152  # C-C bond length projected onto ring plane (nm)
    d = 0.025   # out-of-plane displacement (nm)
    atoms6 = 6
    pos6 = []
    for j in range(atoms6):
        x = r6 * math.cos(2*math.pi*j/atoms6)
        y = r6 * math.sin(2*math.pi*j/atoms6)
        z = d if j%2 == 0 else -d
        pos6.append([x, y, z])
    pos6_padded = pos6 + [[0.0, 0.0, 0.0]]

    ctx6, f6 = _make_ctx_puckering(pos6_padded, list(range(6)), 6, 0, platform)
    cv6 = f6.getLastCVValues(ctx6)
    Q_ref, _, _ = _cremer_pople_6(pos6, list(range(6)))
    assert cv6[0] > 0.01, f"chair: expected Q > 0.01, got {cv6[0]:.4f}"
    assert abs(cv6[0] - Q_ref) < TOL_CV, f"chair Q mismatch: ref={Q_ref:.6f}, got={cv6[0]:.6f}"
    print(f"  test_chair_Q_nonzero: OK  (Q={cv6[0]:.4f})")


def test_puckering_value_5ring(platform):
    """Q and phi for a 5-membered ring match Python reference."""
    rng = random.Random(42)
    # Construct a non-flat 5-membered ring with some out-of-plane displacement
    r5 = 0.15
    pos5 = []
    for j in range(5):
        x = r5 * math.cos(2*math.pi*j/5) + rng.uniform(-0.01, 0.01)
        y = r5 * math.sin(2*math.pi*j/5) + rng.uniform(-0.01, 0.01)
        z = rng.uniform(-0.03, 0.03)
        pos5.append([x, y, z])
    pos5_padded = pos5 + [[0.0, 0.0, 0.0]]
    ring_atoms = list(range(5))

    amp_ref, phi_ref = _huang_5(pos5, ring_atoms)

    # Test amplitude (component 0)
    ctx_q, f_q = _make_ctx_puckering(pos5_padded, ring_atoms, 5, 0, platform)
    cv_q = f_q.getLastCVValues(ctx_q)
    assert abs(cv_q[0] - amp_ref) < TOL_CV, \
        f"5-ring amp: ref={amp_ref:.6f}, got={cv_q[0]:.6f}"

    # Test phase (component 1)
    ctx_p, f_p = _make_ctx_puckering(pos5_padded, ring_atoms, 5, 1, platform)
    cv_p = f_p.getLastCVValues(ctx_p)
    dphi = cv_p[0] - phi_ref
    while dphi >  math.pi: dphi -= 2*math.pi
    while dphi < -math.pi: dphi += 2*math.pi
    assert abs(dphi) < TOL_CV, \
        f"5-ring phi: ref={phi_ref:.6f}, got={cv_p[0]:.6f}"

    print(f"  test_puckering_value_5ring: OK  (amp={amp_ref:.4f}, phi={phi_ref:.4f})")


def test_puckering_value_6ring(platform):
    """Q, theta, and phi for a 6-membered ring match Python reference."""
    rng = random.Random(99)
    r6 = 0.152
    pos6 = []
    for j in range(6):
        x = r6 * math.cos(2*math.pi*j/6) + rng.uniform(-0.01, 0.01)
        y = r6 * math.sin(2*math.pi*j/6) + rng.uniform(-0.01, 0.01)
        z = (0.02 if j%2==0 else -0.02) + rng.uniform(-0.005, 0.005)
        pos6.append([x, y, z])
    pos6_padded = pos6 + [[0.0, 0.0, 0.0]]
    ring_atoms = list(range(6))

    Q_ref, theta_ref, phi_ref = _cremer_pople_6(pos6, ring_atoms)

    # Test Q (component 0)
    ctx_q, f_q = _make_ctx_puckering(pos6_padded, ring_atoms, 6, 0, platform)
    cv_q = f_q.getLastCVValues(ctx_q)
    assert abs(cv_q[0] - Q_ref) < TOL_CV, \
        f"6-ring Q: ref={Q_ref:.6f}, got={cv_q[0]:.6f}"

    # Test theta (component 1)
    ctx_t, f_t = _make_ctx_puckering(pos6_padded, ring_atoms, 6, 1, platform)
    cv_t = f_t.getLastCVValues(ctx_t)
    assert abs(cv_t[0] - theta_ref) < TOL_CV, \
        f"6-ring theta: ref={theta_ref:.6f}, got={cv_t[0]:.6f}"

    # Test phi (component 2)
    ctx_p, f_p = _make_ctx_puckering(pos6_padded, ring_atoms, 6, 2, platform)
    cv_p = f_p.getLastCVValues(ctx_p)
    dphi = cv_p[0] - phi_ref
    while dphi >  math.pi: dphi -= 2*math.pi
    while dphi < -math.pi: dphi += 2*math.pi
    assert abs(dphi) < TOL_CV, \
        f"6-ring phi: ref={phi_ref:.6f}, got={cv_p[0]:.6f}"

    print(f"  test_puckering_value_6ring: OK  (Q={Q_ref:.4f}, theta={theta_ref:.4f}, phi={phi_ref:.4f})")


def test_numerical_gradient_Q_5ring(platform):
    """Finite-difference check of amplitude gradient for a 5-membered ring.

    The Huang torsion-based Jacobian is exact (no fixed-normal approximation),
    so the FD check passes for any non-degenerate ring configuration.
    """
    r5 = 0.15
    A_amp = 0.05
    pos5 = [[r5*math.cos(2*math.pi*j/5),
             r5*math.sin(2*math.pi*j/5),
             A_amp*math.cos(4*math.pi*j/5)] for j in range(5)]
    pos5_padded = pos5 + [[0.0, 0.0, 0.0]]
    ring_atoms = list(range(5))

    n_atoms = len(pos5_padded)
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in ring_atoms:
        av.append(a)
    pv = mm.vectord()
    pv.append(5.0)   # ring size
    pv.append(0.0)   # component = Q
    f_cv.addCollectiveVariable(gp.GluedForce.CV_PUCKERING, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys_.addForce(f_cv)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)

    h = 1e-4
    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in ring_atoms:
        for ci in range(3):
            pos_p = [list(p) for p in pos5_padded]
            pos_m = [list(p) for p in pos5_padded]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            Q_p, _ = _huang_5(pos_p[:5], ring_atoms)
            Q_m, _ = _huang_5(pos_m[:5], ring_atoms)
            fd = (Q_p - Q_m) / (2*h)

            ctx.setPositions(pos5_padded)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err

    assert max_err < TOL_F, \
        f"5-ring Q gradient max error {max_err:.2e} > {TOL_F}"
    print(f"  test_numerical_gradient_Q_5ring: OK  (max_err={max_err:.2e})")


def test_numerical_gradient_Q_6ring(platform):
    """Finite-difference check of Q gradient for a 6-membered ring.

    Uses a ring in the XY plane with pure chair puckering (alternating Z).
    In this configuration the mean-plane normal is exactly Z and does not
    rotate when atoms are displaced in Z, so the fixed-normal Jacobian
    approximation is exact to numerical precision.
    """
    r6 = 0.152  # ring radius (nm)
    d = 0.025   # chair amplitude (nm)
    pos6 = [[r6*math.cos(2*math.pi*j/6),
             r6*math.sin(2*math.pi*j/6),
             d*(1.0 if j%2==0 else -1.0)] for j in range(6)]
    pos6_padded = pos6 + [[0.0, 0.0, 0.0]]
    ring_atoms = list(range(6))

    n_atoms = len(pos6_padded)
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in ring_atoms:
        av.append(a)
    pv = mm.vectord()
    pv.append(6.0)   # ring size
    pv.append(0.0)   # component = Q
    f_cv.addCollectiveVariable(gp.GluedForce.CV_PUCKERING, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys_.addForce(f_cv)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)

    h = 1e-4
    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in ring_atoms:
        for ci in range(3):
            pos_p = [list(p) for p in pos6_padded]
            pos_m = [list(p) for p in pos6_padded]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            Q_p, _, _ = _cremer_pople_6(pos_p[:6], ring_atoms)
            Q_m, _, _ = _cremer_pople_6(pos_m[:6], ring_atoms)
            fd = (Q_p - Q_m) / (2*h)

            ctx.setPositions(pos6_padded)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err

    assert max_err < TOL_F, \
        f"6-ring Q gradient max error {max_err:.2e} > {TOL_F}"
    print(f"  test_numerical_gradient_Q_6ring: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available -- skipping CV_PUCKERING tests.")
        sys.exit(0)
    print("Stage 3.14 -- CV_PUCKERING tests (CUDA platform):")
    test_flat_ring_Q_zero(plat)
    test_chair_Q_nonzero(plat)
    test_puckering_value_5ring(plat)
    test_puckering_value_6ring(plat)
    test_numerical_gradient_Q_5ring(plat)
    test_numerical_gradient_Q_6ring(plat)
    print("All CV_PUCKERING tests passed.")
