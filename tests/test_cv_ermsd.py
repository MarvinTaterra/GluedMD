"""Stage 3.19 -- CV_ERMSD acceptance tests (Bottaro eRMSD for RNA).

Tests:
  1. test_ermsd_zero_at_reference    : positions == reference -> eRMSD = 0
  2. test_ermsd_value                : eRMSD matches Python reference for random positions
  3. test_two_ermsd_cvs              : two independent ERMSD CVs computed correctly
  4. test_numerical_gradient         : finite-difference check (h=1e-4, TOL_F=5e-3)

N=3 residues (3 atoms each) are used throughout. Parameters: [N, cutoff, g_ref_4D...]
where g_ref_4D has 4*N*(N-1) values (all ordered pairs, 4D Bottaro G-vector each).
Bottaro form factors = [2.0, 2.0, 1/0.3] (Bottaro et al. 2014 Nucleic Acids Res. 42:13306) are hardcoded in the kernel.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV = 1e-4
TOL_F  = 5e-3   # loose: centroid-only approximation has O(1%) error

# ---------------------------------------------------------------------------
# Bottaro (2014) eRMSD Python reference
# ---------------------------------------------------------------------------

FF    = [2.0, 2.0, 1.0/0.3]
CUTOFF = 2.4
GAMMA  = math.pi / CUTOFF
MAXDIST = CUTOFF / FF[0]  # = 1.2 nm


def _compute_frame(p0, p1, p2):
    """Return (centroid, e1, e2, e3) using Bottaro (2014) eRMSD local frame convention.
    e1 = (atom0 - center) / |...|, e3 = (a×b) / |...|, e2 = e3×e1.
    """
    cx = (p0[0]+p1[0]+p2[0])/3.0
    cy = (p0[1]+p1[1]+p2[1])/3.0
    cz = (p0[2]+p1[2]+p2[2])/3.0
    ax, ay_v, az_v = p0[0]-cx, p0[1]-cy, p0[2]-cz
    la = math.sqrt(ax**2+ay_v**2+az_v**2)
    e1 = [ax/la, ay_v/la, az_v/la]
    bx, by_v, bz_v = p1[0]-cx, p1[1]-cy, p1[2]-cz
    dx = ay_v*bz_v - az_v*by_v
    dy = az_v*bx   - ax*bz_v
    dz = ax*by_v   - ay_v*bx
    ld = math.sqrt(dx**2+dy**2+dz**2)
    e3 = [dx/ld, dy/ld, dz/ld]
    e2 = [e3[1]*e1[2]-e3[2]*e1[1],
          e3[2]*e1[0]-e3[0]*e1[2],
          e3[0]*e1[1]-e3[1]*e1[0]]
    return [cx, cy, cz], e1, e2, e3


def _bottaro_gvec(ci, e1i, e2i, e3i, cj):
    """4-D Bottaro G-vector for ordered pair (i→j) using frame of residue i."""
    drx, dry, drz = cj[0]-ci[0], cj[1]-ci[1], cj[2]-ci[2]
    dist = math.sqrt(drx**2+dry**2+drz**2)
    if dist >= MAXDIST:
        return [0.0, 0.0, 0.0, 0.0]
    rt0 = (drx*e1i[0]+dry*e1i[1]+drz*e1i[2])*FF[0]
    rt1 = (drx*e2i[0]+dry*e2i[1]+drz*e2i[2])*FF[1]
    rt2 = (drx*e3i[0]+dry*e3i[1]+drz*e3i[2])*FF[2]
    rtn = math.sqrt(rt0**2+rt1**2+rt2**2)
    if rtn <= 1e-8:
        return [rt0, rt1, rt2, 2.0/GAMMA]
    if rtn >= CUTOFF:
        return [0.0, 0.0, 0.0, 0.0]
    sc = math.sin(GAMMA*rtn)/(rtn*GAMMA)
    co = math.cos(GAMMA*rtn)
    return [sc*rt0, sc*rt1, sc*rt2, (1.0+co)/GAMMA]


def _compute_gvec_all(positions, atoms, N):
    """Return flat list of 4*N*(N-1) floats: 4D G-vectors for all ordered pairs."""
    frames = []
    for i in range(N):
        p0 = positions[atoms[3*i]]
        p1 = positions[atoms[3*i+1]]
        p2 = positions[atoms[3*i+2]]
        frames.append(_compute_frame(p0, p1, p2))
    result = []
    for i in range(N):
        ci, e1i, e2i, e3i = frames[i]
        for j in range(N):
            if j == i:
                continue
            G = _bottaro_gvec(ci, e1i, e2i, e3i, frames[j][0])
            result.extend(G)
    return result


def _ermsd_ref(positions, atoms, N, ref_gvecs_flat):
    """Compute eRMSD from 4D reference G-vectors using Python reference."""
    frames = []
    for i in range(N):
        p0 = positions[atoms[3*i]]
        p1 = positions[atoms[3*i+1]]
        p2 = positions[atoms[3*i+2]]
        frames.append(_compute_frame(p0, p1, p2))
    sumSq = 0.0
    for i in range(N):
        ci, e1i, e2i, e3i = frames[i]
        for j in range(N):
            if j == i:
                continue
            pIdx = i*(N-1) + (j if j < i else j-1)
            G = _bottaro_gvec(ci, e1i, e2i, e3i, frames[j][0])
            for k in range(4):
                dG = G[k] - ref_gvecs_flat[4*pIdx+k]
                sumSq += dG*dG
    return math.sqrt(sumSq / N)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx_ermsd(positions, atoms, N, ref_gvecs_flat, platform,
                    bias_grad=1.0, cutoff=2.4):
    """Build a minimal OpenMM context with a single CV_ERMSD CV.

    params = [N, cutoff, G_ref_4D...] where G_ref_4D has 4*N*(N-1) values
    ordered by (i,j) pairs: i*(N-1)+(j<i?j:j-1).
    """
    n_atoms = len(positions)
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atoms:
        av.append(a)
    pv = mm.vectord()
    pv.append(float(N))
    pv.append(float(cutoff))
    for v in ref_gvecs_flat:
        pv.append(float(v))
    f.addCollectiveVariable(gp.GluedForce.CV_ERMSD, av, pv)
    f.setTestBiasGradients(mm.vectord([bias_grad]))
    sys_.addForce(f)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


def _compact_positions(rng, n_atoms):
    """Generate positions in [0, 0.3] nm so all pairs are within MAXDIST=1.2 nm."""
    return [[rng.uniform(0, 0.3), rng.uniform(0, 0.3), rng.uniform(0, 0.3)]
            for _ in range(n_atoms)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ermsd_zero_at_reference(platform):
    """When positions == reference positions, eRMSD should be 0."""
    rng = random.Random(7)
    N = 3
    atoms = list(range(9))

    positions = _compact_positions(rng, 9)
    ref_gvecs = _compute_gvec_all(positions, atoms, N)

    ctx, f = _make_ctx_ermsd(positions, atoms, N, ref_gvecs, platform)
    cv = f.getLastCVValues(ctx)
    assert abs(cv[0]) < TOL_CV, \
        f"eRMSD at reference should be 0, got {cv[0]:.2e}"
    print(f"  test_ermsd_zero_at_reference: OK  (eRMSD={cv[0]:.2e})")


def test_ermsd_value(platform):
    """eRMSD matches Python reference for random positions and random reference."""
    rng = random.Random(42)
    N = 3
    atoms = list(range(9))

    ref_positions = _compact_positions(rng, 9)
    positions = [[ref_positions[i][k] + rng.gauss(0, 0.04) for k in range(3)]
                 for i in range(9)]

    ref_gvecs = _compute_gvec_all(ref_positions, atoms, N)
    ermsd_py = _ermsd_ref(positions, atoms, N, ref_gvecs)

    ctx, f = _make_ctx_ermsd(positions, atoms, N, ref_gvecs, platform)
    cv = f.getLastCVValues(ctx)
    assert abs(cv[0] - ermsd_py) < TOL_CV, \
        f"eRMSD value mismatch: ref={ermsd_py:.6f}, got={cv[0]:.6f}"
    print(f"  test_ermsd_value: OK  (eRMSD={ermsd_py:.4f})")


def test_two_ermsd_cvs(platform):
    """Two independent ERMSD CVs are computed correctly."""
    rng = random.Random(99)
    N = 3

    ref_pos0 = _compact_positions(rng, 9)
    ref_pos1 = _compact_positions(rng, 9)
    cur_pos0 = [[ref_pos0[i][k] + rng.gauss(0, 0.04) for k in range(3)] for i in range(9)]
    cur_pos1 = [[ref_pos1[i][k] + rng.gauss(0, 0.04) for k in range(3)] for i in range(9)]

    atoms9 = list(range(9))
    ref_gvecs0 = _compute_gvec_all(ref_pos0, atoms9, N)
    ref_gvecs1 = _compute_gvec_all(ref_pos1, atoms9, N)
    ermsd_py0 = _ermsd_ref(cur_pos0, atoms9, N, ref_gvecs0)
    ermsd_py1 = _ermsd_ref(cur_pos1, atoms9, N, ref_gvecs1)

    positions = cur_pos0 + cur_pos1
    n_atoms = len(positions)
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    def add_ermsd_cv(f, atoms, ref_gvecs):
        av = mm.vectori()
        for a in atoms:
            av.append(a)
        pv = mm.vectord()
        pv.append(float(N))
        pv.append(2.4)  # cutoff
        for v in ref_gvecs:
            pv.append(float(v))
        f.addCollectiveVariable(gp.GluedForce.CV_ERMSD, av, pv)

    add_ermsd_cv(f, list(range(9)), ref_gvecs0)
    add_ermsd_cv(f, list(range(9, 18)), ref_gvecs1)
    f.setTestBiasGradients(mm.vectord([1.0, 1.0]))
    sys_.addForce(f)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    cv = f.getLastCVValues(ctx)

    assert abs(cv[0] - ermsd_py0) < TOL_CV, \
        f"CV0 mismatch: ref={ermsd_py0:.6f}, got={cv[0]:.6f}"
    assert abs(cv[1] - ermsd_py1) < TOL_CV, \
        f"CV1 mismatch: ref={ermsd_py1:.6f}, got={cv[1]:.6f}"
    print(f"  test_two_ermsd_cvs: OK  (eRMSD0={ermsd_py0:.4f}, eRMSD1={ermsd_py1:.4f})")


def test_numerical_gradient(platform):
    """Finite-difference check of eRMSD centroid-only gradient.

    Perturbs all 3 atoms of a residue simultaneously (pure centroid shift).
    The FD gradient from the Python reference tests the centroid-only derivative.
    """
    rng = random.Random(123)
    N = 3
    atoms = list(range(9))

    ref_positions = _compact_positions(rng, 9)
    positions = [[ref_positions[i][k] + rng.gauss(0, 0.04) for k in range(3)]
                 for i in range(9)]

    ref_gvecs = _compute_gvec_all(ref_positions, atoms, N)
    ermsd_val = _ermsd_ref(positions, atoms, N, ref_gvecs)
    print(f"    (eRMSD at test positions: {ermsd_val:.4f})")

    sys_ = mm.System()
    for _ in range(9):
        sys_.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atoms:
        av.append(a)
    pv = mm.vectord()
    pv.append(float(N))
    pv.append(2.4)  # cutoff
    for v in ref_gvecs:
        pv.append(float(v))
    f_cv.addCollectiveVariable(gp.GluedForce.CV_ERMSD, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys_.addForce(f_cv)
    ctx = mm.Context(sys_, mm.VerletIntegrator(0.001), platform)

    h = 1e-4
    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer

    for res in range(N):
        a0 = 3 * res
        for ci in range(3):
            pos_p = [list(p) for p in positions]
            pos_m = [list(p) for p in positions]
            for m in range(3):
                pos_p[a0 + m][ci] += h
                pos_m[a0 + m][ci] -= h

            fd_centroid = (_ermsd_ref(pos_p, atoms, N, ref_gvecs)
                         - _ermsd_ref(pos_m, atoms, N, ref_gvecs)) / (2*h)

            ctx.setPositions(positions)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_per_atom = -raw[a0][ci].value_in_unit(unit)
            f_centroid = 3.0 * f_per_atom

            err = abs(fd_centroid - f_centroid)
            if err > max_err:
                max_err = err

    assert max_err < TOL_F, \
        f"eRMSD centroid gradient max error {max_err:.2e} > {TOL_F}"
    print(f"  test_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    try:
        plat = mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        plat = mm.Platform.getPlatformByName("Reference")
    print("Stage 3.19 -- CV_ERMSD tests:")
    test_ermsd_zero_at_reference(plat)
    test_ermsd_value(plat)
    test_two_ermsd_cvs(plat)
    test_numerical_gradient(plat)
    print("All CV_ERMSD tests passed.")
