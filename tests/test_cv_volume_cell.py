"""Stage 3.15+3.16 — CV_VOLUME and CV_CELL acceptance tests.

CV_VOLUME: V = a·(b×c) = boxVecX.x * boxVecY.y * boxVecZ.z (OpenMM reduced form).
CV_CELL:   Cell lengths |a|, |b|, |c| from the box vectors.

Tests: value correctness against OpenMM box vector queries.
Both CVs require a periodic system (periodic=True).
"""

import sys, math
import openmm as mm
import gluedplugin as gp

TOL = 1e-5


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _make_periodic_ctx(box_vecs, cv_specs, platform):
    """
    box_vecs: [(ax,ay,az), (bx,by,bz), (cx,cy,cz)] in nm (OpenMM reduced triclinic).
    cv_specs: list of (cv_type, atoms_list, params_list).
    Returns (ctx, force).
    """
    n_atoms = 2  # need at least 2 atoms
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    sys.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in box_vecs])

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(True)

    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms: av.append(a)
        pv = mm.vectord()
        for p in params: pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    grads = mm.vectord([0.0] * len(cv_specs))
    f.setTestBiasGradients(grads)
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    ctx.getState(getForces=True)
    return ctx, f


def test_volume_cubic(platform):
    """Cubic box: V = L^3."""
    L = 3.0
    box = [(L, 0, 0), (0, L, 0), (0, 0, L)]
    ctx, f = _make_periodic_ctx(box, [(gp.GluedForce.CV_VOLUME, [], [])], platform)
    cvs = f.getLastCVValues(ctx)
    expected = L**3
    assert abs(cvs[0] - expected) < TOL, f"expected {expected:.4f}, got {cvs[0]:.4f}"
    print(f"  test_volume_cubic: OK  (V={cvs[0]:.4f})")


def test_volume_orthorhombic(platform):
    """Orthorhombic box: V = Lx*Ly*Lz."""
    Lx, Ly, Lz = 2.0, 3.0, 4.0
    box = [(Lx, 0, 0), (0, Ly, 0), (0, 0, Lz)]
    ctx, f = _make_periodic_ctx(box, [(gp.GluedForce.CV_VOLUME, [], [])], platform)
    cvs = f.getLastCVValues(ctx)
    expected = Lx * Ly * Lz
    assert abs(cvs[0] - expected) < TOL, f"expected {expected:.4f}, got {cvs[0]:.4f}"
    print(f"  test_volume_orthorhombic: OK  (V={cvs[0]:.4f})")


def test_volume_triclinic(platform):
    """Triclinic box: V = ax * by * cz (OpenMM reduced form)."""
    # OpenMM reduced triclinic: a=(ax,0,0), b=(bx,by,0), c=(cx,cy,cz)
    ax, by, cz = 2.5, 3.1, 4.2
    bx, cx, cy = 0.3, 0.4, 0.5
    box = [(ax, 0, 0), (bx, by, 0), (cx, cy, cz)]
    ctx, f = _make_periodic_ctx(box, [(gp.GluedForce.CV_VOLUME, [], [])], platform)
    cvs = f.getLastCVValues(ctx)
    expected = ax * by * cz  # OpenMM's reduced form: V = ax * by * cz
    assert abs(cvs[0] - expected) < TOL, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_volume_triclinic: OK  (V={cvs[0]:.6f})")


def test_cell_lengths_cubic(platform):
    """Cubic box: all three cell lengths equal L."""
    L = 2.7
    box = [(L, 0, 0), (0, L, 0), (0, 0, L)]
    specs = [(gp.GluedForce.CV_CELL, [], [0]),
             (gp.GluedForce.CV_CELL, [], [1]),
             (gp.GluedForce.CV_CELL, [], [2])]
    ctx, f = _make_periodic_ctx(box, specs, platform)
    cvs = f.getLastCVValues(ctx)
    for comp in range(3):
        assert abs(cvs[comp] - L) < TOL, \
            f"comp={comp}: expected {L:.4f}, got {cvs[comp]:.4f}"
    print(f"  test_cell_lengths_cubic: OK  (a={cvs[0]:.4f}, b={cvs[1]:.4f}, c={cvs[2]:.4f})")


def test_cell_length_b_triclinic(platform):
    """|b| = sqrt(bx^2 + by^2) for triclinic box."""
    bx, by = 0.4, 3.0
    box = [(2.5, 0, 0), (bx, by, 0), (0.2, 0.3, 4.0)]
    ctx, f = _make_periodic_ctx(box, [(gp.GluedForce.CV_CELL, [], [1])], platform)
    cvs = f.getLastCVValues(ctx)
    expected = math.sqrt(bx**2 + by**2)
    assert abs(cvs[0] - expected) < TOL, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_cell_length_b_triclinic: OK  (|b|={cvs[0]:.6f})")


def test_cell_length_c_triclinic(platform):
    """|c| = sqrt(cx^2 + cy^2 + cz^2) for triclinic box."""
    cx, cy, cz = 0.3, 0.5, 3.5
    box = [(2.5, 0, 0), (0.2, 3.0, 0), (cx, cy, cz)]
    ctx, f = _make_periodic_ctx(box, [(gp.GluedForce.CV_CELL, [], [2])], platform)
    cvs = f.getLastCVValues(ctx)
    expected = math.sqrt(cx**2 + cy**2 + cz**2)
    assert abs(cvs[0] - expected) < TOL, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_cell_length_c_triclinic: OK  (|c|={cvs[0]:.6f})")


def test_volume_and_cell_together(platform):
    """Volume and two cell CVs in a single force object."""
    L = 4.0
    box = [(L, 0, 0), (0, L, 0), (0, 0, L)]
    specs = [(gp.GluedForce.CV_VOLUME, [], []),
             (gp.GluedForce.CV_CELL,   [], [0]),
             (gp.GluedForce.CV_CELL,   [], [2])]
    ctx, f = _make_periodic_ctx(box, specs, platform)
    cvs = f.getLastCVValues(ctx)
    assert abs(cvs[0] - L**3) < TOL, f"volume mismatch: {cvs[0]}"
    assert abs(cvs[1] - L) < TOL, f"|a| mismatch: {cvs[1]}"
    assert abs(cvs[2] - L) < TOL, f"|c| mismatch: {cvs[2]}"
    print(f"  test_volume_and_cell_together: OK  (V={cvs[0]:.4f}, a={cvs[1]:.4f}, c={cvs[2]:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_VOLUME/CV_CELL tests.")
        sys.exit(0)
    print("Stage 3.15+3.16 — CV_VOLUME / CV_CELL tests (CUDA platform):")
    test_volume_cubic(plat)
    test_volume_orthorhombic(plat)
    test_volume_triclinic(plat)
    test_cell_lengths_cubic(plat)
    test_cell_length_b_triclinic(plat)
    test_cell_length_c_triclinic(plat)
    test_volume_and_cell_together(plat)
    print("All CV_VOLUME / CV_CELL tests passed.")
