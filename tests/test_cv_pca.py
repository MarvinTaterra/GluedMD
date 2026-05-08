"""Stage 3.18 — CV_PCA acceptance tests.

CV = dot(r_flat - mean_flat, eigvec_flat)
   = Σ_i dot(r_i - ref_i, ev_i)

API:
  atoms  = [a0, ..., aN-1]
  params = [mean_x0, mean_y0, mean_z0, ..., ev_x0, ev_y0, ev_z0, ...]
           (3N reference mean values followed by 3N eigenvector values)

Tests: CV value, force direction, orthogonality of two PCs, numerical gradient.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV = 1e-5
TOL_F  = 1e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _pca_cv(positions, atoms, ref, eigvec):
    """Python reference: CV = dot(r_flat - ref_flat, eigvec_flat)."""
    cv = 0.0
    for i, a in enumerate(atoms):
        for k in range(3):
            cv += (positions[a][k] - ref[3*i + k]) * eigvec[3*i + k]
    return cv


def _make_ctx(positions, atom_list, ref, eigvec, platform):
    n_atoms = len(positions)
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atom_list: av.append(a)
    pv = mm.vectord()
    for v in ref: pv.append(v)
    for v in eigvec: pv.append(v)
    f.addCollectiveVariable(gp.GluedForce.CV_PCA, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)
    return ctx, f


def test_pca_at_reference(platform):
    """CV = 0 when positions equal the reference mean."""
    n_atoms = 4
    positions = [[0.1*i, 0.2*i, 0.3*i] for i in range(n_atoms)]
    atoms = list(range(n_atoms))
    ref = [v for p in positions for v in p]  # same as positions
    ev = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]  # arbitrary
    ctx, f = _make_ctx(positions, atoms, ref, ev, platform)
    cvs = f.getLastCVValues(ctx)
    assert abs(cvs[0]) < TOL_CV, f"expected CV=0 at reference, got {cvs[0]}"
    print(f"  test_pca_at_reference: OK  (cv={cvs[0]:.2e})")


def test_pca_single_atom_x(platform):
    """Single atom displaced along x; eigvec=(1,0,0) → CV = displacement."""
    positions = [[0.7, 0.3, 0.1]] + [[0.0]*3]*3
    atoms = [0]
    ref = [0.5, 0.3, 0.1]    # reference at x=0.5
    ev = [1.0, 0.0, 0.0]     # eigvec along x
    ctx, f = _make_ctx(positions, atoms, ref, ev, platform)
    cvs = f.getLastCVValues(ctx)
    expected = 0.7 - 0.5      # displacement projected on x
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.4f}, got {cvs[0]:.6f}"
    print(f"  test_pca_single_atom_x: OK  (cv={cvs[0]:.6f})")


def test_pca_value(platform):
    """CV matches Python reference for multi-atom system."""
    rng = random.Random(42)
    n_atoms = 6
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(n_atoms)]
    atoms = [0, 1, 2, 3]
    N = len(atoms)
    ref = [rng.uniform(-0.5, 0.5) for _ in range(3*N)]
    # Normalized eigvec
    raw_ev = [rng.uniform(-1, 1) for _ in range(3*N)]
    norm = math.sqrt(sum(v**2 for v in raw_ev))
    eigvec = [v/norm for v in raw_ev]

    ctx, f = _make_ctx(positions, atoms, ref, eigvec, platform)
    cvs = f.getLastCVValues(ctx)
    expected = _pca_cv(positions, atoms, ref, eigvec)
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_pca_value: OK  (cv={cvs[0]:.6f}, ref={expected:.6f})")


def test_pca_force(platform):
    """Force on atom i component k = -ev[3i+k] (bias_grad=1)."""
    positions = [[0.3, 0.1, 0.2], [0.5, 0.4, 0.6]] + [[0.0]*3]*2
    atoms = [0, 1]
    ref = [0.0]*6
    eigvec = [0.6, 0.8, 0.0, -0.8, 0.6, 0.0]  # two 2D unit vectors

    ctx, f = _make_ctx(positions, atoms, ref, eigvec, platform)
    state = ctx.getState(getForces=True)
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    raw = state.getForces(asNumpy=False)

    for ai, a in enumerate(atoms):
        for k in range(3):
            expected_f = -eigvec[3*ai + k]
            got_f = raw[a][k].value_in_unit(unit)
            assert abs(got_f - expected_f) < TOL_F, \
                f"atom {a} comp {k}: expected {expected_f:.4f}, got {got_f:.4f}"
    print("  test_pca_force: OK")


def test_pca_two_cvs_orthogonal(platform):
    """Two orthogonal PCs: projections are independent."""
    rng = random.Random(7)
    n_atoms = 4
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(n_atoms)]
    atoms = list(range(n_atoms))
    N = len(atoms)
    ref = [0.5]*( 3*N)

    # Build two orthogonal eigenvectors via Gram-Schmidt
    raw1 = [rng.uniform(-1,1) for _ in range(3*N)]
    l1 = math.sqrt(sum(v**2 for v in raw1))
    ev1 = [v/l1 for v in raw1]

    raw2 = [rng.uniform(-1,1) for _ in range(3*N)]
    dot = sum(raw2[i]*ev1[i] for i in range(3*N))
    raw2 = [raw2[i] - dot*ev1[i] for i in range(3*N)]
    l2 = math.sqrt(sum(v**2 for v in raw2))
    ev2 = [v/l2 for v in raw2]

    # Build system with two PCA CVs
    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    force = gp.GluedForce()
    force.setUsesPeriodicBoundaryConditions(False)

    av = mm.vectori()
    for a in atoms: av.append(a)
    pv1 = mm.vectord()
    for v in ref: pv1.append(v)
    for v in ev1: pv1.append(v)
    pv2 = mm.vectord()
    for v in ref: pv2.append(v)
    for v in ev2: pv2.append(v)
    force.addCollectiveVariable(gp.GluedForce.CV_PCA, av, pv1)
    force.addCollectiveVariable(gp.GluedForce.CV_PCA, av, pv2)
    force.setTestBiasGradients(mm.vectord([0.0, 0.0]))
    sys.addForce(force)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)

    cvs = force.getLastCVValues(ctx)
    ref1 = _pca_cv(positions, atoms, ref, ev1)
    ref2 = _pca_cv(positions, atoms, ref, ev2)
    assert abs(cvs[0] - ref1) < TOL_CV, f"CV0 mismatch: {cvs[0]} vs {ref1}"
    assert abs(cvs[1] - ref2) < TOL_CV, f"CV1 mismatch: {cvs[1]} vs {ref2}"
    print(f"  test_pca_two_cvs_orthogonal: OK  (cv0={cvs[0]:.4f}, cv1={cvs[1]:.4f})")


def test_pca_numerical_gradient(platform):
    """Finite-difference check of PCA Jacobian."""
    rng = random.Random(99)
    n_atoms = 5
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(n_atoms)]
    atoms = [0, 1, 2, 3]
    N = len(atoms)
    ref = [rng.uniform(-0.5, 0.5) for _ in range(3*N)]
    raw_ev = [rng.uniform(-1, 1) for _ in range(3*N)]
    norm = math.sqrt(sum(v**2 for v in raw_ev))
    eigvec = [v/norm for v in raw_ev]

    sys = mm.System()
    for _ in range(n_atoms): sys.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atoms: av.append(a)
    pv = mm.vectord()
    for v in ref: pv.append(v)
    for v in eigvec: pv.append(v)
    f_cv.addCollectiveVariable(gp.GluedForce.CV_PCA, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f_cv)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

    h = 1e-4
    max_err = 0.0
    unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
    for ai in range(n_atoms):
        for ci in range(3):
            pos_p = [list(p) for p in positions]
            pos_m = [list(p) for p in positions]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            fd = (_pca_cv(pos_p, atoms, ref, eigvec) -
                   _pca_cv(pos_m, atoms, ref, eigvec)) / (2*h)
            ctx.setPositions(positions)
            state = ctx.getState(getForces=True)
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err

    assert max_err < TOL_F, f"max gradient error {max_err:.2e} > {TOL_F}"
    print(f"  test_pca_numerical_gradient: OK  (max_err={max_err:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_PCA tests.")
        sys.exit(0)
    print("Stage 3.18 — CV_PCA tests (CUDA platform):")
    test_pca_at_reference(plat)
    test_pca_single_atom_x(plat)
    test_pca_value(plat)
    test_pca_force(plat)
    test_pca_two_cvs_orthogonal(plat)
    test_pca_numerical_gradient(plat)
    print("All CV_PCA tests passed.")
