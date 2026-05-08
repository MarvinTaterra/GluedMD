"""Stage 3.11 — CV_DRMSD acceptance tests.

DRMSD = sqrt(1/N * sum_p (d_p - d0_p)^2)

Tests correctness of CV value and numerical gradient against
a finite-difference reference.
"""

import sys, math, random
import openmm as mm
import gluedplugin as gp

TOL_CV  = 1e-5   # nm
TOL_JAC = 5e-4   # finite-difference tolerance (kJ/mol/nm vs analytical)


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except mm.OpenMMException:
        return None


def _make_ctx(positions, atoms, ref_dists, platform, periodic=False):
    n = len(positions)
    sys = mm.System()
    for _ in range(n):
        sys.addParticle(12.0)
    if periodic:
        sys.setDefaultPeriodicBoxVectors([5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0])

    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(periodic)

    av = mm.vectori()
    for a in atoms:
        av.append(a)
    pv = mm.vectord()
    for d in ref_dists:
        pv.append(d)

    f.addCollectiveVariable(gp.GluedForce.CV_DRMSD, av, pv)
    f.setTestBiasGradients(mm.vectord([1.0]))

    sys.addForce(f)
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys, integrator, platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)   # trigger first CV evaluation
    return ctx, f


def _drmsd_cpu(positions, pairs, ref_dists):
    """Reference Python DRMSD calculation."""
    import numpy as np
    pos = [list(p) for p in positions]
    sq = 0.0
    for (a, b), d0 in zip(pairs, ref_dists):
        dx = pos[b][0] - pos[a][0]
        dy = pos[b][1] - pos[a][1]
        dz = pos[b][2] - pos[a][2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        sq += (d - d0)**2
    return math.sqrt(sq / len(ref_dists))


def test_drmsd_single_pair(platform):
    """DRMSD with 1 pair = |d - d0|."""
    positions = [[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]]
    d_actual = math.sqrt(0.3**2 + 0.4**2)  # 0.5 nm
    d0 = 0.3
    ctx, f = _make_ctx(positions, [0, 1], [d0], platform)
    cvs = f.getLastCVValues(ctx)
    expected = abs(d_actual - d0)
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_drmsd_single_pair: OK  (cv={cvs[0]:.6f}, ref={expected:.6f})")


def test_drmsd_three_pairs(platform):
    """Three-pair DRMSD matches Python reference."""
    rng = random.Random(42)
    positions = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(6)]
    pairs = [(0, 1), (2, 3), (4, 5)]
    ref_dists = [0.4, 0.6, 0.5]
    atoms = [a for p in pairs for a in p]

    ctx, f = _make_ctx(positions, atoms, ref_dists, platform)
    cvs = f.getLastCVValues(ctx)
    expected = _drmsd_cpu(positions, pairs, ref_dists)
    assert abs(cvs[0] - expected) < TOL_CV, f"expected {expected:.6f}, got {cvs[0]:.6f}"
    print(f"  test_drmsd_three_pairs: OK  (cv={cvs[0]:.6f}, ref={expected:.6f})")


def test_drmsd_at_reference(platform):
    """DRMSD is exactly 0 when all distances match the reference."""
    positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.6, 0.0]]
    pairs = [(0, 1), (2, 3)]
    ref_dists = [0.5, 0.6]
    atoms = [a for p in pairs for a in p]
    ctx, f = _make_ctx(positions, atoms, ref_dists, platform)
    cvs = f.getLastCVValues(ctx)
    assert cvs[0] < 1e-6, f"expected ~0, got {cvs[0]}"
    print(f"  test_drmsd_at_reference: OK  (cv={cvs[0]:.2e})")


def test_drmsd_numerical_gradient(platform):
    """Finite-difference verification of DRMSD Jacobian (forces)."""
    rng = random.Random(7)
    positions = [[rng.uniform(0, 1) for _ in range(3)] for _ in range(4)]
    pairs = [(0, 1), (0, 2), (1, 3)]
    ref_dists = [0.35, 0.45, 0.55]
    atoms = [a for p in pairs for a in p]

    h = 1e-4  # nm

    n = len(positions)
    sys = mm.System()
    for _ in range(n):
        sys.addParticle(12.0)
    f_cv = gp.GluedForce()
    f_cv.setUsesPeriodicBoundaryConditions(False)
    av = mm.vectori()
    for a in atoms: av.append(a)
    pv = mm.vectord()
    for d in ref_dists: pv.append(d)
    f_cv.addCollectiveVariable(gp.GluedForce.CV_DRMSD, av, pv)
    f_cv.setTestBiasGradients(mm.vectord([1.0]))
    sys.addForce(f_cv)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)

    max_err = 0.0
    for ai in range(n):
        for ci in range(3):
            pos_p = [list(p) for p in positions]
            pos_m = [list(p) for p in positions]
            pos_p[ai][ci] += h
            pos_m[ai][ci] -= h
            cv_p = _drmsd_cpu(pos_p, pairs, ref_dists)
            cv_m = _drmsd_cpu(pos_m, pairs, ref_dists)
            fd = (cv_p - cv_m) / (2*h)   # dCV/dr_{ai,ci}

            ctx.setPositions(positions)
            state = ctx.getState(getForces=True)
            unit = mm.unit.kilojoules_per_mole / mm.unit.nanometer
            raw = state.getForces(asNumpy=False)
            f_anal = -raw[ai][ci].value_in_unit(unit)  # F = -dU/dr = -1 * dCV/dr
            err = abs(fd - f_anal)
            if err > max_err:
                max_err = err
    assert max_err < TOL_JAC, f"max gradient error {max_err:.2e} > {TOL_JAC}"
    print(f"  test_drmsd_numerical_gradient: OK  (max_err={max_err:.2e})")


def test_two_drmsd_cvs(platform):
    """Two independent DRMSD CVs computed in a single call."""
    positions = [[0.0]*3, [0.5, 0.0, 0.0],
                 [0.0]*3, [0.0, 0.7, 0.0]]
    sys = mm.System()
    for _ in range(4):
        sys.addParticle(12.0)
    f = gp.GluedForce()
    f.setUsesPeriodicBoundaryConditions(False)

    av0 = mm.vectori(); av0.append(0); av0.append(1)
    pv0 = mm.vectord(); pv0.append(0.3)   # ref = 0.3; actual = 0.5
    f.addCollectiveVariable(gp.GluedForce.CV_DRMSD, av0, pv0)

    av1 = mm.vectori(); av1.append(2); av1.append(3)
    pv1 = mm.vectord(); pv1.append(0.7)   # at reference → drmsd = 0
    f.addCollectiveVariable(gp.GluedForce.CV_DRMSD, av1, pv1)

    f.setTestBiasGradients(mm.vectord([0.0, 0.0]))
    sys.addForce(f)
    ctx = mm.Context(sys, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    ctx.getState(getForces=True)

    cvs = f.getLastCVValues(ctx)
    assert abs(cvs[0] - 0.2) < TOL_CV, f"cv0={cvs[0]}"
    assert cvs[1] < 1e-6, f"cv1={cvs[1]}"
    print(f"  test_two_drmsd_cvs: OK  (cv0={cvs[0]:.4f}, cv1={cvs[1]:.2e})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping CV_DRMSD tests.")
        sys.exit(0)
    print("Stage 3.11 — CV_DRMSD tests (CUDA platform):")
    test_drmsd_single_pair(plat)
    test_drmsd_three_pairs(plat)
    test_drmsd_at_reference(plat)
    test_drmsd_numerical_gradient(plat)
    test_two_drmsd_cvs(plat)
    print("All CV_DRMSD tests passed.")
    print("All CV_DRMSD tests passed.")
