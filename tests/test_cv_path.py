"""Stage 3.6 acceptance tests — path CVs (s and z)."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_CV    = 1e-4
TOL_FORCE = 1e-3


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_context(positions_nm, cv_specs, bias_gradients=None, platform=None):
    """cv_specs: list of (cv_type, atoms_list, params_list).
    PATH cv produces 2 entries in getLastCVValues: [s, z]."""
    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)

    f = gp.GluedForce()
    for cv_type, atoms, params in cv_specs:
        av = mm.vectori()
        for a in atoms:
            av.append(a)
        pv = mm.vectord()
        for p in params:
            pv.append(p)
        f.addCollectiveVariable(cv_type, av, pv)

    if bias_gradients is not None:
        gv = mm.vectord()
        for g in bias_gradients:
            gv.append(g)
        f.setTestBiasGradients(gv)

    sys_.addForce(f)
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f


def eval_cv(ctx, f):
    ctx.getState(getForces=True)
    return list(f.getLastCVValues(ctx))


def get_forces(ctx):
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def build_path_params(lambda_, frames):
    """Encode lambda, N, then flat reference coords."""
    params = [lambda_, float(len(frames))]
    for frame in frames:
        for atom_coords in frame:
            params.extend(atom_coords)
    return params


# ---------------------------------------------------------------------------
# Path CV tests
# ---------------------------------------------------------------------------

def test_path_midpoint(platform):
    """1-atom path, 2 frames at (0,0,0) and (2,0,0). Atom at midpoint (1,0,0).
    RMSD0=RMSD1=1, w0=w1=exp(-λ). s=1.5 (1-indexed), z=1−ln(2)/λ at λ=1."""
    lam = 1.0
    frames = [[(0.0, 0.0, 0.0)],   # frame 1 (1-indexed)
              [(2.0, 0.0, 0.0)]]   # frame 2 (1-indexed)
    pos = [(1.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0], params)]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 2, f"Expected 2 CV values, got {len(cvs)}"
    # s = (1*w + 2*w)/(w+w) = 1.5 (1-indexed, matching PLUMED)
    assert abs(cvs[0] - 1.5) < TOL_CV, f"s={cvs[0]:.8f}, expected 1.5"
    # z = 1 - ln(2)/1 = 1 - 0.6931... ≈ 0.3069
    expected_z = 1.0 - math.log(2.0) / lam
    assert abs(cvs[1] - expected_z) < TOL_CV, f"z={cvs[1]:.8f}, expected {expected_z:.6f}"
    print(f"  test_path_midpoint: OK  (s={cvs[0]:.6f}, z={cvs[1]:.6f})")


def test_path_at_frame0(platform):
    """Atom exactly at frame 1 (1-indexed) → s~1, z~−1/λ*ln(1+exp(−λ·R²))."""
    lam = 2.0
    d = 2.0   # distance between frames
    frames = [[(0.0, 0.0, 0.0)],
              [(d,   0.0, 0.0)]]
    pos = [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0], params)]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    # RMSD0=0 → w0=1; RMSD1=d=2 → w1=exp(-lam*d^2)=exp(-8)≈3.35e-4
    w1 = math.exp(-lam * d * d)
    S = 1.0 + w1
    s_expected = (1 * 1.0 + 2 * w1) / S   # 1-indexed: k=1 for frame0, k=2 for frame1
    z_expected = -math.log(S) / lam
    assert abs(cvs[0] - s_expected) < TOL_CV, f"s={cvs[0]:.8f}, expected {s_expected:.8f}"
    assert abs(cvs[1] - z_expected) < TOL_CV, f"z={cvs[1]:.8f}, expected {z_expected:.8f}"
    print(f"  test_path_at_frame0: OK  (s={cvs[0]:.6f}, z={cvs[1]:.6f})")


def test_path_three_frames(platform):
    """3 frames at x=0,1,2; atom at x=1 → middle frame (1-indexed k=2).
    w_side=exp(-λ), w_mid=1. s=(1*w_side+2*1+3*w_side)/(1+2*w_side), z=-ln(1+2*exp(-λ))/λ."""
    lam = 1.0
    frames = [[(0.0, 0.0, 0.0)],
              [(1.0, 0.0, 0.0)],
              [(2.0, 0.0, 0.0)]]
    pos = [(1.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0], params)]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    w_side = math.exp(-lam * 1.0)
    S = w_side + 1.0 + w_side
    s_expected = (1*w_side + 2*1.0 + 3*w_side) / S   # 1-indexed: k=1,2,3
    z_expected = -math.log(S) / lam
    assert abs(cvs[0] - s_expected) < TOL_CV, f"s={cvs[0]:.6f}, expected {s_expected:.6f}"
    assert abs(cvs[1] - z_expected) < TOL_CV, f"z={cvs[1]:.6f}, expected {z_expected:.6f}"
    print(f"  test_path_three_frames: OK  (s={cvs[0]:.6f}, z={cvs[1]:.6f})")


def test_path_force_sum_s(platform):
    """Force sum is zero when all frame COMs equal current COM (symmetric config).
    2 atoms at (±0.5,0,0); frame0 = same; frame1 = (±1.5,0,0).
    Every (current−ref) for atom0 and atom1 cancel by symmetry."""
    lam = 1.0
    frames = [
        [(-0.5, 0.0, 0.0), ( 0.5, 0.0, 0.0)],   # frame 0: COM at origin
        [(-1.5, 0.0, 0.0), ( 1.5, 0.0, 0.0)],    # frame 1: COM at origin
    ]
    pos = [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0, 1], params)]
    ctx, f = make_context(pos, specs, bias_gradients=[1.0, 0.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(2))
        assert abs(s) < TOL_FORCE, f"force_s sum [{c}] = {s:.6f}"
    print("  test_path_force_sum_s: OK  (COM-aligned config, Newton's 3rd law)")


def test_path_force_sum_z(platform):
    """Same symmetric COM config — force sum for z must also be zero."""
    lam = 1.0
    frames = [
        [(-0.5, 0.0, 0.0), ( 0.5, 0.0, 0.0)],
        [(-1.5, 0.0, 0.0), ( 1.5, 0.0, 0.0)],
    ]
    pos = [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (5.0, 5.0, 5.0)]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0, 1], params)]
    ctx, f = make_context(pos, specs, bias_gradients=[0.0, 1.0], platform=platform)
    forces = get_forces(ctx)
    for c in range(3):
        s = sum(forces[i][c] for i in range(2))
        assert abs(s) < TOL_FORCE, f"force_z sum [{c}] = {s:.6f}"
    print("  test_path_force_sum_z: OK  (COM-aligned config, Newton's 3rd law)")


def test_path_numerical_derivative_s(platform):
    """FD check of ds/d(atom0.x) for 1-atom 2-frame path (1-indexed frames).
    At midpoint x=1 between frames at 0 and 2, λ=1, w0=w1=exp(-1):
    s = (w0+2*w1)/(w0+w1); ds/dx = 1 at x=1 (d_den/dx=0 by symmetry)."""
    dx = 1e-3
    lam = 1.0
    frames = [[(0.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)]]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0], params)]

    base = [(1.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    pos_p = list(base); pos_p[0] = (1.0 + dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    s_p = eval_cv(ctx_p, f_p)[0]

    pos_m = list(base); pos_m[0] = (1.0 - dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    s_m = eval_cv(ctx_m, f_m)[0]

    jac_num = (s_p - s_m) / (2.0 * dx)
    # Analytical: ds/dx = 1.0 at the midpoint
    jac_ana = 1.0
    assert abs(jac_num - jac_ana) < 2e-3, \
        f"ds/dx jac_num={jac_num:.6f}, expected {jac_ana:.6f}"
    print(f"  test_path_numerical_derivative_s: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_path_numerical_derivative_z(platform):
    """FD check of dz/d(atom0.x) at midpoint.
    dz/dx = (2/(M·S)) · Σᵢ wᵢ·(x−xᵢ)
    At x=1, frame0 at 0, frame1 at 2: (x-x0)=1, (x-x1)=-1, w0=w1=e^-1
    dz/dx = 2/1 * (e^-1*1 + e^-1*(-1)) / (2*e^-1) = 0 (symmetric)."""
    dx = 1e-3
    lam = 1.0
    frames = [[(0.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)]]
    params = build_path_params(lam, frames)
    specs = [(gp.GluedForce.CV_PATH, [0], params)]

    base = [(1.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    pos_p = list(base); pos_p[0] = (1.0 + dx, 0.0, 0.0)
    ctx_p, f_p = make_context(pos_p, specs, platform=platform)
    z_p = eval_cv(ctx_p, f_p)[1]

    pos_m = list(base); pos_m[0] = (1.0 - dx, 0.0, 0.0)
    ctx_m, f_m = make_context(pos_m, specs, platform=platform)
    z_m = eval_cv(ctx_m, f_m)[1]

    jac_num = (z_p - z_m) / (2.0 * dx)
    jac_ana = 0.0
    assert abs(jac_num - jac_ana) < 2e-3, \
        f"dz/dx jac_num={jac_num:.6f}, expected {jac_ana:.6f}"
    print(f"  test_path_numerical_derivative_z: OK  "
          f"(jac_num={jac_num:.6f}, jac_ana={jac_ana:.6f})")


def test_path_mixed_with_distance(platform):
    """PATH (2 values) + DISTANCE (1 value) in same Force → 3 total CV values."""
    lam = 1.0
    frames = [[(0.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)]]
    pos = [(1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    path_params = build_path_params(lam, frames)
    specs = [
        (gp.GluedForce.CV_PATH,     [0], path_params),
        (gp.GluedForce.CV_DISTANCE, [0, 1], []),
    ]
    ctx, f = make_context(pos, specs, platform=platform)
    cvs = eval_cv(ctx, f)
    assert len(cvs) == 3, f"Expected 3 CVs, got {len(cvs)}"
    assert abs(cvs[0] - 1.5) < TOL_CV, f"s={cvs[0]:.6f}"  # 1-indexed: midpoint = 1.5
    assert abs(cvs[2] - 2.0) < TOL_CV, f"dist={cvs[2]:.6f}"
    print(f"  test_path_mixed_with_distance: OK  "
          f"(s={cvs[0]:.4f}, z={cvs[1]:.4f}, dist={cvs[2]:.4f})")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping path CV tests.")
        sys.exit(0)

    print("Stage 3.6 path CV tests (CUDA platform):")
    test_path_midpoint(plat)
    test_path_at_frame0(plat)
    test_path_three_frames(plat)
    test_path_force_sum_s(plat)
    test_path_force_sum_z(plat)
    test_path_numerical_derivative_s(plat)
    test_path_numerical_derivative_z(plat)
    test_path_mixed_with_distance(plat)
    print("All path CV tests passed.")
