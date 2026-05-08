"""Stage 2.1 + 2.2 acceptance tests — GPU position reads, force writes, PBC geometry."""
import sys
import math
import openmm as mm
import gluedplugin as gp

CUDA_PLATFORM = "CUDA"
TOL_FORCE = 1e-9   # fixed-point quantization ~2.3e-10 kJ/mol/nm
TOL_DIST  = 1e-5   # nm


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_context(n_atoms, platform, pbc=False, box_nm=None,
                 test_mode=0, test_scale=0.0):
    """Create a minimal Context with n_atoms and GluedForce.
    test_mode/test_scale must be set here (before Context creation) because
    the kernel copies them from the Force at initialize() time."""
    sys_ = mm.System()
    for _ in range(n_atoms):
        sys_.addParticle(1.0)
    if pbc and box_nm is not None:
        a = mm.Vec3(box_nm, 0, 0)
        b = mm.Vec3(0, box_nm, 0)
        c = mm.Vec3(0, 0, box_nm)
        sys_.setDefaultPeriodicBoxVectors(a, b, c)
    f = gp.GluedForce()
    if pbc:
        f.setUsesPeriodicBoundaryConditions(True)
    if test_mode != 0:
        f.setTestForce(test_mode, test_scale)
    sys_.addForce(f)
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, platform)
    return ctx, f


def set_positions(ctx, positions_nm):
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])


def get_forces(ctx):
    """Return forces as list of (fx, fy, fz) in kJ/mol/nm."""
    state = ctx.getState(getForces=True)
    raw = state.getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit), v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def test_constant_force(platform):
    """mode=1: all atoms should receive force (1.0, 0.0, 0.0) kJ/mol/nm."""
    n = 10
    ctx, f = make_context(n, platform, test_mode=1, test_scale=1.0)
    positions = [(i * 0.1, 0.0, 0.0) for i in range(n)]
    set_positions(ctx, positions)
    forces = get_forces(ctx)
    for i, (fx, fy, fz) in enumerate(forces):
        assert abs(fx - 1.0) < TOL_FORCE, f"atom {i}: fx={fx}, expected 1.0"
        assert abs(fy) < TOL_FORCE, f"atom {i}: fy={fy}, expected 0.0"
        assert abs(fz) < TOL_FORCE, f"atom {i}: fz={fz}, expected 0.0"
    print(f"  mode=1 constant force: OK (n={n})")


def test_per_atom_force(platform):
    """mode=2: atom i should receive force (i, 0, 0) — tests atom reordering."""
    n = 20
    ctx, f = make_context(n, platform, test_mode=2, test_scale=1.0)
    positions = [(i * 0.1, 0.0, 0.0) for i in range(n)]
    set_positions(ctx, positions)
    forces = get_forces(ctx)
    for i, (fx, fy, fz) in enumerate(forces):
        assert abs(fx - float(i)) < TOL_FORCE, \
            f"atom {i}: fx={fx}, expected {float(i)}"
        assert abs(fy) < TOL_FORCE
        assert abs(fz) < TOL_FORCE
    # Specifically verify atom 7
    fx7 = forces[7][0]
    assert abs(fx7 - 7.0) < TOL_FORCE, f"atom 7: fx={fx7}, expected 7.0"
    print(f"  mode=2 per-atom force: OK (atom 7 fx={fx7:.6f})")


def test_pbc_distance(platform):
    """mode=3: atoms 0 at (0,0,0), atom 1 at (0.99,0,0) in 1 nm box.
    Minimum-image distance = 0.01 nm.  Written as force.x of atom 0."""
    box_nm = 1.0
    n = 4
    ctx, f = make_context(n, platform, pbc=True, box_nm=box_nm,
                          test_mode=3, test_scale=0.0)
    positions = [(0.0, 0.0, 0.0),
                 (0.99, 0.0, 0.0),
                 (0.5, 0.5, 0.5),
                 (0.2, 0.3, 0.4)]
    set_positions(ctx, positions)
    forces = get_forces(ctx)
    # Atom 0 force.x encodes the PBC distance
    dist_gpu = forces[0][0]
    dist_expected = 0.01  # nm, but force buffer is in nm-based units
    assert abs(dist_gpu - dist_expected) < TOL_DIST, \
        f"PBC distance: got {dist_gpu:.8f}, expected {dist_expected:.8f}"
    print(f"  mode=3 PBC distance: OK (dist={dist_gpu:.8f} nm, expected 0.01)")


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA platform not available — skipping GPU I/O tests.")
        sys.exit(0)

    print("Stage 2.1 + 2.2 GPU I/O tests (CUDA platform):")
    test_constant_force(plat)
    test_per_atom_force(plat)
    test_pbc_distance(plat)
    print("All GPU I/O tests passed.")
