"""Stage 3.8 acceptance tests — PyTorch (TorchScript) CVs.

Test 1: Trivial model — returns x-coordinate of atom 0 (x[0,0]).
         Value = position[0][0]; gradient = [1,0,0] on atom 0, zeros elsewhere.

Test 2: Squared L2 norm — sum of squared coordinates of a single atom.
         Value = x^2 + y^2 + z^2; gradient = 2*[x,y,z].

Test 3: Sum of pairwise distances — compare GLUED result against direct torch eval.

Test 4: FD force check — numerical derivative matches forces from scatter kernel.

All tests create TorchScript models inline with torch.jit.trace / torch.jit.script,
save them to a temp file, and pass the path to addPyTorchCV.
"""
import sys
import math
import os
import tempfile

CUDA_PLATFORM = "CUDA"
TOL = 1e-4

def get_cuda_platform():
    try:
        import openmm as mm
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except Exception:
        return None

def has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False

def make_ctx_pytorch(positions_nm, model_path, atom_indices, bias_gradient,
                     extra_cv_specs=None, extra_bias_grads=None, platform=None):
    """Create a context with a single PyTorch CV (plus optional extras for FD tests).

    Parameters
    ----------
    positions_nm : list of (x,y,z) tuples
    model_path   : path to saved TorchScript model
    atom_indices : list of int user atom indices for the PyTorch CV
    bias_gradient: float (dU/dCV for the PyTorch CV)
    extra_cv_specs  : list of (type, atoms, params) for additional CVs
    extra_bias_grads: list of floats for extra CV bias gradients
    platform     : OpenMM Platform
    """
    import openmm as mm
    import gluedplugin as gp

    n = len(positions_nm)
    sys_ = mm.System()
    for _ in range(n):
        sys_.addParticle(1.0)

    f = gp.GluedForce()

    # Add the PyTorch CV first
    av = mm.vectori()
    for a in atom_indices:
        av.append(a)
    pv = mm.vectord()  # no extra parameters
    pt_cv_idx = f.addPyTorchCV(model_path, av, pv)

    bias_grads = [bias_gradient]

    # Optional extra CVs
    if extra_cv_specs:
        for cvtype, atoms, params in extra_cv_specs:
            av2 = mm.vectori()
            for a in atoms:
                av2.append(a)
            pv2 = mm.vectord()
            for p in params:
                pv2.append(p)
            f.addCollectiveVariable(cvtype, av2, pv2)
        if extra_bias_grads:
            bias_grads.extend(extra_bias_grads)

    gv = mm.vectord()
    for g in bias_grads:
        gv.append(g)
    f.setTestBiasGradients(gv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(*p) for p in positions_nm])
    return ctx, f, pt_cv_idx


def get_forces(ctx):
    raw = ctx.getState(getForces=True).getForces(asNumpy=False)
    unit = raw[0].unit
    return [(v[0].value_in_unit(unit),
             v[1].value_in_unit(unit),
             v[2].value_in_unit(unit)) for v in raw]


def get_cv_values(ctx, f):
    import openmm as mm
    return f.getLastCVValues(ctx)


# ---------------------------------------------------------------------------
# Test 1: Trivial model — CV = x-coordinate of atom 0
# ---------------------------------------------------------------------------

def test_trivial_x_coordinate(platform):
    """CV = x-coordinate of atom 0.
    Value must equal position[0][0].
    Gradient wrt atom 0 = [1, 0, 0], all others zero.
    With bias_gradient=1.0, force on atom 0 = [−1, 0, 0]."""
    import torch

    class XCoordModel(torch.nn.Module):
        def forward(self, x):
            # x: [N, 3] float32
            return x[0, 0]

    model = torch.jit.script(XCoordModel())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_:
        path = f_.name
    model.save(path)

    try:
        pos = [(1.5, 2.3, 0.7), (5.0, 5.0, 5.0)]  # 2 atoms
        ctx, force, cv_idx = make_ctx_pytorch(
            pos, path, atom_indices=[0], bias_gradient=1.0,
            platform=platform)

        # Trigger force evaluation
        ctx.getState(getForces=True)

        # Check forces: atom 0 should have F.x = -1.0
        forces = get_forces(ctx)
        assert abs(forces[0][0] - (-1.0)) < TOL, \
            f"F[0].x = {forces[0][0]:.6f}, expected -1.0"
        assert abs(forces[0][1]) < TOL, \
            f"F[0].y = {forces[0][1]:.6f}, expected 0"
        assert abs(forces[0][2]) < TOL, \
            f"F[0].z = {forces[0][2]:.6f}, expected 0"
        # Atom 1 should have zero force
        for c in range(3):
            assert abs(forces[1][c]) < TOL, \
                f"F[1][{c}] = {forces[1][c]:.6f}, expected 0"

        print(f"  test_trivial_x_coordinate: OK  "
              f"(F[0]=[{forces[0][0]:.4f}, {forces[0][1]:.4f}, {forces[0][2]:.4f}])")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 2: Squared L2 norm — CV = x^2 + y^2 + z^2 for a single atom
# ---------------------------------------------------------------------------

def test_squared_norm(platform):
    """CV = x^2 + y^2 + z^2 for atom 0.
    Analytical gradient: [2x, 2y, 2z].
    With bias_gradient=1.0, force on atom 0 = -[2x, 2y, 2z]."""
    import torch

    class SquaredNormModel(torch.nn.Module):
        def forward(self, x):
            # x: [1, 3]
            r = x[0]
            return r[0]*r[0] + r[1]*r[1] + r[2]*r[2]

    model = torch.jit.script(SquaredNormModel())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_:
        path = f_.name
    model.save(path)

    try:
        px, py, pz = 1.2, 0.5, 0.8
        pos = [(px, py, pz), (5.0, 5.0, 5.0)]
        ctx, force, cv_idx = make_ctx_pytorch(
            pos, path, atom_indices=[0], bias_gradient=1.0,
            platform=platform)

        ctx.getState(getForces=True)
        forces = get_forces(ctx)

        expected_fx = -2.0 * px
        expected_fy = -2.0 * py
        expected_fz = -2.0 * pz

        assert abs(forces[0][0] - expected_fx) < TOL, \
            f"F[0].x={forces[0][0]:.6f}, expected {expected_fx:.6f}"
        assert abs(forces[0][1] - expected_fy) < TOL, \
            f"F[0].y={forces[0][1]:.6f}, expected {expected_fy:.6f}"
        assert abs(forces[0][2] - expected_fz) < TOL, \
            f"F[0].z={forces[0][2]:.6f}, expected {expected_fz:.6f}"

        print(f"  test_squared_norm: OK  "
              f"(F[0]=[{forces[0][0]:.4f}, {forces[0][1]:.4f}, {forces[0][2]:.4f}])")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 3: Multi-atom model — sum of squared distances from origin
# ---------------------------------------------------------------------------

def test_sum_sq_distances(platform):
    """CV = sum_i (x_i^2 + y_i^2 + z_i^2) for N atoms.
    Gradient on atom i = 2*[x_i, y_i, z_i].
    Compare GLUED forces against analytically expected values."""
    import torch

    class SumSqDistModel(torch.nn.Module):
        def forward(self, x):
            # x: [N, 3]
            return (x * x).sum()

    model = torch.jit.script(SumSqDistModel())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_:
        path = f_.name
    model.save(path)

    try:
        # 4 atoms in the CV, 1 spectator
        positions = [
            (1.0, 0.5, 0.2),
            (0.3, 1.1, 0.7),
            (0.9, 0.6, 1.4),
            (0.2, 0.4, 0.8),
            (9.0, 9.0, 9.0),  # spectator — no force expected
        ]
        atom_indices = [0, 1, 2, 3]
        ctx, force, cv_idx = make_ctx_pytorch(
            positions, path, atom_indices=atom_indices, bias_gradient=1.0,
            platform=platform)

        ctx.getState(getForces=True)
        forces = get_forces(ctx)

        for i, ai in enumerate(atom_indices):
            px, py, pz = positions[ai]
            assert abs(forces[ai][0] - (-2.0 * px)) < TOL, \
                f"Atom {ai} F.x={forces[ai][0]:.6f}, expected {-2.0*px:.6f}"
            assert abs(forces[ai][1] - (-2.0 * py)) < TOL, \
                f"Atom {ai} F.y={forces[ai][1]:.6f}, expected {-2.0*py:.6f}"
            assert abs(forces[ai][2] - (-2.0 * pz)) < TOL, \
                f"Atom {ai} F.z={forces[ai][2]:.6f}, expected {-2.0*pz:.6f}"

        # Spectator atom should have zero force
        for c in range(3):
            assert abs(forces[4][c]) < TOL, \
                f"Spectator F[4][{c}]={forces[4][c]:.6f}, expected 0"

        print("  test_sum_sq_distances: OK  (4 atoms, forces match analytical)")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: Cross-validation — GLUED result vs direct torch eval
# ---------------------------------------------------------------------------

def test_cross_validate_mlp(platform):
    """Two-layer MLP: compare GLUED CV value against direct torch forward pass."""
    import torch

    torch.manual_seed(42)

    class TwoLayerMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(6, 8)  # 2 atoms * 3 coords
            self.fc2 = torch.nn.Linear(8, 1)

        def forward(self, x):
            # x: [2, 3]
            flat = x.reshape(-1)   # [6]
            h = torch.tanh(self.fc1(flat))
            out = self.fc2(h)
            return out.squeeze()   # scalar

    model_orig = TwoLayerMLP()
    model_scripted = torch.jit.script(model_orig)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_:
        path = f_.name
    model_scripted.save(path)

    try:
        positions = [(0.5, 1.2, 0.3), (1.0, 0.4, 0.8), (5.0, 5.0, 5.0)]
        atom_indices = [0, 1]

        ctx, force, cv_idx = make_ctx_pytorch(
            positions, path, atom_indices=atom_indices, bias_gradient=0.0,
            platform=platform)

        # Trigger evaluation but get forces (bias_grad=0 → forces are zero)
        ctx.getState(getForces=True)
        cv_vals = get_cv_values(ctx, force)

        # Direct torch evaluation for comparison
        pos_tensor = torch.tensor(
            [[positions[0][0], positions[0][1], positions[0][2]],
             [positions[1][0], positions[1][1], positions[1][2]]],
            dtype=torch.float32)
        with torch.no_grad():
            expected_cv = model_orig(pos_tensor).item()

        assert abs(cv_vals[cv_idx] - expected_cv) < TOL, \
            (f"GLUED CV={cv_vals[cv_idx]:.6f}, "
             f"direct torch={expected_cv:.6f}")
        print(f"  test_cross_validate_mlp: OK  "
              f"(CV={cv_vals[cv_idx]:.6f}, torch_ref={expected_cv:.6f})")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 5: Finite-difference force check
# ---------------------------------------------------------------------------

def test_fd_force_check(platform):
    """Numerical derivative (finite-difference) matches forces from scatter kernel.

    Using CV = x^2 + y^2 + z^2 for atom 0 (analytical gradient known),
    perturb each coordinate by eps and verify the gradient matches.
    """
    import torch
    import openmm as mm
    import gluedplugin as gp

    class SquaredNormModel(torch.nn.Module):
        def forward(self, x):
            r = x[0]
            return r[0]*r[0] + r[1]*r[1] + r[2]*r[2]

    model = torch.jit.script(SquaredNormModel())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f_:
        path = f_.name
    model.save(path)

    try:
        px, py, pz = 0.7, 1.1, 0.4
        positions_base = [(px, py, pz), (5.0, 5.0, 5.0)]
        eps = 1e-4

        def eval_cv(pos):
            """Evaluate the PyTorch CV value at given positions."""
            n = len(pos)
            sys_ = mm.System()
            for _ in range(n):
                sys_.addParticle(1.0)
            f = gp.GluedForce()
            av = mm.vectori()
            av.append(0)
            pv = mm.vectord()
            cv_idx = f.addPyTorchCV(path, av, pv)
            gv = mm.vectord()
            gv.append(0.0)  # bias_grad=0 so forces are zero
            f.setTestBiasGradients(gv)
            sys_.addForce(f)
            integ = mm.VerletIntegrator(0.001)
            ctx = mm.Context(sys_, integ, platform)
            ctx.setPositions([mm.Vec3(*p) for p in pos])
            ctx.getState(getForces=True)
            vals = f.getLastCVValues(ctx)
            return vals[cv_idx]

        # FD gradient
        cv_xp = eval_cv([(px+eps, py, pz), (5,5,5)])
        cv_xm = eval_cv([(px-eps, py, pz), (5,5,5)])
        cv_yp = eval_cv([(px, py+eps, pz), (5,5,5)])
        cv_ym = eval_cv([(px, py-eps, pz), (5,5,5)])
        cv_zp = eval_cv([(px, py, pz+eps), (5,5,5)])
        cv_zm = eval_cv([(px, py, pz-eps), (5,5,5)])

        fd_gx = (cv_xp - cv_xm) / (2 * eps)
        fd_gy = (cv_yp - cv_ym) / (2 * eps)
        fd_gz = (cv_zp - cv_zm) / (2 * eps)

        # Analytical: ∂(x^2+y^2+z^2)/∂x = 2x, etc.
        an_gx, an_gy, an_gz = 2.0*px, 2.0*py, 2.0*pz

        fd_tol = 1e-3
        assert abs(fd_gx - an_gx) < fd_tol, f"FD gx={fd_gx:.6f}, expected {an_gx:.6f}"
        assert abs(fd_gy - an_gy) < fd_tol, f"FD gy={fd_gy:.6f}, expected {an_gy:.6f}"
        assert abs(fd_gz - an_gz) < fd_tol, f"FD gz={fd_gz:.6f}, expected {an_gz:.6f}"

        # Now check that GLUED forces equal -FD gradient with bias_grad=1.0
        ctx, force, cv_idx = make_ctx_pytorch(
            positions_base, path, atom_indices=[0], bias_gradient=1.0,
            platform=platform)
        ctx.getState(getForces=True)
        forces = get_forces(ctx)

        assert abs(forces[0][0] - (-fd_gx)) < fd_tol, \
            f"F[0].x={forces[0][0]:.6f}, expected {-fd_gx:.6f}"
        assert abs(forces[0][1] - (-fd_gy)) < fd_tol, \
            f"F[0].y={forces[0][1]:.6f}, expected {-fd_gy:.6f}"
        assert abs(forces[0][2] - (-fd_gz)) < fd_tol, \
            f"F[0].z={forces[0][2]:.6f}, expected {-fd_gz:.6f}"

        print(f"  test_fd_force_check: OK  "
              f"(FD=[{fd_gx:.4f},{fd_gy:.4f},{fd_gz:.4f}], "
              f"F=[{forces[0][0]:.4f},{forces[0][1]:.4f},{forces[0][2]:.4f}])")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA not available — skip"); sys.exit(0)

    if not has_torch():
        print("PyTorch not installed — skip"); sys.exit(0)

    print("Stage 3.8 PyTorch CV tests (CUDA):")
    test_trivial_x_coordinate(plat)
    test_squared_norm(plat)
    test_sum_sq_distances(plat)
    test_cross_validate_mlp(plat)
    test_fd_force_check(plat)
    print("All PyTorch CV tests passed.")
