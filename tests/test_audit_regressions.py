"""
Regression tests for the defects found in the September 2026 audit.

Each test is a small deterministic system that reproduces one finding:
validation, checkpoint continuation, side-effect-free energy evaluation,
grid boundary forces, expression/energy CV ordering, periodic OPES, shared
walker state, replica exchange acceptance and the pool scheduler.
"""
import io
import math
import struct

import numpy as np
import pytest
import openmm as mm
from openmm import unit

import glued
import gluedplugin as gp
from MultiGPUManager import BiasStateMerger, MultiWalkerPool
from ReplicaExchange import ReplicaExchange, exchange_log_acceptance


@pytest.fixture
def gpu():
    try:
        return mm.Platform.getPlatformByName("CUDA")
    except Exception:
        pytest.skip("CUDA runtime required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_x(ctx, x):
    """Three heavy particles; only particle 0's x coordinate is varied."""
    ctx.setPositions([[x, 0, 0], [0, 1, 0], [0, 0, 1]])


def make_context(force, platform, other_force=None, integrator=None):
    system = mm.System()
    for _ in range(3):
        system.addParticle(1000.0)
    if other_force is not None:
        system.addForce(other_force)
    system.addForce(force)
    props = {"Precision": "double"} if platform.getName() == "CUDA" else {}
    ctx = mm.Context(system, integrator or mm.VerletIntegrator(1e-6), platform, props)
    set_x(ctx, 0.0)
    return ctx


def energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)


def opes_force(capacity=100, adaptive=False):
    f = glued.Force(temperature=300.0)
    cv = f.add_position(0, 0)
    f.add_opes(cv, None if adaptive else 0.01, pace=1, max_kernels=capacity,
               adaptive_sigma_stride=4 if adaptive else None)
    return f


def parsed_state(force, ctx):
    return BiasStateMerger.parse(force.getBiasState(ctx), force)


def opes_section(force, ctx, index=0):
    """Named view of one parsed OPES section."""
    fields = ("num_kernels", "sum_uprob", "num_samples", "deposit_count", "sum_weights",
              "sum_w", "running_mean", "running_m2", "sigma0", "centers", "sigmas",
              "log_weights")
    return dict(zip(fields, parsed_state(force, ctx)["opes"][index]))


def opes_centers(force, ctx):
    return np.frombuffer(opes_section(force, ctx)["centers"], dtype="<f8")


# ---------------------------------------------------------------------------
# F01 / F02 / F22: validation errors are catchable Python exceptions
# ---------------------------------------------------------------------------

def test_validation_errors_are_catchable():
    f = glued.Force()
    with pytest.raises(RuntimeError):
        f.addBias(gp.GluedForce.BIAS_HARMONIC, [999], [1, 0], [])
    cv = f.add_position(0, 0)
    bad_layouts = [
        (gp.GluedForce.BIAS_HARMONIC, [], []),                       # missing parameters
        (gp.GluedForce.BIAS_METAD, [1, .1, 10, 2.5, 0, 1], [0, 5, 0]),  # pace 0
        (99, [], []),                                                # unknown type
    ]
    for kind, params, ints in bad_layouts:
        with pytest.raises(RuntimeError):
            f.addBias(kind, [cv], params, ints)
    with pytest.raises(RuntimeError):
        f.add_expression("cv0", [1])   # CV slot 1 does not exist yet
    for temperature in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(RuntimeError):
            f.setTemperature(temperature)


def test_reweighting_and_reporter_validation():
    from COLVARReporter import COLVARReporter
    from OPESConvergenceReporter import OPESConvergenceReporter
    for bad in [[], [float("nan")], [float("inf")], [[1, 2]]]:
        with pytest.raises(ValueError):
            glued.kish_ess(bad)
    with pytest.raises(ValueError):
        glued.multithermal_log_weights([1, 2], [0], 300, 310)
    with pytest.raises(ValueError):
        glued.reweight_to_temperature([1, 2], [0, 0], 300, 310, observable=[1])
    f = glued.Force()
    f.add_position(0, 0)
    with pytest.raises(ValueError):
        COLVARReporter(io.StringIO(), 0, f)
    with pytest.raises(ValueError):
        COLVARReporter(io.StringIO(), 1, f, cvNames=["a", "b"])
    with pytest.raises(ValueError):
        OPESConvergenceReporter(f, check_interval=0)
    assert glued.kish_ess([1000, 1000]) == pytest.approx(2)


def test_xml_preserves_force_name():
    f = glued.Force()
    f.setName("audit named force")
    f.add_position(0, 0)
    restored = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))
    assert restored.getName() == "audit named force"


# ---------------------------------------------------------------------------
# F21: pool scheduling is exact and persists across run() calls
# ---------------------------------------------------------------------------

def test_schedule_exact_and_persistent():
    # Drive the scheduling loop with a step-recording stand-in; no GPU needed.
    pool = MultiWalkerPool.__new__(MultiWalkerPool)
    pool._elapsed = 0
    pool._sync_interval = 50
    pool._sharing = True
    pool._ordered = False
    pool._re_mode = "H-REUS"
    pool._re_interval = 75
    now = [0]
    merges, exchanges = [], []
    pool._step_all = lambda n: now.__setitem__(0, now[0] + n)
    pool._sync_bias = lambda: merges.append(now[0])
    pool._attempt_re_swaps = lambda: exchanges.append(now[0])
    for _ in range(30):
        pool.run(10)
    assert merges == [50, 100, 150, 200, 250, 300]
    assert exchanges == [75, 150, 225, 300]


# ---------------------------------------------------------------------------
# F15: replica exchange acceptance uses complete reduced potentials
# ---------------------------------------------------------------------------

def test_replica_exchange_keeps_common_bias():
    replicas = []
    for x in [0, 1]:
        system = mm.System()
        system.addParticle(1.0)
        bias = mm.CustomExternalForce("10*x*x")
        bias.addParticle(0, [])
        bias.setForceGroup(1)
        system.addForce(bias)
        ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions([[x, 0, 0]])
        ctx.setVelocities([[0, 0, 0]])
        replicas.append((ctx, None))
    re = ReplicaExchange(replicas, mode="T-REMD", temperatures=[300, 600],
                         bias_force_group=1, seed=0)
    # Acceptance probability is 0.135; the seeded draw is 0.844.
    assert re._attempt_swap(0, 1) is False


def test_replica_exchange_npt_pressure_volume():
    contexts, states = [], []
    for side, pressure in [(2, 1), (3, 2)]:
        system = mm.System()
        system.addParticle(1.0)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(side, 0, 0), mm.Vec3(0, side, 0),
                                            mm.Vec3(0, 0, side))
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(nb.CutoffPeriodic)
        nb.setCutoffDistance(0.5)
        nb.addParticle(0, 1, 0)
        system.addForce(nb)
        system.addForce(mm.MonteCarloBarostat(pressure, 300))
        ctx = mm.Context(system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions([[0, 0, 0]])
        contexts.append(ctx)
        states.append(ctx.getState(getPositions=True, getEnergy=True))
    delta = exchange_log_acceptance(*contexts, *states, 1, 1)
    bar_to_kj_per_nm3 = 0.0602214076
    assert delta == pytest.approx(bar_to_kj_per_nm3 * (1 - 2) * (8 - 27))


# ---------------------------------------------------------------------------
# F03 / F04 / F14: OPES checkpoints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("adaptive", [False, True])
def test_checkpoint_continuation_matches_uninterrupted(gpu, adaptive):
    f = opes_force(adaptive=adaptive)
    c = make_context(f, gpu)
    for step in range(8):
        set_x(c, step * 0.1)
        c.getIntegrator().step(1)
    saved = f.getBiasState(c)

    f2 = opes_force(adaptive=adaptive)
    c2 = make_context(f2, gpu)
    c2.setStepCount(c.getStepCount())
    f2.setBiasState(saved, c2)
    assert f.getOPESMetrics(c, 0) == pytest.approx(f2.getOPESMetrics(c2, 0))

    for step in range(8, 13):
        for ctx in (c, c2):
            set_x(ctx, step * 0.1)
            ctx.getIntegrator().step(1)
    assert f.getBiasState(c) == f2.getBiasState(c2)
    assert energy(c) == pytest.approx(energy(c2))


def test_checkpoint_read_cannot_unfreeze_deposition(gpu):
    # A constant CV compresses every deposit into one kernel; with capacity 3
    # deposition must continue past three deposits, and reading the state
    # in between must not change the outcome.
    f = opes_force(capacity=3)
    c = make_context(f, gpu)
    c.getIntegrator().step(10)
    first = opes_section(f, c)
    c.getIntegrator().step(10)
    second = opes_section(f, c)
    assert first["num_kernels"] == second["num_kernels"] == 1
    assert first["num_samples"] == 9
    assert second["num_samples"] == 19


def test_checkpoint_rejects_damage_without_mutation(gpu):
    f = opes_force()
    c = make_context(f, gpu)
    c.getIntegrator().step(4)
    blob = f.getBiasState(c)
    damaged = [b"", blob[:12], blob[:-1], blob + b"junk",
               blob[:4] + struct.pack("<i", 99) + blob[8:]]
    for bad in damaged:
        with pytest.raises(RuntimeError):
            f.setBiasState(bad, c)
        assert f.getBiasState(c) == blob

    # Restoring into a larger kernel table is allowed ...
    larger = opes_force(capacity=101)
    larger_c = make_context(larger, gpu)
    larger.setBiasState(blob, larger_c)
    assert larger.getOPESMetrics(larger_c, 0) == pytest.approx(f.getOPESMetrics(c, 0))
    # ... but any other configuration difference is rejected.
    other = glued.Force(temperature=300.0)
    cv = other.add_position(0, 0)
    other.add_opes(cv, 0.01, pace=1, gamma=11)
    other_c = make_context(other, gpu)
    with pytest.raises(RuntimeError, match="configuration"):
        other.setBiasState(blob, other_c)


def test_capacity_exhaustion_is_explicit(gpu):
    f = opes_force(capacity=1)
    c = make_context(f, gpu)
    c.getIntegrator().step(2)
    set_x(c, 10.0)   # far from the only kernel: cannot merge, no free slot
    with pytest.raises(Exception, match="kernel table is full"):
        c.getIntegrator().step(1)


def test_two_dimensional_metad_and_multithermal_roundtrip(gpu):
    f = glued.Force(temperature=300.0)
    x = f.add_position(0, 0)
    y = f.add_position(0, 1)
    f.add_metad([x, y], [0.1, 0.2], 1, 1, grid_min=[0, 0], grid_max=[1, 1], bins=[2, 3])
    u = f.add_energy_cv()
    kT = 2.4943387854
    f.addBias(gp.GluedForce.BIAS_OPES_MULTITHERMAL, [u], [kT, 1 / kT, 1 / 4.157231309], [1])
    c = make_context(f, gpu)
    c.getIntegrator().step(3)
    blob = f.getBiasState(c)
    state = parsed_state(f, c)
    assert state["multithermal"]
    assert BiasStateMerger.pack(state) == blob


def test_context_specific_checkpoint_and_global_bias_index(gpu):
    f = glued.Force(temperature=300.0)
    cv = f.add_position(0, 0)
    f.add_harmonic(cv, 1, 0)
    b = f.add_opes(cv, 0.1, pace=1)
    system = mm.System()
    for _ in range(3):
        system.addParticle(1000.0)
    system.addForce(f)
    c = mm.Context(system, mm.VerletIntegrator(1e-6), gpu, {"Precision": "double"})
    c2 = mm.Context(system, mm.VerletIntegrator(1e-6), gpu, {"Precision": "double"})
    set_x(c, 0.0)
    set_x(c2, 0.0)
    with pytest.raises(RuntimeError):
        f.getBiasState()   # ambiguous: two live Contexts
    c.getIntegrator().step(3)
    assert f.getOPESMetrics(c, b)[2] > 0
    assert f.getOPESMetrics(c2, b)[2] == 0
    with pytest.raises(RuntimeError):
        f.getOPESMetrics(c, 0)   # bias 0 is the harmonic restraint
    f.setBiasState(f.getBiasState(c), c2)
    assert f.getOPESMetrics(c, b) == pytest.approx(f.getOPESMetrics(c2, b))


# ---------------------------------------------------------------------------
# F05 / F06: evaluation is side-effect free; history commits on the current step
# ---------------------------------------------------------------------------

def test_energy_reads_do_not_commit_history(gpu):
    for adaptive in [False, True]:
        f = opes_force(adaptive=adaptive)
        c = make_context(f, gpu)
        c.getIntegrator().step(6)
        before = f.getBiasState(c)
        for x in [5, 1, 0, 3]:
            set_x(c, x)
            energy(c)
        assert f.getBiasState(c) == before

    f = glued.Force()
    cv = f.add_position(0, 0)
    f.add_abmd(cv, 1, 0)
    c = make_context(f, gpu)
    set_x(c, 2.0)
    c.getIntegrator().step(1)
    before = f.getBiasState(c)
    set_x(c, 1.0)
    energy(c)   # a trial configuration closer to the target
    set_x(c, 2.0)
    assert energy(c) == pytest.approx(0)
    assert f.getBiasState(c) == before


def test_auxiliary_history_is_not_initialized_by_energy_query(gpu):
    f = glued.Force()
    cv = f.add_position(0, 0)
    f.addBias(gp.GluedForce.BIAS_EXT_LAGRANGIAN, [cv], [1, 1], [])
    c = make_context(f, gpu)
    before = f.getBiasState(c)
    set_x(c, 2.0)
    energy(c)
    assert f.getBiasState(c) == before


def test_deposition_uses_current_coordinates(gpu):
    f = opes_force()
    c = make_context(f, gpu)
    set_x(c, 0.0)
    energy(c)
    set_x(c, 2.0)
    c.setStepCount(1)
    c.getIntegrator().step(1)
    assert opes_centers(f, c) == pytest.approx([2.0])


def test_moving_bias_invalidates_cached_forces(gpu):
    # Velocity Verlet as a CustomIntegrator reuses the previous step's forces
    # unless the force reports that its potential changed.
    integrator = mm.CustomIntegrator(0.01)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    f = glued.Force()
    cv = f.add_position(0, 0)
    f.add_moving_restraint(cv, [(0, 0, 0), (10, 10, 0)])
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    system.addForce(f)
    c = mm.Context(system, integrator, gpu, {"Precision": "double"})
    set_x(c, 1.0)
    c.setVelocities([[0, 0, 0]] * 3)
    integrator.step(2)
    state = c.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[0, 0]
    v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)[0, 0]
    assert x == pytest.approx(0.99995)
    assert v == pytest.approx(-0.00999975)


def test_moving_restraint_uses_64_bit_step_index(gpu):
    f = glued.Force()
    cv = f.add_position(0, 0)
    f.add_moving_restraint(cv, [(0, 0, 0), (4_000_000_000, 4, 0)])
    c = make_context(f, gpu)
    set_x(c, 1.0)
    c.setStepCount(3_000_000_000)
    assert energy(c) == pytest.approx(1.5)
    c.getIntegrator().step(1)
    assert c.getStepCount() == 3_000_000_001


# ---------------------------------------------------------------------------
# F07: periodic OPES on a torsion
# ---------------------------------------------------------------------------

def test_periodic_opes_torsion_boundary(gpu):
    f = glued.Force(temperature=300.0)
    cv = f.add_dihedral([0, 1, 2, 3])
    f.add_opes(cv, 0.1, pace=1)
    system = mm.System()
    for _ in range(4):
        system.addParticle(1000.0)
    system.addForce(f)
    c = mm.Context(system, mm.VerletIntegrator(1e-6), gpu, {"Precision": "double"})

    def set_angle(phi):
        c.setPositions([[1, 0, 0], [0, 0, 0], [0, 0, 1], [math.cos(phi), math.sin(phi), 1]])

    set_angle(math.pi)
    c.setStepCount(1)
    c.getIntegrator().step(1)   # deposit a kernel at +/- pi
    set_angle(math.pi - 0.01)
    a = energy(c)
    set_angle(-math.pi + 0.01)
    b = energy(c)
    assert a == pytest.approx(b, abs=1e-7)
    # Merging across the branch cut must keep the center near pi.
    c.getIntegrator().step(1)
    center = opes_centers(f, c)[0]
    assert abs(abs(center) - math.pi) < 0.02


# ---------------------------------------------------------------------------
# F16 / F17 / F18 / F19: CV evaluation order and numerics
# ---------------------------------------------------------------------------

def test_energy_expression_and_global_parameter(gpu):
    physical = mm.CustomExternalForce("k*x*x")
    physical.addGlobalParameter("k", 2)
    physical.addParticle(0, [])
    f = glued.Force()
    cv = f.add_energy_cv()
    f.add_expression("2*cv0", [cv])
    c = make_context(f, gpu, physical)
    set_x(c, 1.0)
    energy(c)
    set_x(c, 2.0)
    energy(c)
    assert list(f.getLastCVValues(c)) == pytest.approx([8, 16])
    c.setParameter("k", 5)
    energy(c)
    assert list(f.getLastCVValues(c)) == pytest.approx([20, 40])


def test_expression_indices_are_cv_slots(gpu):
    f = glued.Force()
    for _ in range(4):
        f.add_position(0, 0)
    f.add_expression("2*cv0", [3])   # slot 3 exceeds the particle count
    c = make_context(f, gpu)
    set_x(c, 2.0)
    energy(c)
    assert f.getLastCVValues(c)[4] == pytest.approx(4)


def test_box_expression_cannot_bypass_readonly_guard(gpu):
    f = glued.Force()
    cv = f.addCollectiveVariable(gp.GluedForce.CV_VOLUME, [], [])
    expr = f.add_expression("cv0", [cv])
    f.add_harmonic(expr, 1, 0)
    with pytest.raises(Exception, match="read-only"):
        make_context(f, gpu)


def test_grid_clamping_force_matches_energy(gpu):
    f = glued.Force()
    cv = f.add_position(0, 0)
    f.addBias(gp.GluedForce.BIAS_EXTERNAL, [cv], [0, 1, 0, 1], [1, 0])
    c = make_context(f, gpu)
    for x in [-1.0, 2.0]:
        set_x(c, x)
        forces = c.getState(getForces=True).getForces(asNumpy=True)
        fx = forces.value_in_unit(unit.kilojoules_per_mole / unit.nanometer)[0, 0]
        set_x(c, x + 0.001)
        e_plus = energy(c)
        set_x(c, x - 0.001)
        e_minus = energy(c)
        assert fx == pytest.approx(-(e_plus - e_minus) / 0.002, abs=1e-8)


def test_path_far_from_frames_has_finite_distance(gpu):
    f = glued.Force()
    f.add_path([0], [[0, 0, 0], [0.1, 0, 0]], 100)
    c = make_context(f, gpu)
    set_x(c, 10.0)
    energy(c)
    assert list(f.getLastCVValues(c)) == pytest.approx([2, 98.01], abs=1e-5)


# ---------------------------------------------------------------------------
# F09 / F10 / F11 / F13: shared walker state
# ---------------------------------------------------------------------------

def test_shared_opes_walkers_agree(gpu):
    forces = [opes_force(), opes_force()]
    contexts = [make_context(f, gpu) for f in forces]
    pool = MultiWalkerPool([[c] for c in contexts], [[f] for f in forces], sync_interval=2)
    pool.run(8)
    for c in contexts:
        set_x(c, 0.0)
    assert energy(contexts[0]) == pytest.approx(energy(contexts[1]))
    # One shared history: 2 walkers x 8 steps, minus the two step-0 skips.
    assert opes_section(forces[0], contexts[0])["num_samples"] == 14
    assert opes_section(forces[1], contexts[1]) == opes_section(forces[0], contexts[0])
    pool.close()


def test_grid_merge_preserves_unselected_bias_and_is_idempotent(gpu):
    forces, contexts = [], []
    for x in [0.0, 1.0]:
        f = glued.Force(temperature=300.0)
        cv = f.add_position(0, 0)
        f.add_metad(cv, 0.1, 1, 1, grid_min=-2, grid_max=2, bins=20)
        f.add_abmd(cv, 1, 0)
        c = make_context(f, gpu)
        set_x(c, x)
        forces.append(f)
        contexts.append(c)
    pool = MultiWalkerPool([[c] for c in contexts], [[f] for f in forces], sync_interval=2)
    pool.run(4)
    states = [parsed_state(f, c) for f, c in zip(forces, contexts)]
    assert states[0]["abmd"] != states[1]["abmd"]   # private history stays private
    assert states[0]["metad"] == states[1]["metad"]
    for _ in range(3):
        pool._sync_bias()
    assert states == [parsed_state(f, c) for f, c in zip(forces, contexts)]
    pool.sync_bias_state_from(1)
    pool._sync_bias()
    assert states == [parsed_state(f, c) for f, c in zip(forces, contexts)]
    pool.close()
