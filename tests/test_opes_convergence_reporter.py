"""Tests for OPESConvergenceReporter.

Unit tests use lightweight mocks so no GPU is required.
The integration test (test_reporter_full_simulation) runs on whatever platform
pytest's `platform` fixture provides.
"""

import io
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
import openmm as mm
import gluedplugin as gp
from OPESConvergenceReporter import OPESConvergenceReporter

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockForce:
    """Returns a scripted sequence of metrics tuples from getOPESMetrics.

    Two input shapes are accepted:
      * iterable of scalar rct → [zed, rct, nker=10, neff=50.0]
      * iterable of (zed, rct, nker, neff) tuples → returned verbatim
    """
    def __init__(self, values):
        self._iter = iter(values)

    def getOPESMetrics(self, context, bias_idx):
        v = next(self._iter)
        if isinstance(v, (tuple, list)):
            return list(v)
        rct = float(v)
        return [math.exp(rct / 2.479), rct, 10.0, 50.0]

    def getTemperature(self):
        # Reporter calls this for the rct_relative denominator.
        return 300.0


class _MockSimulation:
    """Minimal simulation stub that drives reporter calls."""
    def __init__(self):
        self.currentStep = 0
        self.context = None   # reporter passes it to force.getOPESMetrics
        self.reporters = []

    def _fire_reporter(self, reporter):
        reporter.report(self, None)

    def step(self, n):
        self.currentStep += n
        for r in list(self.reporters):
            r.report(self, None)


# ---------------------------------------------------------------------------
# Unit tests — convergence detection logic
# ---------------------------------------------------------------------------

def test_no_convergence_on_first_call():
    """Reporter must not declare convergence on the very first rct reading."""
    buf = io.StringIO()
    force = _MockForce([1.0])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)
    assert not reporter.converged
    assert not reporter.done


def test_converges_when_drct_below_tol():
    """Convergence triggers when |Δrct| < tol."""
    buf = io.StringIO()
    # First call: rct=1.0.  Second call: rct=1.05, |Δrct|=0.05 < tol=0.1 → converge.
    force = _MockForce([1.0, 1.05])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)   # baseline
    sim.currentStep = 1
    reporter.report(sim, None)   # |Δrct|=0.05 < 0.1 → converged
    assert reporter.converged
    assert reporter.converged_at_step == 1


def test_no_convergence_when_drct_above_tol():
    """No convergence when |Δrct| >= tol."""
    buf = io.StringIO()
    force = _MockForce([1.0, 1.5, 2.2])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=False)
    reporter._open()
    for step in range(3):
        sim.currentStep = step
        reporter.report(sim, None)
    assert not reporter.converged


def test_done_after_post_convergence_steps():
    """done becomes True only after post_convergence_steps additional steps."""
    buf = io.StringIO()
    # rct sequence: [1.0, 1.05 (converge), 1.06, 1.07]
    force = _MockForce([1.0, 1.05, 1.06, 1.07])
    sim = _MockSimulation()
    POST = 5
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=POST, file=buf, verbose=False)
    reporter._open()
    # step 0 — baseline
    reporter.report(sim, None)
    # step 1 — converge (step=1, post window ends at 1+5=6)
    sim.currentStep = 1
    reporter.report(sim, None)
    assert reporter.converged
    assert not reporter.done
    # step 3 — within post window
    sim.currentStep = 3
    reporter.report(sim, None)
    assert not reporter.done
    # step 6 — at end of post window
    sim.currentStep = 6
    reporter.report(sim, None)
    assert reporter.done


def test_zero_post_convergence_steps():
    """With post_convergence_steps=0, done fires in the same call as converged."""
    buf = io.StringIO()
    force = _MockForce([1.0, 1.05])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)   # baseline at step 0
    sim.currentStep = 1
    reporter.report(sim, None)   # converge → done immediately
    assert reporter.converged
    assert reporter.done


def test_verbose_log_contains_rct():
    """Verbose mode must include rct and Δrct in each log line."""
    buf = io.StringIO()
    force = _MockForce([2.0, 2.05])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.5, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=True)
    reporter._open()
    reporter.report(sim, None)
    sim.currentStep = 1
    reporter.report(sim, None)
    output = buf.getvalue()
    assert "rct=" in output
    assert "nker=" in output


def test_convergence_message_logged():
    """A convergence announcement is written to the log."""
    buf = io.StringIO()
    force = _MockForce([1.0, 1.02])
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=10, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)
    sim.currentStep = 1
    reporter.report(sim, None)
    assert "Converged" in buf.getvalue()


def test_done_message_logged():
    """A completion message is written after post_convergence_steps."""
    buf = io.StringIO()
    force = _MockForce([1.0, 1.02, 1.03])
    sim = _MockSimulation()
    POST = 5
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=POST, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)          # step 0
    sim.currentStep = 1
    reporter.report(sim, None)          # converge at 1
    sim.currentStep = 1 + POST
    reporter.report(sim, None)          # done
    assert "Post-convergence run complete" in buf.getvalue()


# ---------------------------------------------------------------------------
# New criteria: rct_relative, neff_rate, force_converge
# ---------------------------------------------------------------------------

def test_reporter_rct_relative_is_scale_invariant():
    """rct_relative should converge at the same tol regardless of rct magnitude."""
    buf = io.StringIO()
    for scale in (1.0, 100.0):
        # rct trace approaches a plateau; |Δsignal| shrinks below 0.01 after
        # a couple of steps. Two consecutive passes must trigger convergence.
        rct_trace = [1.0, 1.5, 1.8, 1.95, 1.98, 1.99, 1.995]
        force = _MockForce([r * scale for r in rct_trace])
        sim = _MockSimulation()
        reporter = OPESConvergenceReporter(
            force, criterion='rct_relative', tol=0.01,
            check_interval=1, min_consecutive_passes=2,
            min_kernels=0, post_convergence_steps=0,
            file=buf, verbose=False)
        reporter._open()
        for step, _ in enumerate(rct_trace):
            sim.currentStep = step
            reporter.report(sim, None)
        assert reporter.converged, \
            f"rct_relative should be scale-invariant; failed at scale={scale}"


def test_reporter_neff_rate_criterion():
    """neff_rate criterion should converge once neff/step stabilizes."""
    buf = io.StringIO()
    # neff sequence settling: ratios 0.10, 0.10, 0.10, 0.10 over consecutive checks.
    # nker held at 100 to clear the warm-up gate.
    metrics_seq = [
        (1.0, 0.0,  100, 100.0),   # step 1   neff/step = 100.0
        (1.0, 0.1,  100, 200.0),   # step 2   100.0
        (1.0, 0.15, 100, 300.0),   # step 3   100.0
        (1.0, 0.18, 100, 400.0),   # step 4   100.0
        (1.0, 0.19, 100, 500.0),   # step 5   100.0  ← drift = 0
    ]
    force = _MockForce(metrics_seq)
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(
        force, criterion='neff_rate', tol=0.005,
        check_interval=1, min_consecutive_passes=2,
        min_kernels=10, post_convergence_steps=0,
        file=buf, verbose=False)
    reporter._open()
    for step in range(len(metrics_seq)):
        sim.currentStep = step + 1
        reporter.report(sim, None)
    assert reporter.converged, "neff_rate criterion should detect convergence"


def test_reporter_force_converge():
    """force_converge() flips state and writes the manual-trigger log line."""
    buf = io.StringIO()
    force = _MockForce([1.0, 2.0, 3.0])  # huge drift — would never auto-converge
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(
        force, criterion='rct_absolute', tol=1e-9,
        check_interval=1, min_consecutive_passes=1,
        min_kernels=0, post_convergence_steps=2,
        file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)
    sim.currentStep = 5
    reporter.force_converge(sim)
    assert reporter.converged
    assert reporter.converged_at_step == 5
    assert "Manually marked converged" in buf.getvalue()
    # Post-window must elapse before done
    sim.currentStep = 6
    reporter.report(sim, None)
    assert not reporter.done
    sim.currentStep = 7
    reporter.report(sim, None)
    assert reporter.done


def test_reporter_warmup_blocks_convergence():
    """min_kernels gate must prevent convergence even if drift is below tol."""
    buf = io.StringIO()
    # Below-tol drifts but nker=5 < min_kernels=20 → should NOT converge.
    metrics_seq = [(1.0, 1.0, 5, 50.0), (1.0, 1.001, 5, 51.0), (1.0, 1.001, 5, 52.0)]
    force = _MockForce(metrics_seq)
    sim = _MockSimulation()
    reporter = OPESConvergenceReporter(
        force, criterion='rct_absolute', tol=0.01,
        check_interval=1, min_consecutive_passes=1,
        min_kernels=20, post_convergence_steps=0,
        file=buf, verbose=False)
    reporter._open()
    for step in range(3):
        sim.currentStep = step
        reporter.report(sim, None)
    assert not reporter.converged, "warm-up gate should block convergence"


# ---------------------------------------------------------------------------
# Integration test — run() helper attaches and detaches reporter
# ---------------------------------------------------------------------------

def test_run_method_attaches_and_detaches():
    """run() should add the reporter before stepping and remove it after done."""
    buf = io.StringIO()
    # Build a mock simulation whose step() fires the reporter manually
    class _CountingSim(_MockSimulation):
        def step(self, n):
            self.currentStep += n
            for r in list(self.reporters):
                r.report(self, None)

    # rct sequence: 10 values rising quickly (never converge), then max_steps hit
    force = _MockForce([float(i) for i in range(100)])
    sim = _CountingSim()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=0, file=buf, verbose=False)
    reporter.run(sim, max_steps=5)
    # After run() the reporter should be detached
    assert reporter not in sim.reporters


def test_run_method_stops_at_max_steps():
    """run() must not exceed max_steps even if convergence never fires."""
    buf = io.StringIO()

    class _CountingSim(_MockSimulation):
        def step(self, n):
            self.currentStep += n
            for r in list(self.reporters):
                r.report(self, None)

    # rct keeps rising — never converges
    force = _MockForce([float(i) * 10 for i in range(1000)])
    sim = _CountingSim()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=100,
                                        post_convergence_steps=50_000, file=buf, verbose=False)
    reporter.run(sim, max_steps=300)
    assert sim.currentStep <= 300
    assert not reporter.done


def test_run_stops_after_convergence_plus_post():
    """run() stops as soon as done=True, well before max_steps."""
    buf = io.StringIO()

    class _CountingSim(_MockSimulation):
        def step(self, n):
            self.currentStep += n
            for r in list(self.reporters):
                r.report(self, None)

    # Converges on 2nd call (|Δrct|=0.01 < 0.1), POST=5, so done at step ≥ converged+5
    force = _MockForce([1.0, 1.01] + [1.01] * 50)
    sim = _CountingSim()
    POST = 5
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=0.1, check_interval=1,
                                        post_convergence_steps=POST, file=buf, verbose=False)
    reporter.run(sim, max_steps=10_000)
    assert reporter.done
    # Should have stopped well before 10_000 steps
    assert sim.currentStep < 100


# ---------------------------------------------------------------------------
# Integration test — full OpenMM simulation with Reference platform
# ---------------------------------------------------------------------------

_MASS_AMU = 1000.0
_DT_PS    = 0.0001


def _make_opes_simulation(platform):
    """3-atom system with one OPES bias on the interatomic distance."""
    sys_ = mm.System()
    for _ in range(3):
        sys_.addParticle(_MASS_AMU)

    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    pv = mm.vectord()
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, pv)

    civ = mm.vectori(); civ.append(0)
    bv = mm.vectord()
    iv = mm.vectori()
    # OPES params: kT, sigma0, gamma, pace, nkerMax
    for x in [2.479, 0.05, 10.0]: bv.append(x)
    for x in [10, 1000]: iv.append(x)
    f.addBias(gp.GluedForce.BIAS_OPES, civ, bv, iv)

    sys_.addForce(f)
    integ = mm.VerletIntegrator(_DT_PS)
    ctx = mm.Context(sys_, integ, platform)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0), mm.Vec3(5, 5, 5)])

    class _Sim:
        def __init__(self):
            self.context = ctx
            self.currentStep = 0
            self.reporters = []

        def step(self, n):
            integ.step(n)
            self.currentStep += n
            for r in list(self.reporters):
                r.report(self, None)

    return _Sim(), f


def test_reporter_getOPESMetrics_runs_on_reference(platform):
    """Smoke test: reporter can call getOPESMetrics without error (all platforms)."""
    sim, force = _make_opes_simulation(platform)
    buf = io.StringIO()
    reporter = OPESConvergenceReporter(force, criterion='rct_absolute', min_consecutive_passes=1, min_kernels=0, tol=1e10, check_interval=2,
                                        post_convergence_steps=0, file=buf, verbose=True)
    # Warm up so cvValuesReady is True
    sim.step(2)
    sim.step(2)
    # Manually fire the reporter
    reporter._open()
    reporter.report(sim, None)
    reporter.report(sim, None)
    output = buf.getvalue()
    assert "rct=" in output
    # rct value must be finite
    assert reporter._prev_signal is not None
    assert math.isfinite(reporter._prev_signal)
