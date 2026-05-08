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
    """Returns a scripted sequence of rct values from getOPESMetrics."""
    def __init__(self, rct_values):
        self._iter = iter(rct_values)

    def getOPESMetrics(self, context, bias_idx):
        rct = next(self._iter)
        return [math.exp(rct / 2.479), rct, 10.0, 50.0]  # [zed, rct, nker, neff]


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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.5, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
                                        post_convergence_steps=POST, file=buf, verbose=False)
    reporter._open()
    reporter.report(sim, None)          # step 0
    sim.currentStep = 1
    reporter.report(sim, None)          # converge at 1
    sim.currentStep = 1 + POST
    reporter.report(sim, None)          # done
    assert "Post-convergence run complete" in buf.getvalue()


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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=100,
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
    reporter = OPESConvergenceReporter(force, tol=0.1, check_interval=1,
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
    reporter = OPESConvergenceReporter(force, tol=1e10, check_interval=2,
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
    assert reporter._prev_rct is not None
    assert math.isfinite(reporter._prev_rct)
