"""Stage 6.3 + 6.4 acceptance tests.

6.4 — getCurrentCVValues (wraps getCurrentCollectiveVariables): triggers a
      fresh force eval and returns CV values for the current positions.
6.3 — COLVARReporter: writes a PLUMED-compatible COLVAR file with correct
      header, field values, and time stamps.
"""

import sys
import os
import math
import tempfile
import openmm as mm
import gluedplugin as gp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from COLVARReporter import COLVARReporter

CUDA_PLATFORM = "CUDA"
TOL = 1e-4


def get_cuda_platform():
    try:
        return mm.Platform.getPlatformByName(CUDA_PLATFORM)
    except mm.OpenMMException:
        return None


def make_distance_system(pos_nm, platform=None):
    """One distance CV (atoms 0-1), no bias."""
    sys_ = mm.System()
    for _ in pos_nm:
        sys_.addParticle(1.0)
    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform) if platform \
          else mm.Context(sys_, integ)
    ctx.setPositions([mm.Vec3(*p) for p in pos_nm])
    return ctx, f


class _MockSim:
    """Minimal simulation-duck used by the reporter tests.
    Advances the integrator directly and increments currentStep."""
    def __init__(self, ctx):
        self.context = ctx
        self.currentStep = 0


def _run_reporters(ctx, reporters, n_steps):
    """Drive a list of reporters: call report when currentStep % interval == 0."""
    sim = _MockSim(ctx)
    for _ in range(n_steps):
        ctx.getIntegrator().step(1)
        sim.currentStep += 1
        state = ctx.getState(getEnergy=False)
        for r in reporters:
            # describeNextReport returns interval - (step % interval), so
            # it equals the full interval exactly when step % interval == 0.
            nxt, *_ = r.describeNextReport(sim)
            if nxt == r._interval:
                r.report(sim, state)


def _run_reporter(ctx, reporter, n_steps):
    """Drive a single reporter (interval=1) for n_steps."""
    _run_reporters(ctx, [reporter], n_steps)


# ---------------------------------------------------------------------------
# 6.4 — getCurrentCVValues
# ---------------------------------------------------------------------------

def test_getCurrentCV_matches_distance(platform):
    """getCurrentCVValues returns the correct distance after setPositions."""
    d = 1.234
    ctx, f = make_distance_system([(0, 0, 0), (d, 0, 0), (5, 5, 5)], platform)
    values = f.getCurrentCVValues(ctx)
    assert len(values) == 1, f"expected 1 CV, got {len(values)}"
    assert abs(values[0] - d) < TOL, f"CV={values[0]:.6f}, expected {d}"
    print(f"  test_getCurrentCV_matches_distance: OK  (CV={values[0]:.6f}, expected {d})")


def test_getCurrentCV_updates_after_setPositions(platform):
    """After setPositions, getCurrentCVValues reflects the new distance."""
    d1, d2 = 1.0, 2.0
    ctx, f = make_distance_system([(0, 0, 0), (d1, 0, 0), (5, 5, 5)], platform)

    v1 = f.getCurrentCVValues(ctx)
    assert abs(v1[0] - d1) < TOL, f"first CV={v1[0]:.6f}, expected {d1}"

    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(d2, 0, 0), mm.Vec3(5, 5, 5)])
    v2 = f.getCurrentCVValues(ctx)
    assert abs(v2[0] - d2) < TOL, f"updated CV={v2[0]:.6f}, expected {d2}"
    print(f"  test_getCurrentCV_updates_after_setPositions: OK  "
          f"(before={v1[0]:.4f}, after={v2[0]:.4f})")


# ---------------------------------------------------------------------------
# 6.3 — COLVARReporter
# ---------------------------------------------------------------------------

def test_reporter_header_and_values(platform):
    """COLVARReporter writes correct PLUMED header and CV column."""
    d = 0.8
    ctx, f = make_distance_system([(0, 0, 0), (d, 0, 0), (5, 5, 5)], platform)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp:
        tmpname = tmp.name

    try:
        reporter = COLVARReporter(tmpname, 1, f)
        _run_reporter(ctx, reporter, 3)

        with open(tmpname) as fh:
            lines = [l.rstrip() for l in fh if l.strip()]

        assert lines[0].startswith('#! FIELDS time cv0'), \
            f"bad header: {lines[0]!r}"
        assert len(lines) == 4, \
            f"expected header + 3 data lines, got {len(lines)}"

        parts = lines[1].split()
        assert len(parts) == 2, f"expected 2 columns, got {len(parts)}"
        cv_val = float(parts[1])
        assert abs(cv_val - d) < 0.01, f"CV={cv_val:.5f}, expected ~{d}"
        print(f"  test_reporter_header_and_values: OK  (CV={cv_val:.5f}, expected ~{d})")
    finally:
        os.unlink(tmpname)


def test_reporter_custom_names(platform):
    """COLVARReporter uses user-supplied cvNames in the header."""
    ctx, f = make_distance_system([(0, 0, 0), (1.0, 0, 0), (5, 5, 5)], platform)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp:
        tmpname = tmp.name

    try:
        reporter = COLVARReporter(tmpname, 1, f, cvNames=['d01'])
        _run_reporter(ctx, reporter, 1)

        with open(tmpname) as fh:
            header = fh.readline().rstrip()
        assert 'd01' in header, f"custom name missing: {header!r}"
        print(f"  test_reporter_custom_names: OK  (header={header!r})")
    finally:
        os.unlink(tmpname)


def test_reporter_interval(platform):
    """reportInterval=2 produces 3 data lines over 6 steps."""
    ctx, f = make_distance_system([(0, 0, 0), (1.0, 0, 0), (5, 5, 5)], platform)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp:
        tmpname = tmp.name

    try:
        reporter = COLVARReporter(tmpname, 2, f)
        _run_reporters(ctx, [reporter], 6)

        with open(tmpname) as fh:
            lines = [l for l in fh if l.strip()]
        # header + 3 data lines (steps 2, 4, 6)
        assert len(lines) == 4, f"expected 4 lines, got {len(lines)}"
        print(f"  test_reporter_interval: OK  ({len(lines)-1} data lines for 6 steps at interval=2)")
    finally:
        os.unlink(tmpname)


def test_reporter_two_cvs(platform):
    """Two-CV system: header has cv0+cv1, data has three columns."""
    sys_ = mm.System()
    for _ in range(3): sys_.addParticle(1.0)
    f = gp.GluedForce()
    av01 = mm.vectori(); av01.append(0); av01.append(1)
    av02 = mm.vectori(); av02.append(0); av02.append(2)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av01, mm.vectord())
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av02, mm.vectord())
    sys_.addForce(f)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(sys_, integ, platform) if platform \
          else mm.Context(sys_, integ)
    ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1.0, 0, 0), mm.Vec3(0, 1.5, 0)])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as tmp:
        tmpname = tmp.name

    try:
        reporter = COLVARReporter(tmpname, 1, f)
        _run_reporter(ctx, reporter, 1)

        with open(tmpname) as fh:
            header = fh.readline()
            data   = fh.readline()

        assert 'cv0' in header and 'cv1' in header, f"header: {header!r}"
        parts = data.split()
        assert len(parts) == 3, f"expected 3 columns (time cv0 cv1), got {parts}"
        cv0, cv1 = float(parts[1]), float(parts[2])
        assert abs(cv0 - 1.0) < 0.01, f"cv0={cv0:.5f}"
        assert abs(cv1 - 1.5) < 0.01, f"cv1={cv1:.5f}"
        print(f"  test_reporter_two_cvs: OK  (cv0={cv0:.5f}, cv1={cv1:.5f})")
    finally:
        os.unlink(tmpname)


if __name__ == "__main__":
    plat = get_cuda_platform()
    if plat is None:
        print("CUDA not available — running on Reference.")

    print("Stage 6.4 — getCurrentCVValues:")
    test_getCurrentCV_matches_distance(plat)
    test_getCurrentCV_updates_after_setPositions(plat)

    print("Stage 6.3 — COLVARReporter:")
    test_reporter_header_and_values(plat)
    test_reporter_custom_names(plat)
    test_reporter_interval(plat)
    test_reporter_two_cvs(plat)

    print("All CV query and reporter tests passed.")
