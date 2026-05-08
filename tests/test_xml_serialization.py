"""Stage 6.2 acceptance tests — GluedForce XML round-trip serialization.

Each test constructs a force with some combination of CVs and biases, serializes
it to XML via OpenMM's XmlSerializer, deserializes back, and verifies that every
field is preserved exactly.
"""

import sys
import os
import tempfile
import openmm as mm
import gluedplugin as gp
from openmm import XmlSerializer

TOL = 1e-10


def _roundtrip(force):
    xml = XmlSerializer.serialize(force)
    base = XmlSerializer.deserialize(xml)
    rt = gp.GluedForce.cast(base)
    if rt is None:
        raise RuntimeError(f"cast failed; deserialized type: {type(base)}")
    # Transfer SWIG ownership: base would free the C++ object on GC; rt must own it instead.
    base.thisown = False
    rt.thisown = True
    return rt, xml


def _check_cv(orig, rt, idx):
    t0, a0, p0 = orig.getCollectiveVariableParameters(idx)
    t1, a1, p1 = rt.getCollectiveVariableParameters(idx)
    assert t0 == t1, f"CV[{idx}] type mismatch: {t0} vs {t1}"
    assert a0 == a1, f"CV[{idx}] atoms mismatch: {a0} vs {a1}"
    assert len(p0) == len(p1), f"CV[{idx}] param count mismatch"
    for i, (x, y) in enumerate(zip(p0, p1)):
        assert abs(x - y) < TOL, f"CV[{idx}] param[{i}]: {x} vs {y}"


def _check_bias(orig, rt, idx):
    t0, cv0, p0, ip0 = orig.getBiasParameters(idx)
    t1, cv1, p1, ip1 = rt.getBiasParameters(idx)
    assert t0 == t1, f"Bias[{idx}] type mismatch"
    assert cv0 == cv1, f"Bias[{idx}] cvIndices mismatch"
    assert len(p0) == len(p1), f"Bias[{idx}] param count mismatch"
    for i, (x, y) in enumerate(zip(p0, p1)):
        assert abs(x - y) < TOL, f"Bias[{idx}] param[{i}]: {x} vs {y}"
    assert ip0 == ip1, f"Bias[{idx}] intParams mismatch"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_empty_force():
    """Empty force (no CVs, no biases) round-trips cleanly."""
    f = gp.GluedForce()
    f.setTemperature(300.0)
    f.setUsesPeriodicBoundaryConditions(True)
    rt, xml = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 0
    assert rt.getNumBiases() == 0
    assert abs(rt.getTemperature() - 300.0) < TOL
    assert rt.usesPeriodicBoundaryConditions() is True
    print("  test_empty_force: OK")


def test_single_distance_cv():
    """Distance CV (type=1, 2 atoms, no params) round-trips."""
    f = gp.GluedForce()
    av = mm.vectori(); av.append(3); av.append(7)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    rt, _ = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 1
    _check_cv(f, rt, 0)
    print("  test_single_distance_cv: OK")


def test_angle_and_dihedral_cvs():
    """Angle + dihedral CVs preserve atom lists."""
    f = gp.GluedForce()
    av3 = mm.vectori(); av3.append(0); av3.append(1); av3.append(2)
    av4 = mm.vectori()
    for i in range(4): av4.append(i)
    f.addCollectiveVariable(gp.GluedForce.CV_ANGLE, av3, mm.vectord())
    f.addCollectiveVariable(gp.GluedForce.CV_DIHEDRAL, av4, mm.vectord())
    rt, _ = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 2
    _check_cv(f, rt, 0)
    _check_cv(f, rt, 1)
    print("  test_angle_and_dihedral_cvs: OK")


def test_cv_with_params():
    """CV with non-empty parameter list preserves params exactly."""
    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    pv = mm.vectord(); pv.append(0.35); pv.append(6.0); pv.append(12.0)
    f.addCollectiveVariable(gp.GluedForce.CV_COORDINATION, av, pv)
    rt, _ = _roundtrip(f)
    _check_cv(f, rt, 0)
    print("  test_cv_with_params: OK")


def test_expression_cv():
    """Expression CV preserves expression string and input CV indices."""
    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    cvi = mm.vectori(); cvi.append(0)
    f.addExpressionCV("cv0^2", cvi)

    rt, _ = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 2

    expr0, ins0 = f.getExpressionCVParameters(1)
    expr1, ins1 = rt.getExpressionCVParameters(1)
    assert expr0 == expr1, f"expression mismatch: {expr0!r} vs {expr1!r}"
    assert ins0 == ins1, f"inputCVIndices mismatch: {ins0} vs {ins1}"
    print(f"  test_expression_cv: OK  (expression={expr1!r}, inputs={ins1})")


def test_pytorch_cv():
    """PyTorch CV preserves model path, atom indices, and params."""
    f = gp.GluedForce()
    atoms = mm.vectori(); atoms.append(0); atoms.append(5)
    pv = mm.vectord(); pv.append(1.5)
    f.addPyTorchCV(os.path.join(tempfile.gettempdir(), "model.pt"), atoms, pv)
    rt, _ = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 1
    _check_cv(f, rt, 0)
    path0 = f.getPyTorchCVModelPath(0)
    path1 = rt.getPyTorchCVModelPath(0)
    assert path0 == path1, f"model path mismatch: {path0!r} vs {path1!r}"
    print(f"  test_pytorch_cv: OK  (path={path1!r})")


def test_harmonic_bias():
    """Harmonic bias (type=1) round-trips with cvIndices, params, intParams."""
    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    cvi = mm.vectori(); cvi.append(0)
    pv = mm.vectord(); pv.append(1.0); pv.append(500.0)
    iv = mm.vectori()
    f.addBias(gp.GluedForce.BIAS_HARMONIC, cvi, pv, iv)
    rt, _ = _roundtrip(f)
    assert rt.getNumBiases() == 1
    _check_bias(f, rt, 0)
    print("  test_harmonic_bias: OK")


def test_multiple_biases():
    """Two biases on the same CV both survive the round-trip."""
    f = gp.GluedForce()
    av = mm.vectori(); av.append(0); av.append(1)
    f.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, av, mm.vectord())
    cvi = mm.vectori(); cvi.append(0)
    pv1 = mm.vectord(); pv1.append(0.5); pv1.append(200.0)
    pv2 = mm.vectord(); pv2.append(0.8); pv2.append(300.0); pv2.append(0.05)
    iv = mm.vectori()
    f.addBias(gp.GluedForce.BIAS_HARMONIC, cvi, pv1, iv)
    f.addBias(gp.GluedForce.BIAS_METAD, cvi, pv2, iv)
    rt, _ = _roundtrip(f)
    assert rt.getNumBiases() == 2
    _check_bias(f, rt, 0)
    _check_bias(f, rt, 1)
    print("  test_multiple_biases: OK")


def test_force_group_preserved():
    """Non-default forceGroup survives the round-trip."""
    f = gp.GluedForce()
    f.setForceGroup(3)
    rt, _ = _roundtrip(f)
    assert rt.getForceGroup() == 3, f"forceGroup={rt.getForceGroup()}, expected 3"
    print("  test_force_group_preserved: OK")


def test_cv_count_consistency():
    """getNumCollectiveVariables() == 2 after PATH CV round-trip (counts as 2 slots)."""
    f = gp.GluedForce()
    av = mm.vectori()
    for i in range(4): av.append(i)
    pv = mm.vectord()
    pv.append(30.0); pv.append(2)   # lambda, N_frames
    pv.append(0.0); pv.append(0.0); pv.append(0.0)   # frame0
    pv.append(1.0); pv.append(0.0); pv.append(0.0)   # frame1
    f.addCollectiveVariable(gp.GluedForce.CV_PATH, av, pv)
    rt, _ = _roundtrip(f)
    assert rt.getNumCollectiveVariableSpecs() == 1
    assert rt.getNumCollectiveVariables() == 2, \
        f"expected 2 CV values for PATH, got {rt.getNumCollectiveVariables()}"
    _check_cv(f, rt, 0)
    print("  test_cv_count_consistency: OK  (PATH CV → 2 value slots)")


if __name__ == "__main__":
    print("Stage 6.2 — XML serialization round-trip tests:")
    test_empty_force()
    test_single_distance_cv()
    test_angle_and_dihedral_cvs()
    test_cv_with_params()
    test_expression_cv()
    test_pytorch_cv()
    test_harmonic_bias()
    test_multiple_biases()
    test_force_group_preserved()
    test_cv_count_consistency()
    print("All XML serialization tests passed.")
