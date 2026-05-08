"""pytest configuration for the unit test suite.

Provides a `platform` fixture that returns a CUDA platform when available,
falling back to the Reference platform for environments without a GPU.
"""
import pytest
import openmm as mm


@pytest.fixture(scope="session")
def platform():
    for name in ("CUDA", "OpenCL", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except mm.OpenMMException:
            continue
    raise RuntimeError("No OpenMM platform available")
