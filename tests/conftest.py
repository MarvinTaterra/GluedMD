"""pytest configuration for the unit test suite.

Provides a CUDA/OpenCL `platform` fixture. Numerical tests skip when neither
is available because GLUED's Reference backend is only an empty-force stub.
"""
import pytest
import openmm as mm


@pytest.fixture(scope="session")
def platform():
    for name in ("CUDA", "OpenCL"):
        try:
            return mm.Platform.getPlatformByName(name)
        except mm.OpenMMException:
            continue
    pytest.skip("GLUED numerical tests require CUDA or OpenCL")
