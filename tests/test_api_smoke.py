"""Stage 1.2/1.3/1.4 smoke test — from design plan §1.2 acceptance tests.

The script tries every available compute platform and prints a one-line result
per platform. A failure on one platform does not abort the run — we still want
to see whether CUDA works even if OpenCL kernel compilation breaks (or vice
versa). The exit code is non-zero iff at least one platform was attempted and
reached a real assertion failure or compile error.
"""
import sys
import traceback
import openmm as mm

# Load the plugin so its registerKernelFactories() is called.
import gluedplugin as gp

# Reference first as a sanity floor; CUDA before OpenCL so a CUDA install is
# always verified before an OpenCL kernel-compile failure can be reached.
PLATFORMS = ["Reference", "CUDA", "OpenCL"]

def run_smoke(platform_name):
    """Return True on success, False on failure, None if the platform is absent."""
    try:
        plat = mm.Platform.getPlatformByName(platform_name)
    except mm.OpenMMException:
        print(f"  {platform_name}: platform not available, skipping")
        return None

    try:
        sys_ = mm.System()
        for _ in range(4):
            sys_.addParticle(1.0)
        f = gp.GluedForce()
        sys_.addForce(f)

        integ = mm.LangevinIntegrator(300, 1, 0.001)
        ctx = mm.Context(sys_, integ, plat)
        ctx.setPositions([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])

        state = ctx.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(
            state.getPotentialEnergy().unit)
        assert energy == 0.0, f"Expected 0.0 energy on {platform_name}, got {energy}"
        print(f"  {platform_name}: OK (energy={energy})")
        return True
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  {platform_name}: FAILED — {first_line}")
        traceback.print_exc(limit=2, file=sys.stdout)
        return False

if __name__ == "__main__":
    print("Stage 1 smoke test:")
    results = {name: run_smoke(name) for name in PLATFORMS}

    # LocalEnergyMinimizer must not crash with no CVs/biases (Reference only).
    try:
        sys_ = mm.System()
        for _ in range(4):
            sys_.addParticle(1.0)
        sys_.addForce(gp.GluedForce())
        integ = mm.LangevinIntegrator(300, 1, 0.001)
        ctx = mm.Context(sys_, integ, mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=10)
        print("  LocalEnergyMinimizer: OK")
        minimizer_ok = True
    except Exception as exc:
        print(f"  LocalEnergyMinimizer: FAILED — {exc}")
        minimizer_ok = False

    attempted = [n for n, r in results.items() if r is not None]
    failed    = [n for n, r in results.items() if r is False]
    if failed or not minimizer_ok:
        print(f"Smoke tests: {len(failed)} platform failure(s) "
              f"({', '.join(failed) or 'none'}); "
              f"{len(attempted) - len(failed)} passed.")
        sys.exit(1)
    print(f"All smoke tests passed ({', '.join(attempted)}).")
