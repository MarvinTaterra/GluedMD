"""Stage 1.2/1.3/1.4 smoke test — from design plan §1.2 acceptance tests."""
import sys
import openmm as mm

# Load the plugin so its registerKernelFactories() is called.
import gluedplugin as gp

PLATFORMS = ["Reference", "OpenCL", "CUDA"]

def run_smoke(platform_name):
    try:
        plat = mm.Platform.getPlatformByName(platform_name)
    except mm.OpenMMException:
        print(f"  {platform_name}: platform not available, skipping")
        return

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

if __name__ == "__main__":
    print("Stage 1 smoke test:")
    for name in PLATFORMS:
        run_smoke(name)

    # LocalEnergyMinimizer must not crash with no CVs/biases
    sys_ = mm.System()
    for _ in range(4):
        sys_.addParticle(1.0)
    sys_.addForce(gp.GluedForce())
    integ = mm.LangevinIntegrator(300, 1, 0.001)
    ctx = mm.Context(sys_, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=10)
    print("  LocalEnergyMinimizer: OK")
    print("All smoke tests passed.")
