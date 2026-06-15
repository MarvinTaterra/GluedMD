"""CV_ENERGY correctness on virtual-site water models (TIP4P-Ew, TIP5P).

These are the force-field families that matter for production biomolecular runs:
TIP4P-Ew / OPC / TIP4P-D carry a single M-site (ThreeParticleAverageSite); TIP5P
carries two out-of-plane lone pairs (OutOfPlaneSite); a99SB-disp is run with such
water. The energy CV builds a *vsite-free* linked inner context (clone of the System
minus this GluedForce) — the virtual-site chain rule is completed by the OUTER context's
distributeForcesFromVirtualSites, which runs after GLUED's chain-rule scatter. This test
proves that path end-to-end on the GPU.

Decisive check: bias the energy CV linearly with V = k·U (k = 1 ⇒ dV/dU = 1). Biasing the
total potential energy by V(U) scales every atomic force by (1 + dV/dU), so the total
force on every REAL atom must be exactly (1+k)·F_ref = 2·F_ref. This is sensitive to the
virtual-site term g·M·F_vsite: if the M-site force were stranded on the massless site
(not redistributed) the parent forces would be short by M·F_vsite; if it were counted
twice they'd be long by M·F_vsite. Only the correct single redistribution gives 2·F_ref.
"""
import os, sys, math
import openmm as mm
from openmm import app, unit

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))
import glued

KJ = unit.kilojoule_per_mole
KJNM = unit.kilojoule_per_mole / unit.nanometer


def _platform():
    for name in ("CUDA", "OpenCL"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            pass
    return None


def _build_water_box(model):
    """Small PME water box with virtual sites. model in {'tip4pew','tip5p'}."""
    ff_file = {"tip4pew": "tip4pew.xml", "tip5p": "tip5p.xml"}[model]
    ff = app.ForceField(ff_file)
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model=model, boxSize=mm.Vec3(1.9, 1.9, 1.9) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.8 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    return system, modeller.positions


def _reference(platform, sys_xml, positions):
    s = mm.XmlSerializer.deserialize(sys_xml)
    c = mm.Context(s, mm.VerletIntegrator(0.001), platform)
    c.setPositions(positions)
    st = c.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(KJ),
            st.getForces(asNumpy=True).value_in_unit(KJNM))


def _glued(platform, sys_xml, positions, k):
    s = mm.XmlSerializer.deserialize(sys_xml)
    f = glued.Force(pbc=True, temperature=300.0)
    e = f.add_energy_cv()
    f.add_linear(e, float(k))          # V = k·U  ⇒  dV/dU = k
    s.addForce(f)
    c = mm.Context(s, mm.VerletIntegrator(0.001), platform)
    c.setPositions(positions)
    st = c.getState(getEnergy=True, getForces=True)
    U = list(f.getLastCVValues(c))[e]
    return (U,
            st.getPotentialEnergy().value_in_unit(KJ),
            st.getForces(asNumpy=True).value_in_unit(KJNM))


def run_model(platform, model):
    system, positions = _build_water_box(model)
    nat = system.getNumParticles()
    nvsite = sum(1 for i in range(nat) if system.isVirtualSite(i))
    assert nvsite > 0, f"{model}: expected virtual sites, found none"
    sys_xml = mm.XmlSerializer.serialize(system)

    PE_ref, F_ref = _reference(platform, sys_xml, positions)
    k = 1.0
    U, PE_g, F_g = _glued(platform, sys_xml, positions, k)

    print(f"  [{model}] {nat} particles, {nvsite} virtual sites")

    # 1. energy CV value == unbiased PE
    assert abs(U - PE_ref) < max(1e-1, 1e-4 * abs(PE_ref)), \
        f"{model}: U={U:.3f} vs PE_ref={PE_ref:.3f}"
    print(f"     value:  U = {U:.2f} kJ/mol  ==  unbiased PE {PE_ref:.2f}  ✓")

    # 2. biased PE == (1+k)·PE_ref   (V = k·U added)
    assert abs(PE_g - (1.0 + k) * PE_ref) < max(1e-1, 1e-4 * abs(PE_ref)), \
        f"{model}: PE_glued={PE_g:.3f} vs (1+k)·PE_ref={(1+k)*PE_ref:.3f}"
    print(f"     energy: PE_glued = {PE_g:.2f}  ==  (1+{k:g})·PE_ref {(1+k)*PE_ref:.2f}  ✓")

    # 3. DECISIVE: every atom's force == (1+k)·F_ref  (vsite chain rule complete, once)
    fmax = float(abs(F_ref).max())
    diff = F_g - (1.0 + k) * F_ref          # numpy arrays (nat,3)
    max_err = float(abs(diff).max())
    # tolerance generous vs float/PME noise but FAR below the M·Fv term a bug would create
    tol = max(2.0, 0.02 * fmax)
    assert max_err < tol, \
        f"{model}: force-factor mismatch max |F_glued-(1+k)F_ref| = {max_err:.2f} kJ/mol/nm (tol {tol:.2f})"
    print(f"     force:  max |F_glued - {1+k:g}·F_ref| = {max_err:.3f} kJ/mol/nm "
          f"(tol {tol:.2f}, fmax {fmax:.0f})  ✓")

    # 4. sanity: the multithermal bias itself runs finite on a vsite system
    s = mm.XmlSerializer.deserialize(sys_xml)
    fm = glued.Force(pbc=True, temperature=300.0)
    fm.add_multithermal(300.0, 360.0, n_temps=8, pace=50)
    s.addForce(fm)
    cm = mm.Context(s, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                   0.002 * unit.picoseconds), platform)
    cm.setPositions(positions)
    Em = cm.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    assert math.isfinite(Em), f"{model}: multithermal energy non-finite"
    print(f"     multithermal runs: E = {Em:.1f} kJ/mol (finite)  ✓")


def main():
    plat = _platform()
    if plat is None:
        print("SKIP: CV_ENERGY requires the CUDA or OpenCL platform.")
        return
    print(f"CV_ENERGY virtual-site tests ({plat.getName()} platform):")
    for model in ("tip4pew", "tip5p"):
        run_model(plat, model)
    print("All virtual-site energy-CV tests passed.")


if __name__ == "__main__":
    main()
