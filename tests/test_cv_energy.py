"""Stage 1 (OPES multithermal): CV_ENERGY acceptance tests.

CV_ENERGY returns the system's UNBIASED total potential energy U (evaluated in a
linked inner context that omits the GluedForce). Its Jacobian is dU/dx = -F.

Validates:
  1. value:    CV_ENERGY == potential energy of the same system without GLUED.
  2. gradient: d(CV_ENERGY)/dx == -F (finite difference on the CV value vs the
               reference forces).
  3. scatter:  with cvBiasGradient[energy]=1, GLUED adds force +F to each atom
               (force = -dV/dCV * dCV/dx = -(1)*(-F) = +F).
"""
import sys
sys.path.insert(0, ".")
import openmm as mm
from openmm import unit
import gluedplugin as gp

PLATFORM = "CUDA"
N = 6
KJ = unit.kilojoule_per_mole
KJNM = unit.kilojoule_per_mole / unit.nanometer
# Off-equilibrium positions so PE > 0 and forces are non-trivial.
POS = [mm.Vec3(0.15 * i, 0.03 * ((-1) ** i), 0.01 * i) for i in range(N)]


def _add_forces(system):
    bond = mm.HarmonicBondForce()
    for i in range(N - 1):
        bond.addBond(i, i + 1, 0.15, 5000.0)   # r0=0.15 nm, k=5000 (moderate forces)
    system.addForce(bond)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for i in range(N):
        nb.addParticle(0.2 if i % 2 == 0 else -0.2, 0.30, 0.6)
    system.addForce(nb)


def _make_system():
    s = mm.System()
    for _ in range(N):
        s.addParticle(12.0)
    _add_forces(s)
    return s


def main(platform):
    # --- reference: same forces, NO GluedForce ---
    sref = _make_system()
    cref = mm.Context(sref, mm.VerletIntegrator(0.001), platform)
    cref.setPositions(POS)
    stref = cref.getState(getEnergy=True, getForces=True)
    PE_ref = stref.getPotentialEnergy().value_in_unit(KJ)
    F_ref = stref.getForces()

    # --- GLUED with a CV_ENERGY (+ test bias gradient 1.0 to exercise the scatter) ---
    sg = _make_system()
    f = gp.GluedForce()
    eidx = f.addCollectiveVariable(gp.GluedForce.CV_ENERGY, mm.vectori(), mm.vectord())
    f.setTestBiasGradients(mm.vectord([1.0]))
    sg.addForce(f)
    cg = mm.Context(sg, mm.VerletIntegrator(0.001), platform)
    cg.setPositions(POS)
    stg = cg.getState(getForces=True)
    U = list(f.getLastCVValues(cg))[eidx]

    # 1. value
    assert abs(U - PE_ref) < max(1e-2, 1e-4 * abs(PE_ref)), \
        f"CV_ENERGY value {U:.6f} != PE_ref {PE_ref:.6f}"
    print(f"  test_energy_value: OK  (U={U:.4f} kJ/mol, PE_ref={PE_ref:.4f})")

    # 2. gradient: FD of U vs -F_ref on a few atoms/components
    dx = 1e-3
    max_err = 0.0
    for a in (0, 2, 5):
        for comp in range(3):
            pp = list(POS); v = list(pp[a]); v[comp] += dx; pp[a] = mm.Vec3(*v)
            cg.setPositions(pp); cg.getState(getForces=True)
            Up = list(f.getLastCVValues(cg))[eidx]
            pm = list(POS); v = list(pm[a]); v[comp] -= dx; pm[a] = mm.Vec3(*v)
            cg.setPositions(pm); cg.getState(getForces=True)
            Um = list(f.getLastCVValues(cg))[eidx]
            dUdx = (Up - Um) / (2.0 * dx)
            expected = -F_ref[a][comp].value_in_unit(KJNM)   # dU/dx = -F
            err = abs(dUdx - expected)
            tol = max(5.0, 0.02 * abs(expected))
            assert err < tol, f"atom {a} comp {comp}: dU/dx={dUdx:.3f}, -F={expected:.3f} (err {err:.3f})"
            max_err = max(max_err, err)
    print(f"  test_energy_gradient_vs_minusF: OK  (max FD err {max_err:.3f} kJ/mol/nm)")

    # 3. scatter: GLUED adds +F (so total force = F_ref + F_ref = 2*F_ref)
    cg.setPositions(POS)
    Fg = cg.getState(getForces=True).getForces()
    max_ferr = 0.0
    for a in range(N):
        for comp in range(3):
            fg = Fg[a][comp].value_in_unit(KJNM)
            fr = F_ref[a][comp].value_in_unit(KJNM)
            # GLUED scatter contributes +fr, so fg ≈ 2*fr
            max_ferr = max(max_ferr, abs((fg - fr) - fr))
    assert max_ferr < max(5.0, 0.01 * max(abs(F_ref[a][c].value_in_unit(KJNM))
                          for a in range(N) for c in range(3))), \
        f"scatter force mismatch (max {max_ferr:.3f})"
    print(f"  test_energy_scatter_force: OK  (max |F_glued-2F_ref| {max_ferr:.3f} kJ/mol/nm)")

    print("All CV_ENERGY tests passed.")


if __name__ == "__main__":
    try:
        plat = mm.Platform.getPlatformByName(PLATFORM)
    except Exception:
        print("CUDA platform not available — skipping CV_ENERGY tests.")
        sys.exit(0)
    print("Stage 1 — CV_ENERGY tests (CUDA platform):")
    main(plat)
