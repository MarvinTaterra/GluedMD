# Examples

These recipes assume an OpenMM `System`, `Topology`, and starting `positions` are already set up. All distances are in nm, angles in radians, energies in kJ/mol.

```python
import openmm as mm
import openmm.app as app
import glued
from COLVARReporter import COLVARReporter
```

---

## Well-tempered Metadynamics on a dihedral

```python
PHI = [4, 6, 8, 14]   # φ backbone atoms (0-based)

force = glued.Force(pbc=True, temperature=300.0)
phi = force.add_dihedral(PHI)

force.add_metad(
    phi, sigma=0.35, height=1.0, pace=500,
    grid_min=-3.14159, grid_max=3.14159,
    bins=360, periodic=True, gamma=15.0,
)
system.addForce(force)

integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
simulation = app.Simulation(topology, system, integ,
                             mm.Platform.getPlatformByName("CUDA"))
simulation.context.setPositions(positions)
simulation.reporters.append(
    COLVARReporter('colvar.dat', 500, force, cvNames=['phi'])
)
simulation.step(10_000_000)
```

---

## 2-D Metadynamics (φ/ψ Ramachandran)

```python
PHI = [4, 6,  8, 14]
PSI = [6, 8, 14, 16]

force = glued.Force(pbc=True, temperature=300.0)
phi = force.add_dihedral(PHI)
psi = force.add_dihedral(PSI)

force.add_metad(
    [phi, psi], sigma=0.35, height=0.5, pace=500,
    grid_min=-3.14159, grid_max=3.14159,
    bins=180, periodic=True, gamma=15.0,
)
```

---

## OPES

```python
force = glued.Force(pbc=True, temperature=300.0)
phi = force.add_dihedral([4, 6, 8, 14])

force.add_opes(phi, sigma=0.35, gamma=10.0, pace=500)

system.addForce(force)
simulation = app.Simulation(topology, system,
                             mm.LangevinMiddleIntegrator(300, 1.0, 0.002),
                             mm.Platform.getPlatformByName("CUDA"))
simulation.context.setPositions(positions)
simulation.step(10_000_000)

# Check convergence
zed, rct, nker, neff = force.getOPESMetrics(simulation.context, 0)
print(f"OPES: nker={nker:.0f}  neff={neff:.0f}  rct={rct:.3f} kJ/mol")
```

---

## Bias state checkpoint / restore

```python
# ── Save during a run ────────────────────────────────────────────────────────
simulation.step(5_000_000)

blob = force.getBiasState()
with open("metad_checkpoint.bin", "wb") as fh:
    fh.write(blob)

# Also save the OpenMM state (positions, velocities, box)
simulation.saveState("state.xml")

# ── Resume in a new script ───────────────────────────────────────────────────
force2 = glued.Force(pbc=True, temperature=300.0)
phi2 = force2.add_dihedral([4, 6, 8, 14])
force2.add_metad(phi2, sigma=0.35, height=1.0, pace=500,
                 grid_min=-3.14159, grid_max=3.14159,
                 bins=360, periodic=True, gamma=15.0)
system2.addForce(force2)

simulation2 = app.Simulation(topology, system2, integ, platform)
simulation2.loadState("state.xml")

with open("metad_checkpoint.bin", "rb") as fh:
    force2.setBiasState(fh.read())

simulation2.step(5_000_000)   # continues from where we left off
```

---

## COLVAR reporter

```python
force = glued.Force(pbc=True, temperature=300.0)
phi = force.add_dihedral([4, 6,  8, 14])
psi = force.add_dihedral([6, 8, 14, 16])
force.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500)
system.addForce(force)

simulation = app.Simulation(topology, system, integ, platform)
simulation.reporters.append(
    COLVARReporter('colvar.dat', 100, force, cvNames=['phi', 'psi'])
)
simulation.step(1_000_000)
```

Output file format:
```
#! FIELDS time phi psi
  0.00000    -1.23456     2.34567
  0.20000    -1.19321     2.29810
```

To append to an existing COLVAR file when restarting:
```python
simulation.reporters.append(
    COLVARReporter('colvar.dat', 100, force, append=True)
)
```

---

## H-REUS (Hamiltonian Replica Exchange Umbrella Sampling)

```python
from ReplicaExchange import ReplicaExchange

kT      = glued.GluedForce.kTFromTemperature(300.0)
targets = [-2.0, -1.0, 0.0, 1.0]   # window centres (rad)
replicas = []

for i, target in enumerate(targets):
    sys_i = build_system()   # fresh copy of the OpenMM System each time
    f = glued.Force(pbc=True)
    cv = f.add_dihedral(PHI)
    f.add_harmonic(cv, kappa=200.0, at=target, periodic=True)
    sys_i.addForce(f)
    ctx = mm.Context(sys_i, mm.LangevinMiddleIntegrator(300, 1.0, 0.002),
                     mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(start_positions[i])
    replicas.append((ctx, f))

re = ReplicaExchange(replicas, mode="H-REUS", kT=kT, seed=42)
re.run(n_cycles=500, steps_per_cycle=500)   # 500 × 500 = 250 000 MD steps per replica

print(f"Overall acceptance rate: {re.acceptance_rate:.1%}")
for i in range(len(targets) - 1):
    print(f"  pair ({i},{i+1}): {re.pair_acceptance_rate(i, i+1):.1%}")
```

---

## T-REMD

```python
from ReplicaExchange import ReplicaExchange

temperatures = [300, 320, 340, 360]   # K
replicas = []

for T in temperatures:
    sys_T = build_system()
    f = glued.Force(pbc=True, group=1)   # group=1 isolates bias from MM energy
    cv = f.add_dihedral(PHI)
    f.add_abmd(cv, kappa=100.0, to=-1.5)   # optional — same bias at all T
    sys_T.addForce(f)
    ctx = mm.Context(sys_T, mm.LangevinMiddleIntegrator(T, 1.0, 0.002),
                     mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(start_positions)
    replicas.append((ctx, f))

re = ReplicaExchange(
    replicas, mode="T-REMD",
    temperatures=temperatures, bias_force_group=1, seed=99
)
re.run(n_cycles=500, steps_per_cycle=1000)
```

---

## Multi-walker Metadynamics

Multiple walkers deposit into a single shared MetaD grid on the GPU with no CPU merge step.

```python
def make_walker(system):
    f = glued.Force(pbc=True, temperature=300.0)
    cv = f.add_dihedral(PHI)
    f.add_metad(cv, sigma=0.35, height=1.0, pace=500,
                grid_min=-3.14159, grid_max=3.14159,
                bins=360, periodic=True, gamma=15.0)
    system.addForce(f)
    ctx = mm.Context(system, mm.LangevinMiddleIntegrator(300, 1, 0.002), platform)
    return ctx, f

ctx0, f0 = make_walker(build_system())   # primary — owns the shared grid
ctx1, f1 = make_walker(build_system())   # secondary

# Wire the shared grid (must happen after both contexts are created)
ptrs = f0.getMultiWalkerPtrs(ctx0, 0)   # bias index 0
f1.setMultiWalkerPtrs(ctx1, 0, ptrs)

# Run both walkers — deposits from both go into the shared grid
for _ in range(10000):
    ctx0.getIntegrator().step(1)
    ctx1.getIntegrator().step(1)
```

---

## Applying a precomputed external bias

Load a 1-D free energy profile (e.g., from a separate calculation) and apply it as a bias.

```python
import numpy as np

data    = np.loadtxt("pmf_phi.dat")   # columns: [phi_rad, pmf_kJmol]
phi_vals = data[:, 0]
pmf      = data[:, 1]

force = glued.Force(pbc=True)
phi = force.add_dihedral(PHI)

origin  = float(phi_vals[0])
maximum = float(phi_vals[-1])
n_bins  = len(pmf) - 1   # non-periodic: actualPoints = numBins + 1

force.addBias(
    glued.GluedForce.BIAS_EXTERNAL,
    [phi],
    [origin, maximum] + list(pmf),   # origin, max, then grid values
    [n_bins, 0]                       # numBins, isPeriodic=False
)
```

---

## Expression CV

Combine existing CVs algebraically using the Lepton parser.

```python
force = glued.Force(pbc=True)

d_oh = force.add_distance([O, H])
d_nh = force.add_distance([N, H])

# Proton transfer coordinate: d(O-H) − d(N-H)
pt = force.add_expression("cv0 - cv1", [d_oh, d_nh])

force.add_harmonic(pt, kappa=500.0, at=0.0)
```

---

## XML serialization round-trip

```python
# Build and run
system = build_system_with_glued_force()
# ... simulation ...

# Serialize
xml = mm.XmlSerializer.serialize(system)
with open("system.xml", "w") as fh:
    fh.write(xml)

# Deserialize
with open("system.xml") as fh:
    system2 = mm.XmlSerializer.deserialize(fh.read())

# Recover GluedForce from the deserialized system
import gluedplugin as gp
force2 = None
for i in range(system2.getNumForces()):
    f = gp.GluedForce.cast(system2.getForce(i))
    if f is not None:
        force2 = f
        break

# Restore bias state separately (grid data is not in the XML)
with open("bias.bin", "rb") as fh:
    force2.setBiasState(fh.read())
```
