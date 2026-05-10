# Bias Methods

Biases are registered after all CVs have been added. Each call returns a 0-based bias index used for checkpoint and multi-walker operations.

**Recommended** — use named methods on `glued.Force`:

```python
import glued
force = glued.Force(pbc=True, temperature=300.0)
force.add_harmonic(cv_idx, kappa=100.0, at=0.0)
```

**Low-level** — use `addBias` directly (also works on `glued.Force`):

```python
import gluedplugin as gp
bias_idx = force.addBias(gp.GluedForce.BIAS_HARMONIC, [cv_idx], [0.0, 100.0], [])
```

Multiple biases can be stacked on the same force object. They are evaluated in registration order and their `cvBiasGradients` contributions are accumulated additively before the chain-rule scatter kernel fires.

---

## BIAS_HARMONIC (1)

Harmonic restraint: `V = Σ_d (k_d/2) * (s_d − s0_d)²`

| `parameters` | `[at_0, k_0, at_1, k_1, ...]` — equilibrium position and force constant for each CV, interleaved |
| `integerParameters` | `[]` |

```python
# Restrain phi to −1.0 rad with k = 200 kJ/mol/rad²
force.add_harmonic(phi_idx, kappa=200.0, at=-1.0)

# Multi-CV: restrain both phi and psi simultaneously
force.add_harmonic([phi_idx, psi_idx], kappa=200.0, at=[-1.0, 1.2])

# For periodic CVs (dihedrals) wrap the difference to (−π, π]
force.add_harmonic(phi_idx, kappa=200.0, at=-1.0, periodic=True)
```

**Low-level** parameter layout: `[k_0, at_0, k_1, at_1, ...]` — force constant and equilibrium position, interleaved per CV.

---

## BIAS_MOVING_RESTRAINT (2)

Harmonic restraint whose equilibrium position and/or force constant changes linearly between user-specified schedule points.

| `parameters` | `[step_0, at_0_cv0, k_0_cv0, step_1, at_1_cv0, k_1_cv0, ...]` — schedule as a flat sequence of `(step, at, k)` triples per CV (all CV0 schedule first, then CV1, etc.) |
| `integerParameters` | `[numSchedulePoints]` — number of (step, at, k) triples per CV |

```python
# Sweep phi from −π to 0 over 10000 steps
force.addBias(gp.GluedForce.BIAS_MOVING_RESTRAINT,
              [phi_idx],
              [0.0,    100.0, −3.14159,   # step=0:  k=100, at=−π
               10000.0, 100.0,  0.0],      # step=10000: k=100, at=0
              [2])                          # 2 schedule points
```

---

## BIAS_METAD (3)

Well-tempered Metadynamics. Deposits Gaussian hills on a fixed grid at regular intervals.

`V(s) = Σ_deposited  h_eff * exp[−Σ_d (s_d − c_d)²/(2σ_d²)]`

where `h_eff = height * exp[−V(s) / (kT*(γ−1))]` (well-tempering).

| `parameters` | `[height, sigma_0, gamma, kT, origin_0, max_0, sigma_1, origin_1, max_1, ...]` — hill height (kJ/mol), sigma per CV (same units as CV), biasfactor γ, thermal energy kT (kJ/mol), then grid bounds per CV |
| `integerParameters` | `[pace, numBins_0, isPeriodic_0, numBins_1, isPeriodic_1, ...]` — deposition interval (steps), grid bins and periodicity per CV |

```python
force.add_metad(phi_idx, sigma=0.35, height=1.0, pace=500,
                grid_min=-3.14159, grid_max=3.14159,
                bins=360, periodic=True, gamma=15.0)
```

Set `gamma=None` or a very large value (e.g., 1e6) to recover non-tempered MetaD.

---

## BIAS_PBMETAD (4)

Parallel Bias Metadynamics (Pfaendtner & Bonomi 2015). Deposits independent 1-D Gaussians for each CV and combines them multiplicatively, enabling efficient multi-CV sampling.

| `parameters` | `[height, gamma, kT, sigma_0, origin_0, max_0, sigma_1, origin_1, max_1, ...]` — global height, biasfactor, kT; then per-CV: sigma, grid origin, grid max |
| `integerParameters` | `[pace, numBins_0, isPeriodic_0, numBins_1, isPeriodic_1, ...]` |

```python
force.addBias(gp.GluedForce.BIAS_PBMETAD,
              [phi_idx, psi_idx],
              [1.0, 15.0, 2.479,      # height, gamma, kT
               0.35, −3.14159, 3.14159,  # phi: sigma, origin, max
               0.35, −3.14159, 3.14159], # psi: sigma, origin, max
              [500, 360, 1, 360, 1])      # pace, bins_phi, per_phi, bins_psi, per_psi
```

---

## BIAS_OPES (5)

On-the-fly Probability Enhanced Sampling (Invernizzi & Parrinello 2020). Builds a compressed non-parametric kernel density estimate of the CV probability distribution and uses it to construct a flat-histogram bias.

| `parameters` | `[kT, gamma, sigma_0, sigmaMin_0, sigma_1, sigmaMin_1, ...]` — thermal energy, biasfactor, initial bandwidth and minimum bandwidth per CV |
| `integerParameters` | `[variant, pace, maxKernels]` — variant: `0` = standard OPES_METAD, `1` = uniform target + fixed σ, `2` = OPES_METAD_EXPLORE; pace = deposition interval; maxKernels = kernel list capacity |

```python
force.add_opes(phi_idx, sigma=0.35, gamma=15.0, pace=500)
# temperature is inherited from Force(temperature=300.0);
# sigma_min defaults to 5% of sigma; max_kernels defaults to 100 000
```

### Modes

OPES has three sub-variants selectable via the `mode` keyword on `add_opes`
(or the `variant` integer in the low-level API):

* **`mode='metad'`** *(variant=0, default)* — standard OPES_METAD. Targets the
  well-tempered distribution `p^WT ∝ P(s)^(1/γ)` by reweighting the unbiased
  distribution `P(s)`. Adaptive Silverman σ. Bias prefactor `(γ−1)/γ`.
  Reference: Invernizzi & Parrinello, *JPCL* **11**:2731 (2020).

* **`mode='explore'`** *(variant=2)* — OPES_METAD_EXPLORE. Targets the same
  WT distribution but estimates it directly from biased samples (plain KDE)
  rather than via reweighting. Bias prefactor `(γ−1)`. Use when CVs may be
  degenerate or the barrier is unknown. Reference: Invernizzi, Piaggi &
  Parrinello, *JCTC* **18**:3988 (2022).

  The user-supplied σ is internally broadened by √γ to target the wider WT
  distribution — pass the same σ you would use with METAD. PLUMED users
  matching `SIGMA=0.05 BIASFACTOR=10` should pass `sigma=0.05` here too;
  the broadening is applied automatically.

* **`mode='fixed_uniform'`** *(variant=1)* — uniform target (γ→∞ equivalent)
  with fixed σ. Specialized; rarely needed.

```python
# Standard OPES_METAD on alanine dipeptide
force.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500, mode='metad')

# OPES_METAD_EXPLORE — same call, different mode
force.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500, mode='explore')

# Fully-adaptive σ (PLUMED's SIGMA=ADAPTIVE) — Welford running variance
# updated every step; deposition is gated until adaptive_sigma_stride
# steps have elapsed. Defaults to 10×pace if omitted.
force.add_opes([phi, psi], sigma=None, gamma=10.0, pace=500,
               adaptive_sigma_stride=5000)   # 'adaptive' is also accepted
```

### When to use which σ mode

| σ flavor | When | API |
|---|---|---|
| **Explicit fixed** | You know a sensible σ from prior pilots — e.g. σ ≈ 5 % of the CV range | `sigma=0.5` |
| **Mixed-adaptive** *(default)* | Explicit σ rescaled per deposit by Silverman's rule. Robust default | `sigma=0.5` (any positive value) |
| **Fully adaptive** | First exploration of a new CV; CV scale is unknown; want PLUMED-like ADAPTIVE behaviour | `sigma=None` or `sigma='adaptive'` |

**Convergence diagnostics** (query at runtime):

```python
zed, rct, nker, neff = force.getOPESMetrics(context, bias_idx)
# zed:  sum_uprob / (KDEnorm·nker)  — bounded near order 1 at convergence
# rct:  kT · log(sum_weights / counter)  — c(t) reweighting indicator (kJ/mol)
# nker: number of compressed kernels
# neff: effective sample size
```

**Auto-stop** — use `OPESConvergenceReporter` to halt the simulation automatically once rct is flat, without running longer than necessary:

```python
from OPESConvergenceReporter import OPESConvergenceReporter

reporter = OPESConvergenceReporter(force, tol=0.05, check_interval=2000,
                                    post_convergence_steps=100_000)
reporter.run(simulation, max_steps=10_000_000)
print(f"Converged at step {reporter.converged_at_step}")
```

See [Python API — OPESConvergenceReporter](python_api.md#opesconvergencereporter) for full details.

---

## BIAS_EXTERNAL (6)

Apply a precomputed bias from a regular D-dimensional grid (e.g., a PMF computed by free energy perturbation or imported from another code).

| `parameters` | `[origin_0, ..., origin_{D-1}, max_0, ..., max_{D-1}, grid_val_0, ..., grid_val_{G-1}]` — grid extents then flattened grid values in kJ/mol |
| `integerParameters` | `[numBins_0, ..., numBins_{D-1}, isPeriodic_0, ..., isPeriodic_{D-1}]` |

Grid layout:
- `actualPoints_d = numBins_d + (isPeriodic_d ? 0 : 1)`
- Total points G = Π actualPoints_d
- Flat index: `sum_d(bin_d * stride_d)` where `stride_0 = 1`, `stride_d = stride_{d-1} * actualPoints_{d-1}`

```python
# 1-D example: 5-bin periodic grid from 0 to 2π
import numpy as np
grid = np.sin(np.linspace(0, 2*np.pi, 5, endpoint=False))
force.addBias(gp.GluedForce.BIAS_EXTERNAL,
              [phi_idx],
              [0.0, 2*np.pi] + list(grid),
              [5, 1])   # 5 bins, periodic
```

---

## BIAS_ABMD (7)

Adiabatic Bias MD (Marchi & Ballone 1999). A one-sided restraint that ratchets the CV floor forward as the simulation progresses: `V = κ/2 * max(0, ρ_min − s)²`. The ratchet floor `ρ_min` is updated on the GPU each step.

| `parameters` | `[kappa, s0]` — force constant (kJ/mol/unit²) and initial floor position |
| `integerParameters` | `[]` |

```python
force.add_abmd(phi_idx, kappa=200.0, to=-1.5)
```

---

## BIAS_UPPER_WALL (8) / BIAS_LOWER_WALL (9)

Polynomial wall that activates only on one side of a threshold:

`V = κ * max(0, δ)^n * exp(ε * δ)`

where `δ = s − at` for UPPER_WALL and `δ = at − s` for LOWER_WALL.

| `parameters` | `[kappa, at, epsilon, n]` — force constant, threshold, exponential prefactor, polynomial exponent |
| `integerParameters` | `[]` |

```python
# Keep phi below 0.5 rad
force.add_upper_wall(phi_idx, at=0.5, kappa=100.0)          # n=2, eps=1.0 are defaults

# Keep phi above −0.5 rad
force.add_lower_wall(phi_idx, at=-0.5, kappa=100.0)

# Custom exponent and exponential prefactor (low-level)
force.addBias(gp.GluedForce.BIAS_UPPER_WALL,
              [phi_idx],
              [100.0, 0.5, 0.5, 4.0],   # kappa, at, eps, n
              [])
```

---

## BIAS_LINEAR (10)

Linear coupling: `V = Σ_d k_d * s_d`. Constant gradient, no grid.

| `parameters` | `[k_0, k_1, ...]` — one coupling constant per CV |
| `integerParameters` | `[]` |

```python
force.add_linear(phi_idx, k=5.0)

# Multi-CV (low-level)
force.addBias(gp.GluedForce.BIAS_LINEAR, [phi_idx, psi_idx], [5.0, -2.0], [])
```

---

## BIAS_OPES_EXPANDED (11)

OPES Expanded Ensemble (Invernizzi et al. 2022). Simultaneously targets a mixture of thermodynamic states (different temperatures, pressures, or Hamiltonians). CVs should be energy-like quantities (potential energy at different temperatures, etc.).

| `parameters` | `[kT_0, kT_1, ..., gamma]` — thermal energies for each thermodynamic state, then the biasfactor |
| `integerParameters` | `[pace]` |

---

## BIAS_EXT_LAGRANGIAN (12)

Extended Lagrangian / Adiabatic Free Energy Dynamics (AFED). Introduces a fictitious particle `s` coupled to the CV via a stiff spring. The particle evolves as a separate degree of freedom, decoupled from the fast molecular vibrations.

| `parameters` | `[kappa, mass_s]` — coupling constant (kJ/mol/unit²) and fictitious mass (kJ/mol·ps²/nm²) |
| `integerParameters` | `[]` |

```python
force.addBias(gp.GluedForce.BIAS_EXT_LAGRANGIAN,
              [phi_idx],
              [5000.0, 50.0],   # kappa, mass
              [])
```

The extended variable `s` is initialised to the CV value on the first `updateState()` call and then evolves under Langevin dynamics driven by the coupling potential.

---

## BIAS_MAXENT (13)

Maximum Entropy / Experiment-Directed Simulation. Adaptively adjusts a linear coupling λ so that the ensemble average of the CV matches a target value, subject to a Gaussian or uniform error model.

| `parameters` | `[kT, sigma, alpha, at_0, kappa_0, tau_0, at_1, kappa_1, tau_1, ...]` — thermal energy, noise level, learning rate; then per-CV: target value, initial κ, adaptation time constant |
| `integerParameters` | `[pace, constraintType, errorType]` — pace: adaptation interval; constraintType: 0 = EQUAL, 1 = GREATER, 2 = LESS; errorType: 0 = GAUSSIAN, 1 = FLAT |

```python
force.addBias(gp.GluedForce.BIAS_MAXENT,
              [phi_idx],
              [2.479,   # kT
               0.0,     # sigma (Gaussian noise)
               1.0,     # alpha (learning rate)
               −1.0,    # target phi
               10.0,    # initial kappa
               1000.0], # tau
              [500,     # pace
               0,       # EQUAL constraint
               0])      # GAUSSIAN error
```

---

## BIAS_EDS (14)

Experiment-Directed Simulation (White & Voth 2014). An adaptive linear restraint that steers the CV mean toward a target by adjusting the coupling λ via a self-consistent gradient descent update.

| `parameters` | `[target_0, max_range_0, kT, target_1, max_range_1, ...]` — target mean, maximum λ range, thermal energy; then per-CV: target, max_range |
| `integerParameters` | `[pace]` — adaptation interval |

```python
force.add_eds(phi_idx, target=-1.0, max_range=5.0, pace=500)
# temperature is inherited from Force(temperature=300.0)

# Multi-CV (low-level)
force.addBias(gp.GluedForce.BIAS_EDS,
              [phi_idx, psi_idx],
              [−1.0, 5.0,   # phi: target, max_range
               1.2,  5.0,   # psi: target, max_range
               2.479],      # kT
              [500])
```

---

## Stacking multiple biases

Multiple biases may be applied to the same force object. They run sequentially and accumulate into the shared `cvBiasGradients` array before the scatter kernel fires. There is no interaction between bias objects themselves.

```python
# MetaD on phi + harmonic restraint on psi
force.add_metad(phi_idx, sigma=0.35, height=1.0, pace=500,
                grid_min=-3.14159, grid_max=3.14159, bins=360, periodic=True, gamma=15.0)
force.add_harmonic(psi_idx, kappa=50.0, at=0.0)
```

## Bias state checkpoint / restore

All stateful biases (MetaD grid, OPES kernels, ABMD floor, EDS λ, ExtLag s, MaxEnt λ) can be checkpointed and restored without rebuilding the simulation:

```python
# Save
blob = force.getBiasState()   # returns Python bytes

# Restore (after e.g. reloading a serialized system)
force.setBiasState(blob)
```

The binary format includes a `GLUED` magic header and version byte so that stale blobs are detected and rejected.
