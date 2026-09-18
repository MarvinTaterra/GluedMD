# Changelog

## Unreleased — audit remediation

Fixes for the findings of the September 2026 audit. Behaviour changes that
existing scripts may notice:

- **Bias checkpoints are version 2.** They now contain everything needed to
  continue an OPES run exactly (normalization, weight sums, deposit count,
  learned bandwidths) plus a hash of the force configuration. Version-1 blobs
  cannot be restored. A rejected blob leaves the current state untouched.
  `getBiasState(context)` / `setBiasState(blob, context)` take an optional
  Context; the Context-free form requires exactly one live Context.
- **`getMultiWalkerPtrs` / `setMultiWalkerPtrs` are removed.** Walkers share a
  bias through `MultiWalkerPool`, which merges MetaD/PBMetaD grid increments at
  `sync_interval` and passes OPES-family state from walker to walker one step
  at a time. `sync_interval=0` means independent walkers.
- **Bias history is committed once per step, at the current coordinates.**
  Energy queries, minimization and rejected trial moves no longer alter OPES,
  ABMD or extended-Lagrangian history. Custom integrators must call
  `addUpdateContextState()` once per step. Cached forces are invalidated when
  the bias potential changes.
- **Invalid configurations fail early.** `addBias` validates parameter layouts,
  strides, grid sizes and finiteness; CV definitions are checked against the
  System at Context creation. Native exceptions surface as Python
  `RuntimeError`.
- **OPES on dihedral and puckering-phase CVs is periodic**, and the adaptive
  bandwidth follows PLUMED's OPES_METAD recurrence. Deposition continues while
  merging is possible when the kernel table is full; an unmergeable deposit
  into a full table raises instead of being dropped silently.
- **OPES diagnostics take the global bias index** returned by `add_opes`.
- **Replica exchange** evaluates all four cross energies including every bias
  and the pressure–volume term for isotropic NPT; `bias_force_group` is no
  longer needed. Unsupported barostats are rejected.
- The energy CV follows global parameter changes and excludes all GluedForces
  from its inner Context; expression CVs see current energy/Torch CV values.
- Grid biases apply zero force outside a nonperiodic grid; path CVs no longer
  collapse far from the reference frames; XML serialization keeps the force
  name; step indices are 64-bit.
- Tests: the suite collects in a fresh checkout (missing datasets skip), the
  Reference platform is no longer used as a silent fallback for numerical
  tests, CTest runs the pytest suite, and `tests/test_audit_regressions.py`
  covers the findings above.
