"""
ReplicaExchange — GLUED replica exchange driver.

Supports two modes:

  H-REUS  (Hamiltonian Replica Exchange Umbrella Sampling)
      All replicas run at the *same* temperature but with different bias
      parameters (e.g. different harmonic restraint centres).

  T-REMD  (Temperature Replica Exchange MD)
      All replicas share the same Hamiltonian / bias but run at different
      temperatures:
          re = ReplicaExchange(..., mode="T-REMD",
                                   temperatures=[300, 320, 340, 360])

Both modes use the same Metropolis criterion on the complete reduced potentials
of the four (replica, configuration) combinations, including every bias and,
under an isotropic barostat, the pressure-volume term. A bias therefore never
needs to be isolated in its own force group for the exchange to be correct.

Usage:
    from ReplicaExchange import ReplicaExchange
    import gluedplugin as gp

    replicas = [(ctx0, f0), (ctx1, f1), (ctx2, f2), (ctx3, f3)]
    re = ReplicaExchange(replicas, mode="H-REUS", kT=2.479)
    re.run(n_cycles=200, steps_per_cycle=500)
    print("acceptance rate:", re.acceptance_rate)
"""

import math
import random
import warnings

import numpy as np
import openmm as mm
from openmm import unit


def _total_energy_kJ(ctx):
    """Return total potential energy in kJ/mol."""
    s = ctx.getState(getEnergy=True)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _pressure_kJ_per_nm3(ctx):
    """Reference pressure of an isotropic MonteCarloBarostat, or 0 without one."""
    barostats = [f for f in ctx.getSystem().getForces() if "Barostat" in type(f).__name__]
    if not barostats:
        return 0.0
    if len(barostats) != 1 or type(barostats[0]) is not mm.MonteCarloBarostat:
        raise ValueError("replica exchange supports NVT or an isotropic MonteCarloBarostat")
    pressure_bar = ctx.getParameter(mm.MonteCarloBarostat.Pressure())
    return pressure_bar * _BAR_TO_KJ_PER_MOL_NM3


_BAR_TO_KJ_PER_MOL_NM3 = 0.0602214076   # 1 bar = 1e5 J/m^3 = 6.02214076e-2 kJ/mol/nm^3


def _volume_nm3(state):
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    return abs(float(np.linalg.det(box)))


def exchange_log_acceptance(ctx_i, ctx_j, state_i, state_j, beta_i, beta_j):
    """Log of the Metropolis acceptance ratio for swapping the configurations of
    replicas i and j.

    ``state_i``/``state_j`` are the replicas' current States (positions,
    energy, box). Each Context evaluates the other's configuration with its own
    Hamiltonian, so replica-specific biases are handled exactly:

        Δ = β_i [U_i(x_i) − U_i(x_j) + p_i (V_i − V_j)]
          + β_j [U_j(x_j) − U_j(x_i) + p_j (V_j − V_i)]

    Both Contexts are restored to their own configuration before returning.
    """
    pressure_i, pressure_j = _pressure_kJ_per_nm3(ctx_i), _pressure_kJ_per_nm3(ctx_j)
    box_i, box_j = state_i.getPeriodicBoxVectors(), state_j.getPeriodicBoxVectors()
    x_i, x_j = state_i.getPositions(), state_j.getPositions()
    volume_i, volume_j = _volume_nm3(state_i), _volume_nm3(state_j)
    e_ii = state_i.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_jj = state_j.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    try:
        ctx_i.setPeriodicBoxVectors(*box_j)
        ctx_i.setPositions(x_j)
        e_ij = _total_energy_kJ(ctx_i)
    finally:
        ctx_i.setPeriodicBoxVectors(*box_i)
        ctx_i.setPositions(x_i)
    try:
        ctx_j.setPeriodicBoxVectors(*box_i)
        ctx_j.setPositions(x_i)
        e_ji = _total_energy_kJ(ctx_j)
    finally:
        ctx_j.setPeriodicBoxVectors(*box_j)
        ctx_j.setPositions(x_j)
    delta = (beta_i * (e_ii - e_ij + pressure_i * (volume_i - volume_j))
             + beta_j * (e_jj - e_ji + pressure_j * (volume_j - volume_i)))
    if not math.isfinite(delta):
        raise ValueError("nonfinite replica-exchange reduced potential")
    return delta


class ReplicaExchange:
    """
    Drives replica exchange between N OpenMM contexts.

    Parameters
    ----------
    replicas : list of (Context, GluedForce)
        One entry per replica.  Replicas are ordered by the exchange ladder
        (neighbouring replicas are proposed first in "neighbor" scheme).
    mode : str
        "H-REUS" or "T-REMD".
    kT : float
        Thermal energy in kJ/mol.  Required for H-REUS; used for all replicas
        (same temperature).
    temperatures : list of float
        Temperatures in Kelvin, one per replica.  Required for T-REMD.
    bias_force_group : int or None
        Accepted for backward compatibility and not used: the acceptance test
        always uses the complete potential energy.
    scheme : str
        "neighbor" — try only adjacent (i, i+1) pairs each cycle (fast, standard).
        "all"      — try every unique pair each cycle.
    seed : int or None
        RNG seed for reproducibility.
    """

    _GAS_CONSTANT_kJmol = 8.314462618e-3   # kJ mol⁻¹ K⁻¹

    def __init__(self, replicas, mode="H-REUS", *,
                 kT=None, temperatures=None,
                 bias_force_group=None,
                 scheme="neighbor", seed=None):
        if not replicas:
            raise ValueError("replicas list must be non-empty")
        self._replicas = list(replicas)
        self._n = len(replicas)
        self._mode = mode.upper()
        self._scheme = scheme
        self._bias_group = bias_force_group
        self._rng = random.Random(seed)
        self._n_attempts = 0
        self._n_accepted  = 0
        # per-pair acceptance counters: key (i,j) i<j
        self._pair_attempts = {}
        self._pair_accepted = {}

        if self._mode == "H-REUS":
            if kT is None or not math.isfinite(kT) or kT <= 0:
                raise ValueError("H-REUS requires kT")
            self._betas = [1.0 / kT] * self._n

        elif self._mode == "T-REMD":
            if temperatures is None or len(temperatures) != self._n:
                raise ValueError("T-REMD requires temperatures list, one per replica")
            self._temperatures = list(temperatures)
            if any(not math.isfinite(t) or t <= 0 for t in self._temperatures):
                raise ValueError("replica temperatures must be finite and positive")
            self._betas = [1.0 / (self._GAS_CONSTANT_kJmol * T)
                           for T in temperatures]
            # Verify integrator temperatures match
            for i, (ctx, _) in enumerate(self._replicas):
                integ = ctx.getIntegrator()
                if hasattr(integ, "getTemperature"):
                    T_integ = integ.getTemperature().value_in_unit(unit.kelvin)
                    if abs(T_integ - temperatures[i]) > 1.0:
                        warnings.warn(
                            f"Replica {i}: integrator temperature {T_integ:.1f} K "
                            f"does not match RE temperature {temperatures[i]:.1f} K. "
                            "Velocities will be rescaled after each accepted swap.",
                            stacklevel=2)
        else:
            raise ValueError(f"Unknown mode '{mode}'; use 'H-REUS' or 'T-REMD'")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, n_cycles, steps_per_cycle=500):
        """
        Run *n_cycles* of replica exchange.

        Each cycle: run `steps_per_cycle` MD steps in every replica (serially),
        then attempt swaps between all proposed pairs.
        """
        for _ in range(n_cycles):
            self._step_all(steps_per_cycle)
            self._attempt_swaps()

    @property
    def acceptance_rate(self):
        """Overall fraction of accepted swap proposals."""
        if self._n_attempts == 0:
            return 0.0
        return self._n_accepted / self._n_attempts

    def pair_acceptance_rate(self, i, j):
        """Acceptance rate for the (i, j) pair (i < j)."""
        key = (min(i, j), max(i, j))
        n = self._pair_attempts.get(key, 0)
        if n == 0:
            return 0.0
        return self._pair_accepted.get(key, 0) / n

    def sync_bias_state(self, source_idx, target_indices=None):
        """
        Copy bias state from replica *source_idx* to target replicas.

        Useful for multi-walker MetaD where all replicas share a growing bias:
        after each exchange cycle, propagate the primary's MetaD grid to all
        secondaries.  Pass ``target_indices=None`` to broadcast to every other
        replica.
        """
        ctx_src, f_src = self._replicas[source_idx]
        blob = f_src.getBiasState(ctx_src)
        if target_indices is None:
            target_indices = [i for i in range(self._n) if i != source_idx]
        for i in target_indices:
            ctx_tgt, f_tgt = self._replicas[i]
            f_tgt.setBiasState(blob, ctx_tgt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_all(self, n_steps):
        for ctx, _ in self._replicas:
            ctx.getIntegrator().step(n_steps)

    def _attempt_swaps(self):
        if self._scheme == "neighbor":
            # Alternating even/odd pass (standard Sugita-Okamoto scheme).
            parity = self._rng.randint(0, 1)
            pairs = [(i, i + 1) for i in range(parity, self._n - 1, 2)]
        else:
            pairs = [(i, j) for i in range(self._n) for j in range(i + 1, self._n)]

        for i, j in pairs:
            self._attempt_swap(i, j)

    def _attempt_swap(self, i, j):
        ctx_i, _ = self._replicas[i]
        ctx_j, _ = self._replicas[j]

        # Snapshot current state (positions, velocities, energies).
        # Do NOT enforce periodic box: we want the actual (unwrapped) positions
        # so that setPositions restores them byte-for-byte.
        si = ctx_i.getState(getPositions=True, getVelocities=True, getEnergy=True)
        sj = ctx_j.getState(getPositions=True, getVelocities=True, getEnergy=True)

        x_i = si.getPositions(asNumpy=True)
        x_j = sj.getPositions(asNumpy=True)
        v_i = si.getVelocities(asNumpy=True)
        v_j = sj.getVelocities(asNumpy=True)
        # Snapshot periodic box vectors. Under NPT each replica has its own box
        # (the barostat changes it independently), so a configuration swap must
        # carry its box along — otherwise foreign-energy evals and the swapped
        # state use the wrong volume and corrupt the simulation.
        box_i = si.getPeriodicBoxVectors(asNumpy=True)
        box_j = sj.getPeriodicBoxVectors(asNumpy=True)

        beta_i, beta_j = self._betas[i], self._betas[j]

        delta = exchange_log_acceptance(ctx_i, ctx_j, si, sj, beta_i, beta_j)
        # Velocities are rescaled to the receiving replica's temperature
        # (no-op for H-REUS where all betas are equal).
        scale = math.sqrt(beta_j / beta_i)
        v_i_new, v_j_new = v_j * scale, v_i / scale

        key = (min(i, j), max(i, j))
        self._pair_attempts[key] = self._pair_attempts.get(key, 0) + 1
        self._n_attempts += 1

        accepted = delta >= 0.0 or self._rng.random() < math.exp(delta)
        if accepted:
            self._n_accepted += 1
            self._pair_accepted[key] = self._pair_accepted.get(key, 0) + 1
            # Swap full configurations, including periodic box vectors so the
            # exchanged volume travels with the positions (required for NPT).
            ctx_i.setPeriodicBoxVectors(*box_j)
            ctx_i.setPositions(x_j)
            ctx_i.setVelocities(v_i_new)
            ctx_j.setPeriodicBoxVectors(*box_i)
            ctx_j.setPositions(x_i)
            ctx_j.setVelocities(v_j_new)

        return accepted
