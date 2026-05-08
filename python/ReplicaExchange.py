"""
ReplicaExchange — GLUED replica exchange driver.

Supports two modes:

  H-REUS  (Hamiltonian Replica Exchange Umbrella Sampling)
      All replicas run at the *same* temperature but with different bias
      parameters (e.g. different harmonic restraint centres).  Metropolis
      criterion uses total potential energy; the identical MM contributions
      cancel, leaving only the bias-energy difference.

  T-REMD  (Temperature Replica Exchange MD)
      All replicas share the same Hamiltonian / bias but run at different
      temperatures.  Criterion uses MM-only energy (total minus bias).
      The user must place GluedForce in a non-default force group so the
      bias energy can be isolated:
          f.setForceGroup(1)        # before context creation
          re = ReplicaExchange(..., mode="T-REMD",
                                   temperatures=[300, 320, 340, 360],
                                   bias_force_group=1)

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
from openmm import unit


def _total_energy_kJ(ctx):
    """Return total potential energy in kJ/mol."""
    s = ctx.getState(getEnergy=True)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _group_energy_kJ(ctx, groups_mask):
    """Return potential energy restricted to the given force-group mask, kJ/mol."""
    s = ctx.getState(getEnergy=True, groups=groups_mask)
    return s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _mm_energy_kJ(ctx, bias_group):
    """MM-only energy: total minus bias group, kJ/mol."""
    total = _total_energy_kJ(ctx)
    bias  = _group_energy_kJ(ctx, 1 << bias_group)
    return total - bias


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
        Force group index of GluedForce.  Required for T-REMD when a bias
        is present so the MM energy can be isolated.  Set None if no bias (all
        energy is MM energy).
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
            if kT is None:
                raise ValueError("H-REUS requires kT")
            self._betas = [1.0 / kT] * self._n

        elif self._mode == "T-REMD":
            if temperatures is None or len(temperatures) != self._n:
                raise ValueError("T-REMD requires temperatures list, one per replica")
            self._temperatures = list(temperatures)
            self._betas = [1.0 / (self._GAS_CONSTANT_kJmol * T)
                           for T in temperatures]
            # Verify integrator temperatures match
            for i, (ctx, _) in enumerate(self._replicas):
                integ = ctx.getIntegrator()
                if hasattr(integ, "getTemperature"):
                    T_integ = integ.getTemperature().value_in_unit(unit.kelvin)
                    if abs(T_integ - temperatures[i]) > 1.0:
                        import warnings
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
        _, f_src = self._replicas[source_idx]
        blob = f_src.getBiasState()
        if target_indices is None:
            target_indices = [i for i in range(self._n) if i != source_idx]
        for i in target_indices:
            _, f_tgt = self._replicas[i]
            f_tgt.setBiasState(blob)

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

        beta_i, beta_j = self._betas[i], self._betas[j]

        if self._mode == "H-REUS":
            # Total energy: MM terms cancel in criterion (same T).
            E_ii = si.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            E_jj = sj.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            ctx_i.setPositions(x_j)
            E_ij = _total_energy_kJ(ctx_i)
            ctx_i.setPositions(x_i)  # restore

            ctx_j.setPositions(x_i)
            E_ji = _total_energy_kJ(ctx_j)
            ctx_j.setPositions(x_j)  # restore

            delta = -beta_i * (E_ij + E_ji - E_ii - E_jj)
            v_i_new, v_j_new = v_j, v_i   # no scaling (same T)

        else:  # T-REMD
            # Only U(x_i) and U(x_j) are needed — no foreign-position evaluations.
            if self._bias_group is not None:
                U_ii = _mm_energy_kJ(ctx_i, self._bias_group)
                U_jj = _mm_energy_kJ(ctx_j, self._bias_group)
            else:
                # No bias: total energy equals MM energy.
                U_ii = si.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                U_jj = sj.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            # Sugita-Okamoto criterion: Δ = (β_i − β_j)(U(x_i) − U(x_j))
            # β_i > β_j when T_i < T_j; U(x_i) < U(x_j) typically →  Δ < 0 → P_acc < 1
            delta = (beta_i - beta_j) * (U_ii - U_jj)

            # Velocity rescaling for T-REMD.
            Ti, Tj = self._temperatures[i], self._temperatures[j]
            scale_i = math.sqrt(Ti / Tj)   # velocities going from j→i context
            scale_j = math.sqrt(Tj / Ti)   # velocities going from i→j context
            v_i_new = v_j * scale_i
            v_j_new = v_i * scale_j

        key = (min(i, j), max(i, j))
        self._pair_attempts[key] = self._pair_attempts.get(key, 0) + 1
        self._n_attempts += 1

        accepted = delta >= 0.0 or self._rng.random() < math.exp(delta)
        if accepted:
            self._n_accepted += 1
            self._pair_accepted[key] = self._pair_accepted.get(key, 0) + 1
            ctx_i.setPositions(x_j)
            ctx_i.setVelocities(v_i_new)
            ctx_j.setPositions(x_i)
            ctx_j.setVelocities(v_j_new)

        return accepted
