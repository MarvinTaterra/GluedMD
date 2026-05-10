"""OPESConvergenceReporter — auto-stop OPES simulations on convergence.

Supports multiple convergence criteria:

* ``'rct_relative'`` (default) — flatness of rct in dimensionless units. Robust
  across OPES variants. Converged when ``|Δrct| / max(|rct|, kT) < tol`` over
  ``min_consecutive_passes`` consecutive checks.
* ``'neff_rate'`` — stability of the effective-sample-size growth rate.
  Variant-agnostic; recommended for OPES_METAD_EXPLORE where the bias is
  intentionally non-stationary by design.
* ``'rct_absolute'`` — original behavior: ``|Δrct| < tol`` in kJ/mol.

Convergence is confirmed only after ``min_consecutive_passes`` consecutive
passing checks, and the test does not start until at least ``min_kernels``
kernels and ``min_steps`` steps have elapsed. After convergence is detected,
the simulation runs ``post_convergence_steps`` more steps before stopping.

Typical usage
-------------
::

    from OPESConvergenceReporter import OPESConvergenceReporter

    reporter = OPESConvergenceReporter(
        force, criterion='rct_relative', tol=0.01,
        check_interval=2000, post_convergence_steps=100_000,
        file='convergence.log')
    reporter.run(simulation, max_steps=10_000_000)
    print(f"Converged at step {reporter.converged_at_step}")

For OPES_METAD_EXPLORE (``mode='explore'`` on ``add_opes``) prefer
``criterion='neff_rate'`` — rct is non-stationary by design in EXPLORE::

    reporter = OPESConvergenceReporter(
        force, criterion='neff_rate', tol=0.02,
        check_interval=2000, post_convergence_steps=100_000)
"""

import sys

# Boltzmann constant in kJ/mol/K (matches GluedForce.kTFromTemperature).
_R_KJ = 8.314462618e-3


class OPESConvergenceReporter:
    """Auto-stop reporter for OPES simulations.

    Parameters
    ----------
    force : GluedForce
        The force containing the OPES bias.
    bias_idx : int
        Index of the OPES bias within the force (default 0).
    criterion : {'rct_relative', 'neff_rate', 'rct_absolute'}
        Convergence signal. See module docstring.
    tol : float
        Tolerance. Units depend on criterion:

          - ``'rct_relative'`` : dimensionless (~0.01)
          - ``'neff_rate'``    : dimensionless (~0.02)
          - ``'rct_absolute'`` : kJ/mol (~0.05)
    check_interval : int
        MD steps between successive checks.
    min_consecutive_passes : int
        Number of consecutive checks below tol before declaring convergence
        (default 3).
    min_kernels : int
        Don't start checking until at least this many kernels are deposited
        (default 50).
    min_steps : int
        Don't start checking until at least this many MD steps (default 0).
    post_convergence_steps : int
        MD steps to run after convergence is detected.
    file : str, file-like, or None
        Log destination (None = stdout).
    verbose : bool
        Print one line per check.
    """

    _VALID_CRITERIA = ('rct_relative', 'neff_rate', 'rct_absolute')

    def __init__(self, force, bias_idx=0, *,
                 criterion='rct_relative',
                 tol=0.01,
                 check_interval=1000,
                 min_consecutive_passes=3,
                 min_kernels=50,
                 min_steps=0,
                 post_convergence_steps=50_000,
                 file=None, verbose=True):
        # Initialize attributes that __del__ inspects BEFORE any code that
        # can raise. Otherwise a constructor failure (e.g. invalid criterion)
        # leaves the object partially-built and __del__ raises AttributeError,
        # which Python ignores but prints to stderr.
        self._out = None
        self._file = None

        if criterion not in self._VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {self._VALID_CRITERIA}, got {criterion!r}")

        self._force = force
        self._bias_idx = bias_idx
        self._criterion = criterion
        self._tol = tol
        self._interval = check_interval
        self._min_passes = max(1, int(min_consecutive_passes))
        self._min_kernels = max(0, int(min_kernels))
        self._min_steps = max(0, int(min_steps))
        self._post_steps = post_convergence_steps
        self._verbose = verbose
        self._file = file
        self._out = None

        # Cache kT for the relative-flatness denominator.
        try:
            T = force.getTemperature()
            self._kT = _R_KJ * T if T > 0 else _R_KJ * 300.0
        except Exception:
            self._kT = _R_KJ * 300.0   # safe fallback

        self._prev_signal = None
        self._prev_rct    = None     # tracked for rct_relative drift
        self._consecutive_passes = 0
        self._converged = False
        self._done = False
        self._converged_at_step = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def converged(self):
        """True once the chosen criterion has been satisfied N times in a row."""
        return self._converged

    @property
    def done(self):
        """True once post-convergence sampling is also complete."""
        return self._done

    @property
    def converged_at_step(self):
        """Step index at which convergence was first declared, or None."""
        return self._converged_at_step

    def force_converge(self, simulation):
        """Manually mark the run as converged.

        Useful for interactive sessions where the user has decided from
        external evidence (e.g. visual inspection of the FES) that the run
        is good enough. Triggers the post-convergence window normally.
        """
        if not self._converged:
            self._converged = True
            self._converged_at_step = simulation.currentStep
            self._open()
            self._log(
                f"[OPESConvergence] Manually marked converged at step "
                f"{simulation.currentStep}; running {self._post_steps:,} more steps.")

    def run(self, simulation, max_steps):
        """Attach, run in check_interval batches, then detach."""
        was_attached = self in simulation.reporters
        if not was_attached:
            simulation.reporters.append(self)
        steps_done = 0
        try:
            while not self._done and steps_done < max_steps:
                batch = min(self._interval, max_steps - steps_done)
                simulation.step(batch)
                steps_done += batch
        finally:
            if not was_attached:
                simulation.reporters.remove(self)

    # ------------------------------------------------------------------
    # OpenMM reporter protocol
    # ------------------------------------------------------------------

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        self._open()
        step = simulation.currentStep
        metrics = self._force.getOPESMetrics(simulation.context, self._bias_idx)
        zed, rct, nker, neff = metrics

        signal = self._compute_signal(step, rct, neff)

        # Don't evaluate convergence until warm-up gates pass.
        warming_up = (nker < self._min_kernels) or (step < self._min_steps)

        drift = None
        if not warming_up and self._prev_signal is not None:
            drift = self._compute_drift(signal, rct,
                                        self._prev_signal, self._prev_rct)
            if drift < self._tol:
                self._consecutive_passes += 1
                if (not self._converged
                        and self._consecutive_passes >= self._min_passes):
                    self._converged = True
                    self._converged_at_step = step
                    self._log(
                        f"[OPESConvergence] Converged at step {step} "
                        f"({self._criterion}): signal={signal:.4f}, "
                        f"|Δ|={drift:.5f} < tol={self._tol:.5f} "
                        f"for {self._consecutive_passes} consecutive checks. "
                        f"Running {self._post_steps:,} more steps.")
            else:
                self._consecutive_passes = 0

        if self._verbose:
            self._log_check(step, rct, nker, neff, signal, drift, warming_up)

        if self._converged and step >= self._converged_at_step + self._post_steps:
            self._done = True
            self._log(
                f"[OPESConvergence] Post-convergence run complete at step {step}. "
                f"Total post-convergence steps: "
                f"{step - self._converged_at_step:,}.")

        self._prev_signal = signal
        self._prev_rct    = rct

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_signal(self, step, rct, neff):
        """Return the dimensionless quantity to monitor (for display).

        For 'rct_relative' the signal saturates at ±1 once |rct| > kT —
        intuitive for humans reading the log but useless for diff-based
        convergence (|Δsignal| → 0 trivially). The actual drift used for
        the convergence check lives in `_compute_drift`.
        """
        if self._criterion == 'rct_relative':
            denom = max(abs(rct), self._kT)
            return rct / denom              # bounded in [-1, 1] roughly
        elif self._criterion == 'neff_rate':
            return neff / max(step, 1)      # asymptotically constant at convergence
        else:                               # 'rct_absolute'
            return rct

    def _compute_drift(self, signal, rct, prev_signal, prev_rct):
        """Criterion-aware drift used to evaluate convergence.

        For 'rct_relative' we cannot just diff the signal — once |rct|>kT
        the signal saturates at ±1 and |Δsignal| becomes identically 0,
        triggering false convergence even while rct is still drifting by
        ~kT per check. Use the unsaturated form
            |Δrct| / max(|rct|, |prev_rct|, kT)
        which gives a meaningful relative-change in both regimes:
            * |rct| < kT  ⇒ |Δrct|/kT (matches rct_absolute up to scale)
            * |rct| > kT  ⇒ |Δrct|/|rct| (true fractional drift)

        The other criteria don't saturate; signal-diff is fine.
        """
        if self._criterion == 'rct_relative':
            denom = max(abs(rct), abs(prev_rct), self._kT)
            return abs(rct - prev_rct) / denom
        return abs(signal - prev_signal)

    def _log_check(self, step, rct, nker, neff, signal, drift, warming_up):
        if self._converged:
            tag = "  [converged]"
        elif warming_up:
            tag = "  [warmup]"
        else:
            tag = ""
        drift_str = f"{drift:8.5f}" if drift is not None else "     ---"
        passes_str = (f"  passes={self._consecutive_passes}/{self._min_passes}"
                      if drift is not None and not warming_up else "")
        self._log(
            f"step {step:>10d}  rct={rct:9.4f}  nker={int(nker):5d}  "
            f"neff={neff:8.1f}  signal={signal:9.4f}  |Δ|={drift_str}"
            f"{passes_str}{tag}")

    def _open(self):
        if self._out is not None:
            return
        if self._file is None:
            self._out = sys.stdout
        elif isinstance(self._file, str):
            self._out = open(self._file, 'w')
        else:
            self._out = self._file

    def _log(self, msg):
        print(msg, file=self._out, flush=True)

    def __del__(self):
        if (self._out is not None
                and self._out is not sys.stdout
                and isinstance(self._file, str)):
            try:
                self._out.close()
            except Exception:
                pass
