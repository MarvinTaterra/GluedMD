"""OPESConvergenceReporter — auto-stop an OPES simulation when rct converges.

The convergence criterion is |Δrct| < tol (kJ/mol) over one check_interval.
After convergence is detected the simulation continues for post_convergence_steps
additional steps before stopping — providing extra sampling to confirm the result
and collect post-convergence statistics.

Metric background
-----------------
rct = kT·log(Z) is the free-energy normalization estimate returned by OPES.
It starts near zero and rises as new phase-space regions are explored.
When the simulation has converged, rct becomes roughly flat — |Δrct| per
check_interval falls below the thermal noise floor.

Typical usage
-------------
::

    from OPESConvergenceReporter import OPESConvergenceReporter

    reporter = OPESConvergenceReporter(force, tol=0.05, check_interval=2000,
                                       post_convergence_steps=100_000,
                                       file='convergence.log')
    reporter.run(simulation, max_steps=10_000_000)
    print(f"Converged at step {reporter.converged_at_step}")

Manual loop (if you need finer control)::

    simulation.reporters.append(reporter)
    while not reporter.done and simulation.currentStep < max_steps:
        simulation.step(check_interval)
"""

import sys


class OPESConvergenceReporter:
    """Reporter that stops an OPES simulation when rct has converged.

    Parameters
    ----------
    force : GluedForce
        The force containing the OPES bias.
    bias_idx : int
        Index of the OPES bias within the force (0 for the first/only bias).
    tol : float
        Convergence tolerance in kJ/mol.  The simulation is considered converged
        when |rct_now − rct_prev| < tol over one check_interval.
        Default 0.1 kJ/mol (~0.04 kBT at 300 K).
    check_interval : int
        Number of MD steps between successive rct checks.
    post_convergence_steps : int
        Additional MD steps to run after convergence is first detected.
    file : str, file-like, or None
        Destination for log output.  Defaults to stdout.
    verbose : bool
        If True, print rct at every check.  If False, only print convergence
        and completion events.
    """

    def __init__(self, force, bias_idx=0, tol=0.1, check_interval=1000,
                 post_convergence_steps=50_000, file=None, verbose=True):
        self._force = force
        self._bias_idx = bias_idx
        self._tol = tol
        self._interval = check_interval
        self._post_steps = post_convergence_steps
        self._verbose = verbose
        self._file = file
        self._out = None

        self._prev_rct = None
        self._converged = False
        self._done = False
        self._converged_at_step = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def converged(self):
        """True once the rct convergence criterion has been satisfied."""
        return self._converged

    @property
    def done(self):
        """True once post-convergence sampling is also complete."""
        return self._done

    @property
    def converged_at_step(self):
        """Step index at which convergence was first detected, or None."""
        return self._converged_at_step

    def run(self, simulation, max_steps):
        """Run until convergence + post-convergence steps, or max_steps.

        Attaches this reporter to *simulation*, runs in check_interval-sized
        batches, then detaches and returns.

        Parameters
        ----------
        simulation : openmm.app.Simulation
        max_steps : int
            Hard upper limit on total MD steps regardless of convergence.
        """
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
        """Return (steps_until_next, needs_pos, needs_vel, needs_forces, needs_energy)."""
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        """Check rct; update convergence state; write log line."""
        self._open()
        step = simulation.currentStep
        metrics = self._force.getOPESMetrics(simulation.context, self._bias_idx)
        # getOPESMetrics returns [zed, rct, nker, neff]
        rct  = metrics[1]
        nker = metrics[2]
        neff = metrics[3]

        if self._prev_rct is None:
            drct = None
            drct_str = "      ---"
        else:
            drct = abs(rct - self._prev_rct)
            drct_str = f"{drct:9.4f}"
            if not self._converged and drct < self._tol:
                self._converged = True
                self._converged_at_step = step
                self._log(
                    f"[OPESConvergence] Converged at step {step}: "
                    f"rct={rct:.4f} kJ/mol, |Δrct|={drct:.4f} < tol={self._tol:.4f}. "
                    f"Running {self._post_steps:,} more steps."
                )

        if self._verbose:
            tag = "  [converged]" if self._converged else ""
            self._log(
                f"step {step:>10d}  rct={rct:10.4f} kJ/mol"
                f"  |Δrct|={drct_str} kJ/mol"
                f"  nker={int(nker):6d}  neff={neff:8.1f}{tag}"
            )

        if self._converged and step >= self._converged_at_step + self._post_steps:
            self._done = True
            self._log(
                f"[OPESConvergence] Post-convergence run complete at step {step}. "
                f"Total post-convergence steps: {step - self._converged_at_step:,}."
            )

        self._prev_rct = rct

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        if self._out is not None \
                and self._out is not sys.stdout \
                and isinstance(self._file, str):
            try:
                self._out.close()
            except Exception:
                pass
