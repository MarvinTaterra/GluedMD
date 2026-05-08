"""COLVAR reporter — writes GluedForce collective variables in COLVAR format.

Usage::

    from COLVARReporter import COLVARReporter
    reporter = COLVARReporter('colvar.dat', 100, force)
    simulation.reporters.append(reporter)

Output format::

    #! FIELDS time cv0 cv1 ...
     0.00000  1.23456  2.34567

Time is in picoseconds; CV values are in their native OpenMM units
(nm for distances, radians for angles/dihedrals, dimensionless for coordination, etc.).
"""

import openmm.unit as unit


class COLVARReporter:
    """Reporter that writes collective variable values to a COLVAR file.

    Parameters
    ----------
    file : str or file-like
        Output path or an open file object.  If a string, the file is created
        (or appended if *append* is True) on the first report call.
    reportInterval : int
        Number of steps between successive output lines.
    force : GluedForce
        The force whose CVs to log.  Must be added to the System before
        creating the Context.
    cvNames : list[str] or None
        Column names for the header line.  Length must equal the number of CV
        values returned by the force.  Defaults to ``cv0``, ``cv1``, ...
    append : bool
        If True and *file* is a string path, open in append mode and assume
        the header has already been written (skips writing it again).
    """

    def __init__(self, file, reportInterval, force, cvNames=None, append=False):
        self._interval = reportInterval
        self._force = force
        self._cv_names = cvNames
        self._append = append
        self._file = file
        self._out = None
        self._header_written = False

    # ------------------------------------------------------------------
    # OpenMM reporter protocol
    # ------------------------------------------------------------------

    def describeNextReport(self, simulation):
        """Return (steps_until_next, needs_pos, needs_vel, needs_forces, needs_energy)."""
        steps = self._interval - simulation.currentStep % self._interval
        # CV values come from GPU directly — no simulation state components needed.
        return (steps, False, False, False, False)

    def report(self, simulation, state):
        """Write one line of CV values for the current step."""
        self._open()
        values = self._force.getLastCVValues(simulation.context)
        if not self._header_written:
            self._write_header(len(values))
        time_ps = state.getTime().value_in_unit(unit.picoseconds)
        cols = [f'{time_ps:12.5f}'] + [f'{v:12.5f}' for v in values]
        print(' '.join(cols), file=self._out)
        self._out.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self):
        if self._out is not None:
            return
        if isinstance(self._file, str):
            self._out = open(self._file, 'a' if self._append else 'w')
            # In append mode, assume a header already exists if the file is non-empty.
            if self._append and self._out.tell() > 0:
                self._header_written = True
        else:
            self._out = self._file

    def _write_header(self, num_cvs):
        names = list(self._cv_names) if self._cv_names is not None \
                else [f'cv{i}' for i in range(num_cvs)]
        print('#! FIELDS time ' + ' '.join(names), file=self._out)
        self._header_written = True

    def __del__(self):
        if self._out is not None and isinstance(self._file, str):
            try:
                self._out.close()
            except Exception:
                pass
