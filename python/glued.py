"""glued — Pythonic API for the GLUED enhanced-sampling plugin.

Wraps ``gluedplugin.GluedForce`` so that plain Python lists are accepted
everywhere and common CVs/biases have named, keyword-argument methods.

Minimal OPES example::

    import glued
    f = glued.Force(pbc=True, temperature=300.0)
    phi = f.add_dihedral([4, 6,  8, 14])
    psi = f.add_dihedral([6, 8, 14, 16])
    f.add_opes([phi, psi], sigma=0.05, gamma=10.0, pace=500)
    system.addForce(f)

The raw ``GluedForce`` API (``addCollectiveVariable``, ``addBias``, all enum
constants) remains fully accessible on ``Force`` instances.
"""

import gluedplugin as _gp
import openmm as _mm

_R_KJ = 8.314462618e-3   # kJ / (mol · K)

# Re-export the underlying class so callers can do ``isinstance(f, glued.GluedForce)``.
GluedForce = _gp.GluedForce


def _vi(seq):
    v = _mm.vectori()
    for x in seq:
        v.append(int(x))
    return v


def _vd(seq):
    v = _mm.vectord()
    for x in seq:
        v.append(float(x))
    return v


def _scalar_or_list(val, n):
    """Return a list of length n, broadcasting a scalar."""
    if hasattr(val, '__iter__') and not isinstance(val, str):
        out = list(val)
        if len(out) != n:
            raise ValueError(f"expected {n} values, got {len(out)}")
        return out
    return [val] * n


class Force(_gp.GluedForce):
    """``GluedForce`` with a Pythonic interface.

    Parameters
    ----------
    pbc : bool
        Enable periodic boundary conditions (default False).
    temperature : float or None
        System temperature in Kelvin.  Required for temperature-dependent biases
        (OPES, MetaD, EDS) unless passed directly to the bias method.
    group : int or None
        OpenMM force group (0–31).  Leave None to use the default group.
    """

    def __init__(self, *, pbc=False, temperature=None, group=None):
        super().__init__()
        self._temperature = temperature
        if pbc:
            self.setUsesPeriodicBoundaryConditions(True)
        if temperature is not None:
            self.setTemperature(float(temperature))
        if group is not None:
            self.setForceGroup(int(group))

    # ------------------------------------------------------------------
    # Override raw methods to accept plain Python lists
    # ------------------------------------------------------------------

    def addCollectiveVariable(self, cv_type, atoms, params=()):
        return super().addCollectiveVariable(
            int(cv_type),
            atoms if isinstance(atoms, _mm.vectori) else _vi(atoms),
            params if isinstance(params, _mm.vectord) else _vd(params),
        )

    def addBias(self, bias_type, cv_indices, params=(), int_params=()):
        return super().addBias(
            int(bias_type),
            cv_indices if isinstance(cv_indices, _mm.vectori) else _vi(cv_indices),
            params if isinstance(params, _mm.vectord) else _vd(params),
            int_params if isinstance(int_params, _mm.vectori) else _vi(int_params),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_kT(self, temperature):
        T = temperature if temperature is not None else self._temperature
        if T is None:
            raise ValueError(
                "temperature is required — pass temperature=... to Force() "
                "or to the bias method"
            )
        return _R_KJ * float(T)

    def _cv_list(self, cvs):
        return [cvs] if isinstance(cvs, int) else list(cvs)

    # ==================================================================
    # CV convenience methods
    # ==================================================================

    def add_distance(self, atoms):
        """Distance between two atoms (nm)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_DISTANCE, atoms)

    def add_angle(self, atoms):
        """Angle at the middle atom of a triplet (radians)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_ANGLE, atoms)

    def add_dihedral(self, atoms):
        """Torsion angle of four atoms (radians, −π … π)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_DIHEDRAL, atoms)

    def add_com_distance(self, group_a, group_b):
        """Distance between the centres of mass of two atom groups (nm)."""
        atoms = list(group_a) + list(group_b)
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_COM_DISTANCE, atoms, [len(group_a)]
        )

    def add_gyration(self, atoms):
        """Radius of gyration of an atom selection (nm)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_GYRATION, atoms)

    def add_coordination(self, group_a, group_b, r0, n=6, m=12, d0=0.0):
        """Coordination number using a rational switching function.

        Parameters
        ----------
        group_a, group_b : list[int]  Atom indices for the two groups.
        r0 : float  Switching distance (nm).
        n, m : int  Numerator/denominator exponents of the switch (default 6/12).
        d0 : float  Offset distance (nm, default 0).
        """
        atoms = list(group_a) + list(group_b)
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_COORDINATION, atoms, [len(group_a), r0, n, m, d0]
        )

    def add_rmsd(self, atoms, reference_positions):
        """Translation-fit RMSD to a reference structure (nm).

        Parameters
        ----------
        atoms : list[int]
        reference_positions : sequence of (x, y, z) in nm.
        """
        params = [v for pos in reference_positions for v in pos]
        return self.addCollectiveVariable(_gp.GluedForce.CV_RMSD, atoms, params)

    def add_drmsd(self, atom_pairs, ref_distances):
        """Distance-RMSD.

        Parameters
        ----------
        atom_pairs : flat list [a0, b0, a1, b1, …].
        ref_distances : reference pairwise distances (nm), same order as pairs.
        """
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_DRMSD, atom_pairs, ref_distances
        )

    def add_contact_map(self, pairs, *, r0, n=6, m=12, w=1.0, ref=0.0):
        """Contact map (sum of switching functions over atom pairs).

        Parameters
        ----------
        pairs : flat list [a0, b0, a1, b1, …].
        r0, n, m, w, ref : per-pair scalars or lists.
        """
        n_pairs = len(pairs) // 2
        r0_  = _scalar_or_list(r0,  n_pairs)
        n_   = _scalar_or_list(n,   n_pairs)
        m_   = _scalar_or_list(m,   n_pairs)
        w_   = _scalar_or_list(w,   n_pairs)
        ref_ = _scalar_or_list(ref, n_pairs)
        params = []
        for i in range(n_pairs):
            params.extend([r0_[i], n_[i], m_[i], w_[i], ref_[i]])
        return self.addCollectiveVariable(_gp.GluedForce.CV_CONTACTMAP, pairs, params)

    def add_path(self, atoms, frames, lambda_):
        """Path CV returning (s, z).

        The return value is the index of s; z is at index + 1.

        Parameters
        ----------
        atoms : list[int]  Atoms defining the path space.
        frames : list of flat position lists (nm) — each frame has 3*N values.
        lambda_ : float  Path metric scaling parameter.
        """
        params = [lambda_, len(frames)]
        for frame in frames:
            params.extend(frame)
        return self.addCollectiveVariable(_gp.GluedForce.CV_PATH, atoms, params)

    def add_position(self, atom, component):
        """Cartesian position of a single atom.

        component : 0=x, 1=y, 2=z (nm).
        """
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_POSITION, [atom], [component]
        )

    def add_plane_distance(self, plane_atoms, query_atom):
        """Signed distance of *query_atom* from the plane defined by three atoms."""
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PLANE, list(plane_atoms) + [query_atom]
        )

    def add_projection(self, atom_a, atom_b, axis_a, axis_b):
        """Projection of vector (a→b) onto unit axis (axis_a→axis_b) (nm)."""
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PROJECTION, [atom_a, atom_b, axis_a, axis_b]
        )

    def add_dipole(self, atoms, component=0):
        """Electric dipole moment.

        component : 0=magnitude, 1=x, 2=y, 3=z (e·nm).
        """
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_DIPOLE, atoms, [component]
        )

    def add_volume(self):
        """Simulation box volume (nm³)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_VOLUME, [], [])

    def add_cell(self, component):
        """Box parameter.  component: 0=a, 1=b, 2=c (nm), 3=α, 4=β, 5=γ (rad)."""
        return self.addCollectiveVariable(_gp.GluedForce.CV_CELL, [], [component])

    def add_puckering(self, ring_atoms, component):
        """Cremer–Pople ring-puckering amplitude/angle.

        ring_atoms : all atoms in the ring (3–8 atoms).
        component : 0 = total amplitude Q, 1..N−3 = angles.
        """
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PUCKERING, ring_atoms, [len(ring_atoms), component]
        )

    def add_secondary_structure(self, atoms, subtype, r0=0.08):
        """Secondary-structure content via translation-fit RMSD.

        atoms   : flat list of backbone atoms, 5 per residue in order
                  (N, CA, CB, C, O).  Alpha needs ≥ 6 residues (30 atoms);
                  beta needs multiples of 6 residues (30 atoms per window).
        subtype : 0=alpha-helix, 1=antiparallel-beta, 2=parallel-beta.
        r0      : switching-function cutoff in nm (default 0.08, same as PLUMED).
        """
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_SECONDARY_STRUCTURE, atoms, [subtype, r0]
        )

    def add_pca(self, atoms, mean_positions, eigenvector):
        """Projection of (positions − mean) onto a principal-component vector.

        atoms          : atom indices (length N).
        mean_positions : reference/mean structure, shape (N, 3) or flat length-3N,
                         in nm.
        eigenvector    : unit PC vector, same shape as mean_positions.

        Returns the CV index.  The CV value is in nm (same units as positions).
        """
        import math as _math
        flat_mean = [c for xyz in mean_positions for c in xyz] \
            if hasattr(mean_positions[0], '__iter__') else list(mean_positions)
        flat_ev   = [c for xyz in eigenvector   for c in xyz] \
            if hasattr(eigenvector[0],   '__iter__') else list(eigenvector)
        if len(flat_mean) != 3 * len(atoms) or len(flat_ev) != 3 * len(atoms):
            raise ValueError("mean_positions and eigenvector must each have 3*len(atoms) values")
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PCA, atoms, flat_mean + flat_ev
        )

    def add_ermsd(self, atoms_per_residue, reference_positions, cutoff=2.4):
        """Bottaro eRMSD for RNA structural similarity.

        atoms_per_residue  : list of [P, C4', N1/N9] atom index triplets,
                             one per residue.  E.g. [[0,1,2], [3,4,5], ...].
        reference_positions: atom positions for the reference structure,
                             indexable by the same atom indices, in nm.
                             Only the atoms in atoms_per_residue are used.
        cutoff             : eRMSD cutoff radius in nm (default 2.4, Bottaro 2014).

        Returns the CV index.  Pre-computes the 4D Bottaro G-vectors from the
        reference positions; the kernel uses these at every step.
        """
        import math as _math

        N = len(atoms_per_residue)
        flat_atoms = [a for triplet in atoms_per_residue for a in triplet]

        # Bottaro (2014) form factors and parameters
        ff0, ff1, ff2 = 2.0, 2.0, 1.0 / 0.3
        gamma = _math.pi / cutoff
        maxdist = cutoff / ff0

        def _frame(p0, p1, p2):
            cx = (p0[0]+p1[0]+p2[0]) / 3.0
            cy = (p0[1]+p1[1]+p2[1]) / 3.0
            cz = (p0[2]+p1[2]+p2[2]) / 3.0
            ax, ay, az = p0[0]-cx, p0[1]-cy, p0[2]-cz
            la = _math.sqrt(ax*ax + ay*ay + az*az) or 1e-8
            e1 = [ax/la, ay/la, az/la]
            bx, by, bz = p1[0]-cx, p1[1]-cy, p1[2]-cz
            dx = ay*bz - az*by; dy = az*bx - ax*bz; dz = ax*by - ay*bx
            ld = _math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-8
            e3 = [dx/ld, dy/ld, dz/ld]
            e2 = [e3[1]*e1[2]-e3[2]*e1[1],
                  e3[2]*e1[0]-e3[0]*e1[2],
                  e3[0]*e1[1]-e3[1]*e1[0]]
            return [cx, cy, cz], e1, e2, e3

        def _gvec(ci, e1i, e2i, e3i, cj):
            drx, dry, drz = cj[0]-ci[0], cj[1]-ci[1], cj[2]-ci[2]
            dist = _math.sqrt(drx*drx + dry*dry + drz*drz)
            if dist >= maxdist:
                return [0.0, 0.0, 0.0, 0.0]
            rt0 = (drx*e1i[0]+dry*e1i[1]+drz*e1i[2]) * ff0
            rt1 = (drx*e2i[0]+dry*e2i[1]+drz*e2i[2]) * ff1
            rt2 = (drx*e3i[0]+dry*e3i[1]+drz*e3i[2]) * ff2
            rtn = _math.sqrt(rt0*rt0 + rt1*rt1 + rt2*rt2)
            if rtn <= 1e-8:
                return [rt0, rt1, rt2, 2.0/gamma]
            if rtn >= cutoff:
                return [0.0, 0.0, 0.0, 0.0]
            sc = _math.sin(gamma*rtn) / (rtn*gamma)
            co = _math.cos(gamma*rtn)
            return [sc*rt0, sc*rt1, sc*rt2, (1.0+co)/gamma]

        def _pos(idx):
            p = reference_positions[idx]
            return list(p) if hasattr(p, '__iter__') else [p[0], p[1], p[2]]

        frames = [_frame(_pos(t[0]), _pos(t[1]), _pos(t[2])) for t in atoms_per_residue]
        gvecs = []
        for i in range(N):
            ci, e1i, e2i, e3i = frames[i]
            for j in range(N):
                if j == i:
                    continue
                gvecs.extend(_gvec(ci, e1i, e2i, e3i, frames[j][0]))

        params = [float(N), float(cutoff)] + gvecs
        return self.addCollectiveVariable(_gp.GluedForce.CV_ERMSD, flat_atoms, params)

    def add_expression(self, expression, input_cvs):
        """Algebraic expression CV (Lepton parser).

        expression : string using cv0, cv1, … for *input_cvs*.
        input_cvs : list of CV indices to bind to cv0, cv1, …
        """
        return self.addExpressionCV(expression, _vi(input_cvs))

    # ==================================================================
    # Bias convenience methods
    # ==================================================================

    def add_harmonic(self, cvs, kappa, at, *, periodic=False):
        """Harmonic restraint: V = ½ κ (s − s₀)².

        Parameters
        ----------
        cvs : int or list[int]  CV index/indices.
        kappa : float or list  Force constant(s) (kJ/mol/unit²).
        at : float or list  Equilibrium value(s).
        periodic : bool  Wrap difference to (−π, π] (for dihedrals).
        """
        cvs   = self._cv_list(cvs)
        kappa = _scalar_or_list(kappa, len(cvs))
        at    = _scalar_or_list(at,    len(cvs))
        params = []
        for k, s in zip(kappa, at):
            params.extend([k, s])
        return self.addBias(
            _gp.GluedForce.BIAS_HARMONIC, cvs, params, [1 if periodic else 0]
        )

    def add_upper_wall(self, cv, at, kappa, n=2, eps=1.0):
        """One-sided upper wall: V = κ·((s − at)/eps)ⁿ when s > at."""
        return self.addBias(
            _gp.GluedForce.BIAS_UPPER_WALL, [cv], [at, kappa, eps, n]
        )

    def add_lower_wall(self, cv, at, kappa, n=2, eps=1.0):
        """One-sided lower wall: V = κ·((at − s)/eps)ⁿ when s < at."""
        return self.addBias(
            _gp.GluedForce.BIAS_LOWER_WALL, [cv], [at, kappa, eps, n]
        )

    def add_linear(self, cvs, k):
        """Linear coupling: V = −k·s.

        Parameters
        ----------
        cvs : int or list[int]
        k : float or list  Coupling constant(s) (kJ/mol/unit).
        """
        cvs = self._cv_list(cvs)
        k   = _scalar_or_list(k, len(cvs))
        return self.addBias(_gp.GluedForce.BIAS_LINEAR, cvs, k)

    def add_abmd(self, cvs, kappa, to):
        """Ratchet-and-pawl ABMD restraint (Marchi & Ballone 1999).

        Applies a harmonic restoring force whenever a CV moves away from
        its closest approach to *to*.

        Parameters
        ----------
        cvs : int or list[int]
        kappa : float or list  Spring constant(s) (kJ/mol/nm² or per-unit²).
        to : float or list  Target value(s).
        """
        cvs   = self._cv_list(cvs)
        kappa = _scalar_or_list(kappa, len(cvs))
        to    = _scalar_or_list(to,    len(cvs))
        params = []
        for k, t in zip(kappa, to):
            params.extend([k, t])
        return self.addBias(_gp.GluedForce.BIAS_ABMD, cvs, params)

    def add_opes(self, cvs, sigma, *, gamma=10.0, pace=500, temperature=None,
                 sigma_min=None, max_kernels=100000, mode='metad'):
        """OPES bias.

        Parameters
        ----------
        cvs : int or list[int]  CV index/indices.
        sigma : float or list  Initial kernel bandwidth per CV (same units as CV).
        gamma : float  Biasfactor γ > 1 (default 10).
        pace : int  Deposition stride in steps (default 500).
        temperature : float or None  Override Force-level temperature (K).
        sigma_min : float or None  Minimum sigma (default 5 % of sigma).
        max_kernels : int  Kernel table capacity (default 100 000).
        mode : {'metad', 'explore', 'fixed_uniform'}, default 'metad'
            * ``'metad'`` — standard OPES_METAD: well-tempered target,
              adaptive Silverman σ. Bias prefactor (γ−1)/γ.
              Reference: Invernizzi & Parrinello, JPCL 11:2731 (2020).
            * ``'explore'`` — OPES_METAD_EXPLORE: well-tempered sampled
              distribution, plain (unweighted) KDE on biased samples.
              Bias prefactor (γ−1). Use when CVs may be degenerate or the
              barrier height is unknown. The user-supplied σ is internally
              broadened by √γ; pass the same σ you would use with METAD.
              Reference: Invernizzi, Piaggi & Parrinello, JCTC 18:3988 (2022).
            * ``'fixed_uniform'`` — uniform target (γ→∞ equivalent) with
              fixed σ. Specialized; rarely needed.
        """
        variant_map = {'metad': 0, 'fixed_uniform': 1, 'explore': 2}
        if mode not in variant_map:
            raise ValueError(
                f"mode must be one of {list(variant_map)}, got {mode!r}")
        if mode == 'explore':
            import math as _math
            if not (gamma > 1.0 and _math.isfinite(gamma)):
                raise ValueError(
                    "OPES EXPLORE requires finite gamma > 1 (no BIASFACTOR=inf)")
        variant = variant_map[mode]

        cvs   = self._cv_list(cvs)
        kT    = self._resolve_kT(temperature)
        sigma = _scalar_or_list(sigma, len(cvs))
        s_min = sigma_min if sigma_min is not None else min(sigma) * 0.05
        params     = [kT, float(gamma)] + sigma + [float(s_min)]
        int_params = [variant, int(pace), int(max_kernels)]
        return self.addBias(_gp.GluedForce.BIAS_OPES, cvs, params, int_params)

    def add_metad(self, cvs, sigma, height, pace, *, grid_min, grid_max,
                  bins=100, periodic=False, gamma=None, temperature=None):
        """Well-tempered metadynamics on a fixed grid.

        Parameters
        ----------
        cvs : int or list[int]
        sigma : float or list  Hill width per CV.
        height : float  Initial Gaussian height (kJ/mol).
        pace : int  Hill deposition stride (steps).
        grid_min, grid_max : float or list  Grid boundaries per CV.
        bins : int or list  Number of grid bins per CV (default 100).
        periodic : bool or list  Whether each CV is periodic (default False).
        gamma : float or None  Biasfactor; None → non-tempered (classic MetaD).
        temperature : float or None  Override Force-level temperature (K).
        """
        cvs   = self._cv_list(cvs)
        D     = len(cvs)
        kT    = self._resolve_kT(temperature)
        gamma = float(gamma) if gamma is not None else 0.0
        sigma    = _scalar_or_list(sigma,    D)
        g_min    = _scalar_or_list(grid_min, D)
        g_max    = _scalar_or_list(grid_max, D)
        bins_    = _scalar_or_list(bins,     D)
        periodic_ = _scalar_or_list(periodic, D)
        params     = [float(height)] + sigma + [gamma, kT] + g_min + g_max
        int_params = [int(pace)] + [int(b) for b in bins_] + \
                     [1 if p else 0 for p in periodic_]
        return self.addBias(_gp.GluedForce.BIAS_METAD, cvs, params, int_params)

    def add_moving_restraint(self, cvs, schedule):
        """Time-varying harmonic restraint.

        Parameters
        ----------
        cvs : int or list[int]
        schedule : list of (step, kappa, at) tuples.
            kappa and at may be scalars (same for all CVs) or lists.
            Values are linearly interpolated between schedule points.
        """
        cvs = self._cv_list(cvs)
        D   = len(cvs)
        params = []
        for entry in schedule:
            step, kappa, at = entry
            kappa = _scalar_or_list(kappa, D)
            at    = _scalar_or_list(at,    D)
            params.append(float(step))
            for k, s in zip(kappa, at):
                params.extend([k, s])
        return self.addBias(
            _gp.GluedForce.BIAS_MOVING_RESTRAINT, cvs, params, [len(schedule)]
        )

    def add_eds(self, cvs, target, max_range=None, *, temperature=None):
        """EDS adaptive linear restraint (White & Voth, JCTC 2014).

        Parameters
        ----------
        cvs : int or list[int]
        target : float or list  Target mean value(s) for each CV.
        max_range : float or list or None  Max coupling range (default 25 kT).
        temperature : float or None  Override Force-level temperature.
        """
        cvs    = self._cv_list(cvs)
        D      = len(cvs)
        kT     = self._resolve_kT(temperature)
        target = _scalar_or_list(target, D)
        if max_range is None:
            max_range = [25.0 * kT] * D
        else:
            max_range = _scalar_or_list(max_range, D)
        params = []
        for t, r in zip(target, max_range):
            params.extend([float(t), float(r)])
        params.append(kT)
        return self.addBias(_gp.GluedForce.BIAS_EDS, cvs, params)
