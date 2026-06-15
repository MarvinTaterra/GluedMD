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

Minimal OPES multithermal example (run cold, sample hot, reweight back)::

    import glued
    f = glued.Force(temperature=300.0)          # simulation temperature = temp0
    energy_cv, _ = f.add_multithermal(300.0, 600.0, n_temps=16, pace=500)
    system.addForce(f)                          # integrator must run at 300 K
    # … run dynamics, collecting per-frame U and V …
    U, V = [], []
    for _ in range(n_frames):
        integrator.step(stride)
        u, v = f.multithermal_uv(context)
        U.append(u); V.append(v)
    mean_at_500, ess = glued.reweight_to_temperature(U, V, 300.0, 500.0, observable)
"""

# Import openmm BEFORE gluedplugin: gluedplugin reuses openmm's SWIG std::vector
# typemaps (vectori/vectord) via %import, and that cross-module type table is only
# registered if openmm is loaded first. Importing gluedplugin first leaves the
# wrapper's vector arguments unrecognised (TypeError on every addCollectiveVariable).
import openmm as _mm
import gluedplugin as _gp

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

    def add_coordination(self, group_a, group_b, r0, n=6, m=12):
        """Coordination number using a rational switching function
        f(r) = (1 − (r/r0)ⁿ) / (1 − (r/r0)ᵐ).

        Parameters
        ----------
        group_a, group_b : list[int]  Atom indices for the two groups.
        r0 : float  Switching distance (nm).
        n, m : int  Numerator/denominator exponents of the switch (default 6/12).

        Notes
        -----
        The kernel contract is ``atoms = [len(group_a), *group_a, *group_b]`` and
        ``params = [r0, n, m]`` (the group-A size is the first atom slot, not a param).
        PLUMED's ``D_0`` offset is not implemented in the kernel and is therefore not
        exposed here.
        """
        atoms = [len(group_a)] + list(group_a) + list(group_b)
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_COORDINATION, atoms, [r0, n, m]
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

    def add_contact_map(self, pairs, *, r0, n=6, m=12, w=1.0):
        """Contact map (weighted sum of rational switching functions over atom pairs).

        Parameters
        ----------
        pairs : flat list [a0, b0, a1, b1, …].
        r0, n, m, w : per-pair scalars or lists.

        Notes
        -----
        Kernel contract is 4 params per pair ``[r0, n, m, w]``. (PLUMED's per-pair
        reference offset is not implemented in the kernel and is not exposed.)
        """
        n_pairs = len(pairs) // 2
        r0_  = _scalar_or_list(r0,  n_pairs)
        n_   = _scalar_or_list(n,   n_pairs)
        m_   = _scalar_or_list(m,   n_pairs)
        w_   = _scalar_or_list(w,   n_pairs)
        params = []
        for i in range(n_pairs):
            params.extend([r0_[i], n_[i], m_[i], w_[i]])
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

    def add_plane(self, plane_atoms, component=2):
        """A Cartesian component of the unit normal to the plane through three atoms.

        Parameters
        ----------
        plane_atoms : sequence of 3 atom indices defining the plane.
        component : int  ``0=nx, 1=ny, 2=nz`` of the unit normal n̂ = (B−A)×(C−A)/|…|.

        Notes
        -----
        This is what the CV_PLANE kernel actually computes. A true signed
        point-to-plane distance is not implemented (see :meth:`add_plane_distance`).
        """
        plane_atoms = list(plane_atoms)
        if len(plane_atoms) != 3:
            raise ValueError("plane_atoms must be exactly 3 atom indices")
        if int(component) not in (0, 1, 2):
            raise ValueError("component must be 0=nx, 1=ny, or 2=nz")
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PLANE, plane_atoms, [int(component)]
        )

    def add_plane_distance(self, plane_atoms, query_atom):
        """Not implemented — the CV_PLANE kernel returns a unit-normal component, not a
        point-to-plane distance. Use :meth:`add_plane` for the normal component."""
        raise NotImplementedError(
            "signed point-to-plane distance is not implemented; the CV_PLANE kernel "
            "returns a component of the plane's unit normal — use add_plane(plane_atoms, "
            "component) instead."
        )

    def add_projection(self, atom_a, atom_b, direction):
        """Projection of the vector (a→b) onto a fixed direction (nm).

        Parameters
        ----------
        atom_a, atom_b : int  The vector is r_b − r_a.
        direction : sequence of 3 floats  Projection axis (normalized internally).

        Notes
        -----
        The kernel uses a *fixed* direction (``params=[nx,ny,nz]``), not a dynamic
        two-atom axis.
        """
        direction = list(direction)
        if len(direction) != 3:
            raise ValueError("direction must be a 3-vector (nx, ny, nz)")
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_PROJECTION, [atom_a, atom_b], direction
        )

    def add_dipole(self, atoms, charges, component=3):
        """Electric dipole moment of an atom group (e·nm).

        Parameters
        ----------
        atoms : list[int]  Atom indices in the group.
        charges : sequence of float  Per-atom charges (e), one per atom — the kernel
            needs these explicitly (they are not read from the System's NonbondedForce).
        component : int  ``0=x, 1=y, 2=z, 3=|μ|`` (magnitude). Default 3.

        Notes
        -----
        Kernel contract: ``params = [q0, …, qN-1, component]``. For a non-neutral group
        the dipole is taken about the group's mean charge (PLUMED's Q/N neutralization).
        """
        atoms = list(atoms)
        charges = _scalar_or_list(charges, len(atoms))
        if int(component) not in (0, 1, 2, 3):
            raise ValueError("component must be 0=x, 1=y, 2=z, or 3=|mu|")
        return self.addCollectiveVariable(
            _gp.GluedForce.CV_DIPOLE, atoms, list(charges) + [int(component)]
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
        """Linear coupling: V = +k·s (force −k on the CV).

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

    def add_opes(self, cvs, sigma=None, *, gamma=10.0, pace=500,
                 temperature=None, sigma_min=None, max_kernels=100000,
                 mode='metad', adaptive_sigma_stride=None):
        """OPES bias.

        Parameters
        ----------
        cvs : int or list[int]
            CV index/indices.
        sigma : float, list, ``None``, or ``'adaptive'``
            Initial kernel bandwidth per CV (same units as the CV). Set to
            ``None`` (default) or ``'adaptive'`` for **fully-adaptive σ**
            — a per-step Welford running variance is used, so kernels are
            broadened/narrowed automatically as the run explores new
            regions (PLUMED's ``SIGMA=ADAPTIVE``). Passing an explicit
            value gives the standard mixed-adaptive behaviour: σ is fixed
            for the deposit centre but Silverman's rule still rescales it
            on every deposit (PLUMED's default).
        gamma : float
            Biasfactor γ > 1 (default 10).
        pace : int
            Deposition stride in steps (default 500).
        temperature : float or None
            Override Force-level temperature (K).
        sigma_min : float or None
            Minimum sigma. With explicit ``sigma`` defaults to 5 % of σ;
            with adaptive σ defaults to 1e-3 (CV units), which is the same
            order PLUMED uses.
        max_kernels : int
            Kernel table capacity (default 100 000).
        mode : {'metad', 'explore', 'fixed_uniform'}, default 'metad'
            * ``'metad'`` — standard OPES_METAD: well-tempered target,
              Silverman σ-rescaling. Bias prefactor (γ−1)/γ.
              Reference: Invernizzi & Parrinello, JPCL 11:2731 (2020).
            * ``'explore'`` — OPES_METAD_EXPLORE: well-tempered sampled
              distribution, plain (unweighted) KDE on biased samples.
              Bias prefactor (γ−1). Use when CVs may be degenerate or the
              barrier height is unknown. The user-supplied σ is internally
              broadened by √γ; pass the same σ you would use with METAD.
              Reference: Invernizzi, Piaggi & Parrinello, JCTC 18:3988 (2022).
            * ``'fixed_uniform'`` — uniform target (γ→∞ equivalent) with
              fixed σ. Specialized; rarely needed. Not compatible with
              adaptive σ.
        adaptive_sigma_stride : int or None
            Only used in fully-adaptive mode. Number of MD steps before
            kernel deposition is allowed (warm-up window over which the
            running variance stabilises). Defaults to ``10 * pace`` —
            matches PLUMED's behaviour.
        """
        variant_map = {'metad': 0, 'fixed_uniform': 1, 'explore': 2}
        if mode not in variant_map:
            raise ValueError(
                f"mode must be one of {list(variant_map)}, got {mode!r}")
        import math as _math
        if mode == 'explore':
            if not (gamma > 1.0 and _math.isfinite(gamma)):
                raise ValueError(
                    "OPES EXPLORE requires finite gamma > 1 (no BIASFACTOR=inf)")
        variant = variant_map[mode]

        # Detect fully-adaptive σ.  Sentinel passed to the kernel is sigma=0.0.
        adaptive = (sigma is None
                    or (isinstance(sigma, str) and sigma.lower() == 'adaptive'))
        if adaptive and mode == 'fixed_uniform':
            raise ValueError(
                "mode='fixed_uniform' is incompatible with adaptive σ; "
                "pass an explicit sigma value.")

        cvs = self._cv_list(cvs)
        kT  = self._resolve_kT(temperature)

        if adaptive:
            sigma_list = [0.0] * len(cvs)            # sentinel
            s_min      = (sigma_min if sigma_min is not None else 1e-3)
        else:
            sigma_list = _scalar_or_list(sigma, len(cvs))
            s_min      = (sigma_min if sigma_min is not None
                          else min(sigma_list) * 0.05)

        params     = [kT, float(gamma)] + sigma_list + [float(s_min)]
        int_params = [variant, int(pace), int(max_kernels)]
        if adaptive:
            int_params.append(int(adaptive_sigma_stride
                                  if adaptive_sigma_stride is not None
                                  else 10 * pace))
        elif adaptive_sigma_stride is not None:
            raise ValueError(
                "adaptive_sigma_stride is only used when sigma is None or "
                "'adaptive'; remove it or set sigma=None.")
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

    # ==================================================================
    # OPES multithermal (energy as CV) — Stage 3/4
    # ==================================================================

    def add_energy_cv(self):
        """Add a total-potential-energy collective variable.

        The CV value is the system's **unbiased** potential energy U, evaluated
        fully GPU-resident in a linked inner context that holds every System force
        *except* this GluedForce (so the bias never enters U). Its per-atom Jacobian
        is dU/dx = −F. This is the CV that OPES multithermal expands.

        Returns
        -------
        int  The CV index.

        Notes
        -----
        * Requires the **CUDA or OpenCL** platform (the Reference platform has no
          device-resident inner context).
        * The inner context omits constraints and virtual sites — both are handled
          correctly without copying them. Constraints contribute no potential energy or
          force, so constrained systems (H-bond constraints, rigid water) are fine.
          **Virtual-site water/force fields are fully supported** (OPC, TIP4P-Ew,
          TIP4P-D, TIP5P, a99SB-disp): the bias force on a virtual site is redistributed
          to its parent atoms by the outer context's normal virtual-site machinery, after
          the chain-rule scatter — validated in ``tests/test_cv_energy_vsite.py``. (This
          assumes a standard integrator that recomputes virtual-site positions each step.)
        """
        return self.addCollectiveVariable(_gp.GluedForce.CV_ENERGY, [], [])

    def add_multithermal(self, temp_min, temp_max, n_temps=16, *,
                         temp0=None, temps=None, pace=500, spacing='geometric'):
        """OPES multithermal: sample a temperature range from a single replica.

        Adds a total-energy CV (:meth:`add_energy_cv`) and expands it over an
        inverse-temperature ladder spanning ``[temp_min, temp_max]``, applying a bias
        that flattens the marginal distribution of U over that range. A single
        trajectory run at ``temp0`` then visits configurations characteristic of the
        whole range; reweight back to any target temperature in the range afterwards
        with :func:`reweight_to_temperature`.

        Reference: Invernizzi, Piaggi & Parrinello, *Unified approach to enhanced
        sampling*, PRX 10, 041034 (2020).

        Parameters
        ----------
        temp_min, temp_max : float
            Lowest / highest temperature (K) of the expanded ensemble.
        n_temps : int
            Number of temperatures in the ladder (default 16). A few tens are plenty;
            neighbouring canonical distributions only need to overlap.
        temp0 : float or None
            Reference (simulation) temperature in K. Defaults to the Force-level
            ``temperature``. **Your integrator must run at this temperature.** With
            flat ΔF the bias force is zero when the ladder reduces to {temp0}.
        temps : list[float] or None
            Explicit temperature list (K); overrides ``temp_min/temp_max/n_temps``.
        pace : int
            ΔF-update stride in steps (default 500).
        spacing : {'geometric', 'uniform'}
            Ladder spacing (default 'geometric' — near-constant neighbour overlap).

        Returns
        -------
        (energy_cv, bias_index) : tuple[int, int]
        """
        if temps is not None:
            ladder = [float(t) for t in temps]
            if not ladder:
                raise ValueError("temps must be a non-empty list of temperatures")
        else:
            ladder = multithermal_temperature_ladder(
                temp_min, temp_max, int(n_temps), spacing=spacing)
        T0 = temp0 if temp0 is not None else self._temperature
        if T0 is None:
            raise ValueError(
                "temp0 (reference/simulation temperature) is required — pass "
                "temp0=... or construct Force(temperature=...)")
        kT0   = _R_KJ * float(T0)
        betas = [1.0 / (_R_KJ * float(t)) for t in ladder]
        energy_cv = self.add_energy_cv()
        params    = [kT0] + betas            # [kT0, β_0, …, β_{N-1}]
        bias = self.addBias(_gp.GluedForce.BIAS_OPES_MULTITHERMAL,
                            [energy_cv], params, [int(pace)])
        # Stash the configuration so reweighting can be done without re-deriving β0.
        self._multithermal = {
            'energy_cv': energy_cv, 'temp0': float(T0),
            'temps': ladder, 'bias': bias,
        }
        return energy_cv, bias

    def multithermal_uv(self, context, energy_cv=None):
        """Return ``(U, V)`` for the current frame.

        U is the total potential energy CV value (kJ/mol) and V is the total applied
        bias energy (kJ/mol). Collect these every frame during a multithermal run and
        pass the arrays to :func:`reweight_to_temperature`. Call after a
        ``context.getState(getEnergy=True)`` (or any force evaluation) so the cached
        values are current.

        ``energy_cv`` defaults to the CV index returned by :meth:`add_multithermal`.
        Assumes the multithermal bias is the only bias on this Force (so V is the
        multithermal bias); if you add other biases, track V yourself.
        """
        if energy_cv is None:
            if not hasattr(self, '_multithermal'):
                raise ValueError("no multithermal bias on this Force; pass energy_cv=")
            energy_cv = self._multithermal['energy_cv']
        U = list(self.getLastCVValues(context))[energy_cv]
        V = self.getLastBias(context)
        return U, V


# ======================================================================
# OPES multithermal: temperature ladder + reweighting (Stage 3)
# ======================================================================

def multithermal_temperature_ladder(temp_min, temp_max, n_temps, *,
                                     spacing='geometric'):
    """Build a temperature ladder (K) for OPES multithermal.

    Parameters
    ----------
    temp_min, temp_max : float  Range endpoints (K), inclusive.
    n_temps : int               Number of temperatures (>= 1).
    spacing : {'geometric', 'uniform'}
        'geometric' (default): ``T_k = temp_min·(temp_max/temp_min)**(k/(n-1))`` —
        constant ratio between neighbours, giving near-uniform overlap of the
        canonical distributions (the natural choice for an expanded ensemble).
        'uniform': linear in temperature.
    """
    import math
    n = int(n_temps)
    if n < 1:
        raise ValueError("n_temps must be >= 1")
    if temp_min <= 0 or temp_max <= 0:
        raise ValueError("temperatures must be positive")
    if n == 1:
        return [float(temp_min)]
    if spacing == 'geometric':
        ratio = (temp_max / temp_min) ** (1.0 / (n - 1))
        return [float(temp_min) * ratio ** k for k in range(n)]
    if spacing == 'uniform':
        return [float(temp_min) + (temp_max - temp_min) * k / (n - 1)
                for k in range(n)]
    raise ValueError(f"spacing must be 'geometric' or 'uniform', got {spacing!r}")


def multithermal_log_weights(potential_energy, bias, temp0, temp_target,
                             kB=_R_KJ):
    """Per-frame log reweighting weights from temp0 to temp_target.

    ``log w_i = β0·V_i − (β′ − β0)·U_i``  with β0 = 1/(kB·temp0),
    β′ = 1/(kB·temp_target). U is the (unbiased) potential energy CV value and V is
    the applied multithermal bias for each frame (see :meth:`Force.multithermal_uv`).
    Returns a NumPy array of unnormalised log-weights.
    """
    import numpy as np
    U = np.asarray(potential_energy, dtype=float)
    V = np.asarray(bias, dtype=float)
    beta0 = 1.0 / (kB * float(temp0))
    betap = 1.0 / (kB * float(temp_target))
    return beta0 * V - (betap - beta0) * U


def kish_ess(log_weights):
    """Kish effective sample size ``(Σw)²/Σw²`` from log-weights (stable).

    Ranges from 1 (one frame dominates) to N (uniform weights). A healthy
    reweighting keeps ESS a sizeable fraction of N.
    """
    import numpy as np
    lw = np.asarray(log_weights, dtype=float)
    lw = lw - np.max(lw)
    w = np.exp(lw)
    return float(w.sum() ** 2 / np.sum(w * w))


def reweight_to_temperature(potential_energy, bias, temp0, temp_target,
                            observable=None, kB=_R_KJ):
    """Reweight multithermal samples from temp0 to temp_target.

    Parameters
    ----------
    potential_energy, bias : sequence[float]
        Per-frame U and V (kJ/mol). See :meth:`Force.multithermal_uv`.
    temp0 : float        Simulation (reference) temperature (K).
    temp_target : float  Temperature to reweight to (K), within the ladder range.
    observable : sequence[float] or None
        Per-frame values of an observable to average. If None, just return the
        normalised weights.

    Returns
    -------
    (weights, ess)         if ``observable`` is None
    (weighted_mean, ess)   otherwise
        ``weights`` are normalised (sum to 1); ``ess`` is the Kish effective
        sample size.
    """
    import numpy as np
    log_w = multithermal_log_weights(potential_energy, bias, temp0, temp_target, kB)
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w /= w.sum()
    ess = float(1.0 / np.sum(w * w))
    if observable is None:
        return w, ess
    obs = np.asarray(observable, dtype=float)
    return float(np.sum(w * obs)), ess
