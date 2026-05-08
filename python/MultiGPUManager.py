"""
MultiGPUManager — GLUED multi-GPU scenarios.

Three supported patterns:

  A. Single system across multiple GPUs (OpenMM built-in distribution)
       platform, props = MultiGPUManager.multi_device_platform(devices=[0, 1])
       ctx = mm.Context(system, integrator, platform, props)
     OpenMM natively distributes non-bonded work across the listed devices.
     Our custom force runs on the primary context — no additional changes needed.

  B. One system per GPU — Replica Exchange (H-REUS / T-REMD)
       # Build (context, force) pairs; each context is pinned to a specific GPU:
       replicas = MultiGPUManager.build_replicas(factories, devices=[0, 1, 2, 3])
       # factories is a list of callables: device_idx → (context, force)
       re = ReplicaExchange(replicas, mode="H-REUS", kT=kT)
       re.run(n_cycles=500, steps_per_cycle=500)
     MultiGPUManager.build_replicas() passes DeviceIndex to each factory so each
     context lands on the requested device without further intervention.

  C. Multiple walkers per GPU across multiple GPUs
       pool = MultiWalkerPool(
           walker_groups,    # [[ctx_gpu0_w0, ctx_gpu0_w1], [ctx_gpu1_w0, ...]],
           force_groups,     # [[f_gpu0_w0,  f_gpu0_w1],  [f_gpu1_w0,  ...]],
           bias_index=0,     # which bias to share
           sync_interval=50, # steps between cross-GPU bias state merges
       )
       pool.run(n_steps=100_000)
     Within each GPU group, walkers share GPU bias arrays via getMultiWalkerPtrs /
     setMultiWalkerPtrs (atomic, no CPU round-trip).  Across GPUs, bias state is
     periodically merged at the Python level using getBiasState / setBiasState with
     an additive MetaD-grid merge or an OPES-union merge.

     MultiWalkerPool can also drive replica exchange *between* GPU groups:
       pool = MultiWalkerPool(..., re_mode="H-REUS", kT=kT, re_interval=500)

Requirements
------------
- openmm >= 8.0
- gluedplugin (this repo, CUDA platform build)
- numpy (for additive grid merge)

The module is deliberately import-time-light — numpy is only imported when an
additive MetaD merge is actually requested.
"""

import math
import random
import struct
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import openmm as mm
import openmm.unit as unit

try:
    import gluedplugin as gsp
    _GSP_AVAILABLE = True
except ImportError:
    _GSP_AVAILABLE = False
    gsp = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_energy_kJ(ctx: mm.Context) -> float:
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)


def _group_energy_kJ(ctx: mm.Context, groups_mask: int) -> float:
    return ctx.getState(getEnergy=True, groups=groups_mask).getPotentialEnergy(
    ).value_in_unit(unit.kilojoules_per_mole)


def _bias_type_name(btype: int) -> str:
    if not _GSP_AVAILABLE:
        return f"BIAS_{btype}"
    names = {
        gsp.GluedForce.BIAS_METAD:    "METAD",
        gsp.GluedForce.BIAS_PBMETAD:  "PBMETAD",
        gsp.GluedForce.BIAS_OPES:     "OPES",
        gsp.GluedForce.BIAS_ABMD:     "ABMD",
        gsp.GluedForce.BIAS_EDS:      "EDS",
        gsp.GluedForce.BIAS_EXT_LAGRANGIAN: "EXT_LAGRANGIAN",
        gsp.GluedForce.BIAS_MAXENT:   "MAXENT",
    }
    return names.get(btype, f"BIAS_{btype}")


# ---------------------------------------------------------------------------
# BiasStateMerger — parse / merge / repack getBiasState() blobs
#
# Binary layout (little-endian, version 1):
#   4B magic 'GPUS' + int32 version
#   int32  n_opes
#   per OPES bias (D = numCVs for that bias):
#     int32 nk, double logZ, int32 nSamples
#     double[D] runningMean, double[D] runningM2
#     float[nk*D] centers, float[nk*D] sigmas, float[nk] logWeights
#   int32  n_abmd
#   per ABMD bias (D = numCVs):
#     double[D] rhoMin
#   int32  n_metad
#   per MetaD bias (G = totalGridPoints):
#     int32 numDeposited, double[G] grid
#   int32  n_pbmetad
#   per PBMetaD bias:
#     int32  n_subgrids
#     per sub-grid: int32 numDeposited, double[G_k] grid
#   int32  n_external   (stateless — count tag only)
#   int32  n_linear     (stateless — count tag only)
#   int32  n_wall       (stateless — count tag only)
#   int32  n_opes_expanded
#   per OPES_EXPANDED bias:
#     double logZ, int32 numUpdates
#   int32  n_ext_lagrangian
#   per ExtLag bias (D = numCVs):
#     double[D] s, double[D] p
#   int32  n_eds
#   per EDS bias (D = numCVs):
#     double[D] lambda, mean, ssd, accum; int32[D] count
#   int32  n_maxent
#   per MaxEnt bias (D = numCVs):
#     double[D] lambda
# ---------------------------------------------------------------------------

class BiasStateMerger:
    """
    Parse, inspect, and merge getBiasState() binary blobs.

    Primarily used for cross-GPU MetaD-grid additive merge:
      merged = BiasStateMerger.merge_additive([blob0, blob1], force)

    For OPES, the merge strategy is "select the blob with the most kernels"
    (the richest kernel list) and broadcast it.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def merge_additive(blobs: List[bytes], force) -> bytes:
        """
        Merge multiple bias state blobs into one.

        MetaD grids: element-wise sum of all blobs (walkers collectively fill
        the shared grid — correct for independent-walker multi-walker MetaD).
        numDeposited: sum across blobs.

        OPES: select the blob whose OPES section has the most kernels.

        All other sections (ABMD rhoMin, ExtLag s/p, EDS λ, MaxEnt λ):
        take from blob[0] (walkers in separate groups have independent states).

        Requires numpy.
        """
        if len(blobs) == 0:
            raise ValueError("no blobs to merge")
        if len(blobs) == 1:
            return blobs[0]

        import numpy as np

        # Parse all blobs into structured dicts.
        parsed = [BiasStateMerger._parse(b, force) for b in blobs]

        # Build merged result from parsed[0] as base, then override.
        result = {k: v for k, v in parsed[0].items()}

        # --- MetaD grids: additive sum ---
        if parsed[0]["metad"]:
            merged_metad = []
            for i, (nd0, g0) in enumerate(parsed[0]["metad"]):
                grids_i = [np.frombuffer(bytes(p["metad"][i][1]), dtype="<f8")
                            for p in parsed]
                summed = sum(grids_i)
                total_nd = sum(p["metad"][i][0] for p in parsed)
                merged_metad.append((total_nd, summed.tobytes()))
            result["metad"] = merged_metad

        # --- PBMetaD sub-grids: additive sum ---
        if parsed[0]["pbmetad"]:
            merged_pb = []
            for bi, subgrids0 in enumerate(parsed[0]["pbmetad"]):
                merged_subs = []
                for si, (nd0, g0) in enumerate(subgrids0):
                    grids_i = [np.frombuffer(bytes(p["pbmetad"][bi][si][1]), dtype="<f8")
                                for p in parsed]
                    summed = sum(grids_i)
                    total_nd = sum(p["pbmetad"][bi][si][0] for p in parsed)
                    merged_subs.append((total_nd, summed.tobytes()))
                merged_pb.append(merged_subs)
            result["pbmetad"] = merged_pb

        # --- OPES: pick blob with most kernels ---
        if parsed[0]["opes"]:
            best_idx = 0
            best_total = sum(nk for (nk, _, _, _, _, _, _, _) in parsed[0]["opes"])
            for i, p in enumerate(parsed[1:], 1):
                total = sum(nk for (nk, _, _, _, _, _, _, _) in p["opes"])
                if total > best_total:
                    best_total = total
                    best_idx = i
            result["opes"] = parsed[best_idx]["opes"]
            result["opes_expanded"] = parsed[best_idx]["opes_expanded"]

        return BiasStateMerger._pack(result)

    @staticmethod
    def _parse(blob: bytes, force) -> dict:
        """Parse a getBiasState() blob into a structured dict using force config."""
        p = [0]   # mutable position

        def read_fmt(fmt):
            size = struct.calcsize(fmt)
            val = struct.unpack_from(fmt, blob, p[0])
            p[0] += size
            return val

        def read_i32():
            return read_fmt("<i")[0]

        def read_bytes(n):
            out = blob[p[0]:p[0]+n]
            p[0] += n
            return out

        # Header
        has_header = (len(blob) >= 8 and blob[:4] == b'GPUS')
        if has_header:
            p[0] += 4; read_i32()  # version

        # Gather bias configuration from force.
        opes_dims, abmd_dims, metad_specs, pbmetad_specs = [], [], [], []
        eds_dims, maxent_dims, extlag_dims, opes_expanded_count = [], [], [], 0
        if _GSP_AVAILABLE and force is not None:
            for i in range(force.getNumBiases()):
                btype, cv_idxs, params, intparams = force.getBiasParameters(i)
                D = len(cv_idxs)
                if btype == gsp.GluedForce.BIAS_OPES:
                    opes_dims.append(D)
                elif btype == gsp.GluedForce.BIAS_ABMD:
                    abmd_dims.append(D)
                elif btype == gsp.GluedForce.BIAS_METAD:
                    # Grid size: prod(numBins_d + (0 if isPeriodic else 1))
                    # intparams layout: [pace, numBins_0, isPeriodic_0, numBins_1, ...]
                    G = 1
                    for d in range(D):
                        nb = intparams[1 + d * 2]
                        periodic = intparams[2 + d * 2]
                        G *= (nb if periodic else nb + 1)
                    metad_specs.append(G)
                elif btype == gsp.GluedForce.BIAS_PBMETAD:
                    # PBMetaD: D independent 1D sub-grids
                    # intparams: [pace, numBins_0, isPeriodic_0, numBins_1, ...]
                    sub_sizes = []
                    for d in range(D):
                        nb = intparams[1 + d * 2]
                        periodic = intparams[2 + d * 2]
                        sub_sizes.append(nb if periodic else nb + 1)
                    pbmetad_specs.append(sub_sizes)
                elif btype == gsp.GluedForce.BIAS_EDS:
                    eds_dims.append(D)
                elif btype == gsp.GluedForce.BIAS_MAXENT:
                    maxent_dims.append(D)
                elif btype == gsp.GluedForce.BIAS_EXT_LAGRANGIAN:
                    extlag_dims.append(D)
                elif btype == gsp.GluedForce.BIAS_OPES_EXPANDED:
                    opes_expanded_count += 1

        # Parse OPES section.
        opes = []
        n_opes = read_i32()
        for d_i, D in enumerate(opes_dims[:n_opes]):
            nk = read_i32()
            logZ = read_fmt("<d")[0]
            ns = read_i32()
            mean_bytes = read_bytes(D * 8)
            m2_bytes   = read_bytes(D * 8)
            centers_bytes    = read_bytes(nk * D * 4)
            sigmas_bytes     = read_bytes(nk * D * 4)
            logweights_bytes = read_bytes(nk * 4)
            opes.append((nk, logZ, ns, mean_bytes, m2_bytes,
                         centers_bytes, sigmas_bytes, logweights_bytes))

        # Parse ABMD section.
        abmd = []
        if p[0] < len(blob):
            n_abmd = read_i32()
            for d_i, D in enumerate(abmd_dims[:n_abmd]):
                abmd.append(read_bytes(D * 8))

        # Parse MetaD section.
        metad = []
        if p[0] < len(blob):
            n_metad = read_i32()
            for G in metad_specs[:n_metad]:
                nd = read_i32()
                grid_bytes = read_bytes(G * 8)
                metad.append((nd, grid_bytes))

        # Parse PBMetaD section.
        pbmetad = []
        if p[0] < len(blob):
            n_pbmetad = read_i32()
            for sub_sizes in pbmetad_specs[:n_pbmetad]:
                n_sub = read_i32()
                subs = []
                for G in sub_sizes[:n_sub]:
                    nd = read_i32()
                    grid_bytes = read_bytes(G * 8)
                    subs.append((nd, grid_bytes))
                pbmetad.append(subs)

        # Stateless count tags.
        n_external = read_i32() if p[0] < len(blob) else 0
        n_linear   = read_i32() if p[0] < len(blob) else 0
        n_wall     = read_i32() if p[0] < len(blob) else 0

        # OPES expanded.
        opes_expanded = []
        if p[0] < len(blob):
            n_oe = read_i32()
            for _ in range(n_oe):
                logZ_oe = read_fmt("<d")[0]
                nu_oe   = read_i32()
                opes_expanded.append((logZ_oe, nu_oe))

        # Extended Lagrangian.
        ext_lag = []
        if p[0] < len(blob):
            n_el = read_i32()
            for d_i, D in enumerate(extlag_dims[:n_el]):
                s_bytes = read_bytes(D * 8)
                p_bytes = read_bytes(D * 8)
                ext_lag.append((s_bytes, p_bytes))

        # EDS.
        eds = []
        if p[0] < len(blob):
            n_eds = read_i32()
            for d_i, D in enumerate(eds_dims[:n_eds]):
                lam  = read_bytes(D * 8)
                mean = read_bytes(D * 8)
                ssd  = read_bytes(D * 8)
                acc  = read_bytes(D * 8)
                cnt  = read_bytes(D * 4)
                eds.append((lam, mean, ssd, acc, cnt))

        # MaxEnt.
        maxent = []
        if p[0] < len(blob):
            n_mx = read_i32()
            for d_i, D in enumerate(maxent_dims[:n_mx]):
                maxent.append(read_bytes(D * 8))

        return dict(
            opes=opes, abmd=abmd, metad=metad, pbmetad=pbmetad,
            n_external=n_external, n_linear=n_linear, n_wall=n_wall,
            opes_expanded=opes_expanded, ext_lag=ext_lag,
            eds=eds, maxent=maxent
        )

    @staticmethod
    def _pack(d: dict) -> bytes:
        """Repack a parsed dict back into the GPUS binary format."""
        buf = bytearray()

        def w_bytes(b):
            buf.extend(b)

        def w_i32(v):
            buf.extend(struct.pack("<i", v))

        def w_d(v):
            buf.extend(struct.pack("<d", v))

        # Header.
        buf.extend(b'GPUS')
        w_i32(1)  # version

        # OPES.
        w_i32(len(d["opes"]))
        for (nk, logZ, ns, mean_b, m2_b, cen_b, sig_b, lw_b) in d["opes"]:
            w_i32(nk); w_d(logZ); w_i32(ns)
            w_bytes(mean_b); w_bytes(m2_b)
            w_bytes(cen_b); w_bytes(sig_b); w_bytes(lw_b)

        # ABMD.
        w_i32(len(d["abmd"]))
        for rho_b in d["abmd"]:
            w_bytes(rho_b)

        # MetaD.
        w_i32(len(d["metad"]))
        for (nd, grid_b) in d["metad"]:
            w_i32(nd); w_bytes(grid_b)

        # PBMetaD.
        w_i32(len(d["pbmetad"]))
        for subs in d["pbmetad"]:
            w_i32(len(subs))
            for (nd, grid_b) in subs:
                w_i32(nd); w_bytes(grid_b)

        # Stateless tags.
        w_i32(d.get("n_external", 0))
        w_i32(d.get("n_linear", 0))
        w_i32(d.get("n_wall", 0))

        # OPES expanded.
        w_i32(len(d["opes_expanded"]))
        for (logZ_oe, nu_oe) in d["opes_expanded"]:
            w_d(logZ_oe); w_i32(nu_oe)

        # ExtLag.
        w_i32(len(d["ext_lag"]))
        for (s_b, p_b) in d["ext_lag"]:
            w_bytes(s_b); w_bytes(p_b)

        # EDS.
        w_i32(len(d["eds"]))
        for (lam_b, mean_b, ssd_b, acc_b, cnt_b) in d["eds"]:
            w_bytes(lam_b); w_bytes(mean_b); w_bytes(ssd_b)
            w_bytes(acc_b); w_bytes(cnt_b)

        # MaxEnt.
        w_i32(len(d["maxent"]))
        for lam_b in d["maxent"]:
            w_bytes(lam_b)

        return bytes(buf)


# ---------------------------------------------------------------------------
# Scenario A helper
# ---------------------------------------------------------------------------

class MultiGPUManager:
    """
    Static helpers for multi-GPU OpenMM context setup.
    """

    @staticmethod
    def multi_device_platform(devices: List[int]) -> Tuple[mm.Platform, Dict[str, str]]:
        """
        Return (Platform, props) for distributing one system across multiple GPUs.

        Usage::

            platform, props = MultiGPUManager.multi_device_platform([0, 1])
            ctx = mm.Context(system, integrator, platform, props)

        OpenMM distributes non-bonded work across the listed devices natively.
        GluedForce runs on the primary (first) device — no code changes needed.
        The platform must be "CUDA" (OpenCL does not support multi-device natively
        in the same context).

        Parameters
        ----------
        devices : list of int
            CUDA device indices to use (e.g. [0, 1]).

        Returns
        -------
        platform : mm.Platform
            The CUDA Platform object.
        props : dict
            Platform properties dict to pass to mm.Context().
        """
        try:
            platform = mm.Platform.getPlatformByName("CUDA")
        except Exception as exc:
            raise RuntimeError(
                "CUDA platform not available — cannot create multi-GPU context. "
                "Install openmm with CUDA support."
            ) from exc

        props = {"DeviceIndex": ",".join(str(d) for d in devices)}
        return platform, props

    @staticmethod
    def build_replicas(
        factories: List[Callable[[int], Tuple[mm.Context, object]]],
        devices: Optional[List[int]] = None,
    ) -> List[Tuple[mm.Context, object]]:
        """
        Build a list of (Context, GluedForce) pairs, each on a specific GPU.

        Each factory callable receives a CUDA device index and must return a
        ``(context, force)`` tuple for a fully initialised simulation.

        Parameters
        ----------
        factories : list of callables
            ``factory(device_idx) → (context, force)``
        devices : list of int, optional
            GPU device indices, one per factory.  Defaults to [0, 1, ..., N-1].

        Returns
        -------
        list of (Context, GluedForce)
            Ready for use with ReplicaExchange.
        """
        n = len(factories)
        if devices is None:
            devices = list(range(n))
        if len(devices) != n:
            raise ValueError(f"len(factories)={n} != len(devices)={len(devices)}")

        replicas = []
        for factory, dev in zip(factories, devices):
            ctx, force = factory(dev)
            replicas.append((ctx, force))
        return replicas

    @staticmethod
    def cuda_properties(device_idx: int = 0,
                        precision: str = "mixed") -> Dict[str, str]:
        """
        Return a platform properties dict for a single CUDA device.

        Usage::

            props = MultiGPUManager.cuda_properties(device_idx=1)
            ctx = mm.Context(system, integrator,
                             mm.Platform.getPlatformByName("CUDA"), props)

        Parameters
        ----------
        device_idx : int
            CUDA device index.
        precision : str
            "single", "mixed", or "double".
        """
        return {"DeviceIndex": str(device_idx), "Precision": precision}


# ---------------------------------------------------------------------------
# Scenario C — MultiWalkerPool
# ---------------------------------------------------------------------------

class MultiWalkerPool:
    """
    Multi-walker ensemble across one or more GPUs.

    Architecture
    ~~~~~~~~~~~~
    *Intra-GPU sharing* (within each GPU group): The first walker in each group
    is the "group primary".  All other walkers in the group have their bias GPU
    arrays redirected to the primary's arrays via ``setMultiWalkerPtrs``.
    Deposits from all walkers in the group land atomically into a single set of
    GPU arrays — no CPU round-trip within a group.

    *Cross-GPU merge* (between GPU groups, every ``sync_interval`` steps):
    The group primaries download their bias state via ``getBiasState()``, the
    states are merged (additive for MetaD grids, richest-OPES for OPES kernels),
    and the merged state is uploaded to all group primaries via ``setBiasState()``.

    Parameters
    ----------
    walker_groups : list of list of mm.Context
        ``walker_groups[g][w]`` is the OpenMM Context for walker *w* in GPU
        group *g*.  All walkers within a group must be on the same GPU device.
    force_groups : list of list of GluedForce
        Matching force objects: ``force_groups[g][w]``.
    bias_index : int
        Index of the bias to share (0-based, in registration order).
    sync_interval : int
        Steps between cross-GPU bias-state merges.  Set to 0 to disable
        cross-GPU merging (useful when each GPU group is a fully independent
        replica and you want no shared bias at all).
    sync_mode : str
        "additive" (default) — merge MetaD grids by element-wise sum.
        "broadcast" — copy group-primary-0's state to all other primaries.
    re_mode : str or None
        If not None, also drive H-REUS or T-REMD between the group primaries
        after every ``re_interval`` steps.  Uses one replica per group (the
        group primary).  See :class:`ReplicaExchange` for criteria.
    re_interval : int
        Steps between replica exchange attempts (ignored if re_mode is None).
    kT : float
        Thermal energy in kJ/mol — required for H-REUS.
    temperatures : list of float
        Temperatures in K — required for T-REMD, one per group.
    bias_force_group : int or None
        OpenMM force group of the bias force — required for T-REMD to isolate
        MM energy from bias energy.
    seed : int or None
        RNG seed.
    """

    _GAS_CONSTANT = 8.314462618e-3   # kJ mol⁻¹ K⁻¹

    def __init__(
        self,
        walker_groups: List[List[mm.Context]],
        force_groups: List[List],
        bias_index: int = 0,
        sync_interval: int = 50,
        sync_mode: str = "additive",
        re_mode: Optional[str] = None,
        re_interval: int = 500,
        kT: Optional[float] = None,
        temperatures: Optional[List[float]] = None,
        bias_force_group: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if len(walker_groups) != len(force_groups):
            raise ValueError("walker_groups and force_groups must have the same length")
        for g, (wg, fg) in enumerate(zip(walker_groups, force_groups)):
            if len(wg) != len(fg):
                raise ValueError(f"Group {g}: len(contexts)={len(wg)} != len(forces)={len(fg)}")
            if len(wg) == 0:
                raise ValueError(f"Group {g} has no walkers")

        self._groups = list(walker_groups)
        self._forces = list(force_groups)
        self._bias_index = bias_index
        self._sync_interval = sync_interval
        self._sync_mode = sync_mode
        self._re_mode = re_mode.upper() if re_mode else None
        self._re_interval = re_interval
        self._kT = kT
        self._temperatures = list(temperatures) if temperatures else []
        self._bias_group = bias_force_group
        self._rng = random.Random(seed)

        # RE statistics (between group primaries).
        self._re_attempts = 0
        self._re_accepted = 0
        self._pair_attempts: Dict[Tuple[int,int], int] = defaultdict(int)
        self._pair_accepted: Dict[Tuple[int,int], int] = defaultdict(int)

        # Validate RE parameters.
        if self._re_mode == "H-REUS":
            if kT is None:
                raise ValueError("H-REUS requires kT")
            self._betas = [1.0 / kT] * len(self._groups)
        elif self._re_mode == "T-REMD":
            if not self._temperatures or len(self._temperatures) != len(self._groups):
                raise ValueError("T-REMD requires temperatures list, one per group")
            self._betas = [1.0 / (self._GAS_CONSTANT * T) for T in self._temperatures]
        elif self._re_mode is None:
            self._betas = []
        else:
            raise ValueError(f"Unknown re_mode '{re_mode}'; use 'H-REUS', 'T-REMD', or None")

        # Wire intra-GPU shared GPU arrays.
        self._setup_intra_gpu_sharing()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_intra_gpu_sharing(self):
        """
        Wire setMultiWalkerPtrs so all secondary walkers within each GPU group
        point to the group primary's bias GPU arrays.

        This is called once during __init__.  It requires that all contexts in
        a group are already created and on the same GPU device.
        """
        for g, (wg, fg) in enumerate(zip(self._groups, self._forces)):
            if len(wg) < 2:
                continue  # single walker — nothing to wire
            primary_ctx   = wg[0]
            primary_force = fg[0]
            try:
                ptrs = primary_force.getMultiWalkerPtrs(primary_ctx, self._bias_index)
            except Exception as exc:
                raise RuntimeError(
                    f"Group {g}: getMultiWalkerPtrs failed — ensure this is the CUDA "
                    f"platform and bias_index={self._bias_index} is valid."
                ) from exc
            for w in range(1, len(wg)):
                fg[w].setMultiWalkerPtrs(wg[w], self._bias_index, ptrs)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, n_steps: int, steps_per_sync: Optional[int] = None):
        """
        Run *n_steps* MD steps across all walkers.

        All walkers in every group are advanced in lockstep.  Cross-GPU bias
        merge happens every ``sync_interval`` steps.  Replica exchange (if
        configured) happens every ``re_interval`` steps.

        Parameters
        ----------
        n_steps : int
            Total MD steps to run.
        steps_per_sync : int, optional
            Override ``sync_interval`` for this call only.
        """
        sync_every = steps_per_sync if steps_per_sync is not None else self._sync_interval
        if sync_every <= 0:
            sync_every = n_steps + 1  # never sync

        # Determine the step size per inner loop iteration.
        inner = sync_every
        if self._re_mode is not None:
            inner = min(inner, self._re_interval)
        if inner <= 0:
            inner = 1

        steps_done = 0
        steps_since_sync = 0
        steps_since_re   = 0

        while steps_done < n_steps:
            batch = min(inner, n_steps - steps_done)
            self._step_all(batch)
            steps_done    += batch
            steps_since_sync += batch
            steps_since_re   += batch

            if sync_every <= n_steps and steps_since_sync >= sync_every:
                self._sync_bias()
                steps_since_sync = 0

            if self._re_mode is not None and steps_since_re >= self._re_interval:
                self._attempt_re_swaps()
                steps_since_re = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step_all(self, n: int):
        """Advance all walkers in all groups by n MD steps."""
        for wg in self._groups:
            for ctx in wg:
                ctx.getIntegrator().step(n)

    # ------------------------------------------------------------------
    # Cross-GPU bias sync
    # ------------------------------------------------------------------

    def _sync_bias(self):
        """Merge bias states across GPU groups and broadcast the result."""
        if len(self._groups) < 2:
            return  # nothing to merge for a single group

        # Collect bias states from group primaries.
        primary_forces = [fg[0] for fg in self._forces]
        blobs = [bytes(f.getBiasState()) for f in primary_forces]

        if self._sync_mode == "additive":
            # Representative force for configuration lookup.
            ref_force = primary_forces[0]
            try:
                merged = BiasStateMerger.merge_additive(blobs, ref_force)
            except Exception as exc:
                warnings.warn(
                    f"Additive bias merge failed ({exc}); falling back to broadcast.",
                    stacklevel=2)
                merged = blobs[0]
        else:
            merged = blobs[0]

        # Upload merged state to all group primaries.
        for f in primary_forces:
            f.setBiasState(merged)

    # ------------------------------------------------------------------
    # Replica exchange between group primaries
    # ------------------------------------------------------------------

    def _attempt_re_swaps(self):
        """Attempt H-REUS or T-REMD swaps between group primaries."""
        n = len(self._groups)
        if n < 2:
            return

        # Alternating even/odd pairs.
        parity = self._rng.randint(0, 1)
        pairs  = [(i, i + 1) for i in range(parity, n - 1, 2)]

        for i, j in pairs:
            self._attempt_swap(i, j)

    def _attempt_swap(self, i: int, j: int):
        ctx_i = self._groups[i][0]
        ctx_j = self._groups[j][0]
        beta_i, beta_j = self._betas[i], self._betas[j]

        si = ctx_i.getState(getPositions=True, getVelocities=True, getEnergy=True)
        sj = ctx_j.getState(getPositions=True, getVelocities=True, getEnergy=True)

        x_i = si.getPositions(asNumpy=True)
        x_j = sj.getPositions(asNumpy=True)
        v_i = si.getVelocities(asNumpy=True)
        v_j = sj.getVelocities(asNumpy=True)

        if self._re_mode == "H-REUS":
            E_ii = si.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            E_jj = sj.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            ctx_i.setPositions(x_j);  E_ij = _total_energy_kJ(ctx_i);  ctx_i.setPositions(x_i)
            ctx_j.setPositions(x_i);  E_ji = _total_energy_kJ(ctx_j);  ctx_j.setPositions(x_j)

            delta     = -beta_i * (E_ij + E_ji - E_ii - E_jj)
            v_i_new, v_j_new = v_j, v_i

        else:  # T-REMD
            if self._bias_group is not None:
                U_ii = _total_energy_kJ(ctx_i) - _group_energy_kJ(ctx_i, 1 << self._bias_group)
                U_jj = _total_energy_kJ(ctx_j) - _group_energy_kJ(ctx_j, 1 << self._bias_group)
            else:
                U_ii = si.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                U_jj = sj.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            delta = (beta_i - beta_j) * (U_ii - U_jj)
            Ti, Tj = self._temperatures[i], self._temperatures[j]
            v_i_new = v_j * math.sqrt(Ti / Tj)
            v_j_new = v_i * math.sqrt(Tj / Ti)

        key = (min(i, j), max(i, j))
        self._pair_attempts[key]  += 1
        self._re_attempts         += 1

        accepted = delta >= 0.0 or self._rng.random() < math.exp(min(0.0, delta))
        if accepted:
            self._re_accepted          += 1
            self._pair_accepted[key]   += 1
            ctx_i.setPositions(x_j);  ctx_i.setVelocities(v_i_new)
            ctx_j.setPositions(x_i);  ctx_j.setVelocities(v_j_new)

        return accepted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def re_acceptance_rate(self) -> float:
        """Overall replica exchange acceptance rate across all group-pair swaps."""
        if self._re_attempts == 0:
            return 0.0
        return self._re_accepted / self._re_attempts

    def re_pair_acceptance_rate(self, i: int, j: int) -> float:
        """Acceptance rate for the (i, j) group pair."""
        key = (min(i, j), max(i, j))
        n = self._pair_attempts.get(key, 0)
        return self._pair_accepted.get(key, 0) / n if n > 0 else 0.0

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    @property
    def n_walkers_per_group(self) -> List[int]:
        return [len(wg) for wg in self._groups]

    @property
    def total_walkers(self) -> int:
        return sum(len(wg) for wg in self._groups)

    # ------------------------------------------------------------------
    # Convenience: forward-compatible with ReplicaExchange protocol
    # ------------------------------------------------------------------

    def sync_bias_state_from(self, source_group: int):
        """Broadcast bias state from group *source_group* to all others."""
        _, f_src = self._forces[source_group][0], self._forces[source_group][0]
        f_src = self._forces[source_group][0]
        blob = f_src.getBiasState()
        for g, fg in enumerate(self._forces):
            if g != source_group:
                fg[0].setBiasState(blob)

    def get_cv_values(self, group: int = 0, walker: int = 0) -> List[float]:
        """Return current CV values for a specific walker."""
        ctx   = self._groups[group][walker]
        force = self._forces[group][walker]
        return list(force.getCurrentCVValues(ctx))
