"""
MultiGPUManager — GLUED multi-GPU scenarios.

Three supported patterns:

  A. Single system across multiple GPUs (OpenMM built-in distribution)
       platform, props = MultiGPUManager.multi_device_platform(devices=[0, 1])
       ctx = mm.Context(system, integrator, platform, props)
     OpenMM natively distributes non-bonded work across the listed devices.
     Our custom force runs on the primary context — no additional changes needed.

  B. One system per GPU — Replica Exchange (H-REUS / T-REMD)
       replicas = MultiGPUManager.build_replicas(factories, devices=[0, 1, 2, 3])
       re = ReplicaExchange(replicas, mode="H-REUS", kT=kT)
       re.run(n_cycles=500, steps_per_cycle=500)
     MultiGPUManager.build_replicas() passes DeviceIndex to each factory so each
     context lands on the requested device without further intervention.

  C. Multiple walkers sharing one bias, across one or more GPUs
       pool = MultiWalkerPool(
           walker_groups,    # [[ctx_gpu0_w0, ctx_gpu0_w1], [ctx_gpu1_w0, ...]],
           force_groups,     # [[f_gpu0_w0,  f_gpu0_w1],  [f_gpu1_w0,  ...]],
           bias_index=0,     # which bias to share
           sync_interval=50, # steps between bias-state merges
       )
       pool.run(n_steps=100_000)
     See MultiWalkerPool for how the shared bias is kept consistent.

Requirements
------------
- openmm >= 8.0
- gluedplugin (this repo, CUDA platform build)
- numpy (for the grid merge)
"""

import math
import random
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import openmm as mm
import openmm.unit as unit

from ReplicaExchange import exchange_log_acceptance

# Bias type codes (GluedForce::BiasType). Spelled out so the parser only needs
# an object with getNumBiases()/getBiasParameters(), not the compiled plugin.
_BIAS_METAD = 3
_BIAS_PBMETAD = 4
_BIAS_OPES = 5
_BIAS_ABMD = 7
_BIAS_OPES_EXPANDED = 11
_BIAS_EXT_LAGRANGIAN = 12
_BIAS_MAXENT = 13
_BIAS_EDS = 14
_BIAS_OPES_MULTITHERMAL = 15


# ---------------------------------------------------------------------------
# BiasStateMerger — parse / merge / repack getBiasState() blobs
#
# Binary layout (little-endian, version 2; mirrors CommonSerialization.cpp):
#   char[4] 'GPUS', int32 version
#   uint64 configurationHash, int64 lastUpdateStep
#   int32 n_opes; per OPES bias (D = numCVs, K = numKernels):
#     int32 K, double sumUprob, int32 numSamples, int32 depositCount
#     double sumWeights, double[2] sumW
#     double[D] runningMean, double[D] runningM2, double[D] sigma0
#     double[K*D] centers, double[K*D] sigmas, double[K] logWeights
#   int32 n_abmd;          per bias: double[D] rhoMin
#   int32 n_metad;         per bias: int32 numDeposited, double[G] grid
#   int32 n_pbmetad;       per bias: int32 n_subgrids, then per sub-grid as MetaD
#   int32 n_external, int32 n_linear, int32 n_wall   (stateless, counts only)
#   int32 n_opes_expanded; per bias: double logZ, int32 numUpdates
#   int32 n_ext_lagrangian; per bias: int32 initialized, double[D] s, double[D] p
#   int32 n_eds;           per bias: double[D] lambda, mean, ssd, accum; int32[D] count
#   int32 n_maxent;        per bias: double[D] lambda
#   int32 n_multithermal;  per bias: int32 N, double[N] deltaF, double rct, double counter
# ---------------------------------------------------------------------------

class BiasStateMerger:
    """Parse, repack and merge ``getBiasState()`` checkpoints.

    ``parse`` returns a dict with one list per bias section (see the layout
    above); ``pack`` is its exact inverse. Grid entries are ``(numDeposited,
    grid_bytes)`` tuples, PBMetaD entries are lists of those, and OPES entries
    keep every field of the section so a walker's complete state can be moved
    between Contexts.

    ``merge_additive`` sums MetaD/PBMetaD grids across blobs. Everything else
    in the merged blob is copied from the first input; grids are the only
    state for which a sum is meaningful.
    """

    @staticmethod
    def parse(blob: bytes, force) -> dict:
        """Parse a blob. ``force`` supplies the per-bias dimensions the blob omits."""
        pos = 0

        def read(fmt):
            nonlocal pos
            size = struct.calcsize(fmt)
            if pos + size > len(blob):
                raise ValueError(
                    f"bias-state blob truncated at offset {pos} (need {size} bytes)")
            values = struct.unpack_from(fmt, blob, pos)
            pos += size
            return values

        def read_i32():
            return read("<i")[0]

        def read_bytes(n):
            nonlocal pos
            if n < 0 or pos + n > len(blob):
                raise ValueError(
                    f"bias-state blob truncated at offset {pos} (need {n} bytes)")
            out = blob[pos:pos + n]
            pos += n
            return out

        def read_count(section, expected):
            n = read_i32()
            if n != expected:
                raise ValueError(f"bias-state {section} count mismatch: blob declares "
                                 f"{n} but the force has {expected}")

        if blob[:4] != b"GPUS":
            raise ValueError("missing bias-state header")
        pos = 4
        version = read_i32()
        if version != 2:
            raise ValueError(f"unsupported bias-state version {version}")
        config_hash = read("<Q")[0]
        last_update = read("<q")[0]

        # Per-bias dimensions from the force configuration, in registration order.
        dims = defaultdict(list)
        for i in range(force.getNumBiases()):
            btype, cv_idxs, _, intparams = force.getBiasParameters(i)
            D = len(cv_idxs)
            if btype == _BIAS_METAD:
                # intparams: [pace, bins_0..bins_D-1, periodic_0..periodic_D-1]
                points = 1
                for d in range(D):
                    points *= intparams[1 + d] + (0 if intparams[1 + D + d] else 1)
                dims[btype].append(points)
            elif btype == _BIAS_PBMETAD:
                # intparams: [pace, bins_0, periodic_0, bins_1, periodic_1, ...]
                dims[btype].append([intparams[1 + 2 * d] + (0 if intparams[2 + 2 * d] else 1)
                                    for d in range(D)])
            else:
                dims[btype].append(D)

        def grid(points):
            return (read_i32(), read_bytes(points * 8))

        read_count("OPES", len(dims[_BIAS_OPES]))
        opes = []
        for D in dims[_BIAS_OPES]:
            K = read_i32()
            sum_uprob = read("<d")[0]
            num_samples, deposit_count = read("<ii")
            sum_weights = read("<d")[0]
            sum_w = read_bytes(16)
            running_mean = read_bytes(D * 8)
            running_m2 = read_bytes(D * 8)
            sigma0 = read_bytes(D * 8)
            centers = read_bytes(K * D * 8)
            sigmas = read_bytes(K * D * 8)
            log_weights = read_bytes(K * 8)
            opes.append((K, sum_uprob, num_samples, deposit_count, sum_weights, sum_w,
                         running_mean, running_m2, sigma0, centers, sigmas, log_weights))

        read_count("ABMD", len(dims[_BIAS_ABMD]))
        abmd = [read_bytes(D * 8) for D in dims[_BIAS_ABMD]]

        read_count("MetaD", len(dims[_BIAS_METAD]))
        metad = [grid(points) for points in dims[_BIAS_METAD]]

        read_count("PBMetaD", len(dims[_BIAS_PBMETAD]))
        pbmetad = []
        for sub_sizes in dims[_BIAS_PBMETAD]:
            read_count("PBMetaD sub-grid", len(sub_sizes))
            pbmetad.append([grid(points) for points in sub_sizes])

        n_external, n_linear, n_wall = read("<iii")

        read_count("OPES_EXPANDED", len(dims[_BIAS_OPES_EXPANDED]))
        opes_expanded = [read("<di") for _ in dims[_BIAS_OPES_EXPANDED]]

        read_count("EXT_LAGRANGIAN", len(dims[_BIAS_EXT_LAGRANGIAN]))
        ext_lag = [(read_i32(), read_bytes(D * 8), read_bytes(D * 8))
                   for D in dims[_BIAS_EXT_LAGRANGIAN]]

        read_count("EDS", len(dims[_BIAS_EDS]))
        eds = [(read_bytes(D * 8), read_bytes(D * 8), read_bytes(D * 8), read_bytes(D * 8),
                read_bytes(D * 4))
               for D in dims[_BIAS_EDS]]

        read_count("MAXENT", len(dims[_BIAS_MAXENT]))
        maxent = [read_bytes(D * 8) for D in dims[_BIAS_MAXENT]]

        read_count("OPES multithermal", len(dims[_BIAS_OPES_MULTITHERMAL]))
        multithermal = []
        for _ in dims[_BIAS_OPES_MULTITHERMAL]:
            n_states = read_i32()
            multithermal.append((read_bytes(n_states * 8), *read("<dd")))
        if pos != len(blob):
            raise ValueError("unexpected trailing bytes in bias-state blob")

        return dict(
            version=version, config_hash=config_hash, last_update=last_update,
            opes=opes, abmd=abmd, metad=metad, pbmetad=pbmetad,
            n_external=n_external, n_linear=n_linear, n_wall=n_wall,
            opes_expanded=opes_expanded, ext_lag=ext_lag, eds=eds, maxent=maxent,
            multithermal=multithermal)

    @staticmethod
    def pack(state: dict) -> bytes:
        """Inverse of :meth:`parse`."""
        buf = bytearray(b"GPUS")
        buf += struct.pack("<iQq", state["version"], state["config_hash"], state["last_update"])

        def i32(v):
            buf.extend(struct.pack("<i", v))

        def grid(entry):
            i32(entry[0])
            buf.extend(entry[1])

        i32(len(state["opes"]))
        for (K, sum_uprob, num_samples, deposit_count, sum_weights, sum_w, running_mean,
             running_m2, sigma0, centers, sigmas, log_weights) in state["opes"]:
            buf += struct.pack("<idiid", K, sum_uprob, num_samples, deposit_count, sum_weights)
            for section in (sum_w, running_mean, running_m2, sigma0, centers, sigmas, log_weights):
                buf.extend(section)
        i32(len(state["abmd"]))
        for rho_min in state["abmd"]:
            buf.extend(rho_min)
        i32(len(state["metad"]))
        for entry in state["metad"]:
            grid(entry)
        i32(len(state["pbmetad"]))
        for sub_grids in state["pbmetad"]:
            i32(len(sub_grids))
            for entry in sub_grids:
                grid(entry)
        buf += struct.pack("<iii", state["n_external"], state["n_linear"], state["n_wall"])
        i32(len(state["opes_expanded"]))
        for log_z, num_updates in state["opes_expanded"]:
            buf += struct.pack("<di", log_z, num_updates)
        i32(len(state["ext_lag"]))
        for initialized, s, p in state["ext_lag"]:
            i32(initialized)
            buf.extend(s)
            buf.extend(p)
        i32(len(state["eds"]))
        for sections in state["eds"]:
            for section in sections:
                buf.extend(section)
        i32(len(state["maxent"]))
        for lam in state["maxent"]:
            buf.extend(lam)
        i32(len(state["multithermal"]))
        for delta_f, rct, counter in state["multithermal"]:
            i32(len(delta_f) // 8)
            buf.extend(delta_f)
            buf += struct.pack("<dd", rct, counter)
        return bytes(buf)

    @staticmethod
    def merge_grid_increments(entries, baselines):
        """Merge grid entries ``(numDeposited, grid_bytes)`` from several walkers.

        ``baselines[i]`` is the entry walker ``i`` started from after the previous
        merge (all baselines are identical then). The result is that common
        baseline plus every walker's increment since, so repeated merges never
        count a hill twice. Pass ``baselines=None`` to sum absolute grids.
        """
        import numpy as np
        if baselines is None:
            baselines = [(0, bytes(len(entries[0][1])))] * len(entries)
        num_deposited, grid = baselines[0]
        grid = np.frombuffer(grid, dtype="<f8").copy()
        for (n, g), (base_n, base_g) in zip(entries, baselines):
            if n < base_n:
                raise ValueError("a walker's grid has fewer deposits than its baseline; "
                                 "broadcast a new baseline before merging")
            num_deposited += n - base_n
            grid += np.frombuffer(g, dtype="<f8") - np.frombuffer(base_g, dtype="<f8")
        return int(num_deposited), grid.tobytes()

    @staticmethod
    def merge_additive(blobs: List[bytes], force) -> bytes:
        """Sum the MetaD/PBMetaD grids of several blobs (first merge only).

        Non-grid sections are copied from ``blobs[0]``. For repeated merges of
        a persistent shared grid use :meth:`merge_additive_incremental`.
        """
        merged, _ = BiasStateMerger.merge_additive_incremental(blobs, force)
        return merged

    @staticmethod
    def merge_additive_incremental(blobs: List[bytes], force, baselines=None):
        """Sum grid increments since the previous merge.

        ``baselines`` is the list returned by the previous call (one parsed
        state per blob, all equal to the state that was broadcast); with it only
        each blob's increment since then is added, so hills are never counted
        twice. Returns ``(merged_blob, new_baselines)``.
        """
        if not blobs:
            raise ValueError("no blobs to merge")
        parsed = [BiasStateMerger.parse(b, force) for b in blobs]
        if baselines is not None and len(baselines) != len(parsed):
            raise ValueError("one baseline per blob is required")
        merged = dict(parsed[0])
        merged["metad"] = [
            BiasStateMerger.merge_grid_increments(
                [p["metad"][i] for p in parsed],
                None if baselines is None else [b["metad"][i] for b in baselines])
            for i in range(len(merged["metad"]))]
        merged["pbmetad"] = [
            [BiasStateMerger.merge_grid_increments(
                 [p["pbmetad"][i][k] for p in parsed],
                 None if baselines is None else [b["pbmetad"][i][k] for b in baselines])
             for k in range(len(sub_grids))]
            for i, sub_grids in enumerate(merged["pbmetad"])]
        return BiasStateMerger.pack(merged), [merged] * len(parsed)


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

# Bias types a pool can share: checkpoint section, and whether the state must
# be passed between walkers one step at a time.
_SHARED_SECTIONS = {
    _BIAS_METAD: ("metad", False),
    _BIAS_PBMETAD: ("pbmetad", False),
    _BIAS_OPES: ("opes", True),
    _BIAS_OPES_EXPANDED: ("opes_expanded", True),
    _BIAS_OPES_MULTITHERMAL: ("multithermal", True),
}


class MultiWalkerPool:
    """
    Several walkers (Contexts) sharing one bias, optionally with replica
    exchange between groups.

    Every walker has its own Context and its own GluedForce built from the same
    definition. ``walker_groups[g]`` lists the walkers that run on one device;
    groups step concurrently on worker threads, walkers within a group step in
    turn. ``bias_index`` selects the shared bias (in ``addBias`` order); all
    other biases keep a private per-walker history.

    How the shared bias is kept consistent depends on its type:

    * **MetaD / PBMetaD** — hills are additive, so each walker deposits into its
      own grid and every ``sync_interval`` steps the increments accumulated by
      all walkers since the previous merge are summed and written back to every
      walker. Between merges a walker does not yet see the others' newest hills,
      so choose ``sync_interval`` around the deposition pace.
    * **OPES / OPES-expanded / multithermal** — the state is a compressed kernel
      table with running statistics that cannot be merged after the fact. The
      walkers therefore advance one MD step at a time in a fixed order, each
      starting from the state the previous walker left, and ``sync_interval``
      only switches sharing on (> 0) or off (0). This is exact but serial: it
      costs a checkpoint transfer per walker step and no concurrency.

    ``sync_interval=0`` leaves every walker's bias independent, which is what
    you want for H-REUS windows that only exchange configurations.

    Parameters
    ----------
    walker_groups : list of list of mm.Context
    force_groups : list of list of GluedForce
        ``force_groups[g][w]`` belongs to ``walker_groups[g][w]``.
    bias_index : int
        Index of the bias to share.
    sync_interval : int
        Steps between grid merges (MetaD/PBMetaD); 0 disables sharing.
    sync_mode : str
        "additive" (default) merges grid increments; "broadcast" copies group 0's
        grid to everyone instead.
    re_mode : str or None
        "H-REUS" or "T-REMD" to also attempt exchanges between the first walker
        of each group every ``re_interval`` steps. See :class:`ReplicaExchange`.
    kT, temperatures, bias_force_group, seed
        As for :class:`ReplicaExchange`. ``bias_force_group`` is accepted for
        backward compatibility; the acceptance test uses complete energies.
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
        if len(walker_groups) != len(force_groups) or not walker_groups:
            raise ValueError("walker_groups and force_groups must be nonempty and the same length")
        for g, (wg, fg) in enumerate(zip(walker_groups, force_groups)):
            if len(wg) != len(fg) or not wg:
                raise ValueError(f"Group {g}: needs one force per walker and at least one walker")
        if not isinstance(sync_interval, int) or sync_interval < 0:
            raise ValueError("sync_interval must be a nonnegative integer")
        if sync_mode not in ("additive", "broadcast"):
            raise ValueError("sync_mode must be 'additive' or 'broadcast'")

        self._groups = [list(wg) for wg in walker_groups]
        self._forces = [list(fg) for fg in force_groups]
        self._walkers = [(ctx, force) for wg, fg in zip(self._groups, self._forces)
                         for ctx, force in zip(wg, fg)]
        self._bias_index = bias_index
        self._sync_interval = sync_interval
        self._sync_mode = sync_mode
        self._re_mode = re_mode.upper() if re_mode else None
        self._re_interval = re_interval
        self._temperatures = list(temperatures) if temperatures else []
        self._rng = random.Random(seed)
        self._elapsed = 0   # MD steps run through this pool so far

        self._re_attempts = 0
        self._re_accepted = 0
        self._pair_attempts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._pair_accepted: Dict[Tuple[int, int], int] = defaultdict(int)

        if self._re_mode == "H-REUS":
            if kT is None or not math.isfinite(kT) or kT <= 0:
                raise ValueError("H-REUS requires a positive kT")
            self._betas = [1.0 / kT] * len(self._groups)
        elif self._re_mode == "T-REMD":
            if len(self._temperatures) != len(self._groups):
                raise ValueError("T-REMD requires one temperature per group")
            if any(not math.isfinite(t) or t <= 0 for t in self._temperatures):
                raise ValueError("replica temperatures must be finite and positive")
            self._betas = [1.0 / (self._GAS_CONSTANT * T) for T in self._temperatures]
        elif self._re_mode is None:
            self._betas = []
        else:
            raise ValueError(f"Unknown re_mode '{re_mode}'; use 'H-REUS', 'T-REMD', or None")
        if self._re_mode is not None and (not isinstance(re_interval, int) or re_interval <= 0):
            raise ValueError("re_interval must be a positive integer")

        # Groups on distinct devices step concurrently: Integrator.step() releases
        # the GIL during the GPU computation.
        self._executor = None
        if len(self._groups) > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=len(self._groups), thread_name_prefix="glued-walker-group")

        self._setup_sharing()

    # ------------------------------------------------------------------
    # Shared-state plumbing
    # ------------------------------------------------------------------

    def _setup_sharing(self):
        """Check the walkers agree on the shared bias and record the starting state."""
        self._sharing = self._sync_interval > 0
        if not self._sharing:
            return
        reference = self._walkers[0][1].getBiasParameters(self._bias_index)
        for _, force in self._walkers:
            if force.getBiasParameters(self._bias_index) != reference:
                raise ValueError("every walker must define the shared bias identically")
        btype = reference[0]
        if btype not in _SHARED_SECTIONS:
            raise ValueError("the shared bias must be MetaD, PBMetaD, OPES, "
                             "OPES-expanded or multithermal")
        self._section, self._ordered = _SHARED_SECTIONS[btype]
        # Position of the shared bias within its section (biases of one type
        # are stored together, in registration order).
        self._local = sum(self._walkers[0][1].getBiasParameters(i)[0] == btype
                          for i in range(self._bias_index))

        # Walkers may differ in their other biases (e.g. H-REUS windows), but
        # the shared bias must start out identical everywhere.
        states = [self._state(ctx, force) for ctx, force in self._walkers]
        if any(self._selected(s) != self._selected(states[0]) for s in states):
            raise ValueError("walkers must start from the same shared bias state")
        # For grids: the entry each walker last received (increments are measured
        # against it). For ordered biases: the single live state.
        self._shared = self._selected(states[0])

    @staticmethod
    def _state(ctx, force) -> dict:
        return BiasStateMerger.parse(force.getBiasState(ctx), force)

    def _selected(self, state: dict):
        return state[self._section][self._local]

    def _write_selected(self, entry, ctx, force):
        """Replace only the shared bias in a walker; its other biases stay as they are."""
        state = self._state(ctx, force)
        state[self._section][self._local] = entry
        force.setBiasState(BiasStateMerger.pack(state), ctx)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, n_steps: int, steps_per_sync: Optional[int] = None):
        """
        Advance all walkers by ``n_steps`` MD steps.

        Merges and exchange attempts fall on absolute multiples of their
        intervals counted from the pool's creation, so repeated short calls
        keep the same schedule as one long call. ``steps_per_sync`` overrides
        ``sync_interval`` for this call.
        """
        if not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        sync_every = self._sync_interval if steps_per_sync is None else steps_per_sync
        if not isinstance(sync_every, int) or sync_every < 0:
            raise ValueError("steps_per_sync must be a nonnegative integer")
        if bool(sync_every) != self._sharing:
            raise ValueError("bias sharing cannot be switched on or off after construction")
        merges = sync_every if (self._sharing and not self._ordered) else 0
        exchanges = self._re_interval if self._re_mode else 0

        def next_multiple(interval):
            return (self._elapsed // interval + 1) * interval

        end = self._elapsed + n_steps
        while self._elapsed < end:
            deadline = end
            if merges:
                deadline = min(deadline, next_multiple(merges))
            if exchanges:
                deadline = min(deadline, next_multiple(exchanges))
            self._step_all(deadline - self._elapsed)
            self._elapsed = deadline
            if merges and self._elapsed % merges == 0:
                self._sync_bias()
            if exchanges and self._elapsed % exchanges == 0:
                self._attempt_re_swaps()

    def _step_all(self, n: int):
        if self._sharing and self._ordered:
            # One shared history: each walker steps from the state the previous
            # walker left, one MD step at a time.
            state = self._shared
            for _ in range(n):
                for ctx, force in self._walkers:
                    self._write_selected(state, ctx, force)
                    ctx.getIntegrator().step(1)
                    state = self._selected(self._state(ctx, force))
            self._shared = state
            for ctx, force in self._walkers[:-1]:
                self._write_selected(state, ctx, force)
            return

        def step_group(walkers):
            for ctx in walkers:
                ctx.getIntegrator().step(n)

        if self._executor is None:
            step_group(self._groups[0])
            return
        futures = [self._executor.submit(step_group, wg) for wg in self._groups]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 — collect, then re-raise the first
                errors.append(exc)
        if errors:
            raise errors[0]

    def close(self):
        """Shut down the per-group stepping thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------
    # Grid merging
    # ------------------------------------------------------------------

    def _sync_bias(self):
        """Merge every walker's grid increments and give all walkers the result."""
        entries = [self._selected(self._state(ctx, force)) for ctx, force in self._walkers]
        if self._sync_mode == "broadcast":
            merged = entries[0]
        else:
            baselines = [self._shared] * len(entries)
            if self._section == "metad":
                merged = BiasStateMerger.merge_grid_increments(entries, baselines)
            else:
                merged = [BiasStateMerger.merge_grid_increments(
                              [e[k] for e in entries], [b[k] for b in baselines])
                          for k in range(len(entries[0]))]
        self._broadcast(merged)

    def _broadcast(self, entry):
        for ctx, force in self._walkers:
            self._write_selected(entry, ctx, force)
        self._shared = entry

    def sync_bias_state_from(self, source_group: int):
        """Replace every walker's shared bias with group ``source_group``'s copy."""
        if not self._sharing:
            raise ValueError("bias sharing is disabled for this pool")
        ctx, force = self._groups[source_group][0], self._forces[source_group][0]
        self._broadcast(self._selected(self._state(ctx, force)))

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
        for i in range(parity, n - 1, 2):
            self._attempt_swap(i, i + 1)

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
        # Box vectors travel with the configuration (they differ under NPT).
        box_i = si.getPeriodicBoxVectors()
        box_j = sj.getPeriodicBoxVectors()

        delta = exchange_log_acceptance(ctx_i, ctx_j, si, sj, beta_i, beta_j)
        scale = math.sqrt(beta_j / beta_i)   # velocity rescaling between temperatures
        v_i_new, v_j_new = v_j * scale, v_i / scale

        key = (min(i, j), max(i, j))
        self._pair_attempts[key] += 1
        self._re_attempts += 1

        accepted = delta >= 0.0 or self._rng.random() < math.exp(delta)
        if accepted:
            self._re_accepted += 1
            self._pair_accepted[key] += 1
            ctx_i.setPeriodicBoxVectors(*box_j)
            ctx_i.setPositions(x_j)
            ctx_i.setVelocities(v_i_new)
            ctx_j.setPeriodicBoxVectors(*box_i)
            ctx_j.setPositions(x_i)
            ctx_j.setVelocities(v_j_new)
        return accepted

    # ------------------------------------------------------------------
    # Statistics and convenience
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
        return len(self._walkers)

    def get_cv_values(self, group: int = 0, walker: int = 0) -> List[float]:
        """Return current CV values for a specific walker."""
        ctx = self._groups[group][walker]
        force = self._forces[group][walker]
        return list(force.getCurrentCVValues(ctx))
