# Collective Variables

All CVs are registered before the OpenMM `Context` is created. Each call returns the **CV value index** — an integer used to refer to this CV when registering biases.

**Recommended** — use named methods on `glued.Force`:

```python
import glued
force = glued.Force(pbc=True, temperature=300.0)
cv_idx = force.add_distance([0, 5])   # plain Python list, no mm.vectori()
```

**Low-level** — use `addCollectiveVariable` directly (also works on `glued.Force`):

```python
import gluedplugin as gp
cv_idx = force.addCollectiveVariable(gp.GluedForce.CV_DISTANCE, [0, 5], [])
```

`CV_PATH` is special: it produces **two** consecutive output values (`s` and `z`). `addCollectiveVariable` returns the index of `s`; `z` is at `s_index + 1`.

Units follow OpenMM conventions: distances in **nm**, angles in **radians**, energies in **kJ/mol**.

---

## CV_DISTANCE (1)

Euclidean distance between two atoms, with optional PBC minimum-image.

| Argument | Value |
|---|---|
| `atoms` | `[atom_a, atom_b]` |
| `parameters` | `[]` |

```python
d = force.add_distance([0, 5])
```

---

## CV_ANGLE (2)

Angle at the central atom formed by three atoms (a–b–c), in radians ∈ [0, π].

| Argument | Value |
|---|---|
| `atoms` | `[atom_a, atom_b, atom_c]` — b is the vertex |
| `parameters` | `[]` |

```python
ang = force.add_angle([2, 4, 6])
```

---

## CV_DIHEDRAL (3)

Torsion angle defined by four atoms, in radians ∈ (−π, π].

| Argument | Value |
|---|---|
| `atoms` | `[atom_a, atom_b, atom_c, atom_d]` |
| `parameters` | `[]` |

```python
phi = force.add_dihedral([4, 6, 8, 14])
```

---

## CV_COM_DISTANCE (4)

Distance between the centres of mass of two atom groups.

| Argument | Value |
|---|---|
| `atoms` | `[n1, a0, a1, ..., a_{n1-1}, b0, b1, ..., b_{n2-1}]` — first element is the count of group-1 atoms; remainder are group-1 then group-2 atom indices |
| `parameters` | `[m0, m1, ..., m_{n1+n2-1}]` — masses in the same order as the non-count atoms |

```python
# Group 1: atoms 0,1,2 (mass 12 each); Group 2: atoms 5,6 (mass 14 each)
atoms  = [3, 0, 1, 2, 5, 6]
masses = [12.0, 12.0, 12.0, 14.0, 14.0]
com_d = force.addCollectiveVariable(gp.GluedForce.CV_COM_DISTANCE, atoms, masses)
```

---

## CV_GYRATION (5)

Mass-weighted radius of gyration of an atom group.

| Argument | Value |
|---|---|
| `atoms` | `[atom_0, atom_1, ..., atom_{N-1}]` |
| `parameters` | `[m_0, m_1, ..., m_{N-1}]` — atomic masses |

```python
atoms  = list(range(10))
masses = [12.0] * 10
rg = force.addCollectiveVariable(gp.GluedForce.CV_GYRATION, atoms, masses)
```

---

## CV_COORDINATION (6)

Smooth coordination number between two groups using a switching function:

`CN = Σ_{i∈G1} Σ_{j∈G2} [ 1 − (r_ij/r0)^n ] / [ 1 − (r_ij/r0)^m ]`

| Argument | Value |
|---|---|
| `atoms` | `[n1, a0...a_{n1-1}, b0...b_{n2-1}]` — same layout as COM_DISTANCE but no masses |
| `parameters` | `[r0, n, m]` — cutoff distance (nm), numerator exponent, denominator exponent |

```python
# 3 atoms in group 1, 4 in group 2; r0=0.35 nm, n=6, m=12
atoms = [3, 0, 1, 2, 5, 6, 7, 8]
cn = force.addCollectiveVariable(gp.GluedForce.CV_COORDINATION, atoms, [0.35, 6.0, 12.0])
```

---

## CV_RMSD (7)

RMSD from a reference structure, computed over a selected set of atoms (mass-weighted if masses are provided; equal-weight if all masses are 1).

| Argument | Value |
|---|---|
| `atoms` | `[atom_0, ..., atom_{N-1}]` |
| `parameters` | `[m_0, ..., m_{N-1}, x0_0, y0_0, z0_0, x0_1, y0_1, z0_1, ...]` — masses first, then reference positions in nm (row-major, xyz per atom) |

```python
ref_pos = [0.1, 0.2, 0.3,  0.4, 0.5, 0.6]  # 2 atoms × xyz
atoms   = [0, 1]
masses  = [12.0, 14.0]
rmsd = force.addCollectiveVariable(gp.GluedForce.CV_RMSD, atoms, masses + ref_pos)
```

---

## CV_EXPRESSION (8)

An algebraic function of other CVs, differentiated symbolically via Lepton. Use `addExpressionCV` instead of `addCollectiveVariable`.

```python
d1 = force.add_distance([0, 1])
d2 = force.add_distance([2, 3])
ratio = force.add_expression("cv0 / cv1", [d1, d2])
```

The expression string uses `cv0`, `cv1`, … as variables, mapped to `inputCVIndices` in order. Any algebraic expression supported by Lepton (including `sin`, `cos`, `exp`, `log`, `sqrt`, `min`, `max`, `step`) is valid.

---

## CV_PYTORCH (9)

A TorchScript model evaluated on GPU. The model receives atom positions as a `[N, 3]` float32 tensor (nm) and must return a scalar.

> **Requires:** CUDA platform built with `libtorch` (`GLUED_HAS_TORCH`).

```python
pt_idx = force.addPyTorchCV(
    "/path/to/model.pt",   # TorchScript model (model.save())
    [0, 1, 2, 5, 6],       # atom indices fed to the model
    []                      # optional scalar hyperparameters
)
```

The model's output is treated as the CV value; GLUED computes forces via automatic differentiation through the model.

---

## CV_PATH (10)

Path collective variable (Branduardi et al. 2007). Defines a progress coordinate `s` along a path of reference frames and a distance-from-path `z`. Returns **two** CV values: `s` at the returned index, `z` at `index + 1`.

| Argument | Value |
|---|---|
| `atoms` | `[atom_0, ..., atom_{N-1}]` |
| `parameters` | `[lambda, N_frames, x0_f0, y0_f0, z0_f0, ..., x_{N-1}_fM, y_{N-1}_fM, z_{N-1}_fM]` — λ, frame count, then all reference positions row-by-row (atom-major within each frame) |

```python
s_idx = force.addCollectiveVariable(
    gp.GluedForce.CV_PATH,
    [0, 1, 2],          # 3-atom path
    [100.0, 3,          # lambda=100, 3 frames
     0.1,0.2,0.3,       # frame 0, atom 0
     0.4,0.5,0.6,       # frame 0, atom 1
     0.7,0.8,0.9,       # frame 0, atom 2
     # ... frames 1 and 2 ...
    ]
)
z_idx = s_idx + 1
```

---

## CV_POSITION (11)

Cartesian position of a single atom along one axis.

| Argument | Value |
|---|---|
| `atoms` | `[atom]` |
| `parameters` | `[component]` — 0 = x, 1 = y, 2 = z |

```python
x_pos = force.add_position(0, component=0)   # component: 0=x, 1=y, 2=z
```

---

## CV_DRMSD (12)

Distance RMSD: RMSD between the set of pairwise distances in the current configuration and a reference set.

| Argument | Value |
|---|---|
| `atoms` | flat list of atom pairs: `[a0, b0, a1, b1, ..., a_{P-1}, b_{P-1}]` — P pairs |
| `parameters` | `[ref_d0, ref_d1, ..., ref_d_{P-1}]` — reference distances in nm |

```python
# 3 pairs: (0,1), (1,2), (0,2)
atoms = [0, 1, 1, 2, 0, 2]
refs  = [0.15, 0.15, 0.21]
drmsd = force.addCollectiveVariable(gp.GluedForce.CV_DRMSD, atoms, refs)
```

---

## CV_CONTACTMAP (13)

Weighted sum of switching-function contact indicators over atom pairs.

For each pair `(i, j)` with reference contact `ref_ij`:

`CV = Σ w_ij * { [1−(r_ij/r0_ij)^n_ij] / [1−(r_ij/r0_ij)^m_ij] }`

| Argument | Value |
|---|---|
| `atoms` | flat pair list: `[a0, b0, a1, b1, ...]` |
| `parameters` | per-pair `[r0, n, m, w, ref]` — 5 values per pair |

```python
atoms  = [0, 5, 1, 6]   # 2 pairs
params = [
    0.35, 6.0, 12.0, 1.0, 1.0,   # pair 0: r0, n, m, weight, ref
    0.35, 6.0, 12.0, 1.0, 1.0,   # pair 1
]
cmap = force.addCollectiveVariable(gp.GluedForce.CV_CONTACTMAP, atoms, params)
```

---

## CV_PLANE (14)

Signed distance from an atom to a plane defined by three other atoms.

| Argument | Value |
|---|---|
| `atoms` | `[p0, p1, p2, query]` — first three define the plane, fourth is the query point |
| `parameters` | `[]` |

```python
plane_d = force.add_plane_distance([0, 1, 2], 5)
```

---

## CV_PROJECTION (15)

Projection of the vector between two atoms onto the axis defined by two other atoms.

| Argument | Value |
|---|---|
| `atoms` | `[vec_a, vec_b, axis_a, axis_b]` — vector is b−a, axis is axis_b−axis_a |
| `parameters` | `[]` |

```python
proj = force.add_projection(0, 1, 2, 3)
```

---

## CV_PUCKERING (16)

Cremer-Pople ring-puckering coordinate. Supports 5-membered (furanose) and 6-membered (pyranose) rings.

| Argument | Value |
|---|---|
| `atoms` | ring atoms in order `[C1, C2, C3, C4, C5]` or `[C1, C2, C3, C4, C5, C6]` |
| `parameters` | `[ring_size, component]` — ring_size = 5 or 6; component: 0 = Q (amplitude), 1 = θ (polar angle), 2 = φ (azimuthal angle for 6-ring) |

```python
# 6-membered ring, azimuthal angle φ
phi_cp = force.addCollectiveVariable(
    gp.GluedForce.CV_PUCKERING,
    [0, 1, 2, 3, 4, 5],  # ring atoms
    [6.0, 2.0]           # ring_size=6, component=φ
)
```

---

## CV_DIPOLE (17)

Dipole moment of a group of atoms, or one component of it.

| Argument | Value |
|---|---|
| `atoms` | `[atom_0, ..., atom_{N-1}]` |
| `parameters` | `[component, q_0, q_1, ..., q_{N-1}]` — component: 0 = magnitude, 1 = x, 2 = y, 3 = z; followed by partial charges |

```python
dip_z = force.addCollectiveVariable(
    gp.GluedForce.CV_DIPOLE,
    [0, 1, 2],
    [3.0, -0.5, 0.25, 0.25]   # component=z, then charges
)
```

---

## CV_VOLUME (18)

Simulation box volume in nm³. No atoms or parameters required.

```python
vol = force.add_volume()
```

---

## CV_CELL (19)

One cell parameter (length or angle) of the simulation box.

| Argument | Value |
|---|---|
| `atoms` | `[]` |
| `parameters` | `[component]` — 0 = a, 1 = b, 2 = c (lengths in nm); 3 = α, 4 = β, 5 = γ (angles in radians) |

```python
cell_c = force.addCollectiveVariable(gp.GluedForce.CV_CELL, [], [2.0])
```

---

## CV_SECONDARY_STRUCTURE (20)

Translation-fit RMSD-based secondary structure score against an ideal reference geometry. Measures how closely a backbone segment resembles α-helix, antiparallel β-sheet, or parallel β-sheet.

The score is a sum of rational switching functions `s(RMSD)` over overlapping 30-atom windows:

`s(x) = (1 − (x/r₀)⁶) / (1 − (x/r₀)¹²)`

| Argument | Value |
|---|---|
| `atoms` | flat list of backbone atoms, **5 per residue** in order: N, CA, CB, C, O |
| `parameters` | `[subtype, r0]` — subtype: 0=α-helix, 1=antiparallel-β, 2=parallel-β; r0: switching cutoff in nm |

**Atom requirements:**
- α-helix (`subtype=0`): at least 6 consecutive residues (30 atoms); each additional 5 atoms adds one window.
- β-sheet (`subtype=1,2`): multiples of 6 residues (30 atoms); each block of 30 atoms is one window.

```python
# Alpha-helix score for a 10-residue segment (50 atoms, 4 windows)
backbone = [N0, CA0, CB0, C0, O0,  N1, CA1, CB1, C1, O1, ...]  # 50 atoms
helix = force.add_secondary_structure(backbone, subtype=0, r0=0.08)

# Antiparallel β-sheet: 6-residue block (30 atoms, 1 window)
beta = force.add_secondary_structure(backbone30, subtype=1, r0=0.08)
```

---

## CV_PCA (21)

Projection of `(position − mean)` onto a principal-component vector:

`CV = Σ_a (r_a − μ_a) · v_a`

| Argument | Value |
|---|---|
| `atoms` | `[atom_0, ..., atom_{N-1}]` |
| `parameters` | `[μ_x0, μ_y0, μ_z0, ..., μ_x{N-1}, μ_y{N-1}, μ_z{N-1}, v_x0, v_y0, v_z0, ..., v_x{N-1}, v_y{N-1}, v_z{N-1}]` — mean structure first (3N values), then unit eigenvector (3N values) |

The named method computes the flat parameter list from separate mean/eigenvector arrays:

```python
import numpy as np

atoms = [0, 1, 2, 3]                         # 4 CA atoms
mean  = np.array([[0.1,0.2,0.3],              # mean positions, nm
                  [0.4,0.5,0.6],
                  [0.7,0.8,0.9],
                  [1.0,1.1,1.2]])
ev    = np.array([[0.5, 0.5, 0.0],            # unit PC vector
                  [-0.5, 0.5, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0]])
ev   /= np.linalg.norm(ev)                    # ensure unit length

pc1 = force.add_pca(atoms, mean, ev)
```

Low-level (params must be mean then eigenvector, each of length 3N):

```python
params = mean.ravel().tolist() + ev.ravel().tolist()  # mean first, then ev
pc1 = force.addCollectiveVariable(gp.GluedForce.CV_PCA, atoms, params)
```

---

## CV_ERMSD (22)

Bottaro eRMSD — an RNA-specific structural similarity metric based on relative
nucleotide frame orientations (Bottaro et al. 2014, Nucleic Acids Res. 42:13306).
Centroid-only gradient approximation; forces have O(1%) error vs. the exact
analytical gradient.

Each residue is represented by three atoms (P, C4', N1 or N9). The kernel builds
local reference frames per residue and computes 4D G-vectors between all ordered
residue pairs. The parameter list stores **pre-computed G-vectors from the reference
structure** — `add_ermsd` computes these automatically.

| Argument | Value |
|---|---|
| `atoms` | flat atom list: `[P₀, C4'₀, N1/N9₀, P₁, C4'₁, N1/N9₁, ...]` — 3 atoms per residue |
| `parameters` | `[N, cutoff, G₀₁_0, G₀₁_1, G₀₁_2, G₀₁_3, G₀₂_0, ..., G_{N-1,N-2}_3]` — residue count, cutoff radius (nm), then 4×N×(N−1) pre-computed G-vector components |

**Use `add_ermsd` — the raw parameter list requires computing Bottaro G-vectors:**

```python
# 3 RNA residues; atoms: [P, C4', N1] per residue
atoms_per_residue = [[0, 1, 2],   # residue 0
                     [3, 4, 5],   # residue 1
                     [6, 7, 8]]   # residue 2

# reference_positions: any indexable structure giving (x, y, z) in nm
ermsd = force.add_ermsd(
    atoms_per_residue,
    reference_positions=ref_pos,  # e.g. np.array of shape (N_atoms, 3)
    cutoff=2.4,                   # nm, Bottaro default
)
```

The `cutoff` parameter (default 2.4 nm) controls which residue pairs contribute.
Pairs whose centroid separation exceeds `cutoff / 2` are excluded.
