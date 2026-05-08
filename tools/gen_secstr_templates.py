#!/usr/bin/env python3
"""
gen_secstr_templates.py — Generate secondary structure reference templates.

Derives coordinates independently from:
  - Engh & Huber (1991) Acta Cryst. A47:392 — ideal bond geometry
  - Pauling & Corey (1951) Proc. Natl. Acad. Sci. — backbone dihedrals
  - NERF algorithm, Parsons et al. (2005) J. Comput. Chem. 26:1063

Atom order per residue: N, CA, CB, C, O  (5 atoms × 6 residues = 30 per window).
Output: C++ float literals (nm, centred to origin) ready for CommonGluedKernels.cpp.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Engh & Huber (1991) ideal bond lengths (Angstroms) and angles (radians)
# ---------------------------------------------------------------------------
B_NC   = 1.329   # peptide bond C(i-1)-N(i)
B_NCA  = 1.458   # N-CA
B_CAC  = 1.525   # CA-C
B_CO   = 1.229   # C=O
B_CACB = 1.521   # CA-CB

A_CNC   = np.radians(121.7)   # bond angle at N: C-N-CA
A_NCAC  = np.radians(111.2)   # bond angle at CA: N-CA-C  (tau)
A_CACN  = np.radians(116.2)   # bond angle at C: CA-C-N
A_CACO  = np.radians(120.8)   # bond angle at C: CA-C=O
A_NCACB = np.radians(110.5)   # bond angle at CA: N-CA-CB


def nerf(A, B, C, d, theta, phi):
    """
    Place D given chain A-B-C-D.
    |CD|=d, angle(BCD)=theta, dihedral(ABCD)=phi.
    Parsons et al. (2005) eq. (2).
    """
    bc = C - B
    bc = bc / np.linalg.norm(bc)
    ba = A - B
    ba_perp = ba - np.dot(ba, bc) * bc
    n_perp = np.linalg.norm(ba_perp)
    if n_perp < 1e-10:
        perp = np.array([1.0, 0.0, 0.0]) if abs(bc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ba_perp = perp - np.dot(perp, bc) * bc
    n = ba_perp / np.linalg.norm(ba_perp)
    m = np.cross(bc, n)
    return C + d * (-np.cos(theta) * bc
                    + np.sin(theta) * np.cos(phi) * n
                    + np.sin(theta) * np.sin(phi) * m)


def build_strand(n_res, phi_deg, psi_deg, omega_deg=180.0):
    """
    Build an n_res-residue backbone using ideal Engh & Huber geometry
    and Pauling & Corey (1951) backbone dihedral angles.

    Returns five lists of np.array([x,y,z]) in Angstroms:
      N_pos, CA_pos, CB_pos, C_pos, O_pos  (each length n_res)
    """
    phi   = np.radians(phi_deg)
    psi   = np.radians(psi_deg)
    omega = np.radians(omega_deg)

    N_pos  = [None] * n_res
    CA_pos = [None] * n_res
    C_pos  = [None] * n_res
    CB_pos = [None] * n_res
    O_pos  = [None] * n_res

    # --- seed the first residue ---
    # N1 at the origin; CA1 along +x; C1 in the xy-plane.
    N_pos[0]  = np.array([0.0, 0.0, 0.0])
    CA_pos[0] = np.array([B_NCA, 0.0, 0.0])
    # Angle N-CA-C = A_NCAC at CA1: C1 is in the +y half-plane.
    C_pos[0]  = CA_pos[0] + B_CAC * np.array(
        [-np.cos(A_NCAC), np.sin(A_NCAC), 0.0])

    # --- extend backbone, residue by residue ---
    for i in range(1, n_res):
        pN  = N_pos[i-1];  pCA = CA_pos[i-1];  pC = C_pos[i-1]
        # N_i: dihedral(N_{i-1}-CA_{i-1}-C_{i-1}-N_i) = psi
        N_pos[i]  = nerf(pN,  pCA, pC,   B_NC,   A_CACN,  psi)
        # CA_i: dihedral(CA_{i-1}-C_{i-1}-N_i-CA_i) = omega
        CA_pos[i] = nerf(pCA, pC,  N_pos[i],  B_NCA,  A_CNC,   omega)
        # C_i: dihedral(C_{i-1}-N_i-CA_i-C_i) = phi
        C_pos[i]  = nerf(pC,  N_pos[i], CA_pos[i], B_CAC,  A_NCAC,  phi)

    # --- CB (L-amino-acid gauche-minus placement) ---
    # dihedral(C_i - N_i - CA_i - CB_i) = -120°  (staggered g- rotamer)
    phi_CB = np.radians(-120.0)
    for i in range(n_res):
        CB_pos[i] = nerf(C_pos[i], N_pos[i], CA_pos[i],
                         B_CACB, A_NCACB, phi_CB)

    # --- O (carbonyl, trans-peptide plane) ---
    # dihedral(N_i - CA_i - C_i - O_i) = psi + 180°
    phi_O = psi + np.pi
    for i in range(n_res):
        O_pos[i] = nerf(N_pos[i], CA_pos[i], C_pos[i],
                        B_CO, A_CACO, phi_O)

    return N_pos, CA_pos, CB_pos, C_pos, O_pos


def interleave(N, CA, CB, C, O):
    """Return flat list of atoms in per-residue order: N,CA,CB,C,O."""
    atoms = []
    for i in range(len(N)):
        atoms.extend([N[i], CA[i], CB[i], C[i], O[i]])
    return atoms


def centroid(atoms):
    n = len(atoms)
    return np.array([sum(a[k] for a in atoms) / n for k in range(3)])


def centre_and_nm(atoms_ang):
    """Subtract centroid and convert Angstroms → nm."""
    c = centroid(atoms_ang)
    return [(a - c) / 10.0 for a in atoms_ang]


# ---------------------------------------------------------------------------
# Build antiparallel beta sheet
# ---------------------------------------------------------------------------
def build_antibeta(phi_deg=-139.0, psi_deg=135.0):
    """
    Two 3-residue beta strands, antiparallel.
    Strand 1 runs +x; strand 2 runs -x at ~4.7 Å interstrand distance (+z).
    Returns 30-atom list [strand1 (15 atoms), strand2 (15 atoms)].
    """
    N1, CA1, CB1, C1, O1 = build_strand(3, phi_deg, psi_deg)
    N2, CA2, CB2, C2, O2 = build_strand(3, phi_deg, psi_deg)

    # Flip strand 2: reverse residue order and mirror the strand direction
    # For antiparallel: strand 2 runs in the -x direction relative to strand 1.
    # Reflect x-coordinate and reverse residue list.
    def flip_x(lst):
        flipped = [np.array([-a[0], a[1], a[2]]) for a in lst]
        return flipped[::-1]

    N2  = flip_x(N2);  CA2 = flip_x(CA2);  CB2 = flip_x(CB2)
    C2  = flip_x(C2);  O2  = flip_x(O2)

    # Translate strand 2 in z by the interstrand spacing.
    # Antiparallel beta: CA-CA interstrand ~4.72 Å (Pauling & Corey 1951).
    dz = 4.72  # Angstroms

    # Align strand 2 centroids in x and y, offset in z only.
    c1_xy = centroid(interleave(N1, CA1, CB1, C1, O1))
    c2_xy = centroid(interleave(N2, CA2, CB2, C2, O2))
    shift = np.array([c1_xy[0] - c2_xy[0], c1_xy[1] - c2_xy[1], dz])

    N2  = [a + shift for a in N2];  CA2 = [a + shift for a in CA2]
    CB2 = [a + shift for a in CB2]; C2  = [a + shift for a in C2]
    O2  = [a + shift for a in O2]

    atoms = interleave(N1, CA1, CB1, C1, O1) + interleave(N2, CA2, CB2, C2, O2)
    return centre_and_nm(atoms)


# ---------------------------------------------------------------------------
# Build parallel beta sheet
# ---------------------------------------------------------------------------
def build_parabeta(phi_deg=-119.0, psi_deg=113.0):
    """
    Two 3-residue beta strands, parallel, at ~4.85 Å interstrand distance.
    Returns two 30-atom templates (template1 and template2); parallel beta requires two H-bond register templates.
    """
    N1, CA1, CB1, C1, O1 = build_strand(3, phi_deg, psi_deg)
    N2, CA2, CB2, C2, O2 = build_strand(3, phi_deg, psi_deg)

    # Parallel: strand 2 runs in the same direction (+x) as strand 1.
    # Translate strand 2 in z by ~4.85 Å (Pauling & Corey 1951).
    dz = 4.85  # Angstroms

    c1_xy = centroid(interleave(N1, CA1, CB1, C1, O1))
    c2_xy = centroid(interleave(N2, CA2, CB2, C2, O2))
    shift = np.array([c1_xy[0] - c2_xy[0], c1_xy[1] - c2_xy[1], dz])

    N2  = [a + shift for a in N2];  CA2 = [a + shift for a in CA2]
    CB2 = [a + shift for a in CB2]; C2  = [a + shift for a in C2]
    O2  = [a + shift for a in O2]

    atoms = interleave(N1, CA1, CB1, C1, O1) + interleave(N2, CA2, CB2, C2, O2)
    tmpl1 = centre_and_nm(atoms)

    # Template 2: shift both strands by half a residue (register shift).
    # Pauling & Corey parallel beta has two distinct H-bond registers.
    # Second template: strand 2 shifted by ~3.4 Å (rise per residue) in x.
    N2b  = [a + np.array([3.4, 0.0, 0.0]) for a in N2]
    CA2b = [a + np.array([3.4, 0.0, 0.0]) for a in CA2]
    CB2b = [a + np.array([3.4, 0.0, 0.0]) for a in CB2]
    C2b  = [a + np.array([3.4, 0.0, 0.0]) for a in C2]
    O2b  = [a + np.array([3.4, 0.0, 0.0]) for a in O2]

    atoms2 = interleave(N1, CA1, CB1, C1, O1) + interleave(N2b, CA2b, CB2b, C2b, O2b)
    tmpl2 = centre_and_nm(atoms2)

    return tmpl1, tmpl2


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------
def fmt_cpp(atoms, varx, vary, varz, label):
    lines = [f" // {label}"]
    for i, a in enumerate(atoms):
        lines.append(
            f" {varx}[{i:2d}]={a[0]:.6f}f; {vary}[{i:2d}]={a[1]:.6f}f; {varz}[{i:2d}]={a[2]:.6f}f;"
        )
    return "\n".join(lines)


def fmt_python(atoms, name):
    lines = [f"{name} = ["]
    for i, a in enumerate(atoms):
        comma = "," if i < len(atoms) - 1 else ""
        lines.append(f"    [{a[0]:12.6f}, {a[1]:12.6f}, {a[2]:12.6f}]{comma}")
    lines.append("]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Alpha helix: 6 consecutive residues, phi=-57, psi=-47 (Pauling & Corey 1951)
    N_a, CA_a, CB_a, C_a, O_a = build_strand(6, -57.0, -47.0)
    alpha_atoms_ang = interleave(N_a, CA_a, CB_a, C_a, O_a)
    alpha = centre_and_nm(alpha_atoms_ang)

    # Anti-parallel beta: phi=-139, psi=135 (Pauling & Corey 1951)
    antibeta = build_antibeta(-139.0, 135.0)

    # Parallel beta: phi=-119, psi=113 (Pauling & Corey 1951)
    parabeta1, parabeta2 = build_parabeta(-119.0, 113.0)

    print("=" * 70)
    print("C++ float literals (for CommonGluedKernels.cpp)")
    print("=" * 70)
    print()
    print(fmt_cpp(alpha,    "tx1", "ty1", "tz1",
                  "Alpha helix — Engh&Huber (1991) geometry, Pauling&Corey phi=-57,psi=-47"))
    print()
    print(fmt_cpp(antibeta, "tx1", "ty1", "tz1",
                  "Antiparallel beta — Engh&Huber (1991) geometry, Pauling&Corey phi=-139,psi=135"))
    print()
    print(fmt_cpp(parabeta1, "tx1", "ty1", "tz1",
                  "Parallel beta template 1 — Engh&Huber (1991) geometry, Pauling&Corey phi=-119,psi=113"))
    print()
    print(fmt_cpp(parabeta2, "tx2", "ty2", "tz2",
                  "Parallel beta template 2 — Engh&Huber (1991) geometry, register-shifted"))
    print()
    print("=" * 70)
    print("Python lists (for test_cv_secondary_structure.py)")
    print("=" * 70)
    print()
    print(fmt_python(alpha,    "ALPHA_TEMPLATE"))
    print()
    print(fmt_python(antibeta, "ANTIBETA_TEMPLATE"))
    print()
    print(fmt_python(parabeta1, "PARABETA_TEMPLATE1"))
    print()
    print(fmt_python(parabeta2, "PARABETA_TEMPLATE2"))

    # Sanity checks
    print()
    print("Sanity checks:")
    for name, atoms in [("alpha", alpha), ("antibeta", antibeta),
                         ("parabeta1", parabeta1), ("parabeta2", parabeta2)]:
        cx = sum(a[0] for a in atoms) / 30
        cy = sum(a[1] for a in atoms) / 30
        cz = sum(a[2] for a in atoms) / 30
        rms_size = np.sqrt(sum(sum(a[k]**2 for k in range(3)) for a in atoms) / 30)
        print(f"  {name}: centroid=({cx:.2e},{cy:.2e},{cz:.2e}), RMS_size={rms_size:.4f} nm")

    # Verify alpha helix geometry: check CA-CA distance between consecutive residues
    print()
    print("Alpha helix CA-CA distances (should be ~0.35 nm for helix rise):")
    for i in range(5):
        ca_i  = np.array(alpha[i*5 + 1])  # CA is atom index 1 in each residue
        ca_ip1 = np.array(alpha[(i+1)*5 + 1])
        d = np.linalg.norm(ca_ip1 - ca_i)
        print(f"  CA{i+1}-CA{i+2}: {d:.4f} nm")

    print()
    print("Beta strand rise per residue (CA-CA within strand 1):")
    for name, atoms in [("antibeta", antibeta), ("parabeta1", parabeta1)]:
        ca0 = np.array(atoms[1])
        ca1 = np.array(atoms[6])
        ca2 = np.array(atoms[11])
        d01 = np.linalg.norm(ca1 - ca0)
        d12 = np.linalg.norm(ca2 - ca1)
        print(f"  {name}: CA1-CA2={d01:.4f} nm, CA2-CA3={d12:.4f} nm (expect ~0.34 nm)")
