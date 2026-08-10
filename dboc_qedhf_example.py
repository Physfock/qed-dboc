"""
Diagonal Born-Oppenheimer correction (DBOC) for a diatomic molecule in a
single-mode optical cavity, evaluated at the QED-CI(2,0) level (CISD built
on top of a QED-HF reference).

The DBOC is obtained fully numerically from wavefunction overlaps at
displaced nuclear geometries, following the finite-difference scheme of

    E. F. Valeev and C. D. Sherrill, J. Chem. Phys. 118, 3921 (2003)

generalized to the cavity QED case in

    T. Zalialiutdinov, D. Solovyev, J. J. Lopez-Rodriguez, A. Anikin,
    and A. Kotov, "Non-adiabatic Effects Induced by Strong Light-Matter
    Coupling in Cavity QED" (2026).

This script computes the DBOC for LiH at a single bond length in the
cc-pVDZ basis set (chosen for speed in this example), both without a
cavity and with a finite light-matter coupling strength, and reports
the resulting cavity-induced shift.

As a sanity check, the cavity-free result can be compared against an
independent CCSD-DBOC calculation performed with CFOUR (see the
reference input/output reproduced at the bottom of this file).

Dependencies
------------
PySCF   https://pyscf.org
OpenMS  https://github.com/lanl/OpenMS   (provides openms.mqed.qedhf)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np
from pyscf import ci, gto

from openms.lib import boson
from openms.mqed import qedhf, scqedhf

# --------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------- #
BOHR_TO_ANGSTROM = 0.52917721067121
HARTREE_TO_CM1 = 219474.6313632

# Nuclear masses in units of the electron mass, from standard atomic
# weights (natural isotopic abundance). Use the pure-isotope mass
# (e.g. 7.016003 u for 7Li) instead if isotope-resolved DBOC values
# are required.
AMU_TO_ME = 1822.888486209
ATOMIC_MASS = {
    "Li": 6.941 * AMU_TO_ME,
    "H": 1.008 * AMU_TO_ME,
}

# --------------------------------------------------------------------- #
# Molecule / cavity / SCF configuration
# --------------------------------------------------------------------- #
BASIS = "cc-pvdz"

SCF_CONV_TOL = 1e-10
SCF_CONV_TOL_GRAD = 1e-10
SCF_MAX_CYCLE = 300
CISD_CONV_TOL = 1e-10
CISD_CONV_TOL_NORMT = 1e-10


@dataclass
class CavityConfig:
    """Single-mode cavity, polarized along z by construction."""

    omega: float = 0.5      # cavity mode frequency, a.u.
    coupling: float = 0.0   # coupling strength lambda_z, a.u. (0 = no cavity)

    @property
    def frequency(self) -> np.ndarray:
        return np.array([self.omega])

    @property
    def mode_vector(self) -> np.ndarray:
        return np.array([[0.0, 0.0, self.coupling]])


# --------------------------------------------------------------------- #
# Molecule construction (PySCF atom-string convention)
# --------------------------------------------------------------------- #
def build_lih(R: float, atom_index: int | None = None,
              cart_index: int | None = None, delta: float = 0.0) -> gto.Mole:
    """
    Build a LiH molecule with Li at the origin and H along +z at distance R,
    using the standard PySCF atom-string convention.

    Parameters
    ----------
    R : float
        Li-H bond length in Angstrom.
    atom_index : {0, 1}, optional
        Atom to displace (0 = Li, 1 = H). Leave as None for the
        undisplaced (equilibrium) geometry.
    cart_index : {0, 1, 2}, optional
        Cartesian axis to displace along (0 = x, 1 = y, 2 = z).
    delta : float
        Displacement in Angstrom, applied to atom `atom_index` along
        axis `cart_index`. Ignored if `atom_index` is None.
    """
    li_xyz = [0.0, 0.0, 0.0]
    h_xyz = [0.0, 0.0, R]

    if atom_index is not None:
        coords = [li_xyz, h_xyz]
        coords[atom_index][cart_index] += delta

    atom = f"Li {li_xyz[0]} {li_xyz[1]} {li_xyz[2]}; H {h_xyz[0]} {h_xyz[1]} {h_xyz[2]}"

    mol = gto.M(
        atom=atom,
        basis=BASIS,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    mol.linear_dep_threshold = 1e-10
    return mol


# --------------------------------------------------------------------- #
# Electronic structure: QED-HF + CISD
# --------------------------------------------------------------------- #
def run_qedhf_cisd(mol: gto.Mole, cavity: CavityConfig):
    """Run QED-HF, then CISD on top of the converged QED-HF reference."""
    qed = boson.Photon(mol, omega=cavity.frequency, vec=cavity.mode_vector)

    # Generate the mean-field object with QED-HF or SC-QED-HF.
    mf = qedhf.RHF(mol, qed=qed)
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.max_cycle = SCF_MAX_CYCLE
    mf.kernel()

    # Run CISD on top of the previously obtained QED-HF reference.
    myci = ci.CISD(mf)
    myci.conv_tol = CISD_CONV_TOL
    myci.conv_tol_normt = CISD_CONV_TOL_NORMT
    myci.kernel()

    return mf, myci


# --------------------------------------------------------------------- #
# See https://pyscf.org/user/ci.html for the CISD overlap convention.
# --------------------------------------------------------------------- #
def cisd_overlap(mf_ref, ci_ref, mol_ref, mf_disp, ci_disp, mol_disp) -> complex:
    """Overlap <CISD(mol_ref)|CISD(mol_disp)> entering the finite-difference Laplacian."""
    nmo = mf_ref.mo_energy.size
    nocc = mf_ref.mol.nelectron // 2

    s12 = gto.intor_cross("cint1e_ovlp_sph", mol_ref, mol_disp)
    s12 = reduce(np.dot, (mf_ref.mo_coeff.T, s12, mf_disp.mo_coeff))
    return ci.cisd.overlap(ci_ref.ci, ci_disp.ci, nmo, nocc, s12)


# --------------------------------------------------------------------- #
# Finite-difference Laplacian and DBOC
# --------------------------------------------------------------------- #
def laplacian_component(
    R: float,
    atom_index: int,
    cart_index: int,
    dR: float,
    cavity: CavityConfig,
    mf0,
    ci0,
    mol0,
    verbose: bool = True,
) -> float:
    """
    Second derivative of the CISD overlap along one Cartesian direction
    for one atom, evaluated by central finite differences.
    """
    mol_minus = build_lih(R, atom_index, cart_index, delta=-dR)
    mol_plus = build_lih(R, atom_index, cart_index, delta=+dR)

    mf_minus, ci_minus = run_qedhf_cisd(mol_minus, cavity)
    mf_plus, ci_plus = run_qedhf_cisd(mol_plus, cavity)

    s_minus = cisd_overlap(mf0, ci0, mol0, mf_minus, ci_minus, mol_minus)
    s_plus = cisd_overlap(mf0, ci0, mol0, mf_plus, ci_plus, mol_plus)
    laplace = (s_plus + s_minus - 2.0) / dR**2

    if verbose:
        axis_label = "xyz"[cart_index]
        print(
            f"    atom {atom_index} axis {axis_label}: "
            f"S(-dR) = {s_minus:.12f}  S(+dR) = {s_plus:.12f}  "
            f"S(-dR)+S(+dR)-2 = {s_plus + s_minus - 2.0: .3e}  "
            f"d2S/dR2 = {laplace: .6e}"
        )

    return laplace


def compute_dboc(
    R: float,
    dR: float,
    cavity: CavityConfig,
    exploit_linear_symmetry: bool = True,
    verbose: bool = True,
):
    """
    Evaluate the DBOC and the QED-CI(2,0) total energy for LiH at bond
    length R (Angstrom).

    Parameters
    ----------
    exploit_linear_symmetry : bool
        For a linear molecule aligned along z, the x and y Laplacian
        components are related by symmetry, so only x needs to be
        computed explicitly and doubled. Set to False to compute all
        three Cartesian directions independently (useful as a
        consistency check, at roughly 1.5x the cost).
    verbose : bool
        Print the wavefunction overlaps entering the finite-difference
        Laplacian for each atom and Cartesian direction.
    """
    mol0 = build_lih(R)
    mf0, ci0 = run_qedhf_cisd(mol0, cavity)

    masses = {0: ATOMIC_MASS["Li"], 1: ATOMIC_MASS["H"]}
    laplacian_sum = 0.0

    for atom_index, mass in masses.items():
        lap_xx = laplacian_component(R, atom_index, 0, dR, cavity, mf0, ci0, mol0, verbose)
        lap_zz = laplacian_component(R, atom_index, 2, dR, cavity, mf0, ci0, mol0, verbose)

        if exploit_linear_symmetry:
            lap_atom = 2.0 * lap_xx + lap_zz
        else:
            lap_yy = laplacian_component(R, atom_index, 1, dR, cavity, mf0, ci0, mol0, verbose)
            lap_atom = lap_xx + lap_yy + lap_zz

        laplacian_sum += lap_atom / mass

    dboc = -0.5 * BOHR_TO_ANGSTROM**2 * laplacian_sum
    return dboc, ci0.e_tot


# --------------------------------------------------------------------- #
# Example: single-point DBOC for LiH, with and without cavity coupling
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    R_EQ = 1.5957     # LiH equilibrium bond length, Angstrom
    DR = 1.0e-4       # finite-difference displacement, Angstrom

    print("Cavity-free (lambda = 0):")
    cavity_free = CavityConfig(omega=0.5, coupling=0.0)
    dboc_free, e_free = compute_dboc(R_EQ, DR, cavity_free)
    print(f"  QED-CI(2,0) energy : {e_free: .10f} Ha")
    print(f"  DBOC               : {dboc_free: .10e} Ha  ({dboc_free * HARTREE_TO_CM1: .4f} cm^-1)")
    print(f"  PEC + DBOC         : {e_free + dboc_free: .10f} Ha")

    print(f"\nWith cavity coupling (lambda = 0.05 a.u.):")
    cavity_on = CavityConfig(omega=0.5, coupling=0.05)
    dboc_on, e_on = compute_dboc(R_EQ, DR, cavity_on)
    print(f"  QED-CI(2,0) energy : {e_on: .10f} Ha")
    print(f"  DBOC               : {dboc_on: .10e} Ha  ({dboc_on * HARTREE_TO_CM1: .4f} cm^-1)")
    print(f"  PEC + DBOC         : {e_on + dboc_on: .10f} Ha")

    shift_cm1 = abs(dboc_free - dboc_on) * HARTREE_TO_CM1
    print(f"\nCavity-induced shift in DBOC: {shift_cm1: .4f} cm^-1")

    cfour_dboc_cm1 = 196.570455
    print(f"\nComparison with the cavity-free CFOUR CCSD-DBOC reference below:")
    print(f"  CFOUR (CCSD)                      : {cfour_dboc_cm1: .4f} cm^-1")


# --------------------------------------------------------------------- #
# Reference calculation (no cavity): CFOUR CCSD/cc-pVDZ, analytic DBOC
# --------------------------------------------------------------------- #
#
# Input file (LiH, cc-pVDZ, RHF reference, CCSD, analytic relaxed DBOC):
#
#   LiH
#   Li 0.000   0.000   0.000
#   H  0.000   0.000   1.5957
#
#   *CFOUR(COORD=CARTESIAN,BASIS=PVDZ
#   REFERENCE=RHF,UNITS=ANGSTROM
#   CALC=CCSD
#   DIFF_TYPE=RELAXED
#   ABCDTYPE=AOBASIS
#   SPHERICAL=ON,CC_MAXCYC=200
#   CHARGE=0,MULTIPLICITY=1
#   DBOC=ON
#   SYM=OFF
#   SCF_CONV=10,SCF_MAXCYC=200
#   CC_CONV=10,LINEQ_CONV=10,MEM_UNIT=GB,MEMORY_SIZE=8)
#
# Relevant output:
#
#   The total diagonal Born-Oppenheimer correction (DBOC) is:  196.570455 cm-1
#   The total diagonal Born-Oppenheimer correction (DBOC) is:    2.352 kJ/mole
#   The final electronic energy is        -8.013824766683138 a.u.
#
# This CCSD/cc-pVDZ value is the natural cavity-free (lambda = 0) benchmark
# for the QED-CI(2,0) result computed above. Note that CFOUR's DBOC is
# obtained from an analytic CCSD relaxed-density treatment, whereas the
# present script uses CISD built on a QED-HF reference and a numerical
# finite-difference Laplacian, so exact agreement is not expected.
# Note: default masses in CFOUR may differ.
