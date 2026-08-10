# DBOC in Cavity QED — Example (LiH)

Example code for evaluating the diagonal Born-Oppenheimer correction (DBOC)
for a molecule embedded in a single-mode optical cavity, at the QED-CI(2,0)
level (CISD built on a QED-HF reference). Accompanies:

> T. Zalialiutdinov, D. Solovyev, J. J. Lopez-Rodriguez, A. Anikin, and
> A. Kotov, *Non-adiabatic Effects Induced by Strong Light-Matter Coupling
> in Cavity QED* (2026).

## What it does

`dboc_qedhf_example.py` computes the DBOC for LiH at a single bond
length (cc-pVDZ basis, chosen for speed), both without a cavity and with a
finite light-matter coupling strength, and reports the cavity-induced
shift in cm⁻¹. The cavity-free result is compared against an independent
CCSD/cc-pVDZ analytic DBOC calculation from CFOUR (input and output
reproduced as a comment at the end of the script).

The DBOC is evaluated fully numerically from CISD wavefunction overlaps
at displaced nuclear geometries, following the finite-difference scheme
of Valeev and Sherrill, *J. Chem. Phys.* **118**, 3921 (2003), generalized
here to the cavity QED case.

## Dependencies

- [PySCF](https://pyscf.org)
- [OpenMS](https://github.com/lanl/OpenMS) (provides `openms.mqed.qedhf`)

## Usage

```bash
python dboc_qedhf_example.py
```

No command-line arguments. Molecule, basis set, bond length, cavity
frequency/coupling, and finite-difference step are set as constants near
the top of the script and in the `if __name__ == "__main__":` block.

## Output

For each case (cavity-free and with coupling) the script prints:

- the QED-CI(2,0) total energy,
- the DBOC in Hartree and cm⁻¹,
- the total (electronic + DBOC) energy,
- per-atom, per-axis wavefunction overlaps entering the finite-difference
  Laplacian, for inspection/debugging,

followed by the cavity-induced shift in the DBOC and a comparison of the
cavity-free result against the CFOUR reference value.

## Notes on the numerics

- `exploit_linear_symmetry=True` (default) uses the fact that for a
  linear molecule aligned along z, the x and y Laplacian components are
  equal by symmetry, so only x is computed explicitly and doubled. Set
  to `False` to compute all three Cartesian directions independently as
  a consistency check.
- Nuclear masses use standard atomic weights (natural isotopic
  abundance). Replace with pure-isotope masses in `ATOMIC_MASS` if
  isotope-resolved DBOC values are needed.
- Exact numerical agreement with the CFOUR reference is not expected: the
  reference is an analytic CCSD relaxed-density DBOC, while this script
  uses CISD on a QED-HF reference and a numerical finite-difference
  Laplacian. 

