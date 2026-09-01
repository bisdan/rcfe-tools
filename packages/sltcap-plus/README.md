# SLTCAP+

SLTCAP+ estimates cation and anion counts that combine a target bulk salt
concentration with global electroneutrality in a finite simulation volume. The
method extends the original mean-field SLTCAP construction to fully dissociated
binary salts of arbitrary valence. The command also implements the
add-then-neutralize (AN) heuristic.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
sltcap-plus --help
```

## Definition

For stoichiometric coefficients $\nu_+$ and $\nu_-$, signed valences
$q_+>0$ and $q_-<0$, formula-unit concentration $c^0$, solvent volume $V$,
Avogadro constant $N_\mathrm{A}$, and solute charge $Q$, define the dimensionless
bulk formula-unit count $N_0=c^0VN_\mathrm{A}$. SLTCAP+ solves

$$
\rho=t^{q_+}-t^{q_-}, \qquad
\rho=-\frac{Q}{N_0\nu_+q_+},
$$

and calculates $N_\pm=\nu_\pm N_0t^{q_\pm}$. Charges are in elementary-charge
units. The command accepts concentration in mM salt formula units and solvent
volume in litres, applying the required unit conversion when calculating
$N_0$. The stoichiometry must be neutral: $\nu_+q_+ + \nu_-q_-=0$.

Specify the solvent by either volume in litres (`--solvent-volume`) or number
of water molecules (`--num-water-molecules`, using 55.5 M water).

## Usage

NaCl at 150 mM with 5,550 water molecules and solute charge -10:

```bash
sltcap-plus sltcap -Q -10 -W 5550 -c 150 --solver analytic
```

MgCl2 at 100 mM under the same solvent definition:

```bash
sltcap-plus sltcap -Q -10 -W 5550 -c 100 \
  -np 1 -nn 2 -qp 2 -qn -1 --solver analytic
```

Use the `an` subcommand for add-then-neutralize and `--output-format json` for
machine-readable output. `--solver numerical` supports arbitrary valid binary
stoichiometries. `auto` uses a supported analytical form when stable, while
`analytic` requires one. Reported integer counts minimize distance from the
continuous solution subject to exact neutrality whenever the valences permit
it.

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests cover the defining equations, analytical/numerical agreement, solvent
conversion, integer neutrality, AN behavior, and command-line interfaces.

## Citation and license

SLTCAP was introduced by Schmit, J. D., Kariyawasam, N. L., Needham, V. &
Smith, P. E. *SLTCAP: A Simple Method for Calculating the Number of Ions Needed
for MD Simulation.* Journal of Chemical Theory and Computation **14**,
1823--1827 (2018). [doi:10.1021/acs.jctc.7b01254](https://doi.org/10.1021/acs.jctc.7b01254).

If you use this software, please cite the associated data and analysis record:
<https://doi.org/10.5281/zenodo.21687546>.

This software is licensed under the MIT License; see `LICENSE`.
