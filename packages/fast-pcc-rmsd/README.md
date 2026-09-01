# fast-pcc-rmsd

`fast-pcc-rmsd` calculates the RMSD of individual pairwise crystal contacts
(PCCs) in protein-crystal molecular dynamics trajectories. The installed
`fpcc` command uses MDAnalysis to identify or load residue-pair contacts, map
them to the nearest periodic image, and compare every trajectory frame with a
reference crystal structure.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
fpcc --help
```

For development:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

For NVIDIA CUDA acceleration, install one matching optional dependency:

```bash
python -m pip install '.[cuda12]'  # CUDA 12
python -m pip install '.[cuda13]'  # CUDA 13
```

Do not install both CuPy variants in the same environment. CUDA execution also
requires a compatible NVIDIA GPU and driver.

## Usage

```bash
fpcc \
  --topology system.tpr \
  --trajectory trajectory.xtc \
  --reference reference.gro \
  --contact-index contacts.ndx \
  --transfer-topology-to-reference \
  --output contact_rmsd.xvg
```

`--transfer-topology-to-reference` is useful for coordinate-only reference
formats such as GRO. After confirming that the primary system and reference
contain the same atoms in the same order, `fpcc` copies masses and bonds from
the topology.

The short forms `-s`, `-f`, `-r`, and `-o` are equivalent to the corresponding
long options. The output option is required. Run `fpcc --help` for all options.

## Inputs and constraints

- The topology must explicitly provide atom masses and bonds. Attributes
  guessed by MDAnalysis are not accepted.
- The trajectory must provide instantaneous unit-cell parameters for every
  analyzed frame.
- The reference must provide coordinates and unit-cell parameters. If
  `--reference` is omitted, `fpcc` tries to use the topology as the reference.
- A separate reference must provide masses and bonds unless
  `--transfer-topology-to-reference` is used, and it must be topology-equivalent
  to the primary system.
- Analysis uses the MDAnalysis `protein` selection by default. If the prepared
  system has multiple chains or segments, they must have identical amino-acid
  sequences.

Only the first frame of a multi-frame reference is used.

## Contact selection

Without `--contact-index`, contacts are detected automatically from the
unwrapped reference:

- only inter-chain residue pairs are considered;
- the default atom-distance cutoff is `0.6 nm`;
- hydrogens are included in the search by default;
- biological and crystal interfaces are not distinguished.

Change the cutoff with `--contact-cutoff-nm`, or omit hydrogens from automatic
detection with `--exclude-hydrogens-from-contact-search`.

To supply contacts explicitly, pass `--contact-index contacts.ndx`. Only groups
whose names start with `pcc_` are used. Valid group headers include:

```text
[ pcc_A42_B107 ]
[ pcc_ASP42_LYS107 ]
[ pcc_contact_001 ]
```

The lowercase `pcc_` prefix is case-sensitive. The suffix is only a descriptive
label; the two residues are inferred from the group's atom indices, not its
name. Each group must map to exactly two residues. Inter-chain pairs are
accepted directly. Same-chain periodic-image pairs are accepted when their
reference residue center-of-mass distance exceeds
`--indexed-intra-chain-com-cutoff-nm` (default `1.2 nm`).

A named group from the same index can replace the default protein selection
through `--system-selection-index-group NAME`.

Hydrogens are excluded from the RMSD atom subsets by default, independently of
contact detection. Use `--include-hydrogens-in-rmsd` to retain them.

## Analysis

`fpcc` unwraps the reference protein before preparing contacts. For each
contact, it keeps the first residue fixed and translates the second by unit-cell
vectors to minimize their center-of-mass distance. During trajectory analysis,
the contact residues are made whole and the same nearest-image mapping is
applied on every frame.

RMSD is calculated directly from Cartesian displacements against the prepared
reference coordinates; no rotational or translational fitting is performed.
Reported RMSD values are in `nm`.

## Outputs

The `--output` path selects one of two layouts:

- A path ending in `.xvg` produces one table containing the analyzed frame in
  the first column and one RMSD series per contact in the remaining columns.
- Any other path is treated as a directory. It receives one prepared `.gro`
  reference and one RMSD `.xvg` file per contact.

Directory mode can additionally write one remapped `.xtc` trajectory per
contact when `--write-remapped-contact-trajectories` is enabled.

## Acceleration

CPU analysis with optimized residue unwrapping is the default. After installing
a CUDA extra, GPU acceleration must be selected explicitly:

```bash
fpcc ... --compute-backend cuda
```

CUDA accelerates contact remapping, RMSD calculation, and optimized residue
unwrapping. It is never selected automatically and does not silently fall back
to CPU if setup fails.

For compatibility comparisons, select MDAnalysis residue unwrapping with
`--unwrap-backend mdanalysis`. Use `--no-progress-bar` for non-interactive runs
and `--verbose` for a run summary.

## Citation and license

If you use this software, please cite the associated data and analysis record:
<https://doi.org/10.5281/zenodo.21687546>.

This software is licensed under the MIT License; see `LICENSE`.
