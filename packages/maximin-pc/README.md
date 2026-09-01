# maximin-pc

`maximin-pc` places `k` well-separated points in a periodic simulation cell while enforcing clearance from an obstacle selection. It is designed for crowded systems under periodic boundary conditions (PBC), including the placement of restrained co-alchemical ions in protein-crystal solvent channels.

The program supports:

- a fast sampled search, recommended for most applications;
- a deterministic fractional-grid search for structured or reference calculations;
- optional output structures containing dummy `CLC` atoms at the selected positions.

## Input requirement

The obstacle structure must contain the full contents of the primary unit cell. Expand crystallographic asymmetric-unit coordinates before use; otherwise, symmetry-related obstacles will be missing and the selected positions may be invalid. Cell expansion must be performed with an external crystallographic tool such as Gemmi.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
maximin-pc --help
```

For development:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Usage

Sampled search:

```bash
maximin-pc -t input.gro -c input.gro -k 4 -m 15 --no-h -n 80000 \
  --seed 12345 -o sampled_out.gro
```

Sampled search balancing point separation and obstacle clearance:

```bash
maximin-pc -t input.gro -c input.gro -k 4 -m 15 --no-h -n 80000 \
  --seed 12345 --joint-min-distance -o sampled_joint_out.gro
```

Deterministic thresholded grid search:

```bash
maximin-pc -t input.gro -c input.gro -k 4 -m 15 --no-h \
  --grid 40 40 40 --thresholded-maximin --max-dist 20 -o grid_out.gro
```

Distances are specified in Angstrom. The default obstacle selection is `protein`. Use `-s` for another MDAnalysis selection and record `--seed` when using sampled mode. Run `maximin-pc --help` for all options.

## Search modes

Sampled mode generates low-discrepancy candidates, filters them by obstacle clearance, maximizes the minimum point-to-point distance, and locally refines the result. `--joint-min-distance` instead maximizes the shared bottleneck between point-to-point distance and obstacle clearance.

Deterministic mode is enabled by `--grid NU NV NW`. With `--thresholded-maximin`, pair separations strictly above `--max-dist` are treated as equivalent. Grid results can depend strongly on the chosen resolution, and exhaustive searches may be slow; use this mode deliberately for structured or reproducibility workflows.

The program reports the selected coordinates, optimization score, and minimum obstacle clearance. Supplying `-o` writes the selected positions as dummy atoms.

The system-specific calculations from the associated study can be reproduced
from the TPR files and analysis script deposited on Zenodo.

## Citation and license

If you use this software, please cite the associated data and analysis record:
<https://doi.org/10.5281/zenodo.21687546>.

This software is licensed under the MIT License; see `LICENSE`.
