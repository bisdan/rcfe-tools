#!/usr/bin/env python3
"""Find special positions in a periodic MDAnalysis Universe.

Modes
-----
sampled (default)
    Generate quasi-random candidate points in the unit cell, filter them by
    obstacle clearance, and then maximize either the minimum pairwise distance
    or, with --joint-min-distance, the shared bottleneck between pairwise
    distance and obstacle clearance under periodic boundary conditions.

deterministic (enabled automatically by --grid NU NV NW)
    Build a regular fractional grid in the unit cell, filter grid points by a
    minimum obstacle-distance threshold, and then maximize the minimum pairwise
    distance between the selected positions under periodic boundary conditions.

evaluate_only (enabled by --evaluate-only)
    Read existing special-point atoms with atom name and residue name matching
    --point-label from the chosen frame and report their periodic pair
    distances and obstacle clearances without performing a search.

Thresholded maximin mode
------------------------
If --thresholded-maximin is enabled, pairwise distances larger than
--max-dist are saturated to a common large sentinel value (1000.0)
during selection, so distances above the cutoff are not distinguished further.

With a deterministic grid, an exhaustive subset search is performed when the
number of combinations is within the configured computational limit.
"""

from __future__ import annotations

from .cli import parse_args
from . import HydrogenMaskError


def main() -> int:
    args = parse_args()
    from .runner import run_workflow

    return run_workflow(args)


def cli_main() -> int:
    try:
        return main()
    except HydrogenMaskError as err:
        raise SystemExit(str(err))


if __name__ == "__main__":
    raise SystemExit(cli_main())
