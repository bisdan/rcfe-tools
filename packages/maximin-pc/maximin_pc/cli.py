from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find or evaluate special positions in the simulation cell under "
            "periodic boundary conditions."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-t",
        "--tpr",
        required=True,
        help="Input topology in any format supported by MDAnalysis",
    )
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        help=(
            "Optional input structure/trajectory in any format supported by "
            "MDAnalysis. If omitted, -t must itself contain coordinates and "
            "unit-cell information, for example a .gro file."
        ),
    )
    parser.add_argument(
        "-k",
        "--n-points",
        type=int,
        default=None,
        help=(
            "Number of special positions to place. In --evaluate-only mode, "
            "this may be omitted and will then be inferred from the number of "
            "input special-point atoms matching --point-label."
        ),
    )

    parser.add_argument(
        "-s",
        "--obstacle-selection",
        dest="obstacle_selection",
        default="protein",
        help="MDAnalysis selection string for the obstacle atoms (default: protein)",
    )
    parser.add_argument(
        "--exclude-hydrogens",
        "-H",
        "--no-h",
        action="store_true",
        help="Exclude hydrogen atoms from the obstacle set; also omit them from output.",
    )
    parser.add_argument(
        "--ignore-waters",
        "-W",
        "--no-water",
        action="store_true",
        help="Ignore water atoms in the search/evaluation and omit them from output.",
    )
    parser.add_argument(
        "--ignore-selection",
        "-S",
        "--exclude-selection",
        dest="ignore_selection",
        default=None,
        help=(
            "Optional MDAnalysis selection string to exclude from the obstacle set "
            "and omit from output."
        ),
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to analyze (default: 0)",
    )
    parser.add_argument(
        "--evaluate-only",
        "-e",
        action="store_true",
        help=(
            "Do not search for new positions. Instead, evaluate the existing "
            "special-point atoms in the input structure/trajectory frame whose "
            "atom name and residue name both match --point-label. If "
            "--n-points is provided, it must match the number found in the "
            "input."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "Print additional per-point details for the final configuration, "
            "including distances to all other special-point atoms and the "
            "nearest obstacle atom."
        ),
    )
    parser.add_argument(
        "--pym",
        "-p",
        default=None,
        help=(
            "Evaluation-only: write a PyMOL script that loads the evaluated "
            ".gro file, draws transparent clearance spheres around the "
            "special-point atoms, adds an optional camera-facing circle overlay, "
            "and registers PyMOL commands for adjusting overlay colors and sphere "
            "transparency after loading."
        ),
    )
    parser.add_argument(
        "--point-label",
        "-l",
        default="CLC",
        help=(
            "Atom name and residue name used for the special-point markers in "
            "written output and in --evaluate-only mode (default: CLC). For "
            "broad structure-writer compatibility this must be 1-4 "
            "alphanumeric characters."
        ),
    )

    parser.add_argument(
        "-n",
        "--n-candidates",
        type=int,
        default=20000,
        help="Number of candidate points for sampled mode (default: 20000)",
    )
    parser.add_argument(
        "--grid",
        "-g",
        nargs=3,
        type=int,
        metavar=("NU", "NV", "NW"),
        default=None,
        help=(
            "Use a deterministic regular grid instead of sampled candidate generation. "
            "Example: --grid 40 40 40"
        ),
    )

    parser.add_argument(
        "--max-survivors",
        type=int,
        default=50000,
        help=(
            "Maximum number of filtered candidate points used in the approximate "
            "combinatorial optimization step. If more candidates survive the "
            "obstacle-distance filter, the highest-clearance candidates are retained "
            "as a computational cap (default: 50000)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for sampled mode (default: 12345)",
    )
    parser.add_argument(
        "--binary-steps",
        type=int,
        default=24,
        help="Binary search iterations for the sampled optimization step (default: 24)",
    )
    parser.add_argument(
        "--joint-min-distance",
        "-j",
        dest="joint_min_distance",
        action="store_true",
        help=(
            "In sampled mode, optimize the shared bottleneck distance, i.e. "
            "min(min point-point distance, minimum obstacle clearance). In "
            "evaluate-only mode, report that same joint score for the input "
            "special-point configuration."
        ),
    )

    parser.add_argument(
        "-m",
        "--min-dist",
        dest="min_obstacle_distance",
        type=float,
        default=10.0,
        help=(
            "Minimum allowed obstacle clearance for candidate points in Angstrom. "
            "Candidates with clearance < min_obstacle_distance are discarded "
            "(default: 10.0 A = 1.0 nm)."
        ),
    )
    parser.add_argument(
        "-M",
        "--max-dist",
        dest="max_obstacle_distance",
        type=float,
        default=None,
        help=(
            "Maximum obstacle-distance cutoff. In sampled mode, candidates with "
            "clearance > max_obstacle_distance are discarded. In deterministic "
            "grid mode, this is not used as a hard validity filter; with "
            "--thresholded-maximin it instead defines the pair-score threshold."
        ),
    )
    parser.add_argument(
        "--min-charged-dist",
        "--m-charged",
        "-mq",
        dest="cavity_lower_cutoff_charged",
        type=float,
        default=None,
        help=(
            "Minimum required clearance to charged obstacle residues in deterministic "
            "grid mode. Charged residues are currently defined as resname ARG, LYS, "
            "ASP, or GLU. If omitted, no separate charged cutoff is applied."
        ),
    )

    parser.add_argument(
        "--thresholded-maximin",
        "--thresholded",
        dest="thresholded_mode",
        action="store_true",
        help=(
            "Enable the thresholded maximin objective. Pair distances larger than "
            "--max-dist are saturated to a common large value during "
            "selection and are therefore treated as equivalent."
        ),
    )
    parser.add_argument(
        "--legacy",
        dest="thresholded_mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--maxcomb",
        dest="thresholded_exact_max_combinations",
        type=int,
        default=3000000,
        help=(
            "With deterministic grid + thresholded maximin, perform an exhaustive "
            "subset search when n_survivors choose k does not exceed this value "
            "(default: 3000000)."
        ),
    )
    parser.add_argument(
        "--thresholded-exact-chunk-size",
        type=int,
        default=200000,
        help=(
            "Chunk size for the exhaustive thresholded maximin subset search. Larger values can be "
            "faster but use more memory (default: 200000)."
        ),
    )
    parser.add_argument(
        "--legacy-exact-chunk-size",
        dest="thresholded_exact_chunk_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=True,
        help="Disable the progress bar for the exhaustive thresholded maximin subset search.",
    )
    parser.add_argument(
        "--print-ties",
        action="store_true",
        help=(
            "Print all solutions that are indistinguishable under the thresholded "
            "maximin score. This is supported when the exhaustive deterministic "
            "grid search is used."
        ),
    )
    parser.add_argument(
        "--max-tied",
        dest="max_printed_tied_solutions",
        type=int,
        default=10000,
        help=(
            "Maximum number of tied solutions to print when --print-ties "
            "is enabled. Use 0 for unlimited (default: 10000)."
        ),
    )
    parser.add_argument(
        "--pick",
        dest="pick_tied_solution",
        type=int,
        default=1,
        help=(
            "1-based index of the tied optimal solution to select during exhaustive "
            "thresholded maximin grid search (default: 1). The selected tied solution "
            "is the one printed and written to --output."
        ),
    )

    parser.add_argument(
        "-o",
        dest="output_path",
        default=None,
        help=(
            "Optional output structure file to write with dummy atoms at the selected positions. "
            "The writer format is inferred from the filename extension and may be any format "
            "supported by MDAnalysis. "
            "Dummy atom name/resname default to CLC and can be changed with --point-label. "
            "Atoms excluded from the obstacle set are also omitted from the written output."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.n_points is not None and args.n_points < 1:
        parser.error("--n-points must be >= 1")

    if not args.evaluate_only and args.n_points is None:
        parser.error("--n-points is required unless --evaluate-only is used")

    if args.n_points is not None and args.max_survivors < args.n_points:
        parser.error("--max-survivors must be >= --n-points")

    if args.n_points is not None and args.grid is None and args.n_candidates < args.n_points:
        parser.error("--n-candidates must be >= --n-points")

    if args.conf is None and args.frame != 0:
        parser.error("--frame must be 0 when -c/--conf is omitted")

    if (
        args.min_obstacle_distance is not None
        and args.max_obstacle_distance is not None
        and args.min_obstacle_distance > args.max_obstacle_distance
    ):
        parser.error("--min-dist cannot be larger than --max-dist")

    if args.min_obstacle_distance is not None and args.min_obstacle_distance < 0.0:
        parser.error("--min-dist must be >= 0")

    if args.cavity_lower_cutoff_charged is not None and args.cavity_lower_cutoff_charged < 0.0:
        parser.error("--min-charged-dist must be >= 0")

    if args.grid is not None and any(n <= 0 for n in args.grid):
        parser.error("--grid values must all be positive integers")

    if args.thresholded_mode and args.max_obstacle_distance is None:
        parser.error("--thresholded-maximin requires --max-dist")

    if args.joint_min_distance and args.grid is not None:
        parser.error("--joint-min-distance is only supported in sampled mode")

    if args.joint_min_distance and args.thresholded_mode:
        parser.error("--joint-min-distance cannot be combined with --thresholded-maximin")

    if args.thresholded_exact_max_combinations < 1:
        parser.error("--maxcomb must be >= 1")

    if args.thresholded_exact_chunk_size < 1:
        parser.error("--thresholded-exact-chunk-size must be >= 1")

    if args.max_printed_tied_solutions < 0:
        parser.error("--max-tied must be >= 0")

    if args.pick_tied_solution < 1:
        parser.error("--pick must be >= 1")

    if args.evaluate_only and args.print_ties:
        parser.error("--print-ties is not supported with --evaluate-only")

    if args.evaluate_only and args.pick_tied_solution != 1:
        parser.error("--pick is not supported with --evaluate-only")

    if args.evaluate_only and args.output_path is not None:
        parser.error("-o is not supported with --evaluate-only")

    if args.pym is not None and not args.evaluate_only:
        parser.error("--pym is only supported with --evaluate-only")

    if args.pym is not None:
        pym_source = args.conf if args.conf is not None else args.tpr
        if not pym_source.lower().endswith(".gro"):
            parser.error("--pym currently requires the evaluated structure file to be a .gro file")

    if not args.point_label:
        parser.error("--point-label must not be empty")

    if len(args.point_label) > 4:
        parser.error("--point-label must be at most 4 characters long")

    if not args.point_label.isascii() or not args.point_label.isalnum():
        parser.error("--point-label must contain only ASCII letters and digits")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args
