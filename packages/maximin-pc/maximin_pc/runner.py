from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError

from .core import (
    CHARGED_RESNAMES,
    THRESHOLDED_SENTINEL,
    SearchSetup,
    Solution,
    apply_deterministic_min_distance_filter,
    build_solution_from_indices,
    build_search_setup,
    charged_obstacle_mask,
    compute_clearance,
    exact_thresholded_subset_search,
    filter_sampled_candidates_by_window,
    generate_grid_candidates,
    generate_random_candidates,
    get_cell_matrix,
    n_choose_k,
    print_solution,
    print_tied_solutions,
    print_verbose_solution,
    refine_sampled_solution,
    solve_joint_min_distance,
    solve_max_pair_distance,
    wrap_points,
    write_pymol_sphere_script,
    write_output_structure,
)


@dataclass
class ExecutionContext:
    universe: mda.Universe
    setup: SearchSetup
    box: np.ndarray
    cell: np.ndarray
    inv_cell: np.ndarray
    structure_path: Path


@dataclass
class CandidatePool:
    candidates: np.ndarray
    clearance: np.ndarray


@dataclass
class OptimizationSummary:
    solution: Solution
    replaced_atom_indices: Set[int] = field(default_factory=set)
    used_exact_thresholded: bool = False
    was_capped: bool = False
    capped_to: int | None = None
    tied_combinations: List[Tuple[int, ...]] = field(default_factory=list)
    total_ties: int = 0
    report_obstacles: mda.AtomGroup | None = None


def find_point_atoms(universe: mda.Universe, point_label: str) -> mda.AtomGroup:
    try:
        atom_names = np.asarray(universe.atoms.names, dtype=object)
        residue_names = np.asarray(universe.atoms.resnames, dtype=object)
    except NoDataError as err:
        raise SystemExit(
            "Point-label handling requires atom names and residue names in the input topology."
        ) from err

    mask = (atom_names == point_label) & (residue_names == point_label)
    return universe.atoms[np.asarray(mask, dtype=bool)]


def run_workflow(args: argparse.Namespace) -> int:
    context = load_execution_context(args)
    point_atoms = find_point_atoms(context.universe, args.point_label)
    mode_name = determine_mode(args)

    print(f"Mode: {mode_name}")
    print(f"Selected obstacle atoms: {len(context.setup.obstacle)}")
    if args.evaluate_only:
        print(
            "Evaluating existing special-point atoms from the input frame with "
            f"name/resname {args.point_label!r}."
        )
    elif len(point_atoms) > 0:
        raise SystemExit(
            "The input already contains "
            f"{len(point_atoms)} atom(s) with atom name and residue name "
            f"{args.point_label!r}. This risks reusing or duplicating previously "
            "written marker atoms during a search. Choose a different --point-label, "
            "remove those atoms, or use --evaluate-only."
        )
    if args.joint_min_distance:
        objective_label = "Evaluation objective" if args.evaluate_only else "Sampled objective"
        print(
            f"{objective_label}: maximize min(min point-point distance, "
            "minimum obstacle clearance)."
        )

    candidate_pool = None
    if args.evaluate_only:
        optimization = evaluate_existing_configuration(args, context, point_atoms)
    else:
        candidate_pool = build_candidate_pool(args, context)
        ensure_candidate_count(args, candidate_pool)
        optimization = optimize_candidates(args, context, candidate_pool)
    report_optimization(args, candidate_pool, optimization)
    maybe_print_tied_solutions(args, candidate_pool, optimization, context.box)
    maybe_write_output_structure(
        args,
        context,
        optimization.solution,
        optimization.replaced_atom_indices,
    )
    maybe_print_verbose_configuration(args, context, optimization)
    maybe_write_pymol_script(args, context, optimization)
    return 0


def load_execution_context(args: argparse.Namespace) -> ExecutionContext:
    structure_path = Path(args.conf if args.conf is not None else args.tpr).expanduser().resolve()
    try:
        if args.conf is None:
            universe = mda.Universe(args.tpr)
        else:
            universe = mda.Universe(args.tpr, args.conf)
    except Exception as err:
        if args.conf is None:
            raise SystemExit(
                "Failed to load coordinates from -t alone. When -c/--conf is omitted, "
                "-t must contain coordinates and unit-cell information."
            ) from err
        raise

    n_frames = len(universe.trajectory)
    if args.frame < 0 or args.frame >= n_frames:
        frame_source = args.conf if args.conf is not None else args.tpr
        raise SystemExit(
            f"--frame {args.frame} is out of range for {frame_source} with {n_frames} frame(s)"
        )
    universe.trajectory[args.frame]

    setup = build_search_setup(
        u=universe,
        selection=args.obstacle_selection,
        exclude_hydrogens=args.exclude_hydrogens,
        ignore_waters=args.ignore_waters,
        ignore_selection=args.ignore_selection,
    )

    box = universe.trajectory.ts.dimensions.copy()
    cell = get_cell_matrix(universe.trajectory.ts)
    inv_cell = np.linalg.inv(cell)
    return ExecutionContext(
        universe=universe,
        setup=setup,
        box=box,
        cell=cell,
        inv_cell=inv_cell,
        structure_path=structure_path,
    )


def determine_mode(args: argparse.Namespace) -> str:
    if args.evaluate_only:
        return "evaluate_only"
    if args.grid is not None:
        return "deterministic_grid"
    return "sampled"


def evaluate_existing_configuration(
    args: argparse.Namespace,
    context: ExecutionContext,
    point_atoms: mda.AtomGroup,
) -> OptimizationSummary:
    if len(point_atoms) == 0:
        raise SystemExit(
            "Evaluate-only mode requires existing atoms whose atom name and residue "
            f"name both equal {args.point_label!r}."
        )
    if args.n_points is None:
        args.n_points = len(point_atoms)
        print(f"Inferred number of special-point atoms from input: {args.n_points}")
    elif len(point_atoms) != args.n_points:
        raise SystemExit(
            "Evaluate-only mode found "
            f"{len(point_atoms)} atom(s) with name/resname {args.point_label!r}, "
            f"but --n-points={args.n_points}."
        )

    replaced_atom_indices = set(int(idx) for idx in point_atoms.indices.tolist())
    overlap_mask = np.isin(context.setup.obstacle.indices, point_atoms.indices)
    report_obstacles = context.setup.obstacle
    if np.any(overlap_mask):
        report_obstacles = context.setup.obstacle[~overlap_mask]
        if len(report_obstacles) == 0:
            raise SystemExit(
                "Evaluate-only mode removed all obstacle atoms after excluding the evaluated "
                f"{args.point_label!r} atoms from the obstacle set."
            )
        print(
            "Evaluate-only mode note: excluded "
            f"{int(np.count_nonzero(overlap_mask))} evaluated {args.point_label!r} atom(s) "
            "from the obstacle set."
        )

    points = wrap_points(point_atoms.positions.copy(), context.cell, context.inv_cell)
    clearance = compute_clearance(points, report_obstacles.positions, context.box)
    solution = build_solution_from_indices(
        candidates=points,
        clearance=clearance,
        box=context.box,
        indices=range(len(points)),
        thresholded_mode=args.thresholded_mode,
        threshold_cutoff=args.max_obstacle_distance,
    )
    return OptimizationSummary(
        solution=solution,
        replaced_atom_indices=replaced_atom_indices,
        report_obstacles=report_obstacles,
    )


def build_candidate_pool(
    args: argparse.Namespace,
    context: ExecutionContext,
) -> CandidatePool:
    if args.grid is not None:
        nu, nv, nw = args.grid
        candidates = generate_grid_candidates(context.cell, nu, nv, nw)
        source_count = len(candidates)
        print(f"Grid dimensions: {nu} x {nv} x {nw}")
    else:
        rng = np.random.default_rng(args.seed)
        candidates = generate_random_candidates(context.cell, args.n_candidates, rng)
        source_count = args.n_candidates

    # Always wrap before distance evaluation so random/grid generation behaves the same
    # near periodic boundaries.
    wrapped_candidates = wrap_points(candidates, context.cell, context.inv_cell)

    if args.grid is not None:
        full_clearance = compute_clearance(
            wrapped_candidates,
            context.setup.obstacle.positions,
            context.box,
        )
        return filter_deterministic_candidates(
            args,
            context,
            wrapped_candidates,
            full_clearance,
            source_count,
        )

    return filter_sampled_candidates(
        args,
        wrapped_candidates,
        source_count,
        context.setup.obstacle.positions,
        context.box,
    )


def filter_deterministic_candidates(
    args: argparse.Namespace,
    context: ExecutionContext,
    candidates: np.ndarray,
    full_clearance: np.ndarray,
    source_count: int,
) -> CandidatePool:
    charged_mask = charged_obstacle_mask(context.setup.obstacle)
    if np.any(charged_mask):
        clearance_charged = compute_clearance(
            candidates,
            context.setup.obstacle.positions[charged_mask],
            context.box,
        )
    else:
        clearance_charged = None

    filtered_candidates, filtered_clearance = apply_deterministic_min_distance_filter(
        candidates=candidates,
        clearance_all=full_clearance,
        clearance_charged=clearance_charged,
        min_obstacle_distance=args.min_obstacle_distance,
        cavity_lower_cutoff_charged=args.cavity_lower_cutoff_charged,
    )

    print(f"Grid points before obstacle-distance filtering: {source_count}")
    if args.min_obstacle_distance is not None:
        print(
            "Applied minimum obstacle-distance threshold (A): clearance >= "
            f"{args.min_obstacle_distance:.6f}"
        )
    if args.cavity_lower_cutoff_charged is not None:
        if clearance_charged is None:
            print(
                "Applied charged-residue cavity cutoff (A): none present "
                f"(charged residues currently defined as {sorted(CHARGED_RESNAMES)})"
            )
        else:
            print(
                "Applied charged-residue cavity cutoff (A): "
                f"clearance_to_{{ARG,LYS,ASP,GLU}} >= {args.cavity_lower_cutoff_charged:.6f}"
            )
    if args.max_obstacle_distance is not None:
        print(
            "Deterministic mode note: --max-dist is not used as a hard "
            "validity filter in this mode."
        )
    if args.thresholded_mode:
        print(
            f"Thresholded maximin enabled: pair distances > {args.max_obstacle_distance:.6f} A "
            f"are saturated to {THRESHOLDED_SENTINEL:.1f} during selection."
        )
    print(f"Grid points after obstacle-distance filtering: {len(filtered_candidates)}")

    return CandidatePool(
        candidates=filtered_candidates,
        clearance=filtered_clearance,
    )


def filter_sampled_candidates(
    args: argparse.Namespace,
    candidates: np.ndarray,
    source_count: int,
    obstacle_positions: np.ndarray,
    box: np.ndarray,
) -> CandidatePool:
    filtered_candidates, filtered_clearance = filter_sampled_candidates_by_window(
        candidates=candidates,
        obstacle_positions=obstacle_positions,
        box=box,
        min_obstacle_distance=args.min_obstacle_distance,
        max_obstacle_distance=args.max_obstacle_distance,
    )

    if args.cavity_lower_cutoff_charged is not None:
        print(
            "Sampled mode note: --cavity-lower-cutoff-charged is ignored; "
            "it only applies to deterministic grid mode."
        )

    print(f"Candidate points before obstacle-distance filtering: {source_count}")
    if args.min_obstacle_distance is not None or args.max_obstacle_distance is not None:
        min_txt = "-inf" if args.min_obstacle_distance is None else f"{args.min_obstacle_distance:.6f}"
        max_txt = "inf" if args.max_obstacle_distance is None else f"{args.max_obstacle_distance:.6f}"
        print(f"Applied obstacle-distance window (A): {min_txt} <= clearance <= {max_txt}")
    if args.thresholded_mode:
        print(
            f"Thresholded maximin enabled: pair distances > {args.max_obstacle_distance:.6f} A "
            f"are saturated to {THRESHOLDED_SENTINEL:.1f} during selection."
        )
    print(f"Candidate points after obstacle-distance filtering: {len(filtered_candidates)}")

    return CandidatePool(
        candidates=filtered_candidates,
        clearance=filtered_clearance,
    )


def ensure_candidate_count(args: argparse.Namespace, candidate_pool: CandidatePool) -> None:
    if len(candidate_pool.candidates) < args.n_points:
        raise SystemExit(
            "Only "
            f"{len(candidate_pool.candidates)} candidate points remain after obstacle-distance "
            f"filtering, but --n-points={args.n_points}"
        )


def optimize_candidates(
    args: argparse.Namespace,
    context: ExecutionContext,
    candidate_pool: CandidatePool,
) -> OptimizationSummary:
    if args.grid is not None and args.thresholded_mode:
        # Exhaustively enumerate manageable subset spaces, then fall back to the
        # numerical feasibility solver when the subset space is too large.
        max_store = args.max_printed_tied_solutions if args.print_ties else 0
        try:
            solution, tied_combinations, total_ties = exact_thresholded_subset_search(
                candidates=candidate_pool.candidates,
                clearance=candidate_pool.clearance,
                box=context.box,
                k=args.n_points,
                threshold_cutoff=args.max_obstacle_distance,
                max_combinations=args.thresholded_exact_max_combinations,
                collect_ties=args.print_ties,
                max_stored_ties=max_store,
                chunk_size=args.thresholded_exact_chunk_size,
                show_progress=args.progress,
                selected_tie_index=args.pick_tied_solution,
            )
        except ValueError as err:
            raise SystemExit(str(err))

        if solution is not None:
            return OptimizationSummary(
                solution=solution,
                used_exact_thresholded=True,
                tied_combinations=tied_combinations,
                total_ties=total_ties,
            )

        if args.pick_tied_solution != 1:
            raise SystemExit(
                "Requested a specific tied solution, but exhaustive thresholded maximin "
                "search was not used. "
                "Increase --maxcomb or reduce the number of surviving "
                "grid points."
            )

    elif args.pick_tied_solution != 1:
        raise SystemExit(
            "--pick is only supported when exhaustive thresholded maximin grid "
            "search is used."
        )

    if args.joint_min_distance:
        solution, was_capped, capped_to = solve_joint_min_distance(
            candidates=candidate_pool.candidates,
            clearance=candidate_pool.clearance,
            cell=context.cell,
            box=context.box,
            k=args.n_points,
            max_survivors=args.max_survivors,
            binary_steps=args.binary_steps,
        )
    else:
        solution, was_capped, capped_to = solve_max_pair_distance(
            candidates=candidate_pool.candidates,
            clearance=candidate_pool.clearance,
            cell=context.cell,
            box=context.box,
            k=args.n_points,
            max_survivors=args.max_survivors,
            binary_steps=args.binary_steps,
            thresholded_mode=args.thresholded_mode,
            threshold_cutoff=args.max_obstacle_distance,
        )

    if args.grid is None and not args.thresholded_mode:
        solution = refine_sampled_solution(
            solution=solution,
            obstacle_positions=context.setup.obstacle.positions,
            box=context.box,
            cell=context.cell,
            inv_cell=context.inv_cell,
            min_obstacle_distance=args.min_obstacle_distance,
            max_obstacle_distance=args.max_obstacle_distance,
            seed=args.seed,
            joint_objective=args.joint_min_distance,
        )
    return OptimizationSummary(solution=solution, was_capped=was_capped, capped_to=capped_to)


def report_optimization(
    args: argparse.Namespace,
    candidate_pool: CandidatePool | None,
    optimization: OptimizationSummary,
) -> None:
    if optimization.used_exact_thresholded:
        if candidate_pool is None:
            raise RuntimeError(
                "Internal error: missing candidate pool for exhaustive thresholded reporting"
            )
        ncomb = n_choose_k(len(candidate_pool.candidates), args.n_points)
        print(f"Used exhaustive thresholded maximin search over {ncomb} combinations.")
        print(f"Selected tied solution index: {args.pick_tied_solution}")
        print(f"Total tied optimal solutions: {optimization.total_ties}")
    elif optimization.was_capped:
        if optimization.capped_to is None:
            raise RuntimeError("Internal error: missing cap size for capped optimization")
        if args.joint_min_distance:
            if optimization.capped_to < args.max_survivors:
                print(
                    "Optimization candidate set was capped to "
                    f"{optimization.capped_to} filtered candidates using a "
                    "joint-aware diversity preselection to keep the dense "
                    "sampled solver tractable."
                )
            else:
                print(
                    "Optimization candidate set was capped to "
                    f"{optimization.capped_to} filtered candidates using a "
                    "joint-aware diversity preselection."
                )
        else:
            if optimization.capped_to < args.max_survivors:
                print(
                    "Optimization candidate set was capped to "
                    f"{optimization.capped_to} highest-clearance filtered candidates "
                    "to keep the dense sampled solver tractable."
                )
            else:
                print(
                    "Optimization candidate set was capped to "
                    f"{optimization.capped_to} highest-clearance filtered candidates."
                )

    print("Selected positions (A):")
    print_solution(
        optimization.solution,
        thresholded_mode=args.thresholded_mode,
        joint_objective=args.joint_min_distance,
        evaluated_only=args.evaluate_only,
    )


def maybe_print_tied_solutions(
    args: argparse.Namespace,
    candidate_pool: CandidatePool | None,
    optimization: OptimizationSummary,
    box: np.ndarray,
) -> None:
    if not args.print_ties:
        return

    if candidate_pool is None:
        return

    if optimization.used_exact_thresholded:
        print_tied_solutions(
            candidates=candidate_pool.candidates,
            clearance=candidate_pool.clearance,
            box=box,
            tied_combinations=optimization.tied_combinations,
            total_ties=optimization.total_ties,
            threshold_cutoff=args.max_obstacle_distance,
            max_printed=args.max_printed_tied_solutions,
            selected_tie_index=args.pick_tied_solution,
        )
        return

    print(
        "Note: tied-solution printing is only supported when exhaustive "
        "thresholded maximin grid search is used."
    )


def maybe_write_output_structure(
    args: argparse.Namespace,
    context: ExecutionContext,
    solution: Solution,
    replaced_atom_indices: Set[int] | None = None,
) -> None:
    if not args.output_path:
        return

    excluded_atom_indices = set(context.setup.excluded_atom_indices)
    if replaced_atom_indices:
        excluded_atom_indices |= set(replaced_atom_indices)

    n_removed = write_output_structure(
        context.universe,
        solution.points,
        args.output_path,
        excluded_atom_indices,
        args.point_label,
    )
    print(f"Wrote output structure with dummy atoms to: {args.output_path}")
    print(
        "Omitted atoms from written output: "
        f"total={n_removed} hydrogens={len(context.setup.excluded_hydrogen_indices)} "
        f"waters={len(context.setup.excluded_water_indices)} "
        f"ignore_selection={len(context.setup.excluded_selection_indices)} "
        f"replaced_points={0 if replaced_atom_indices is None else len(replaced_atom_indices)}"
    )


def maybe_print_verbose_configuration(
    args: argparse.Namespace,
    context: ExecutionContext,
    optimization: OptimizationSummary,
) -> None:
    if not args.verbose:
        return

    report_obstacles = optimization.report_obstacles
    if report_obstacles is None:
        report_obstacles = context.setup.obstacle

    print_verbose_solution(
        optimization.solution,
        report_obstacles,
        context.box,
        args.point_label,
        output_order_matches=bool(args.output_path),
    )


def maybe_write_pymol_script(
    args: argparse.Namespace,
    context: ExecutionContext,
    optimization: OptimizationSummary,
) -> None:
    if args.pym is None:
        return

    script_path = Path(args.pym).expanduser().resolve()
    write_pymol_sphere_script(
        context.structure_path,
        script_path,
        optimization.solution,
        args.point_label,
    )
    print(f"Wrote PyMOL script to: {script_path}")
