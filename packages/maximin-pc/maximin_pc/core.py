from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array
from tqdm.auto import tqdm

from . import HydrogenMaskError

THRESHOLDED_SENTINEL = 1000.0
CHARGED_RESNAMES = {"ARG", "LYS", "ASP", "GLU"}
DISTANCE_ARRAY_BACKEND = "OpenMP"
CLEARANCE_DISTANCE_WORKSPACE_BYTES = 512 * 1024 * 1024
SAMPLED_OBSTACLE_CHUNK_TARGET = 1000
DENSE_PAIR_SOLVER_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
DENSE_PAIR_SOLVER_BYTES_PER_PAIR = 16
JOINT_CAP_SHORTLIST_MULTIPLIER = 4
SAMPLED_REFINEMENT_DISTANCE_EVAL_BUDGET = 64 * 1024 * 1024
SAMPLED_REFINEMENT_SCHEDULE: Tuple[Tuple[float, int], ...] = (
    (4.0, 512),
    (2.0, 512),
    (1.0, 512),
    (0.5, 256),
)


@dataclass
class Solution:
    indices: List[int]
    points: np.ndarray
    point_clearances: np.ndarray
    pair_min_distance: float
    selection_pair_score: float


@dataclass
class SearchSetup:
    obstacle: mda.AtomGroup
    excluded_atom_indices: Set[int]
    excluded_hydrogen_indices: Set[int]
    excluded_water_indices: Set[int]
    excluded_selection_indices: Set[int]


def atom_is_hydrogen(atom) -> bool:
    element = getattr(atom, "element", None)
    if element is not None:
        element = str(element).strip()
        if element:
            return element.upper() == "H"

    atom_type = getattr(atom, "type", None)
    if atom_type is not None:
        atom_type = str(atom_type).strip()
        if atom_type and atom_type.upper() == "H":
            return True

    name = getattr(atom, "name", "")
    if name is None:
        name = ""
    name = str(name).strip().upper()
    return name.startswith("H")


def build_search_setup(
    u: mda.Universe,
    selection: str,
    exclude_hydrogens: bool,
    ignore_waters: bool = False,
    ignore_selection: Optional[str] = None,
) -> SearchSetup:
    obstacle = u.select_atoms(selection)
    if len(obstacle) == 0:
        raise ValueError(f"Selection {selection!r} returned no atoms")

    excluded_hydrogen_indices: Set[int] = set()
    excluded_water_indices: Set[int] = set()
    excluded_selection_indices: Set[int] = set()

    if ignore_waters:
        water_atoms = u.select_atoms("water")
        excluded_water_indices = set(water_atoms.indices.tolist())
        if excluded_water_indices:
            keep = np.array(
                [atom.index not in excluded_water_indices for atom in obstacle],
                dtype=bool,
            )
            obstacle = obstacle[keep]

    if ignore_selection:
        extra = u.select_atoms(ignore_selection)
        excluded_selection_indices = set(extra.indices.tolist())
        if excluded_selection_indices:
            keep = np.array(
                [atom.index not in excluded_selection_indices for atom in obstacle],
                dtype=bool,
            )
            obstacle = obstacle[keep]

    if exclude_hydrogens:
        hydrogen_mask = np.array([atom_is_hydrogen(atom) for atom in obstacle], dtype=bool)
        excluded_hydrogen_indices = set(obstacle[hydrogen_mask].indices.tolist())
        obstacle = obstacle[~hydrogen_mask]

    if len(obstacle) == 0:
        raise HydrogenMaskError(
            "The obstacle set became empty after applying the exclusion filters; "
            "cannot continue with an empty obstacle set"
        )

    excluded_atom_indices = (
        excluded_hydrogen_indices | excluded_water_indices | excluded_selection_indices
    )

    return SearchSetup(
        obstacle=obstacle,
        excluded_atom_indices=excluded_atom_indices,
        excluded_hydrogen_indices=excluded_hydrogen_indices,
        excluded_water_indices=excluded_water_indices,
        excluded_selection_indices=excluded_selection_indices,
    )


def get_cell_matrix(ts) -> np.ndarray:
    tri = np.asarray(ts.triclinic_dimensions, dtype=float)
    if tri.shape != (3, 3):
        raise ValueError("Could not obtain a valid triclinic cell matrix from trajectory frame")
    return tri


def radical_inverse_sequence(n: int, base: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)

    indices = np.arange(1, n + 1, dtype=np.int64)
    inv_base = 1.0 / float(base)
    factor = inv_base
    result = np.zeros(n, dtype=float)

    while np.any(indices):
        indices, digit = np.divmod(indices, base)
        result += digit * factor
        factor *= inv_base

    return result


def generate_random_candidates(
    cell: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frac = np.column_stack(
        (
            radical_inverse_sequence(n, 2),
            radical_inverse_sequence(n, 3),
            radical_inverse_sequence(n, 5),
        )
    )
    # A Cranley-Patterson rotation preserves the Halton sequence's space-filling
    # coverage while keeping sampled mode randomized under the requested seed.
    frac = (frac + rng.random(3)) % 1.0
    return frac @ cell


def generate_grid_candidates(cell: np.ndarray, nu: int, nv: int, nw: int) -> np.ndarray:
    us = np.arange(nu, dtype=float) / float(nu)
    vs = np.arange(nv, dtype=float) / float(nv)
    ws = np.arange(nw, dtype=float) / float(nw)
    uu, vv, ww = np.meshgrid(us, vs, ws, indexing="ij")
    frac = np.column_stack((uu.ravel(), vv.ravel(), ww.ravel()))
    return frac @ cell


def wrap_points(points: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    # Wrapping in fractional coordinates keeps triclinic cells consistent.
    frac = points @ inv_cell
    frac_wrapped = frac - np.floor(frac)
    return frac_wrapped @ cell


def compute_clearance(
    candidates: np.ndarray,
    obstacle_positions: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(obstacle_positions) == 0:
        return np.full(len(candidates), np.inf, dtype=np.float64)

    row_bytes = max(1, len(obstacle_positions) * np.dtype(np.float64).itemsize)
    chunk_size = max(1, CLEARANCE_DISTANCE_WORKSPACE_BYTES // row_bytes)

    if len(candidates) <= chunk_size:
        d = distance_array(
            candidates,
            obstacle_positions,
            box=box,
            backend=DISTANCE_ARRAY_BACKEND,
        )
        return d.min(axis=1)

    clearance = np.empty(len(candidates), dtype=np.float64)
    work = np.empty((min(chunk_size, len(candidates)), len(obstacle_positions)), dtype=np.float64)

    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        chunk = work[: stop - start]
        distance_array(
            candidates[start:stop],
            obstacle_positions,
            box=box,
            result=chunk,
            backend=DISTANCE_ARRAY_BACKEND,
        )
        clearance[start:stop] = chunk.min(axis=1)

    return clearance


def filter_sampled_candidates_by_window(
    candidates: np.ndarray,
    obstacle_positions: np.ndarray,
    box: np.ndarray,
    min_obstacle_distance: Optional[float],
    max_obstacle_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidates) == 0:
        return candidates, np.zeros(0, dtype=np.float64)

    if min_obstacle_distance is None:
        clearance = compute_clearance(candidates, obstacle_positions, box)
        return apply_sampled_obstacle_distance_window(
            candidates=candidates,
            clearance=clearance,
            min_obstacle_distance=min_obstacle_distance,
            max_obstacle_distance=max_obstacle_distance,
        )

    if len(obstacle_positions) == 0:
        clearance = np.full(len(candidates), np.inf, dtype=np.float64)
        return apply_sampled_obstacle_distance_window(
            candidates=candidates,
            clearance=clearance,
            min_obstacle_distance=min_obstacle_distance,
            max_obstacle_distance=max_obstacle_distance,
        )

    obstacle_chunk_size = choose_sampled_obstacle_chunk_size(
        len(obstacle_positions),
        target=SAMPLED_OBSTACLE_CHUNK_TARGET,
    )
    if obstacle_chunk_size == len(obstacle_positions):
        clearance = compute_clearance(candidates, obstacle_positions, box)
        return apply_sampled_obstacle_distance_window(
            candidates=candidates,
            clearance=clearance,
            min_obstacle_distance=min_obstacle_distance,
            max_obstacle_distance=max_obstacle_distance,
        )

    ordered_obstacles = obstacle_positions[np.random.default_rng(0).permutation(len(obstacle_positions))]
    candidate_chunk_size = max(
        1,
        CLEARANCE_DISTANCE_WORKSPACE_BYTES
        // (obstacle_chunk_size * np.dtype(np.float64).itemsize),
    )
    candidate_chunk_size = min(candidate_chunk_size, len(candidates))
    result_work = np.empty((candidate_chunk_size, obstacle_chunk_size), dtype=np.float64)

    kept_candidates: List[np.ndarray] = []
    kept_clearance: List[np.ndarray] = []

    for start in range(0, len(candidates), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidates))
        active_candidates = candidates[start:stop]
        active_clearance = np.full(len(active_candidates), np.inf, dtype=np.float64)

        for obstacle_start in range(0, len(ordered_obstacles), obstacle_chunk_size):
            if len(active_candidates) == 0:
                break

            obstacle_stop = obstacle_start + obstacle_chunk_size
            obstacle_chunk = ordered_obstacles[obstacle_start:obstacle_stop]
            chunk_result = result_work[: len(active_candidates), :]
            distance_array(
                active_candidates,
                obstacle_chunk,
                box=box,
                result=chunk_result,
                backend=DISTANCE_ARRAY_BACKEND,
            )
            active_clearance = np.minimum(active_clearance, chunk_result.min(axis=1))
            keep = active_clearance >= min_obstacle_distance
            if np.all(keep):
                continue
            active_candidates = active_candidates[keep]
            active_clearance = active_clearance[keep]

        if len(active_candidates) == 0:
            continue

        active_candidates, active_clearance = apply_sampled_obstacle_distance_window(
            candidates=active_candidates,
            clearance=active_clearance,
            min_obstacle_distance=min_obstacle_distance,
            max_obstacle_distance=max_obstacle_distance,
        )
        if len(active_candidates) == 0:
            continue

        kept_candidates.append(active_candidates)
        kept_clearance.append(active_clearance)

    if not kept_candidates:
        return (
            np.empty((0, candidates.shape[1]), dtype=candidates.dtype),
            np.zeros(0, dtype=np.float64),
        )

    return np.vstack(kept_candidates), np.concatenate(kept_clearance)


def pairwise_distances(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return distance_array(points, points, box=box, backend=DISTANCE_ARRAY_BACKEND)


def choose_sampled_obstacle_chunk_size(n_obstacles: int, target: int) -> int:
    if n_obstacles <= target:
        return n_obstacles

    best = n_obstacles
    best_score = (abs(n_obstacles - target), -n_obstacles)
    limit = int(math.isqrt(n_obstacles))
    for divisor in range(1, limit + 1):
        if n_obstacles % divisor != 0:
            continue
        for candidate in (divisor, n_obstacles // divisor):
            score = (abs(candidate - target), -candidate)
            if score < best_score:
                best = candidate
                best_score = score
    return best


def sample_ball_offsets(
    rng: np.random.Generator,
    n: int,
    radius: float,
) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(norms[:, None], 1e-12)
    radii = radius * np.cbrt(rng.random(n))
    return directions * radii[:, None]


def summarize_solution_objective(
    points: np.ndarray,
    point_clearances: np.ndarray,
    box: np.ndarray,
) -> Tuple[float, float]:
    if len(points) > 1:
        pd = pairwise_distances(points, box=box)
        pair_min = float(np.min(pd[np.triu_indices(len(points), 1)]))
    else:
        pair_min = math.inf

    min_clearance = float(np.min(point_clearances)) if len(point_clearances) > 0 else math.inf
    return pair_min, min_clearance


def combined_min_distance_score(
    pair_min_distance: float,
    min_clearance: float,
) -> float:
    return float(min(pair_min_distance, min_clearance))


def numerically_tied(
    values: np.ndarray,
    reference: float,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Compare finite values by tolerance and infinities by exact equality."""
    if math.isfinite(reference):
        finite = np.isfinite(values)
        result = np.zeros(values.shape, dtype=bool)
        result[finite] = np.abs(values[finite] - reference) <= tolerance
        return result
    return values == reference


def choose_refinement_schedule(
    n_points: int,
    n_obstacles: int,
) -> Tuple[Tuple[float, int], ...]:
    if n_points <= 0 or n_obstacles <= 0:
        return SAMPLED_REFINEMENT_SCHEDULE

    total_local = sum(n_local for _, n_local in SAMPLED_REFINEMENT_SCHEDULE)
    base_evaluations = n_points * n_obstacles * total_local
    if base_evaluations <= SAMPLED_REFINEMENT_DISTANCE_EVAL_BUDGET:
        return SAMPLED_REFINEMENT_SCHEDULE

    scale = SAMPLED_REFINEMENT_DISTANCE_EVAL_BUDGET / float(base_evaluations)
    adjusted_schedule: List[Tuple[float, int]] = []
    for radius, n_local in SAMPLED_REFINEMENT_SCHEDULE:
        scaled_n_local = int(round(n_local * scale))
        if scaled_n_local >= 16:
            adjusted_schedule.append((radius, scaled_n_local))

    if adjusted_schedule:
        return tuple(adjusted_schedule)

    final_radius, _ = SAMPLED_REFINEMENT_SCHEDULE[-1]
    return ((final_radius, 16),)


def charged_obstacle_mask(obstacle: mda.AtomGroup) -> np.ndarray:
    if len(obstacle) == 0:
        return np.zeros(0, dtype=bool)
    resnames = np.array([str(atom.resname).strip().upper() for atom in obstacle], dtype=object)
    return np.array([resn in CHARGED_RESNAMES for resn in resnames], dtype=bool)


def apply_sampled_obstacle_distance_window(
    candidates: np.ndarray,
    clearance: np.ndarray,
    min_obstacle_distance: Optional[float],
    max_obstacle_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(candidates), dtype=bool)
    if min_obstacle_distance is not None:
        keep &= clearance >= min_obstacle_distance
    if max_obstacle_distance is not None:
        keep &= clearance <= max_obstacle_distance
    return candidates[keep], clearance[keep]


def apply_deterministic_min_distance_filter(
    candidates: np.ndarray,
    clearance_all: np.ndarray,
    clearance_charged: Optional[np.ndarray],
    min_obstacle_distance: Optional[float],
    cavity_lower_cutoff_charged: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(candidates), dtype=bool)
    if min_obstacle_distance is not None:
        keep &= clearance_all >= min_obstacle_distance
    if cavity_lower_cutoff_charged is not None and clearance_charged is not None:
        keep &= clearance_charged >= cavity_lower_cutoff_charged
    return candidates[keep], clearance_all[keep]


def max_pair_distance_upper_bound(cell: np.ndarray) -> float:
    return 0.5 * float(
        np.linalg.norm(cell[0]) + np.linalg.norm(cell[1]) + np.linalg.norm(cell[2])
    )


def greedy_seed(order: np.ndarray, conflict: np.ndarray, k: int) -> List[int]:
    chosen: List[int] = []
    for idx in order:
        if all(not conflict[idx, j] for j in chosen):
            chosen.append(int(idx))
            if len(chosen) >= k:
                break
    return chosen


def find_independent_set(
    conflict: np.ndarray,
    priority: np.ndarray,
    k: int,
) -> Optional[List[int]]:
    order = np.argsort(-priority)
    greedy = greedy_seed(order, conflict, k)
    if len(greedy) >= k:
        return greedy[:k]

    # Independent-set feasibility is equivalent to finding a k-clique in the
    # compatibility graph (the complement of the conflict graph). A Tomita-style
    # branch-and-bound search with a greedy coloring bound prunes far more
    # aggressively than the older depth-first search on sparse conflict graphs,
    # which is exactly the hard regime for sampled mode at larger k.
    compatibility = ~conflict.copy()
    np.fill_diagonal(compatibility, False)

    degrees = compatibility.sum(axis=1, dtype=np.int32)
    vertex_order = np.lexsort((-priority, -degrees))
    ordered_compatibility = compatibility[vertex_order][:, vertex_order]
    packed_compatibility = np.packbits(
        ordered_compatibility.astype(np.uint8),
        axis=1,
        bitorder="little",
    )
    compatibility_masks = []
    for vertex, row in enumerate(packed_compatibility):
        mask = int.from_bytes(row.tobytes(), "little")
        mask &= ~(1 << vertex)
        compatibility_masks.append(mask)

    n_vertices = int(ordered_compatibility.shape[0])
    full_mask = (1 << n_vertices) - 1
    chosen_ordered: List[int] = []

    def color_sort(available_mask: int) -> Tuple[List[int], List[int]]:
        ordered_vertices: List[int] = []
        color_bounds: List[int] = []
        remaining = available_mask
        color = 0

        while remaining:
            color += 1
            candidates = remaining
            while candidates:
                bit = candidates & -candidates
                vertex = bit.bit_length() - 1
                ordered_vertices.append(vertex)
                color_bounds.append(color)
                remaining &= ~bit
                candidates &= ~bit
                candidates &= ~compatibility_masks[vertex]

        return ordered_vertices, color_bounds

    def dfs(available_mask: int) -> bool:
        need = k - len(chosen_ordered)
        if need == 0:
            return True
        if available_mask.bit_count() < need:
            return False

        ordered_vertices, color_bounds = color_sort(available_mask)
        for position in range(len(ordered_vertices) - 1, -1, -1):
            if len(chosen_ordered) + color_bounds[position] < k:
                return False

            vertex = ordered_vertices[position]
            bit = 1 << vertex
            chosen_ordered.append(vertex)
            next_available = available_mask & compatibility_masks[vertex] & full_mask
            if dfs(next_available):
                return True
            chosen_ordered.pop()
            available_mask &= ~bit

        return False

    if not dfs(full_mask):
        return None

    chosen = np.asarray(chosen_ordered, dtype=int)
    return vertex_order[chosen].tolist()


def cap_candidates_for_optimization(
    candidates: np.ndarray,
    clearance: np.ndarray,
    max_survivors: int,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    dense_solver_cap = max(
        1,
        math.isqrt(DENSE_PAIR_SOLVER_MEMORY_BUDGET_BYTES // DENSE_PAIR_SOLVER_BYTES_PER_PAIR),
    )
    effective_cap = min(max_survivors, dense_solver_cap)

    if len(candidates) <= effective_cap:
        return candidates, clearance, False, effective_cap

    # We only need the top max_survivors entries, so avoid sorting the full array.
    keep = np.argpartition(clearance, -effective_cap)[-effective_cap:]
    keep = keep[np.argsort(-clearance[keep])]
    return candidates[keep], clearance[keep], True, effective_cap


def cap_candidates_for_joint_optimization(
    candidates: np.ndarray,
    clearance: np.ndarray,
    box: np.ndarray,
    max_survivors: int,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    dense_solver_cap = max(
        1,
        math.isqrt(DENSE_PAIR_SOLVER_MEMORY_BUDGET_BYTES // DENSE_PAIR_SOLVER_BYTES_PER_PAIR),
    )
    effective_cap = min(max_survivors, dense_solver_cap)

    if len(candidates) <= effective_cap:
        return candidates, clearance, False, effective_cap

    # Clearance-only capping can badly bias the joint objective toward "very safe
    # but mutually crowded" points. Instead, first keep a moderate clearance-based
    # shortlist, then apply a greedy joint-aware farthest-point thinning inside
    # that shortlist. This retains high-clearance regions while keeping the
    # diversity preselection much cheaper than operating on every survivor.
    shortlist_size = min(len(candidates), max(effective_cap, effective_cap * JOINT_CAP_SHORTLIST_MULTIPLIER))
    if shortlist_size < len(candidates):
        keep = np.argpartition(clearance, -shortlist_size)[-shortlist_size:]
        keep = keep[np.argsort(-clearance[keep])]
        candidates = candidates[keep]
        clearance = clearance[keep]

    selected = np.empty(effective_cap, dtype=np.int32)
    chosen_mask = np.zeros(len(candidates), dtype=bool)
    min_pair_distance = np.full(len(candidates), np.inf, dtype=np.float64)
    distance_work = np.empty((1, len(candidates)), dtype=np.float64)

    current = int(np.argmax(clearance))
    n_selected = 0

    while n_selected < effective_cap:
        selected[n_selected] = current
        n_selected += 1
        chosen_mask[current] = True

        if n_selected >= effective_cap:
            break

        distance_array(
            candidates[current : current + 1],
            candidates,
            box=box,
            result=distance_work,
            backend=DISTANCE_ARRAY_BACKEND,
        )
        min_pair_distance = np.minimum(min_pair_distance, distance_work[0])
        min_pair_distance[chosen_mask] = -np.inf

        joint_score = np.minimum(clearance, min_pair_distance)
        current = int(np.argmax(joint_score))

    keep = selected[:n_selected]
    return candidates[keep], clearance[keep], True, effective_cap


def build_effective_pair_matrix(
    points: np.ndarray,
    box: np.ndarray,
    thresholded_mode: bool,
    threshold_cutoff: Optional[float],
) -> np.ndarray:
    pd = pairwise_distances(points, box=box)

    if thresholded_mode:
        if threshold_cutoff is None:
            raise ValueError("threshold_cutoff must be provided in thresholded maximin mode")

        # Thresholded maximin distinguishes distances only up to the cutoff;
        # everything larger is treated as equally good via a common sentinel.
        eff = np.full(pd.shape, np.float16(THRESHOLDED_SENTINEL), dtype=np.float16)
        within = pd <= threshold_cutoff
        np.fill_diagonal(within, False)
        eff[within] = pd[within].astype(np.float16)
        np.fill_diagonal(eff, np.float16(THRESHOLDED_SENTINEL))
        return eff

    np.fill_diagonal(pd, np.inf)
    return pd


def feasible_for_pair_radius_from_matrix(
    effective_pair_matrix: np.ndarray,
    clearance: np.ndarray,
    k: int,
    radius: float,
) -> Optional[List[int]]:
    # Two candidates conflict if they cannot simultaneously support the target
    # minimum pair distance.
    conflict = effective_pair_matrix < (radius - 1e-9)
    np.fill_diagonal(conflict, False)

    local_choice = find_independent_set(conflict, clearance, k)
    if local_choice is None:
        return None
    return [int(i) for i in local_choice]


def feasible_for_joint_radius_from_sorted_matrix(
    effective_pair_matrix: np.ndarray,
    clearance_desc: np.ndarray,
    k: int,
    radius: float,
) -> Optional[List[int]]:
    radius_tol = max(0.0, radius - 1e-9)
    eligible = int(np.searchsorted(-clearance_desc, -radius_tol, side="right"))
    if eligible < k:
        return None

    conflict = effective_pair_matrix[:eligible, :eligible] < radius_tol
    np.fill_diagonal(conflict, False)

    local_choice = find_independent_set(conflict, clearance_desc[:eligible], k)
    if local_choice is None:
        return None
    return [int(i) for i in local_choice]


def greedy_pair_fallback_from_matrix(
    effective_pair_matrix: np.ndarray,
    clearance: np.ndarray,
    k: int,
) -> List[int]:
    if k < 1:
        return []

    chosen = [int(np.argmax(clearance))]
    while len(chosen) < k:
        score = effective_pair_matrix[:, np.asarray(chosen, dtype=int)].min(axis=1)
        score[np.asarray(chosen, dtype=int)] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def greedy_joint_fallback_from_matrix(
    effective_pair_matrix: np.ndarray,
    clearance: np.ndarray,
    k: int,
) -> List[int]:
    if k < 1:
        return []

    chosen = [int(np.argmax(clearance))]
    while len(chosen) < k:
        score = np.minimum(
            clearance,
            effective_pair_matrix[:, np.asarray(chosen, dtype=int)].min(axis=1),
        )
        score[np.asarray(chosen, dtype=int)] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def refine_sampled_solution(
    solution: Solution,
    obstacle_positions: np.ndarray,
    box: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    min_obstacle_distance: Optional[float],
    max_obstacle_distance: Optional[float],
    seed: int,
    joint_objective: bool = False,
) -> Solution:
    if len(solution.points) == 0:
        return solution

    points = solution.points.astype(np.float64, copy=True)
    point_clearances = solution.point_clearances.astype(np.float64, copy=True)
    current_pair_min, current_min_clearance = summarize_solution_objective(
        points,
        point_clearances,
        box,
    )
    current_score = combined_min_distance_score(current_pair_min, current_min_clearance)
    refinement_schedule = choose_refinement_schedule(len(points), len(obstacle_positions))

    rng = np.random.default_rng(seed)

    for radius, n_local in refinement_schedule:
        for point_idx in range(len(points)):
            other_mask = np.ones(len(points), dtype=bool)
            other_mask[point_idx] = False
            other_points = points[other_mask]
            other_clearances = point_clearances[other_mask]

            if len(other_points) > 1:
                other_pd = pairwise_distances(other_points, box=box)
                base_other_pair_min = float(np.min(other_pd[np.triu_indices(len(other_points), 1)]))
            else:
                base_other_pair_min = math.inf

            base_other_clearance = (
                float(np.min(other_clearances)) if len(other_clearances) > 0 else math.inf
            )

            local_points = points[point_idx] + sample_ball_offsets(rng, n_local, radius)
            local_points = np.vstack([points[point_idx], local_points])
            local_points = wrap_points(local_points, cell, inv_cell)
            local_clearance = compute_clearance(local_points, obstacle_positions, box)
            local_points, local_clearance = apply_sampled_obstacle_distance_window(
                candidates=local_points,
                clearance=local_clearance,
                min_obstacle_distance=min_obstacle_distance,
                max_obstacle_distance=max_obstacle_distance,
            )
            if len(local_points) == 0:
                continue

            if len(other_points) > 0:
                d_to_others = distance_array(
                    local_points,
                    other_points,
                    box=box,
                    backend=DISTANCE_ARRAY_BACKEND,
                )
                candidate_pair_min = np.minimum(d_to_others.min(axis=1), base_other_pair_min)
            else:
                candidate_pair_min = np.full(len(local_points), math.inf, dtype=np.float64)

            candidate_min_clearance = np.minimum(local_clearance, base_other_clearance)
            candidate_score = np.minimum(candidate_pair_min, candidate_min_clearance)
            if joint_objective:
                score_improved = candidate_score > current_score + 1e-9
                score_tied = numerically_tied(candidate_score, current_score)
                pair_improved = candidate_pair_min > current_pair_min + 1e-9
                pair_tied = numerically_tied(candidate_pair_min, current_pair_min)
                clearance_improved = candidate_min_clearance > current_min_clearance + 1e-9
                better = np.flatnonzero(
                    score_improved
                    | (score_tied & pair_improved)
                    | (score_tied & pair_tied & clearance_improved)
                )
            else:
                better = np.flatnonzero(
                    (candidate_pair_min > current_pair_min + 1e-9)
                    | (
                        numerically_tied(candidate_pair_min, current_pair_min)
                        & (candidate_min_clearance > current_min_clearance + 1e-9)
                    )
                )
            if len(better) == 0:
                continue

            if joint_objective:
                choice_order = np.lexsort(
                    (
                        candidate_min_clearance[better],
                        candidate_pair_min[better],
                        candidate_score[better],
                    )
                )
            else:
                choice_order = np.lexsort(
                    (candidate_min_clearance[better], candidate_pair_min[better])
                )
            best_local = better[choice_order[-1]]
            points[point_idx] = local_points[best_local]
            point_clearances[point_idx] = local_clearance[best_local]
            current_pair_min = float(candidate_pair_min[best_local])
            current_min_clearance = float(candidate_min_clearance[best_local])
            current_score = float(candidate_score[best_local])

    return build_solution_from_indices(
        candidates=points,
        clearance=point_clearances,
        box=box,
        indices=range(len(points)),
    )


def build_solution_from_indices(
    candidates: np.ndarray,
    clearance: np.ndarray,
    box: np.ndarray,
    indices: Sequence[int],
    thresholded_mode: bool = False,
    threshold_cutoff: Optional[float] = None,
) -> Solution:
    idx = np.asarray(indices, dtype=int)
    points = candidates[idx]
    point_clearances = clearance[idx]

    if len(idx) > 1:
        pd = pairwise_distances(points, box=box)
        pair_min = float(np.min(pd[np.triu_indices(len(idx), 1)]))

        if thresholded_mode:
            if threshold_cutoff is None:
                raise ValueError("threshold_cutoff must be provided in thresholded maximin mode")

            eff = np.full(pd.shape, np.float16(THRESHOLDED_SENTINEL), dtype=np.float16)
            within = pd <= threshold_cutoff
            np.fill_diagonal(within, False)
            eff[within] = pd[within].astype(np.float16)
            np.fill_diagonal(eff, np.float16(THRESHOLDED_SENTINEL))
            selection_pair_score = float(np.min(eff[np.triu_indices(len(idx), 1)]))
        else:
            selection_pair_score = pair_min
    else:
        pair_min = math.inf
        selection_pair_score = THRESHOLDED_SENTINEL if thresholded_mode else math.inf

    return Solution(
        indices=idx.tolist(),
        points=points,
        point_clearances=point_clearances,
        pair_min_distance=float(pair_min),
        selection_pair_score=float(selection_pair_score),
    )


def n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def exact_thresholded_subset_search(
    candidates: np.ndarray,
    clearance: np.ndarray,
    box: np.ndarray,
    k: int,
    threshold_cutoff: float,
    max_combinations: int,
    collect_ties: bool = False,
    max_stored_ties: int = 0,
    chunk_size: int = 200000,
    show_progress: bool = False,
    selected_tie_index: int = 1,
) -> Tuple[Optional[Solution], List[Tuple[int, ...]], int]:
    n = len(candidates)
    ncomb = n_choose_k(n, k)
    if ncomb == 0 or ncomb > max_combinations:
        return None, [], 0

    effective_pair_matrix = build_effective_pair_matrix(
        points=candidates,
        box=box,
        thresholded_mode=True,
        threshold_cutoff=threshold_cutoff,
    )

    stored_ties: List[Tuple[int, ...]] = []
    total_ties = 0

    if k == 1:
        # The k=1 case degenerates to picking the row with the best thresholded score,
        # so we can skip the combination machinery entirely.
        row_scores = effective_pair_matrix.sum(axis=1, dtype=np.float64)
        best_score = float(np.max(row_scores))
        tied_idx = np.flatnonzero(row_scores == best_score)
        total_ties = int(len(tied_idx))

        if selected_tie_index > total_ties:
            raise ValueError(
                f"Requested tied solution {selected_tie_index}, but only {total_ties} "
                "tied optimal solutions exist"
            )

        if collect_ties:
            if max_stored_ties == 0:
                stored_ties = [(int(i),) for i in tied_idx.tolist()]
            else:
                stored_ties = [(int(i),) for i in tied_idx[:max_stored_ties].tolist()]

        selected_idx = int(tied_idx[selected_tie_index - 1])
        sol = build_solution_from_indices(
            candidates=candidates,
            clearance=clearance,
            box=box,
            indices=[selected_idx],
            thresholded_mode=True,
            threshold_cutoff=threshold_cutoff,
        )
        sol.selection_pair_score = best_score
        return sol, stored_ties, total_ties

    pair_pos = np.array(list(combinations(range(k), 2)), dtype=int)
    pair_i = pair_pos[:, 0]
    pair_j = pair_pos[:, 1]

    best_score = -np.inf
    selected_indices: Optional[Tuple[int, ...]] = None
    combo_iter = combinations(range(n), k)

    progress_bar = (
        tqdm(total=ncomb, desc="Exhaustive thresholded maximin search", unit="comb")
        if show_progress else None
    )

    try:
        while True:
            # Enumerate combinations in chunks so exact mode can stay memory-bounded
            # even when the total number of subsets is large but still admissible.
            chunk = list(islice(combo_iter, chunk_size))
            if not chunk:
                break

            comb_dtype = np.uint16 if n < 65536 else np.uint32
            combs = np.asarray(chunk, dtype=comb_dtype)

            pair_vals = effective_pair_matrix[combs[:, pair_i], combs[:, pair_j]]
            scores = pair_vals.min(axis=1).astype(np.float32)

            chunk_best = float(np.max(scores))
            best_rows = np.flatnonzero(scores == chunk_best)

            if chunk_best > best_score:
                best_score = chunk_best
                total_ties = int(best_rows.size)

                if selected_tie_index <= total_ties:
                    selected_indices = tuple(
                        int(x) for x in combs[best_rows[selected_tie_index - 1]]
                    )
                else:
                    selected_indices = None

                if collect_ties:
                    if max_stored_ties == 0:
                        stored_ties = [tuple(int(x) for x in row) for row in combs[best_rows]]
                    else:
                        stored_ties = [
                            tuple(int(x) for x in row) for row in combs[best_rows[:max_stored_ties]]
                        ]
                else:
                    stored_ties = []

            elif chunk_best == best_score:
                prev_ties = total_ties
                total_ties += int(best_rows.size)

                if selected_indices is None and prev_ties < selected_tie_index <= total_ties:
                    local_pos = selected_tie_index - prev_ties - 1
                    selected_indices = tuple(int(x) for x in combs[best_rows[local_pos]])

                if collect_ties:
                    if max_stored_ties == 0:
                        stored_ties.extend(tuple(int(x) for x in row) for row in combs[best_rows])
                    else:
                        remaining = max_stored_ties - len(stored_ties)
                        if remaining > 0:
                            stored_ties.extend(
                                tuple(int(x) for x in row) for row in combs[best_rows[:remaining]]
                            )

            if progress_bar is not None:
                progress_bar.update(len(chunk))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if total_ties == 0 or selected_indices is None:
        raise ValueError(
            f"Requested tied solution {selected_tie_index}, but only {total_ties} "
            "tied optimal solutions exist"
        )

    sol = build_solution_from_indices(
        candidates=candidates,
        clearance=clearance,
        box=box,
        indices=selected_indices,
        thresholded_mode=True,
        threshold_cutoff=threshold_cutoff,
    )
    return sol, stored_ties, total_ties


def solve_max_pair_distance(
    candidates: np.ndarray,
    clearance: np.ndarray,
    cell: np.ndarray,
    box: np.ndarray,
    k: int,
    max_survivors: int,
    binary_steps: int,
    thresholded_mode: bool = False,
    threshold_cutoff: Optional[float] = None,
) -> Tuple[Solution, bool, int]:
    if candidates.shape[0] < k:
        raise ValueError("Need at least as many candidates as points to place")

    capped_candidates, capped_clearance, was_capped, effective_cap = cap_candidates_for_optimization(
        candidates=candidates,
        clearance=clearance,
        max_survivors=max_survivors,
    )

    effective_pair_matrix = build_effective_pair_matrix(
        points=capped_candidates,
        box=box,
        thresholded_mode=thresholded_mode,
        threshold_cutoff=threshold_cutoff,
    )

    low = 0.0
    high = THRESHOLDED_SENTINEL if thresholded_mode else max_pair_distance_upper_bound(cell)
    best_indices: Optional[List[int]] = None

    # Feasibility is monotone in the target radius, so binary search works well here.
    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        chosen = feasible_for_pair_radius_from_matrix(
            effective_pair_matrix=effective_pair_matrix,
            clearance=capped_clearance,
            k=k,
            radius=mid,
        )
        if chosen is not None:
            low = mid
            best_indices = chosen
        else:
            high = mid

    if best_indices is None:
        best_indices = greedy_pair_fallback_from_matrix(
            effective_pair_matrix,
            capped_clearance,
            k,
        )

    sol = build_solution_from_indices(
        candidates=capped_candidates,
        clearance=capped_clearance,
        box=box,
        indices=best_indices,
        thresholded_mode=thresholded_mode,
        threshold_cutoff=threshold_cutoff,
    )
    return sol, was_capped, effective_cap


def solve_joint_min_distance(
    candidates: np.ndarray,
    clearance: np.ndarray,
    cell: np.ndarray,
    box: np.ndarray,
    k: int,
    max_survivors: int,
    binary_steps: int,
) -> Tuple[Solution, bool, int]:
    if candidates.shape[0] < k:
        raise ValueError("Need at least as many candidates as points to place")

    capped_candidates, capped_clearance, was_capped, effective_cap = cap_candidates_for_joint_optimization(
        candidates=candidates,
        clearance=clearance,
        box=box,
        max_survivors=max_survivors,
    )
    order = np.argsort(-capped_clearance, kind="stable")
    capped_candidates = capped_candidates[order]
    capped_clearance = capped_clearance[order]

    if k == 1:
        sol = build_solution_from_indices(
            candidates=capped_candidates,
            clearance=capped_clearance,
            box=box,
            indices=[0],
        )
        return sol, was_capped, effective_cap

    effective_pair_matrix = build_effective_pair_matrix(
        points=capped_candidates,
        box=box,
        thresholded_mode=False,
        threshold_cutoff=None,
    )

    low = 0.0
    high = min(max_pair_distance_upper_bound(cell), float(capped_clearance[0]))
    best_indices: Optional[List[int]] = None

    # Joint feasibility remains monotone: a radius is feasible iff there are k
    # candidates with clearance >= radius whose pair distances are also all >= radius.
    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        chosen = feasible_for_joint_radius_from_sorted_matrix(
            effective_pair_matrix=effective_pair_matrix,
            clearance_desc=capped_clearance,
            k=k,
            radius=mid,
        )
        if chosen is not None:
            low = mid
            best_indices = chosen
        else:
            high = mid

    if best_indices is None:
        best_indices = greedy_joint_fallback_from_matrix(
            effective_pair_matrix,
            capped_clearance,
            k,
        )

    sol = build_solution_from_indices(
        candidates=capped_candidates,
        clearance=capped_clearance,
        box=box,
        indices=best_indices,
    )
    return sol, was_capped, effective_cap


def make_dummy_universe(points: np.ndarray, cell: np.ndarray, point_label: str) -> mda.Universe:
    n = len(points)
    atom_resindex = np.arange(n, dtype=int)
    residue_segindex = np.zeros(n, dtype=int)

    u = mda.Universe.empty(
        n_atoms=n,
        n_residues=n,
        n_segments=1,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )
    u.add_TopologyAttr("name", np.array([point_label] * n, dtype=object))
    u.add_TopologyAttr("type", np.array([point_label] * n, dtype=object))
    u.add_TopologyAttr("resname", np.array([point_label] * n, dtype=object))
    u.add_TopologyAttr("resid", np.arange(1, n + 1, dtype=int))
    u.add_TopologyAttr("segid", np.array([point_label], dtype=object))
    u.atoms.positions = points.astype(np.float32, copy=False)

    a, b, c = cell
    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    lc = np.linalg.norm(c)
    alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (la * lc), -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0)))
    u.dimensions = np.array([la, lb, lc, alpha, beta, gamma], dtype=np.float32)
    return u


def write_output_structure(
    source_u: mda.Universe,
    points: np.ndarray,
    outpath: str,
    excluded_atom_indices: Set[int],
    point_label: str,
) -> int:
    # MDAnalysis selects the concrete writer from the output filename extension.
    keep_mask = np.ones(source_u.atoms.n_atoms, dtype=bool)
    if excluded_atom_indices:
        excluded_idx = np.fromiter(sorted(excluded_atom_indices), dtype=int)
        keep_mask[excluded_idx] = False
    kept_atoms = source_u.atoms[keep_mask]

    dummy_u = make_dummy_universe(points, get_cell_matrix(source_u.trajectory.ts), point_label)
    merged = mda.Merge(kept_atoms, dummy_u.atoms)
    merged.atoms.positions = np.vstack([kept_atoms.positions, dummy_u.atoms.positions]).astype(
        np.float32
    )
    merged.dimensions = source_u.trajectory.ts.dimensions.copy()

    with mda.Writer(outpath, n_atoms=merged.atoms.n_atoms) as w:
        w.write(merged.atoms)

    return int(source_u.atoms.n_atoms - kept_atoms.n_atoms)


def format_atom_label(atom) -> str:
    segid = str(getattr(atom, "segid", "") or "").strip()
    chainid = str(getattr(atom, "chainID", "") or "").strip()
    name = str(getattr(atom, "name", "") or "").strip()
    resname = str(getattr(atom, "resname", "") or "").strip()
    resid = getattr(atom, "resid", "")
    atom_index = int(atom.index) + 1

    parts = [f"atom_index={atom_index}"]
    if segid:
        parts.append(f"segid={segid}")
    if chainid:
        parts.append(f"chainID={chainid}")
    if resname or resid != "":
        parts.append(f"residue={resname}{resid}")
    if name:
        parts.append(f"name={name}")
    return " ".join(parts)


def print_solution(
    sol: Solution,
    thresholded_mode: bool,
    joint_objective: bool = False,
    evaluated_only: bool = False,
) -> None:
    min_clearance = float(np.min(sol.point_clearances)) if len(sol.point_clearances) > 0 else math.inf
    pair_label = "Evaluated" if evaluated_only else "Optimal"

    if thresholded_mode:
        if math.isfinite(sol.selection_pair_score):
            print(f"Thresholded maximin pair score (A): {sol.selection_pair_score:.6f}")
        else:
            print("Thresholded maximin pair score (A): inf (only one point requested)")

        if math.isfinite(sol.pair_min_distance):
            print(f"Actual minimum point-point distance (A): {sol.pair_min_distance:.6f}")
        else:
            print("Actual minimum point-point distance (A): inf (only one point requested)")
    elif joint_objective:
        joint_score = combined_min_distance_score(sol.pair_min_distance, min_clearance)
        if math.isfinite(joint_score):
            print(f"{pair_label} joint minimum distance (A): {joint_score:.6f}")
        else:
            print(f"{pair_label} joint minimum distance (A): inf (only one point requested)")

        if math.isfinite(sol.pair_min_distance):
            print(f"Actual minimum point-point distance (A): {sol.pair_min_distance:.6f}")
        else:
            print("Actual minimum point-point distance (A): inf (only one point requested)")
    else:
        if math.isfinite(sol.pair_min_distance):
            print(f"{pair_label} minimum point-point distance (A): {sol.pair_min_distance:.6f}")
        else:
            print(f"{pair_label} minimum point-point distance (A): inf (only one point requested)")

    print(f"Minimum clearance to obstacle among selected points (A): {min_clearance:.6f}")
    print("index x y z clearance_to_obstacle")
    for i, (p, c) in enumerate(zip(sol.points, sol.point_clearances), start=1):
        print(f"{i:3d} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f} {c:12.6f}")


def print_verbose_solution(
    sol: Solution,
    obstacle: mda.AtomGroup,
    box: np.ndarray,
    point_label: str,
    output_order_matches: bool = False,
) -> None:
    header = "Verbose final configuration"
    if output_order_matches:
        header += f" ({point_label} order matches written output)"
    print(header + ":")

    if len(sol.points) == 0:
        print(f"  No {point_label} atoms to report.")
        return

    if len(sol.points) > 1:
        pair_matrix = pairwise_distances(sol.points, box=box)
    else:
        pair_matrix = np.full((1, 1), np.inf, dtype=np.float64)

    nearest_indices = np.zeros(len(sol.points), dtype=int)
    nearest_distances = np.full(len(sol.points), np.inf, dtype=np.float64)
    if len(obstacle) > 0:
        obstacle_distances = distance_array(
            sol.points,
            obstacle.positions,
            box=box,
            backend=DISTANCE_ARRAY_BACKEND,
        )
        nearest_indices = np.argmin(obstacle_distances, axis=1)
        nearest_distances = obstacle_distances[np.arange(len(sol.points)), nearest_indices]

    for i, point in enumerate(sol.points, start=1):
        print(f"{point_label} {i}:")
        print(f"  position (A): {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
        print(f"  minimum obstacle distance (A): {nearest_distances[i - 1]:.6f}")
        if len(obstacle) > 0:
            nearest_atom = obstacle[int(nearest_indices[i - 1])]
            print(f"  nearest obstacle atom: {format_atom_label(nearest_atom)}")
        else:
            print("  nearest obstacle atom: none")

        if len(sol.points) == 1:
            print(f"  distances to other {point_label} atoms (A): none")
            continue

        print(f"  distances to other {point_label} atoms (A):")
        for j in range(len(sol.points)):
            if j == i - 1:
                continue
            print(f"    to {point_label} {j + 1}: {pair_matrix[i - 1, j]:.6f}")


def write_pymol_sphere_script(
    structure_path: Path,
    script_path: Path,
    sol: Solution,
    point_label: str,
) -> None:
    structure_literal = repr(str(structure_path))
    object_name = f"{point_label.lower()}_evaluation"
    sphere_object_name = f"{point_label}_clearance_spheres"
    circle_object_name = f"{point_label}_clearance_circles"
    group_by_molecule = structure_path.suffix.lower() == ".gro"
    points_literal = repr(
        [
            (
                idx,
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(clearance),
            )
            for idx, (point, clearance) in enumerate(
                zip(sol.points, sol.point_clearances),
                start=1,
            )
        ]
    )

    script = f"""# PyMOL helper script generated by maximin-pc
# Run from within PyMOL with:
#   run {script_path}

import math
import numpy as np
from pymol import CmdException, cmd

STRUCTURE_PATH = {structure_literal}
OBJECT_NAME = {object_name!r}
POINT_LABEL = {point_label!r}
SPHERE_OBJECT_NAME = {sphere_object_name!r}
CIRCLE_OBJECT_NAME = {circle_object_name!r}
GROUP_BY_MOLECULE = {group_by_molecule!r}
POINTS = {points_literal}
POINT_SELECTION = "(" + OBJECT_NAME + ") and resn " + POINT_LABEL + " and name " + POINT_LABEL
SPHERE_COLOR = (0.20, 0.70, 1.00)
SPHERE_ALPHA = 0.25
SPHERE_QUALITY = 4
SPHERE_COLOR_NAME = f"{{POINT_LABEL.lower()}}_clearance_sphere_color"
CIRCLE_COLOR = (0.20, 0.70, 1.00)
CIRCLE_LINE_WIDTH = 2.0
CIRCLE_SEGMENTS = 720

try:
    from pymol.callback import Callback
    from OpenGL.GL import (
        glBegin,
        glColor3f,
        glDisable,
        glEnable,
        glEnd,
        glGetDoublev,
        glLineWidth,
        glMultMatrixd,
        glPopMatrix,
        glPushMatrix,
        glTranslatef,
        glVertex3f,
        GL_DEPTH_TEST,
        GL_LINE_LOOP,
        GL_LIGHTING,
        GL_MODELVIEW_MATRIX,
    )
    OPENGL_AVAILABLE = True
    OPENGL_IMPORT_ERROR = None
except Exception as open_gl_import_error:
    OPENGL_AVAILABLE = False
    OPENGL_IMPORT_ERROR = str(open_gl_import_error)


def build_cell_matrices(symmetry):
    if not symmetry or len(symmetry) < 6:
        return None, None

    a, b, c, alpha_deg, beta_deg, gamma_deg = [float(value) for value in symmetry[:6]]
    if min(a, b, c) <= 0.0:
        return None, None

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-8:
        return None, None

    cell = np.array(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [
                c * cos_beta,
                c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                0.0,
            ],
        ],
        dtype=float,
    )
    cz_sq = c * c - cell[2, 0] ** 2 - cell[2, 1] ** 2
    if cz_sq < 0.0 and abs(cz_sq) < 1.0e-8:
        cz_sq = 0.0
    if cz_sq < 0.0:
        return None, None
    cell[2, 2] = math.sqrt(cz_sq)
    return cell, np.linalg.inv(cell)


def wrap_protein_chains_into_unit_cell(object_name):
    cell, inv_cell = build_cell_matrices(cmd.get_symmetry(object_name, state=1, quiet=1))
    if cell is None:
        return

    protein_selection = f"({{object_name}}) and polymer.protein"
    protein_model = cmd.get_model(protein_selection, state=1)
    if not protein_model.atom:
        return

    groups = []
    if GROUP_BY_MOLECULE:
        visited = set()
        atom_indices = [int(atom.index) for atom in protein_model.atom]
        for atom_index in atom_indices:
            if atom_index in visited:
                continue
            molecule_selection = (
                f"bymolecule (({{object_name}}) and polymer.protein and index {{atom_index}})"
            )
            molecule_model = cmd.get_model(molecule_selection, state=1)
            if not molecule_model.atom:
                continue
            group_indices = [int(atom.index) for atom in molecule_model.atom]
            visited.update(group_indices)
            groups.append(
                {{
                    "indices": group_indices,
                    "coords": [
                        [float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2])]
                        for atom in molecule_model.atom
                    ],
                }}
            )
    else:
        chain_groups = {{}}
        for atom in protein_model.atom:
            key = (
                str(getattr(atom, "segi", "") or ""),
                str(getattr(atom, "chain", "") or ""),
            )
            group = chain_groups.setdefault(key, {{"indices": [], "coords": []}})
            group["indices"].append(int(atom.index))
            group["coords"].append(
                [
                    float(atom.coord[0]),
                    float(atom.coord[1]),
                    float(atom.coord[2]),
                ]
            )
        groups = [chain_groups[key] for key in sorted(chain_groups)]

    for group in groups:
        coords = np.asarray(group["coords"], dtype=float)
        center = coords.mean(axis=0)
        frac = center @ inv_cell
        shift_frac = -np.floor(frac)
        if np.allclose(shift_frac, 0.0):
            continue

        shift_cart = shift_frac @ cell
        selection = f"({{object_name}}) and index {{'+'.join(str(idx) for idx in group['indices'])}}"
        cmd.translate(
            [float(shift_cart[0]), float(shift_cart[1]), float(shift_cart[2])],
            selection,
            state=1,
            camera=0,
        )


CURRENT_SPHERE_ALPHA = float(SPHERE_ALPHA)
CURRENT_SPHERE_COLOR = tuple(float(value) for value in SPHERE_COLOR)
CURRENT_CIRCLE_COLOR = tuple(float(value) for value in CIRCLE_COLOR)
CURRENT_RADIUS_MODE = "clearance"
POINT_IDENTIFIERS = []


def get_static_point_records():
    return [
        (
            int(idx),
            float(x),
            float(y),
            float(z),
            abs(float(radius)),
            None,
        )
        for idx, x, y, z, radius in POINTS
    ]


def build_point_identifier(atom):
    atom_id = getattr(atom, "id", None)
    if atom_id not in (None, ""):
        atom_id = int(atom_id)

    resi_number = getattr(atom, "resi_number", None)
    if resi_number not in (None, ""):
        resi_number = int(resi_number)

    atom_index = getattr(atom, "index", None)
    if atom_index not in (None, ""):
        atom_index = int(atom_index)

    return {{
        "id": atom_id,
        "resi_number": resi_number,
        "resi": str(getattr(atom, "resi", "") or "").strip(),
        "index": atom_index,
    }}


def initialize_point_mapping():
    global POINT_IDENTIFIERS
    point_model = cmd.get_model(POINT_SELECTION, state=1)
    if len(point_model.atom) != len(POINTS):
        raise CmdException(
            "Loaded structure contains "
            + str(len(point_model.atom))
            + " "
            + POINT_LABEL
            + " marker atoms, but the overlay expects "
            + str(len(POINTS))
            + "."
        )
    POINT_IDENTIFIERS = [build_point_identifier(atom) for atom in point_model.atom]


def assign_atoms_by_key(matched_atoms, used_positions, atoms, key_name):
    lookup = {{}}
    duplicates = set()
    for position, atom in enumerate(atoms):
        key = build_point_identifier(atom).get(key_name)
        if key in (None, ""):
            continue
        if key in lookup:
            duplicates.add(key)
        else:
            lookup[key] = position

    for key in duplicates:
        lookup.pop(key, None)

    for slot, identifier in enumerate(POINT_IDENTIFIERS):
        if matched_atoms[slot] is not None:
            continue
        key = identifier.get(key_name)
        if key in (None, ""):
            continue
        position = lookup.get(key)
        if position is None or position in used_positions:
            continue
        matched_atoms[slot] = atoms[position]
        used_positions.add(position)


def resolve_current_point_atoms():
    if not POINT_IDENTIFIERS:
        initialize_point_mapping()

    point_model = cmd.get_model(POINT_SELECTION, state=1)
    atoms = list(point_model.atom)
    if len(atoms) != len(POINTS):
        raise CmdException(
            "Loaded structure currently contains "
            + str(len(atoms))
            + " "
            + POINT_LABEL
            + " marker atoms, but the overlay expects "
            + str(len(POINTS))
            + "."
        )

    matched_atoms = [None] * len(POINT_IDENTIFIERS)
    used_positions = set()
    for key_name in ("id", "resi_number", "resi", "index"):
        assign_atoms_by_key(matched_atoms, used_positions, atoms, key_name)

    remaining_positions = [position for position in range(len(atoms)) if position not in used_positions]
    remaining_atoms = [atoms[position] for position in remaining_positions]
    remaining_slots = [slot for slot, atom in enumerate(matched_atoms) if atom is None]
    if len(remaining_atoms) != len(remaining_slots):
        raise CmdException(
            "Could not match the current "
            + POINT_LABEL
            + " marker atoms to the overlay records."
        )
    for slot, atom in zip(remaining_slots, remaining_atoms):
        matched_atoms[slot] = atom

    return matched_atoms


def get_current_point_records(allow_static_fallback=False):
    try:
        matched_atoms = resolve_current_point_atoms()
    except CmdException:
        if allow_static_fallback:
            return get_static_point_records()
        raise

    records = []
    for atom, (idx, _x, _y, _z, radius) in zip(matched_atoms, POINTS):
        atom_identifier = build_point_identifier(atom)
        records.append(
            (
                int(idx),
                float(atom.coord[0]),
                float(atom.coord[1]),
                float(atom.coord[2]),
                abs(float(radius)),
                atom_identifier.get("index"),
            )
        )

    if CURRENT_RADIUS_MODE == "pair":
        coords = np.asarray([[x, y, z] for _, x, y, z, _radius, _atom_index in records], dtype=float)
        if len(coords) > 1:
            cell, inv_cell = build_cell_matrices(cmd.get_symmetry(OBJECT_NAME, state=1, quiet=1))
            pair_radii = np.full(len(coords), np.inf, dtype=float)
            for idx in range(len(coords)):
                diffs = coords - coords[idx]
                if cell is not None and inv_cell is not None:
                    frac = diffs @ inv_cell
                    frac -= np.rint(frac)
                    diffs = frac @ cell
                distances = np.linalg.norm(diffs, axis=1)
                distances[idx] = np.inf
                pair_radii[idx] = float(np.min(distances))
            records = [
                (idx, x, y, z, float(pair_radius), atom_index)
                for (idx, x, y, z, _radius, atom_index), pair_radius in zip(records, pair_radii)
            ]
    return records


def compute_point_extent(records):
    if not records:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for _, x, y, z, radius, _atom_index in records:
        mins[0] = min(mins[0], x - radius)
        mins[1] = min(mins[1], y - radius)
        mins[2] = min(mins[2], z - radius)
        maxs[0] = max(maxs[0], x + radius)
        maxs[1] = max(maxs[1], y + radius)
        maxs[2] = max(maxs[2], z + radius)
    return [mins, maxs]


def object_is_enabled(name, default_enabled):
    if name not in set(cmd.get_names("objects")):
        return default_enabled
    return name in set(cmd.get_names("objects", enabled_only=1))


def parse_alpha_value(alpha):
    try:
        value = float(alpha)
    except ValueError as err:
        raise CmdException("Sphere alpha must be a numeric value between 0 and 1") from err
    if value < 0.0 or value > 1.0:
        raise CmdException("Sphere alpha must be between 0 and 1")
    return value


def parse_color_value(*components):
    tokens = []
    for component in components:
        for token in str(component).split(","):
            token = token.strip()
            if token:
                tokens.append(token)

    if len(tokens) != 3:
        raise CmdException("Color must be provided as r,g,b with each component in [0,1]")

    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as err:
        raise CmdException("Color values must be numeric") from err

    if any(value < 0.0 or value > 1.0 for value in values):
        raise CmdException("Color components must each be between 0 and 1")
    return values


def reload_clearance_spheres(enabled=None):
    records = get_current_point_records()
    if enabled is None:
        enabled = object_is_enabled(SPHERE_OBJECT_NAME, True)
    cmd.delete(SPHERE_OBJECT_NAME)
    for idx, x, y, z, radius, _atom_index in records:
        cmd.pseudoatom(
            SPHERE_OBJECT_NAME,
            pos=[float(x), float(y), float(z)],
            vdw=abs(float(radius)),
            name="SPH",
            resn="SPH",
            resi=str(int(idx)),
            elem="Xe",
            state=1,
        )
    cmd.hide("everything", SPHERE_OBJECT_NAME)
    cmd.show("spheres", SPHERE_OBJECT_NAME)
    cmd.set("sphere_scale", 1.0, SPHERE_OBJECT_NAME)
    cmd.set("sphere_quality", SPHERE_QUALITY, SPHERE_OBJECT_NAME)
    cmd.set("sphere_transparency", 1.0 - CURRENT_SPHERE_ALPHA, SPHERE_OBJECT_NAME)
    cmd.set_color(SPHERE_COLOR_NAME, list(CURRENT_SPHERE_COLOR))
    cmd.color(SPHERE_COLOR_NAME, SPHERE_OBJECT_NAME)
    cmd.rebuild(SPHERE_OBJECT_NAME)
    if not enabled:
        cmd.disable(SPHERE_OBJECT_NAME)


if OPENGL_AVAILABLE:
    class ClearanceCircleBillboards(Callback):
        def __init__(
            self,
            color=CIRCLE_COLOR,
            line_width=CIRCLE_LINE_WIDTH,
            segments=CIRCLE_SEGMENTS,
        ):
            self.color = tuple(float(c) for c in color)
            self.line_width = float(line_width)
            self.segments = int(segments)

        def get_extent(self):
            return compute_point_extent(get_current_point_records(allow_static_fallback=True))

        def __call__(self):
            points = get_current_point_records(allow_static_fallback=True)
            if not points:
                return

            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            billboard = [
                modelview[0][0], modelview[1][0], modelview[2][0], 0.0,
                modelview[0][1], modelview[1][1], modelview[2][1], 0.0,
                modelview[0][2], modelview[1][2], modelview[2][2], 0.0,
                0.0,             0.0,             0.0,             1.0,
            ]

            glDisable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)
            glColor3f(*self.color)
            glLineWidth(self.line_width)

            step = 2.0 * math.pi / float(self.segments)
            for _, x, y, z, radius, _atom_index in points:
                glPushMatrix()
                glTranslatef(float(x), float(y), float(z))
                glMultMatrixd(billboard)
                glBegin(GL_LINE_LOOP)
                for i in range(self.segments):
                    angle = i * step
                    glVertex3f(
                        float(radius) * math.cos(angle),
                        float(radius) * math.sin(angle),
                        0.0,
                    )
                glEnd()
                glPopMatrix()
            glEnable(GL_LIGHTING)


def reload_clearance_circles(enabled=None):
    if not OPENGL_AVAILABLE:
        raise CmdException(
            "Camera-facing circle overlay requires OpenGL bindings in the PyMOL environment"
        )
    get_current_point_records()
    if enabled is None:
        enabled = object_is_enabled(CIRCLE_OBJECT_NAME, False)
    cmd.delete(CIRCLE_OBJECT_NAME)
    cmd.load_callback(
        ClearanceCircleBillboards(color=CURRENT_CIRCLE_COLOR),
        CIRCLE_OBJECT_NAME,
    )
    if not enabled:
        cmd.disable(CIRCLE_OBJECT_NAME)


def maximin_pc_refresh_overlays(_self=cmd):
    reload_clearance_spheres()
    if OPENGL_AVAILABLE:
        reload_clearance_circles()
    print("Refreshed maximin-pc overlays from current " + POINT_LABEL + " marker positions.")


def parse_radius_mode(mode):
    value = str(mode).strip().lower()
    if value in ("clearance", "pair"):
        return value
    raise CmdException("Radius mode must be either 'clearance' or 'pair'")


def maximin_pc_set_radius_mode(mode, _self=cmd):
    global CURRENT_RADIUS_MODE
    CURRENT_RADIUS_MODE = parse_radius_mode(mode)
    maximin_pc_refresh_overlays()
    print(
        "maximin-pc overlay radius mode is now "
        + CURRENT_RADIUS_MODE
        + " (clearance or nearest point-point distance)."
    )


def maximin_pc_toggle_radius_mode(_self=cmd):
    next_mode = "pair" if CURRENT_RADIUS_MODE == "clearance" else "clearance"
    maximin_pc_set_radius_mode(next_mode)


def maximin_pc_set_sphere_alpha(alpha, _self=cmd):
    global CURRENT_SPHERE_ALPHA
    CURRENT_SPHERE_ALPHA = parse_alpha_value(alpha)
    reload_clearance_spheres()
    print(f"{{SPHERE_OBJECT_NAME}} alpha set to {{CURRENT_SPHERE_ALPHA:.3f}}")


def maximin_pc_set_sphere_color(*components, _self=cmd):
    global CURRENT_SPHERE_COLOR
    CURRENT_SPHERE_COLOR = parse_color_value(*components)
    reload_clearance_spheres()
    print(f"{{SPHERE_OBJECT_NAME}} color set to {{CURRENT_SPHERE_COLOR}}")


def maximin_pc_set_circle_color(*components, _self=cmd):
    global CURRENT_CIRCLE_COLOR
    CURRENT_CIRCLE_COLOR = parse_color_value(*components)
    reload_clearance_circles()
    print(f"{{CIRCLE_OBJECT_NAME}} color set to {{CURRENT_CIRCLE_COLOR}}")


cmd.delete(OBJECT_NAME)
cmd.delete(SPHERE_OBJECT_NAME)
cmd.delete(CIRCLE_OBJECT_NAME)
cmd.load(STRUCTURE_PATH, OBJECT_NAME)
wrap_protein_chains_into_unit_cell(OBJECT_NAME)
initialize_point_mapping()
cmd.show("spheres", f"({{OBJECT_NAME}}) and resn {{POINT_LABEL}} and name {{POINT_LABEL}}")
cmd.set("sphere_scale", 0.3, f"({{OBJECT_NAME}}) and resn {{POINT_LABEL}} and name {{POINT_LABEL}}")
cmd.color("yellow", f"({{OBJECT_NAME}}) and resn {{POINT_LABEL}} and name {{POINT_LABEL}}")
reload_clearance_spheres(enabled=True)
if OPENGL_AVAILABLE:
    reload_clearance_circles(enabled=False)
else:
    print(
        "maximin-pc PyMOL overlay note: camera-facing circle overlay was not loaded "
        f"because OpenGL bindings are unavailable ({{OPENGL_IMPORT_ERROR}})."
    )
cmd.extend("maximin_pc_refresh_overlays", maximin_pc_refresh_overlays)
cmd.extend("maximin_pc_set_radius_mode", maximin_pc_set_radius_mode)
cmd.extend("maximin_pc_toggle_radius_mode", maximin_pc_toggle_radius_mode)
cmd.extend("maximin_pc_set_sphere_alpha", maximin_pc_set_sphere_alpha)
cmd.extend("maximin_pc_set_sphere_color", maximin_pc_set_sphere_color)
cmd.extend("maximin_pc_set_circle_color", maximin_pc_set_circle_color)
print("maximin-pc PyMOL commands:")
print("  maximin_pc_refresh_overlays")
print("  maximin_pc_set_radius_mode clearance")
print("  maximin_pc_set_radius_mode pair")
print("  maximin_pc_toggle_radius_mode")
print("  maximin_pc_set_sphere_alpha 0.25")
print("  maximin_pc_set_sphere_color 0.20,0.70,1.00")
print("  maximin_pc_set_circle_color 0.20,0.70,1.00")
"""
    script_path.write_text(script)


def print_tied_solutions(
    candidates: np.ndarray,
    clearance: np.ndarray,
    box: np.ndarray,
    tied_combinations: Sequence[Tuple[int, ...]],
    total_ties: int,
    threshold_cutoff: float,
    max_printed: int,
    selected_tie_index: int,
) -> None:
    if total_ties == 0:
        return

    n_stored = len(tied_combinations)
    print(f"Tied optimal solutions under the thresholded maximin score: {total_ties}")
    print(f"Selected tied solution index: {selected_tie_index}")

    if max_printed != 0 and total_ties > max_printed:
        print(
            f"Printing the first {n_stored} tied solutions due to "
            f"--max-tied={max_printed}."
        )
    elif n_stored < total_ties:
        print(f"Printing {n_stored} tied solutions (storage truncated during enumeration).")

    for sol_idx, comb in enumerate(tied_combinations, start=1):
        marker = "  [selected]" if sol_idx == selected_tie_index else ""
        sol = build_solution_from_indices(
            candidates=candidates,
            clearance=clearance,
            box=box,
            indices=comb,
            thresholded_mode=True,
            threshold_cutoff=threshold_cutoff,
        )
        print(f"Tied solution {sol_idx}:{marker}")
        print(f"  Thresholded maximin score (A): {sol.selection_pair_score:.6f}")
        if math.isfinite(sol.pair_min_distance):
            print(f"  Actual minimum point-point distance (A): {sol.pair_min_distance:.6f}")
        else:
            print("  Actual minimum point-point distance (A): inf (only one point requested)")
        print(
            "  Minimum clearance to obstacle among selected points (A): "
            f"{float(np.min(sol.point_clearances)):.6f}"
        )
        print("  index x y z clearance_to_obstacle")
        for i, (p, c) in enumerate(zip(sol.points, sol.point_clearances), start=1):
            print(f"  {i:3d} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f} {c:12.6f}")
