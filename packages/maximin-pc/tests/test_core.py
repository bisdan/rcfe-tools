from __future__ import annotations

import warnings

import numpy as np
import pytest

from maximin_pc import __version__
from maximin_pc.cli import build_parser, parse_args
from maximin_pc.core import (
    Solution,
    generate_grid_candidates,
    generate_random_candidates,
    pairwise_distances,
    refine_sampled_solution,
    solve_max_pair_distance,
)


BOX = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
CELL = np.diag([10.0, 10.0, 10.0])


def test_parser_reports_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip().endswith(__version__)


def test_search_requires_point_count():
    with pytest.raises(SystemExit):
        parse_args(["-t", "input.gro"])


def test_candidate_generators_are_bounded_and_reproducible():
    first = generate_random_candidates(CELL, 8, np.random.default_rng(7))
    second = generate_random_candidates(CELL, 8, np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    assert np.all((first >= 0.0) & (first < 10.0))

    grid = generate_grid_candidates(CELL, 2, 2, 2)
    assert grid.shape == (8, 3)
    assert {tuple(point) for point in grid} == {
        (x, y, z) for x in (0.0, 5.0) for y in (0.0, 5.0) for z in (0.0, 5.0)
    }


def test_periodic_distance_and_deterministic_selection():
    points = np.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0], [5.0, 5.0, 5.0]])
    distances = pairwise_distances(points, BOX)
    assert distances[0, 1] == pytest.approx(1.0)

    solution, was_capped, effective_cap = solve_max_pair_distance(
        candidates=points,
        clearance=np.array([2.0, 2.0, 3.0]),
        cell=CELL,
        box=BOX,
        k=2,
        max_survivors=10,
        binary_steps=20,
    )
    assert len(solution.indices) == 2
    assert solution.pair_min_distance > 0.0
    assert not was_capped
    assert effective_cap == 10


def test_one_point_refinement_emits_no_runtime_warning(monkeypatch):
    monkeypatch.setattr("maximin_pc.core.choose_refinement_schedule", lambda *_: ((0.1, 4),))
    solution = Solution(
        indices=[0],
        points=np.array([[1.0, 1.0, 1.0]]),
        point_clearances=np.array([np.sqrt(3.0)]),
        pair_min_distance=np.inf,
        selection_pair_score=np.inf,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        refined = refine_sampled_solution(
            solution=solution,
            obstacle_positions=np.array([[0.0, 0.0, 0.0]]),
            box=BOX,
            cell=CELL,
            inv_cell=np.linalg.inv(CELL),
            min_obstacle_distance=0.0,
            max_obstacle_distance=None,
            seed=7,
        )
    assert refined.points.shape == (1, 3)
    assert np.isinf(refined.pair_min_distance)
