from __future__ import annotations

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from fast_pcc_rmsd import __version__
from fast_pcc_rmsd.cli import (
    FPCCError,
    build_index_selection,
    build_parser,
    effective_unwrap_engine,
    has_valid_unit_cell,
    parse_gromacs_index_groups,
    remap_contact_coordinates,
    resolve_output_layout,
    rmsd_without_fitting,
    select_rmsd_subset,
)
from fast_pcc_rmsd.unwrapping import ResidueUnwrapError, ResidueUnwrapPlan


def make_universe(positions: object, dimensions: object):
    universe = mda.Universe.empty(
        4,
        n_residues=2,
        atom_resindex=[0, 0, 1, 1],
        trajectory=False,
    )
    universe.add_TopologyAttr("names", ["C1", "C2", "C1", "C2"])
    universe.add_TopologyAttr("masses", [12.0, 12.0, 12.0, 12.0])
    universe.add_TopologyAttr("resnames", ["ALA", "GLY"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.add_TopologyAttr("segids", ["A"])
    universe.add_bonds([(0, 1), (2, 3)])
    universe.load_new(
        np.asarray(positions, dtype=np.float32),
        format=MemoryReader,
        dimensions=np.asarray(dimensions, dtype=np.float32),
    )
    return universe


def test_optimized_unwrap_handles_boundary_crossing_and_invalid_box():
    universe = make_universe(
        [[[9.5, 1.0, 1.0], [0.2, 1.0, 1.0], [2.0, 2.0, 2.0], [2.5, 2.0, 2.0]]],
        [[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]],
    )
    plan = ResidueUnwrapPlan.from_atom_group(universe.atoms)
    unwrapped = plan.unwrap_numpy(universe.dimensions, 0)
    np.testing.assert_allclose(unwrapped[1], [10.2, 1.0, 1.0])

    with pytest.raises(ResidueUnwrapError, match="unit cell is invalid"):
        plan.unwrap_numpy([10.0, 10.0, 0.0, 90.0, 90.0, 90.0], 4)


def test_unwrap_rejects_incomplete_and_disconnected_residues():
    universe = make_universe(
        np.zeros((1, 4, 3)),
        [[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]],
    )
    with pytest.raises(ResidueUnwrapError, match="not represented by all"):
        ResidueUnwrapPlan.from_atom_group(universe.atoms[[0, 1, 2]])

    universe.del_TopologyAttr("bonds")
    with pytest.raises(ResidueUnwrapError, match="without bonds"):
        ResidueUnwrapPlan.from_atom_group(universe.atoms)


def test_contact_remapping_and_rmsd_do_not_fit_coordinates():
    box = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    first = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    second = np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    remapped = remap_contact_coordinates(first, second, [12.0, 12.0], [12.0, 12.0], box)
    np.testing.assert_allclose(remapped[2:], [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    reference = np.zeros((2, 3))
    translated = np.ones((2, 3))
    assert rmsd_without_fitting(translated, reference) == pytest.approx(np.sqrt(3.0))


def test_hydrogen_subset_and_all_hydrogen_failure():
    coordinates = np.arange(9, dtype=float).reshape(3, 3)
    indices, positions, selected = select_rmsd_subset(
        (4, 5, 6), [12.0, 1.0, 16.0], coordinates, include_hydrogens=False
    )
    assert indices == (4, 6)
    assert positions == (0, 2)
    np.testing.assert_array_equal(selected, coordinates[[0, 2]])

    with pytest.raises(FPCCError, match="removed all atoms"):
        select_rmsd_subset((1,), [1.0], coordinates[:1], include_hydrogens=False)


def test_index_parser_and_selection_reject_bad_input(tmp_path):
    index_file = tmp_path / "contacts.ndx"
    index_file.write_text("[ pcc_demo ]\n1 2 2\n\n[ other ]\n3\n")
    assert parse_gromacs_index_groups(index_file) == [("pcc_demo", (0, 1, 1)), ("other", (2,))]
    assert build_index_selection([3, 1, 3]) == "index 1 3"

    bad_file = tmp_path / "bad.ndx"
    bad_file.write_text("[ pcc_demo ]\n1 nope\n")
    with pytest.raises(FPCCError, match="non-integer"):
        parse_gromacs_index_groups(bad_file)
    with pytest.raises(FPCCError, match="empty"):
        build_index_selection([])


def test_output_layout_and_unit_cell_validation(tmp_path):
    combined = resolve_output_layout(tmp_path / "results.XVG", False)
    assert combined.mode == "combined-xvg"
    assert combined.combined_xvg_output_path.name == "results.XVG"
    directory = resolve_output_layout(tmp_path / "contacts", True)
    assert directory.mode == "directory"
    assert directory.write_remapped_contact_trajectories
    with pytest.raises(FPCCError, match="requires -o"):
        resolve_output_layout(tmp_path / "results.xvg", True)

    assert has_valid_unit_cell([1, 2, 3, 90, 90, 90])
    assert not has_valid_unit_cell([1, 2, 3, 90, 90])
    assert not has_valid_unit_cell([1, 2, 0, 90, 90, 90])
    assert effective_unwrap_engine("cuda", "optimized") == "optimized-cuda"


def test_parser_default_matches_documented_contact_cutoff():
    args = build_parser().parse_args(["-s", __file__, "-f", __file__, "-o", "out.xvg"])
    assert args.contact_cutoff_nm == pytest.approx(0.6)


def test_parser_reports_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip().endswith(__version__)
