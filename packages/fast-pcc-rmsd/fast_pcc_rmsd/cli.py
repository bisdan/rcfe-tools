#!/usr/bin/env python3
"""Pairwise-contact RMSD analysis with MDAnalysis.

The script validates its inputs, prepares an RMSD reference, determines
pairwise contacts, prepares reference coordinates for each contact, and then
analyzes every trajectory frame without rotational fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError
import numpy as np
from tqdm import tqdm

from . import __version__
from .unwrapping import ResidueUnwrapError, ResidueUnwrapPlan


REQUIRED_TOPOLOGY_ATTRS = frozenset({"masses", "bonds"})
REFERENCE_ORDER_ATTRS = ("names", "resindices", "resnames")
MASS_ABS_TOLERANCE = 1e-6
MASS_REL_TOLERANCE = 1e-6
DEFAULT_CONTACT_CUTOFF_NM = 0.6
DEFAULT_INDEXED_INTRACHAIN_COM_CUTOFF_NM = 1.2
NM_TO_ANGSTROM = 10.0
ANGSTROM_TO_NM = 0.1
PROTEIN_SELECTION = "protein"
HYDROGEN_MASS_MAX = 1.5
LOGGER = logging.getLogger(__name__)


class FPCCError(RuntimeError):
    """Raised for user-facing CLI and validation errors."""


@dataclass(frozen=True)
class PairwiseContactSelection:
    """Atom-level selection describing a pairwise residue pair."""

    label: str
    selection: str
    atom_indices: tuple[int, ...]
    residue_pair: tuple[int, int]


@dataclass(frozen=True)
class PreparedReferenceContact:
    """Prepared reference coordinates and output path for one contact."""

    selection: PairwiseContactSelection
    full_reference_coordinates: object
    first_residue_atom_count: int
    first_residue_masses: object
    second_residue_masses: object
    reference_first_residue_com: object
    rmsd_atom_indices: tuple[int, ...]
    rmsd_coordinate_positions: tuple[int, ...]
    rmsd_reference_coordinates: object
    gro_output_path: Path
    xtc_output_path: Path
    xvg_output_path: Path


@dataclass(frozen=True)
class PreparedContactBatch:
    """Batch of contacts with a shared coordinate layout for vectorized analysis."""

    contact_indices: tuple[int, ...]
    first_atom_indices: object
    second_atom_indices: object
    first_residue_masses: object
    second_residue_masses: object
    first_mass_sums: object
    second_mass_sums: object
    reference_first_residue_coms: object
    rmsd_coordinate_positions: object
    reference_rmsd_coordinates: object


@dataclass(frozen=True)
class OutputLayout:
    """Resolved output mode and paths for one CLI run."""

    mode: str
    output_path: Path
    contact_file_dir: Path
    combined_xvg_output_path: Path | None
    write_prepared_contact_gro_files: bool
    write_per_contact_xvg_files: bool
    write_remapped_contact_trajectories: bool


def existing_file(path_str: str) -> Path:
    """Argparse type that resolves a file path and ensures it exists."""

    path = Path(path_str).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def path_argument(path_str: str) -> Path:
    """Argparse type that resolves a path without checking it exists."""

    return Path(path_str).expanduser()


def configure_logging(verbose: bool) -> None:
    """Configure CLI logging."""

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for the fpcc script."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate a topology/trajectory pair, prepare pairwise-contact "
            "reference structures, and calculate per-contact RMSD values "
            "across the trajectory. The topology must explicitly contain "
            "masses and bonds, and the trajectory must contain instantaneous "
            "unit cell parameters."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-s",
        "--topology",
        required=True,
        type=existing_file,
        help=(
            "Topology file to load. Only formats whose parser reads both masses "
            "and bonds are accepted."
        ),
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        required=True,
        type=existing_file,
        help=(
            "Trajectory file to load alongside the topology. The trajectory "
            "must provide per-frame unit cell parameters."
        ),
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=existing_file,
        help=(
            "Optional reference file to use later as the RMSD reference "
            "structure. If omitted, the script will try to use the topology "
            "file itself as the reference source. Any reference source must "
            "contain coordinates and unit cell parameters. A separate "
            "reference must expose masses and bonds unless "
            "--transfer-topology-to-reference is used. A separate reference "
            "must still be topology-equivalent to the primary system."
        ),
    )
    parser.add_argument(
        "--transfer-topology-to-reference",
        action="store_true",
        help=(
            "Only valid together with --reference. First validate that the "
            "primary system and reference contain the same atoms in the same "
            "order, using atom names plus residue indices and names. If that "
            "passes, copy masses and bonds from the primary topology into the "
            "reference before continuing."
        ),
    )
    parser.add_argument(
        "--contact-index",
        type=existing_file,
        help=(
            "Optional Gromacs index file defining pairwise contacts. Relevant "
            "groups must be named with the prefix 'pcc_'. If omitted, "
            "pairwise inter-chain contacts are determined automatically from "
            "the unwrapped protein reference."
        ),
    )
    parser.add_argument(
        "--system-selection-index-group",
        help=(
            "Optional exact name of a Gromacs index group in --contact-index. "
            "If provided, that group's atoms are converted into an MDAnalysis "
            "selection and used instead of the default 'protein' selection "
            "during system preparation."
        ),
    )
    parser.add_argument(
        "--indexed-intra-chain-com-cutoff-nm",
        type=float,
        default=DEFAULT_INDEXED_INTRACHAIN_COM_CUTOFF_NM,
        help=(
            "When --contact-index is provided, accept an otherwise intra-chain "
            "pcc_ residue pair only if the residue COM distance in the "
            "unwrapped reference exceeds this cutoff in nm "
            f"(default: {DEFAULT_INDEXED_INTRACHAIN_COM_CUTOFF_NM})."
        ),
    )
    parser.add_argument(
        "--contact-cutoff-nm",
        type=float,
        default=DEFAULT_CONTACT_CUTOFF_NM,
        help=(
            "Distance cutoff in nm for automatically determining pairwise "
            "inter-chain contacts from the unwrapped protein reference "
            f"(default: {DEFAULT_CONTACT_CUTOFF_NM})."
        ),
    )
    parser.add_argument(
        "--exclude-hydrogens-from-contact-search",
        action="store_true",
        help=(
            "Exclude hydrogen atoms from the automatic pairwise contact search "
            "by using only atoms with mass greater than "
            f"{HYDROGEN_MASS_MAX:.1f} Da."
        ),
    )
    parser.add_argument(
        "--include-hydrogens-in-rmsd",
        action="store_true",
        help=(
            "Include hydrogens in the prepared atom subsets for later RMSD "
            "calculations. By default, hydrogens are excluded from RMSD."
        ),
    )
    parser.add_argument(
        "--write-remapped-contact-trajectories",
        action="store_true",
        help=(
            "Write remapped per-contact .xtc trajectories during analysis. "
            "Only valid when -o points to a directory. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm progress bar during trajectory analysis.",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Numerical backend for trajectory contact analysis (default: cpu). "
            "The optional cuda backend requires CuPy and an NVIDIA GPU."
        ),
    )
    parser.add_argument(
        "--unwrap-backend",
        choices=("optimized", "mdanalysis"),
        default="optimized",
        help=(
            "Residue-unwrapping implementation for every trajectory frame "
            "(default: optimized). The optimized implementation follows the "
            "selected compute backend; mdanalysis uses the public MDAnalysis "
            "residue unwrap operation as a compatibility path."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable info-level logging for the run summary and status output.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_path",
        required=True,
        type=path_argument,
        help=(
            "Output target. If it ends with '.xvg', write one combined RMSD "
            "table there with frame in the first column and one RMSD column "
            "per pairwise contact. Otherwise treat it as an output directory "
            "and write per-contact .xvg files plus prepared .gro references "
            "there, with optional per-contact .xtc trajectories."
        ),
    )
    return parser


def topology_attr_names(attributes: Iterable[object]) -> set[str]:
    """Normalize MDAnalysis topology attribute objects to their attr names."""

    names: set[str] = set()
    for attribute in attributes:
        attr_name = getattr(attribute, "attrname", None)
        if isinstance(attr_name, str):
            names.add(attr_name)
            continue
        if isinstance(attribute, str):
            names.add(attribute)
    return names


def topology_attribute_status(universe: object) -> tuple[set[str], set[str]]:
    """Return the topology attributes that were read and guessed."""

    topology = getattr(universe, "_topology", None)
    if topology is None:
        raise FPCCError(
            "The loaded universe does not expose topology metadata needed for "
            "topology validation."
        )

    if not hasattr(topology, "read_attributes"):
        raise FPCCError(
            "This MDAnalysis version does not expose the topology metadata "
            "needed to verify that masses and bonds came from the topology file."
        )

    read_attributes = topology_attr_names(getattr(topology, "read_attributes", ()))
    guessed_attributes = topology_attr_names(getattr(topology, "guessed_attributes", ()))
    return read_attributes, guessed_attributes


def validate_topology_data_access(
    universe: object,
    source_path: Path,
    no_data_error: type[Exception],
    source_label: str,
) -> None:
    """Ensure masses and bonds are accessible on the loaded universe."""

    try:
        masses = universe.atoms.masses
    except (AttributeError, no_data_error) as exc:
        raise FPCCError(
            f"{source_label} '{source_path}' does not expose atom masses after loading."
        ) from exc

    if len(masses) != universe.atoms.n_atoms:
        raise FPCCError(
            f"{source_label} '{source_path}' returned {len(masses)} masses for "
            f"{universe.atoms.n_atoms} atoms."
        )

    try:
        bonds = universe.bonds
    except (AttributeError, no_data_error) as exc:
        raise FPCCError(
            f"{source_label} '{source_path}' does not expose bond information after loading."
        ) from exc

    len(bonds)


def validate_supported_topology(universe: object, topology_path: Path, no_data_error: type[Exception]) -> None:
    """Ensure the topology explicitly provided masses and bonds."""

    read_attributes, guessed_attributes = topology_attribute_status(universe)

    missing_attributes = REQUIRED_TOPOLOGY_ATTRS - read_attributes
    guessed_required_attributes = REQUIRED_TOPOLOGY_ATTRS & guessed_attributes

    if missing_attributes:
        missing_text = ", ".join(sorted(missing_attributes))
        raise FPCCError(
            f"Unsupported topology '{topology_path}': MDAnalysis did not read the "
            f"required topology attributes {missing_text}. Use a topology format "
            "that explicitly stores both masses and bonds."
        )

    if guessed_required_attributes:
        guessed_text = ", ".join(sorted(guessed_required_attributes))
        raise FPCCError(
            f"Unsupported topology '{topology_path}': MDAnalysis guessed {guessed_text} "
            "instead of reading them from the topology file. Use a topology format "
            "that explicitly stores both masses and bonds."
        )

    validate_topology_data_access(universe, topology_path, no_data_error, source_label="Topology")


def has_valid_unit_cell(dimensions: object) -> bool:
    """Return True when MDAnalysis exposes usable unit cell dimensions."""

    if dimensions is None:
        return False

    try:
        values = tuple(float(value) for value in dimensions)
    except (TypeError, ValueError):
        return False

    if len(values) != 6:
        return False

    if not all(math.isfinite(value) for value in values):
        return False

    lengths = values[:3]
    angles = values[3:]

    if any(length <= 0.0 for length in lengths):
        return False

    if any(angle <= 0.0 for angle in angles):
        return False

    if all(abs(value) <= 1e-12 for value in values):
        return False

    return True


def validate_trajectory_unit_cell(
    universe: object,
    source_path: Path,
    source_label: str = "trajectory",
) -> None:
    """Ensure the current frame exposes instantaneous unit cell parameters."""

    trajectory = getattr(universe, "trajectory", None)
    if trajectory is None:
        raise FPCCError(
            f"{source_label.capitalize()} '{source_path}' was not attached to the "
            "loaded universe."
        )

    timestep = getattr(trajectory, "ts", None)
    if timestep is None:
        raise FPCCError(
            f"{source_label.capitalize()} '{source_path}' does not expose a "
            "current timestep."
        )

    dimensions = getattr(timestep, "dimensions", None)
    if not has_valid_unit_cell(dimensions):
        raise FPCCError(
            f"Unsupported {source_label} '{source_path}': the loaded current frame "
            "does not provide instantaneous unit cell parameters."
        )


def validate_reference_coordinates(
    universe: object,
    reference_path: Path,
    no_data_error: type[Exception],
) -> None:
    """Ensure the reference universe exposes coordinates for the current frame."""

    try:
        positions = universe.atoms.positions
    except (AttributeError, no_data_error) as exc:
        raise FPCCError(
            f"Reference source '{reference_path}' does not expose coordinates "
            "for the reference frame."
        ) from exc

    if len(positions) != universe.atoms.n_atoms:
        raise FPCCError(
            f"Reference source '{reference_path}' returned coordinates for "
            f"{len(positions)} atoms, but the universe contains "
            f"{universe.atoms.n_atoms} atoms."
        )


def protein_grouping_mode(protein: object) -> str:
    """Choose whether protein partitions should follow chain IDs or segments."""

    try:
        chain_ids = tuple(str(value).strip() for value in protein.chainIDs)
    except AttributeError:
        chain_ids = ()

    if chain_ids and all(chain_ids) and len(set(chain_ids)) > 1:
        return "chain"
    return "segment"


def atom_group_partition_labels(atom_group: object, grouping_mode: str) -> tuple[str, ...]:
    """Return per-atom chain or segment labels for a protein atom group."""

    if grouping_mode == "chain":
        try:
            labels = tuple(str(value).strip() for value in atom_group.chainIDs)
        except AttributeError as exc:
            raise FPCCError(
                "The loaded protein system does not expose chain IDs needed for "
                "protein chain handling."
            ) from exc

        if any(not label for label in labels):
            raise FPCCError(
                "The loaded protein system contains empty chain IDs, so protein "
                "chains cannot be compared safely."
            )
        return labels

    segids = tuple(str(value).strip() for value in atom_group.segids)
    segindices = tuple(int(value) for value in atom_group.segindices)
    return tuple(
        segid if segid else f"segindex:{segindex}"
        for segid, segindex in zip(segids, segindices)
    )


def residue_partition_label(
    residue: object,
    grouping_mode: str,
    source_path: Path,
    source_label: str,
) -> str:
    """Return the unique chain or segment label for a residue."""

    labels = set(atom_group_partition_labels(residue.atoms, grouping_mode))
    if len(labels) != 1:
        label_kind = "chain IDs" if grouping_mode == "chain" else "segments"
        raise FPCCError(
            f"{source_label.capitalize()} '{source_path}' contains a protein residue "
            f"that spans multiple {label_kind}, which is not supported."
        )
    return next(iter(labels))


def validate_identical_protein_sequences(
    protein: object,
    source_path: Path,
    source_label: str,
) -> None:
    """Require all protein chains or segments to share the same residue sequence."""

    grouping_mode = protein_grouping_mode(protein)
    sequences_by_group: dict[str, list[str]] = {}

    for residue in protein.residues:
        group_label = residue_partition_label(
            residue,
            grouping_mode,
            source_path,
            source_label,
        )
        sequences_by_group.setdefault(group_label, []).append(str(residue.resname))

    if len(sequences_by_group) <= 1:
        return

    group_kind = "chains" if grouping_mode == "chain" else "segments"
    sequence_items = list(sequences_by_group.items())
    reference_group, reference_sequence = sequence_items[0]

    for group_label, sequence in sequence_items[1:]:
        if len(sequence) != len(reference_sequence):
            raise FPCCError(
                f"{source_label.capitalize()} '{source_path}' contains multiple "
                f"protein {group_kind}, but '{group_label}' has {len(sequence)} "
                f"residues while '{reference_group}' has {len(reference_sequence)}. "
                "All protein chains or segments must have the exact same amino acid "
                "sequence."
            )

        for residue_index, (resname, reference_resname) in enumerate(
            zip(sequence, reference_sequence),
            start=1,
        ):
            if resname != reference_resname:
                raise FPCCError(
                    f"{source_label.capitalize()} '{source_path}' contains multiple "
                    f"protein {group_kind}, but '{group_label}' differs from "
                    f"'{reference_group}' at sequence position {residue_index} "
                    f"({resname} vs {reference_resname}). All protein chains or "
                    "segments must have the exact same amino acid sequence."
                )


def prepare_protein_selection(
    universe: object,
    source_path: Path,
    source_label: str,
    selection_text: str = PROTEIN_SELECTION,
    apply_full_unwrap: bool = True,
):
    """Restrict subsequent analysis to the requested selection and optionally unwrap it."""

    protein = universe.select_atoms(selection_text)
    if protein.n_atoms == 0:
        raise FPCCError(
            f"{source_label.capitalize()} '{source_path}' does not contain any atoms "
            f"matching the selection '{selection_text}'."
        )

    validate_identical_protein_sequences(protein, source_path, source_label)

    if apply_full_unwrap:
        try:
            from MDAnalysis.transformations import unwrap
        except ImportError as exc:
            raise FPCCError(
                "MDAnalysis transformations are not available in the active environment."
            ) from exc

        universe.trajectory.add_transformations(unwrap(protein))
        universe.trajectory[universe.trajectory.frame]
    return protein


def build_index_selection(atom_indices: Iterable[int]) -> str:
    """Convert 0-based atom indices to an MDAnalysis selection string."""

    sorted_indices = sorted({int(atom_index) for atom_index in atom_indices})
    if not sorted_indices:
        raise FPCCError("Cannot build a contact selection from an empty atom index set.")
    return "index " + " ".join(str(atom_index) for atom_index in sorted_indices)


def ensure_output_directory(output_dir: Path) -> Path:
    """Create the output directory when needed and validate the result."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FPCCError(f"Could not create output directory '{output_dir}': {exc}") from exc

    if not output_dir.is_dir():
        raise FPCCError(f"Output path '{output_dir}' is not a directory.")

    return output_dir


def ensure_output_file_parent(output_path: Path) -> Path:
    """Create the parent directory for an output file when needed."""

    parent_dir = output_path.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FPCCError(
            f"Could not create parent directory for output file '{output_path}': {exc}"
        ) from exc

    if not parent_dir.is_dir():
        raise FPCCError(
            f"Parent path '{parent_dir}' for output file '{output_path}' is not a directory."
        )

    if output_path.exists() and output_path.is_dir():
        raise FPCCError(
            f"Output file path '{output_path}' is an existing directory, not an .xvg file."
        )

    return output_path


def resolve_output_layout(
    output_path: Path,
    write_remapped_contact_trajectories: bool,
) -> OutputLayout:
    """Resolve whether -o selects combined-file mode or directory mode."""

    if output_path.suffix.lower() == ".xvg":
        if write_remapped_contact_trajectories:
            raise FPCCError(
                "--write-remapped-contact-trajectories requires -o to point to a "
                "directory, not to a single .xvg file."
            )
        output_file = ensure_output_file_parent(output_path)
        return OutputLayout(
            mode="combined-xvg",
            output_path=output_file,
            contact_file_dir=output_file.parent,
            combined_xvg_output_path=output_file,
            write_prepared_contact_gro_files=False,
            write_per_contact_xvg_files=False,
            write_remapped_contact_trajectories=False,
        )

    output_dir = ensure_output_directory(output_path)
    return OutputLayout(
        mode="directory",
        output_path=output_dir,
        contact_file_dir=output_dir,
        combined_xvg_output_path=None,
        write_prepared_contact_gro_files=True,
        write_per_contact_xvg_files=True,
        write_remapped_contact_trajectories=write_remapped_contact_trajectories,
    )


def select_contact_search_atoms(
    protein: object,
    exclude_hydrogens: bool,
):
    """Choose which protein atoms participate in automatic contact searching."""

    if not exclude_hydrogens:
        return protein

    heavy_atoms = protein[protein.masses > HYDROGEN_MASS_MAX]
    if heavy_atoms.n_atoms == 0:
        raise FPCCError(
            "Hydrogen exclusion removed all atoms from the automatic pairwise "
            "contact search."
        )

    return heavy_atoms


def select_rmsd_subset(
    atom_indices: tuple[int, ...],
    atom_masses: Iterable[float],
    remapped_coordinates: object,
    include_hydrogens: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], object]:
    """Choose which remapped contact atoms will later participate in RMSD."""

    if include_hydrogens:
        return (
            atom_indices,
            tuple(range(len(atom_indices))),
            remapped_coordinates.copy(),
        )

    keep_positions = [
        position
        for mass, position in zip(atom_masses, remapped_coordinates)
        if float(mass) > HYDROGEN_MASS_MAX
    ]
    keep_indices = tuple(
        int(atom_index)
        for atom_index, mass in zip(atom_indices, atom_masses)
        if float(mass) > HYDROGEN_MASS_MAX
    )
    keep_coordinate_positions = tuple(
        coordinate_index
        for coordinate_index, mass in enumerate(atom_masses)
        if float(mass) > HYDROGEN_MASS_MAX
    )
    if not keep_indices:
        raise FPCCError(
            "Hydrogen exclusion removed all atoms from a prepared pairwise-contact "
            "RMSD subset."
        )

    return keep_indices, keep_coordinate_positions, np.asarray(keep_positions, dtype=float)


def coordinate_center_of_mass(coordinates: object, masses: object) -> object:
    """Return the mass-weighted center of mass for a coordinate array."""

    return np.average(coordinates, axis=0, weights=masses)


def residue_center_of_mass(residue: object) -> object:
    """Return the mass-weighted center of mass for a residue."""

    return coordinate_center_of_mass(
        np.asarray(residue.atoms.positions, dtype=float),
        np.asarray(residue.atoms.masses, dtype=float),
    )


def residue_com_distance_nm(
    universe: object,
    residue_pair: tuple[int, int],
) -> float:
    """Return the direct COM distance in nm for two residues."""

    first_residue, second_residue = residue_pair
    first_com = residue_center_of_mass(universe.residues[first_residue])
    second_com = residue_center_of_mass(universe.residues[second_residue])
    return float(np.linalg.norm(second_com - first_com) * ANGSTROM_TO_NM)


def residue_label(universe: object, residue_index: int) -> str:
    """Format a residue label for diagnostic output."""

    residue = universe.residues[residue_index]
    return f"{residue.segid}:{residue.resname}:{residue.resid}"


def residue_pair_label(universe: object, residue_pair: tuple[int, int]) -> str:
    """Format a residue-pair label for a pairwise contact."""

    left, right = residue_pair
    return f"{residue_label(universe, left)}__{residue_label(universe, right)}"


def contact_atom_indices_for_residue_pair(
    universe: object,
    residue_pair: tuple[int, int],
) -> tuple[int, ...]:
    """Return all atoms from the two contact residues in residue order."""

    first_residue, second_residue = residue_pair
    first_atom_indices = tuple(int(atom_index) for atom_index in universe.residues[first_residue].atoms.indices)
    second_atom_indices = tuple(int(atom_index) for atom_index in universe.residues[second_residue].atoms.indices)
    return first_atom_indices + second_atom_indices


def build_pairwise_contact_selection(
    universe: object,
    residue_pair: tuple[int, int],
    label: str,
) -> PairwiseContactSelection:
    """Build a full two-residue contact selection."""

    atom_indices = contact_atom_indices_for_residue_pair(universe, residue_pair)
    return PairwiseContactSelection(
        label=label,
        selection=build_index_selection(atom_indices),
        atom_indices=atom_indices,
        residue_pair=residue_pair,
    )


def sanitize_output_stem(label: str) -> str:
    """Convert a contact label into a filesystem-friendly stem."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return sanitized or "contact"


def contact_output_path(
    output_dir: Path,
    contact_selection: PairwiseContactSelection,
    contact_index: int,
    total_contacts: int,
) -> Path:
    """Return the output path for one prepared contact GRO file."""

    width = max(3, len(str(total_contacts)))
    filename = (
        f"contact_{contact_index:0{width}d}_"
        f"{sanitize_output_stem(contact_selection.label)}.gro"
    )
    return output_dir / filename


def analysis_trajectory_output_path(gro_output_path: Path) -> Path:
    """Return the per-contact remapped trajectory path."""

    return gro_output_path.with_suffix(".xtc")


def analysis_xvg_output_path(gro_output_path: Path) -> Path:
    """Return the per-contact RMSD time-series path."""

    return gro_output_path.with_suffix(".xvg")


def validate_frame_unit_cell(
    dimensions: object,
    source_label: str,
    frame_index: int,
) -> object:
    """Validate the current frame unit cell during analysis."""

    if not has_valid_unit_cell(dimensions):
        raise FPCCError(
            f"The {source_label} does not provide a usable unit cell at frame "
            f"{frame_index}."
        )

    return np.asarray(dimensions, dtype=float)


def nearest_image_translations(
    source_coordinates: object,
    target_coordinates: object,
    box: object,
) -> object:
    """Return lattice translations that bring targets nearest to sources."""

    from MDAnalysis.lib.distances import minimize_vectors

    source_array = np.asarray(source_coordinates, dtype=float)
    target_array = np.asarray(target_coordinates, dtype=float)
    coordinate_deltas = target_array - source_array
    minimized_deltas = minimize_vectors(coordinate_deltas, box)
    return minimized_deltas - coordinate_deltas


def nearest_image_translation(
    source_com: object,
    target_com: object,
    box: object,
) -> object:
    """Return the lattice translation that brings target nearest to source."""

    return nearest_image_translations(
        np.asarray([source_com], dtype=float),
        np.asarray([target_com], dtype=float),
        box,
    )[0]


def remap_contact_coordinates(
    first_coordinates: object,
    second_coordinates: object,
    first_masses: object,
    second_masses: object,
    box: object,
) -> object:
    """Remap the second residue to the periodic image nearest the first."""

    remapped_first = first_coordinates.copy()
    remapped_second = second_coordinates.copy()
    first_com = coordinate_center_of_mass(remapped_first, first_masses)
    second_com = coordinate_center_of_mass(remapped_second, second_masses)
    remapped_second += nearest_image_translation(first_com, second_com, box)

    return np.concatenate((remapped_first, remapped_second), axis=0)


def prepare_contact_reference_coordinates(
    reference_universe: object,
    contact_selection: PairwiseContactSelection,
):
    """Prepare the remapped reference coordinates for one pairwise contact."""

    first_residue, second_residue = contact_selection.residue_pair
    first_atoms = reference_universe.residues[first_residue].atoms
    second_atoms = reference_universe.residues[second_residue].atoms
    expected_atom_indices = (
        tuple(int(atom_index) for atom_index in first_atoms.indices)
        + tuple(int(atom_index) for atom_index in second_atoms.indices)
    )
    if contact_selection.atom_indices != expected_atom_indices:
        raise FPCCError(
            f"Contact selection '{contact_selection.label}' does not match the "
            "expected residue-ordered atom list for its residue pair."
        )

    box = validate_frame_unit_cell(
        reference_universe.trajectory.ts.dimensions,
        source_label="reference source",
        frame_index=int(reference_universe.trajectory.frame),
    )
    return remap_contact_coordinates(
        first_atoms.positions,
        second_atoms.positions,
        np.asarray(first_atoms.masses, dtype=float),
        np.asarray(second_atoms.masses, dtype=float),
        box,
    )


def write_prepared_contact_gro(
    reference_universe: object,
    prepared_contact: PreparedReferenceContact,
) -> None:
    """Write one prepared pairwise-contact reference as a GRO file."""

    contact_atoms = reference_universe.atoms[list(prepared_contact.selection.atom_indices)]
    contact_universe = mda.Merge(contact_atoms)
    contact_universe.atoms.positions = prepared_contact.full_reference_coordinates.copy()
    contact_universe.dimensions = reference_universe.trajectory.ts.dimensions.copy()
    contact_universe.atoms.write(str(prepared_contact.gro_output_path))


def rmsd_without_fitting(coordinates: object, reference_coordinates: object) -> float:
    """Return the un-fitted RMSD between two coordinate arrays in Angstrom."""

    squared_displacements = np.sum((coordinates - reference_coordinates) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_displacements)))


def write_contact_rmsd_xvg(
    prepared_contact: PreparedReferenceContact,
    frame_indices: object,
    rmsd_values_angstrom: object,
    hydrogens_in_rmsd: bool,
    compute_backend: str = "cpu",
    unwrap_engine: str = "optimized-cpu",
) -> None:
    """Write the per-frame RMSD values for one contact as a Gromacs-style XVG."""

    rmsd_values_nm = np.asarray(rmsd_values_angstrom, dtype=float) * ANGSTROM_TO_NM
    with prepared_contact.xvg_output_path.open("w", encoding="ascii") as handle:
        handle.write("# Pairwise-contact RMSD without rotational fitting.\n")
        handle.write(f"# contact {prepared_contact.selection.label}\n")
        handle.write(f"# compute_backend {compute_backend}\n")
        handle.write(f"# unwrap_backend {unwrap_engine}\n")
        handle.write("# analysis nearest-image remapping\n")
        handle.write(
            f"# rmsd_hydrogens {'included' if hydrogens_in_rmsd else 'excluded'}\n"
        )
        handle.write("@ title \"Pairwise-contact RMSD\"\n")
        handle.write(f"@ subtitle \"{prepared_contact.selection.label}\"\n")
        handle.write("@ xaxis label \"Frame\"\n")
        handle.write("@ yaxis label \"RMSD (nm)\"\n")
        handle.write("@TYPE xy\n")
        for frame_index, rmsd_value_nm in zip(frame_indices, rmsd_values_nm):
            handle.write(f"{int(frame_index)} {rmsd_value_nm:.8f}\n")


def write_combined_contact_rmsd_xvg(
    output_path: Path,
    prepared_contacts: list[PreparedReferenceContact],
    frame_indices: object,
    rmsd_values_angstrom: object,
    hydrogens_in_rmsd: bool,
    compute_backend: str = "cpu",
    unwrap_engine: str = "optimized-cpu",
) -> None:
    """Write all pairwise-contact RMSD series into one Gromacs-style XVG."""

    rmsd_values_nm = np.asarray(rmsd_values_angstrom, dtype=float).T * ANGSTROM_TO_NM
    with output_path.open("w", encoding="ascii") as handle:
        handle.write("# Pairwise-contact RMSD without rotational fitting.\n")
        handle.write(f"# compute_backend {compute_backend}\n")
        handle.write(f"# unwrap_backend {unwrap_engine}\n")
        handle.write("# analysis nearest-image remapping\n")
        handle.write(
            f"# rmsd_hydrogens {'included' if hydrogens_in_rmsd else 'excluded'}\n"
        )
        handle.write("# columns frame")
        for prepared_contact in prepared_contacts:
            handle.write(f" {prepared_contact.selection.label}")
        handle.write("\n")
        handle.write("@ title \"Pairwise-contact RMSD\"\n")
        handle.write("@ subtitle \"All contacts\"\n")
        handle.write("@ xaxis label \"Frame\"\n")
        handle.write("@ yaxis label \"RMSD (nm)\"\n")
        handle.write("@TYPE nxy\n")
        for series_index, prepared_contact in enumerate(prepared_contacts):
            label = prepared_contact.selection.label.replace("\"", "\\\"")
            handle.write(f"@ s{series_index} legend \"{label}\"\n")

        for frame_row_index, frame_index in enumerate(frame_indices):
            row_values = rmsd_values_nm[frame_row_index]
            if row_values.size == 0:
                handle.write(f"{int(frame_index)}\n")
                continue
            values_text = " ".join(f"{float(value):.8f}" for value in row_values)
            handle.write(f"{int(frame_index)} {values_text}\n")


def build_prepared_contact_batches(
    prepared_contacts: list[PreparedReferenceContact],
) -> list[PreparedContactBatch]:
    """Group prepared contacts by shared shape for vectorized frame analysis."""

    grouped_contacts: dict[
        tuple[object, ...],
        list[tuple[int, PreparedReferenceContact]],
    ] = {}
    for contact_index, prepared_contact in enumerate(prepared_contacts):
        first_atom_count = prepared_contact.first_residue_atom_count
        total_atom_count = len(prepared_contact.selection.atom_indices)
        group_key = (
            first_atom_count,
            total_atom_count,
            prepared_contact.rmsd_coordinate_positions,
        )
        grouped_contacts.setdefault(group_key, []).append((contact_index, prepared_contact))

    batches: list[PreparedContactBatch] = []
    for _, group_contacts in sorted(grouped_contacts.items()):
        contact_indices = tuple(contact_index for contact_index, _ in group_contacts)
        first_atom_indices = np.stack(
            [
                np.asarray(
                    prepared_contact.selection.atom_indices[
                        : prepared_contact.first_residue_atom_count
                    ],
                    dtype=int,
                )
                for _, prepared_contact in group_contacts
            ]
        )
        second_atom_indices = np.stack(
            [
                np.asarray(
                    prepared_contact.selection.atom_indices[
                        prepared_contact.first_residue_atom_count :
                    ],
                    dtype=int,
                )
                for _, prepared_contact in group_contacts
            ]
        )
        first_residue_masses = np.stack(
            [prepared_contact.first_residue_masses for _, prepared_contact in group_contacts]
        )
        second_residue_masses = np.stack(
            [prepared_contact.second_residue_masses for _, prepared_contact in group_contacts]
        )
        reference_first_residue_coms = np.stack(
            [prepared_contact.reference_first_residue_com for _, prepared_contact in group_contacts]
        )
        reference_rmsd_coordinates = np.stack(
            [prepared_contact.rmsd_reference_coordinates for _, prepared_contact in group_contacts]
        )
        rmsd_coordinate_positions = np.asarray(
            group_contacts[0][1].rmsd_coordinate_positions,
            dtype=int,
        )
        batches.append(
            PreparedContactBatch(
                contact_indices=contact_indices,
                first_atom_indices=first_atom_indices,
                second_atom_indices=second_atom_indices,
                first_residue_masses=first_residue_masses,
                second_residue_masses=second_residue_masses,
                first_mass_sums=first_residue_masses.sum(axis=1, keepdims=True),
                second_mass_sums=second_residue_masses.sum(axis=1, keepdims=True),
                reference_first_residue_coms=reference_first_residue_coms,
                rmsd_coordinate_positions=rmsd_coordinate_positions,
                reference_rmsd_coordinates=reference_rmsd_coordinates,
            )
        )

    return batches


def build_contact_unwrap_atom_group(
    universe: object,
    prepared_contacts: list[PreparedReferenceContact],
):
    """Return the union of contact atoms that should be made whole per frame."""

    unique_atom_indices = sorted(
        {
            int(atom_index)
            for prepared_contact in prepared_contacts
            for atom_index in prepared_contact.selection.atom_indices
        }
    )
    if not unique_atom_indices:
        raise FPCCError(
            "Cannot build a trajectory contact-unwrapping group from zero contacts."
        )
    return universe.atoms[unique_atom_indices]


def effective_unwrap_engine(compute_backend: str, unwrap_backend: str) -> str:
    """Return the concrete residue-unwrapping implementation for metadata."""

    if unwrap_backend == "mdanalysis":
        return "mdanalysis"
    if unwrap_backend == "optimized" and compute_backend in {"cpu", "cuda"}:
        return f"optimized-{compute_backend}"
    raise FPCCError(
        f"Unknown compute/unwrap backend combination '{compute_backend}/{unwrap_backend}'."
    )


def analyze_pairwise_contact_trajectories(
    universe: object,
    prepared_contacts: list[PreparedReferenceContact],
    include_hydrogens_in_rmsd: bool,
    write_per_contact_xvg_files: bool,
    write_remapped_contact_trajectories: bool,
    show_progress_bar: bool = True,
    locally_unwrap_contact_residues: bool = True,
    compute_backend: str = "cpu",
    unwrap_backend: str = "optimized",
) -> tuple[object, object]:
    """Run the main per-frame RMSD analysis and return frame indices plus RMSDs."""

    unwrap_engine = effective_unwrap_engine(compute_backend, unwrap_backend)
    if compute_backend == "cuda":
        return analyze_pairwise_contact_trajectories_cuda(
            universe,
            prepared_contacts,
            include_hydrogens_in_rmsd,
            write_per_contact_xvg_files,
            write_remapped_contact_trajectories,
            show_progress_bar=show_progress_bar,
            unwrap_backend=unwrap_backend,
        )
    if compute_backend != "cpu":
        raise FPCCError(f"Unknown trajectory compute backend '{compute_backend}'.")

    n_frames = universe.trajectory.n_frames
    frame_indices = np.zeros(n_frames, dtype=int)
    rmsd_values = np.zeros((len(prepared_contacts), n_frames), dtype=float)
    if not prepared_contacts:
        for frame_array_index, ts in enumerate(universe.trajectory):
            frame_indices[frame_array_index] = int(ts.frame)
        return frame_indices, rmsd_values

    prepared_batches = build_prepared_contact_batches(prepared_contacts)
    contact_unwrap_atoms = (
        build_contact_unwrap_atom_group(universe, prepared_contacts)
        if locally_unwrap_contact_residues
        else None
    )
    try:
        contact_unwrap_plan = (
            ResidueUnwrapPlan.from_atom_group(contact_unwrap_atoms)
            if contact_unwrap_atoms is not None
            else None
        )
    except ResidueUnwrapError as exc:
        raise FPCCError(str(exc)) from exc
    writer_entries: list[tuple[object, object] | None] = [None] * len(prepared_contacts)

    try:
        if write_remapped_contact_trajectories:
            for contact_index, prepared_contact in enumerate(prepared_contacts):
                contact_atoms = universe.atoms[list(prepared_contact.selection.atom_indices)]
                contact_universe = mda.Merge(contact_atoms)
                writer = mda.Writer(
                    str(prepared_contact.xtc_output_path),
                    n_atoms=contact_universe.atoms.n_atoms,
                )
                writer_entries[contact_index] = (contact_universe, writer)

        progress_bar = tqdm(
            universe.trajectory,
            total=n_frames,
            desc="Analyzing pairwise contacts",
            unit="frame",
            disable=not show_progress_bar,
        )
        try:
            for frame_array_index, ts in enumerate(progress_bar):
                frame_indices[frame_array_index] = int(ts.frame)
                box = validate_frame_unit_cell(
                    ts.dimensions,
                    source_label="trajectory",
                    frame_index=int(ts.frame),
                )
                if contact_unwrap_plan is not None:
                    try:
                        if unwrap_backend == "mdanalysis":
                            contact_unwrap_plan.unwrap_mdanalysis(int(ts.frame))
                        else:
                            contact_unwrap_plan.unwrap_numpy(box, int(ts.frame))
                    except ResidueUnwrapError as exc:
                        raise FPCCError(str(exc)) from exc
                all_positions = universe.atoms.positions

                for prepared_batch in prepared_batches:
                    first_coordinates = all_positions[prepared_batch.first_atom_indices]
                    second_coordinates = all_positions[prepared_batch.second_atom_indices]

                    first_coms = (
                        first_coordinates * prepared_batch.first_residue_masses[..., None]
                    ).sum(axis=1) / prepared_batch.first_mass_sums
                    # First choose the residue-1 image closest to the prepared reference.
                    first_translations = nearest_image_translations(
                        prepared_batch.reference_first_residue_coms,
                        first_coms,
                        box,
                    )
                    first_coordinates = first_coordinates + first_translations[:, None, :]
                    first_coms = first_coms + first_translations

                    second_coms = (
                        second_coordinates * prepared_batch.second_residue_masses[..., None]
                    ).sum(axis=1) / prepared_batch.second_mass_sums
                    second_translations = nearest_image_translations(
                        first_coms,
                        second_coms,
                        box,
                    )
                    second_coordinates = second_coordinates + second_translations[:, None, :]

                    remapped_coordinates = np.concatenate(
                        (first_coordinates, second_coordinates),
                        axis=1,
                    )
                    rmsd_coordinates = remapped_coordinates[
                        :,
                        prepared_batch.rmsd_coordinate_positions,
                        :,
                    ]
                    squared_displacements = np.sum(
                        (
                            rmsd_coordinates
                            - prepared_batch.reference_rmsd_coordinates
                        )
                        ** 2,
                        axis=2,
                    )
                    batch_rmsd_values = np.sqrt(np.mean(squared_displacements, axis=1))
                    rmsd_values[
                        np.asarray(prepared_batch.contact_indices, dtype=int),
                        frame_array_index,
                    ] = batch_rmsd_values

                    if write_remapped_contact_trajectories:
                        for batch_row_index, contact_index in enumerate(prepared_batch.contact_indices):
                            writer_entry = writer_entries[contact_index]
                            if writer_entry is None:
                                raise FPCCError(
                                    "Internal error: missing writer for remapped contact trajectory."
                                )
                            contact_universe, writer = writer_entry
                            contact_universe.atoms.positions = remapped_coordinates[
                                batch_row_index
                            ]
                            contact_universe.dimensions = box.copy()
                            writer.write(contact_universe.atoms)
        finally:
            progress_bar.close()
    finally:
        for writer_entry in writer_entries:
            if writer_entry is not None:
                _, writer = writer_entry
                writer.close()

    if write_per_contact_xvg_files:
        for contact_index, prepared_contact in enumerate(prepared_contacts):
            write_contact_rmsd_xvg(
                prepared_contact,
                frame_indices,
                rmsd_values[contact_index],
                hydrogens_in_rmsd=include_hydrogens_in_rmsd,
                compute_backend="cpu",
                unwrap_engine=unwrap_engine,
            )

    return frame_indices, rmsd_values


def analyze_pairwise_contact_trajectories_cuda(
    universe: object,
    prepared_contacts: list[PreparedReferenceContact],
    include_hydrogens_in_rmsd: bool,
    write_per_contact_xvg_files: bool,
    write_remapped_contact_trajectories: bool,
    show_progress_bar: bool = True,
    unwrap_backend: str = "optimized",
) -> tuple[object, object]:
    """Run the direct-RMSD frame loop as one padded CUDA contact batch."""

    from .cuda_backend import CudaBackendError, CudaContactAnalyzer

    n_frames = universe.trajectory.n_frames
    frame_indices = np.zeros(n_frames, dtype=int)
    rmsd_values = np.zeros((len(prepared_contacts), n_frames), dtype=float)
    if not prepared_contacts:
        for frame_array_index, ts in enumerate(universe.trajectory):
            frame_indices[frame_array_index] = int(ts.frame)
        return frame_indices, rmsd_values

    try:
        cuda_analyzer = CudaContactAnalyzer(
            universe,
            prepared_contacts,
            unwrap_backend=unwrap_backend,
        )
    except CudaBackendError as exc:
        raise FPCCError(str(exc)) from exc
    except Exception as exc:
        raise FPCCError(f"CUDA backend initialization failed: {exc}") from exc

    LOGGER.info("CUDA trajectory backend: %s.", cuda_analyzer.device_name)
    writer_entries: list[tuple[object, object] | None] = [None] * len(prepared_contacts)

    try:
        if write_remapped_contact_trajectories:
            for contact_index, prepared_contact in enumerate(prepared_contacts):
                contact_atoms = universe.atoms[list(prepared_contact.selection.atom_indices)]
                contact_universe = mda.Merge(contact_atoms)
                writer = mda.Writer(
                    str(prepared_contact.xtc_output_path),
                    n_atoms=contact_universe.atoms.n_atoms,
                )
                writer_entries[contact_index] = (contact_universe, writer)

        progress_bar = tqdm(
            universe.trajectory,
            total=n_frames,
            desc="Analyzing pairwise contacts (CUDA)",
            unit="frame",
            disable=not show_progress_bar,
        )
        try:
            for frame_array_index, ts in enumerate(progress_bar):
                frame_indices[frame_array_index] = int(ts.frame)
                box = validate_frame_unit_cell(
                    ts.dimensions,
                    source_label="trajectory",
                    frame_index=int(ts.frame),
                )
                try:
                    frame_rmsd_values, output_coordinates = cuda_analyzer.analyze_frame(
                        int(ts.frame),
                        box,
                        return_full_coordinates=write_remapped_contact_trajectories,
                    )
                except CudaBackendError as exc:
                    raise FPCCError(str(exc)) from exc
                except Exception as exc:
                    raise FPCCError(
                        f"CUDA contact analysis failed at frame {int(ts.frame)}: {exc}"
                    ) from exc
                rmsd_values[:, frame_array_index] = frame_rmsd_values

                if write_remapped_contact_trajectories:
                    if output_coordinates is None:
                        raise FPCCError(
                            "Internal error: CUDA did not return remapped coordinates."
                        )
                    for contact_index, prepared_contact in enumerate(prepared_contacts):
                        writer_entry = writer_entries[contact_index]
                        if writer_entry is None:
                            raise FPCCError(
                                "Internal error: missing CUDA contact trajectory writer."
                            )
                        contact_universe, writer = writer_entry
                        atom_count = len(prepared_contact.selection.atom_indices)
                        contact_universe.atoms.positions = output_coordinates[
                            contact_index,
                            :atom_count,
                        ]
                        contact_universe.dimensions = box.copy()
                        writer.write(contact_universe.atoms)
        finally:
            progress_bar.close()
    finally:
        for writer_entry in writer_entries:
            if writer_entry is not None:
                _, writer = writer_entry
                writer.close()

    if write_per_contact_xvg_files:
        for contact_index, prepared_contact in enumerate(prepared_contacts):
            write_contact_rmsd_xvg(
                prepared_contact,
                frame_indices,
                rmsd_values[contact_index],
                hydrogens_in_rmsd=include_hydrogens_in_rmsd,
                compute_backend="cuda",
                unwrap_engine=effective_unwrap_engine("cuda", unwrap_backend),
            )

    return frame_indices, rmsd_values


def prepare_pairwise_contact_references(
    reference_universe: object,
    contact_selections: list[PairwiseContactSelection],
    output_dir: Path,
    include_hydrogens_in_rmsd: bool,
    write_prepared_contact_gro_files: bool,
) -> list[PreparedReferenceContact]:
    """Prepare all pairwise-contact reference coordinate arrays and optional GRO files."""

    prepared_contacts: list[PreparedReferenceContact] = []
    total_contacts = len(contact_selections)
    for contact_index, contact_selection in enumerate(contact_selections, start=1):
        gro_output_path = contact_output_path(
            output_dir,
            contact_selection,
            contact_index,
            total_contacts,
        )
        full_reference_coordinates = prepare_contact_reference_coordinates(
            reference_universe,
            contact_selection,
        )
        contact_atom_group = reference_universe.atoms[list(contact_selection.atom_indices)]
        first_residue, second_residue = contact_selection.residue_pair
        first_atom_group = reference_universe.residues[first_residue].atoms
        second_atom_group = reference_universe.residues[second_residue].atoms
        first_residue_masses = np.asarray(first_atom_group.masses, dtype=float)
        second_residue_masses = np.asarray(second_atom_group.masses, dtype=float)
        rmsd_atom_indices, rmsd_coordinate_positions, rmsd_reference_coordinates = select_rmsd_subset(
            contact_selection.atom_indices,
            contact_atom_group.masses,
            full_reference_coordinates,
            include_hydrogens_in_rmsd,
        )
        prepared_contact = PreparedReferenceContact(
            selection=contact_selection,
            full_reference_coordinates=full_reference_coordinates,
            first_residue_atom_count=first_atom_group.n_atoms,
            first_residue_masses=first_residue_masses,
            second_residue_masses=second_residue_masses,
            reference_first_residue_com=coordinate_center_of_mass(
                full_reference_coordinates[: first_atom_group.n_atoms],
                first_residue_masses,
            ),
            rmsd_atom_indices=rmsd_atom_indices,
            rmsd_coordinate_positions=rmsd_coordinate_positions,
            rmsd_reference_coordinates=rmsd_reference_coordinates,
            gro_output_path=gro_output_path,
            xtc_output_path=analysis_trajectory_output_path(gro_output_path),
            xvg_output_path=analysis_xvg_output_path(gro_output_path),
        )
        if write_prepared_contact_gro_files:
            write_prepared_contact_gro(reference_universe, prepared_contact)
        prepared_contacts.append(prepared_contact)

    return prepared_contacts


def build_automatic_pairwise_contact_selections(
    topology_universe: object,
    reference_universe: object,
    reference_contact_search_atoms: object,
    cutoff_nm: float,
    system_selection_text: str,
) -> list[PairwiseContactSelection]:
    """Determine unique inter-chain residue-pair contacts from the reference."""

    from MDAnalysis.lib.distances import capped_distance

    cutoff_angstrom = cutoff_nm * NM_TO_ANGSTROM
    pairs = capped_distance(
        reference_contact_search_atoms.positions,
        reference_contact_search_atoms.positions,
        max_cutoff=cutoff_angstrom,
        min_cutoff=1e-6,
        box=reference_universe.trajectory.ts.dimensions,
        return_distances=False,
    )

    atom_indices = reference_contact_search_atoms.indices
    topology_atoms = topology_universe.atoms[atom_indices]
    residue_indices = topology_atoms.resindices
    grouping_mode = protein_grouping_mode(topology_universe.select_atoms(system_selection_text))
    partition_labels = atom_group_partition_labels(topology_atoms, grouping_mode)
    contact_residue_pairs: set[tuple[int, int]] = set()

    for atom_i, atom_j in pairs:
        residue_i = int(residue_indices[atom_i])
        residue_j = int(residue_indices[atom_j])
        if residue_i == residue_j:
            continue

        if partition_labels[atom_i] == partition_labels[atom_j]:
            continue

        pair_key = (residue_i, residue_j) if residue_i < residue_j else (residue_j, residue_i)
        contact_residue_pairs.add(pair_key)

    contacts: list[PairwiseContactSelection] = []
    for residue_pair in sorted(contact_residue_pairs):
        contacts.append(
            build_pairwise_contact_selection(
                topology_universe,
                residue_pair,
                residue_pair_label(topology_universe, residue_pair),
            )
        )

    return contacts


def parse_gromacs_index_groups(index_path: Path) -> list[tuple[str, tuple[int, ...]]]:
    """Parse a Gromacs index file into named 0-based atom index groups."""

    groups: list[tuple[str, tuple[int, ...]]] = []
    current_name: str | None = None
    current_indices: list[int] = []

    try:
        with index_path.open() as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("["):
                    if current_name is not None:
                        groups.append((current_name, tuple(current_indices)))
                    current_name = line.strip("[] ").strip()
                    current_indices = []
                    continue
                try:
                    current_indices.extend(int(token) - 1 for token in line.split())
                except ValueError as exc:
                    raise FPCCError(
                        f"Gromacs index file '{index_path}' contains a non-integer atom "
                        f"index on line {line_number}."
                    ) from exc
    except OSError as exc:
        raise FPCCError(f"Could not read Gromacs index file '{index_path}': {exc}") from exc

    if current_name is not None:
        groups.append((current_name, tuple(current_indices)))

    return groups


def resolve_system_selection(
    index_path: Path | None,
    group_name: str | None,
) -> tuple[str, str]:
    """Return the MDAnalysis selection and description used for system preparation."""

    if group_name is None:
        return PROTEIN_SELECTION, f"default selection '{PROTEIN_SELECTION}'"

    if index_path is None:
        raise FPCCError(
            "--system-selection-index-group requires --contact-index so the named "
            "group can be read from a Gromacs index file."
        )

    for current_name, atom_indices in parse_gromacs_index_groups(index_path):
        if current_name != group_name:
            continue
        if not atom_indices:
            raise FPCCError(f"Index group '{group_name}' in '{index_path}' is empty.")
        return (
            build_index_selection(atom_indices),
            f"index group '{group_name}' from '{index_path}'",
        )

    raise FPCCError(
        f"Index file '{index_path}' does not contain a group named '{group_name}'."
    )


def build_index_pairwise_contact_selections(
    topology_universe: object,
    reference_universe: object,
    reference_protein: object,
    index_path: Path,
    system_selection_text: str,
    system_selection_description: str,
    intra_chain_com_cutoff_nm: float,
) -> tuple[list[PairwiseContactSelection], int]:
    """Convert pcc_ groups from a Gromacs index file into contact selections."""

    groups = parse_gromacs_index_groups(index_path)
    pcc_groups = [(name, indices) for name, indices in groups if name.startswith("pcc_")]
    if not pcc_groups:
        raise FPCCError(
            f"Index file '{index_path}' does not contain any groups whose name starts "
            "with 'pcc_'."
        )

    protein_index_set = {int(atom_index) for atom_index in reference_protein.indices}
    max_atom_index = reference_universe.atoms.n_atoms - 1
    grouping_mode = protein_grouping_mode(topology_universe.select_atoms(system_selection_text))

    contacts: list[PairwiseContactSelection] = []
    for group_name, atom_indices in pcc_groups:
        unique_atom_indices = tuple(sorted({int(atom_index) for atom_index in atom_indices}))
        if not unique_atom_indices:
            raise FPCCError(f"Index group '{group_name}' in '{index_path}' is empty.")
        if unique_atom_indices[0] < 0 or unique_atom_indices[-1] > max_atom_index:
            raise FPCCError(
                f"Index group '{group_name}' in '{index_path}' references atoms outside "
                "the loaded reference universe."
            )
        if not set(unique_atom_indices).issubset(protein_index_set):
            raise FPCCError(
                f"Index group '{group_name}' in '{index_path}' contains atoms outside "
                f"the prepared system selection ({system_selection_description})."
            )

        topology_atoms = topology_universe.atoms[list(unique_atom_indices)]
        residue_pair = tuple(sorted({int(residue_index) for residue_index in topology_atoms.resindices}))
        partition_labels = set(atom_group_partition_labels(topology_atoms, grouping_mode))
        partition_kind = "chain" if grouping_mode == "chain" else "segment"

        if len(residue_pair) != 2:
            raise FPCCError(
                f"Index group '{group_name}' in '{index_path}' maps to {len(residue_pair)} "
                "residues instead of exactly 2."
            )
        if len(partition_labels) == 1:
            com_distance_nm = residue_com_distance_nm(reference_universe, residue_pair)
            if com_distance_nm <= intra_chain_com_cutoff_nm:
                raise FPCCError(
                    f"Index group '{group_name}' in '{index_path}' maps to an intra-"
                    f"{partition_kind} residue pair whose residue COM distance in the "
                    f"unwrapped reference is {com_distance_nm:.3f} nm, which does not "
                    "exceed --indexed-intra-chain-com-cutoff-nm "
                    f"({intra_chain_com_cutoff_nm:.3f} nm)."
                )
            LOGGER.info(
                "Accepting intra-%s index group '%s' from '%s' because the residue "
                "COM distance in the unwrapped reference is %.3f nm (> %.3f nm).",
                partition_kind,
                group_name,
                index_path,
                com_distance_nm,
                intra_chain_com_cutoff_nm,
            )
        elif len(partition_labels) != 2:
            raise FPCCError(
                f"Index group '{group_name}' in '{index_path}' spans an unsupported "
                f"number of {partition_kind} labels ({len(partition_labels)})."
            )

        contacts.append(
            build_pairwise_contact_selection(
                topology_universe,
                residue_pair,
                group_name,
            )
        )

    return contacts, len(pcc_groups)

def validate_same_atoms_in_same_order(
    universe: object,
    reference_universe: object,
    topology_path: Path,
    reference_path: Path,
    no_data_error: type[Exception],
) -> None:
    """Validate atom identity and order between primary and explicit reference."""

    if universe.atoms.n_atoms != reference_universe.atoms.n_atoms:
        raise FPCCError(
            f"Reference '{reference_path}' cannot receive topology information "
            f"from '{topology_path}': atom counts differ "
            f"({reference_universe.atoms.n_atoms} vs {universe.atoms.n_atoms})."
        )

    primary_values: dict[str, object] = {}
    reference_values: dict[str, object] = {}
    for attr_name in REFERENCE_ORDER_ATTRS:
        try:
            primary_values[attr_name] = getattr(universe.atoms, attr_name)
            reference_values[attr_name] = getattr(reference_universe.atoms, attr_name)
        except (AttributeError, no_data_error) as exc:
            raise FPCCError(
                f"Cannot compare atom order between '{topology_path}' and "
                f"'{reference_path}' because attribute '{attr_name}' is missing."
            ) from exc

    for atom_index, values in enumerate(
        zip(
            primary_values["names"],
            primary_values["resindices"],
            primary_values["resnames"],
            reference_values["names"],
            reference_values["resindices"],
            reference_values["resnames"],
        ),
        start=1,
    ):
        primary_name, primary_resindex, primary_resname, ref_name, ref_resindex, ref_resname = values
        if (
            primary_name != ref_name
            or int(primary_resindex) != int(ref_resindex)
            or primary_resname != ref_resname
        ):
            raise FPCCError(
                f"Reference '{reference_path}' cannot receive topology information "
                f"from '{topology_path}': atom {atom_index} does not match in order "
                f"(primary: {primary_resname}:{primary_resindex}:{primary_name}; "
                f"reference: {ref_resname}:{ref_resindex}:{ref_name})."
            )


def transfer_primary_topology_to_reference(
    universe: object,
    reference_universe: object,
) -> None:
    """Copy masses and bond connectivity from the primary universe to the reference."""

    read_attributes, guessed_attributes = topology_attribute_status(reference_universe)
    available_attributes = read_attributes | guessed_attributes

    if "masses" in available_attributes:
        reference_universe.atoms.masses = universe.atoms.masses.copy()
    else:
        reference_universe.add_TopologyAttr("masses", universe.atoms.masses.copy())

    if "bonds" in available_attributes:
        reference_universe.del_TopologyAttr("bonds")
    reference_universe.add_bonds(universe.bonds.to_indices())


def normalized_bond_indices(universe: object) -> set[tuple[int, int]]:
    """Return bond connectivity as an order-independent set of atom index pairs."""

    return {
        tuple(sorted((int(atom_i), int(atom_j))))
        for atom_i, atom_j in universe.bonds.to_indices()
    }


def validate_equivalent_topologies(
    universe: object,
    reference_universe: object,
    topology_path: Path,
    reference_path: Path,
) -> None:
    """Ensure the external reference topology matches the primary topology."""

    if universe.atoms.n_atoms != reference_universe.atoms.n_atoms:
        raise FPCCError(
            f"Reference topology '{reference_path}' is not equivalent to "
            f"'{topology_path}': atom counts differ "
            f"({reference_universe.atoms.n_atoms} vs {universe.atoms.n_atoms})."
        )

    for atom_index, (mass, reference_mass) in enumerate(
        zip(universe.atoms.masses, reference_universe.atoms.masses),
        start=1,
    ):
        if not math.isclose(
            float(mass),
            float(reference_mass),
            rel_tol=MASS_REL_TOLERANCE,
            abs_tol=MASS_ABS_TOLERANCE,
        ):
            raise FPCCError(
                f"Reference topology '{reference_path}' is not equivalent to "
                f"'{topology_path}': atom {atom_index} has mass "
                f"{reference_mass} in the reference and {mass} in the primary topology."
            )

    bonds = normalized_bond_indices(universe)
    reference_bonds = normalized_bond_indices(reference_universe)
    if bonds != reference_bonds:
        missing_bonds = sorted(bonds - reference_bonds)
        extra_bonds = sorted(reference_bonds - bonds)
        detail = "bond connectivity differs"
        if missing_bonds:
            detail = f"missing reference bond {missing_bonds[0]}"
        elif extra_bonds:
            detail = f"unexpected reference bond {extra_bonds[0]}"

        raise FPCCError(
            f"Reference topology '{reference_path}' is not equivalent to "
            f"'{topology_path}': {detail}."
        )


def load_universe(topology_path: Path, trajectory_path: Path):
    """Load a topology/trajectory pair with MDAnalysis and validate it."""

    try:
        universe = mda.Universe(str(topology_path), str(trajectory_path))
    except Exception as exc:
        raise FPCCError(
            "MDAnalysis could not load the requested topology/trajectory pair: "
            f"{exc}"
        ) from exc

    validate_supported_topology(universe, topology_path, NoDataError)
    validate_trajectory_unit_cell(universe, trajectory_path)
    return universe


def warn_if_multiframe_reference(universe: object, reference_path: Path) -> None:
    """Warn when more than one frame is available for the chosen reference."""

    if universe.trajectory.n_frames > 1:
        LOGGER.warning(
            "Reference source '%s' contains %s frames; only the first frame "
            "will be used as the RMSD reference.",
            reference_path,
            universe.trajectory.n_frames,
        )


def load_reference_universe(
    reference_path: Path,
    primary_universe: object,
    topology_path: Path,
    transfer_topology_to_reference: bool,
):
    """Load and validate an RMSD reference universe from a separate file."""

    try:
        reference_universe = mda.Universe(str(reference_path))
    except Exception as exc:
        raise FPCCError(
            f"MDAnalysis could not load the requested RMSD reference '{reference_path}': "
            f"{exc}"
        ) from exc

    if transfer_topology_to_reference:
        validate_same_atoms_in_same_order(
            primary_universe,
            reference_universe,
            topology_path,
            reference_path,
            NoDataError,
        )
        transfer_primary_topology_to_reference(primary_universe, reference_universe)
    else:
        validate_topology_data_access(
            reference_universe,
            reference_path,
            NoDataError,
            source_label="Reference source",
        )

    validate_reference_coordinates(reference_universe, reference_path, NoDataError)
    validate_trajectory_unit_cell(
        reference_universe,
        reference_path,
        source_label="reference source",
    )
    warn_if_multiframe_reference(reference_universe, reference_path)
    reference_universe.trajectory[0]
    return reference_universe


def infer_reference_from_topology(topology_path: Path):
    """Build the default RMSD reference from the topology file itself."""

    try:
        universe = mda.Universe(str(topology_path))
    except Exception as exc:
        raise FPCCError(
            f"No separate RMSD reference was provided, and topology '{topology_path}' "
            "cannot be loaded as a standalone reference source. The topology must "
            "contain reference coordinates and unit cell parameters if "
            "`--reference` is omitted. "
            f"MDAnalysis raised: {exc}"
        ) from exc

    validate_supported_topology(universe, topology_path, NoDataError)
    validate_reference_coordinates(universe, topology_path, NoDataError)
    validate_trajectory_unit_cell(universe, topology_path, source_label="reference source")
    warn_if_multiframe_reference(universe, topology_path)

    universe.trajectory[0]
    return universe


def resolve_reference_source(
    args: argparse.Namespace,
    primary_universe: object,
) -> tuple[object, Path, str]:
    """Load the chosen reference source and return its universe, path, and mode."""

    if args.reference is not None:
        reference_universe = load_reference_universe(
            args.reference,
            primary_universe,
            args.topology,
            args.transfer_topology_to_reference,
        )
        validate_equivalent_topologies(
            primary_universe,
            reference_universe,
            args.topology,
            args.reference,
        )
        if args.transfer_topology_to_reference:
            reference_mode = "external+transferred-topology"
        else:
            reference_mode = "external"
        return reference_universe, args.reference, reference_mode

    reference_universe = infer_reference_from_topology(args.topology)
    return reference_universe, args.topology, "topology-derived"


def determine_pairwise_contacts(
    args: argparse.Namespace,
    trajectory_universe: object,
    reference_universe: object,
    reference_protein: object,
    reference_contact_search_atoms: object,
    system_selection_text: str,
    system_selection_description: str,
) -> tuple[list[PairwiseContactSelection], list[PairwiseContactSelection], int | None, str]:
    """Determine automatic and active pairwise-contact selections for the run."""

    if args.transfer_topology_to_reference and args.reference is not None:
        contact_topology_universe = trajectory_universe
    else:
        contact_topology_universe = reference_universe

    automatic_contacts = build_automatic_pairwise_contact_selections(
        contact_topology_universe,
        reference_universe,
        reference_contact_search_atoms,
        args.contact_cutoff_nm,
        system_selection_text,
    )

    if args.contact_index is not None:
        contact_selections, raw_index_group_count = build_index_pairwise_contact_selections(
            contact_topology_universe,
            reference_universe,
            reference_protein,
            args.contact_index,
            system_selection_text,
            system_selection_description,
            args.indexed_intra_chain_com_cutoff_nm,
        )
        contact_source = "gromacs-index"
    else:
        contact_selections = automatic_contacts
        raw_index_group_count = None
        contact_source = "automatic"

    return contact_selections, automatic_contacts, raw_index_group_count, contact_source


def rmsd_atom_count_text(prepared_contacts: list[PreparedReferenceContact]) -> str:
    """Return a compact text summary of the prepared RMSD subset sizes."""

    rmsd_atom_counts = [
        len(prepared_contact.rmsd_atom_indices) for prepared_contact in prepared_contacts
    ]
    if not rmsd_atom_counts:
        return "0 contacts"

    min_rmsd_atoms = min(rmsd_atom_counts)
    max_rmsd_atoms = max(rmsd_atom_counts)
    if min_rmsd_atoms == max_rmsd_atoms:
        return f"{min_rmsd_atoms} atoms/contact"
    return f"{min_rmsd_atoms}-{max_rmsd_atoms} atoms/contact"


def log_run_summary(
    args: argparse.Namespace,
    output_layout: OutputLayout,
    universe: object,
    reference_universe: object,
    reference_source: Path,
    reference_mode: str,
    system_selection_description: str,
    trajectory_protein: object,
    reference_protein: object,
    reference_contact_search_atoms: object,
    contact_source: str,
    contact_selections: list[PairwiseContactSelection],
    automatic_contacts: list[PairwiseContactSelection],
    raw_index_group_count: int | None,
    prepared_contacts: list[PreparedReferenceContact],
) -> None:
    """Log a compact post-run summary for the CLI."""

    LOGGER.info(
        "Loaded universe "
        f"({universe.atoms.n_atoms} atoms, {len(universe.bonds)} bonds, "
        f"{universe.trajectory.n_frames} frames)."
    )
    LOGGER.info(
        f"RMSD reference [{reference_mode}] "
        f"({reference_universe.atoms.n_atoms} atoms, "
        f"{len(reference_universe.bonds)} bonds, "
        f"{reference_universe.trajectory.n_frames} frames): "
        f"{reference_source}"
    )
    LOGGER.info("System preparation selection: %s.", system_selection_description)
    LOGGER.info(
        "Prepared analysis systems "
        f"({trajectory_protein.n_atoms} trajectory atoms, "
        f"{reference_protein.n_atoms} reference atoms)."
    )
    LOGGER.info(
        "Automatic contact-search atoms "
        f"({reference_contact_search_atoms.n_atoms} reference atoms, "
        f"{'hydrogens excluded' if args.exclude_hydrogens_from_contact_search else 'hydrogens included'})."
    )
    LOGGER.info(
        "Pairwise contacts [%s] (%s selections).",
        contact_source,
        len(contact_selections),
    )
    LOGGER.info("Trajectory compute backend: %s.", args.compute_backend)
    LOGGER.info(
        "Trajectory residue-unwrapping backend: %s.",
        effective_unwrap_engine(args.compute_backend, args.unwrap_backend),
    )
    LOGGER.info(
        "Output mode [%s]: %s",
        output_layout.mode,
        output_layout.output_path,
    )
    if output_layout.mode == "directory":
        LOGGER.info(
            "Prepared pairwise-contact reference files (%s GRO files): %s",
            len(prepared_contacts),
            output_layout.output_path,
        )
        LOGGER.info(
            "Pairwise-contact analysis outputs (%s XTC trajectories, %s XVG files).",
            len(prepared_contacts) if output_layout.write_remapped_contact_trajectories else 0,
            len(prepared_contacts),
        )
    else:
        LOGGER.info("Prepared pairwise-contact reference files (0 GRO files).")
        LOGGER.info("Pairwise-contact analysis outputs (0 XTC trajectories, 1 XVG file).")
    LOGGER.info(
        "Prepared pairwise-contact RMSD subsets "
        f"({rmsd_atom_count_text(prepared_contacts)}, "
        f"{'hydrogens included' if args.include_hydrogens_in_rmsd else 'hydrogens excluded by default'})."
    )
    LOGGER.info(
        "Automatic inter-chain contacts from the unwrapped reference "
        f"({args.contact_cutoff_nm:.3f} nm): {len(automatic_contacts)} selections."
    )
    LOGGER.info(
        "Note: automatic pairwise contact detection does not distinguish "
        "biological interfaces from crystal contact interfaces."
    )
    if raw_index_group_count is not None:
        LOGGER.info(
            f"Index groups starting with 'pcc_' in {args.contact_index}: "
            f"{raw_index_group_count}."
        )
        LOGGER.info(
            "Indexed intra-chain periodic-image exception "
            f"(accept if residue COM distance in unwrapped reference > "
            f"{args.indexed_intra_chain_com_cutoff_nm:.3f} nm)."
        )
        LOGGER.info(
            "Automatic-vs-index comparison "
            f"({len(automatic_contacts)} automatic vs {raw_index_group_count} pcc_ groups)."
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.contact_cutoff_nm <= 0.0:
        parser.error("--contact-cutoff-nm must be positive.")
    if args.indexed_intra_chain_com_cutoff_nm <= 0.0:
        parser.error("--indexed-intra-chain-com-cutoff-nm must be positive.")
    if args.transfer_topology_to_reference and args.reference is None:
        parser.error("--transfer-topology-to-reference requires --reference.")
    if args.system_selection_index_group is not None and args.contact_index is None:
        parser.error("--system-selection-index-group requires --contact-index.")
    try:
        system_selection_text, system_selection_description = resolve_system_selection(
            args.contact_index,
            args.system_selection_index_group,
        )
        output_layout = resolve_output_layout(
            args.output_path,
            args.write_remapped_contact_trajectories,
        )
        universe = load_universe(args.topology, args.trajectory)
    except FPCCError as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        reference_universe, reference_source, reference_mode = resolve_reference_source(
            args,
            universe,
        )
    except FPCCError as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        trajectory_protein = prepare_protein_selection(
            universe,
            args.trajectory,
            source_label="trajectory",
            selection_text=system_selection_text,
            apply_full_unwrap=False,
        )
        reference_protein = prepare_protein_selection(
            reference_universe,
            reference_source,
            source_label="reference source",
            selection_text=system_selection_text,
        )
        reference_contact_search_atoms = select_contact_search_atoms(
            reference_protein,
            args.exclude_hydrogens_from_contact_search,
        )
        (
            contact_selections,
            automatic_contacts,
            raw_index_group_count,
            contact_source,
        ) = determine_pairwise_contacts(
            args,
            universe,
            reference_universe,
            reference_protein,
            reference_contact_search_atoms,
            system_selection_text,
            system_selection_description,
        )
        prepared_contacts = prepare_pairwise_contact_references(
            reference_universe,
            contact_selections,
            output_layout.contact_file_dir,
            args.include_hydrogens_in_rmsd,
            output_layout.write_prepared_contact_gro_files,
        )
        frame_indices, rmsd_values = analyze_pairwise_contact_trajectories(
            universe,
            prepared_contacts,
            include_hydrogens_in_rmsd=args.include_hydrogens_in_rmsd,
            write_per_contact_xvg_files=output_layout.write_per_contact_xvg_files,
            write_remapped_contact_trajectories=output_layout.write_remapped_contact_trajectories,
            show_progress_bar=not args.no_progress_bar,
            compute_backend=args.compute_backend,
            unwrap_backend=args.unwrap_backend,
        )
        if output_layout.combined_xvg_output_path is not None:
            write_combined_contact_rmsd_xvg(
                output_layout.combined_xvg_output_path,
                prepared_contacts,
                frame_indices,
                rmsd_values,
                hydrogens_in_rmsd=args.include_hydrogens_in_rmsd,
                compute_backend=args.compute_backend,
                unwrap_engine=effective_unwrap_engine(
                    args.compute_backend,
                    args.unwrap_backend,
                ),
            )
    except FPCCError as exc:
        LOGGER.error("%s", exc)
        return 1

    log_run_summary(
        args,
        output_layout,
        universe,
        reference_universe,
        reference_source,
        reference_mode,
        system_selection_description,
        trajectory_protein,
        reference_protein,
        reference_contact_search_atoms,
        contact_source,
        contact_selections,
        automatic_contacts,
        raw_index_group_count,
        prepared_contacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
