"""Optional CUDA acceleration for the pairwise-contact analysis loop.

The module deliberately imports CuPy only when CUDA was explicitly requested.
The normal CPU installation and execution path therefore remain independent of
CuPy and of an NVIDIA driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from .unwrapping import ResidueUnwrapError, ResidueUnwrapPlan


class CudaBackendError(RuntimeError):
    """Raised when the optional CUDA backend cannot be initialized or used."""


def load_cupy():
    """Import CuPy and verify that a usable CUDA device is available."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise CudaBackendError(
            "CUDA acceleration requires CuPy. Install the CUDA 12 build with "
            "`python -m pip install 'fast-pcc-rmsd[cuda12]'` or the CUDA 13 "
            "build with `python -m pip install 'fast-pcc-rmsd[cuda13]'`."
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise CudaBackendError("CuPy did not find an NVIDIA CUDA device.")
        # Force CUDA context creation here so setup errors are reported before
        # the trajectory loop starts.
        cp.zeros(1, dtype=cp.float32).sum().item()
    except CudaBackendError:
        raise
    except Exception as exc:
        raise CudaBackendError(
            "CuPy is installed, but CUDA initialization failed. Check that the "
            "NVIDIA driver is available and that the installed CuPy wheel matches "
            f"the CUDA major version. CuPy reported: {exc}"
        ) from exc

    return cp


def cuda_device_name(cp: object) -> str:
    """Return a human-readable name for CuPy's active CUDA device."""

    try:
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties.get("name", "unknown CUDA device")
        if isinstance(name, bytes):
            return name.decode(errors="replace")
        return str(name)
    except Exception:
        return "unknown CUDA device"


def _round_half_away_from_zero(values: object, cp: object) -> object:
    """Match the C ``round`` operation used by MDAnalysis minimum images."""

    return cp.copysign(cp.floor(cp.abs(values) + 0.5), values)


def cuda_minimize_vectors(
    vectors: object,
    dimensions: object,
    cp: object,
    triclinic_offsets: object | None = None,
) -> object:
    """Apply MDAnalysis-compatible minimum-image handling on the GPU.

    ``dimensions`` uses the MDAnalysis ``(a, b, c, alpha, beta, gamma)``
    convention. Orthogonal cells use direct component wrapping. Triclinic
    cells reproduce MDAnalysis' initial lower-triangular reduction followed by
    an exhaustive search of the 27 neighboring images.
    """

    from MDAnalysis.lib.mdamath import triclinic_vectors

    box = np.asarray(dimensions, dtype=float)
    if box.shape != (6,):
        raise CudaBackendError(
            f"CUDA minimum-image handling expected 6 unit-cell values, got {box.shape}."
        )

    if np.all(box[3:] == 90.0):
        lengths = cp.asarray(box[:3], dtype=vectors.dtype)
        scaled = vectors / lengths
        return vectors - _round_half_away_from_zero(scaled, cp) * lengths

    # MDAnalysis builds this matrix in float32 and only then promotes it to
    # the vector dtype inside ``minimize_vectors``. Keep that ordering so the
    # CUDA path has the same numerical behavior for float64 coordinates too.
    box_matrix_host = triclinic_vectors(box)
    if not np.any(box_matrix_host):
        raise CudaBackendError("CUDA minimum-image handling received an invalid unit cell.")
    box_matrix = cp.asarray(box_matrix_host, dtype=vectors.dtype)

    reduced = vectors.copy()
    shift = _round_half_away_from_zero(reduced[:, 2] / box_matrix[2, 2], cp)
    reduced -= shift[:, None] * box_matrix[2]
    shift = _round_half_away_from_zero(reduced[:, 1] / box_matrix[1, 1], cp)
    reduced -= shift[:, None] * box_matrix[1]
    shift = _round_half_away_from_zero(reduced[:, 0] / box_matrix[0, 0], cp)
    reduced -= shift[:, None] * box_matrix[0]

    if triclinic_offsets is None:
        triclinic_offsets = cp.asarray(
            tuple(product((-1.0, 0.0, 1.0), repeat=3)),
            dtype=vectors.dtype,
        )
    lattice_offsets = cp.matmul(triclinic_offsets, box_matrix)
    candidates = reduced[:, None, :] + lattice_offsets[None, :, :]
    squared_lengths = cp.sum(candidates * candidates, axis=2)
    best_images = cp.argmin(squared_lengths, axis=1)
    return candidates[cp.arange(vectors.shape[0]), best_images]


def cuda_nearest_image_translations(
    sources: object,
    targets: object,
    dimensions: object,
    cp: object,
    triclinic_offsets: object,
) -> object:
    """Return lattice translations bringing targets nearest to sources."""

    deltas = targets - sources
    minimized = cuda_minimize_vectors(deltas, dimensions, cp, triclinic_offsets)
    return minimized - deltas


def cuda_unwrap_residues(
    plan: ResidueUnwrapPlan,
    dimensions: object,
    cp: object,
    traversal_layers: object | None = None,
    triclinic_offsets: object | None = None,
) -> object:
    """Apply a cached residue bond traversal to coordinates on the GPU."""

    if traversal_layers is None:
        traversal_layers = tuple(
            (cp.asarray(parents), cp.asarray(children))
            for parents, children in plan.traversal_layers
        )
    original = cp.asarray(plan.atom_group.positions)
    unwrapped = original.copy()
    for parents, children in traversal_layers:
        bond_vectors = original[children] - unwrapped[parents]
        minimized = cuda_minimize_vectors(
            bond_vectors,
            dimensions,
            cp,
            triclinic_offsets,
        )
        unwrapped[children] = unwrapped[parents] + minimized
    return unwrapped


@dataclass(frozen=True)
class CudaContactArrays:
    """One padded, device-resident batch containing all prepared contacts."""

    full_atom_positions: object
    first_coordinate_mask: object
    second_coordinate_mask: object
    first_masses: object
    second_masses: object
    first_mass_sums: object
    second_mass_sums: object
    reference_first_coms: object
    rmsd_coordinate_positions: object
    rmsd_coordinate_mask: object
    rmsd_atom_counts: object
    reference_rmsd_coordinates: object
    triclinic_offsets: object


class CudaContactAnalyzer:
    """Analyze every prepared contact in one CUDA batch per trajectory frame."""

    def __init__(
        self,
        universe: object,
        prepared_contacts: Iterable[object],
        unwrap_backend: str = "optimized",
    ) -> None:
        self.cp = load_cupy()
        self.device_name = cuda_device_name(self.cp)
        self.prepared_contacts = tuple(prepared_contacts)
        if unwrap_backend not in {"optimized", "mdanalysis"}:
            raise CudaBackendError(
                f"Unknown residue unwrap backend '{unwrap_backend}'."
            )
        self.unwrap_backend = unwrap_backend

        unique_atom_indices = sorted(
            {
                int(atom_index)
                for contact in self.prepared_contacts
                for atom_index in contact.selection.atom_indices
            }
        )
        if not unique_atom_indices:
            raise CudaBackendError("Cannot initialize CUDA analysis with zero contacts.")
        contact_atoms = universe.atoms[unique_atom_indices]
        try:
            self.unwrap_plan = ResidueUnwrapPlan.from_atom_group(contact_atoms)
        except ResidueUnwrapError as exc:
            raise CudaBackendError(str(exc)) from exc
        self.cuda_traversal_layers = tuple(
            (self.cp.asarray(parents), self.cp.asarray(children))
            for parents, children in self.unwrap_plan.traversal_layers
        )
        self.arrays = self._build_contact_arrays(unique_atom_indices)

    def _unwrap(self, frame_index: int, dimensions: object) -> object:
        """Return device coordinates unwrapped by the selected implementation."""

        cp = self.cp
        try:
            if self.unwrap_backend == "mdanalysis":
                return cp.asarray(self.unwrap_plan.unwrap_mdanalysis(frame_index))

            return cuda_unwrap_residues(
                self.unwrap_plan,
                dimensions,
                cp,
                traversal_layers=self.cuda_traversal_layers,
                triclinic_offsets=self.arrays.triclinic_offsets,
            )
        except ResidueUnwrapError as exc:
            raise CudaBackendError(str(exc)) from exc
        except CudaBackendError:
            raise
        except Exception as exc:
            raise CudaBackendError(
                "Could not make the CUDA trajectory contact residues whole under "
                f"PBC at frame {frame_index}."
            ) from exc

    def _build_contact_arrays(self, unique_atom_indices: list[int]) -> CudaContactArrays:
        cp = self.cp
        contacts = self.prepared_contacts
        contact_count = len(contacts)
        max_total_atoms = max(len(contact.selection.atom_indices) for contact in contacts)
        max_rmsd_atoms = max(len(contact.rmsd_coordinate_positions) for contact in contacts)

        union_positions = {
            atom_index: local_index
            for local_index, atom_index in enumerate(unique_atom_indices)
        }
        full_atom_positions = np.zeros((contact_count, max_total_atoms), dtype=int)
        first_mask = np.zeros((contact_count, max_total_atoms), dtype=bool)
        second_mask = np.zeros((contact_count, max_total_atoms), dtype=bool)
        first_masses = np.zeros((contact_count, max_total_atoms), dtype=float)
        second_masses = np.zeros((contact_count, max_total_atoms), dtype=float)
        rmsd_positions = np.zeros((contact_count, max_rmsd_atoms), dtype=int)
        rmsd_mask = np.zeros((contact_count, max_rmsd_atoms), dtype=bool)
        reference_rmsd = np.zeros((contact_count, max_rmsd_atoms, 3), dtype=float)
        reference_first_coms = np.zeros((contact_count, 3), dtype=float)

        for contact_index, contact in enumerate(contacts):
            atom_indices = tuple(int(index) for index in contact.selection.atom_indices)
            total_atom_count = len(atom_indices)
            first_atom_count = int(contact.first_residue_atom_count)
            rmsd_atom_count = len(contact.rmsd_coordinate_positions)
            full_atom_positions[contact_index, :total_atom_count] = [
                union_positions[index] for index in atom_indices
            ]
            first_mask[contact_index, :first_atom_count] = True
            second_mask[contact_index, first_atom_count:total_atom_count] = True
            first_masses[contact_index, :first_atom_count] = np.asarray(
                contact.first_residue_masses,
                dtype=float,
            )
            second_masses[contact_index, first_atom_count:total_atom_count] = np.asarray(
                contact.second_residue_masses,
                dtype=float,
            )
            reference_first_coms[contact_index] = np.asarray(
                contact.reference_first_residue_com,
                dtype=float,
            )
            rmsd_positions[contact_index, :rmsd_atom_count] = np.asarray(
                contact.rmsd_coordinate_positions,
                dtype=int,
            )
            rmsd_mask[contact_index, :rmsd_atom_count] = True
            reference_rmsd[contact_index, :rmsd_atom_count] = np.asarray(
                contact.rmsd_reference_coordinates,
                dtype=float,
            )

        rmsd_counts = rmsd_mask.sum(axis=1).astype(float)

        return CudaContactArrays(
            full_atom_positions=cp.asarray(full_atom_positions),
            first_coordinate_mask=cp.asarray(first_mask),
            second_coordinate_mask=cp.asarray(second_mask),
            first_masses=cp.asarray(first_masses),
            second_masses=cp.asarray(second_masses),
            first_mass_sums=cp.asarray(first_masses.sum(axis=1)),
            second_mass_sums=cp.asarray(second_masses.sum(axis=1)),
            reference_first_coms=cp.asarray(reference_first_coms),
            rmsd_coordinate_positions=cp.asarray(rmsd_positions),
            rmsd_coordinate_mask=cp.asarray(rmsd_mask),
            rmsd_atom_counts=cp.asarray(rmsd_counts),
            reference_rmsd_coordinates=cp.asarray(reference_rmsd),
            triclinic_offsets=cp.asarray(
                tuple(product((-1.0, 0.0, 1.0), repeat=3)),
                dtype=float,
            ),
        )

    def analyze_frame(
        self,
        frame_index: int,
        dimensions: object,
        return_full_coordinates: bool,
    ) -> tuple[object, object | None]:
        """Unwrap and analyze one frame, returning host RMSDs and coordinates."""

        cp = self.cp
        arrays = self.arrays
        union_positions = self._unwrap(frame_index, dimensions)
        full_coordinates = union_positions[arrays.full_atom_positions]

        first_coms = cp.sum(
            full_coordinates * arrays.first_masses[:, :, None],
            axis=1,
        ) / arrays.first_mass_sums[:, None]
        first_translations = cuda_nearest_image_translations(
            arrays.reference_first_coms,
            first_coms,
            dimensions,
            cp,
            arrays.triclinic_offsets,
        )
        first_coms = first_coms + first_translations

        second_coms = cp.sum(
            full_coordinates * arrays.second_masses[:, :, None],
            axis=1,
        ) / arrays.second_mass_sums[:, None]
        second_translations = cuda_nearest_image_translations(
            first_coms,
            second_coms,
            dimensions,
            cp,
            arrays.triclinic_offsets,
        )
        remapped = full_coordinates.copy()
        remapped += (
            arrays.first_coordinate_mask[:, :, None] * first_translations[:, None, :]
        )
        remapped += (
            arrays.second_coordinate_mask[:, :, None] * second_translations[:, None, :]
        )

        rmsd_coordinates = cp.take_along_axis(
            remapped,
            arrays.rmsd_coordinate_positions[:, :, None],
            axis=1,
        )
        displacements = rmsd_coordinates - arrays.reference_rmsd_coordinates

        squared_displacements = cp.sum(displacements * displacements, axis=2)
        squared_displacements *= arrays.rmsd_coordinate_mask
        rmsd_values = cp.sqrt(
            cp.sum(squared_displacements, axis=1) / arrays.rmsd_atom_counts
        )
        rmsd_values_host = cp.asnumpy(rmsd_values)
        output_host = cp.asnumpy(remapped) if return_full_coordinates else None
        return rmsd_values_host, output_host
