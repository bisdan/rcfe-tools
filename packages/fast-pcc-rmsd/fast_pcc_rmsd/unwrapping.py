"""Cached, batched residue unwrapping for trajectory contact atoms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


class ResidueUnwrapError(RuntimeError):
    """Raised when a residue-unwrapping plan cannot be built or applied."""


def _validated_dimensions(dimensions: object, frame_index: int) -> np.ndarray:
    """Return a usable MDAnalysis unit cell for one trajectory frame."""

    if dimensions is None:
        raise ResidueUnwrapError(
            "Could not make the trajectory contact residues whole under PBC at "
            f"frame {frame_index}: no unit cell is available."
        )
    box = np.asarray(dimensions, dtype=float)
    if box.shape != (6,) or not np.all(np.isfinite(box)) or np.any(box[:3] <= 0.0):
        raise ResidueUnwrapError(
            "Could not make the trajectory contact residues whole under PBC at "
            f"frame {frame_index}: the unit cell is invalid."
        )
    return box


@dataclass(frozen=True)
class ResidueUnwrapPlan:
    """A coordinate-independent bond traversal for a union of whole residues."""

    atom_group: object
    traversal_layers: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def from_atom_group(cls, atom_group: object) -> "ResidueUnwrapPlan":
        """Build stable residue-local breadth-first traversals once."""

        from MDAnalysis.exceptions import NoDataError

        global_indices = np.asarray(atom_group.indices, dtype=np.int64)
        if len(np.unique(global_indices)) != len(global_indices):
            raise ResidueUnwrapError(
                "Cannot build a residue-unwrapping plan from duplicate atoms."
            )

        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(global_indices)
        }
        residue_indices = np.asarray(atom_group.resindices, dtype=np.int64)

        for residue in atom_group.residues:
            missing = [
                int(atom_index)
                for atom_index in residue.atoms.indices
                if int(atom_index) not in global_to_local
            ]
            if missing:
                raise ResidueUnwrapError(
                    "A trajectory contact residue was not represented by all of "
                    "its atoms."
                )

        adjacency: list[list[int]] = [[] for _ in global_indices]
        try:
            bond_indices = atom_group.universe.bonds.to_indices()
        except (AttributeError, NoDataError, TypeError) as exc:
            if any(len(residue.atoms) > 1 for residue in atom_group.residues):
                raise ResidueUnwrapError(
                    "Cannot build a residue-unwrapping plan without bonds."
                ) from exc
            bond_indices = np.empty((0, 2), dtype=np.int64)

        for first_global, second_global in bond_indices:
            first_local = global_to_local.get(int(first_global))
            second_local = global_to_local.get(int(second_global))
            if first_local is None or second_local is None:
                continue
            if residue_indices[first_local] != residue_indices[second_local]:
                continue
            adjacency[first_local].append(second_local)
            adjacency[second_local].append(first_local)

        layers: list[list[tuple[int, int]]] = []
        for residue in atom_group.residues:
            residue_local_indices = np.asarray(
                [global_to_local[int(index)] for index in residue.atoms.indices],
                dtype=np.int64,
            )
            root = int(residue_local_indices[0])
            visited = {root}
            queue: deque[tuple[int, int]] = deque([(root, 0)])

            while queue:
                parent, depth = queue.popleft()
                for child in sorted(adjacency[parent]):
                    if child in visited:
                        continue
                    visited.add(child)
                    while len(layers) <= depth:
                        layers.append([])
                    layers[depth].append((parent, child))
                    queue.append((child, depth + 1))

            if len(visited) != len(residue_local_indices):
                residue_label = (
                    f"{residue.segid}:{residue.resname}:{residue.resid}"
                )
                raise ResidueUnwrapError(
                    "Could not build a bond traversal for trajectory contact "
                    f"residue {residue_label}; its atoms are not connected by "
                    "intra-residue bonds."
                )

        traversal_layers = tuple(
            (
                np.asarray([parent for parent, _ in layer], dtype=np.int64),
                np.asarray([child for _, child in layer], dtype=np.int64),
            )
            for layer in layers
        )
        return cls(atom_group=atom_group, traversal_layers=traversal_layers)

    def unwrap_numpy(self, dimensions: object, frame_index: int) -> np.ndarray:
        """Unwrap all planned residues with batched NumPy operations."""

        from MDAnalysis.lib.distances import minimize_vectors

        box = _validated_dimensions(dimensions, frame_index)
        original = np.asarray(self.atom_group.positions)
        unwrapped = original.copy()
        try:
            for parents, children in self.traversal_layers:
                bond_vectors = original[children] - unwrapped[parents]
                minimized = minimize_vectors(bond_vectors, box)
                unwrapped[children] = unwrapped[parents] + minimized
            self.atom_group.positions = unwrapped
        except (AttributeError, TypeError, ValueError) as exc:
            raise ResidueUnwrapError(
                "Could not make the trajectory contact residues whole under PBC "
                f"at frame {frame_index}."
            ) from exc
        return unwrapped

    def unwrap_mdanalysis(self, frame_index: int) -> np.ndarray:
        """Unwrap with MDAnalysis' public residue implementation."""

        from MDAnalysis.exceptions import NoDataError

        try:
            coordinates = self.atom_group.unwrap(
                compound="residues",
                reference=None,
                inplace=True,
            )
        except (AttributeError, NoDataError, TypeError, ValueError) as exc:
            raise ResidueUnwrapError(
                "Could not make the trajectory contact residues whole under PBC "
                f"at frame {frame_index} with MDAnalysis."
            ) from exc
        return np.asarray(coordinates)
