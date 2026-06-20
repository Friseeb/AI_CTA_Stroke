"""Voxel/physical geometry helpers shared across the CTA pipelines.

Consolidates the two previously-duplicated ``geometry.py`` modules from
``stroke-cta-osa`` and ``cta-dental-opportunistic-screening``. The bounding box
operates in numpy array order (axis 0, 1, 2). Both legacy attribute spellings
are exposed as aliases so existing call sites keep working:

- stroke used ``min_zyx`` / ``max_zyx`` / ``slices()``
- dental used ``min_ijk`` / ``max_ijk`` / ``to_slices()``

Scalar helpers carry an axis-order assumption noted per function; ``voxel_volume_mm3``
is order-independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Axes in numpy array order, i.e. ``array[z, y, x]`` for SimpleITK-loaded volumes.
AXIS_Z = 0
AXIS_Y = 1
AXIS_X = 2


@dataclass
class BoundingBox:
    """Inclusive axis-aligned voxel bbox in numpy array order (axis 0, 1, 2)."""

    min_idx: np.ndarray  # shape (3,), int
    max_idx: np.ndarray  # shape (3,), int

    # --- legacy attribute aliases -----------------------------------------
    @property
    def min_zyx(self) -> np.ndarray:
        return self.min_idx

    @property
    def max_zyx(self) -> np.ndarray:
        return self.max_idx

    @property
    def min_ijk(self) -> np.ndarray:
        return self.min_idx

    @property
    def max_ijk(self) -> np.ndarray:
        return self.max_idx

    # --- constructors ------------------------------------------------------
    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "BoundingBox":
        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("Mask is empty — cannot compute bounding box.")
        return cls(coords.min(axis=0).astype(int), coords.max(axis=0).astype(int))

    # --- transforms (return new boxes) ------------------------------------
    def expand_voxels(self, voxels: Sequence[int]) -> "BoundingBox":
        v = np.asarray(voxels, dtype=int)
        return BoundingBox(np.maximum(0, self.min_idx - v), self.max_idx + v)

    def expand_mm(self, margin_mm: float, spacing_mm: Sequence[float]) -> "BoundingBox":
        vox = np.array([math.ceil(margin_mm / s) for s in spacing_mm], dtype=int)
        return self.expand_voxels(vox)

    def clip_to_shape(self, shape: Sequence[int]) -> "BoundingBox":
        sh = np.asarray(shape, dtype=int)
        return BoundingBox(
            np.maximum(0, self.min_idx), np.minimum(sh - 1, self.max_idx)
        )

    # --- accessors ---------------------------------------------------------
    def shape(self) -> np.ndarray:
        return self.max_idx - self.min_idx + 1

    def to_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(int(self.min_idx[0]), int(self.max_idx[0]) + 1),
            slice(int(self.min_idx[1]), int(self.max_idx[1]) + 1),
            slice(int(self.min_idx[2]), int(self.max_idx[2]) + 1),
        )

    # Legacy alias for stroke-cta-osa call sites.
    def slices(self) -> tuple[slice, slice, slice]:
        return self.to_slices()

    def to_dict(self) -> dict:
        return {
            "min_ijk": self.min_idx.tolist(),
            "max_ijk": self.max_idx.tolist(),
            "shape_ijk": self.shape().tolist(),
        }

    def to_physical(self, origin: Sequence[float], spacing: Sequence[float]) -> dict:
        orig = np.asarray(origin, dtype=float)
        sp = np.asarray(spacing, dtype=float)
        return {
            "min_physical_mm": (orig + self.min_idx * sp).tolist(),
            "max_physical_mm": (orig + self.max_idx * sp).tolist(),
            "size_mm": ((self.max_idx - self.min_idx) * sp).tolist(),
        }


def crop_array(arr: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return arr[bbox.to_slices()]


def pad_to_multiple(
    arr: np.ndarray, multiple: int = 16
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Zero-pad so each dim is divisible by ``multiple``; return (padded, pad widths)."""
    pads = []
    for dim in arr.shape:
        remainder = dim % multiple
        pads.append((0, 0) if remainder == 0 else (0, multiple - remainder))
    return np.pad(arr, pads), pads


def voxel_volume_mm3(spacing_mm: Sequence[float]) -> float:
    """Volume of one voxel in mm³ (order-independent)."""
    result = 1.0
    for s in spacing_mm:
        result *= float(s)
    return result


def slice_area_mm2(spacing_xyz_mm: Sequence[float]) -> float:
    """In-plane (axial) voxel area in mm². Expects spacing in (x, y, z) order."""
    sx, sy, _ = (float(v) for v in spacing_xyz_mm)
    return sx * sy


def slice_thickness_mm(spacing_xyz_mm: Sequence[float]) -> float:
    """Z spacing in mm. Expects spacing in (x, y, z) order."""
    return float(spacing_xyz_mm[2])


def mm_to_voxels(value_mm: float, spacing_one_axis_mm: float) -> int:
    return max(1, int(round(value_mm / max(spacing_one_axis_mm, 1e-6))))


def z_index_to_mm(
    z_index: int, origin_xyz_mm: Sequence[float], spacing_xyz_mm: Sequence[float]
) -> float:
    """Approximate z physical position (axis-aligned LPS). Reporting only.

    Expects origin/spacing in (x, y, z) order.
    """
    return float(origin_xyz_mm[2]) + float(z_index) * float(spacing_xyz_mm[2])


def compute_spacing_from_sitk(sitk_image) -> tuple[float, ...]:
    """SimpleITK spacing (x, y, z) -> numpy array order (z, y, x)."""
    return tuple(reversed(sitk_image.GetSpacing()))


__all__ = [
    "AXIS_Z",
    "AXIS_Y",
    "AXIS_X",
    "BoundingBox",
    "crop_array",
    "pad_to_multiple",
    "voxel_volume_mm3",
    "slice_area_mm2",
    "slice_thickness_mm",
    "mm_to_voxels",
    "z_index_to_mm",
    "compute_spacing_from_sitk",
]
