"""Spatial geometry utilities: bbox, crop, resample helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in voxel coordinates (inclusive)."""
    min_ijk: np.ndarray  # shape (3,) int
    max_ijk: np.ndarray  # shape (3,) int

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "BoundingBox":
        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("Mask is empty — cannot compute bounding box.")
        return cls(
            min_ijk=coords.min(axis=0).astype(int),
            max_ijk=coords.max(axis=0).astype(int),
        )

    def expand_voxels(self, voxels: Sequence[int]) -> "BoundingBox":
        v = np.array(voxels, dtype=int)
        return BoundingBox(
            min_ijk=np.maximum(0, self.min_ijk - v),
            max_ijk=self.max_ijk + v,
        )

    def expand_mm(self, margin_mm: float, spacing_mm: Sequence[float]) -> "BoundingBox":
        vox = np.array([math.ceil(margin_mm / s) for s in spacing_mm], dtype=int)
        return self.expand_voxels(vox)

    def clip_to_shape(self, shape: Sequence[int]) -> "BoundingBox":
        sh = np.array(shape, dtype=int)
        return BoundingBox(
            min_ijk=np.maximum(0, self.min_ijk),
            max_ijk=np.minimum(sh - 1, self.max_ijk),
        )

    def to_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(int(self.min_ijk[0]), int(self.max_ijk[0]) + 1),
            slice(int(self.min_ijk[1]), int(self.max_ijk[1]) + 1),
            slice(int(self.min_ijk[2]), int(self.max_ijk[2]) + 1),
        )

    def shape(self) -> np.ndarray:
        return self.max_ijk - self.min_ijk + 1

    def to_dict(self) -> dict:
        return {
            "min_ijk": self.min_ijk.tolist(),
            "max_ijk": self.max_ijk.tolist(),
            "shape_ijk": self.shape().tolist(),
        }

    def to_physical(self, origin: Sequence[float], spacing: Sequence[float]) -> dict:
        orig = np.array(origin)
        sp = np.array(spacing)
        min_phys = orig + self.min_ijk * sp
        max_phys = orig + self.max_ijk * sp
        return {
            "min_physical_mm": min_phys.tolist(),
            "max_physical_mm": max_phys.tolist(),
            "size_mm": ((self.max_ijk - self.min_ijk) * sp).tolist(),
        }


def crop_array(arr: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    s = bbox.to_slices()
    return arr[s]


def pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Zero-pad array so each dim is divisible by *multiple*. Returns padded array and pad widths."""
    pads = []
    for dim in arr.shape:
        remainder = dim % multiple
        if remainder == 0:
            pads.append((0, 0))
        else:
            pads.append((0, multiple - remainder))
    return np.pad(arr, pads), pads


def voxel_volume_mm3(spacing_mm: Sequence[float]) -> float:
    result = 1.0
    for s in spacing_mm:
        result *= s
    return result


def compute_spacing_from_sitk(sitk_image) -> tuple[float, ...]:
    """Return spacing as (i, j, k) matching numpy array axis order."""
    sp = sitk_image.GetSpacing()  # ITK: (x, y, z) = (col, row, slice)
    return tuple(reversed(sp))    # numpy: (slice, row, col)
