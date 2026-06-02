"""Voxel ↔ physical conversions and small axis helpers.

Keep this thin. The dental subproject has a sister `geometry.py` with the
same bounding-box class, but we don't import it: we don't want the stroke
pipeline to break when the dental package isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Axes (in numpy array order, i.e. cta.array[z, y, x])
AXIS_Z = 0
AXIS_Y = 1
AXIS_X = 2


@dataclass
class BoundingBox:
    """Inclusive voxel bbox in (z, y, x) numpy order."""
    min_zyx: np.ndarray
    max_zyx: np.ndarray

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "BoundingBox":
        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("Empty mask cannot produce a bounding box.")
        return cls(coords.min(axis=0).astype(int), coords.max(axis=0).astype(int))

    def shape(self) -> tuple[int, int, int]:
        s = (self.max_zyx - self.min_zyx + 1).tolist()
        return int(s[0]), int(s[1]), int(s[2])

    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(int(self.min_zyx[0]), int(self.max_zyx[0]) + 1),
            slice(int(self.min_zyx[1]), int(self.max_zyx[1]) + 1),
            slice(int(self.min_zyx[2]), int(self.max_zyx[2]) + 1),
        )


def voxel_volume_mm3(spacing_xyz_mm: Sequence[float]) -> float:
    sx, sy, sz = (float(v) for v in spacing_xyz_mm)
    return sx * sy * sz


def slice_area_mm2(spacing_xyz_mm: Sequence[float]) -> float:
    """Area of one in-plane (axial) voxel, mm². Used to convert per-slice
    pixel counts to physical CSA — even when voxels are anisotropic in z.
    """
    sx, sy, _ = (float(v) for v in spacing_xyz_mm)
    return sx * sy


def slice_thickness_mm(spacing_xyz_mm: Sequence[float]) -> float:
    return float(spacing_xyz_mm[2])


def mm_to_voxels(value_mm: float, spacing_one_axis_mm: float) -> int:
    return max(1, int(round(value_mm / max(spacing_one_axis_mm, 1e-6))))


def z_index_to_mm(z_index: int, origin_xyz_mm: Sequence[float], spacing_xyz_mm: Sequence[float]) -> float:
    """Approximate z-physical position assuming axis-aligned LPS direction.
    Used only for reporting, not for resampling.
    """
    return float(origin_xyz_mm[2]) + float(z_index) * float(spacing_xyz_mm[2])
