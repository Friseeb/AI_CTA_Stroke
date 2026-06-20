"""Voxel ↔ physical conversions and small axis helpers.

This module now re-exports the shared implementation from ``cta_common.geometry``
(install with ``pip install -e cta_common``). It is kept as a stable import path
so existing ``from .geometry import ...`` call sites keep working. The shared
``BoundingBox`` exposes both the old ``min_zyx``/``max_zyx`` attributes and the
``slices()`` method used here.
"""

from __future__ import annotations

from cta_common.geometry import (  # noqa: F401
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    BoundingBox,
    mm_to_voxels,
    slice_area_mm2,
    slice_thickness_mm,
    voxel_volume_mm3,
    z_index_to_mm,
)

__all__ = [
    "AXIS_X",
    "AXIS_Y",
    "AXIS_Z",
    "BoundingBox",
    "mm_to_voxels",
    "slice_area_mm2",
    "slice_thickness_mm",
    "voxel_volume_mm3",
    "z_index_to_mm",
]
