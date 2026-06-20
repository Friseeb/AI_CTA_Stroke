"""Spatial geometry utilities: bbox, crop, resample helpers.

This module now re-exports the shared implementation from ``cta_common.geometry``
(install with ``pip install -e cta_common``). It is kept as a stable import path
so existing ``from .geometry import ...`` call sites keep working.
"""

from __future__ import annotations

from cta_common.geometry import (  # noqa: F401
    BoundingBox,
    compute_spacing_from_sitk,
    crop_array,
    pad_to_multiple,
    voxel_volume_mm3,
)

__all__ = [
    "BoundingBox",
    "compute_spacing_from_sitk",
    "crop_array",
    "pad_to_multiple",
    "voxel_volume_mm3",
]
