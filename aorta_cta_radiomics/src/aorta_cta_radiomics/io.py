"""NIfTI image and mask I/O helpers.

This module now re-exports the shared implementation from ``cta_common.io``
(install with ``pip install -e cta_common``). It is kept as a stable import path
so existing ``from .io import ...`` call sites keep working.
"""

from __future__ import annotations

from cta_common.io import (  # noqa: F401
    ImageVolume,
    load_image_and_mask,
    read_mask,
    read_volume,
    resample_mask_to_image,
    same_physical_space,
    voxel_to_physical,
    write_label_like,
    write_mask_like,
    _sitk,
)

__all__ = [
    "ImageVolume",
    "load_image_and_mask",
    "read_mask",
    "read_volume",
    "resample_mask_to_image",
    "same_physical_space",
    "voxel_to_physical",
    "write_label_like",
    "write_mask_like",
]
