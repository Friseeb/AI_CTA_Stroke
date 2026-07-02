"""Helpers for optional anatomy masks used as ROI priors."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from .geometry import mm_to_voxels


DEFAULT_PRIOR_NAMES = (
    "tongue",
    "mandible",
    "oral_cavity",
    "soft_palate",
    "uvula",
    "palatine_tonsil_left",
    "palatine_tonsil_right",
)


def combined_anatomy_exclusion_mask(
    anatomy_masks: Optional[dict[str, Optional[np.ndarray]]],
    *,
    reference_shape: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    dilation_mm: float = 1.0,
    names: tuple[str, ...] = DEFAULT_PRIOR_NAMES,
) -> tuple[np.ndarray, list[str]]:
    """Return a combined mask of known non-fat anatomy.

    TotalSegmentator/VISTA/dental masks are used as conservative exclusion
    priors for fat ROIs. The fat HU window remains the primary classifier; this
    mask only prevents airway-relative ROIs from spilling into known anatomy
    when upstream segmentations are available.
    """
    out = np.zeros(reference_shape, dtype=bool)
    used: list[str] = []
    if not anatomy_masks:
        return out, used

    for name in names:
        mask = anatomy_masks.get(name)
        if mask is None:
            continue
        arr = np.asarray(mask).astype(bool)
        if arr.shape != reference_shape or not arr.any():
            continue
        out |= arr
        used.append(name)

    if out.any() and dilation_mm > 0:
        iterations = mm_to_voxels(dilation_mm, min(spacing_xyz_mm))
        if iterations > 0:
            out = ndimage.binary_dilation(out, iterations=iterations)
    return out, used
