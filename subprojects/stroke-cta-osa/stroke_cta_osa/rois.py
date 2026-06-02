"""Anatomical ROI builders.

Each builder returns a boolean mask + a short string describing how it was
constructed; the string is propagated into the feature row's `*_roi_method`
column so downstream analysis can stratify by ROI provenance (atlas-defined,
airway-relative, landmark-defined, heuristic).

The aim is not a clinically perfect ROI: it's a deterministic, reproducible
ROI that researchers can recompute and compare. Where landmarks exist they
take priority; otherwise heuristic boxes anchored on the airway centroid are
used and clearly flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .geometry import mm_to_voxels
from .logging_utils import get_logger
from .shared_schema import SharedAirwayLandmarks
from .types import AirwayMaskInfo, CTAImage

log = get_logger("rois")


@dataclass
class ROI:
    mask_zyx: np.ndarray
    method: str
    z_range: tuple[int, int] | None = None


# --- Body / soft-tissue envelope -------------------------------------------

def body_mask(image: CTAImage, body_air_hu: float) -> np.ndarray:
    """Connected-component body silhouette.

    Body voxels = HU > body_air_hu. The largest connected component is kept
    so the table/headrest below the patient is dropped. Holes inside the
    silhouette are filled per axial slice to recover internal air-filled
    structures (airway, sinuses, oesophagus) as part of 'body'.
    """
    soft = image.array > body_air_hu
    if not soft.any():
        return np.zeros_like(soft, dtype=bool)
    labeled, n = ndimage.label(soft)
    if n == 0:
        return np.zeros_like(soft, dtype=bool)
    sizes = ndimage.sum_labels(np.ones_like(soft), labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    body = labeled == largest
    filled = np.zeros_like(body)
    for z in range(body.shape[0]):
        filled[z] = ndimage.binary_fill_holes(body[z])
    return filled


def subcutaneous_band(body: np.ndarray, erosion_mm: float, spacing_xyz_mm) -> np.ndarray:
    """Voxels inside `body` within `erosion_mm` of its surface."""
    if not body.any():
        return np.zeros_like(body)
    r = mm_to_voxels(erosion_mm, min(spacing_xyz_mm[0], spacing_xyz_mm[1]))
    eroded = ndimage.binary_erosion(body, iterations=r)
    return body & ~eroded


# --- Z extent of cervical analysis -----------------------------------------

def cervical_z_range(
    image: CTAImage,
    airway: Optional[AirwayMaskInfo],
    landmarks: SharedAirwayLandmarks,
) -> tuple[int, int]:
    """Z range used for "cervical" / "neck" feature extraction.

    Priority:
      1. landmarks (hyoid superior bound, epiglottis or PNS as upper bound).
      2. airway mask z extent if present.
      3. middle 50% of the image as a last resort.
    """
    sz = image.shape_zyx[0]
    if landmarks.hyoid and landmarks.posterior_nasal_spine:
        zs = sorted([landmarks.hyoid[0], landmarks.posterior_nasal_spine[0]])
        return max(0, zs[0]), min(sz - 1, zs[1])
    if airway is not None and airway.is_present:
        zs = np.where(airway.mask_zyx.any(axis=(1, 2)))[0]
        if zs.size:
            return int(zs.min()), int(zs.max())
    return int(sz * 0.25), int(sz * 0.75)


# --- Parapharyngeal lateral bands ------------------------------------------

def parapharyngeal_bands(
    image: CTAImage,
    airway_mask: np.ndarray,
    lateral_band_mm: float,
    axial_window_mm: float,
    z_anchor: Optional[int],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build left and right parapharyngeal ROIs lateral to the airway.

    For each axial slice in ±axial_window_mm/2 around z_anchor:
      • compute the airway centroid (y, x)
      • take a vertical strip ±lateral_band_mm/2 wide on each side of the airway
      • restrict to soft tissue (body) - this is the caller's job downstream.

    Returns (left_mask, right_mask, method_str). Left/right are in image
    column order; LPS data have patient-right as low-x.
    """
    sx, sy, sz_mm = image.spacing_xyz_mm
    sz_image = image.shape_zyx[0]
    if z_anchor is None or z_anchor < 0 or z_anchor >= sz_image:
        # Fall back to the slice of min CSA / centroid of airway
        zs = np.where(airway_mask.any(axis=(1, 2)))[0]
        if zs.size == 0:
            return (np.zeros_like(airway_mask), np.zeros_like(airway_mask),
                    "unavailable_no_airway")
        z_anchor = int(zs[len(zs) // 2])
        method = f"airway_centroid_z={z_anchor}"
    else:
        method = f"anchored_z={z_anchor}"

    half_z = mm_to_voxels(axial_window_mm / 2.0, sz_mm)
    band_x = mm_to_voxels(lateral_band_mm, sx)

    left = np.zeros_like(airway_mask, dtype=bool)
    right = np.zeros_like(airway_mask, dtype=bool)

    z_lo = max(0, z_anchor - half_z)
    z_hi = min(sz_image - 1, z_anchor + half_z)
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        cx = int(round(xs.mean()))
        x_left_lo = max(0, cx - band_x)
        x_left_hi = max(0, cx - 1)
        x_right_lo = min(sl.shape[1] - 1, cx + 1)
        x_right_hi = min(sl.shape[1] - 1, cx + band_x)
        y_lo = max(0, int(ys.min()) - 5)
        y_hi = min(sl.shape[0] - 1, int(ys.max()) + 5)
        if x_left_hi >= x_left_lo:
            left[z, y_lo:y_hi + 1, x_left_lo:x_left_hi + 1] = True
        if x_right_hi >= x_right_lo:
            right[z, y_lo:y_hi + 1, x_right_lo:x_right_hi + 1] = True
    return left, right, method


# --- Retropharyngeal posterior band ----------------------------------------

def retropharyngeal_band(
    image: CTAImage,
    airway_mask: np.ndarray,
    posterior_band_mm: float,
    axial_window_mm: float,
    z_anchor: Optional[int],
) -> tuple[np.ndarray, str]:
    """Posterior to the airway, anterior to the prevertebral region."""
    _, sy, sz_mm = image.spacing_xyz_mm
    sz_image = image.shape_zyx[0]
    if z_anchor is None or z_anchor < 0 or z_anchor >= sz_image:
        zs = np.where(airway_mask.any(axis=(1, 2)))[0]
        if zs.size == 0:
            return np.zeros_like(airway_mask), "unavailable_no_airway"
        z_anchor = int(zs[len(zs) // 2])
        method = f"airway_centroid_z={z_anchor}"
    else:
        method = f"anchored_z={z_anchor}"

    half_z = mm_to_voxels(axial_window_mm / 2.0, sz_mm)
    band_y = mm_to_voxels(posterior_band_mm, sy)

    out = np.zeros_like(airway_mask, dtype=bool)
    z_lo = max(0, z_anchor - half_z)
    z_hi = min(sz_image - 1, z_anchor + half_z)
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        y_back = int(ys.max())  # posterior wall in LPS / RAS axial
        y_lo = min(sl.shape[0] - 1, y_back + 1)
        y_hi = min(sl.shape[0] - 1, y_back + band_y)
        x_lo = max(0, int(xs.min()) - 2)
        x_hi = min(sl.shape[1] - 1, int(xs.max()) + 2)
        if y_hi >= y_lo and x_hi >= x_lo:
            out[z, y_lo:y_hi + 1, x_lo:x_hi + 1] = True
    return out, method


# --- Posterior tongue band (optional / experimental) -----------------------

def posterior_tongue_band(
    image: CTAImage,
    airway_mask: np.ndarray,
    axial_window_mm: float,
    z_anchor: Optional[int],
    anterior_band_mm: float = 30.0,
) -> tuple[np.ndarray, str]:
    """Anterior to the airway, bounded laterally to airway extent.

    Very rough — without a tongue segmentation this also picks up parts of
    floor-of-mouth musculature. Flagged as `heuristic` and gated by the
    caller on whether to emit tongue features.
    """
    sx, sy, sz_mm = image.spacing_xyz_mm
    sz_image = image.shape_zyx[0]
    if z_anchor is None or z_anchor < 0 or z_anchor >= sz_image:
        return np.zeros_like(airway_mask), "unavailable_no_anchor"
    half_z = mm_to_voxels(axial_window_mm / 2.0, sz_mm)
    band_y = mm_to_voxels(anterior_band_mm, sy)

    out = np.zeros_like(airway_mask, dtype=bool)
    for z in range(max(0, z_anchor - half_z), min(sz_image - 1, z_anchor + half_z) + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        y_front = int(ys.min())
        y_lo = max(0, y_front - band_y)
        y_hi = max(0, y_front - 1)
        x_lo = max(0, int(xs.min()))
        x_hi = min(sl.shape[1] - 1, int(xs.max()))
        if y_hi >= y_lo and x_hi >= x_lo:
            out[z, y_lo:y_hi + 1, x_lo:x_hi + 1] = True
    return out, "anterior_to_airway_heuristic"
