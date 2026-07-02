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
    # Memory-conscious: this runs on the full-resolution volume (hundreds of
    # millions of voxels), so we free each large temporary as soon as it is no
    # longer needed and fill holes in place rather than allocating a second
    # full-volume buffer.
    soft = image.array > body_air_hu
    if not soft.any():
        return np.zeros(image.shape_zyx, dtype=bool)
    labeled, n = ndimage.label(soft)
    del soft
    if n == 0:
        del labeled
        return np.zeros(image.shape_zyx, dtype=bool)
    # Largest component via bincount (avoids a full-volume np.ones_like weight
    # array and the per-label sum_labels pass).
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    largest = int(counts.argmax())
    body = labeled == largest
    del labeled
    for z in range(body.shape[0]):
        body[z] = ndimage.binary_fill_holes(body[z])
    return body


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


def parapharyngeal_sector_bands(
    image: CTAImage,
    airway_mask: np.ndarray,
    lateral_band_mm: float,
    axial_window_mm: float,
    z_anchor: Optional[int],
    *,
    anatomy_exclusion_mask: Optional[np.ndarray] = None,
    min_lateral_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build rounded lateral sector ROIs around the airway.

    This is an anatomy-prior alternative to rectangular PP strips. It keeps
    voxels close to the airway but only in left/right lateral sectors, reducing
    pickup from broad anterior/posterior boxes.
    """
    sx, sy, sz_mm = image.spacing_xyz_mm
    sz_image = image.shape_zyx[0]
    if z_anchor is None or z_anchor < 0 or z_anchor >= sz_image:
        zs = np.where(airway_mask.any(axis=(1, 2)))[0]
        if zs.size == 0:
            return (np.zeros_like(airway_mask), np.zeros_like(airway_mask),
                    "unavailable_no_airway")
        z_anchor = int(zs[len(zs) // 2])
        method = f"airway_sector_z={z_anchor}"
    else:
        method = f"airway_sector_anchored_z={z_anchor}"

    half_z = mm_to_voxels(axial_window_mm / 2.0, sz_mm)
    min_frac = max(0.0, float(min_lateral_fraction))
    left = np.zeros_like(airway_mask, dtype=bool)
    right = np.zeros_like(airway_mask, dtype=bool)

    yy, xx = np.indices(airway_mask.shape[1:])
    z_lo = max(0, z_anchor - half_z)
    z_hi = min(sz_image - 1, z_anchor + half_z)
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        cy = float(ys.mean())
        cx = float(xs.mean())
        dx = (xx - cx) * sx
        dy = (yy - cy) * sy
        dist = np.sqrt(dx * dx + dy * dy)
        near = (dist > 0) & (dist <= lateral_band_mm)
        lateral_dominant = np.abs(dx) >= (min_frac * np.abs(dy))
        blocked = (
            anatomy_exclusion_mask[z]
            if anatomy_exclusion_mask is not None
            and anatomy_exclusion_mask.shape == airway_mask.shape
            else np.zeros_like(sl, dtype=bool)
        )
        left[z] = near & lateral_dominant & (dx < 0) & ~sl & ~blocked
        right[z] = near & lateral_dominant & (dx > 0) & ~sl & ~blocked
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


def retropharyngeal_prevertebral_band(
    image: CTAImage,
    airway_mask: np.ndarray,
    posterior_band_mm: float,
    axial_window_mm: float,
    z_anchor: Optional[int],
    *,
    prevertebral_mask: Optional[np.ndarray] = None,
    z_bounds: Optional[tuple[int, int]] = None,
    prevertebral_margin_mm: float = 1.0,
    lateral_margin_mm: float = 5.0,
) -> tuple[np.ndarray, str]:
    """RP ROI: posterior airway wall to anterior prevertebral boundary.

    If `prevertebral_mask` is missing on a slice, this falls back to the fixed
    posterior physical band for that slice. This makes the method usable with
    partial TotalSegmentator/VISTA coverage while keeping provenance explicit.
    """
    sx, sy, sz_mm = image.spacing_xyz_mm
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
    if z_bounds is None:
        z_lo = max(0, int(z_anchor) - half_z)
        z_hi = min(sz_image - 1, int(z_anchor) + half_z)
        z_method = "fixed_window"
    else:
        z_lo = max(0, min(int(z_bounds[0]), int(z_bounds[1])))
        z_hi = min(sz_image - 1, max(int(z_bounds[0]), int(z_bounds[1])))
        z_method = "oropharyngeal_window"

    band_y = mm_to_voxels(posterior_band_mm, sy)
    lateral_x = mm_to_voxels(lateral_margin_mm, sx)
    pv_margin = mm_to_voxels(prevertebral_margin_mm, sy)

    out = np.zeros_like(airway_mask, dtype=bool)
    used_prevertebral = False
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        x_lo = max(0, int(xs.min()) - lateral_x)
        x_hi = min(sl.shape[1] - 1, int(xs.max()) + lateral_x)

        pv_ys = np.array([], dtype=int)
        if (prevertebral_mask is not None
                and prevertebral_mask.shape == airway_mask.shape):
            pv = prevertebral_mask[z]
            if pv.any() and x_hi >= x_lo:
                local = pv[:, x_lo:x_hi + 1].copy()
                pv_ys = np.where(local)[0]
        posterior_sign = -1 if (pv_ys.size and float(pv_ys.mean()) < float(ys.mean())) else 1

        if posterior_sign < 0:
            y_air_post = int(ys.min())
            y_lo = max(0, y_air_post - band_y)
            y_hi = max(0, y_air_post - 1)
            if pv_ys.size:
                pv_ys = pv_ys[pv_ys < y_air_post]
                if pv_ys.size:
                    y_lo = max(y_lo, min(y_hi, int(pv_ys.max()) + pv_margin))
                    used_prevertebral = True
        else:
            y_air_post = int(ys.max())
            y_lo = min(sl.shape[0] - 1, y_air_post + 1)
            y_hi = min(sl.shape[0] - 1, y_air_post + band_y)
            if pv_ys.size:
                pv_ys = pv_ys[pv_ys > y_air_post]
                if pv_ys.size:
                    y_hi = min(y_hi, max(y_lo, int(pv_ys.min()) - pv_margin))
                    used_prevertebral = True

        if y_hi >= y_lo and x_hi >= x_lo:
            out[z, y_lo:y_hi + 1, x_lo:x_hi + 1] = True

    pv_method = "prevertebral_bounded" if used_prevertebral else "fixed_posterior_band"
    return out, f"{method}_{z_method}_{pv_method}"


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
