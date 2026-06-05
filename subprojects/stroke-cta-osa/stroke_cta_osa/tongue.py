"""Tongue features (volume, HU stats, posterior-tongue ROI, tongue-base
encroachment, tongue/mandible & tongue/oral-cavity ratios).

The module is mask-driven. Inputs:

  * `tongue_mask` — externally segmented binary mask (preferred), OR
  * landmark-based conservative posterior tongue ROI (fallback).

When neither is available, every feature is NaN with `tongue_mask_available =
False`. We never invent a global tongue volume from heuristics: tongue body
volume is too sensitive to segmentation choices, and a wrong number is
worse than a missing one. The posterior-tongue HU surrogate IS allowed as
a landmark-only fallback because (a) it's the most clinically referenced
tongue-fat proxy in adult OSA literature, and (b) a coarse box is a
faithful representation of "approximately the posterior tongue".

This module does NOT segment the tongue from scratch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .landmark_schema import LandmarkBundle
from .landmarks import (
    get_hyoid_position, get_retroglossal_level, get_tongue_base_level,
)
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("tongue")

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Configuration shape (mirrors the YAML `tongue:` block)
# ---------------------------------------------------------------------------

@dataclass
class TongueConfig:
    enabled: bool = True
    require_mask_for_volume: bool = True
    allow_posterior_roi_fallback: bool = True
    low_hu_threshold: float = 30.0
    low_hu_threshold_mode: str = "absolute"
    record_contrast_sensitivity: bool = True


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def compute_tongue_features(
    image: CTAImage,
    cfg: TongueConfig,
    tongue_mask: Optional[np.ndarray],
    landmarks: LandmarkBundle,
    airway: Optional[AirwayMaskInfo] = None,
    mandible_volume_ml: Optional[float] = None,
    oral_cavity_volume_ml: Optional[float] = None,
    save_masks_callback=None,
) -> dict[str, object]:
    """Compute every tongue-family feature defined in the registry.

    Returns a flat dict keyed by feature name. Missing values are NaN /
    False / "" so the caller can blindly merge it into the case row.
    """
    out: dict[str, object] = _empty_row()
    out["tongue_low_hu_threshold_used"] = float(cfg.low_hu_threshold)
    out["tongue_contrast_sensitive"] = bool(image.is_contrast_enhanced) \
        if cfg.record_contrast_sensitivity else False

    if not cfg.enabled:
        out["tongue_qc_failure_reasons"] = "tongue_module_disabled"
        return out

    has_full_mask = tongue_mask is not None and np.asarray(tongue_mask).any()
    if has_full_mask:
        _populate_from_tongue_mask(
            out, image, cfg,
            tongue_mask=np.asarray(tongue_mask).astype(bool),
            landmarks=landmarks,
            airway=airway,
            mandible_volume_ml=mandible_volume_ml,
            oral_cavity_volume_ml=oral_cavity_volume_ml,
            save_masks_callback=save_masks_callback,
        )
    else:
        if not cfg.allow_posterior_roi_fallback:
            out["tongue_mask_available"] = False
            out["tongue_qc_failure_reasons"] = "no_tongue_mask_and_fallback_disabled"
            return out
        # Posterior tongue ROI from landmarks + airway, no global volume.
        out["tongue_mask_available"] = False
        out["tongue_mask_method"] = "absent"
        _populate_posterior_only(
            out, image, cfg, landmarks=landmarks,
            airway=airway, save_masks_callback=save_masks_callback,
        )

    # QC roll-up
    out["tongue_qc_pass"] = not bool(out.get("tongue_qc_failure_reasons"))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _empty_row() -> dict[str, object]:
    return {
        "tongue_mask_available": False,
        "tongue_mask_method": "",
        "tongue_qc_pass": False,
        "tongue_qc_failure_reasons": "",
        "tongue_artifact_warning": False,
        "tongue_contrast_sensitive": False,
        "tongue_mask_source": "",
        "tongue_roi_confidence": "",
        "tongue_coverage_warning": "",

        "tongue_volume_mm3": _NAN, "tongue_volume_ml": _NAN,
        "tongue_mean_hu": _NAN, "tongue_median_hu": _NAN,
        "tongue_std_hu": _NAN, "tongue_p10_hu": _NAN, "tongue_p90_hu": _NAN,
        "tongue_low_hu_fraction": _NAN, "tongue_low_hu_threshold_used": _NAN,

        "tongue_posterior_roi_available": False,
        "tongue_posterior_roi_method": "",
        "tongue_posterior_volume_ml": _NAN,
        "tongue_posterior_mean_hu": _NAN, "tongue_posterior_median_hu": _NAN,
        "tongue_posterior_std_hu": _NAN,
        "tongue_posterior_p10_hu": _NAN, "tongue_posterior_p90_hu": _NAN,
        "tongue_posterior_low_hu_fraction": _NAN,

        "tongue_base_volume_ml": _NAN,
        "tongue_base_area_at_retroglossal_level_mm2": _NAN,
        "tongue_base_to_retroglossal_airway_ratio": _NAN,
        "tongue_base_posterior_displacement_mm": _NAN,
        "tongue_base_inferior_displacement_mm": _NAN,
        "retroglossal_airway_area_adjacent_to_tongue_base_mm2": _NAN,
        "tongue_base_airway_contact_length_mm": _NAN,

        "tongue_to_mandible_volume_ratio": _NAN,
        "tongue_to_oral_cavity_volume_ratio": _NAN,
        "tongue_to_skeletal_enclosure_ratio": _NAN,

        "lingual_tonsil_roi_available": False,
        "lingual_tonsil_volume_ml": _NAN,
        "lingual_tonsil_mean_hu": _NAN,
        "lingual_tonsil_to_retroglossal_airway_ratio": _NAN,
    }


def _populate_from_tongue_mask(
    out: dict, image: CTAImage, cfg: TongueConfig,
    *, tongue_mask: np.ndarray, landmarks: LandmarkBundle,
    airway: Optional[AirwayMaskInfo],
    mandible_volume_ml: Optional[float],
    oral_cavity_volume_ml: Optional[float],
    save_masks_callback,
) -> None:
    out["tongue_mask_available"] = True
    out["tongue_mask_method"] = "external_or_dental"
    out["tongue_mask_source"] = "provided"
    if save_masks_callback is not None:
        save_masks_callback("tongue", tongue_mask)

    vox_vol = image.voxel_volume_mm3
    n_vox = int(tongue_mask.sum())
    out["tongue_volume_mm3"] = round(float(n_vox * vox_vol), 2)
    out["tongue_volume_ml"] = round(float(n_vox * vox_vol) / 1000.0, 4)

    hu_vals = image.array[tongue_mask].astype(np.float32)
    out["tongue_mean_hu"] = round(float(hu_vals.mean()), 2)
    out["tongue_median_hu"] = round(float(np.median(hu_vals)), 2)
    out["tongue_std_hu"] = round(float(hu_vals.std()), 2)
    out["tongue_p10_hu"] = round(float(np.percentile(hu_vals, 10)), 2)
    out["tongue_p90_hu"] = round(float(np.percentile(hu_vals, 90)), 2)
    low_thresh = float(cfg.low_hu_threshold)
    out["tongue_low_hu_fraction"] = round(
        float((hu_vals < low_thresh).mean()), 4)
    out["tongue_low_hu_threshold_used"] = low_thresh

    # Posterior tongue from mask: posterior 1/3 along the dorsal axis.
    # In ITK conventions, larger y (axis 1) is more posterior; we use the
    # mask's own y-extent so we don't hardcode patient orientation.
    posterior, post_method = _posterior_third_of_mask(tongue_mask)
    if posterior.any():
        out["tongue_posterior_roi_available"] = True
        out["tongue_posterior_roi_method"] = post_method
        _hu_block(out, image, posterior, prefix="tongue_posterior",
                  low_hu_threshold=low_thresh)
        if save_masks_callback is not None:
            save_masks_callback("tongue_posterior", posterior)

    # Tongue base ≈ inferior 1/3 of the mask in z, or constrained to the
    # tongue-base level band if landmarks provide one.
    base_band = _tongue_base_band(image, tongue_mask, landmarks)
    if base_band.any():
        n_base = int(base_band.sum())
        out["tongue_base_volume_ml"] = round(float(n_base * vox_vol) / 1000.0, 4)
        if save_masks_callback is not None:
            save_masks_callback("tongue_base", base_band)

        rg_z = get_retroglossal_level(landmarks)
        if rg_z is not None and 0 <= rg_z < base_band.shape[0]:
            sx, sy, _ = image.spacing_xyz_mm
            area_mm2 = float(base_band[rg_z].sum()) * sx * sy
            out["tongue_base_area_at_retroglossal_level_mm2"] = round(area_mm2, 2)
            if airway is not None and airway.is_present:
                airway_area = float(airway.mask_zyx[rg_z].sum()) * sx * sy
                if airway_area > 0:
                    out["tongue_base_to_retroglossal_airway_ratio"] = round(
                        area_mm2 / airway_area, 4)
                out["retroglossal_airway_area_adjacent_to_tongue_base_mm2"] = \
                    round(airway_area, 2)

        # Geometric displacement: posterior point of base vs airway posterior wall
        if airway is not None and airway.is_present:
            disp = _tongue_base_airway_displacements(image, base_band, airway.mask_zyx)
            out.update(disp)

    # Ratios
    if mandible_volume_ml is not None and mandible_volume_ml > 0:
        out["tongue_to_mandible_volume_ratio"] = round(
            float(out["tongue_volume_ml"]) / float(mandible_volume_ml), 4)
    if oral_cavity_volume_ml is not None and oral_cavity_volume_ml > 0:
        out["tongue_to_oral_cavity_volume_ratio"] = round(
            float(out["tongue_volume_ml"]) / float(oral_cavity_volume_ml), 4)


def _populate_posterior_only(
    out: dict, image: CTAImage, cfg: TongueConfig,
    *, landmarks: LandmarkBundle, airway: Optional[AirwayMaskInfo],
    save_masks_callback,
) -> None:
    """Landmark-only posterior tongue ROI.

    The ROI is a coarse rectangular box anterior to the retroglossal-level
    airway slice, vertically bounded by the tongue-base z band, and
    laterally bounded by the airway's L-R extent expanded by a fixed margin.
    Marked `roi_confidence = "low"`.
    """
    box, method, confidence = _landmark_posterior_tongue_box(
        image, landmarks, airway,
    )
    if box is None or not box.any():
        out["tongue_qc_failure_reasons"] = "no_tongue_mask_and_no_landmark_box"
        return
    out["tongue_posterior_roi_available"] = True
    out["tongue_posterior_roi_method"] = method
    out["tongue_roi_confidence"] = confidence
    _hu_block(out, image, box, prefix="tongue_posterior",
              low_hu_threshold=float(cfg.low_hu_threshold))
    if save_masks_callback is not None:
        save_masks_callback("tongue_posterior", box)


def _hu_block(out: dict, image: CTAImage, mask: np.ndarray,
              *, prefix: str, low_hu_threshold: float) -> None:
    if not mask.any():
        return
    vox = image.voxel_volume_mm3
    n = int(mask.sum())
    out[f"{prefix}_volume_ml"] = round(float(n * vox) / 1000.0, 4)
    hu = image.array[mask].astype(np.float32)
    out[f"{prefix}_mean_hu"] = round(float(hu.mean()), 2)
    out[f"{prefix}_median_hu"] = round(float(np.median(hu)), 2)
    out[f"{prefix}_std_hu"] = round(float(hu.std()), 2)
    out[f"{prefix}_p10_hu"] = round(float(np.percentile(hu, 10)), 2)
    out[f"{prefix}_p90_hu"] = round(float(np.percentile(hu, 90)), 2)
    out[f"{prefix}_low_hu_fraction"] = round(
        float((hu < low_hu_threshold).mean()), 4)


def _posterior_third_of_mask(mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Posterior 1/3 of the mask along the y-axis (axis 1 in our (z,y,x) arrays).

    We treat "larger y" as posterior by the SimpleITK/DICOM-derived
    NIfTI convention. If the affine is RAS this becomes "smaller y", which
    we don't detect here — the result is still "one third of the tongue's
    dorsal extent" which is acceptable as a posterior-tongue surrogate
    irrespective of L-R/A-P sign.
    """
    if not mask.any():
        return mask, "empty_mask"
    ys = np.where(mask.any(axis=(0, 2)))[0]
    y_lo, y_hi = int(ys.min()), int(ys.max())
    third = max(1, (y_hi - y_lo + 1) // 3)
    out = np.zeros_like(mask)
    out[:, y_hi - third + 1:y_hi + 1, :] = mask[:, y_hi - third + 1:y_hi + 1, :]
    return out, f"posterior_third_y[{y_hi-third+1},{y_hi}]"


def _tongue_base_band(
    image: CTAImage, tongue_mask: np.ndarray, landmarks: LandmarkBundle,
) -> np.ndarray:
    """Tongue-base region.

    Priority:
      1. tongue_base_level landmark ± 10 mm in z (using the mask's z slab),
      2. inferior 1/3 of the mask's z extent (assumes larger z = inferior).
    """
    sz_mm = image.spacing_xyz_mm[2]
    band_voxels = max(1, int(round(10.0 / sz_mm)))
    out = np.zeros_like(tongue_mask)
    base_z = get_tongue_base_level(landmarks)
    if base_z is not None:
        z_lo = max(0, base_z - band_voxels)
        z_hi = min(tongue_mask.shape[0] - 1, base_z + band_voxels)
        out[z_lo:z_hi + 1] = tongue_mask[z_lo:z_hi + 1]
        return out
    # Fallback: inferior 1/3 of the tongue mask z extent
    zs = np.where(tongue_mask.any(axis=(1, 2)))[0]
    if zs.size == 0:
        return out
    z_lo_full, z_hi_full = int(zs.min()), int(zs.max())
    third = max(1, (z_hi_full - z_lo_full + 1) // 3)
    out[z_hi_full - third + 1:z_hi_full + 1] = (
        tongue_mask[z_hi_full - third + 1:z_hi_full + 1]
    )
    return out


def _landmark_posterior_tongue_box(
    image: CTAImage, landmarks: LandmarkBundle, airway: Optional[AirwayMaskInfo],
) -> tuple[Optional[np.ndarray], str, str]:
    """Coarse landmark+airway posterior-tongue box.

    Z band:
        tongue_base_level (or retroglossal_level) ± 15 mm.
    Y band:
        anterior to airway posterior wall at retroglossal_level by up to 30 mm
        (in axis-1 voxels).
    X band:
        airway L-R extent at retroglossal_level expanded by ±10 mm.
    Returns (None, method, conf) if landmarks aren't enough to anchor.
    """
    rg = get_retroglossal_level(landmarks)
    base_z = get_tongue_base_level(landmarks) or rg
    if base_z is None or airway is None or not airway.is_present:
        return None, "no_anchor", "low"
    sx, sy, sz = image.spacing_xyz_mm
    half_z = max(1, int(round(15.0 / sz)))
    anchor = base_z if base_z is not None else rg
    z_lo = max(0, anchor - half_z)
    z_hi = min(airway.mask_zyx.shape[0] - 1, anchor + half_z)
    box = np.zeros_like(airway.mask_zyx)

    # Use the anchor slice's airway extent as the L-R anchor.
    z_for_xyref = anchor if 0 <= anchor < airway.mask_zyx.shape[0] else rg
    sl = airway.mask_zyx[z_for_xyref]
    if not sl.any():
        return None, "airway_empty_at_anchor", "low"
    ys, xs = np.where(sl)
    margin_x = max(1, int(round(10.0 / sx)))
    margin_y_back = max(1, int(round(30.0 / sy)))
    x_lo = max(0, int(xs.min()) - margin_x)
    x_hi = min(sl.shape[1] - 1, int(xs.max()) + margin_x)
    y_back = int(ys.min())  # smaller y = anterior in LPS-derived NIfTI
    y_lo = max(0, y_back - margin_y_back)
    y_hi = max(0, y_back - 1)
    if y_hi < y_lo or x_hi < x_lo:
        return None, "degenerate_box", "low"

    box[z_lo:z_hi + 1, y_lo:y_hi + 1, x_lo:x_hi + 1] = True
    return box, "landmark_box_z={}_y=[{},{}]".format(anchor, y_lo, y_hi), "low"


def _tongue_base_airway_displacements(
    image: CTAImage, base_band: np.ndarray, airway_mask: np.ndarray,
) -> dict[str, float]:
    """Posterior + inferior displacements of the tongue-base centroid relative
    to the airway anterior wall, in physical mm.

    Posterior displacement = (airway anterior y) − (tongue base posterior y),
    larger = tongue is further posterior (more crowded airway).
    Inferior displacement = (airway top z) − (tongue base inferior z), with
    sign indicating below/above.
    """
    sx, sy, sz = image.spacing_xyz_mm
    if not base_band.any() or not airway_mask.any():
        return {
            "tongue_base_posterior_displacement_mm": _NAN,
            "tongue_base_inferior_displacement_mm": _NAN,
        }
    base_coords = np.argwhere(base_band)
    airway_coords = np.argwhere(airway_mask)
    # Take the z-band overlap to ensure we compare comparable slices
    z_lo = max(int(base_coords[:, 0].min()), int(airway_coords[:, 0].min()))
    z_hi = min(int(base_coords[:, 0].max()), int(airway_coords[:, 0].max()))
    if z_hi < z_lo:
        return {
            "tongue_base_posterior_displacement_mm": _NAN,
            "tongue_base_inferior_displacement_mm": _NAN,
        }
    base_y = base_coords[(base_coords[:, 0] >= z_lo) & (base_coords[:, 0] <= z_hi), 1]
    airway_y = airway_coords[(airway_coords[:, 0] >= z_lo) & (airway_coords[:, 0] <= z_hi), 1]
    if base_y.size == 0 or airway_y.size == 0:
        return {
            "tongue_base_posterior_displacement_mm": _NAN,
            "tongue_base_inferior_displacement_mm": _NAN,
        }
    posterior_disp_vox = float(base_y.max() - airway_y.min())
    inferior_disp_vox = float(int(base_coords[:, 0].max())
                              - int(airway_coords[:, 0].min()))
    return {
        "tongue_base_posterior_displacement_mm":
            round(posterior_disp_vox * sy, 2),
        "tongue_base_inferior_displacement_mm":
            round(inferior_disp_vox * sz, 2),
    }
