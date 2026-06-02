"""Cervical / parapharyngeal / retropharyngeal fat features.

Fat voxel definition is configurable but defaults to standard adipose HU
window (−190, −30). For every fat ROI we report:
    * volume (mL) and mean / median / p10 / p90 / std HU
    * derived ratios where they are clinically motivated

Vessel exclusion: contrast-enhanced CTA puts vessels at HU > 120 by default,
which is already above the fat window and so does not contaminate fat
volumes directly. We still record a `vessel_exclusion_hu` value because the
same ROIs are used for soft-tissue radiomics downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .config import FatConfig, HUConfig
from .geometry import mm_to_voxels
from .logging_utils import get_logger
from .rois import (body_mask, cervical_z_range, parapharyngeal_bands,
                   retropharyngeal_band, subcutaneous_band, posterior_tongue_band)
from .shared_schema import SharedAirwayLandmarks
from .types import AirwayMaskInfo, CTAImage

log = get_logger("fat")

_NAN = float("nan")


def compute_fat_features(
    image: CTAImage,
    airway: Optional[AirwayMaskInfo],
    landmarks: SharedAirwayLandmarks,
    hu_cfg: HUConfig,
    fat_cfg: FatConfig,
    airway_min_csa_z_index: Optional[int] = None,
    save_masks_callback=None,
) -> dict[str, float | str | None]:
    """Run every fat-feature block. Returns a flat dict keyed by feature name.

    `save_masks_callback`, if provided, is called as
    `save_masks_callback(name: str, mask: np.ndarray)` for any ROI mask we
    materialise — used by the orchestrator when cfg.output.save_masks=True.
    """
    arr_hu = image.array
    fat_voxels = (arr_hu >= hu_cfg.fat_hu_min) & (arr_hu <= hu_cfg.fat_hu_max)

    # ---- Z range and body envelope ----
    z_lo, z_hi = cervical_z_range(image, airway, landmarks)
    body = body_mask(image, fat_cfg.body_air_threshold_hu)
    sub = subcutaneous_band(body, fat_cfg.subcutaneous_erosion_mm, image.spacing_xyz_mm)
    deep = body & ~sub

    z_slice = slice(z_lo, z_hi + 1)
    cervical_body = body.copy()
    cervical_body[:z_lo] = False
    cervical_body[z_hi + 1:] = False
    cervical_fat = fat_voxels & cervical_body
    cervical_sub_fat = fat_voxels & sub & cervical_body
    cervical_deep_fat = fat_voxels & deep & cervical_body

    if save_masks_callback is not None:
        save_masks_callback("body", cervical_body)
        save_masks_callback("fat_cervical_total", cervical_fat)
        save_masks_callback("fat_cervical_subcutaneous", cervical_sub_fat)
        save_masks_callback("fat_cervical_deep", cervical_deep_fat)

    out: dict[str, float | str | None] = {
        "fat_hu_min_used": float(hu_cfg.fat_hu_min),
        "fat_hu_max_used": float(hu_cfg.fat_hu_max),
        "fat_roi_method": "airway_or_landmark_z_range",
        "fat_cervical_z_lo_index": int(z_lo),
        "fat_cervical_z_hi_index": int(z_hi),
    }

    # ---- A. Total cervical fat ----
    out.update(_block("fat_cervical", arr_hu, cervical_fat, image))

    # ---- B. Subcutaneous cervical fat ----
    out.update(_block("fat_subcutaneous_cervical", arr_hu, cervical_sub_fat, image))
    neck_area_voxels = int(cervical_body.sum())
    sub_voxels = int(cervical_sub_fat.sum())
    out["fat_subcutaneous_fraction_of_neck_area"] = (
        round(sub_voxels / neck_area_voxels, 4) if neck_area_voxels else _NAN
    )

    # ---- C. Deep cervical fat ----
    out.update(_block("fat_deep_cervical", arr_hu, cervical_deep_fat, image))
    sub_vol = out["fat_subcutaneous_cervical_volume_ml"]
    deep_vol = out["fat_deep_cervical_volume_ml"]
    out["fat_deep_to_subcutaneous_ratio"] = (
        round(deep_vol / sub_vol, 3) if (isinstance(sub_vol, float) and sub_vol > 0
                                          and isinstance(deep_vol, float)) else _NAN
    )

    # ---- D. Parapharyngeal ----
    airway_mask = airway.mask_zyx if (airway is not None and airway.is_present) else None
    if airway_mask is not None:
        left, right, pp_method = parapharyngeal_bands(
            image, airway_mask,
            lateral_band_mm=fat_cfg.parapharyngeal_lateral_band_mm,
            axial_window_mm=fat_cfg.parapharyngeal_axial_window_mm,
            z_anchor=airway_min_csa_z_index,
        )
        left_fat = fat_voxels & left & body
        right_fat = fat_voxels & right & body
        if save_masks_callback is not None:
            save_masks_callback("fat_parapharyngeal_left", left_fat)
            save_masks_callback("fat_parapharyngeal_right", right_fat)

        out.update(_block("fat_parapharyngeal_left", arr_hu, left_fat, image))
        out.update(_block("fat_parapharyngeal_right", arr_hu, right_fat, image))
        total = left_fat | right_fat
        out.update(_block("fat_parapharyngeal_total", arr_hu, total, image))

        lv = out["fat_parapharyngeal_left_volume_ml"]
        rv = out["fat_parapharyngeal_right_volume_ml"]
        denom = (lv + rv) if (isinstance(lv, float) and isinstance(rv, float)) else _NAN
        out["fat_parapharyngeal_asymmetry_index"] = (
            round((rv - lv) / denom, 3) if (isinstance(denom, float) and denom > 0) else _NAN
        )
        out["fat_parapharyngeal_roi_method"] = pp_method

        # Ratio to airway volume
        airway_vol_voxels = int(airway_mask.sum())
        airway_vol_ml = airway_vol_voxels * image.voxel_volume_mm3 / 1000.0
        pp_total_vol = out["fat_parapharyngeal_total_volume_ml"]
        out["fat_parapharyngeal_to_airway_ratio"] = (
            round(pp_total_vol / airway_vol_ml, 3) if (airway_vol_ml > 0
                and isinstance(pp_total_vol, float)) else _NAN
        )

        # Area at min-airway-CSA slice (z_anchor) — useful single-slice feature
        if airway_min_csa_z_index is not None and 0 <= airway_min_csa_z_index < arr_hu.shape[0]:
            slice_fat = (left_fat | right_fat)[airway_min_csa_z_index]
            sx, sy, _ = image.spacing_xyz_mm
            out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = round(
                float(slice_fat.sum()) * sx * sy, 2
            )
        else:
            out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = _NAN

        # Region-anchored areas (retropalatal / retroglossal). We rely on
        # the existing landmark anchor — without landmarks we leave NaN.
        out["fat_parapharyngeal_area_retropalatal_mm2"] = _area_at_landmark_z(
            (left_fat | right_fat), landmarks.posterior_nasal_spine or landmarks.soft_palate_inferior,
            image,
        )
        out["fat_parapharyngeal_area_retroglossal_mm2"] = _area_at_landmark_z(
            (left_fat | right_fat), landmarks.epiglottis_tip or landmarks.hyoid, image,
        )
    else:
        out.update(_block("fat_parapharyngeal_left", arr_hu, None, image))
        out.update(_block("fat_parapharyngeal_right", arr_hu, None, image))
        out.update(_block("fat_parapharyngeal_total", arr_hu, None, image))
        out["fat_parapharyngeal_asymmetry_index"] = _NAN
        out["fat_parapharyngeal_roi_method"] = "unavailable_no_airway"
        out["fat_parapharyngeal_to_airway_ratio"] = _NAN
        out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = _NAN
        out["fat_parapharyngeal_area_retropalatal_mm2"] = _NAN
        out["fat_parapharyngeal_area_retroglossal_mm2"] = _NAN

    # ---- E. Retropharyngeal ----
    if airway_mask is not None:
        rp_band, rp_method = retropharyngeal_band(
            image, airway_mask,
            posterior_band_mm=fat_cfg.retropharyngeal_posterior_band_mm,
            axial_window_mm=fat_cfg.retropharyngeal_axial_window_mm,
            z_anchor=airway_min_csa_z_index,
        )
        rp_fat = fat_voxels & rp_band & body
        if save_masks_callback is not None:
            save_masks_callback("fat_retropharyngeal", rp_fat)
        out.update(_block("fat_retropharyngeal", arr_hu, rp_fat, image))
        out["fat_retropharyngeal_roi_method"] = rp_method
        out.update(_thickness_features(rp_fat, image))
    else:
        out.update(_block("fat_retropharyngeal", arr_hu, None, image))
        out["fat_retropharyngeal_roi_method"] = "unavailable_no_airway"
        out["fat_retropharyngeal_max_thickness_mm"] = _NAN
        out["fat_retropharyngeal_mean_thickness_mm"] = _NAN

    # ---- F. Posterior tongue (optional surrogate) ----
    if airway_mask is not None and airway_min_csa_z_index is not None:
        tongue_band, tongue_method = posterior_tongue_band(
            image, airway_mask,
            axial_window_mm=fat_cfg.parapharyngeal_axial_window_mm,
            z_anchor=airway_min_csa_z_index,
        )
        tongue_soft = tongue_band & body
        n_soft = int(tongue_soft.sum())
        if n_soft > 0:
            hu_in = arr_hu[tongue_soft]
            out["tongue_posterior_mean_hu"] = round(float(hu_in.mean()), 2)
            out["tongue_posterior_low_hu_fraction"] = round(
                float(((hu_in >= hu_cfg.fat_hu_min) & (hu_in <= hu_cfg.fat_hu_max)).mean()), 4)
            out["tongue_fat_surrogate_available"] = True
            out["tongue_roi_method"] = tongue_method
        else:
            out["tongue_posterior_mean_hu"] = _NAN
            out["tongue_posterior_low_hu_fraction"] = _NAN
            out["tongue_fat_surrogate_available"] = False
            out["tongue_roi_method"] = "unavailable_empty_band"
    else:
        out["tongue_posterior_mean_hu"] = _NAN
        out["tongue_posterior_low_hu_fraction"] = _NAN
        out["tongue_fat_surrogate_available"] = False
        out["tongue_roi_method"] = "unavailable_no_airway"

    return out


# --- helpers ----------------------------------------------------------------

def _block(prefix: str, arr_hu: np.ndarray, mask: Optional[np.ndarray],
           image: CTAImage) -> dict[str, float | str | None]:
    """Volume + HU summary stats for a single ROI mask, prefixed."""
    if mask is None or not mask.any():
        nan_block = {f"{prefix}_volume_mm3": _NAN, f"{prefix}_volume_ml": _NAN,
                     f"{prefix}_mean_hu": _NAN, f"{prefix}_median_hu": _NAN,
                     f"{prefix}_p10_hu": _NAN, f"{prefix}_p90_hu": _NAN,
                     f"{prefix}_std_hu": _NAN, f"{prefix}_voxel_count": 0}
        return nan_block
    n = int(mask.sum())
    vol_mm3 = n * image.voxel_volume_mm3
    hu = arr_hu[mask].astype(np.float32)
    return {
        f"{prefix}_volume_mm3": round(float(vol_mm3), 2),
        f"{prefix}_volume_ml":  round(float(vol_mm3) / 1000.0, 3),
        f"{prefix}_mean_hu":    round(float(hu.mean()), 2),
        f"{prefix}_median_hu":  round(float(np.median(hu)), 2),
        f"{prefix}_p10_hu":     round(float(np.percentile(hu, 10)), 2),
        f"{prefix}_p90_hu":     round(float(np.percentile(hu, 90)), 2),
        f"{prefix}_std_hu":     round(float(hu.std()), 2),
        f"{prefix}_voxel_count": n,
    }


def _thickness_features(rp_fat: np.ndarray, image: CTAImage) -> dict[str, float]:
    """Per-slice max and mean AP thickness of the retropharyngeal fat band."""
    if not rp_fat.any():
        return {"fat_retropharyngeal_max_thickness_mm": _NAN,
                "fat_retropharyngeal_mean_thickness_mm": _NAN}
    _, sy, _ = image.spacing_xyz_mm
    thicknesses: list[float] = []
    max_t = 0
    for z in range(rp_fat.shape[0]):
        sl = rp_fat[z]
        if not sl.any():
            continue
        # Per-column AP run-length: max number of contiguous fat voxels in y
        for x in range(sl.shape[1]):
            col = sl[:, x]
            if not col.any():
                continue
            # Count longest True run
            run = 0
            cur = 0
            for v in col:
                if v:
                    cur += 1
                    run = max(run, cur)
                else:
                    cur = 0
            if run > 0:
                thicknesses.append(run * sy)
                if run > max_t:
                    max_t = run
    if not thicknesses:
        return {"fat_retropharyngeal_max_thickness_mm": _NAN,
                "fat_retropharyngeal_mean_thickness_mm": _NAN}
    return {
        "fat_retropharyngeal_max_thickness_mm": round(float(max_t * sy), 2),
        "fat_retropharyngeal_mean_thickness_mm": round(float(np.mean(thicknesses)), 2),
    }


def _area_at_landmark_z(mask: np.ndarray, landmark, image: CTAImage) -> float:
    if landmark is None:
        return _NAN
    z = int(landmark[0])
    if z < 0 or z >= mask.shape[0]:
        return _NAN
    sx, sy, _ = image.spacing_xyz_mm
    return round(float(mask[z].sum()) * sx * sy, 2)
