"""Expanded fat features: level-wise areas + per-side parapharyngeal at
RP/RG/subglosso-supraglottic levels + facial / buccal fat (when FOV covers
the face).

This module *complements* :mod:`fat`; it does not replace it. The existing
module already produces the headline cervical / subcutaneous / deep / total
parapharyngeal / retropharyngeal volumes — this module adds the
landmark-anchored area features that downstream OSA modelling typically
needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .config import FatConfig, HUConfig
from .geometry import mm_to_voxels, slice_area_mm2
from .landmark_schema import LandmarkBundle
from .landmarks import (
    get_hyoid_position, get_retroglossal_level, get_retropalatal_level,
)
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("fat_regions")

_NAN = float("nan")


@dataclass
class FatRegionConfig:
    enabled: bool = True
    fat_hu_min: float = -190.0
    fat_hu_max: float = -30.0
    parapharyngeal_lateral_band_mm: float = 25.0
    parapharyngeal_axial_window_mm: float = 30.0
    retropharyngeal_posterior_band_mm: float = 15.0
    retropharyngeal_axial_window_mm: float = 30.0
    body_air_threshold_hu: float = -250.0
    enable_facial_fat: bool = False


def compute_regional_fat_features(
    image: CTAImage,
    cfg: FatRegionConfig,
    *,
    airway: Optional[AirwayMaskInfo],
    body_mask: Optional[np.ndarray],
    landmarks: LandmarkBundle,
    save_masks_callback=None,
) -> dict[str, object]:
    out = _empty_row()
    if not cfg.enabled or airway is None or not airway.is_present:
        return out
    if body_mask is None or not body_mask.any():
        return out

    sx, sy, sz_mm = image.spacing_xyz_mm
    in_plane = slice_area_mm2(image.spacing_xyz_mm)
    fat_voxels = ((image.array >= cfg.fat_hu_min)
                  & (image.array <= cfg.fat_hu_max))

    # ---- Cervical-fat area at standard z levels ----
    cerv_fat = fat_voxels & body_mask
    rp_z = get_retropalatal_level(landmarks)
    rg_z = get_retroglossal_level(landmarks)
    hyoid_z = None
    p = landmarks.points.get("hyoid_centroid")
    if p and p.voxel_zyx:
        hyoid_z = int(p.voxel_zyx[0])
    if hyoid_z is not None and 0 <= hyoid_z < cerv_fat.shape[0]:
        out["fat_cervical_area_at_hyoid_level_mm2"] = round(
            float(cerv_fat[hyoid_z].sum() * in_plane), 2)
    if rp_z is not None and 0 <= rp_z < cerv_fat.shape[0]:
        out["fat_cervical_area_at_retropalatal_level_mm2"] = round(
            float(cerv_fat[rp_z].sum() * in_plane), 2)
    if rg_z is not None and 0 <= rg_z < cerv_fat.shape[0]:
        out["fat_cervical_area_at_retroglossal_level_mm2"] = round(
            float(cerv_fat[rg_z].sum() * in_plane), 2)

    # ---- Per-side parapharyngeal at each level ----
    for level_name, anchor_z in (("retropalatal", rp_z),
                                  ("retroglossal", rg_z),
                                  ("subglosso_supraglottic",
                                   _subglosso_anchor(rg_z, sz_mm))):
        if anchor_z is None or not (0 <= anchor_z < cerv_fat.shape[0]):
            continue
        left, right = _per_side_parapharyngeal(
            image=image, airway_mask=airway.mask_zyx,
            anchor_z=anchor_z,
            lateral_band_mm=cfg.parapharyngeal_lateral_band_mm,
            window_mm=cfg.parapharyngeal_axial_window_mm,
            body_mask=body_mask, fat_voxels=fat_voxels,
        )
        # Area at the anchor slice
        a_left = float(left[anchor_z].sum() * in_plane) if left.any() else 0.0
        a_right = float(right[anchor_z].sum() * in_plane) if right.any() else 0.0
        out[f"fat_parapharyngeal_area_{level_name}_left_mm2"] = round(a_left, 2)
        out[f"fat_parapharyngeal_area_{level_name}_right_mm2"] = round(a_right, 2)
        out[f"fat_parapharyngeal_area_{level_name}_total_mm2"] = round(
            a_left + a_right, 2)
        # Airway-to-PPF ratio at this level
        airway_area = float(airway.mask_zyx[anchor_z].sum() * in_plane)
        if airway_area > 0:
            ratio = (a_left + a_right) / airway_area
            out[f"fat_parapharyngeal_to_airway_ratio_{level_name}"] = round(
                float(ratio), 4)
        if save_masks_callback is not None:
            save_masks_callback(f"fat_parapharyngeal_{level_name}_left", left)
            save_masks_callback(f"fat_parapharyngeal_{level_name}_right", right)

    # ---- Retropharyngeal area at standard z levels ----
    rp_band = _retropharyngeal_band(
        image=image, airway_mask=airway.mask_zyx,
        posterior_band_mm=cfg.retropharyngeal_posterior_band_mm,
        window_mm=cfg.retropharyngeal_axial_window_mm,
        anchor_z=rp_z if rp_z is not None else rg_z,
    )
    rp_fat = rp_band & fat_voxels & body_mask
    if rp_z is not None and 0 <= rp_z < rp_fat.shape[0]:
        out["fat_retropharyngeal_area_at_retropalatal_level_mm2"] = round(
            float(rp_fat[rp_z].sum() * in_plane), 2)
    if rg_z is not None and 0 <= rg_z < rp_fat.shape[0]:
        out["fat_retropharyngeal_area_at_retroglossal_level_mm2"] = round(
            float(rp_fat[rg_z].sum() * in_plane), 2)
    if save_masks_callback is not None:
        save_masks_callback("fat_retropharyngeal_regional", rp_fat)

    # ---- Facial / buccal (optional, FOV-dependent) ----
    if cfg.enable_facial_fat:
        facial = _facial_buccal_fat(image=image, body_mask=body_mask,
                                     fat_voxels=fat_voxels, airway=airway)
        out.update(facial)

    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _empty_row() -> dict[str, object]:
    base = {
        "fat_cervical_area_at_hyoid_level_mm2": _NAN,
        "fat_cervical_area_at_retropalatal_level_mm2": _NAN,
        "fat_cervical_area_at_retroglossal_level_mm2": _NAN,
        "fat_facial_total_volume_ml": _NAN,
        "fat_buccal_left_volume_ml": _NAN,
        "fat_buccal_right_volume_ml": _NAN,
        "fat_facial_to_parapharyngeal_ratio": _NAN,
    }
    for level in ("retropalatal", "retroglossal", "subglosso_supraglottic"):
        for side in ("left", "right", "total"):
            base[f"fat_parapharyngeal_area_{level}_{side}_mm2"] = _NAN
        base[f"fat_parapharyngeal_to_airway_ratio_{level}"] = _NAN
    base["fat_retropharyngeal_area_at_retropalatal_level_mm2"] = _NAN
    base["fat_retropharyngeal_area_at_retroglossal_level_mm2"] = _NAN
    return base


def _subglosso_anchor(rg_z: Optional[int], sz_mm: float) -> Optional[int]:
    if rg_z is None:
        return None
    # ~15 mm below the RG level
    return rg_z + max(1, int(round(15.0 / sz_mm)))


def _per_side_parapharyngeal(
    *, image: CTAImage, airway_mask: np.ndarray, anchor_z: int,
    lateral_band_mm: float, window_mm: float,
    body_mask: np.ndarray, fat_voxels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz_mm = image.spacing_xyz_mm
    band_x = mm_to_voxels(lateral_band_mm, sx)
    half_z = mm_to_voxels(window_mm / 2.0, sz_mm)
    sz_image = airway_mask.shape[0]
    left = np.zeros_like(airway_mask)
    right = np.zeros_like(airway_mask)
    z_lo = max(0, anchor_z - half_z)
    z_hi = min(sz_image - 1, anchor_z + half_z)
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
    left = left & fat_voxels & body_mask
    right = right & fat_voxels & body_mask
    return left, right


def _retropharyngeal_band(
    *, image: CTAImage, airway_mask: np.ndarray,
    posterior_band_mm: float, window_mm: float,
    anchor_z: Optional[int],
) -> np.ndarray:
    sx, sy, sz_mm = image.spacing_xyz_mm
    band_y = mm_to_voxels(posterior_band_mm, sy)
    half_z = mm_to_voxels(window_mm / 2.0, sz_mm)
    sz_image = airway_mask.shape[0]
    if anchor_z is None or not (0 <= anchor_z < sz_image):
        zs = np.where(airway_mask.any(axis=(1, 2)))[0]
        if zs.size == 0:
            return np.zeros_like(airway_mask)
        anchor_z = int(zs[len(zs) // 2])
    out = np.zeros_like(airway_mask)
    z_lo = max(0, anchor_z - half_z)
    z_hi = min(sz_image - 1, anchor_z + half_z)
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        y_back = int(ys.max())
        y_lo = min(sl.shape[0] - 1, y_back + 1)
        y_hi = min(sl.shape[0] - 1, y_back + band_y)
        x_lo = max(0, int(xs.min()) - 2)
        x_hi = min(sl.shape[1] - 1, int(xs.max()) + 2)
        if y_hi >= y_lo and x_hi >= x_lo:
            out[z, y_lo:y_hi + 1, x_lo:x_hi + 1] = True
    return out


def _facial_buccal_fat(
    *, image: CTAImage, body_mask: np.ndarray, fat_voxels: np.ndarray,
    airway: AirwayMaskInfo,
) -> dict[str, object]:
    """Naive face / buccal fat = fat voxels in the upper 1/3 of body extent,
    split by axial midline."""
    sx, sy, _ = image.spacing_xyz_mm
    zs = np.where(body_mask.any(axis=(1, 2)))[0]
    if zs.size == 0:
        return {}
    z_lo, z_hi = int(zs.min()), int(zs.max())
    third = max(1, (z_hi - z_lo + 1) // 3)
    upper = np.zeros_like(body_mask)
    upper[z_lo:z_lo + third] = True
    facial_fat = fat_voxels & body_mask & upper
    if not facial_fat.any():
        return {}
    vol_mm3 = float(int(facial_fat.sum()) * image.voxel_volume_mm3)
    midx = facial_fat.shape[2] // 2
    left = facial_fat.copy()
    left[..., midx:] = False
    right = facial_fat & ~left
    out = {
        "fat_facial_total_volume_ml": round(vol_mm3 / 1000.0, 4),
        "fat_buccal_left_volume_ml":
            round(float(int(left.sum()) * image.voxel_volume_mm3) / 1000.0, 4),
        "fat_buccal_right_volume_ml":
            round(float(int(right.sum()) * image.voxel_volume_mm3) / 1000.0, 4),
    }
    return out
