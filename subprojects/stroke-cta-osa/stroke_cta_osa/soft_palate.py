"""Soft palate, uvula, lateral pharyngeal wall, and palatine-tonsil features.

All features here are mask- or landmark-driven; the module never invents a
mask. Three input modes:

  1. External masks (soft_palate, uvula, palatine_tonsil_left/right,
     lateral_wall_left/right) — each is optional and independent.
  2. Landmark-only length and thickness (soft palate length from PNS to
     uvula tip; lateral wall thickness from coarse cross-section).
  3. Missing — every feature NaN.

Lateral pharyngeal wall thickness is computed from a thin band lateral to
the airway at the retropalatal level, using a body mask as the outer
bound and the airway as the inner bound. The thickness is the median
radial distance from airway L/R extreme to the body silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import mm_to_voxels, slice_area_mm2
from .landmark_schema import LandmarkBundle
from .landmarks import (
    get_hyoid_position, get_retroglossal_level, get_retropalatal_level,
)
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("soft_palate")

_NAN = float("nan")


@dataclass
class SoftTissueConfig:
    enabled: bool = True
    require_masks_for_volumes: bool = True
    allow_landmark_length_fallback: bool = True
    lateral_wall_band_mm: float = 15.0
    lateral_wall_axial_window_mm: float = 20.0
    body_air_threshold_hu: float = -250.0


def compute_soft_palate_features(
    image: CTAImage,
    cfg: SoftTissueConfig,
    *,
    soft_palate_mask: Optional[np.ndarray] = None,
    uvula_mask: Optional[np.ndarray] = None,
    palatine_tonsil_left_mask: Optional[np.ndarray] = None,
    palatine_tonsil_right_mask: Optional[np.ndarray] = None,
    landmarks: Optional[LandmarkBundle] = None,
    airway: Optional[AirwayMaskInfo] = None,
    body_mask: Optional[np.ndarray] = None,
    save_masks_callback=None,
) -> dict[str, object]:
    out: dict[str, object] = _empty_row()
    if not cfg.enabled:
        return out

    # ---- Soft palate mask block ----
    if soft_palate_mask is not None and np.asarray(soft_palate_mask).any():
        m = np.asarray(soft_palate_mask).astype(bool)
        out["soft_palate_mask_available"] = True
        n = int(m.sum())
        out["soft_palate_volume_ml"] = round(
            float(n * image.voxel_volume_mm3) / 1000.0, 4)
        hu = image.array[m].astype(np.float32)
        out["soft_palate_mean_hu"] = round(float(hu.mean()), 2)
        # Length = z extent of mask in mm
        zs = np.where(m.any(axis=(1, 2)))[0]
        if zs.size:
            out["soft_palate_length_mm"] = round(
                float((zs.max() - zs.min() + 1) * image.spacing_xyz_mm[2]), 2)
            out["soft_palate_inferior_tip_z_mm"] = round(
                float(zs.max() * image.spacing_xyz_mm[2]
                      + image.origin_xyz_mm[2]), 2)
        # Thickness: per-slice max y-extent, in mm
        sy = image.spacing_xyz_mm[1]
        thicknesses = []
        for z in zs:
            ys = np.where(m[z].any(axis=1))[0]
            if ys.size:
                thicknesses.append((ys.max() - ys.min() + 1) * sy)
        if thicknesses:
            out["soft_palate_thickness_max_mm"] = round(max(thicknesses), 2)
            out["soft_palate_thickness_mean_mm"] = round(
                float(np.mean(thicknesses)), 2)
        if save_masks_callback is not None:
            save_masks_callback("soft_palate", m)
    elif landmarks is not None and cfg.allow_landmark_length_fallback:
        # Landmark-only length: PNS → uvula tip.
        pts = landmarks.points
        pns = pts.get("posterior_nasal_spine")
        uvula = pts.get("uvula_tip")
        if pns and uvula and pns.physical_mm and uvula.physical_mm:
            d = float(np.linalg.norm(
                np.array(pns.physical_mm) - np.array(uvula.physical_mm)))
            out["soft_palate_length_mm"] = round(d, 2)
            out["soft_palate_mask_available"] = False

    # ---- Uvula mask block ----
    if uvula_mask is not None and np.asarray(uvula_mask).any():
        m = np.asarray(uvula_mask).astype(bool)
        out["uvula_visible"] = True
        n = int(m.sum())
        out["uvula_volume_ml"] = round(
            float(n * image.voxel_volume_mm3) / 1000.0, 4)
        # Length / width approximations
        zs = np.where(m.any(axis=(1, 2)))[0]
        if zs.size:
            out["uvula_length_mm"] = round(
                float((zs.max() - zs.min() + 1) * image.spacing_xyz_mm[2]), 2)
        ys = np.where(m.any(axis=(0, 2)))[0]
        if ys.size:
            sx = image.spacing_xyz_mm[0]
            xs = np.where(m.any(axis=(0, 1)))[0]
            if xs.size:
                out["uvula_width_mm"] = round(
                    float((xs.max() - xs.min() + 1) * sx), 2)
        if save_masks_callback is not None:
            save_masks_callback("uvula", m)

    # ---- Palatine tonsils ----
    if palatine_tonsil_left_mask is not None and np.asarray(palatine_tonsil_left_mask).any():
        m = np.asarray(palatine_tonsil_left_mask).astype(bool)
        out["palatine_tonsil_left_visible"] = True
        out["palatine_tonsil_left_volume_ml"] = round(
            float(int(m.sum()) * image.voxel_volume_mm3) / 1000.0, 4)
        if save_masks_callback is not None:
            save_masks_callback("palatine_tonsil_left", m)
    if palatine_tonsil_right_mask is not None and np.asarray(palatine_tonsil_right_mask).any():
        m = np.asarray(palatine_tonsil_right_mask).astype(bool)
        out["palatine_tonsil_right_visible"] = True
        out["palatine_tonsil_right_volume_ml"] = round(
            float(int(m.sum()) * image.voxel_volume_mm3) / 1000.0, 4)
        if save_masks_callback is not None:
            save_masks_callback("palatine_tonsil_right", m)
    if (out["palatine_tonsil_left_visible"]
            or out["palatine_tonsil_right_visible"]):
        l = out["palatine_tonsil_left_volume_ml"]
        r = out["palatine_tonsil_right_volume_ml"]
        total = (l if isinstance(l, float) and l == l else 0.0) + \
                (r if isinstance(r, float) and r == r else 0.0)
        out["palatine_tonsil_total_volume_ml"] = round(total, 4)

    # ---- Lateral wall thickness (airway + body) ----
    if (airway is not None and airway.is_present
            and body_mask is not None and body_mask.any()):
        lateral = _lateral_wall_thickness(
            image=image, airway_mask=airway.mask_zyx, body_mask=body_mask,
            band_mm=cfg.lateral_wall_band_mm,
            window_mm=cfg.lateral_wall_axial_window_mm,
            landmarks=landmarks,
        )
        out.update(lateral)
    return out


def _empty_row() -> dict[str, object]:
    return {
        "soft_palate_mask_available": False,
        "soft_palate_length_mm": _NAN,
        "soft_palate_thickness_max_mm": _NAN,
        "soft_palate_thickness_mean_mm": _NAN,
        "soft_palate_volume_ml": _NAN,
        "soft_palate_mean_hu": _NAN,
        "soft_palate_inferior_tip_z_mm": _NAN,
        "soft_palate_to_posterior_pharyngeal_wall_distance_mm": _NAN,
        "soft_palate_to_retropalatal_airway_ratio": _NAN,
        "uvula_visible": False,
        "uvula_length_mm": _NAN,
        "uvula_width_mm": _NAN,
        "uvula_volume_ml": _NAN,
        "lateral_pharyngeal_wall_left_thickness_mm": _NAN,
        "lateral_pharyngeal_wall_right_thickness_mm": _NAN,
        "lateral_pharyngeal_wall_mean_thickness_mm": _NAN,
        "lateral_pharyngeal_wall_asymmetry_index": _NAN,
        "lateral_wall_to_airway_ratio_at_retropalatal_level": _NAN,
        "lateral_wall_to_airway_ratio_at_retroglossal_level": _NAN,
        "palatine_tonsil_left_visible": False,
        "palatine_tonsil_right_visible": False,
        "palatine_tonsil_left_volume_ml": _NAN,
        "palatine_tonsil_right_volume_ml": _NAN,
        "palatine_tonsil_total_volume_ml": _NAN,
        "tonsil_to_retropalatal_airway_ratio": _NAN,
    }


def _lateral_wall_thickness(
    *, image: CTAImage, airway_mask: np.ndarray, body_mask: np.ndarray,
    band_mm: float, window_mm: float,
    landmarks: Optional[LandmarkBundle],
) -> dict[str, float]:
    sx, sy, sz_mm = image.spacing_xyz_mm
    band_voxels = mm_to_voxels(band_mm, sx)
    half_z = mm_to_voxels(window_mm / 2.0, sz_mm)
    rp = get_retropalatal_level(landmarks) if landmarks else None
    rg = get_retroglossal_level(landmarks) if landmarks else None
    centre = rp if rp is not None else rg
    if centre is None:
        return {}

    sz_image = airway_mask.shape[0]
    z_lo = max(0, centre - half_z)
    z_hi = min(sz_image - 1, centre + half_z)

    left_thicknesses: list[float] = []
    right_thicknesses: list[float] = []
    for z in range(z_lo, z_hi + 1):
        sl_air = airway_mask[z]
        sl_body = body_mask[z]
        if not sl_air.any() or not sl_body.any():
            continue
        ys, xs = np.where(sl_air)
        y_mid = int(np.median(ys))
        x_lo = max(0, int(xs.min()) - band_voxels)
        x_hi = min(sl_air.shape[1] - 1, int(xs.max()) + band_voxels)
        # Left side: from airway L wall outward to body silhouette
        for direction, side_list in (("left", left_thicknesses),
                                      ("right", right_thicknesses)):
            if direction == "left":
                start_x = int(xs.min())
                step = -1
                bound = x_lo
            else:
                start_x = int(xs.max())
                step = 1
                bound = x_hi
            count = 0
            cx = start_x + step
            while (step > 0 and cx <= bound) or (step < 0 and cx >= bound):
                if 0 <= cx < sl_body.shape[1] and sl_body[y_mid, cx]:
                    count += 1
                cx += step
            if count > 0:
                side_list.append(count * sx)
    if not left_thicknesses and not right_thicknesses:
        return {}
    out: dict[str, float] = {}
    if left_thicknesses:
        out["lateral_pharyngeal_wall_left_thickness_mm"] = round(
            float(np.median(left_thicknesses)), 2)
    if right_thicknesses:
        out["lateral_pharyngeal_wall_right_thickness_mm"] = round(
            float(np.median(right_thicknesses)), 2)
    all_t = left_thicknesses + right_thicknesses
    out["lateral_pharyngeal_wall_mean_thickness_mm"] = round(
        float(np.mean(all_t)), 2)
    l_t = float(np.median(left_thicknesses)) if left_thicknesses else _NAN
    r_t = float(np.median(right_thicknesses)) if right_thicknesses else _NAN
    if l_t == l_t and r_t == r_t and (l_t + r_t) > 0:
        out["lateral_pharyngeal_wall_asymmetry_index"] = round(
            (r_t - l_t) / (r_t + l_t), 4)
    return out
