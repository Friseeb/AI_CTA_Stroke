"""Regional airway compartments + shape features.

This module *extends* the existing :mod:`airway` module with:

  * five anatomical compartments (nasopharyngeal, retropalatal, retroglossal,
    retrolingual, hypopharyngeal) bounded by landmark z-levels;
  * shape features at the min-CSA slice (circularity, narrowing, AP/lateral
    ratios) that are stable additions to the existing AP / lateral / eccentricity;
  * airway-vs-tongue cross-features (RG airway area / tongue base area, etc.).

The module is purely additive — it does not modify the existing global
features in :mod:`airway`. Callers merge both dicts into the case row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import mm_to_voxels, slice_area_mm2, slice_thickness_mm
from .landmark_schema import LandmarkBundle
from .landmarks import (
    get_retroglossal_level, get_retropalatal_level, get_tongue_base_level,
    infer_region_levels_from_landmarks,
)
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("airway_regions")

_NAN = float("nan")


@dataclass
class AirwayRegionConfig:
    enabled: bool = True
    prefer_landmark_defined_regions: bool = True
    allow_axial_approximation: bool = True
    save_csa_profile: bool = False


def compute_regional_airway_features(
    image: CTAImage,
    cfg: AirwayRegionConfig,
    airway: Optional[AirwayMaskInfo],
    landmarks: LandmarkBundle,
    *,
    tongue_mask: Optional[np.ndarray] = None,
    tongue_volume_ml: Optional[float] = None,
    csa_profile_path: Optional[str] = None,
) -> dict[str, object]:
    out: dict[str, object] = _empty_row()
    if not cfg.enabled or airway is None or not airway.is_present:
        return out

    mask = airway.mask_zyx
    sx, sy, sz_mm = image.spacing_xyz_mm
    in_plane = slice_area_mm2(image.spacing_xyz_mm)
    per_slice_vox = mask.sum(axis=(1, 2)).astype(int)
    nonzero_z = np.where(per_slice_vox > 0)[0]
    if nonzero_z.size == 0:
        return out

    # Region z-bounds (voxel indices) ---------------------------------------
    bounds = _region_bounds(image, mask, landmarks, cfg)
    out["airway_region_method"] = bounds["method"]

    # Per-compartment volume + min CSA --------------------------------------
    for compartment in ("nasopharyngeal", "retropalatal", "retroglossal",
                        "retrolingual", "hypopharyngeal"):
        z_lo, z_hi = bounds.get(compartment, (None, None))
        if z_lo is None or z_hi is None or z_hi < z_lo:
            continue
        band = per_slice_vox[z_lo:z_hi + 1]
        if not (band > 0).any():
            continue
        nonzero = band[band > 0]
        volume_ml = float(band.sum() * in_plane * sz_mm) / 1000.0
        min_csa_mm2 = float(nonzero.min() * in_plane)
        out[f"{compartment}_volume_ml"] = round(volume_ml, 4)
        out[f"{compartment}_min_csa_mm2"] = round(min_csa_mm2, 2)

    # Standard-level CSAs ---------------------------------------------------
    rp_z = get_retropalatal_level(landmarks)
    rg_z = get_retroglossal_level(landmarks)
    if rp_z is not None and 0 <= rp_z < mask.shape[0]:
        out["retropalatal_csa_at_standard_level_mm2"] = round(
            float(per_slice_vox[rp_z] * in_plane), 2)
    if rg_z is not None and 0 <= rg_z < mask.shape[0]:
        out["retroglossal_csa_at_standard_level_mm2"] = round(
            float(per_slice_vox[rg_z] * in_plane), 2)

    # Min-CSA region label
    min_z_global = int(np.argmin(np.where(per_slice_vox > 0, per_slice_vox,
                                            np.iinfo(np.int64).max)))
    out["airway_min_csa_region"] = _slice_to_region(min_z_global, bounds)

    # Shape extensions at min CSA -------------------------------------------
    shape = _shape_at_min_csa(mask, image, min_z_global)
    out.update(shape)

    # Lateral narrowing across all slices (median of LAT/AP)
    nar = _lateral_narrowing_index(mask, image)
    out["airway_lateral_narrowing_index"] = nar

    # Number of profile slices
    out["airway_area_profile_n_slices"] = int(nonzero_z.size)
    out["airway_centerline_available"] = False
    if cfg.save_csa_profile and csa_profile_path:
        _save_profile(csa_profile_path, per_slice_vox, in_plane,
                      sz_mm, image.origin_xyz_mm[2])
        out["airway_csa_profile_json_path"] = csa_profile_path

    # Airway × tongue cross features ---------------------------------------
    if (rg_z is not None and 0 <= rg_z < mask.shape[0]
            and tongue_mask is not None
            and 0 <= rg_z < np.asarray(tongue_mask).shape[0]):
        air_area = float(per_slice_vox[rg_z] * in_plane)
        tb_area = float(np.asarray(tongue_mask)[rg_z].sum() * in_plane)
        if tb_area > 0:
            out["retroglossal_airway_to_tongue_base_area_ratio"] = round(
                air_area / tb_area, 4)
    if (tongue_volume_ml is not None and tongue_volume_ml > 0
            and isinstance(out.get("retroglossal_volume_ml"), float)
            and out["retroglossal_volume_ml"] == out["retroglossal_volume_ml"]):
        out["retroglossal_airway_to_tongue_volume_ratio"] = round(
            float(out["retroglossal_volume_ml"]) / float(tongue_volume_ml), 4)

    tongue_base_z = get_tongue_base_level(landmarks)
    if tongue_base_z is not None:
        out["airway_min_csa_adjacent_to_tongue_base_flag"] = bool(
            abs(int(min_z_global) - int(tongue_base_z)) <= 3
        )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _empty_row() -> dict[str, object]:
    base = {
        "airway_region_method": "unavailable",
        "airway_min_csa_region": "",
        "airway_lateral_narrowing_index": _NAN,
        "airway_circularity_at_min_csa": _NAN,
        "airway_ap_to_lateral_ratio_at_min_csa": _NAN,
        "airway_concentricity_index": _NAN,
        "airway_centerline_available": False,
        "airway_area_profile_n_slices": -1,
        "airway_csa_profile_json_path": "",
        "retropalatal_csa_at_standard_level_mm2": _NAN,
        "retroglossal_csa_at_standard_level_mm2": _NAN,
        "retroglossal_airway_to_tongue_base_area_ratio": _NAN,
        "retroglossal_airway_to_tongue_volume_ratio": _NAN,
        "airway_min_csa_adjacent_to_tongue_base_flag": False,
    }
    for c in ("nasopharyngeal", "retropalatal", "retroglossal",
              "retrolingual", "hypopharyngeal"):
        base[f"{c}_volume_ml"] = _NAN
        base[f"{c}_min_csa_mm2"] = _NAN
    return base


def _region_bounds(
    image: CTAImage,
    mask: np.ndarray,
    landmarks: LandmarkBundle,
    cfg: AirwayRegionConfig,
) -> dict:
    """Return per-compartment (z_lo, z_hi) tuples and a `method` string.

    Bounds are derived from landmark z-levels in a single chain so each
    compartment is a slab in z. When landmarks are missing we fall back to
    splitting the airway extent into thirds (top = nasopharyngeal, middle =
    retropalatal+retroglossal, bottom = hypopharyngeal) — the method string
    records this so downstream consumers can stratify.
    """
    nonzero_z = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    if nonzero_z.size == 0:
        return {"method": "no_airway"}
    z_lo_all, z_hi_all = int(nonzero_z.min()), int(nonzero_z.max())

    levels = infer_region_levels_from_landmarks(landmarks)
    hp = levels.get("hard_palate")
    rp = levels.get("retropalatal_level")
    rg = levels.get("retroglossal_level")
    tb = levels.get("tongue_base_level")
    lar = levels.get("laryngeal_inlet_level")

    if any(v is not None for v in (hp, rp, rg, tb, lar)):
        method = "landmark_z_levels"
        # Pick z bounds inside the airway extent; fall back to nearest within range
        def _clamp(z):
            return int(min(max(z, z_lo_all), z_hi_all)) if z is not None else None
        hp = _clamp(hp); rp = _clamp(rp); rg = _clamp(rg); tb = _clamp(tb); lar = _clamp(lar)
        bounds = {}
        # nasopharyngeal: top of airway → hp
        if hp is not None:
            bounds["nasopharyngeal"] = (z_lo_all, hp)
        # retropalatal: hp → rp
        if hp is not None and rp is not None:
            bounds["retropalatal"] = (min(hp, rp), max(hp, rp))
        elif rp is not None:
            half_band = max(1, int(round(15.0 / image.spacing_xyz_mm[2])))
            bounds["retropalatal"] = (max(z_lo_all, rp - half_band),
                                       min(z_hi_all, rp + half_band))
        # retroglossal: rp → rg
        if rp is not None and rg is not None:
            bounds["retroglossal"] = (min(rp, rg), max(rp, rg))
        elif rg is not None:
            half_band = max(1, int(round(15.0 / image.spacing_xyz_mm[2])))
            bounds["retroglossal"] = (max(z_lo_all, rg - half_band),
                                       min(z_hi_all, rg + half_band))
        # retrolingual: rg → tb
        if rg is not None and tb is not None:
            bounds["retrolingual"] = (min(rg, tb), max(rg, tb))
        # hypopharyngeal: tb → laryngeal_inlet_level → bottom of airway
        if tb is not None:
            bottom = lar if lar is not None else z_hi_all
            bounds["hypopharyngeal"] = (min(tb, bottom), max(tb, bottom))
        bounds["method"] = method
        return bounds

    # No landmarks → split by thirds
    extent = z_hi_all - z_lo_all + 1
    third = max(1, extent // 3)
    return {
        "nasopharyngeal": (z_lo_all, z_lo_all + third - 1),
        "retropalatal": (z_lo_all + third, z_lo_all + 2 * third - 1),
        "retroglossal": (z_lo_all + third, z_lo_all + 2 * third - 1),
        "retrolingual": (z_lo_all + 2 * third, z_hi_all),
        "hypopharyngeal": (z_lo_all + 2 * third, z_hi_all),
        "method": "airway_thirds_fallback",
    }


def _slice_to_region(z: int, bounds: dict) -> str:
    # Landmark-derived slabs can overlap because broad nasopharyngeal extent is
    # retained for volume summaries. For a single-slice assignment, prefer the
    # narrower lower-airway compartments before the broad fallback slab.
    for compartment in ("retrolingual", "hypopharyngeal", "retroglossal",
                        "retropalatal", "nasopharyngeal"):
        b = bounds.get(compartment)
        if b is None:
            continue
        lo, hi = b
        if lo <= z <= hi:
            return compartment
    return ""


def _shape_at_min_csa(
    mask: np.ndarray, image: CTAImage, z: int,
) -> dict[str, float]:
    sl = mask[z]
    if not sl.any():
        return {
            "airway_circularity_at_min_csa": _NAN,
            "airway_ap_to_lateral_ratio_at_min_csa": _NAN,
        }
    ys, xs = np.where(sl)
    sx, sy, _ = image.spacing_xyz_mm
    lat = float((xs.max() - xs.min() + 1) * sx)
    ap = float((ys.max() - ys.min() + 1) * sy)
    ratio = ap / max(lat, 1e-9)
    # Approximate area and perimeter from voxel-edge perimeter for circularity
    area = float(sl.sum() * sx * sy)
    # 4-neighbour perimeter, scaled by in-plane spacing (a coarse approximation)
    from scipy import ndimage as ndi
    boundary = sl & ~ndi.binary_erosion(sl)
    perimeter = float(boundary.sum() * 0.5 * (sx + sy))
    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter ** 2)
    else:
        circularity = _NAN
    return {
        "airway_circularity_at_min_csa": round(float(circularity), 3) if circularity == circularity else _NAN,
        "airway_ap_to_lateral_ratio_at_min_csa": round(float(ratio), 3),
    }


def _lateral_narrowing_index(mask: np.ndarray, image: CTAImage) -> float:
    sx, sy, _ = image.spacing_xyz_mm
    lats: list[float] = []
    aps: list[float] = []
    for z in range(mask.shape[0]):
        sl = mask[z]
        if not sl.any():
            continue
        ys, xs = np.where(sl)
        lats.append((xs.max() - xs.min() + 1) * sx)
        aps.append((ys.max() - ys.min() + 1) * sy)
    if not lats or not aps:
        return _NAN
    return round(float(np.median(lats) / max(np.median(aps), 1e-9)), 3)


def _save_profile(path: str, per_slice_vox: np.ndarray, in_plane: float,
                  dz: float, z0: float) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nonzero = [
        {
            "z_index": int(i),
            "z_mm": round(float(z0 + i * dz), 2),
            "csa_mm2": round(float(per_slice_vox[i] * in_plane), 2),
        }
        for i in range(per_slice_vox.size)
        if per_slice_vox[i] > 0
    ]
    Path(path).write_text(json.dumps({"slices": nonzero}, indent=2))
