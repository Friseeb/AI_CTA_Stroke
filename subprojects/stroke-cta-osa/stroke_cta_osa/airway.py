"""Airway geometry features.

All cross-sectional areas in v1 are AXIAL approximations: we count voxels in
the airway mask per axial slice and multiply by in-plane voxel area.
Centerline-orthogonal CSAs (enabled by `cfg.airway.centerline_orthogonal_csa`)
are TODO — the column `airway_csa_orientation` records which method produced
the numbers so downstream analyses can stratify.

Region landmarks come from the shared adapter payload when available; when
they are not, retropalatal/retroglossal/retrolingual outputs are emitted as
NaN with `airway_region_method = 'unavailable'`. We deliberately do NOT
invent fake landmarks from the airway mask alone — the geometric meaning
of "retroglossal" becomes ambiguous without the tongue/palate references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .geometry import mm_to_voxels, slice_area_mm2, slice_thickness_mm, z_index_to_mm
from .logging_utils import get_logger
from .shared_schema import SharedAirwayLandmarks
from .types import AirwayMaskInfo, CTAImage

log = get_logger("airway")


_NAN = float("nan")


@dataclass
class AirwayGeometry:
    features: dict[str, float | str | None]


# --- Public entrypoint ------------------------------------------------------

def compute_airway_features(
    image: CTAImage,
    mask_info: Optional[AirwayMaskInfo],
    landmarks: SharedAirwayLandmarks,
    retropalatal_window_mm: float = 15.0,
    retroglossal_window_mm: float = 15.0,
    retrolingual_window_mm: float = 10.0,
) -> AirwayGeometry:
    if mask_info is None or not mask_info.is_present:
        return AirwayGeometry(features=_missing_airway_features(method=None))

    mask = mask_info.mask_zyx.astype(bool)
    spacing = image.spacing_xyz_mm
    in_plane_area = slice_area_mm2(spacing)
    dz = slice_thickness_mm(spacing)

    # Per-slice voxel counts (axial CSA proxy)
    per_slice_vox = mask.sum(axis=(1, 2)).astype(int)  # (z,)
    csa_mm2 = per_slice_vox * in_plane_area
    nonzero_mask = csa_mm2 > 0
    nonzero_csa = csa_mm2[nonzero_mask]

    if nonzero_csa.size == 0:
        return AirwayGeometry(features=_missing_airway_features(method=mask_info.method))

    # Length: number of axial slices that contain any airway voxel × slice
    # thickness. This is a vertical extent, not a curved length — flagged
    # accordingly in feature metadata.
    airway_length_mm = float(nonzero_csa.size * dz)

    # Volume in mL
    volume_mm3 = float(per_slice_vox.sum()) * image.voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    # Min CSA and percentiles
    min_idx_in_nonzero = int(np.argmin(nonzero_csa))
    min_csa_mm2 = float(nonzero_csa[min_idx_in_nonzero])

    # Map back from "nonzero index" to original z
    nonzero_z_indices = np.where(nonzero_mask)[0]
    min_csa_z = int(nonzero_z_indices[min_idx_in_nonzero])
    min_csa_z_mm = z_index_to_mm(min_csa_z, image.origin_xyz_mm, spacing)

    p05 = float(np.percentile(nonzero_csa, 5))
    p10 = float(np.percentile(nonzero_csa, 10))
    p25 = float(np.percentile(nonzero_csa, 25))
    p50 = float(np.percentile(nonzero_csa, 50))

    # Lateral / AP diameters at min slice (axis-aligned bounding box of the
    # mask in that slice — an approximation, not principal-axis fit)
    slice_at_min = mask[min_csa_z]
    if slice_at_min.any():
        ys, xs = np.where(slice_at_min)
        lateral_d_mm = float((xs.max() - xs.min() + 1) * spacing[0])
        ap_d_mm = float((ys.max() - ys.min() + 1) * spacing[1])
        eccentricity = (
            math.sqrt(1.0 - (min(lateral_d_mm, ap_d_mm) / max(lateral_d_mm, ap_d_mm)) ** 2)
            if max(lateral_d_mm, ap_d_mm) > 0 else _NAN
        )
    else:
        lateral_d_mm = ap_d_mm = eccentricity = _NAN

    # Region windows (retropalatal / retroglossal / retrolingual) require
    # landmarks. When they're missing we leave NaN; we don't invent.
    region_method = "landmarked" if (
        landmarks.posterior_nasal_spine or landmarks.soft_palate_inferior
        or landmarks.epiglottis_tip or landmarks.hyoid
    ) else "unavailable"

    rp_csa, rp_vol = _region_csa_and_volume(
        landmark_zyx=landmarks.posterior_nasal_spine or landmarks.soft_palate_inferior,
        window_mm=retropalatal_window_mm, per_slice_vox=per_slice_vox,
        in_plane_area=in_plane_area, dz=dz,
    )
    rg_csa, rg_vol = _region_csa_and_volume(
        landmark_zyx=landmarks.epiglottis_tip or landmarks.hyoid,
        window_mm=retroglossal_window_mm, per_slice_vox=per_slice_vox,
        in_plane_area=in_plane_area, dz=dz,
    )
    rl_csa, rl_vol = _region_csa_and_volume(
        landmark_zyx=landmarks.hyoid,
        window_mm=retrolingual_window_mm, per_slice_vox=per_slice_vox,
        in_plane_area=in_plane_area, dz=dz,
    )

    features: dict[str, float | str | None] = {
        "airway_mask_available": True,
        "airway_method": mask_info.method,
        "airway_confidence": mask_info.confidence,
        "airway_csa_orientation": "axial_approximation",
        "airway_volume_mm3": round(volume_mm3, 2),
        "airway_volume_ml": round(volume_ml, 3),
        "airway_length_mm": round(airway_length_mm, 2),
        "airway_min_csa_mm2": round(min_csa_mm2, 2),
        "airway_min_csa_slice_index": int(min_csa_z),
        "airway_min_csa_z_mm": round(float(min_csa_z_mm), 2),
        "airway_csa_p05_mm2": round(p05, 2),
        "airway_csa_p10_mm2": round(p10, 2),
        "airway_csa_p25_mm2": round(p25, 2),
        "airway_csa_median_mm2": round(p50, 2),
        "airway_lateral_diameter_min_mm": _round_or_nan(lateral_d_mm, 2),
        "airway_ap_diameter_min_mm": _round_or_nan(ap_d_mm, 2),
        "airway_eccentricity_at_min_csa": _round_or_nan(eccentricity, 3),
        "airway_region_method": region_method,
        "retropalatal_csa_mm2": _round_or_nan(rp_csa, 2),
        "retroglossal_csa_mm2": _round_or_nan(rg_csa, 2),
        "retrolingual_csa_mm2": _round_or_nan(rl_csa, 2),
        "retropalatal_volume_ml": _round_or_nan(rp_vol, 3),
        "retroglossal_volume_ml": _round_or_nan(rg_vol, 3),
    }
    return AirwayGeometry(features=features)


# --- Helpers ----------------------------------------------------------------

def _region_csa_and_volume(
    landmark_zyx: Optional[tuple[int, int, int]],
    window_mm: float,
    per_slice_vox: np.ndarray,
    in_plane_area: float,
    dz: float,
) -> tuple[float, float]:
    """Mean CSA and volume within ±window_mm/2 of a landmark's z index.

    Returns (NaN, NaN) if the landmark is missing.
    """
    if landmark_zyx is None:
        return _NAN, _NAN
    z0 = int(landmark_zyx[0])
    half = mm_to_voxels(window_mm / 2.0, dz)
    z_lo = max(0, z0 - half)
    z_hi = min(per_slice_vox.shape[0] - 1, z0 + half)
    band = per_slice_vox[z_lo:z_hi + 1]
    if band.size == 0 or not (band > 0).any():
        return _NAN, _NAN
    nonzero = band[band > 0]
    csa_mm2 = float(nonzero.mean() * in_plane_area)
    volume_ml = float(band.sum() * in_plane_area * dz) / 1000.0
    return csa_mm2, volume_ml


def _missing_airway_features(method: Optional[str]) -> dict[str, float | str | None]:
    """Stable feature row for the no-mask case — every key present, all NaN."""
    return {
        "airway_mask_available": False,
        "airway_method": method or "null",
        "airway_confidence": "none",
        "airway_csa_orientation": "n/a",
        "airway_volume_mm3": _NAN,
        "airway_volume_ml": _NAN,
        "airway_length_mm": _NAN,
        "airway_min_csa_mm2": _NAN,
        "airway_min_csa_slice_index": -1,
        "airway_min_csa_z_mm": _NAN,
        "airway_csa_p05_mm2": _NAN,
        "airway_csa_p10_mm2": _NAN,
        "airway_csa_p25_mm2": _NAN,
        "airway_csa_median_mm2": _NAN,
        "airway_lateral_diameter_min_mm": _NAN,
        "airway_ap_diameter_min_mm": _NAN,
        "airway_eccentricity_at_min_csa": _NAN,
        "airway_region_method": "unavailable",
        "retropalatal_csa_mm2": _NAN,
        "retroglossal_csa_mm2": _NAN,
        "retrolingual_csa_mm2": _NAN,
        "retropalatal_volume_ml": _NAN,
        "retroglossal_volume_ml": _NAN,
    }


def _round_or_nan(value: float, ndigits: int) -> float:
    if value != value:  # NaN guard
        return _NAN
    return round(float(value), ndigits)
