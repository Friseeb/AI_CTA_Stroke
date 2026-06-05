"""Skeletal / hyoid geometry features.

All distances are computed in physical mm, regardless of voxel anisotropy.
Every feature falls back to NaN when its input landmarks/masks are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .landmark_schema import LandmarkBundle
from .landmarks import voxel_to_physical
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("skeletal")

_NAN = float("nan")


@dataclass
class SkeletalConfig:
    enabled: bool = True
    allow_landmark_only_distances: bool = True
    allow_hyoid_threshold_fallback: bool = False


def compute_skeletal_features(
    image: CTAImage,
    cfg: SkeletalConfig,
    landmarks: LandmarkBundle,
    *,
    airway: Optional[AirwayMaskInfo] = None,
    mandible_mask: Optional[np.ndarray] = None,
    mandibular_plane_to_hyoid_distance_mm: Optional[float] = None,
) -> dict[str, object]:
    out: dict[str, object] = _empty_row()
    if not cfg.enabled:
        return out

    pts = landmarks.points
    hyoid = pts.get("hyoid_centroid")
    if hyoid and hyoid.physical_mm:
        out["hyoid_detected"] = True
        out["hyoid_centroid_x_mm"] = round(float(hyoid.physical_mm[0]), 2)
        out["hyoid_centroid_y_mm"] = round(float(hyoid.physical_mm[1]), 2)
        out["hyoid_centroid_z_mm"] = round(float(hyoid.physical_mm[2]), 2)

    def _d(a: str, b: str) -> Optional[float]:
        if a not in pts or b not in pts:
            return None
        pa, pb = pts[a].physical_mm, pts[b].physical_mm
        if pa is None or pb is None:
            return None
        return round(float(np.linalg.norm(np.array(pa) - np.array(pb))), 2)

    out["hyoid_to_c2_distance_mm"] = _d("hyoid_centroid", "c2_centroid") or _NAN
    out["hyoid_to_c3_distance_mm"] = _d("hyoid_centroid", "c3_centroid") or _NAN
    out["hyoid_to_c4_distance_mm"] = _d("hyoid_centroid", "c4_centroid") or _NAN
    out["hyoid_to_epiglottis_distance_mm"] = _d("hyoid_centroid", "epiglottis_tip") or _NAN
    out["hard_palate_to_hyoid_distance_mm"] = (
        _d("posterior_nasal_spine", "hyoid_centroid") or _NAN
    )
    out["posterior_nasal_spine_to_epiglottis_distance_mm"] = (
        _d("posterior_nasal_spine", "epiglottis_tip") or _NAN
    )

    # Hyoid to posterior pharyngeal wall — uses airway mask if available
    if (hyoid and hyoid.voxel_zyx and airway is not None
            and airway.is_present):
        d_mm = _distance_hyoid_to_airway_posterior_wall(image, hyoid.voxel_zyx,
                                                        airway.mask_zyx)
        out["hyoid_to_posterior_pharyngeal_wall_distance_mm"] = (
            round(d_mm, 2) if d_mm is not None else _NAN
        )

    # Vertical / AP positions relative to mandible / spine
    if (hyoid and hyoid.physical_mm
            and "menton" in pts and pts["menton"].physical_mm):
        out["hyoid_vertical_position_relative_to_mandible_mm"] = round(
            float(hyoid.physical_mm[2] - pts["menton"].physical_mm[2]), 2)
    if (hyoid and hyoid.physical_mm
            and "c3_centroid" in pts and pts["c3_centroid"].physical_mm):
        out["hyoid_ap_position_relative_to_cervical_spine_mm"] = round(
            float(hyoid.physical_mm[1] - pts["c3_centroid"].physical_mm[1]), 2)

    # Neck length: hard palate plane to hyoid (mm)
    if isinstance(out["hard_palate_to_hyoid_distance_mm"], float) \
            and out["hard_palate_to_hyoid_distance_mm"] == out["hard_palate_to_hyoid_distance_mm"]:
        out["neck_length_mm"] = out["hard_palate_to_hyoid_distance_mm"]
    # Laryngeal descent — placeholder = hyoid_to_c4_distance for now
    out["laryngeal_descent_mm"] = out["hyoid_to_c4_distance_mm"]

    # Cervicomandibular ring area: airway + mandible inferior rim area at
    # hyoid level — coarse proxy
    if (hyoid and hyoid.voxel_zyx and mandible_mask is not None
            and mandible_mask.any()):
        cm_area, cm_method = _cervicomandibular_ring_area(
            image, hyoid.voxel_zyx, mandible_mask, airway,
        )
        if cm_area is not None:
            out["cervicomandibular_ring_area_mm2"] = round(float(cm_area), 2)
            out["cervicomandibular_ring_method"] = cm_method

    # Use the mandible module's distance if it already computed one
    if mandibular_plane_to_hyoid_distance_mm is not None:
        out["mandibular_plane_to_hyoid_distance_mm"] = round(
            float(mandibular_plane_to_hyoid_distance_mm), 2)
        out["mandibular_plane_available"] = True

    return out


def _empty_row() -> dict[str, object]:
    return {
        "hyoid_detected": False, "hyoid_mask_available": False,
        "hyoid_centroid_x_mm": _NAN, "hyoid_centroid_y_mm": _NAN,
        "hyoid_centroid_z_mm": _NAN,
        "hyoid_to_posterior_pharyngeal_wall_distance_mm": _NAN,
        "hyoid_to_c2_distance_mm": _NAN, "hyoid_to_c3_distance_mm": _NAN,
        "hyoid_to_c4_distance_mm": _NAN, "hyoid_to_epiglottis_distance_mm": _NAN,
        "hyoid_vertical_position_relative_to_mandible_mm": _NAN,
        "hyoid_ap_position_relative_to_cervical_spine_mm": _NAN,
        "neck_length_mm": _NAN, "laryngeal_descent_mm": _NAN,
        "hard_palate_to_hyoid_distance_mm": _NAN,
        "posterior_nasal_spine_to_epiglottis_distance_mm": _NAN,
        "cervicomandibular_ring_area_mm2": _NAN,
        "cervicomandibular_ring_method": "",
        "skeletal_enclosure_index": _NAN,
        "mandibular_plane_available": False,
        "mandibular_plane_to_hyoid_distance_mm": _NAN,
    }


def _distance_hyoid_to_airway_posterior_wall(
    image: CTAImage,
    hyoid_voxel_zyx: tuple[int, int, int],
    airway_mask: np.ndarray,
) -> Optional[float]:
    z = int(hyoid_voxel_zyx[0])
    if not (0 <= z < airway_mask.shape[0]):
        return None
    sl = airway_mask[z]
    if not sl.any():
        return None
    sy = image.spacing_xyz_mm[1]
    ys, _ = np.where(sl)
    y_post = int(ys.max())
    delta_y = abs(int(hyoid_voxel_zyx[1]) - y_post)
    return float(delta_y * sy)


def _cervicomandibular_ring_area(
    image: CTAImage,
    hyoid_voxel_zyx: tuple[int, int, int],
    mandible_mask: np.ndarray,
    airway: Optional[AirwayMaskInfo],
) -> tuple[Optional[float], str]:
    """Axial slice area enclosed by the mandible inferior rim + hyoid level.

    Cheap proxy: take the bounding box of the mandible inferior 1/3 in axis-1
    (A-P) and axis-2 (L-R), and multiply by in-plane voxel area.
    """
    coords = np.argwhere(mandible_mask)
    if coords.size == 0:
        return None, "no_mandible"
    z_max = int(coords[:, 0].max())
    inferior = coords[coords[:, 0] >= z_max - max(1, (z_max // 10))]
    if inferior.size == 0:
        return None, "no_inferior_extrema"
    y_span = int(inferior[:, 1].max() - inferior[:, 1].min() + 1)
    x_span = int(inferior[:, 2].max() - inferior[:, 2].min() + 1)
    sx, sy, _ = image.spacing_xyz_mm
    return float(y_span * x_span * sx * sy), "mandible_inferior_bbox_proxy"
