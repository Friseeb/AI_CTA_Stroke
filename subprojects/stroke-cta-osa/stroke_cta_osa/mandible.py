"""Mandible + oral cavity feature extraction.

Inputs in priority order:
    1. External mandible mask NIfTI;
    2. Dental/CBCT pipeline mandible mask (via adapter);
    3. High-HU connected-component bone fallback (only when configured);
    4. Missing — every mandible feature NaN.

The mandibular plane (Frankfort-mandible style) is computed from three
landmark points (menton, gonion_left, gonion_right) when available. If
landmarks are missing, the plane is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .landmark_schema import LandmarkBundle, LandmarkPlane
from .landmarks import get_hyoid_position, voxel_to_physical
from .logging_utils import get_logger
from .types import CTAImage

log = get_logger("mandible")

_NAN = float("nan")


@dataclass
class MandibleConfig:
    enabled: bool = True
    allow_bone_threshold_fallback: bool = False
    bone_hu_min: float = 250.0
    require_mask_for_volume: bool = True
    bone_min_volume_ml: float = 5.0


@dataclass
class OralCavityConfig:
    enabled: bool = True


def compute_mandible_features(
    image: CTAImage,
    cfg: MandibleConfig,
    mandible_mask: Optional[np.ndarray],
    landmarks: LandmarkBundle,
    *,
    mandible_mask_method: str = "external_mask",
    oral_cavity_mask: Optional[np.ndarray] = None,
    oral_cavity_cfg: Optional[OralCavityConfig] = None,
    save_masks_callback=None,
) -> dict[str, object]:
    """Returns a flat dict of mandible + oral cavity features."""
    out: dict[str, object] = _empty_row()

    if not cfg.enabled:
        out["mandible_mask_method"] = "disabled"
        return out

    # Mandible mask resolution -----------------------------------------------
    mask, method = _resolve_mandible_mask(
        image, cfg,
        external_mask=mandible_mask,
        external_mask_method=mandible_mask_method,
        save_masks_callback=save_masks_callback,
    )
    out["mandible_mask_method"] = method
    if mask is None or not mask.any():
        out["mandible_mask_available"] = False
        return out

    out["mandible_mask_available"] = True
    n_vox = int(mask.sum())
    vol_mm3 = float(n_vox * image.voxel_volume_mm3)
    out["mandible_volume_mm3"] = round(vol_mm3, 2)
    out["mandible_volume_ml"] = round(vol_mm3 / 1000.0, 4)
    if save_masks_callback is not None:
        save_masks_callback("mandible", mask)

    # Mandibular plane + hyoid distance -------------------------------------
    plane_info, plane_method = _resolve_mandibular_plane(landmarks, mask, image)
    out["mandibular_plane_method"] = plane_method
    out["mandibular_plane_available"] = plane_info is not None

    hyoid_vox = get_hyoid_position(landmarks)
    if plane_info is not None and hyoid_vox is not None:
        d_mm = _point_to_plane_distance(image, hyoid_vox, plane_info)
        out["mandibular_plane_to_hyoid_distance_mm"] = round(d_mm, 2)
        out["hyoid_to_mandible_distance_mm"] = round(d_mm, 2)

    # Oral cavity -----------------------------------------------------------
    if oral_cavity_mask is not None and oral_cavity_mask.any():
        out["oral_cavity_mask_available"] = True
        out["oral_cavity_method"] = "external"
        n_oc = int(np.asarray(oral_cavity_mask).sum())
        out["oral_cavity_volume_ml"] = round(
            float(n_oc * image.voxel_volume_mm3) / 1000.0, 4)
        if save_masks_callback is not None:
            save_masks_callback("oral_cavity", oral_cavity_mask.astype(bool))
    elif oral_cavity_cfg is not None and oral_cavity_cfg.enabled:
        out["oral_cavity_method"] = "absent"

    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _empty_row() -> dict[str, object]:
    return {
        "mandible_mask_available": False,
        "mandible_mask_method": "",
        "mandible_volume_mm3": _NAN,
        "mandible_volume_ml": _NAN,
        "mandibular_plane_available": False,
        "mandibular_plane_method": "",
        "mandibular_plane_to_hyoid_distance_mm": _NAN,
        "hyoid_to_mandible_distance_mm": _NAN,
        "cervicomandibular_ring_area_mm2": _NAN,
        "cervicomandibular_ring_method": "",
        "oral_cavity_mask_available": False,
        "oral_cavity_volume_ml": _NAN,
        "oral_cavity_method": "",
    }


def _resolve_mandible_mask(
    image: CTAImage, cfg: MandibleConfig,
    *, external_mask: Optional[np.ndarray],
    external_mask_method: str,
    save_masks_callback,
) -> tuple[Optional[np.ndarray], str]:
    if external_mask is not None and np.asarray(external_mask).any():
        return np.asarray(external_mask).astype(bool), external_mask_method

    if not cfg.allow_bone_threshold_fallback:
        return None, "absent_threshold_fallback_disabled"

    # High-HU threshold + largest CC heuristic. Conservative: only return
    # the candidate if its volume is plausible (≥ bone_min_volume_ml ≈ 5 mL).
    bone = image.array > cfg.bone_hu_min
    if not bone.any():
        return None, "no_bone_hu_voxels"
    labelled, n = ndimage.label(bone)
    if n == 0:
        return None, "no_bone_components"
    sizes = ndimage.sum_labels(np.ones_like(bone), labelled, range(1, n + 1))
    vox_ml = image.voxel_volume_mm3 / 1000.0
    sorted_idx = np.argsort(-sizes)
    # The largest bone CC is typically the cervical spine; the mandible is
    # the second-largest in many head-neck CTAs. We score the top 3 by an
    # axial-vs-vertical shape ratio: the mandible spans wide L-R but short
    # in z; the spine is tall and narrow.
    best_label = None
    best_score = -1.0
    for idx in sorted_idx[:5]:
        lbl = int(idx) + 1
        size = float(sizes[idx])
        if size * vox_ml < cfg.bone_min_volume_ml:
            continue
        mask = (labelled == lbl)
        zs, ys, xs = np.where(mask)
        if zs.size == 0:
            continue
        z_extent = (zs.max() - zs.min() + 1) * image.spacing_xyz_mm[2]
        x_extent = (xs.max() - xs.min() + 1) * image.spacing_xyz_mm[0]
        if z_extent <= 0:
            continue
        score = x_extent / z_extent
        if score > best_score:
            best_score = score
            best_label = lbl
    if best_label is None:
        return None, "no_plausible_mandible_candidate"
    mask = (labelled == best_label)
    return mask.astype(bool), f"bone_threshold_largest_cc_score={best_score:.2f}"


def _resolve_mandibular_plane(
    landmarks: LandmarkBundle, mask: np.ndarray, image: CTAImage,
) -> tuple[Optional[tuple[np.ndarray, np.ndarray]], str]:
    """Return ((point, normal), method_str) in physical mm.

    Priority:
        1. `mandibular_plane` from bundle.planes if it has point+normal;
        2. Three landmark points: menton + gonion_left + gonion_right;
        3. Three points derived from the mask: lowest anterior + lowest L/R;
        4. None.
    """
    pl = landmarks.planes.get("mandibular_plane")
    if pl is not None and pl.point_phys_mm and pl.normal_phys_mm:
        return (
            (np.array(pl.point_phys_mm), np.array(pl.normal_phys_mm)),
            "bundle_point_normal",
        )
    pts = landmarks.points
    if all(n in pts and pts[n].physical_mm
           for n in ("menton", "gonion_left", "gonion_right")):
        a = np.array(pts["menton"].physical_mm)
        b = np.array(pts["gonion_left"].physical_mm)
        c = np.array(pts["gonion_right"].physical_mm)
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) > 1e-6:
            return ((a, n / np.linalg.norm(n)), "from_menton_gonion_points")

    # Mask-derived fallback: lowest-z (most inferior) extremum point taken as
    # menton, leftmost and rightmost lowest-z points as gonia approximations.
    coords = np.argwhere(mask)
    if coords.size > 0:
        z_max = int(coords[:, 0].max())
        bottom = coords[coords[:, 0] >= (z_max - 2)]
        if bottom.size > 0:
            anterior = bottom[np.argmin(bottom[:, 1])]  # smallest y = anterior in LPS
            left_pt = bottom[np.argmin(bottom[:, 2])]
            right_pt = bottom[np.argmax(bottom[:, 2])]
            ax = np.array(voxel_to_physical(image, tuple(anterior)))
            lx = np.array(voxel_to_physical(image, tuple(left_pt)))
            rx = np.array(voxel_to_physical(image, tuple(right_pt)))
            n = np.cross(lx - ax, rx - ax)
            if np.linalg.norm(n) > 1e-6:
                return ((ax, n / np.linalg.norm(n)),
                        "mask_inferior_extrema_heuristic")
    return None, "no_landmarks_or_extrema"


def _point_to_plane_distance(
    image: CTAImage,
    point_voxel_zyx: tuple[int, int, int],
    plane: tuple[np.ndarray, np.ndarray],
) -> float:
    """Signed perpendicular distance from a voxel-index point to a plane
    (point, normal) given in physical mm. We return the magnitude — the
    caller decides whether sign is meaningful.
    """
    p_phys = np.array(voxel_to_physical(image, point_voxel_zyx))
    plane_point, plane_normal = plane
    n = plane_normal / max(np.linalg.norm(plane_normal), 1e-9)
    return float(abs(np.dot(p_phys - plane_point, n)))
