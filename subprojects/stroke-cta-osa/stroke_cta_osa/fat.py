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

from .anatomy_priors import combined_anatomy_exclusion_mask
from .config import FatConfig, HUConfig
from .geometry import mm_to_voxels
from .logging_utils import get_logger
from .rois import (body_mask, cervical_z_range, parapharyngeal_bands,
                   parapharyngeal_sector_bands, retropharyngeal_band,
                   retropharyngeal_prevertebral_band,
                   subcutaneous_band, posterior_tongue_band)
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
    anatomy_masks: Optional[dict[str, Optional[np.ndarray]]] = None,
    precomputed_body_mask: Optional[np.ndarray] = None,
) -> dict[str, float | str | None]:
    """Run every fat-feature block. Returns a flat dict keyed by feature name.

    `save_masks_callback`, if provided, is called as
    `save_masks_callback(name: str, mask: np.ndarray)` for any ROI mask we
    materialise — used by the orchestrator when cfg.output.save_masks=True.

    `precomputed_body_mask` lets the orchestrator share a single body silhouette
    across the fat / regional-fat / soft-palate modules. Computing it is one of
    the most expensive (CPU + RAM) operations in the pipeline, so reusing it
    avoids a second full-volume connected-component pass.
    """
    arr_hu = image.array
    fat_voxels = (arr_hu >= hu_cfg.fat_hu_min) & (arr_hu <= hu_cfg.fat_hu_max)

    # ---- Z range and body envelope ----
    z_lo, z_hi = cervical_z_range(image, airway, landmarks)
    body = (precomputed_body_mask if precomputed_body_mask is not None
            else body_mask(image, fat_cfg.body_air_threshold_hu))
    sub = subcutaneous_band(body, fat_cfg.subcutaneous_erosion_mm, image.spacing_xyz_mm)
    deep = body & ~sub

    # FOV-robust anchored neck-slab features (computed while sub/deep are alive).
    neck_feats = _anchored_neck_features(
        image=image, fat_voxels=fat_voxels, body=body, sub=sub, deep=deep,
        airway=airway, anchor_z=airway_min_csa_z_index,
        z_range=(z_lo, z_hi), fat_cfg=fat_cfg,
    )

    # Memory discipline: the masks below are each a full-resolution bool volume
    # (hundreds of MB at clinical CTA sizes). We free every intermediate at its
    # last use so they don't all stay resident through the parapharyngeal block,
    # which is what set the per-case memory high-water mark.
    cervical_body = body.copy()
    cervical_body[:z_lo] = False
    cervical_body[z_hi + 1:] = False
    cervical_fat = fat_voxels & cervical_body
    cervical_sub_fat = fat_voxels & sub & cervical_body
    cervical_deep_fat = fat_voxels & deep & cervical_body
    del sub, deep  # only needed to build the two masks above
    anatomy_exclusion, anatomy_used = combined_anatomy_exclusion_mask(
        anatomy_masks,
        reference_shape=image.shape_zyx,
        spacing_xyz_mm=image.spacing_xyz_mm,
        dilation_mm=fat_cfg.anatomy_prior_dilation_mm,
    )
    anatomy_prior_available = bool(fat_cfg.use_anatomy_priors and anatomy_used)
    deep_fat_for_local_rois = (
        cervical_deep_fat & ~anatomy_exclusion
        if anatomy_prior_available else cervical_deep_fat
    )

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
        "fat_anatomy_prior_masks_used": ",".join(anatomy_used),
    }
    out.update(neck_feats)

    # ---- A. Total cervical fat ----
    out.update(_block("fat_cervical", arr_hu, cervical_fat, image))
    del cervical_fat

    # ---- B. Subcutaneous cervical fat ----
    out.update(_block("fat_subcutaneous_cervical", arr_hu, cervical_sub_fat, image))
    neck_area_voxels = int(cervical_body.sum())
    sub_voxels = int(cervical_sub_fat.sum())
    del cervical_sub_fat, cervical_body
    out["fat_subcutaneous_fraction_of_neck_area"] = (
        round(sub_voxels / neck_area_voxels, 4) if neck_area_voxels else _NAN
    )

    # ---- C. Deep cervical fat ----
    out.update(_block("fat_deep_cervical", arr_hu, cervical_deep_fat, image))
    del cervical_deep_fat  # downstream local ROIs use deep_fat_for_local_rois
    sub_vol = out["fat_subcutaneous_cervical_volume_ml"]
    deep_vol = out["fat_deep_cervical_volume_ml"]
    out["fat_deep_to_subcutaneous_ratio"] = (
        round(deep_vol / sub_vol, 3) if (isinstance(sub_vol, float) and sub_vol > 0
                                          and isinstance(deep_vol, float)) else _NAN
    )

    # ---- D. Deep peri-pharyngeal / parapharyngeal ----
    airway_mask = airway.mask_zyx if (airway is not None and airway.is_present) else None
    if airway_mask is not None:
        deep_peripharyngeal_fat, deep_peripharyngeal_method = _deep_peripharyngeal_fat(
            image=image,
            airway_mask=airway_mask,
            deep_fat_mask=deep_fat_for_local_rois,
            radial_band_mm=max(
                fat_cfg.parapharyngeal_lateral_band_mm,
                fat_cfg.retropharyngeal_posterior_band_mm,
            ),
            axial_window_mm=max(
                fat_cfg.parapharyngeal_axial_window_mm,
                fat_cfg.retropharyngeal_axial_window_mm,
            ),
            z_anchor=airway_min_csa_z_index,
        )
        if save_masks_callback is not None:
            save_masks_callback("fat_deep_peripharyngeal", deep_peripharyngeal_fat)
        out.update(_block("fat_deep_peripharyngeal", arr_hu,
                          deep_peripharyngeal_fat, image))
        out["fat_deep_peripharyngeal_roi_method"] = deep_peripharyngeal_method

        if anatomy_prior_available:
            left, right, pp_method = parapharyngeal_sector_bands(
                image, airway_mask,
                lateral_band_mm=fat_cfg.parapharyngeal_lateral_band_mm,
                axial_window_mm=fat_cfg.parapharyngeal_axial_window_mm,
                z_anchor=airway_min_csa_z_index,
                anatomy_exclusion_mask=anatomy_exclusion,
                min_lateral_fraction=(
                    fat_cfg.parapharyngeal_sector_min_lateral_fraction
                ),
            )
            pp_method += "_anatomy_prior_sector"
        else:
            left, right, pp_method = parapharyngeal_bands(
                image, airway_mask,
                lateral_band_mm=fat_cfg.parapharyngeal_lateral_band_mm,
                axial_window_mm=fat_cfg.parapharyngeal_axial_window_mm,
                z_anchor=airway_min_csa_z_index,
            )
        left_fat = fat_voxels & left & deep_fat_for_local_rois
        right_fat = fat_voxels & right & deep_fat_for_local_rois
        del left, right, deep_fat_for_local_rois  # consumed by the local ROIs
        if save_masks_callback is not None:
            save_masks_callback("fat_parapharyngeal_left", left_fat)
            save_masks_callback("fat_parapharyngeal_right", right_fat)

        out.update(_block("fat_parapharyngeal_left", arr_hu, left_fat, image))
        out.update(_block("fat_parapharyngeal_right", arr_hu, right_fat, image))
        # Compute the union once and reuse it for the volume block, the asymmetry
        # ratio, and the level areas — instead of OR-ing two full volumes 4×.
        total = left_fat | right_fat
        del left_fat, right_fat
        out.update(_block("fat_parapharyngeal_total", arr_hu, total, image))

        lv = out["fat_parapharyngeal_left_volume_ml"]
        rv = out["fat_parapharyngeal_right_volume_ml"]
        denom = (lv + rv) if (isinstance(lv, float) and isinstance(rv, float)) else _NAN
        out["fat_parapharyngeal_asymmetry_index"] = (
            round((rv - lv) / denom, 3) if (isinstance(denom, float) and denom > 0) else _NAN
        )
        out["fat_parapharyngeal_roi_method"] = pp_method
        out["fat_parapharyngeal_roi_method"] += "_deep_fat_gated"
        if anatomy_used:
            out["fat_parapharyngeal_roi_method"] += (
                "_priors=" + ",".join(anatomy_used)
            )

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
            slice_fat = total[airway_min_csa_z_index]
            sx, sy, _ = image.spacing_xyz_mm
            out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = round(
                float(slice_fat.sum()) * sx * sy, 2
            )
        else:
            out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = _NAN

        # Region-anchored areas (retropalatal / retroglossal). We rely on
        # the existing landmark anchor — without landmarks we leave NaN.
        out["fat_parapharyngeal_area_retropalatal_mm2"] = _area_at_landmark_z(
            total, landmarks.posterior_nasal_spine or landmarks.soft_palate_inferior,
            image,
        )
        out["fat_parapharyngeal_area_retroglossal_mm2"] = _area_at_landmark_z(
            total, landmarks.epiglottis_tip or landmarks.hyoid, image,
        )
        del total
    else:
        out.update(_block("fat_parapharyngeal_left", arr_hu, None, image))
        out.update(_block("fat_parapharyngeal_right", arr_hu, None, image))
        out.update(_block("fat_parapharyngeal_total", arr_hu, None, image))
        out.update(_block("fat_deep_peripharyngeal", arr_hu, None, image))
        out["fat_deep_peripharyngeal_roi_method"] = "unavailable_no_airway"
        out["fat_parapharyngeal_asymmetry_index"] = _NAN
        out["fat_parapharyngeal_roi_method"] = "unavailable_no_airway"
        out["fat_parapharyngeal_to_airway_ratio"] = _NAN
        out["fat_parapharyngeal_area_at_min_airway_csa_mm2"] = _NAN
        out["fat_parapharyngeal_area_retropalatal_mm2"] = _NAN
        out["fat_parapharyngeal_area_retroglossal_mm2"] = _NAN

    # ---- E. Retropharyngeal ----
    if airway_mask is not None:
        prevertebral_mask = (
            np.asarray(anatomy_masks.get("prevertebral")).astype(bool)
            if anatomy_masks and anatomy_masks.get("prevertebral") is not None
            else None
        )
        rp_z_bounds = (
            _oropharyngeal_z_bounds(
                image=image,
                airway_mask=airway_mask,
                landmarks=landmarks,
                z_anchor=airway_min_csa_z_index,
                axial_window_mm=fat_cfg.retropharyngeal_axial_window_mm,
            )
            if fat_cfg.retropharyngeal_use_oropharyngeal_window else None
        )
        if prevertebral_mask is not None and prevertebral_mask.any():
            rp_band, rp_method = retropharyngeal_prevertebral_band(
                image, airway_mask,
                posterior_band_mm=fat_cfg.retropharyngeal_posterior_band_mm,
                axial_window_mm=fat_cfg.retropharyngeal_axial_window_mm,
                z_anchor=airway_min_csa_z_index,
                prevertebral_mask=prevertebral_mask,
                z_bounds=rp_z_bounds,
                prevertebral_margin_mm=(
                    fat_cfg.retropharyngeal_prevertebral_margin_mm
                ),
                lateral_margin_mm=fat_cfg.retropharyngeal_lateral_margin_mm,
            )
        else:
            rp_band, rp_method = retropharyngeal_band(
                image, airway_mask,
                posterior_band_mm=fat_cfg.retropharyngeal_posterior_band_mm,
                axial_window_mm=fat_cfg.retropharyngeal_axial_window_mm,
                z_anchor=airway_min_csa_z_index,
            )
            if rp_z_bounds is not None:
                z_lo, z_hi = rp_z_bounds
                lo, hi = min(z_lo, z_hi), max(z_lo, z_hi)
                rp_band[:lo] = False          # zero outside the window in place,
                rp_band[hi + 1:] = False      # avoiding a full-volume temp mask
                rp_method += "_oropharyngeal_window"
        rp_fat = fat_voxels & rp_band & deep_peripharyngeal_fat
        del rp_band, deep_peripharyngeal_fat
        if save_masks_callback is not None:
            save_masks_callback("fat_retropharyngeal", rp_fat)
        out.update(_block("fat_retropharyngeal", arr_hu, rp_fat, image))
        out["fat_retropharyngeal_roi_method"] = (
            f"{rp_method}_deep_peripharyngeal_posterior_sector"
        )
        out.update(_thickness_features(rp_fat, image))
        del rp_fat
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

_NECK_KEYS = (
    "fat_neck_slab_height_mm", "fat_neck_slab_volume_ml",
    "fat_neck_slab_subcutaneous_volume_ml", "fat_neck_slab_deep_volume_ml",
    "fat_neck_slab_mean_hu", "fat_neck_slab_fat_fraction",
    "fat_neck_slab_deep_to_subcutaneous_ratio",
    "fat_neck_slab_to_airway_volume_ratio", "fat_neck_area_at_min_csa_mm2",
    "fat_neck_body_area_at_min_csa_mm2", "fat_neck_area_fraction_at_min_csa",
    "fat_neck_roi_radius_mm",
)


def _anchored_neck_features(*, image, fat_voxels, body, sub, deep, airway,
                            anchor_z, z_range, fat_cfg) -> dict:
    """FOV-robust cervical adiposity from a fixed-height neck slab.

    The plain cervical volume scales with the imaged z-extent (badly inflated on
    tall head-to-chest CTAs). Here we anchor a fixed ±``neck_slab_half_height_mm``
    slab on the airway min-CSA slice and report absolute slab volumes plus
    dimensionless fractions/ratios that are invariant to how much neck/chest was
    scanned.
    """
    out: dict = {k: _NAN for k in _NECK_KEYS}
    out["fat_neck_anchor_method"] = "unavailable_no_anchor"
    if not getattr(fat_cfg, "neck_slab_enabled", True) or body is None or not body.any():
        return out

    sz = image.shape_zyx[0]
    dx, dy, dz = (float(v) for v in image.spacing_xyz_mm)
    # Anchor: prefer the airway min-CSA slice; fall back to the cervical-z midpoint.
    a, method = None, None
    if (getattr(fat_cfg, "neck_slab_anchor", "min_csa") == "min_csa"
            and isinstance(anchor_z, int) and 0 <= anchor_z < sz):
        a, method = anchor_z, "min_csa"
    if a is None:
        z_lo, z_hi = z_range
        if 0 <= z_lo <= z_hi < sz:
            a, method = (z_lo + z_hi) // 2, "cervical_zrange"
    if a is None:
        return out

    # Shift the slab centre inferiorly from the min-CSA anchor (which sits at the
    # fat-rich, artifact-prone tongue-base level) to the true mid-cervical neck.
    # RAS: higher z index = superior, so inferior = lower index. Clamp the centre
    # to stay within the imaged airway span.
    offset_vox = int(round(getattr(fat_cfg, "neck_slab_inferior_offset_mm", 0.0)
                           / max(dz, 1e-6)))
    zr_lo, zr_hi = z_range
    center = min(max(a - offset_vox, min(zr_lo, zr_hi)), max(zr_lo, zr_hi))
    center = min(max(center, 0), sz - 1)
    if offset_vox:
        out["fat_neck_anchor_method"] = f"{method}_inferior{int(getattr(fat_cfg, 'neck_slab_inferior_offset_mm', 0))}mm"

    half = max(1, int(round(fat_cfg.neck_slab_half_height_mm / max(dz, 1e-6))))
    z0, z1 = max(0, center - half), min(sz - 1, center + half)
    sl = slice(z0, z1 + 1)
    vox_ml = image.voxel_volume_mm3 / 1000.0

    bslab = body[sl]
    if not bslab.any():
        return out

    # In-plane containment: keep only voxels within `radius_mm` of the airway
    # centroid (y, x), so shoulders / arms / foam padding — all low-HU and
    # otherwise miscounted as fat — are excluded. Center on the airway when
    # present, else the body centroid of the slab.
    radius_mm = float(getattr(fat_cfg, "neck_slab_radius_mm", 0.0) or 0.0)
    disk = None
    if radius_mm > 0:
        if airway is not None and airway.is_present and airway.mask_zyx[sl].any():
            _, yy, xx = np.where(airway.mask_zyx[sl])
        else:
            _, yy, xx = np.where(bslab)
        cy, cx = float(yy.mean()), float(xx.mean())
        ny, nx = body.shape[1], body.shape[2]
        Y, X = np.ogrid[:ny, :nx]
        disk = (((Y - cy) * dy) ** 2 + ((X - cx) * dx) ** 2) <= radius_mm ** 2
        out["fat_neck_roi_radius_mm"] = round(radius_mm, 1)
        bslab = bslab & disk  # contained neck body

    body_n = int(bslab.sum())
    if body_n == 0:
        return out
    fslab = fat_voxels[sl] & bslab
    fat_n = int(fslab.sum())
    sub_n = int((fat_voxels[sl] & sub[sl] & bslab).sum())
    deep_n = int((fat_voxels[sl] & deep[sl] & bslab).sum())

    out["fat_neck_anchor_method"] = method
    out["fat_neck_slab_height_mm"] = round((z1 - z0 + 1) * dz, 2)
    out["fat_neck_slab_volume_ml"] = round(fat_n * vox_ml, 3)
    out["fat_neck_slab_subcutaneous_volume_ml"] = round(sub_n * vox_ml, 3)
    out["fat_neck_slab_deep_volume_ml"] = round(deep_n * vox_ml, 3)
    out["fat_neck_slab_fat_fraction"] = round(fat_n / body_n, 4)
    out["fat_neck_slab_deep_to_subcutaneous_ratio"] = (
        round(deep_n / sub_n, 3) if sub_n > 0 else _NAN)
    if fat_n > 0:
        out["fat_neck_slab_mean_hu"] = round(float(image.array[sl][fslab].mean()), 2)
    if airway is not None and airway.is_present:
        airway_vol = int(airway.mask_zyx.sum()) * vox_ml
        out["fat_neck_slab_to_airway_volume_ratio"] = (
            round(fat_n * vox_ml / airway_vol, 3) if airway_vol > 0 else _NAN)

    # Single-slice area fraction at the anchor (dimensionless, FOV-invariant).
    body_a = body[a] & disk if disk is not None else body[a]
    ba = int(body_a.sum())
    fa = int((fat_voxels[a] & body_a).sum())
    out["fat_neck_area_at_min_csa_mm2"] = round(fa * dx * dy, 2)
    out["fat_neck_body_area_at_min_csa_mm2"] = round(ba * dx * dy, 2)
    out["fat_neck_area_fraction_at_min_csa"] = round(fa / ba, 4) if ba > 0 else _NAN
    return out


def _deep_peripharyngeal_fat(
    *,
    image: CTAImage,
    airway_mask: np.ndarray,
    deep_fat_mask: np.ndarray,
    radial_band_mm: float,
    axial_window_mm: float,
    z_anchor: Optional[int],
) -> tuple[np.ndarray, str]:
    """Deep fat within a physical-distance band around the airway.

    This is the parent compartment for airway-adjacent fat. It avoids calling
    every posterior/lateral box "retropharyngeal" before we know whether the
    voxels are actually deep and close to the pharyngeal airway.
    """
    _, _, sz_mm = image.spacing_xyz_mm
    sy, sx = float(image.spacing_xyz_mm[1]), float(image.spacing_xyz_mm[0])
    sz_image = airway_mask.shape[0]
    if z_anchor is None or z_anchor < 0 or z_anchor >= sz_image:
        zs = np.where(airway_mask.any(axis=(1, 2)))[0]
        if zs.size == 0:
            return np.zeros_like(airway_mask, dtype=bool), "unavailable_no_airway"
        z_anchor = int(zs[len(zs) // 2])
        method = f"airway_centroid_z={z_anchor}"
    else:
        method = f"anchored_z={z_anchor}"

    half_z = mm_to_voxels(axial_window_mm / 2.0, sz_mm)
    z_lo = max(0, int(z_anchor) - half_z)
    z_hi = min(sz_image - 1, int(z_anchor) + half_z)
    out = np.zeros_like(airway_mask, dtype=bool)
    for z in range(z_lo, z_hi + 1):
        sl = airway_mask[z]
        if not sl.any():
            continue
        dist_mm = ndimage.distance_transform_edt(~sl, sampling=(sy, sx))
        out[z] = (dist_mm > 0) & (dist_mm <= radial_band_mm)
    return out & deep_fat_mask, (
        f"{method}_deep_fat_airway_distance_le_{float(radial_band_mm):.1f}mm"
    )


def _oropharyngeal_z_bounds(
    *,
    image: CTAImage,
    airway_mask: np.ndarray,
    landmarks: SharedAirwayLandmarks,
    z_anchor: Optional[int],
    axial_window_mm: float,
) -> Optional[tuple[int, int]]:
    """Best-effort oropharyngeal z window for RP fat.

    The pipeline's available landmark schema does not expose an explicit
    `oropharynx` mask. We use the clinically adjacent levels: inferior soft
    palate / PNS through epiglottis or hyoid. If those are unavailable, fall
    back to the configured anchor window.
    """
    candidates_hi = [
        landmarks.soft_palate_inferior,
        landmarks.posterior_nasal_spine,
    ]
    candidates_lo = [
        landmarks.epiglottis_tip,
        landmarks.hyoid,
    ]
    zs = [int(p[0]) for p in candidates_hi + candidates_lo if p is not None]
    if len(zs) >= 2:
        return max(0, min(zs)), min(image.shape_zyx[0] - 1, max(zs))
    airway_z = np.where(airway_mask.any(axis=(1, 2)))[0]
    if airway_z.size:
        z_lo_all, z_hi_all = int(airway_z.min()), int(airway_z.max())
        third = max(1, (z_hi_all - z_lo_all + 1) // 3)
        return (
            min(image.shape_zyx[0] - 1, z_lo_all + third),
            min(image.shape_zyx[0] - 1, z_lo_all + 2 * third - 1),
        )
    if z_anchor is None or not (0 <= int(z_anchor) < image.shape_zyx[0]):
        return None
    half_z = mm_to_voxels(axial_window_mm / 2.0, image.spacing_xyz_mm[2])
    return (
        max(0, int(z_anchor) - half_z),
        min(image.shape_zyx[0] - 1, int(z_anchor) + half_z),
    )


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
