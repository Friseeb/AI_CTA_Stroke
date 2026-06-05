"""Aortic calcification thresholding and summary features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .features import feature_row
from .shells import external_shell, internal_boundary_shell


@dataclass(frozen=True)
class DynamicWallCalcificationResult:
    """Outputs from local lumen-referenced wall calcium growth."""

    mask: np.ndarray
    high_confidence_seed_mask: np.ndarray
    candidate_mask: np.ndarray
    lumen_core_mask: np.ndarray
    search_roi_mask: np.ndarray
    external_contrast_like_mask: np.ndarray
    external_contrast_rejected_mask: np.ndarray
    lumen_reference_hu_by_slice: np.ndarray
    dynamic_threshold_hu_by_slice: np.ndarray
    seed_threshold_hu_by_slice: np.ndarray
    global_lumen_reference_hu: float


def density_factor_for_hu(max_hu: float) -> int:
    """Return an Agatston-style density factor from maximum HU."""
    if not np.isfinite(max_hu):
        return 0
    if max_hu >= 400:
        return 4
    if max_hu >= 300:
        return 3
    if max_hu >= 200:
        return 2
    if max_hu >= 130:
        return 1
    return 0


def extract_calcification_masks(
    image: np.ndarray,
    roi_mask: np.ndarray,
    thresholds_hu: list[int | float],
) -> dict[int, np.ndarray]:
    """Threshold high-attenuation voxels in a configured aortic ROI."""
    roi = np.asarray(roi_mask, dtype=bool)
    return {int(threshold): roi & (image >= float(threshold)) for threshold in thresholds_hu}


def summarize_calcification(
    image: np.ndarray,
    calcium_masks: dict[int | str, np.ndarray],
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    region: str,
    mask_name: str,
    software_version: str = __version__,
) -> pd.DataFrame:
    """Summarize calcification burden for each HU threshold."""
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    rows: list[dict[str, object]] = []
    for threshold, calcium_mask in calcium_masks.items():
        values = image[calcium_mask]
        voxel_count = int(calcium_mask.sum())
        volume_mm3 = float(voxel_count * voxel_volume_mm3)
        max_hu = float(values.max()) if values.size else np.nan
        mean_hu = float(values.mean()) if values.size else np.nan
        density_factor = density_factor_for_hu(max_hu)
        agatston_like = float(volume_mm3 * density_factor)
        rows.extend(
            [
                _row(
                    case_id,
                    region,
                    "calcification",
                    "calcium_voxel_count",
                    voxel_count,
                    "voxels",
                    threshold,
                    mask_name,
                    software_version,
                ),
                _row(
                    case_id,
                    region,
                    "calcification",
                    "calcium_volume",
                    volume_mm3,
                    "mm3",
                    threshold,
                    mask_name,
                    software_version,
                ),
                _row(
                    case_id,
                    region,
                    "calcification",
                    "calcium_max_hu",
                    max_hu,
                    "HU",
                    threshold,
                    mask_name,
                    software_version,
                ),
                _row(
                    case_id,
                    region,
                    "calcification",
                    "calcium_mean_hu",
                    mean_hu,
                    "HU",
                    threshold,
                    mask_name,
                    software_version,
                ),
                _row(
                    case_id,
                    region,
                    "calcification",
                    "agatston_like_not_ecg_gated",
                    agatston_like,
                    "arbitrary",
                    threshold,
                    mask_name,
                    software_version,
                ),
            ]
        )
    return pd.DataFrame(rows)


def extract_dynamic_wall_calcification(
    image: np.ndarray,
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    seed_threshold_hu: float = 500.0,
    lumen_margin_hu: float = 75.0,
    min_candidate_hu: float = 300.0,
    lumen_core_distance_mm: float = 5.0,
    search_internal_mm: float = 5.0,
    search_external_mm: float = 2.0,
    smooth_lumen_profile_mm: float = 10.0,
    min_core_voxels_per_slice: int = 20,
    exclude_external_contrast_touching: bool = True,
    external_contrast_tolerance_hu: float = 75.0,
) -> DynamicWallCalcificationResult:
    """Grow wall calcium from high-HU seeds using a local lumen-referenced threshold.

    This keeps the conservative high-HU threshold as the seed definition, then
    admits connected wall-adjacent voxels when they remain above the local
    contrast-lumen reference by a configurable margin. It is intended to recover
    boundary/intimal partial-volume tails without thresholding the whole aortic
    lumen as calcium.
    """
    image_array = np.asarray(image)
    aorta = np.asarray(aorta_mask, dtype=bool)
    if image_array.shape != aorta.shape:
        raise ValueError("image and aorta_mask must have the same shape.")

    lumen_core = _lumen_core_mask(
        aorta,
        spacing_xyz=spacing_xyz,
        core_distance_mm=float(lumen_core_distance_mm),
    )
    lumen_reference = _slice_lumen_reference_hu(
        image_array,
        lumen_core,
        fallback_mask=aorta,
        min_voxels_per_slice=int(min_core_voxels_per_slice),
    )
    lumen_reference = _smooth_profile(
        lumen_reference,
        spacing_z_mm=float(spacing_xyz[2]),
        smooth_mm=float(smooth_lumen_profile_mm),
    )
    dynamic_threshold = np.maximum(lumen_reference + float(lumen_margin_hu), float(min_candidate_hu))
    seed_threshold = np.maximum(dynamic_threshold, float(seed_threshold_hu))

    search_roi = _wall_calcium_search_roi(
        aorta,
        spacing_xyz=spacing_xyz,
        internal_mm=float(search_internal_mm),
        external_mm=float(search_external_mm),
    )
    high_confidence_seed = search_roi & (image_array >= seed_threshold[:, None, None])
    candidate = search_roi & (image_array >= dynamic_threshold[:, None, None])
    candidate |= high_confidence_seed
    external_contrast_like = np.zeros_like(aorta, dtype=bool)
    rejected_external_contrast = np.zeros_like(aorta, dtype=bool)
    if exclude_external_contrast_touching:
        external_contrast_like = _external_contrast_like_mask(
            image_array,
            aorta,
            lumen_reference_hu_by_slice=lumen_reference,
            tolerance_hu=float(external_contrast_tolerance_hu),
        )
        rejected_external_contrast = _external_contrast_touching_components(
            candidate,
            aorta,
            external_contrast_like,
        )
        if rejected_external_contrast.any():
            candidate &= ~rejected_external_contrast
            high_confidence_seed &= ~rejected_external_contrast
    grown = _connected_to_seed(candidate, high_confidence_seed)

    return DynamicWallCalcificationResult(
        mask=grown,
        high_confidence_seed_mask=high_confidence_seed,
        candidate_mask=candidate,
        lumen_core_mask=lumen_core,
        search_roi_mask=search_roi,
        external_contrast_like_mask=external_contrast_like,
        external_contrast_rejected_mask=rejected_external_contrast,
        lumen_reference_hu_by_slice=lumen_reference,
        dynamic_threshold_hu_by_slice=dynamic_threshold,
        seed_threshold_hu_by_slice=seed_threshold,
        global_lumen_reference_hu=float(np.nanmedian(lumen_reference)),
    )


def summarize_dynamic_wall_calcification(
    result: DynamicWallCalcificationResult,
    case_id: str,
    mask_name: str,
    software_version: str = __version__,
) -> pd.DataFrame:
    """Summarize diagnostic settings for the dynamic wall calcium map."""
    rows = [
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="lumen_reference_hu_median",
            feature_value=float(np.nanmedian(result.lumen_reference_hu_by_slice)),
            units="HU",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="dynamic_threshold_hu_median",
            feature_value=float(np.nanmedian(result.dynamic_threshold_hu_by_slice)),
            units="HU",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="dynamic_threshold_hu_min",
            feature_value=float(np.nanmin(result.dynamic_threshold_hu_by_slice)),
            units="HU",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="dynamic_threshold_hu_max",
            feature_value=float(np.nanmax(result.dynamic_threshold_hu_by_slice)),
            units="HU",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="seed_threshold_hu_median",
            feature_value=float(np.nanmedian(result.seed_threshold_hu_by_slice)),
            units="HU",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="high_confidence_seed_voxel_count",
            feature_value=int(result.high_confidence_seed_mask.sum()),
            units="voxels",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="dynamic_candidate_voxel_count",
            feature_value=int(result.candidate_mask.sum()),
            units="voxels",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="external_contrast_like_voxel_count",
            feature_value=int(result.external_contrast_like_mask.sum()),
            units="voxels",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
        feature_row(
            case_id=case_id,
            region="aorta_wall_dynamic",
            feature_group="calcification_dynamic_threshold",
            feature_name="external_contrast_touching_rejected_voxel_count",
            feature_value=int(result.external_contrast_rejected_mask.sum()),
            units="voxels",
            threshold_if_applicable="dynamic_lumen_referenced",
            mask_name=mask_name,
            software_version=software_version,
        ),
    ]
    return pd.DataFrame(rows)


def _row(
    case_id: str,
    region: str,
    group: str,
    name: str,
    value: object,
    units: str,
    threshold: object,
    mask_name: str,
    software_version: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "region": region,
        "feature_group": group,
        "feature_name": name,
        "feature_value": value,
        "units": units,
        "threshold_if_applicable": threshold,
        "mask_name": mask_name,
        "software_version": software_version,
    }


def _lumen_core_mask(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    core_distance_mm: float,
) -> np.ndarray:
    aorta = np.asarray(aorta_mask, dtype=bool)
    if not aorta.any():
        return np.zeros_like(aorta, dtype=bool)
    boundary_band = internal_boundary_shell(aorta, spacing_xyz, depth_mm=max(float(core_distance_mm), 0.0))
    core = aorta & ~boundary_band
    if core.any():
        return core
    return aorta


def _wall_calcium_search_roi(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    internal_mm: float,
    external_mm: float,
) -> np.ndarray:
    aorta = np.asarray(aorta_mask, dtype=bool)
    internal = internal_boundary_shell(aorta, spacing_xyz, depth_mm=max(float(internal_mm), 0.0))
    external = (
        external_shell(aorta, spacing_xyz, inner_mm=0.0, outer_mm=float(external_mm))
        if external_mm > 0
        else np.zeros_like(aorta, dtype=bool)
    )
    return internal | external


def _slice_lumen_reference_hu(
    image: np.ndarray,
    lumen_core: np.ndarray,
    fallback_mask: np.ndarray,
    min_voxels_per_slice: int,
    statistic: str = "median",
) -> np.ndarray:
    profile = np.full(image.shape[0], np.nan, dtype=float)
    for z in range(image.shape[0]):
        core_values = image[z][lumen_core[z]]
        if core_values.size >= min_voxels_per_slice:
            profile[z] = _reference_statistic(core_values, statistic)
            continue
        fallback_values = image[z][fallback_mask[z]]
        if fallback_values.size:
            profile[z] = _reference_statistic(fallback_values, statistic)
    valid = np.flatnonzero(np.isfinite(profile))
    if valid.size == 0:
        return np.zeros_like(profile)
    if valid.size == 1:
        profile[:] = profile[valid[0]]
        return profile
    missing = np.flatnonzero(~np.isfinite(profile))
    profile[missing] = np.interp(missing, valid, profile[valid])
    return profile


def _reference_statistic(values: np.ndarray, statistic: str) -> float:
    stat = statistic.lower()
    if stat == "median":
        return float(np.median(values))
    if stat == "mean":
        return float(np.mean(values))
    raise ValueError("lumen reference statistic must be 'median' or 'mean'.")


def _smooth_profile(profile: np.ndarray, spacing_z_mm: float, smooth_mm: float) -> np.ndarray:
    if smooth_mm <= 0 or profile.size < 3:
        return profile
    window = max(1, int(round(float(smooth_mm) / max(float(spacing_z_mm), 1e-6))))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return profile
    try:
        from scipy import ndimage as ndi

        return ndi.median_filter(profile, size=window, mode="nearest")
    except Exception:
        kernel = np.ones(window, dtype=float) / float(window)
        padded = np.pad(profile, window // 2, mode="edge")
        return np.convolve(padded, kernel, mode="valid")


def _connected_to_seed(candidate_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate_mask, dtype=bool)
    seed = np.asarray(seed_mask, dtype=bool) & candidate
    if not candidate.any() or not seed.any():
        return np.zeros_like(candidate, dtype=bool)

    slices = _crop_slices(candidate | seed)
    candidate_crop = candidate[slices]
    seed_crop = seed[slices]
    try:
        from scipy import ndimage as ndi

        structure = np.ones((3, 3, 3), dtype=bool)
        grown_crop = ndi.binary_propagation(seed_crop, structure=structure, mask=candidate_crop)
    except Exception:
        grown_crop = seed_crop
    grown = np.zeros_like(candidate, dtype=bool)
    grown[slices] = grown_crop
    return grown


def _external_contrast_like_mask(
    image: np.ndarray,
    aorta_mask: np.ndarray,
    lumen_reference_hu_by_slice: np.ndarray,
    tolerance_hu: float,
) -> np.ndarray:
    outside_aorta = ~np.asarray(aorta_mask, dtype=bool)
    lower = lumen_reference_hu_by_slice[:, None, None] - float(tolerance_hu)
    upper = lumen_reference_hu_by_slice[:, None, None] + float(tolerance_hu)
    return outside_aorta & (image >= lower) & (image <= upper)


def _external_contrast_touching_components(
    candidate_mask: np.ndarray,
    aorta_mask: np.ndarray,
    external_contrast_like_mask: np.ndarray,
) -> np.ndarray:
    candidate = np.asarray(candidate_mask, dtype=bool)
    if not candidate.any() or not external_contrast_like_mask.any():
        return np.zeros_like(candidate, dtype=bool)

    slices = _crop_slices_with_pad(candidate, pad_voxels=1)
    candidate_crop = candidate[slices]
    aorta_crop = np.asarray(aorta_mask, dtype=bool)[slices]
    contrast_crop = np.asarray(external_contrast_like_mask, dtype=bool)[slices]
    try:
        from scipy import ndimage as ndi

        structure = np.ones((3, 3, 3), dtype=bool)
        labels, _ = ndi.label(candidate_crop, structure=structure)
        if labels.max() == 0:
            return np.zeros_like(candidate, dtype=bool)
        contrast_touch_crop = ndi.binary_dilation(contrast_crop, structure=structure) & candidate_crop & ~aorta_crop
        touching_ids = set(np.unique(labels[contrast_touch_crop]).astype(int).tolist()) - {0}
        internal_ids = set(np.unique(labels[candidate_crop & aorta_crop]).astype(int).tolist()) - {0}
        reject_ids = touching_ids - internal_ids
        rejected_crop = np.isin(labels, list(reject_ids)) if reject_ids else np.zeros_like(candidate_crop, dtype=bool)
    except Exception:
        rejected_crop = np.zeros_like(candidate_crop, dtype=bool)

    rejected = np.zeros_like(candidate, dtype=bool)
    rejected[slices] = rejected_crop
    return rejected


def _crop_slices(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, 0) for _ in range(3))  # type: ignore[return-value]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))  # type: ignore[return-value]


def _crop_slices_with_pad(mask: np.ndarray, pad_voxels: int) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, 0) for _ in range(3))  # type: ignore[return-value]
    pad = int(max(pad_voxels, 0))
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(mask.shape))
    return tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))  # type: ignore[return-value]
