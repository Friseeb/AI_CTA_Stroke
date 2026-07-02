"""Experimental aortic wall candidate from periaortic fat and contrast lumen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .calcification import _lumen_core_mask, _slice_lumen_reference_hu, _smooth_profile
from .features import feature_row
from .shells import _crop_around_mask, _sampling_zyx, external_shell


@dataclass
class FatClosedWallResult:
    contrast_lumen_mask: np.ndarray
    fat_support_mask: np.ndarray
    closed_outer_envelope_mask: np.ndarray
    wall_candidate_mask: np.ndarray
    hu_refined_aorta_mask: np.ndarray
    labelmap: np.ndarray
    features: pd.DataFrame


def extract_fat_closed_aortic_wall(
    image: np.ndarray,
    aorta_mask: np.ndarray,
    fat_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    outer_limit_mm: float = 5.0,
    close_radius_mm: float = 3.0,
    lumen_core_distance_mm: float = 5.0,
    centerline_core_radius_mm: float = 2.0,
    contrast_lower_margin_hu: float = 120.0,
    min_lumen_hu: float = 150.0,
    max_lumen_hu_above_reference: float | None = 300.0,
    lumen_reference_lower_fraction: float | None = None,
    lumen_reference_upper_fraction: float | None = None,
    lumen_reference_statistic: str = "median",
    require_lumen_seed_connectivity: bool = False,
    use_input_aorta_as_lumen_floor: bool = False,
    lumen_floor_mask: np.ndarray | None = None,
    smooth_lumen_profile_mm: float = 10.0,
    min_core_voxels_per_slice: int = 20,
    wall_hu_min: float = -30.0,
    wall_hu_max: float = 1200.0,
    exclude_fat_from_wall: bool = True,
    exclude_calcification_hu: float | None = None,
    include_calcification_in_wall: bool = True,
    lumen_correction_enabled: bool = False,
    lumen_correction_outer_mm: float = 2.0,
    lumen_correction_close_radius_mm: float = 1.0,
    lumen_correction_lower_margin_hu: float | None = None,
    lumen_correction_min_hu: float | None = None,
    lumen_correction_max_above_reference_hu: float | None = None,
    software_version: str = __version__,
) -> FatClosedWallResult:
    """Infer a review-only wall candidate between fat support and contrast lumen.

    This is not a histologic wall segmentation. It creates an inspectable ROI by
    closing discontinuous periaortic fat evidence near the aorta and subtracting
    the contrast-filled lumen estimated from the aortic center/core HU profile.
    """
    image_array = np.asarray(image, dtype=float)
    aorta = np.asarray(aorta_mask, dtype=bool)
    fat = np.asarray(fat_mask, dtype=bool)
    if image_array.shape != aorta.shape or fat.shape != aorta.shape:
        raise ValueError("image, aorta_mask, and fat_mask must have the same shape.")
    extra_lumen_floor = None
    if lumen_floor_mask is not None:
        extra_lumen_floor = np.asarray(lumen_floor_mask, dtype=bool)
        if extra_lumen_floor.shape != aorta.shape:
            raise ValueError("lumen_floor_mask must have the same shape as aorta_mask.")
    if not aorta.any():
        empty = np.zeros_like(aorta, dtype=bool)
        return FatClosedWallResult(
            contrast_lumen_mask=empty.copy(),
            fat_support_mask=empty.copy(),
            closed_outer_envelope_mask=empty.copy(),
            wall_candidate_mask=empty.copy(),
            hu_refined_aorta_mask=empty.copy(),
            labelmap=np.zeros_like(aorta, dtype=np.uint8),
            features=_summarize(
                image_array,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                np.array([], dtype=float),
                lumen_reference_statistic,
                spacing_xyz,
                case_id,
                software_version,
            ),
        )

    external_limit = external_shell(aorta, spacing_xyz, inner_mm=0.0, outer_mm=float(outer_limit_mm))
    analysis_roi = aorta | external_limit
    fat_support = fat & external_limit
    lumen, reference_profile = _contrast_lumen_from_centerline_hu(
        image=image_array,
        aorta_mask=aorta,
        spacing_xyz=spacing_xyz,
        lumen_core_distance_mm=lumen_core_distance_mm,
        centerline_core_radius_mm=centerline_core_radius_mm,
        contrast_lower_margin_hu=contrast_lower_margin_hu,
        min_lumen_hu=min_lumen_hu,
        max_lumen_hu_above_reference=max_lumen_hu_above_reference,
        reference_lower_fraction=lumen_reference_lower_fraction,
        reference_upper_fraction=lumen_reference_upper_fraction,
        reference_statistic=lumen_reference_statistic,
        require_seed_connectivity=require_lumen_seed_connectivity,
        smooth_lumen_profile_mm=smooth_lumen_profile_mm,
        min_core_voxels_per_slice=min_core_voxels_per_slice,
        exclude_hu_at_or_above=exclude_calcification_hu,
    )
    lumen_floor = aorta.copy() if bool(use_input_aorta_as_lumen_floor) else np.zeros_like(aorta, dtype=bool)
    if extra_lumen_floor is not None:
        lumen_floor |= extra_lumen_floor
    if bool(use_input_aorta_as_lumen_floor):
        # The input VISTA aorta is the minimum accepted lumen/aorta trace.
        # HU thresholds may add adjacent contrast-filled voxels, but they must
        # not shrink the input trace. Calcium handling is deferred to downstream
        # wall/thickness stages that receive an explicit calcium mask.
        lumen |= lumen_floor
    if bool(lumen_correction_enabled):
        correction_roi = aorta | external_shell(
            aorta,
            spacing_xyz,
            inner_mm=0.0,
            outer_mm=float(lumen_correction_outer_mm),
        )
        lumen = _correct_lumen_with_hu_threshold(
            image=image_array,
            initial_lumen=lumen,
            correction_roi=correction_roi,
            reference_profile=reference_profile,
            contrast_lower_margin_hu=(
                contrast_lower_margin_hu
                if lumen_correction_lower_margin_hu is None
                else float(lumen_correction_lower_margin_hu)
            ),
            min_lumen_hu=min_lumen_hu if lumen_correction_min_hu is None else float(lumen_correction_min_hu),
            max_lumen_hu_above_reference=(
                max_lumen_hu_above_reference
                if lumen_correction_max_above_reference_hu is None
                else float(lumen_correction_max_above_reference_hu)
            ),
            reference_lower_fraction=lumen_reference_lower_fraction,
            reference_upper_fraction=lumen_reference_upper_fraction,
            exclude_hu_at_or_above=exclude_calcification_hu,
            close_radius_mm=lumen_correction_close_radius_mm,
            spacing_xyz=spacing_xyz,
        )
        if bool(use_input_aorta_as_lumen_floor):
            lumen |= lumen_floor
    closed_outer = _closed_outer_envelope(
        aorta_mask=aorta,
        fat_support=fat_support,
        analysis_roi=analysis_roi,
        spacing_xyz=spacing_xyz,
        close_radius_mm=close_radius_mm,
    )
    closed_outer |= lumen

    wall = closed_outer & ~lumen
    if exclude_fat_from_wall:
        wall &= ~fat_support
    wall &= image_array >= float(wall_hu_min)
    wall &= image_array <= float(wall_hu_max)
    if exclude_calcification_hu is not None and not bool(include_calcification_in_wall):
        wall &= image_array < float(exclude_calcification_hu)
    hu_refined_aorta = lumen | wall

    labelmap = np.zeros_like(aorta, dtype=np.uint8)
    labelmap[closed_outer] = 4
    labelmap[fat_support] = 3
    labelmap[wall] = 2
    labelmap[lumen] = 1

    return FatClosedWallResult(
        contrast_lumen_mask=lumen,
        fat_support_mask=fat_support,
        closed_outer_envelope_mask=closed_outer,
        wall_candidate_mask=wall,
        hu_refined_aorta_mask=hu_refined_aorta,
        labelmap=labelmap,
        features=_summarize(
            image_array,
            aorta,
            lumen,
            fat_support,
            closed_outer,
            wall,
            hu_refined_aorta,
            reference_profile,
            lumen_reference_statistic,
            spacing_xyz,
            case_id,
            software_version,
        ),
    )


def _contrast_lumen_from_centerline_hu(
    image: np.ndarray,
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    lumen_core_distance_mm: float,
    centerline_core_radius_mm: float,
    contrast_lower_margin_hu: float,
    min_lumen_hu: float,
    max_lumen_hu_above_reference: float | None,
    reference_lower_fraction: float | None,
    reference_upper_fraction: float | None,
    reference_statistic: str,
    require_seed_connectivity: bool,
    smooth_lumen_profile_mm: float,
    min_core_voxels_per_slice: int,
    exclude_hu_at_or_above: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    core = _slice_centerline_core_mask(
        aorta_mask,
        spacing_xyz,
        centerline_core_radius_mm=float(centerline_core_radius_mm),
    )
    if not core.any():
        core = _lumen_core_mask(aorta_mask, spacing_xyz, core_distance_mm=float(lumen_core_distance_mm))
    reference = _slice_lumen_reference_hu(
        image,
        core,
        aorta_mask,
        min_voxels_per_slice=int(min_core_voxels_per_slice),
        statistic=reference_statistic,
    )
    reference = _smooth_profile(reference, spacing_z_mm=spacing_xyz[2], smooth_mm=float(smooth_lumen_profile_mm))
    lower, upper = _reference_hu_bounds(
        reference,
        lower_margin_hu=contrast_lower_margin_hu,
        min_hu=min_lumen_hu,
        max_above_reference_hu=max_lumen_hu_above_reference,
        reference_lower_fraction=reference_lower_fraction,
        reference_upper_fraction=reference_upper_fraction,
    )
    lumen = aorta_mask & (image >= lower[:, None, None])
    if upper is not None:
        lumen &= image <= upper[:, None, None]
    if exclude_hu_at_or_above is not None:
        lumen &= image < float(exclude_hu_at_or_above)
    if bool(require_seed_connectivity):
        seed = core & lumen
        if seed.any():
            lumen = _keep_components_touching(lumen, seed)
    return lumen, reference


def _correct_lumen_with_hu_threshold(
    image: np.ndarray,
    initial_lumen: np.ndarray,
    correction_roi: np.ndarray,
    reference_profile: np.ndarray,
    contrast_lower_margin_hu: float,
    min_lumen_hu: float,
    max_lumen_hu_above_reference: float | None,
    reference_lower_fraction: float | None,
    reference_upper_fraction: float | None,
    exclude_hu_at_or_above: float | None,
    close_radius_mm: float,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Expand contrast lumen locally outside the input trace when HU supports it."""
    roi = np.asarray(correction_roi, dtype=bool)
    seed = np.asarray(initial_lumen, dtype=bool)
    lower, upper = _reference_hu_bounds(
        reference_profile,
        lower_margin_hu=contrast_lower_margin_hu,
        min_hu=min_lumen_hu,
        max_above_reference_hu=max_lumen_hu_above_reference,
        reference_lower_fraction=reference_lower_fraction,
        reference_upper_fraction=reference_upper_fraction,
    )
    candidate = roi & (image >= lower[:, None, None])
    if upper is not None:
        candidate &= image <= upper[:, None, None]
    if exclude_hu_at_or_above is not None:
        candidate &= image < float(exclude_hu_at_or_above)
    candidate = _keep_components_touching(candidate | seed, seed)
    close_radius = max(float(close_radius_mm), 0.0)
    if close_radius <= 0 or not candidate.any():
        return candidate | seed
    try:
        from scipy import ndimage as ndi
    except Exception as exc:
        raise ImportError("SciPy ndimage is required for HU lumen correction.") from exc

    cropped_roi, slices = _crop_around_mask(roi | seed, spacing_xyz, margin_mm=close_radius)
    candidate_crop = candidate[slices]
    footprint = _physical_ball_footprint(spacing_xyz, close_radius)
    corrected_crop = ndi.binary_closing(candidate_crop, structure=footprint) | candidate_crop
    corrected_crop &= cropped_roi
    corrected = np.zeros_like(candidate, dtype=bool)
    corrected[slices] = corrected_crop
    corrected = _keep_components_touching(corrected, seed)
    return corrected | seed


def _reference_hu_bounds(
    reference: np.ndarray,
    lower_margin_hu: float,
    min_hu: float,
    max_above_reference_hu: float | None,
    reference_lower_fraction: float | None,
    reference_upper_fraction: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    ref = np.asarray(reference, dtype=float)
    if reference_lower_fraction is None:
        lower = ref - float(lower_margin_hu)
    else:
        lower = ref * (1.0 - float(reference_lower_fraction))
    lower = np.maximum(float(min_hu), lower)

    upper = None
    if reference_upper_fraction is not None:
        upper = ref * (1.0 + float(reference_upper_fraction))
    if max_above_reference_hu is not None:
        upper_margin = ref + float(max_above_reference_hu)
        upper = upper_margin if upper is None else np.minimum(upper, upper_margin)
    return lower, upper


def _slice_centerline_core_mask(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline_core_radius_mm: float,
) -> np.ndarray:
    """Approximate contrast centerline samples from deepest in-plane aortic voxels."""
    aorta = np.asarray(aorta_mask, dtype=bool)
    core = np.zeros_like(aorta, dtype=bool)
    if not aorta.any():
        return core
    try:
        from scipy import ndimage as ndi
    except Exception as exc:
        raise ImportError("SciPy ndimage is required for slice centerline-core HU estimation.") from exc

    sampling_yx = (float(spacing_xyz[1]), float(spacing_xyz[0]))
    radius = max(float(centerline_core_radius_mm), float(min(sampling_yx)))
    for z in np.flatnonzero(aorta.any(axis=(1, 2))):
        distance = ndi.distance_transform_edt(aorta[z], sampling=sampling_yx)
        max_distance = float(distance.max())
        if max_distance <= 0:
            continue
        threshold = max(max_distance - radius, max_distance * 0.7, float(min(sampling_yx)))
        core[z] = distance >= threshold
    return core


def _closed_outer_envelope(
    aorta_mask: np.ndarray,
    fat_support: np.ndarray,
    analysis_roi: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    close_radius_mm: float,
) -> np.ndarray:
    seed = np.asarray(aorta_mask, dtype=bool) | np.asarray(fat_support, dtype=bool)
    if not seed.any():
        return np.zeros_like(seed, dtype=bool)
    close_radius = max(float(close_radius_mm), 0.0)
    if close_radius <= 0:
        return seed & analysis_roi

    cropped_roi, slices = _crop_around_mask(analysis_roi, spacing_xyz, margin_mm=close_radius)
    seed_crop = seed[slices]
    footprint = _physical_ball_footprint(spacing_xyz, close_radius)
    try:
        from scipy import ndimage as ndi

        closed_crop = ndi.binary_closing(seed_crop, structure=footprint) | seed_crop
        closed_crop = ndi.binary_fill_holes(closed_crop)
    except Exception as exc:
        raise ImportError("SciPy ndimage is required for fat-closed aortic wall candidate masks.") from exc

    closed = np.zeros_like(seed, dtype=bool)
    closed[slices] = closed_crop & cropped_roi
    closed &= analysis_roi
    return _keep_components_touching(closed, aorta_mask)


def _physical_ball_footprint(spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    spacing_zyx = np.asarray(_sampling_zyx(spacing_xyz), dtype=float)
    radius = max(float(radius_mm), float(spacing_zyx.min()))
    radii_vox = np.ceil(radius / spacing_zyx).astype(int)
    z, y, x = np.ogrid[
        -radii_vox[0] : radii_vox[0] + 1,
        -radii_vox[1] : radii_vox[1] + 1,
        -radii_vox[2] : radii_vox[2] + 1,
    ]
    distance = (z * spacing_zyx[0]) ** 2 + (y * spacing_zyx[1]) ** 2 + (x * spacing_zyx[2]) ** 2
    return distance <= radius**2


def _keep_components_touching(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    seed_binary = np.asarray(seed, dtype=bool)
    if not binary.any():
        return binary
    try:
        from scipy import ndimage as ndi

        labels, n_labels = ndi.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
        if n_labels == 0:
            return np.zeros_like(binary, dtype=bool)
        keep_labels = np.unique(labels[seed_binary & binary])
        keep_labels = keep_labels[keep_labels > 0]
        if keep_labels.size == 0:
            return np.zeros_like(binary, dtype=bool)
        return np.isin(labels, keep_labels)
    except Exception as exc:
        raise ImportError("SciPy ndimage is required for connected component filtering.") from exc


def _summarize(
    image: np.ndarray,
    input_aorta: np.ndarray,
    lumen: np.ndarray,
    fat_support: np.ndarray,
    closed_outer: np.ndarray,
    wall: np.ndarray,
    hu_refined_aorta: np.ndarray,
    reference_profile: np.ndarray,
    reference_statistic: str,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    voxel_volume = float(np.prod(np.asarray(spacing_xyz, dtype=float)))
    rows: list[dict[str, object]] = []
    for name, mask in [
        ("input_aorta", input_aorta),
        ("contrast_lumen", lumen),
        ("fat_support_0_5mm", fat_support),
        ("closed_outer_envelope", closed_outer),
        ("wall_candidate", wall),
        ("hu_refined_aorta", hu_refined_aorta),
    ]:
        rows.append(
            feature_row(
                case_id,
                "aorta_wall_from_fat",
                "experimental_wall_from_fat_lumen",
                f"{name}_volume_mm3",
                float(np.asarray(mask, dtype=bool).sum() * voxel_volume),
                "mm3",
                "",
                "aortic_wall_from_fat_lumen",
                software_version,
            )
        )
    rows.append(
        feature_row(
            case_id,
            "aorta_wall_from_fat",
            "experimental_wall_from_fat_lumen",
            "lumen_added_outside_input_aorta_volume_mm3",
            float((np.asarray(lumen, dtype=bool) & ~np.asarray(input_aorta, dtype=bool)).sum() * voxel_volume),
            "mm3",
            "",
            "aortic_wall_from_fat_lumen",
            software_version,
        )
    )
    rows.append(
        feature_row(
            case_id,
            "aorta_wall_from_fat",
            "experimental_wall_from_fat_lumen",
            "hu_refined_aorta_added_volume_mm3",
            float(
                (
                    np.asarray(hu_refined_aorta, dtype=bool)
                    & ~np.asarray(input_aorta, dtype=bool)
                ).sum()
                * voxel_volume
            ),
            "mm3",
            "",
            "aortic_wall_from_fat_lumen",
            software_version,
        )
    )
    rows.append(
        feature_row(
            case_id,
            "aorta_wall_from_fat",
            "experimental_wall_from_fat_lumen",
            "wall_candidate_mean_HU",
            _nan_mean(image[wall]),
            "HU",
            "",
            "aortic_wall_from_fat_lumen",
            software_version,
        )
    )
    rows.append(
        feature_row(
            case_id,
            "aorta_wall_from_fat",
            "experimental_wall_from_fat_lumen",
            "lumen_reference_profile_median_HU",
            _nan_median(_active_slice_values(reference_profile, input_aorta)),
            "HU",
            f"per_slice_{reference_statistic.lower()}",
            "aortic_wall_from_fat_lumen",
            software_version,
        )
    )
    return pd.DataFrame(rows)


def _active_slice_values(profile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    active = np.asarray(mask, dtype=bool).any(axis=(1, 2))
    if active.shape[0] != values.shape[0] or not active.any():
        return values
    return values[active]


def _nan_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _nan_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmedian(values))
