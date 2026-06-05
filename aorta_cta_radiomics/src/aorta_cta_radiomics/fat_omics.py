"""Periaortic adipose tissue masks and fat-omics features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .features import feature_row
from .shells import _crop_around_mask, _distance_transform_edt, _sampling_zyx


@dataclass
class FatOmicsResult:
    fat_mask: np.ndarray
    periaortic_roi_mask: np.ndarray
    fat_layer_masks: dict[str, np.ndarray]
    distance_to_aorta_mm: np.ndarray
    features: pd.DataFrame


def extract_periaortic_fat_omics(
    image: np.ndarray,
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    centerline_points: pd.DataFrame | None = None,
    segment_labels: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
    external_radius_mm: float = 5.0,
    adipose_hu_min: float = -190.0,
    adipose_hu_max: float = -30.0,
    high_hu_bins: dict[str, tuple[float, float]] | None = None,
    radial_bins_mm: list[tuple[float, float]] | None = None,
    angle_bins: int = 12,
    texture_levels: int = 16,
    mask_name: str = "periaortic_fat",
    software_version: str = __version__,
) -> FatOmicsResult:
    """Create a periaortic fat mask and hand-crafted fat-omics feature table."""
    image_array = np.asarray(image, dtype=float)
    aorta = np.asarray(aorta_mask, dtype=bool)
    if image_array.shape != aorta.shape:
        raise ValueError("image and aorta_mask must have the same shape.")

    high_hu_bins = high_hu_bins or {
        "m70_m30": (-70.0, -30.0),
        "m50_m30": (-50.0, -30.0),
    }
    radial_bins_mm = radial_bins_mm or [(0.0, 2.0), (2.0, float(external_radius_mm))]

    roi_mask, distance_to_aorta = _periaortic_roi(aorta, spacing_xyz, external_radius_mm)
    fat_mask = roi_mask & (image_array >= adipose_hu_min) & (image_array <= adipose_hu_max)
    fat_layer_masks = create_fat_layer_masks(fat_mask, distance_to_aorta, radial_bins_mm)
    features = summarize_periaortic_fat_omics(
        image=image_array,
        fat_mask=fat_mask,
        periaortic_roi_mask=roi_mask,
        aorta_mask=aorta,
        distance_to_aorta_mm=distance_to_aorta,
        spacing_xyz=spacing_xyz,
        case_id=case_id,
        centerline_points=centerline_points,
        segment_labels=segment_labels,
        segment_names=segment_names or {},
        high_hu_bins=high_hu_bins,
        radial_bins_mm=radial_bins_mm,
        angle_bins=angle_bins,
        texture_levels=texture_levels,
        mask_name=mask_name,
        software_version=software_version,
    )
    return FatOmicsResult(
        fat_mask=fat_mask,
        periaortic_roi_mask=roi_mask,
        fat_layer_masks=fat_layer_masks,
        distance_to_aorta_mm=distance_to_aorta,
        features=features,
    )


def create_fat_layer_masks(
    fat_mask: np.ndarray,
    distance_to_aorta_mm: np.ndarray,
    radial_bins_mm: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Split a fat mask into named physical-distance layers."""
    fat = np.asarray(fat_mask, dtype=bool)
    distances = np.asarray(distance_to_aorta_mm, dtype=float)
    layers: dict[str, np.ndarray] = {}
    for low, high in radial_bins_mm:
        suffix = _bin_suffix(low, high)
        layers[f"periaortic_fat_{suffix}"] = fat & (distances > float(low)) & (distances <= float(high))
    return layers


def summarize_periaortic_fat_omics(
    image: np.ndarray,
    fat_mask: np.ndarray,
    periaortic_roi_mask: np.ndarray,
    aorta_mask: np.ndarray,
    distance_to_aorta_mm: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    centerline_points: pd.DataFrame | None = None,
    segment_labels: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
    high_hu_bins: dict[str, tuple[float, float]] | None = None,
    radial_bins_mm: list[tuple[float, float]] | None = None,
    angle_bins: int = 12,
    texture_levels: int = 16,
    mask_name: str = "periaortic_fat",
    software_version: str = __version__,
) -> pd.DataFrame:
    """Summarize periaortic adipose burden, HU distribution, spatial, and texture proxies."""
    image_array = np.asarray(image, dtype=float)
    fat = np.asarray(fat_mask, dtype=bool)
    roi = np.asarray(periaortic_roi_mask, dtype=bool)
    aorta = np.asarray(aorta_mask, dtype=bool)
    distances = np.asarray(distance_to_aorta_mm, dtype=float)
    if image_array.shape != fat.shape or image_array.shape != roi.shape or image_array.shape != aorta.shape:
        raise ValueError("image, fat_mask, periaortic_roi_mask, and aorta_mask must have the same shape.")

    high_hu_bins = high_hu_bins or {
        "m70_m30": (-70.0, -30.0),
        "m50_m30": (-50.0, -30.0),
    }
    radial_bins_mm = radial_bins_mm or [(0.0, 2.0), (2.0, 5.0)]

    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    values = image_array[fat]
    fat_voxels = int(fat.sum())
    fat_volume_mm3 = float(fat_voxels * voxel_volume_mm3)
    aortic_length_cm = _aortic_length_cm(aorta, spacing_xyz, centerline_points)
    texture = _texture_features(image_array, fat, texture_levels)
    sector = _sector_features(image_array, fat, aorta, angle_bins)
    rows = [
        _row(case_id, "periaortic_fat", "periaortic_fat_volume_mm3", fat_volume_mm3, "mm3", mask_name, software_version),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_fat_volume_per_cm",
            _safe_divide(fat_volume_mm3, aortic_length_cm),
            "mm3/cm",
            mask_name,
            software_version,
        ),
        _row(case_id, "periaortic_fat", "periaortic_mean_HU", _nan_stat(values, np.mean), "HU", mask_name, software_version),
        _row(case_id, "periaortic_fat", "periaortic_median_HU", _nan_stat(values, np.median), "HU", mask_name, software_version),
        _row(case_id, "periaortic_fat", "periaortic_std_HU", _nan_stat(values, np.std), "HU", mask_name, software_version),
        _row(case_id, "periaortic_fat", "periaortic_skewness_HU", _skewness(values), "unitless", mask_name, software_version),
        _row(case_id, "periaortic_fat", "periaortic_kurtosis_HU", _kurtosis(values), "unitless", mask_name, software_version),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_radial_gradient",
            _radial_gradient(image_array, fat, distances),
            "HU/mm",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_circumferential_sector_max_HU",
            sector["max_sector_mean_hu"],
            "HU",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_circumferential_sector_std_HU",
            sector["std_sector_mean_hu"],
            "HU",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glcm_cluster_prominence",
            texture["glcm_cluster_prominence"],
            "unitless",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glcm_cluster_tendency",
            texture["glcm_cluster_tendency"],
            "unitless",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glrlm_short_run_emphasis",
            texture["glrlm_short_run_emphasis"],
            "unitless",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glrlm_long_run_emphasis",
            texture["glrlm_long_run_emphasis"],
            "unitless",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glszm_small_zone_emphasis",
            texture["glszm_small_zone_emphasis"],
            "unitless",
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "periaortic_fat",
            "periaortic_glszm_large_zone_emphasis",
            texture["glszm_large_zone_emphasis"],
            "unitless",
            mask_name,
            software_version,
        ),
    ]

    for label, (low, high) in high_hu_bins.items():
        bin_mask = fat & (image_array >= float(low)) & (image_array <= float(high))
        rows.append(
            _row(
                case_id,
                "periaortic_fat",
                f"periaortic_high_HU_fraction_{label}",
                _safe_divide(int(bin_mask.sum()), fat_voxels),
                "fraction",
                mask_name,
                software_version,
            )
        )

    for low, high in radial_bins_mm:
        bin_mask = fat & (distances > float(low)) & (distances <= float(high))
        suffix = _bin_suffix(low, high)
        rows.append(
            _row(
                case_id,
                "periaortic_fat",
                f"periaortic_fat_volume_{suffix}",
                float(bin_mask.sum() * voxel_volume_mm3),
                "mm3",
                mask_name,
                software_version,
            )
        )
        rows.append(
            _row(
                case_id,
                "periaortic_fat",
                f"periaortic_mean_HU_{suffix}",
                _nan_stat(image_array[bin_mask], np.mean),
                "HU",
                mask_name,
                software_version,
            )
        )

    rows.extend(
        _segment_rows(
            image=image_array,
            fat=fat,
            aorta=aorta,
            segment_labels=segment_labels,
            segment_names=segment_names or {},
            spacing_xyz=spacing_xyz,
            voxel_volume_mm3=voxel_volume_mm3,
            case_id=case_id,
            mask_name=mask_name,
            software_version=software_version,
        )
    )
    return pd.DataFrame(rows)


def _periaortic_roi(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    external_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    aorta = np.asarray(aorta_mask, dtype=bool)
    roi = np.zeros_like(aorta, dtype=bool)
    distances = np.full(aorta.shape, np.nan, dtype=np.float32)
    if external_radius_mm <= 0 or not aorta.any():
        return roi, distances
    cropped, slices = _crop_around_mask(aorta, spacing_xyz, margin_mm=float(external_radius_mm))
    distance_crop = _distance_transform_edt(~cropped, sampling=_sampling_zyx(spacing_xyz))
    roi_crop = (~cropped) & (distance_crop > 0) & (distance_crop <= float(external_radius_mm))
    roi[slices] = roi_crop
    distance_view = distances[slices]
    distance_view[roi_crop] = distance_crop[roi_crop]
    distances[slices] = distance_view
    return roi, distances


def _row(
    case_id: str,
    region: str,
    feature_name: str,
    value: object,
    units: str,
    mask_name: str,
    software_version: str,
) -> dict[str, object]:
    return feature_row(
        case_id=case_id,
        region=region,
        feature_group="fat_omics",
        feature_name=feature_name,
        feature_value=value,
        units=units,
        mask_name=mask_name,
        software_version=software_version,
    )


def _aortic_length_cm(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline_points: pd.DataFrame | None,
) -> float:
    if centerline_points is not None and not centerline_points.empty:
        required = {"x", "y", "z"}
        if required.issubset(centerline_points.columns):
            points = centerline_points[["x", "y", "z"]].to_numpy(dtype=float)
            if len(points) >= 2:
                return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / 10.0
    coords = np.argwhere(aorta_mask)
    if coords.size == 0:
        return 0.0
    return float((coords[:, 0].max() - coords[:, 0].min() + 1) * spacing_xyz[2] / 10.0)


def _nan_stat(values: np.ndarray, func: object) -> float:
    if values.size == 0:
        return float("nan")
    return float(func(values))


def _skewness(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4))


def _radial_gradient(image: np.ndarray, fat: np.ndarray, distances: np.ndarray) -> float:
    valid = fat & np.isfinite(distances)
    if int(valid.sum()) < 2:
        return float("nan")
    x = distances[valid].astype(float)
    y = image[valid].astype(float)
    if float(np.ptp(x)) == 0:
        return 0.0
    return float(np.polyfit(x, y, deg=1)[0])


def _sector_features(image: np.ndarray, fat: np.ndarray, aorta: np.ndarray, angle_bins: int) -> dict[str, float]:
    bin_count = max(int(angle_bins), 4)
    sums = np.zeros(bin_count, dtype=float)
    counts = np.zeros(bin_count, dtype=float)
    for z in np.where(fat.any(axis=(1, 2)))[0]:
        fy, fx = np.where(fat[z])
        ay, ax = np.where(aorta[z])
        if fy.size == 0 or ay.size == 0:
            continue
        center_y = float(ay.mean())
        center_x = float(ax.mean())
        angles = np.mod(np.arctan2(fy - center_y, fx - center_x), 2.0 * np.pi)
        bins = np.floor(angles / (2.0 * np.pi) * bin_count).astype(int)
        values = image[z, fy, fx]
        np.add.at(sums, bins, values)
        np.add.at(counts, bins, 1)
    valid = counts > 0
    if not valid.any():
        return {"max_sector_mean_hu": float("nan"), "std_sector_mean_hu": float("nan")}
    sector_means = sums[valid] / counts[valid]
    return {
        "max_sector_mean_hu": float(np.max(sector_means)),
        "std_sector_mean_hu": float(np.std(sector_means)),
    }


def _texture_features(image: np.ndarray, mask: np.ndarray, levels: int) -> dict[str, float]:
    if not mask.any():
        return {
            "glcm_cluster_prominence": float("nan"),
            "glcm_cluster_tendency": float("nan"),
            "glrlm_short_run_emphasis": float("nan"),
            "glrlm_long_run_emphasis": float("nan"),
            "glszm_small_zone_emphasis": float("nan"),
            "glszm_large_zone_emphasis": float("nan"),
        }
    coords = np.argwhere(mask)
    slices = tuple(slice(int(coords[:, axis].min()), int(coords[:, axis].max()) + 1) for axis in range(3))
    cropped_image = image[slices]
    cropped_mask = mask[slices]
    quantized = _quantize(cropped_image, cropped_mask, levels)
    return {
        **_glcm_features(quantized, cropped_mask, levels),
        **_glrlm_features(quantized, cropped_mask),
        **_glszm_features(quantized, cropped_mask, levels),
    }


def _quantize(image: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    q = np.zeros(image.shape, dtype=np.uint8)
    values = image[mask]
    if values.size == 0:
        return q
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        q[mask] = 0
        return q
    scaled = np.floor((np.clip(image[mask], vmin, vmax) - vmin) / (vmax - vmin) * (levels - 1))
    q[mask] = scaled.astype(np.uint8)
    return q


def _glcm_features(quantized: np.ndarray, mask: np.ndarray, levels: int) -> dict[str, float]:
    matrix = np.zeros((levels, levels), dtype=float)
    for axis in range(3):
        slices_a = [slice(None), slice(None), slice(None)]
        slices_b = [slice(None), slice(None), slice(None)]
        slices_a[axis] = slice(0, -1)
        slices_b[axis] = slice(1, None)
        pair_mask = mask[tuple(slices_a)] & mask[tuple(slices_b)]
        if not pair_mask.any():
            continue
        first = quantized[tuple(slices_a)][pair_mask]
        second = quantized[tuple(slices_b)][pair_mask]
        np.add.at(matrix, (first, second), 1)
        np.add.at(matrix, (second, first), 1)
    total = float(matrix.sum())
    if total == 0:
        return {"glcm_cluster_prominence": float("nan"), "glcm_cluster_tendency": float("nan")}
    p = matrix / total
    i, j = np.indices(p.shape)
    ux = float((p.sum(axis=1) * np.arange(levels)).sum())
    uy = float((p.sum(axis=0) * np.arange(levels)).sum())
    cluster = i + j - ux - uy
    return {
        "glcm_cluster_prominence": float(np.sum((cluster**4) * p)),
        "glcm_cluster_tendency": float(np.sum((cluster**2) * p)),
    }


def _glrlm_features(quantized: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    lengths: list[int] = []
    for axis in range(3):
        moved_mask = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis])
        moved_quantized = np.moveaxis(quantized, axis, -1).reshape(-1, mask.shape[axis])
        for mask_line, q_line in zip(moved_mask, moved_quantized, strict=False):
            active = np.where(mask_line)[0]
            if active.size == 0:
                continue
            run_start = int(active[0])
            previous = int(active[0])
            previous_level = int(q_line[previous])
            for index in active[1:]:
                index = int(index)
                level = int(q_line[index])
                if index == previous + 1 and level == previous_level:
                    previous = index
                    continue
                lengths.append(previous - run_start + 1)
                run_start = previous = index
                previous_level = level
            lengths.append(previous - run_start + 1)
    if not lengths:
        return {"glrlm_short_run_emphasis": float("nan"), "glrlm_long_run_emphasis": float("nan")}
    arr = np.asarray(lengths, dtype=float)
    return {
        "glrlm_short_run_emphasis": float(np.mean(1.0 / (arr**2))),
        "glrlm_long_run_emphasis": float(np.mean(arr**2)),
    }


def _glszm_features(quantized: np.ndarray, mask: np.ndarray, levels: int) -> dict[str, float]:
    try:
        from scipy import ndimage as ndi
    except Exception:
        return {"glszm_small_zone_emphasis": float("nan"), "glszm_large_zone_emphasis": float("nan")}

    zone_sizes: list[float] = []
    structure = np.ones((3, 3, 3), dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {"glszm_small_zone_emphasis": float("nan"), "glszm_large_zone_emphasis": float("nan")}
    slices = tuple(slice(int(coords[:, axis].min()), int(coords[:, axis].max()) + 1) for axis in range(3))
    cropped_mask = mask[slices]
    cropped_q = quantized[slices]
    for level in range(levels):
        level_mask = cropped_mask & (cropped_q == level)
        if not level_mask.any():
            continue
        labels, count = ndi.label(level_mask, structure=structure)
        if count == 0:
            continue
        sizes = ndi.sum(level_mask, labels, index=np.arange(1, count + 1))
        zone_sizes.extend(float(size) for size in np.asarray(sizes).ravel() if float(size) > 0)
    if not zone_sizes:
        return {"glszm_small_zone_emphasis": float("nan"), "glszm_large_zone_emphasis": float("nan")}
    arr = np.asarray(zone_sizes, dtype=float)
    return {
        "glszm_small_zone_emphasis": float(np.mean(1.0 / (arr**2))),
        "glszm_large_zone_emphasis": float(np.mean(arr**2)),
    }


def _segment_rows(
    image: np.ndarray,
    fat: np.ndarray,
    aorta: np.ndarray,
    segment_labels: np.ndarray | None,
    segment_names: dict[int, str],
    spacing_xyz: tuple[float, float, float],
    voxel_volume_mm3: float,
    case_id: str,
    mask_name: str,
    software_version: str,
) -> list[dict[str, object]]:
    if segment_labels is None:
        return []
    labels = np.asarray(segment_labels)
    if labels.shape != fat.shape:
        raise ValueError("segment_labels must have the same shape as fat_mask.")
    nonzero_labels = sorted(int(item) for item in np.unique(labels) if int(item) != 0)
    if not nonzero_labels:
        return []

    if nonzero_labels == [1] and segment_names.get(1, "") == "whole_aorta":
        assigned = np.zeros_like(labels, dtype=np.int16)
        assigned[fat] = 1
    else:
        assigned = _nearest_segment_labels(fat, aorta, labels, spacing_xyz)

    rows: list[dict[str, object]] = []
    for label in nonzero_labels:
        segment_name = segment_names.get(label, f"label_{label}")
        region = f"aorta_segment:{segment_name}"
        segment_fat = fat & (assigned == label)
        rows.append(_row(case_id, region, "aortic_segment_label", label, "label", mask_name, software_version))
        rows.append(
            _row(
                case_id,
                region,
                "periaortic_fat_volume_mm3",
                float(segment_fat.sum() * voxel_volume_mm3),
                "mm3",
                mask_name,
                software_version,
            )
        )
        rows.append(
            _row(
                case_id,
                region,
                "periaortic_mean_HU",
                _nan_stat(image[segment_fat], np.mean),
                "HU",
                mask_name,
                software_version,
            )
        )
    return rows


def _nearest_segment_labels(
    fat: np.ndarray,
    aorta: np.ndarray,
    segment_labels: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    assigned = np.zeros_like(segment_labels, dtype=np.int16)
    if not fat.any() or not aorta.any():
        return assigned
    try:
        from scipy import ndimage as ndi

        _, indices = ndi.distance_transform_edt(
            ~aorta,
            sampling=_sampling_zyx(spacing_xyz),
            return_indices=True,
        )
        nearest = segment_labels[tuple(indices)]
        assigned[fat] = nearest[fat]
        return assigned
    except Exception:
        return assigned


def _bin_suffix(low: float, high: float) -> str:
    return f"{_format_mm(low)}_{_format_mm(high)}mm"


def _format_mm(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _safe_divide(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator) / denominator
