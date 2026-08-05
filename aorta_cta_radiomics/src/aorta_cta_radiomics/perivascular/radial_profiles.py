"""Radial and longitudinal aggregation without discarding local profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ._spatial import validate_spacing, voxel_volume_mm3
from .composition import TISSUE_NAMES, TissueLabel


_PROFILE_LABELS = (
    TissueLabel.ADIPOSE,
    TissueLabel.SKELETAL_MUSCLE,
    TissueLabel.VEIN,
    TissueLabel.BONE,
    TissueLabel.THYROID_GLAND,
    TissueLabel.OTHER_SOFT_TISSUE,
    TissueLabel.HIGH_DENSITY,
    TissueLabel.UNCERTAIN,
)


def _profile_record(
    mask: np.ndarray,
    image: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    voxel_volume: float,
) -> dict[str, object]:
    total = mask
    assessed = total & valid
    count = int(assessed.sum())
    record: dict[str, object] = {
        "total_voxels": int(total.sum()),
        "total_volume_mm3": float(total.sum() * voxel_volume),
        "valid_voxels": count,
        "valid_volume_mm3": float(count * voxel_volume),
    }
    for label in _PROFILE_LABELS:
        name = TISSUE_NAMES[int(label)]
        label_mask = assessed & (labels == int(label))
        label_count = int(label_mask.sum())
        record[f"{name}_voxels"] = label_count
        record[f"{name}_volume_mm3"] = float(label_count * voxel_volume)
        record[f"{name}_fraction"] = float(label_count / count) if count else float("nan")
        if label == TissueLabel.ADIPOSE:
            values = image[label_mask & np.isfinite(image)]
            record["adipose_mean_hu"] = float(np.mean(values)) if values.size else float("nan")
            record["adipose_median_hu"] = float(np.median(values)) if values.size else float("nan")
    values = image[assessed & np.isfinite(image)]
    record["mean_cta_hu"] = float(np.mean(values)) if values.size else float("nan")
    record["median_cta_hu"] = float(np.median(values)) if values.size else float("nan")
    record["qc_status"] = "ok" if count else "empty_profile_bin"
    return record


def aggregate_radial_profiles(
    cta_hu: np.ndarray,
    tissue_labels: np.ndarray,
    radial_bands: Mapping[str, np.ndarray],
    spacing_xyz: Sequence[float],
    *,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Summarize each physical radial band while preserving its identity."""
    image = np.asarray(cta_hu, dtype=float)
    labels = np.asarray(tissue_labels)
    if image.ndim != 3 or labels.shape != image.shape:
        raise ValueError("cta_hu and tissue_labels must be same-shaped 3D arrays.")
    valid = np.ones(image.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != image.shape:
        raise ValueError("valid_mask must share the image shape.")
    voxel_volume = voxel_volume_mm3(validate_spacing(spacing_xyz))
    records: list[dict[str, object]] = []
    for order, (name, mask) in enumerate(radial_bands.items()):
        band = np.asarray(mask, dtype=bool)
        if band.shape != image.shape:
            raise ValueError(f"Radial band {name} does not share the image shape.")
        # Compact each sparse shell once.  _profile_record is dimension
        # agnostic, so this is exactly equivalent to repeatedly combining
        # CTA-sized Boolean arrays for every tissue class.
        compact_mask = np.ones(int(band.sum()), dtype=bool)
        record = _profile_record(
            compact_mask,
            image[band],
            labels[band],
            valid[band],
            voxel_volume,
        )
        record.update({"radial_band": str(name), "radial_band_order": order})
        records.append(record)
    return records


def aggregate_longitudinal_profiles(
    cta_hu: np.ndarray,
    tissue_labels: np.ndarray,
    roi_mask: np.ndarray,
    path_mm_map: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    bin_length_mm: float = 1.0,
    valid_mask: np.ndarray | None = None,
    vessel_length_mm: float | None = None,
) -> list[dict[str, object]]:
    """Aggregate cross-sectional tissue composition along physical path bins."""
    image = np.asarray(cta_hu, dtype=float)
    labels = np.asarray(tissue_labels)
    roi = np.asarray(roi_mask, dtype=bool)
    path = np.asarray(path_mm_map, dtype=float)
    if (
        image.ndim != 3
        or labels.shape != image.shape
        or roi.shape != image.shape
        or path.shape != image.shape
    ):
        raise ValueError("Image, labels, ROI, and path map must be same-shaped 3D arrays.")
    if not np.isfinite(bin_length_mm) or bin_length_mm <= 0:
        raise ValueError("bin_length_mm must be positive and finite.")
    domain = roi & np.isfinite(path)
    valid = domain.copy()
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != image.shape:
            raise ValueError("valid_mask must share the image shape.")
        valid &= supplied
    finite_path = path[domain]
    inferred_length = float(np.max(finite_path)) if finite_path.size else 0.0
    length = inferred_length if vessel_length_mm is None else float(vessel_length_mm)
    if not np.isfinite(length) or length < 0:
        raise ValueError("vessel_length_mm must be finite and non-negative.")
    edges = np.arange(0.0, length + float(bin_length_mm), float(bin_length_mm))
    if len(edges) < 2:
        edges = np.asarray([0.0, float(bin_length_mm)])
    if edges[-1] < length:
        edges = np.append(edges, length)
    voxel_volume = voxel_volume_mm3(validate_spacing(spacing_xyz))
    compact_image = image[domain]
    compact_labels = labels[domain]
    compact_valid = valid[domain]
    compact_path = path[domain]
    records: list[dict[str, object]] = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == len(edges) - 2:
            selected = (compact_path >= lower) & (compact_path <= upper)
        else:
            selected = (compact_path >= lower) & (compact_path < upper)
        record = _profile_record(
            selected,
            compact_image,
            compact_labels,
            compact_valid,
            voxel_volume,
        )
        record.update(
            {
                "longitudinal_bin": index,
                "path_start_mm": lower,
                "path_end_mm": upper,
                "path_mid_mm": 0.5 * (lower + upper),
                "bin_length_mm": upper - lower,
            }
        )
        records.append(record)
    return records


def weighted_profile_gradient(
    records: Sequence[Mapping[str, object]],
    value_key: str,
    *,
    position_key: str = "path_mid_mm",
    weight_key: str = "valid_voxels",
) -> float:
    """Return a weighted least-squares profile gradient in value/mm."""
    x = np.asarray([float(record[position_key]) for record in records], dtype=float)
    y = np.asarray([float(record[value_key]) for record in records], dtype=float)
    weights = np.asarray([float(record.get(weight_key, 1.0)) for record in records], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if keep.sum() < 2 or np.ptp(x[keep]) <= 0:
        return float("nan")
    design = np.column_stack((x[keep], np.ones(int(keep.sum()))))
    weighted = design * np.sqrt(weights[keep])[:, None]
    target = y[keep] * np.sqrt(weights[keep])
    slope, _ = np.linalg.lstsq(weighted, target, rcond=None)[0]
    return float(slope)


def contiguous_profile_runs(
    records: Sequence[Mapping[str, object]],
    value_key: str,
    *,
    threshold: float,
    comparison: str = "below",
) -> list[dict[str, float | int]]:
    """Describe contiguous longitudinal runs above or below a threshold."""
    if comparison not in {"below", "above"}:
        raise ValueError("comparison must be 'below' or 'above'.")
    values = np.asarray([float(record[value_key]) for record in records], dtype=float)
    selected = np.isfinite(values) & (values < threshold if comparison == "below" else values > threshold)
    runs: list[dict[str, float | int]] = []
    start: int | None = None
    for index, is_selected in enumerate(np.append(selected, False)):
        if is_selected and start is None:
            start = index
        elif not is_selected and start is not None:
            end = index - 1
            path_start = float(records[start]["path_start_mm"])
            path_end = float(records[end]["path_end_mm"])
            runs.append(
                {
                    "start_bin": start,
                    "end_bin": end,
                    "path_start_mm": path_start,
                    "path_end_mm": path_end,
                    "length_mm": path_end - path_start,
                }
            )
            start = None
    return runs


# Singular aliases retained for intuitive callers.
aggregate_radial_profile = aggregate_radial_profiles
aggregate_longitudinal_profile = aggregate_longitudinal_profiles


__all__ = [
    "aggregate_longitudinal_profile",
    "aggregate_longitudinal_profiles",
    "aggregate_radial_profile",
    "aggregate_radial_profiles",
    "contiguous_profile_runs",
    "weighted_profile_gradient",
]
