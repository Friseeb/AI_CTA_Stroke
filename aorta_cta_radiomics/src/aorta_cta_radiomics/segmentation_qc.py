"""Quality-control metrics for aorta masks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import _label


def mask_bounding_box(mask: np.ndarray) -> dict[str, int | None]:
    """Return a z/y/x bounding box for a binary mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {
            "bbox_z_min": None,
            "bbox_z_max": None,
            "bbox_y_min": None,
            "bbox_y_max": None,
            "bbox_x_min": None,
            "bbox_x_max": None,
        }
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return {
        "bbox_z_min": int(mins[0]),
        "bbox_z_max": int(maxs[0]),
        "bbox_y_min": int(mins[1]),
        "bbox_y_max": int(maxs[1]),
        "bbox_x_min": int(mins[2]),
        "bbox_x_max": int(maxs[2]),
    }


def calculate_qc_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    components_before_cleaning: int | None = None,
    mask_resampled: bool = False,
    small_mask_volume_mm3: float = 1000.0,
    large_mask_volume_mm3: float = 500000.0,
) -> dict[str, object]:
    """Calculate deterministic mask QC metrics and warning flags."""
    binary = np.asarray(mask, dtype=bool)
    voxel_count = int(binary.sum())
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    volume_mm3 = voxel_count * voxel_volume_mm3
    _, n_components = _label(binary)

    warnings: list[str] = []
    if voxel_count == 0:
        warnings.append("empty_mask")
    if 0 < volume_mm3 < small_mask_volume_mm3:
        warnings.append("mask_volume_below_configured_threshold")
    if volume_mm3 > large_mask_volume_mm3:
        warnings.append("mask_volume_above_configured_threshold")
    if n_components > 1:
        warnings.append("multiple_components_after_cleaning")
    if mask_resampled:
        warnings.append("mask_resampled_to_image_space")

    masked_values = image[binary]
    mean_hu = float(np.mean(masked_values)) if masked_values.size else np.nan

    metrics: dict[str, object] = {
        "case_id": case_id,
        "mask_volume_mm3": float(volume_mm3),
        "mask_voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        "connected_components": int(n_components),
        "components_before_cleaning": components_before_cleaning,
        "spacing_x_mm": float(spacing_xyz[0]),
        "spacing_y_mm": float(spacing_xyz[1]),
        "spacing_z_mm": float(spacing_xyz[2]),
        "shape_z": int(binary.shape[0]),
        "shape_y": int(binary.shape[1]),
        "shape_x": int(binary.shape[2]),
        "mean_hu_inside_mask": mean_hu,
        "warning_flags": ";".join(warnings),
    }
    metrics.update(mask_bounding_box(binary))
    return metrics


def qc_metrics_to_frame(metrics: dict[str, object]) -> pd.DataFrame:
    """Convert one QC metrics dictionary to a DataFrame."""
    return pd.DataFrame([metrics])
