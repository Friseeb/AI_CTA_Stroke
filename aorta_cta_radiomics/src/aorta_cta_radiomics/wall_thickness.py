"""Mask-derived aortic wall thickness measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import __version__


@dataclass(frozen=True)
class WallThicknessResult:
    """Wall thickness maps and tabular summaries."""

    lumen_mask: np.ndarray
    wall_mask: np.ndarray
    inner_surface_mask: np.ndarray
    outer_surface_mask: np.ndarray
    thickness_map_mm: np.ndarray
    inner_surface_thickness_map_mm: np.ndarray
    outer_surface_thickness_map_mm: np.ndarray
    thickness_bin_labelmap: np.ndarray
    summary: pd.DataFrame


def measure_wall_thickness(
    lumen_mask: np.ndarray,
    wall_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    software_version: str = __version__,
) -> WallThicknessResult:
    """Estimate local wall thickness from lumen and wall binary masks.

    The method treats wall thickness as the distance between inner and outer
    wall surfaces in the binary masks. This is a reproducible geometry feature,
    not a histologic wall boundary measurement.
    """
    lumen = np.asarray(lumen_mask, dtype=bool)
    wall = np.asarray(wall_mask, dtype=bool)
    if lumen.shape != wall.shape:
        raise ValueError(f"lumen_mask and wall_mask shapes differ: {lumen.shape} vs {wall.shape}.")
    if lumen.ndim != 3:
        raise ValueError("lumen_mask and wall_mask must be 3D arrays.")

    wall = wall & ~lumen
    full_vessel = lumen | wall
    empty_float = np.zeros(lumen.shape, dtype=np.float32)
    empty_bool = np.zeros(lumen.shape, dtype=bool)
    empty_labels = np.zeros(lumen.shape, dtype=np.uint16)
    if not wall.any() or not lumen.any():
        return WallThicknessResult(
            lumen_mask=lumen,
            wall_mask=wall,
            inner_surface_mask=empty_bool,
            outer_surface_mask=empty_bool,
            thickness_map_mm=empty_float,
            inner_surface_thickness_map_mm=empty_float,
            outer_surface_thickness_map_mm=empty_float,
            thickness_bin_labelmap=empty_labels,
            summary=_summary(case_id, np.array([], dtype=float), np.array([], dtype=float), spacing_xyz, software_version),
        )

    sampling_zyx = _spacing_zyx(spacing_xyz)
    min_spacing = float(min(spacing_xyz))
    structure = ndi.generate_binary_structure(3, 1)
    inner_surface = wall & ndi.binary_dilation(lumen, structure=structure)
    outer_surface = wall & ndi.binary_dilation(~full_vessel, structure=structure)

    if not inner_surface.any():
        inner_surface = wall & (ndi.distance_transform_edt(~lumen, sampling=sampling_zyx) <= min_spacing * 1.5)
    if not outer_surface.any():
        outer_surface = wall & (ndi.distance_transform_edt(full_vessel, sampling=sampling_zyx) <= min_spacing * 1.5)

    distance_to_inner = ndi.distance_transform_edt(~inner_surface, sampling=sampling_zyx)
    distance_to_outer = ndi.distance_transform_edt(~outer_surface, sampling=sampling_zyx)

    thickness = np.zeros(lumen.shape, dtype=np.float32)
    thickness[wall] = (distance_to_inner[wall] + distance_to_outer[wall] + min_spacing).astype(np.float32)

    inner_surface_thickness = np.zeros(lumen.shape, dtype=np.float32)
    inner_surface_thickness[inner_surface] = (distance_to_outer[inner_surface] + min_spacing).astype(np.float32)

    outer_surface_thickness = np.zeros(lumen.shape, dtype=np.float32)
    outer_surface_thickness[outer_surface] = (distance_to_inner[outer_surface] + min_spacing).astype(np.float32)

    labelmap = thickness_bins(thickness, wall)
    summary = _summary(
        case_id=case_id,
        wall_values=thickness[wall],
        outer_surface_values=outer_surface_thickness[outer_surface],
        spacing_xyz=spacing_xyz,
        software_version=software_version,
    )
    return WallThicknessResult(
        lumen_mask=lumen,
        wall_mask=wall,
        inner_surface_mask=inner_surface,
        outer_surface_mask=outer_surface,
        thickness_map_mm=thickness,
        inner_surface_thickness_map_mm=inner_surface_thickness,
        outer_surface_thickness_map_mm=outer_surface_thickness,
        thickness_bin_labelmap=labelmap,
        summary=summary,
    )


def thickness_bins(thickness_map_mm: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Create an integer labelmap for wall thickness bins."""
    values = np.asarray(thickness_map_mm, dtype=float)
    wall = np.asarray(wall_mask, dtype=bool)
    labels = np.zeros(values.shape, dtype=np.uint16)
    labels[wall & (values > 0.0) & (values < 2.0)] = 1
    labels[wall & (values >= 2.0) & (values < 3.0)] = 2
    labels[wall & (values >= 3.0) & (values < 4.0)] = 3
    labels[wall & (values >= 4.0) & (values < 5.0)] = 4
    labels[wall & (values >= 5.0)] = 5
    return labels


def wall_thickness_threshold_mask(
    thickness_map_mm: np.ndarray,
    wall_mask: np.ndarray,
    threshold_mm: float = 4.0,
    inclusive: bool = False,
) -> np.ndarray:
    """Return wall voxels above a thickness threshold.

    By default this uses a strict greater-than threshold, matching wording such
    as "more than 4 mm." Set ``inclusive=True`` for >= threshold.
    """
    values = np.asarray(thickness_map_mm, dtype=float)
    wall = np.asarray(wall_mask, dtype=bool)
    if values.shape != wall.shape:
        raise ValueError(f"thickness_map_mm and wall_mask shapes differ: {values.shape} vs {wall.shape}.")
    if inclusive:
        return wall & (values >= float(threshold_mm))
    return wall & (values > float(threshold_mm))


def thickness_threshold_summary(
    case_id: str,
    threshold_mask: np.ndarray,
    wall_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    threshold_mm: float,
    software_version: str = __version__,
) -> pd.DataFrame:
    """Summarize a thresholded wall-thickness marker mask."""
    threshold = np.asarray(threshold_mask, dtype=bool)
    wall = np.asarray(wall_mask, dtype=bool)
    if threshold.shape != wall.shape:
        raise ValueError(f"threshold_mask and wall_mask shapes differ: {threshold.shape} vs {wall.shape}.")
    voxel_volume = float(np.prod(spacing_xyz))
    count = int(threshold.sum())
    wall_count = int(wall.sum())
    fraction = float(count / wall_count) if wall_count else np.nan
    threshold_text = f"> {float(threshold_mm):g} mm"
    rows = [
        _row(case_id, "wall_thickness_threshold", "wall_thickness_gt4mm_voxel_count", count, "voxels", software_version),
        _row(case_id, "wall_thickness_threshold", "wall_thickness_gt4mm_volume_mm3", count * voxel_volume, "mm3", software_version),
        _row(case_id, "wall_thickness_threshold", "wall_thickness_gt4mm_wall_fraction", fraction, "fraction", software_version),
    ]
    for row in rows:
        row["threshold_if_applicable"] = threshold_text
        row["mask_name"] = "wall_thickness_gt4mm"
    return pd.DataFrame(rows)


def _summary(
    case_id: str,
    wall_values: np.ndarray,
    outer_surface_values: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    software_version: str,
) -> pd.DataFrame:
    voxel_volume = float(np.prod(spacing_xyz))
    values = np.asarray(wall_values, dtype=float)
    outer_values = np.asarray(outer_surface_values, dtype=float)
    rows: list[dict[str, object]] = [
        _row(case_id, "wall_thickness", "wall_voxel_count", int(values.size), "voxels", software_version),
        _row(case_id, "wall_thickness", "wall_volume_mm3", float(values.size * voxel_volume), "mm3", software_version),
    ]
    for name, data in (("wall", values), ("outer_surface", outer_values)):
        if data.size == 0:
            stats = {
                "mean_mm": np.nan,
                "median_mm": np.nan,
                "std_mm": np.nan,
                "min_mm": np.nan,
                "p05_mm": np.nan,
                "p25_mm": np.nan,
                "p75_mm": np.nan,
                "p95_mm": np.nan,
                "max_mm": np.nan,
            }
        else:
            stats = {
                "mean_mm": float(np.mean(data)),
                "median_mm": float(np.median(data)),
                "std_mm": float(np.std(data)),
                "min_mm": float(np.min(data)),
                "p05_mm": float(np.percentile(data, 5)),
                "p25_mm": float(np.percentile(data, 25)),
                "p75_mm": float(np.percentile(data, 75)),
                "p95_mm": float(np.percentile(data, 95)),
                "max_mm": float(np.max(data)),
            }
        for metric, value in stats.items():
            rows.append(_row(case_id, "wall_thickness", f"{name}_{metric}", value, "mm", software_version))
    return pd.DataFrame(rows)


def _row(
    case_id: str,
    feature_group: str,
    feature_name: str,
    feature_value: object,
    units: str,
    software_version: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "region": "aortic_wall",
        "feature_group": feature_group,
        "feature_name": feature_name,
        "feature_value": feature_value,
        "units": units,
        "threshold_if_applicable": "",
        "mask_name": "wall_mask",
        "software_version": software_version,
    }


def _spacing_zyx(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
