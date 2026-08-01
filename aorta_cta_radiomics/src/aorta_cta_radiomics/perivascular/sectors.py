"""Reproducible circumferential sectors defined by vessel frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._spatial import indices_zyx_to_physical_xyz, validate_spacing, voxel_volume_mm3
from .composition import TISSUE_NAMES, TissueLabel
from .coordinates import VesselCoordinateSystem


SUPPORTED_SECTOR_COUNTS = (8, 16, 32)


def sector_index_from_angle(angle_rad: np.ndarray | float, number_of_sectors: int) -> np.ndarray:
    if number_of_sectors not in SUPPORTED_SECTOR_COUNTS:
        raise ValueError(f"number_of_sectors must be one of {SUPPORTED_SECTOR_COUNTS}.")
    angle = np.mod(np.asarray(angle_rad, dtype=float), 2.0 * np.pi)
    width = 2.0 * np.pi / number_of_sectors
    return np.floor(angle / width).astype(np.int16) + 1


@dataclass(frozen=True)
class SectorAssignment:
    labels: np.ndarray
    path_mm: np.ndarray
    radial_mm: np.ndarray
    angle_rad: np.ndarray
    number_of_sectors: int
    orientation_reference: str
    absolute_orientation_available: bool
    vessel_id: str


def assign_angular_sectors(
    roi_mask: np.ndarray,
    coordinate_system: VesselCoordinateSystem,
    spacing_xyz: Sequence[float],
    *,
    number_of_sectors: int = 8,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | np.ndarray | None = None,
) -> SectorAssignment:
    """Assign every ROI voxel to a frame-stable angular sector."""
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.ndim != 3:
        raise ValueError("roi_mask must be three-dimensional.")
    spacing = validate_spacing(spacing_xyz)
    # Validate count before doing any dense work.
    sector_index_from_angle(np.asarray([0.0]), number_of_sectors)
    labels = np.zeros(roi.shape, dtype=np.int16)
    path_map = np.full(roi.shape, np.nan, dtype=np.float32)
    radial_map = np.full(roi.shape, np.nan, dtype=np.float32)
    angle_map = np.full(roi.shape, np.nan, dtype=np.float32)
    indices = np.argwhere(roi)
    if indices.size:
        physical = indices_zyx_to_physical_xyz(indices, spacing, origin_xyz, direction)
        coordinates = coordinate_system.physical_to_vessel(physical)
        sectors = sector_index_from_angle(coordinates.angle_rad, number_of_sectors)
        target = tuple(indices.T)
        labels[target] = sectors
        path_map[target] = coordinates.path_mm.astype(np.float32)
        radial_map[target] = coordinates.radial_mm.astype(np.float32)
        angle_map[target] = coordinates.angle_rad.astype(np.float32)
    absolute = coordinate_system.anatomical_orientation is not None
    reference = (
        str(coordinate_system.anatomical_orientation)
        if absolute
        else "relative_rotation_minimizing_frame_normal_1"
    )
    return SectorAssignment(
        labels=labels,
        path_mm=path_map,
        radial_mm=radial_map,
        angle_rad=angle_map,
        number_of_sectors=number_of_sectors,
        orientation_reference=reference,
        absolute_orientation_available=absolute,
        vessel_id=coordinate_system.vessel_id,
    )


def _statistics(values: np.ndarray, prefix: str) -> dict[str, object]:
    if values.size == 0:
        return {
            f"{prefix}_mean_hu": float("nan"),
            f"{prefix}_median_hu": float("nan"),
            f"{prefix}_std_hu": float("nan"),
            f"{prefix}_p10_hu": float("nan"),
            f"{prefix}_p25_hu": float("nan"),
            f"{prefix}_p75_hu": float("nan"),
            f"{prefix}_p90_hu": float("nan"),
        }
    percentiles = np.percentile(values, (10, 25, 75, 90))
    return {
        f"{prefix}_mean_hu": float(np.mean(values)),
        f"{prefix}_median_hu": float(np.median(values)),
        f"{prefix}_std_hu": float(np.std(values)),
        f"{prefix}_p10_hu": float(percentiles[0]),
        f"{prefix}_p25_hu": float(percentiles[1]),
        f"{prefix}_p75_hu": float(percentiles[2]),
        f"{prefix}_p90_hu": float(percentiles[3]),
    }


def summarize_sectors(
    cta_hu: np.ndarray,
    tissue_labels: np.ndarray,
    assignment: SectorAssignment,
    spacing_xyz: Sequence[float],
    *,
    radial_bands: Mapping[str, np.ndarray] | None = None,
    valid_mask: np.ndarray | None = None,
    attenuation_histogram_edges_hu: Sequence[float] = (-200, -100, -50, 0, 100, 200, 400, 800),
) -> list[dict[str, object]]:
    """Create one composition record per radial band and angular sector."""
    image = np.asarray(cta_hu, dtype=float)
    tissues = np.asarray(tissue_labels)
    sectors = np.asarray(assignment.labels)
    if image.shape != tissues.shape or image.shape != sectors.shape:
        raise ValueError("cta_hu, tissue_labels, and sector labels must share a shape.")
    valid = sectors > 0 if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != image.shape:
        raise ValueError("valid_mask must share the image shape.")
    bands = {"all": sectors > 0} if radial_bands is None else {
        str(name): np.asarray(mask, dtype=bool) for name, mask in radial_bands.items()
    }
    if any(mask.shape != image.shape for mask in bands.values()):
        raise ValueError("Every radial band mask must share the image shape.")
    spacing = validate_spacing(spacing_xyz)
    voxel_volume = voxel_volume_mm3(spacing)
    histogram_edges = np.asarray(attenuation_histogram_edges_hu, dtype=float)
    if histogram_edges.ndim != 1 or len(histogram_edges) < 2 or np.any(np.diff(histogram_edges) <= 0):
        raise ValueError("attenuation_histogram_edges_hu must be strictly increasing.")
    # Only voxels assigned to a vessel sector can contribute to any output.
    # Compact this sparse domain once so each band/sector/tissue summary does
    # not rescan the full CTA volume.
    domain = sectors > 0
    compact_image = image[domain]
    compact_tissues = tissues[domain]
    compact_sectors = sectors[domain]
    compact_valid = valid[domain]
    compact_bands = {name: mask[domain] for name, mask in bands.items()}
    records: list[dict[str, object]] = []
    for band_name, band_mask in compact_bands.items():
        for sector in range(1, assignment.number_of_sectors + 1):
            total = band_mask & (compact_sectors == sector)
            assessed = total & compact_valid
            count = int(assessed.sum())
            record: dict[str, object] = {
                "vessel_id": assignment.vessel_id,
                "radial_band": band_name,
                "sector": sector,
                "number_of_sectors": assignment.number_of_sectors,
                "orientation_reference": assignment.orientation_reference,
                "absolute_orientation_available": assignment.absolute_orientation_available,
                "total_sector_voxels": int(total.sum()),
                "total_sector_volume_mm3": float(total.sum() * voxel_volume),
                "valid_voxels": count,
                "valid_tissue_volume_mm3": float(count * voxel_volume),
                "qc_status": "ok" if count else "missing_sector",
            }
            for label in (
                TissueLabel.ADIPOSE,
                TissueLabel.SKELETAL_MUSCLE,
                TissueLabel.VEIN,
                TissueLabel.BONE,
                TissueLabel.THYROID_GLAND,
                TissueLabel.OTHER_SOFT_TISSUE,
                TissueLabel.HIGH_DENSITY,
                TissueLabel.UNCERTAIN,
            ):
                name = TISSUE_NAMES[int(label)]
                label_count = int((assessed & (compact_tissues == int(label))).sum())
                record[f"{name}_voxels"] = label_count
                record[f"{name}_volume_mm3"] = float(label_count * voxel_volume)
                record[f"{name}_fraction"] = float(label_count / count) if count else float("nan")
            values = compact_image[assessed & np.isfinite(compact_image)]
            record.update(_statistics(values, "cta"))
            fat_values = compact_image[
                assessed
                & (compact_tissues == int(TissueLabel.ADIPOSE))
                & np.isfinite(compact_image)
            ]
            record.update(_statistics(fat_values, "fat"))
            histogram, _ = np.histogram(values, bins=histogram_edges)
            record["attenuation_histogram_edges_hu"] = histogram_edges.tolist()
            record["attenuation_histogram_counts"] = histogram.astype(int).tolist()
            records.append(record)
    return records


__all__ = [
    "SUPPORTED_SECTOR_COUNTS",
    "SectorAssignment",
    "assign_angular_sectors",
    "sector_index_from_angle",
    "summarize_sectors",
]
