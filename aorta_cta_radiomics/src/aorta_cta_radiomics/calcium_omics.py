"""Calcium-omics features for aortic calcium masks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import __version__
from .calcification import density_factor_for_hu
from .features import feature_row


def summarize_calcium_omics(
    image: np.ndarray,
    calcium_mask: np.ndarray,
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    mask_name: str,
    threshold_label: str,
    centerline_points: pd.DataFrame | None = None,
    segment_labels: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
    software_version: str = __version__,
    angle_bins: int = 36,
) -> pd.DataFrame:
    """Create regression-ready calcium burden, mass, lesion, and wall-location features.

    The mass features are HU-volume proxies unless a future calibration phantom
    is supplied. The wall-location features are image-plane approximations and
    are intended for QC/research modeling, not diagnostic classification.
    """
    image_array = np.asarray(image)
    calcium = np.asarray(calcium_mask, dtype=bool)
    aorta = np.asarray(aorta_mask, dtype=bool)
    if image_array.shape != calcium.shape or image_array.shape != aorta.shape:
        raise ValueError("image, calcium_mask, and aorta_mask must have the same shape.")

    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    values = image_array[calcium]
    calcium_voxels = int(calcium.sum())
    calcium_volume_mm3 = float(calcium_voxels * voxel_volume_mm3)
    mass_proxy = float(values.sum() * voxel_volume_mm3) if values.size else 0.0
    dense_mask = calcium & (image_array > 1000)
    dense_volume_mm3 = float(dense_mask.sum() * voxel_volume_mm3)
    dense_fraction = _safe_divide(dense_volume_mm3, calcium_volume_mm3)

    component_summary = _component_summary(image_array, calcium, voxel_volume_mm3)
    num_lesions = int(component_summary["num_components"])
    modified_agatston = float(component_summary["modified_agatston"])
    calcium_span_mm = _z_span_mm(calcium, spacing_xyz)
    aortic_length_cm = _aortic_length_cm(aorta, spacing_xyz, centerline_points)
    calcium_span_cm = calcium_span_mm / 10.0 if np.isfinite(calcium_span_mm) else 0.0
    wall_stats = _wall_distribution_stats(calcium, aorta, angle_bins=angle_bins)

    rows = [
        _row(case_id, "aorta", "mass_total", mass_proxy, "HU*mm3", threshold_label, mask_name, software_version),
        _row(case_id, "aorta", "aortic_mass_proxy", mass_proxy, "HU*mm3", threshold_label, mask_name, software_version),
        _row(
            case_id,
            "aorta",
            "aortic_volume_mm3",
            calcium_volume_mm3,
            "mm3",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "aortic_agatston_modified",
            modified_agatston,
            "arbitrary",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(case_id, "aorta", "num_lesions", num_lesions, "components", threshold_label, mask_name, software_version),
        _row(
            case_id,
            "aorta",
            "log1p_num_lesions",
            float(np.log1p(num_lesions)),
            "log_components",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "hu_gt_1000_volume",
            dense_volume_mm3,
            "mm3",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "hu_gt_1000_fraction",
            dense_fraction,
            "fraction",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "top_bottom_distance_mm",
            calcium_span_mm,
            "mm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "aortic_length_cm",
            aortic_length_cm,
            "cm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "calcium_per_cm",
            _safe_divide(calcium_volume_mm3, aortic_length_cm),
            "mm3/cm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "calcium_mass_proxy_per_cm",
            _safe_divide(mass_proxy, aortic_length_cm),
            "HU*mm3/cm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "diffusivity",
            _safe_divide(num_lesions, aortic_length_cm),
            "lesions/cm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "diffusivity_by_calcium_span",
            _safe_divide(num_lesions, calcium_span_cm),
            "lesions/cm",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "circumferential_arc_mean",
            wall_stats["circumferential_arc_mean"],
            "degrees",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "circumferential_arc_max",
            wall_stats["circumferential_arc_max"],
            "degrees",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "anterior_proxy_fraction",
            wall_stats["anterior_proxy_fraction"],
            "fraction",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "posterior_proxy_fraction",
            wall_stats["posterior_proxy_fraction"],
            "fraction",
            threshold_label,
            mask_name,
            software_version,
        ),
        _row(
            case_id,
            "aorta",
            "anterior_posterior_distribution",
            wall_stats["anterior_posterior_distribution"],
            "fraction_difference",
            threshold_label,
            mask_name,
            software_version,
        ),
    ]

    rows.extend(
        _segment_rows(
            image_array=image_array,
            calcium=calcium,
            segment_labels=segment_labels,
            segment_names=segment_names or {},
            voxel_volume_mm3=voxel_volume_mm3,
            case_id=case_id,
            threshold_label=threshold_label,
            mask_name=mask_name,
            software_version=software_version,
        )
    )
    return pd.DataFrame(rows)


def _row(
    case_id: str,
    region: str,
    feature_name: str,
    value: object,
    units: str,
    threshold_label: str,
    mask_name: str,
    software_version: str,
) -> dict[str, object]:
    return feature_row(
        case_id=case_id,
        region=region,
        feature_group="calcium_omics",
        feature_name=feature_name,
        feature_value=value,
        units=units,
        threshold_if_applicable=threshold_label,
        mask_name=mask_name,
        software_version=software_version,
    )


def _component_summary(image: np.ndarray, calcium: np.ndarray, voxel_volume_mm3: float) -> dict[str, float | int]:
    if not calcium.any():
        return {"num_components": 0, "modified_agatston": 0.0}
    try:
        from scipy import ndimage as ndi

        labels, num_components = ndi.label(calcium, structure=np.ones((3, 3, 3), dtype=bool))
        component_ids = np.arange(1, int(num_components) + 1)
        voxel_counts = np.asarray(ndi.sum(calcium, labels, component_ids), dtype=float)
        max_hu = np.asarray(ndi.maximum(image, labels, component_ids), dtype=float)
    except Exception:
        labels = calcium.astype(np.uint8)
        num_components = 1
        voxel_counts = np.asarray([float(calcium.sum())])
        max_hu = np.asarray([float(image[calcium].max())])

    volumes = voxel_counts * voxel_volume_mm3
    modified_agatston = 0.0
    for volume, hu in zip(volumes, max_hu, strict=False):
        modified_agatston += float(volume) * density_factor_for_hu(float(hu))
    return {"num_components": int(num_components), "modified_agatston": float(modified_agatston)}


def _z_span_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    min_z = int(coords[:, 0].min())
    max_z = int(coords[:, 0].max())
    return float((max_z - min_z + 1) * spacing_xyz[2])


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
                length_mm = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                return length_mm / 10.0
    return _z_span_mm(aorta_mask, spacing_xyz) / 10.0


def _wall_distribution_stats(calcium: np.ndarray, aorta: np.ndarray, angle_bins: int) -> dict[str, float]:
    if not calcium.any():
        return {
            "circumferential_arc_mean": 0.0,
            "circumferential_arc_max": 0.0,
            "anterior_proxy_fraction": 0.0,
            "posterior_proxy_fraction": 0.0,
            "anterior_posterior_distribution": 0.0,
        }

    arcs: list[float] = []
    anterior_count = 0
    posterior_count = 0
    bin_count = max(int(angle_bins), 4)
    for z in np.where(calcium.any(axis=(1, 2)))[0]:
        calcium_y, calcium_x = np.where(calcium[z])
        aorta_y, aorta_x = np.where(aorta[z])
        if calcium_y.size == 0 or aorta_y.size == 0:
            continue
        center_y = float(aorta_y.mean())
        center_x = float(aorta_x.mean())
        angles = np.mod(np.arctan2(calcium_y - center_y, calcium_x - center_x), 2.0 * math.pi)
        occupied = np.unique(np.floor(angles / (2.0 * math.pi) * bin_count).astype(int))
        arcs.append(float(occupied.size * 360.0 / bin_count))
        anterior_count += int((calcium_y < center_y).sum())
        posterior_count += int((calcium_y >= center_y).sum())

    total_ap = anterior_count + posterior_count
    anterior_fraction = _safe_divide(anterior_count, total_ap)
    posterior_fraction = _safe_divide(posterior_count, total_ap)
    return {
        "circumferential_arc_mean": float(np.mean(arcs)) if arcs else 0.0,
        "circumferential_arc_max": float(np.max(arcs)) if arcs else 0.0,
        "anterior_proxy_fraction": anterior_fraction,
        "posterior_proxy_fraction": posterior_fraction,
        "anterior_posterior_distribution": float(anterior_fraction - posterior_fraction),
    }


def _segment_rows(
    image_array: np.ndarray,
    calcium: np.ndarray,
    segment_labels: np.ndarray | None,
    segment_names: dict[int, str],
    voxel_volume_mm3: float,
    case_id: str,
    threshold_label: str,
    mask_name: str,
    software_version: str,
) -> list[dict[str, object]]:
    if segment_labels is None:
        return []
    labels = np.asarray(segment_labels)
    if labels.shape != calcium.shape:
        raise ValueError("segment_labels must have the same shape as calcium_mask.")

    rows: list[dict[str, object]] = []
    territories_involved = 0
    nonzero_labels = sorted(int(item) for item in np.unique(labels) if int(item) != 0)
    whole_aorta_only = nonzero_labels == [1] and segment_names.get(1, "") == "whole_aorta"
    for label in nonzero_labels:
        segment_mask = np.ones_like(calcium, dtype=bool) if whole_aorta_only else labels == label
        segment_calcium = calcium & segment_mask
        if segment_calcium.any():
            territories_involved += 1
        segment_name = segment_names.get(label, f"label_{label}")
        region = f"aorta_segment:{segment_name}"
        volume = float(segment_calcium.sum() * voxel_volume_mm3)
        mass = float(image_array[segment_calcium].sum() * voxel_volume_mm3) if segment_calcium.any() else 0.0
        rows.append(_row(case_id, region, "calcium_by_segment", volume, "mm3", threshold_label, mask_name, software_version))
        rows.append(_row(case_id, region, "mass_by_territory", mass, "HU*mm3", threshold_label, mask_name, software_version))

    rows.append(
        _row(
            case_id,
            "aorta",
            "num_territories_involved",
            territories_involved,
            "territories",
            threshold_label,
            mask_name,
            software_version,
        )
    )
    return rows


def _safe_divide(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator) / denominator
