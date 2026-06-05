"""Local aortic wall-sector morphology features.

These features are candidate-generation signals for review. They are not plaque
segmentations and do not classify tissue as plaque versus non-plaque.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .features import feature_row


@dataclass(frozen=True)
class WallMorphologyResult:
    """Sector table, case-level summary rows, and candidate masks."""

    sector_features: pd.DataFrame
    parcel_features: pd.DataFrame
    summary_features: pd.DataFrame
    candidate_boundary_mask: np.ndarray
    candidate_neighborhood_mask: np.ndarray
    inward_candidate_boundary_mask: np.ndarray
    inward_candidate_neighborhood_mask: np.ndarray
    outward_candidate_boundary_mask: np.ndarray
    outward_candidate_neighborhood_mask: np.ndarray
    boundary_direction_labelmap: np.ndarray
    direction_labelmap: np.ndarray
    inward_focal_mask: np.ndarray
    outward_focal_mask: np.ndarray
    focal_direction_labelmap: np.ndarray
    parcel_labelmap: np.ndarray
    inward_parcel_labelmap: np.ndarray
    outward_parcel_labelmap: np.ndarray


def extract_wall_morphology(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    min_slice_voxels: int = 80,
    axial_step_mm: float = 2.0,
    angular_bins: int = 24,
    smoothing_bins: int = 5,
    candidate_depth_mm: float = 4.0,
    candidate_neighborhood_mm: float = 2.0,
    candidate_focal_radius_mm: float = 1.0,
    wall_parcel_radius_mm: float = 1.25,
    max_components_per_slice: int = 4,
    software_version: str = __version__,
) -> WallMorphologyResult:
    """Compute local wall-sector morphology and 4 mm scale candidate flags.

    Positive radial residuals are labelled outward/crater-like because the local
    contour extends beyond its smoothed expectation. Positive convexity defects
    are labelled inward/protrusion-like because the local contour sits inside the
    convex local lumen envelope.
    """
    binary = np.asarray(mask, dtype=bool)
    candidate_boundary = np.zeros(binary.shape, dtype=bool)
    inward_candidate_boundary = np.zeros(binary.shape, dtype=bool)
    outward_candidate_boundary = np.zeros(binary.shape, dtype=bool)
    inward_seed = np.zeros(binary.shape, dtype=bool)
    outward_seed = np.zeros(binary.shape, dtype=bool)
    rows: list[dict[str, object]] = []

    z_step = max(1, int(round(axial_step_mm / spacing_xyz[2])))
    occupied_slices = np.where(binary.any(axis=(1, 2)))[0]
    sampled_slices = occupied_slices[::z_step]

    for z in sampled_slices:
        labels, n_labels = _label_2d(binary[z])
        component_labels = _component_labels_by_size(labels, n_labels, max_components_per_slice)
        component_count = len(component_labels)
        for component_rank, component_label in enumerate(component_labels):
            component = labels == component_label
            voxel_count = int(component.sum())
            if voxel_count < min_slice_voxels:
                continue
            (
                component_rows,
                component_candidate_boundary,
                component_inward_boundary,
                component_outward_boundary,
                component_inward_seed,
                component_outward_seed,
            ) = _component_wall_sectors(
                component=component,
                z=int(z),
                component_label=int(component_label),
                component_rank=int(component_rank),
                component_count=component_count,
                spacing_xyz=spacing_xyz,
                case_id=case_id,
                angular_bins=angular_bins,
                smoothing_bins=smoothing_bins,
                candidate_depth_mm=candidate_depth_mm,
            )
            rows.extend(component_rows)
            candidate_boundary[z] |= component_candidate_boundary
            inward_candidate_boundary[z] |= component_inward_boundary
            outward_candidate_boundary[z] |= component_outward_boundary
            inward_seed[z] |= component_inward_seed
            outward_seed[z] |= component_outward_seed

    sector_frame = pd.DataFrame(rows)
    if sector_frame.empty:
        empty_mask = np.zeros_like(candidate_boundary, dtype=bool)
        empty_summary = _summary_feature_rows(
            sector_frame,
            candidate_boundary,
            inward_candidate_boundary,
            outward_candidate_boundary,
            empty_mask,
            empty_mask,
            spacing_xyz,
            case_id,
            software_version,
        )
        return WallMorphologyResult(
            sector_features=sector_frame,
            parcel_features=pd.DataFrame(),
            summary_features=empty_summary,
            candidate_boundary_mask=candidate_boundary,
            candidate_neighborhood_mask=empty_mask,
            inward_candidate_boundary_mask=inward_candidate_boundary,
            inward_candidate_neighborhood_mask=empty_mask,
            outward_candidate_boundary_mask=outward_candidate_boundary,
            outward_candidate_neighborhood_mask=empty_mask,
            boundary_direction_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint8),
            direction_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint8),
            inward_focal_mask=empty_mask,
            outward_focal_mask=empty_mask,
            focal_direction_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint8),
            parcel_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint16),
            inward_parcel_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint16),
            outward_parcel_labelmap=np.zeros_like(candidate_boundary, dtype=np.uint16),
        )

    candidate_neighborhood = _candidate_neighborhood(candidate_boundary, candidate_neighborhood_mm, spacing_xyz)
    inward_neighborhood = _candidate_neighborhood(inward_candidate_boundary, candidate_neighborhood_mm, spacing_xyz)
    outward_neighborhood = _candidate_neighborhood(outward_candidate_boundary, candidate_neighborhood_mm, spacing_xyz)
    inward_focal = _candidate_neighborhood(inward_seed, candidate_focal_radius_mm, spacing_xyz)
    outward_focal = _candidate_neighborhood(outward_seed, candidate_focal_radius_mm, spacing_xyz)
    parcel_frame, parcel_labelmap, inward_parcels, outward_parcels = _build_wall_parcels(
        sector_frame=sector_frame,
        inward_boundary=inward_candidate_boundary,
        outward_boundary=outward_candidate_boundary,
        spacing_xyz=spacing_xyz,
        parcel_radius_mm=wall_parcel_radius_mm,
        case_id=case_id,
    )
    summary = _summary_feature_rows(
        sector_frame,
        candidate_neighborhood,
        inward_neighborhood,
        outward_neighborhood,
        inward_focal,
        outward_focal,
        spacing_xyz,
        case_id,
        software_version,
    )
    return WallMorphologyResult(
        sector_features=sector_frame,
        parcel_features=parcel_frame,
        summary_features=summary,
        candidate_boundary_mask=candidate_boundary,
        candidate_neighborhood_mask=candidate_neighborhood,
        inward_candidate_boundary_mask=inward_candidate_boundary,
        inward_candidate_neighborhood_mask=inward_neighborhood,
        outward_candidate_boundary_mask=outward_candidate_boundary,
        outward_candidate_neighborhood_mask=outward_neighborhood,
        boundary_direction_labelmap=_direction_labelmap(inward_candidate_boundary, outward_candidate_boundary),
        direction_labelmap=_direction_labelmap(inward_neighborhood, outward_neighborhood),
        inward_focal_mask=inward_focal,
        outward_focal_mask=outward_focal,
        focal_direction_labelmap=_direction_labelmap(inward_focal, outward_focal),
        parcel_labelmap=parcel_labelmap,
        inward_parcel_labelmap=inward_parcels,
        outward_parcel_labelmap=outward_parcels,
    )


def _direction_labelmap(inward_mask: np.ndarray, outward_mask: np.ndarray) -> np.ndarray:
    """Create a viewable label map: 1 inward, 2 outward, 3 overlap."""
    inward = np.asarray(inward_mask, dtype=bool)
    outward = np.asarray(outward_mask, dtype=bool)
    labels = np.zeros(inward.shape, dtype=np.uint8)
    labels[inward] = 1
    labels[outward] = 2
    labels[inward & outward] = 3
    return labels


def _build_wall_parcels(
    sector_frame: pd.DataFrame,
    inward_boundary: np.ndarray,
    outward_boundary: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    parcel_radius_mm: float,
    case_id: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build unique small parcel labels restricted to candidate wall voxels."""
    parcel_labelmap = np.zeros(inward_boundary.shape, dtype=np.uint16)
    inward_labelmap = np.zeros(inward_boundary.shape, dtype=np.uint16)
    outward_labelmap = np.zeros(inward_boundary.shape, dtype=np.uint16)
    rows: list[dict[str, object]] = []
    parcel_id = 1

    candidates = sector_frame[sector_frame["wall_morphology_candidate"].astype(bool)].copy()
    if candidates.empty:
        return pd.DataFrame(), parcel_labelmap, inward_labelmap, outward_labelmap

    candidates = candidates.sort_values(["slice_index_z", "component_rank_in_slice", "wall_angle_bin"])
    for row in candidates.itertuples(index=False):
        if bool(row.inward_protrusion_like_candidate):
            parcel_id = _add_wall_parcel(
                row=row,
                direction="inward",
                allowed_boundary=inward_boundary,
                direction_labelmap=inward_labelmap,
                combined_labelmap=parcel_labelmap,
                spacing_xyz=spacing_xyz,
                parcel_radius_mm=parcel_radius_mm,
                parcel_id=parcel_id,
                case_id=case_id,
                rows=rows,
            )
        if bool(row.outward_crater_like_candidate):
            parcel_id = _add_wall_parcel(
                row=row,
                direction="outward",
                allowed_boundary=outward_boundary,
                direction_labelmap=outward_labelmap,
                combined_labelmap=parcel_labelmap,
                spacing_xyz=spacing_xyz,
                parcel_radius_mm=parcel_radius_mm,
                parcel_id=parcel_id,
                case_id=case_id,
                rows=rows,
            )

    return pd.DataFrame(rows), parcel_labelmap, inward_labelmap, outward_labelmap


def _add_wall_parcel(
    row: object,
    direction: str,
    allowed_boundary: np.ndarray,
    direction_labelmap: np.ndarray,
    combined_labelmap: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    parcel_radius_mm: float,
    parcel_id: int,
    case_id: str,
    rows: list[dict[str, object]],
) -> int:
    z = int(row.slice_index_z)
    if z < 0 or z >= allowed_boundary.shape[0]:
        return parcel_id

    ys, xs = np.where(allowed_boundary[z])
    if ys.size == 0:
        return parcel_id

    center_y = float(row.center_y_voxel)
    center_x = float(row.center_x_voxel)
    distances = np.sqrt(((xs - center_x) * spacing_xyz[0]) ** 2 + ((ys - center_y) * spacing_xyz[1]) ** 2)
    selected = distances <= parcel_radius_mm
    if not np.any(selected):
        selected[np.argmin(distances)] = True

    selected_y = ys[selected]
    selected_x = xs[selected]
    unassigned = direction_labelmap[z, selected_y, selected_x] == 0
    selected_y = selected_y[unassigned]
    selected_x = selected_x[unassigned]
    if selected_y.size == 0:
        return parcel_id

    direction_labelmap[z, selected_y, selected_x] = parcel_id
    combined_unassigned = combined_labelmap[z, selected_y, selected_x] == 0
    combined_labelmap[z, selected_y[combined_unassigned], selected_x[combined_unassigned]] = parcel_id

    voxel_count = int(selected_y.size)
    rows.append(
        {
            "case_id": case_id,
            "wall_parcel_id": int(parcel_id),
            "wall_parcel_name": f"{case_id}_wall_parcel_{parcel_id:04d}",
            "direction": direction,
            "slice_index_z": int(row.slice_index_z),
            "component_rank_in_slice": int(row.component_rank_in_slice),
            "wall_angle_bin": int(row.wall_angle_bin),
            "center_x_voxel": int(round(float(np.median(selected_x)))),
            "center_y_voxel": int(round(float(np.median(selected_y)))),
            "center_z_voxel": int(z),
            "center_x_mm": float(np.median(selected_x) * spacing_xyz[0]),
            "center_y_mm": float(np.median(selected_y) * spacing_xyz[1]),
            "center_z_mm": float(z * spacing_xyz[2]),
            "parcel_wall_voxel_count": voxel_count,
            "parcel_wall_footprint_mm2": float(voxel_count * spacing_xyz[0] * spacing_xyz[1]),
            "parcel_radius_mm": float(parcel_radius_mm),
            "sector_inward_residual_mm": float(row.sector_inward_residual_mm),
            "sector_outward_residual_mm": float(row.sector_outward_residual_mm),
            "sector_abs_wall_deviation_mm": float(row.sector_abs_wall_deviation_mm),
            "candidate_depth_threshold_mm": float(row.candidate_depth_threshold_mm),
            "parcel_interpretation": "wall_surface_patch_for_review_not_plaque_segmentation",
        }
    )
    return parcel_id + 1


def _component_wall_sectors(
    component: np.ndarray,
    z: int,
    component_label: int,
    component_rank: int,
    component_count: int,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    angular_bins: int,
    smoothing_bins: int,
    candidate_depth_mm: float,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(component)
    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())
    boundary = component ^ _binary_erosion_2d(component)
    bys, bxs = np.where(boundary)
    if bys.size < max(8, angular_bins // 2):
        empty = np.zeros_like(component, dtype=bool)
        return [], empty, empty, empty, empty, empty

    angles = _angles_for_points(bys, bxs, centroid_y, centroid_x, spacing_xyz)
    bins = np.floor(angles / (2.0 * np.pi / angular_bins)).astype(int)
    radii = _radii_for_points(bys, bxs, centroid_y, centroid_x, spacing_xyz)
    radial_profile = _radial_profile(radii, bins, angular_bins)
    if not np.isfinite(radial_profile).any():
        empty = np.zeros_like(component, dtype=bool)
        return [], empty, empty, empty, empty, empty
    radial_profile = _fill_circular_profile(radial_profile)
    smoothed_profile = _smooth_circular(radial_profile, smoothing_bins)
    outward_residual = radial_profile - smoothed_profile

    hull = _convex_hull(component)
    hull_boundary = hull ^ _binary_erosion_2d(hull)
    hys, hxs = np.where(hull_boundary)
    hull_angles = _angles_for_points(hys, hxs, centroid_y, centroid_x, spacing_xyz)
    hull_bins = np.floor(hull_angles / (2.0 * np.pi / angular_bins)).astype(int)
    hull_profile = _radial_profile(
        _radii_for_points(hys, hxs, centroid_y, centroid_x, spacing_xyz),
        hull_bins,
        angular_bins,
    )
    hull_profile = _fill_circular_profile(hull_profile)
    inward_residual = np.maximum(smoothed_profile - radial_profile, 0.0)
    convex_hull_defect = np.maximum(hull_profile - radial_profile, 0.0)

    area_mm2 = float(component.sum() * spacing_xyz[0] * spacing_xyz[1])
    perimeter_mm = _contour_perimeter_mm(component, spacing_xyz)
    smoothed_perimeter_mm = _radial_polygon_perimeter_mm(smoothed_profile)
    circularity = _safe_divide(4.0 * np.pi * area_mm2, perimeter_mm**2)
    compactness = _safe_divide(perimeter_mm**2, 4.0 * np.pi * area_mm2)
    malinowska = _safe_divide(perimeter_mm, 2.0 * np.sqrt(np.pi * area_mm2)) - 1.0
    hull_area_mm2 = float(hull.sum() * spacing_xyz[0] * spacing_xyz[1])
    solidity = _safe_divide(area_mm2, hull_area_mm2)
    convexity_defect_area_fraction = _safe_divide(hull_area_mm2 - area_mm2, hull_area_mm2)
    roughness_ratio = _safe_divide(perimeter_mm, smoothed_perimeter_mm)

    candidate_boundary = np.zeros_like(component, dtype=bool)
    inward_boundary = np.zeros_like(component, dtype=bool)
    outward_boundary = np.zeros_like(component, dtype=bool)
    inward_seed = np.zeros_like(component, dtype=bool)
    outward_seed = np.zeros_like(component, dtype=bool)
    rows: list[dict[str, object]] = []
    for angle_bin in range(angular_bins):
        in_bin = bins == angle_bin
        if not np.any(in_bin):
            continue
        sector_radius = radii[in_bin]
        center_y = int(round(float(np.median(bys[in_bin]))))
        center_x = int(round(float(np.median(bxs[in_bin]))))
        inward_depth = float(inward_residual[angle_bin])
        hull_defect_depth = float(convex_hull_defect[angle_bin])
        outward_depth = float(max(outward_residual[angle_bin], 0.0))
        inward_candidate = bool(inward_depth >= candidate_depth_mm)
        outward_candidate = bool(outward_depth >= candidate_depth_mm)
        candidate = inward_candidate or outward_candidate
        if candidate:
            candidate_boundary[bys[in_bin], bxs[in_bin]] = True
        if inward_candidate:
            inward_boundary[bys[in_bin], bxs[in_bin]] = True
            inward_index = np.flatnonzero(in_bin)[int(np.argmin(sector_radius))]
            inward_seed[bys[inward_index], bxs[inward_index]] = True
        if outward_candidate:
            outward_boundary[bys[in_bin], bxs[in_bin]] = True
            outward_index = np.flatnonzero(in_bin)[int(np.argmax(sector_radius))]
            outward_seed[bys[outward_index], bxs[outward_index]] = True

        rows.append(
            {
                "case_id": case_id,
                "slice_index_z": int(z),
                "component_rank_in_slice": int(component_rank),
                "component_count_in_slice": int(component_count),
                "component_label_2d": int(component_label),
                "wall_angle_bin": int(angle_bin),
                "sector_start_degrees": float(angle_bin * 360.0 / angular_bins),
                "sector_end_degrees": float((angle_bin + 1) * 360.0 / angular_bins),
                "center_x_voxel": int(center_x),
                "center_y_voxel": int(center_y),
                "center_z_voxel": int(z),
                "center_x_mm": float(center_x * spacing_xyz[0]),
                "center_y_mm": float(center_y * spacing_xyz[1]),
                "center_z_mm": float(z * spacing_xyz[2]),
                "component_area_mm2": area_mm2,
                "component_perimeter_mm": perimeter_mm,
                "component_smoothed_perimeter_mm": smoothed_perimeter_mm,
                "component_circularity": float(np.clip(circularity, 0.0, 1.0)) if np.isfinite(circularity) else np.nan,
                "component_compactness": compactness,
                "component_malinowska": malinowska,
                "component_solidity": solidity,
                "component_convexity_defect_area_fraction": convexity_defect_area_fraction,
                "component_roughness_ratio": roughness_ratio,
                "sector_boundary_voxel_count": int(in_bin.sum()),
                "sector_radius_mean_mm": float(np.mean(sector_radius)),
                "sector_radius_sd_mm": float(np.std(sector_radius)),
                "sector_radius_cv": _safe_divide(float(np.std(sector_radius)), float(np.mean(sector_radius))),
                "sector_radius_profile_mm": float(radial_profile[angle_bin]),
                "sector_expected_radius_mm": float(smoothed_profile[angle_bin]),
                "sector_outward_residual_mm": outward_depth,
                "sector_inward_residual_mm": inward_depth,
                "sector_convex_hull_defect_mm": hull_defect_depth,
                "sector_abs_wall_deviation_mm": float(max(inward_depth, outward_depth)),
                "inward_protrusion_like_candidate": inward_candidate,
                "outward_crater_like_candidate": outward_candidate,
                "wall_morphology_candidate": candidate,
                "candidate_depth_threshold_mm": float(candidate_depth_mm),
                "morphology_method": "axial_wall_sector_radial_residual_v1",
                "morphology_interpretation": "review_candidate_not_plaque_segmentation_or_diagnosis",
            }
        )
    return rows, candidate_boundary, inward_boundary, outward_boundary, inward_seed, outward_seed


def _summary_feature_rows(
    sector_frame: pd.DataFrame,
    candidate_neighborhood: np.ndarray,
    inward_neighborhood: np.ndarray,
    outward_neighborhood: np.ndarray,
    inward_focal: np.ndarray,
    outward_focal: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    voxel_volume = float(np.prod(spacing_xyz))
    if sector_frame.empty:
        values: dict[str, object] = {
            "sector_count": 0,
            "wall_morphology_candidate_sector_count": 0,
            "inward_protrusion_like_candidate_sector_count": 0,
            "outward_crater_like_candidate_sector_count": 0,
            "candidate_neighborhood_volume_mm3": 0.0,
            "inward_candidate_neighborhood_volume_mm3": 0.0,
            "outward_candidate_neighborhood_volume_mm3": 0.0,
            "inward_focal_volume_mm3": 0.0,
            "outward_focal_volume_mm3": 0.0,
        }
    else:
        values = {
            "sector_count": int(len(sector_frame)),
            "wall_morphology_candidate_sector_count": int(sector_frame["wall_morphology_candidate"].sum()),
            "inward_protrusion_like_candidate_sector_count": int(
                sector_frame["inward_protrusion_like_candidate"].sum()
            ),
            "outward_crater_like_candidate_sector_count": int(sector_frame["outward_crater_like_candidate"].sum()),
            "max_inward_residual_mm": float(sector_frame["sector_inward_residual_mm"].max()),
            "max_convex_hull_defect_mm": float(sector_frame["sector_convex_hull_defect_mm"].max()),
            "max_outward_residual_mm": float(sector_frame["sector_outward_residual_mm"].max()),
            "max_abs_wall_deviation_mm": float(sector_frame["sector_abs_wall_deviation_mm"].max()),
            "mean_component_malinowska": float(sector_frame["component_malinowska"].mean()),
            "max_component_malinowska": float(sector_frame["component_malinowska"].max()),
            "mean_component_circularity": float(sector_frame["component_circularity"].mean()),
            "max_component_roughness_ratio": float(sector_frame["component_roughness_ratio"].max()),
            "max_sector_radius_cv": float(sector_frame["sector_radius_cv"].max()),
            "candidate_neighborhood_volume_mm3": float(candidate_neighborhood.sum() * voxel_volume),
            "inward_candidate_neighborhood_volume_mm3": float(inward_neighborhood.sum() * voxel_volume),
            "outward_candidate_neighborhood_volume_mm3": float(outward_neighborhood.sum() * voxel_volume),
            "inward_focal_volume_mm3": float(inward_focal.sum() * voxel_volume),
            "outward_focal_volume_mm3": float(outward_focal.sum() * voxel_volume),
        }

    rows = [
        feature_row(
            case_id=case_id,
            region="wall_morphology",
            feature_group="wall_morphology_candidate_review",
            feature_name=name,
            feature_value=value,
            units=_summary_units(name),
            mask_name="wall_morphology_candidate_2mm",
            software_version=software_version,
        )
        for name, value in values.items()
    ]
    return pd.DataFrame(rows)


def _summary_units(name: str) -> str:
    if name.endswith("_mm"):
        return "mm"
    if name.endswith("_mm3"):
        return "mm3"
    if name.endswith("_count"):
        return "count"
    return ""


def _candidate_neighborhood(
    candidate_boundary: np.ndarray,
    outer_mm: float,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    if not candidate_boundary.any():
        return np.zeros_like(candidate_boundary, dtype=bool)
    try:
        from scipy import ndimage as ndi

        distance = ndi.distance_transform_edt(
            ~candidate_boundary,
            sampling=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]),
        )
        return distance <= outer_mm
    except Exception:
        return candidate_boundary.copy()


def _angles_for_points(
    ys: np.ndarray,
    xs: np.ndarray,
    centroid_y: float,
    centroid_x: float,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    return (np.arctan2((ys - centroid_y) * spacing_xyz[1], (xs - centroid_x) * spacing_xyz[0]) + 2.0 * np.pi) % (
        2.0 * np.pi
    )


def _radii_for_points(
    ys: np.ndarray,
    xs: np.ndarray,
    centroid_y: float,
    centroid_x: float,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    return np.sqrt(((xs - centroid_x) * spacing_xyz[0]) ** 2 + ((ys - centroid_y) * spacing_xyz[1]) ** 2)


def _radial_profile(radii: np.ndarray, bins: np.ndarray, angular_bins: int) -> np.ndarray:
    profile = np.full(angular_bins, np.nan, dtype=float)
    for angle_bin in range(angular_bins):
        values = radii[bins == angle_bin]
        if values.size:
            profile[angle_bin] = float(np.percentile(values, 95))
    return profile


def _fill_circular_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    valid = np.flatnonzero(np.isfinite(profile))
    if len(valid) == 0:
        return np.zeros_like(profile)
    if len(valid) == len(profile):
        return profile
    x = np.arange(len(profile))
    extended_x = np.concatenate([valid - len(profile), valid, valid + len(profile)])
    extended_y = np.concatenate([profile[valid], profile[valid], profile[valid]])
    return np.interp(x, extended_x, extended_y)


def _smooth_circular(profile: np.ndarray, smoothing_bins: int) -> np.ndarray:
    window = max(1, int(smoothing_bins))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return profile.astype(float)
    pad = window // 2
    padded = np.pad(profile.astype(float), pad, mode="wrap")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _radial_polygon_perimeter_mm(profile: np.ndarray) -> float:
    if len(profile) < 3:
        return 0.0
    angles = (np.arange(len(profile)) + 0.5) * 2.0 * np.pi / len(profile)
    xy = np.column_stack([profile * np.cos(angles), profile * np.sin(angles)])
    closed = np.vstack([xy, xy[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _contour_perimeter_mm(component: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    try:
        from skimage.measure import find_contours

        contours = find_contours(component.astype(float), 0.5)
        perimeter = 0.0
        for contour in contours:
            if len(contour) < 2:
                continue
            xy = np.column_stack([contour[:, 1] * spacing_xyz[0], contour[:, 0] * spacing_xyz[1]])
            closed = np.vstack([xy, xy[0]])
            perimeter += float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
        return perimeter
    except Exception:
        boundary = component ^ _binary_erosion_2d(component)
        horizontal_edges = np.logical_xor(boundary[:, 1:], boundary[:, :-1]).sum() * spacing_xyz[1]
        vertical_edges = np.logical_xor(boundary[1:, :], boundary[:-1, :]).sum() * spacing_xyz[0]
        return float(horizontal_edges + vertical_edges)


def _convex_hull(component: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import convex_hull_image

        return convex_hull_image(component)
    except Exception:
        ys, xs = np.where(component)
        out = np.zeros_like(component, dtype=bool)
        out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        return out


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator)


def _label_2d(binary: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage as ndi

        labels, n_labels = ndi.label(binary)
        return labels.astype(np.int32), int(n_labels)
    except Exception:
        return _label_2d_fallback(binary)


def _binary_erosion_2d(binary: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi

        return ndi.binary_erosion(binary)
    except Exception:
        padded = np.pad(binary, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(binary, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
        return eroded


def _label_2d_fallback(binary: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(binary.shape, dtype=np.int32)
    current = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def _component_labels_by_size(labels: np.ndarray, n_labels: int, max_components: int) -> list[int]:
    if n_labels == 0:
        return []
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    ordered = np.argsort(counts)[::-1]
    return [int(label) for label in ordered[:max_components] if counts[label] > 0]
