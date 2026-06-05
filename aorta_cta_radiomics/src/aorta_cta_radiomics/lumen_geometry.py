"""Component-wise lumen/mask geometry without irregularity scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def slice_geometry_features(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    min_slice_voxels: int = 20,
    max_branch_link_distance_mm: float = 20.0,
    max_components_per_slice: int = 4,
) -> pd.DataFrame:
    """Compute branch-aware axial fallback geometry features.

    This is intentionally descriptive only. It does not produce an
    irregularity score, adaptive ROI, plaque label, or candidate segmentation.
    """
    binary = np.asarray(mask, dtype=bool)
    area_per_voxel = float(spacing_xyz[0] * spacing_xyz[1])
    rows: list[dict[str, object]] = []

    for z in np.where(binary.any(axis=(1, 2)))[0]:
        labels, n_labels = _label_2d(binary[z])
        components = _component_labels_by_size(labels, n_labels, max_components_per_slice)
        for component_rank, component_label in enumerate(components):
            component = labels == component_label
            voxel_count = int(component.sum())
            if voxel_count < min_slice_voxels:
                continue
            rows.append(
                _component_geometry_row(
                    component=component,
                    z=int(z),
                    component_label=int(component_label),
                    component_rank=int(component_rank),
                    voxel_count=voxel_count,
                    area_per_voxel=area_per_voxel,
                    spacing_xyz=spacing_xyz,
                    case_id=case_id,
                )
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["component_count_in_slice"] = frame.groupby("slice_index_z")["component_label_2d"].transform("count")
    frame = _assign_branch_ids(frame, spacing_xyz, max_branch_link_distance_mm)
    frame = _add_branch_change_features(frame)
    return frame


def _component_geometry_row(
    component: np.ndarray,
    z: int,
    component_label: int,
    component_rank: int,
    voxel_count: int,
    area_per_voxel: float,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
) -> dict[str, object]:
    ys, xs = np.where(component)
    area_mm2 = float(voxel_count * area_per_voxel)
    equivalent_diameter_mm = float(np.sqrt(4.0 * area_mm2 / np.pi))
    width_x_mm = float((xs.max() - xs.min() + 1) * spacing_xyz[0])
    width_y_mm = float((ys.max() - ys.min() + 1) * spacing_xyz[1])
    max_diameter_mm = max(width_x_mm, width_y_mm)
    min_diameter_mm = min(width_x_mm, width_y_mm)
    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())

    boundary = component ^ _binary_erosion_2d(component)
    bys, bxs = np.where(boundary)
    if bys.size:
        radii = np.sqrt(((bxs - centroid_x) * spacing_xyz[0]) ** 2 + ((bys - centroid_y) * spacing_xyz[1]) ** 2)
        radius_mean = float(np.mean(radii))
        radius_sd = float(np.std(radii))
        radius_mad = float(np.median(np.abs(radii - np.median(radii))))
    else:
        radius_mean = np.nan
        radius_sd = np.nan
        radius_mad = np.nan
    radius_cv = float(radius_sd / radius_mean) if radius_mean and np.isfinite(radius_mean) else np.nan
    radius_mad_norm = float(radius_mad / radius_mean) if radius_mean and np.isfinite(radius_mean) else np.nan

    perimeter_mm = _perimeter_mm(boundary, spacing_xyz)
    circularity = float(4.0 * np.pi * area_mm2 / (perimeter_mm**2)) if perimeter_mm > 0 else np.nan
    circularity = float(np.clip(circularity, 0.0, 1.0)) if np.isfinite(circularity) else np.nan
    solidity = _solidity(component, area_mm2, spacing_xyz)
    eccentricity = _eccentricity(component)
    asymmetry_index = float((max_diameter_mm - min_diameter_mm) / max_diameter_mm) if max_diameter_mm else np.nan

    return {
        "case_id": case_id,
        "slice_index_z": z,
        "component_rank_in_slice": component_rank,
        "component_label_2d": component_label,
        "component_voxel_count": voxel_count,
        "centroid_x_voxel": centroid_x,
        "centroid_y_voxel": centroid_y,
        "centroid_x_mm": float(centroid_x * spacing_xyz[0]),
        "centroid_y_mm": float(centroid_y * spacing_xyz[1]),
        "centroid_z_mm": float(z * spacing_xyz[2]),
        "cross_section_area_mm2": area_mm2,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "maximum_diameter_mm": max_diameter_mm,
        "minimum_diameter_mm": min_diameter_mm,
        "radius_mean_mm": radius_mean,
        "radius_sd_mm": radius_sd,
        "radius_cv": radius_cv,
        "radius_mad_norm": radius_mad_norm,
        "eccentricity": eccentricity,
        "asymmetry_index": asymmetry_index,
        "perimeter_mm": perimeter_mm,
        "circularity": circularity,
        "circularity_defect": float(1.0 - circularity) if np.isfinite(circularity) else np.nan,
        "solidity": solidity,
        "solidity_defect": float(1.0 - solidity) if np.isfinite(solidity) else np.nan,
        "geometry_method": "axial_component_branch_fallback_not_orthogonal",
        "geometry_interpretation": "descriptive_geometry_only_not_irregularity_or_plaque_classifier",
    }


def _assign_branch_ids(
    frame: pd.DataFrame,
    spacing_xyz: tuple[float, float, float],
    max_link_distance_mm: float,
) -> pd.DataFrame:
    out = frame.sort_values(["slice_index_z", "component_rank_in_slice"]).copy()
    active: dict[int, tuple[int, np.ndarray]] = {}
    next_branch_id = 1
    branch_ids: list[int] = []

    for row in out.itertuples(index=False):
        current_slice = int(row.slice_index_z)
        point = np.asarray(
            [
                float(row.centroid_x_voxel) * spacing_xyz[0],
                float(row.centroid_y_voxel) * spacing_xyz[1],
                float(current_slice) * spacing_xyz[2],
            ],
            dtype=float,
        )
        best_branch = None
        best_distance = float("inf")
        for branch_id, (previous_slice, previous) in active.items():
            if previous_slice >= current_slice:
                continue
            distance = float(np.linalg.norm(point - previous))
            if distance < best_distance:
                best_branch = branch_id
                best_distance = distance
        if best_branch is None or best_distance > max_link_distance_mm:
            best_branch = next_branch_id
            next_branch_id += 1
        active[best_branch] = (current_slice, point)
        branch_ids.append(best_branch)

    out["branch_id"] = branch_ids
    return out.sort_index()


def _add_branch_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["local_area_change_fraction"] = 0.0
    out["local_diameter_change_fraction"] = 0.0
    out["centroid_step_mm"] = 0.0

    for _, branch in out.groupby("branch_id", sort=False):
        idx = branch.index
        area = branch["cross_section_area_mm2"].astype(float)
        diameter = branch["equivalent_diameter_mm"].astype(float)
        previous_area = area.shift(1)
        next_area = area.shift(-1)
        previous_diameter = diameter.shift(1)
        next_diameter = diameter.shift(-1)
        area_ref = pd.concat([previous_area, next_area], axis=1).median(axis=1)
        diameter_ref = pd.concat([previous_diameter, next_diameter], axis=1).median(axis=1)
        out.loc[idx, "local_area_change_fraction"] = _fractional_change(area, area_ref)
        out.loc[idx, "local_diameter_change_fraction"] = _fractional_change(diameter, diameter_ref)

        coords = branch[["centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]].astype(float).to_numpy()
        steps = np.zeros(len(coords), dtype=float)
        if len(coords) > 1:
            steps[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        out.loc[idx, "centroid_step_mm"] = steps

    return out


def _fractional_change(values: pd.Series, references: pd.Series) -> np.ndarray:
    values_array = values.astype(float).to_numpy()
    references_array = references.astype(float).to_numpy()
    out = np.zeros_like(values_array, dtype=float)
    valid = np.isfinite(references_array) & (references_array > 0)
    out[valid] = np.abs(values_array[valid] - references_array[valid]) / references_array[valid]
    return out


def _label_2d(binary: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage as ndi

        labels, n_labels = ndi.label(binary)
        return labels.astype(np.int32), int(n_labels)
    except Exception:
        return _label_2d_fallback(binary)


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
    sizes = [(label, int(np.sum(labels == label))) for label in range(1, n_labels + 1)]
    sizes.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in sizes[:max_components]]


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


def _perimeter_mm(boundary: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    if not boundary.any():
        return 0.0
    horizontal_edges = np.logical_xor(boundary[:, 1:], boundary[:, :-1]).sum() * spacing_xyz[1]
    vertical_edges = np.logical_xor(boundary[1:, :], boundary[:-1, :]).sum() * spacing_xyz[0]
    return float(horizontal_edges + vertical_edges)


def _solidity(component: np.ndarray, area_mm2: float, spacing_xyz: tuple[float, float, float]) -> float:
    ys, xs = np.where(component)
    bbox_area = float((ys.max() - ys.min() + 1) * spacing_xyz[1] * (xs.max() - xs.min() + 1) * spacing_xyz[0])
    return float(area_mm2 / bbox_area) if bbox_area > 0 else np.nan


def _eccentricity(component: np.ndarray) -> float:
    ys, xs = np.where(component)
    if len(xs) < 3:
        return np.nan
    coords = np.vstack([xs - xs.mean(), ys - ys.mean()])
    covariance = np.cov(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = float(np.max(eigenvalues))
    minor = float(np.min(eigenvalues))
    if major <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - minor / major)))
