"""Anatomical centreline-to-lumen label propagation.

This module does not infer anatomy.  It rasterizes labels that have already
been assigned to trusted centreline branches (for example by FLOWCAT).  Two
strategies are exposed deliberately so their disagreement can be retained as
uncertainty rather than hidden:

``nearest_centerline_labels``
    Euclidean nearest labelled centreline with radius and component checks.

``branch_aware_connected_labels``
    Marker-controlled watershed inside the connected lumen.  This respects
    mask connectivity and creates explicit bifurcation exclusion zones.

Arrays follow NIfTI/nibabel ``(i, j, k)`` order.  Coordinates are physical RAS
millimetres unless the caller converts them before constructing the seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


UNCERTAIN_LABEL = 14


@dataclass(frozen=True)
class CenterlineSeed:
    """One anatomically labelled centreline branch in physical millimetres."""

    branch_id: str
    label_id: int
    label_name: str
    points_ras_mm: np.ndarray
    radii_mm: np.ndarray | None = None
    label_confidence: float | np.ndarray = 1.0
    parent_branch_id: str | None = None

    def validated(self) -> "CenterlineSeed":
        points = np.asarray(self.points_ras_mm, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError(
                f"Branch {self.branch_id!r} points must have shape (N, 3), N >= 2."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Branch {self.branch_id!r} contains non-finite coordinates.")
        if not 1 <= int(self.label_id) <= 13:
            raise ValueError(
                f"Branch {self.branch_id!r} label_id must be in 1..13; "
                "14 is reserved for uncertain assignment."
            )
        if self.radii_mm is not None:
            radii = np.asarray(self.radii_mm, dtype=float)
            if radii.shape != (len(points),):
                raise ValueError(f"Branch {self.branch_id!r} radii length differs from points.")
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
                raise ValueError(f"Branch {self.branch_id!r} radii must be finite and positive.")
        confidence = np.asarray(self.label_confidence, dtype=float)
        if confidence.ndim > 1 or (confidence.ndim == 1 and confidence.shape != (len(points),)):
            raise ValueError(
                f"Branch {self.branch_id!r} confidence must be scalar or have one value per point."
            )
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError(f"Branch {self.branch_id!r} confidence must be within [0, 1].")
        return self


@dataclass
class LabelPropagationResult:
    """Outputs required for anatomical label-volume QC."""

    labels: np.ndarray
    uncertainty_mask: np.ndarray
    confidence: np.ndarray
    bifurcation_exclusion_mask: np.ndarray
    nearest_labels: np.ndarray
    connected_labels: np.ndarray
    disagreement_mask: np.ndarray
    summary: dict[str, object]


def _validate_inputs(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
) -> tuple[np.ndarray, np.ndarray, list[CenterlineSeed]]:
    lumen = np.asarray(lumen_mask, dtype=bool)
    transform = np.asarray(affine, dtype=float)
    if lumen.ndim != 3:
        raise ValueError("lumen_mask must be three-dimensional.")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("affine must be a finite 4x4 matrix.")
    if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-10:
        raise ValueError("affine is singular.")
    checked = [branch.validated() for branch in branches]
    branch_ids = [branch.branch_id for branch in checked]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("branch_id values must be unique.")
    if lumen.any() and not checked:
        raise ValueError("At least one labelled centreline branch is required.")
    return lumen, transform, checked


def _voxel_centres_ras(indices_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(indices_ijk, dtype=float), np.ones(len(indices_ijk), dtype=float)]
    )
    return (homogeneous @ affine.T)[:, :3]


def _points_to_voxels(points_ras_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(affine)
    homogeneous = np.column_stack(
        [np.asarray(points_ras_mm, dtype=float), np.ones(len(points_ras_mm), dtype=float)]
    )
    return (homogeneous @ inverse.T)[:, :3]


def _flatten_branch_points(
    branches: Sequence[CenterlineSeed],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    branch_indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for branch_index, branch in enumerate(branches, start=1):
        branch_points = np.asarray(branch.points_ras_mm, dtype=float)
        n_points = len(branch_points)
        points.append(branch_points)
        branch_indices.append(np.full(n_points, branch_index, dtype=np.int32))
        labels.append(np.full(n_points, int(branch.label_id), dtype=np.uint8))
        radii.append(
            np.asarray(branch.radii_mm, dtype=float)
            if branch.radii_mm is not None
            else np.full(n_points, np.nan, dtype=float)
        )
        confidence = np.asarray(branch.label_confidence, dtype=float)
        confidences.append(
            np.full(n_points, float(confidence), dtype=float)
            if confidence.ndim == 0
            else confidence
        )
    return (
        np.concatenate(points, axis=0),
        np.concatenate(branch_indices),
        np.concatenate(labels),
        np.concatenate(radii),
        np.concatenate(confidences),
    )


def _distinct_branch_neighbours(
    tree: cKDTree,
    query_points: np.ndarray,
    point_branch_indices: np.ndarray,
    *,
    maximum_k: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest and nearest-distinct-branch point indices/distances."""

    k = min(maximum_k, tree.n)
    distances, indices = tree.query(query_points, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    first_index = indices[:, 0].astype(np.int64)
    first_distance = distances[:, 0].astype(float)
    first_branch = point_branch_indices[first_index]
    second_index = np.full(len(query_points), -1, dtype=np.int64)
    second_distance = np.full(len(query_points), np.inf, dtype=float)
    for column in range(1, k):
        candidate = indices[:, column]
        distinct = (second_index < 0) & (
            point_branch_indices[candidate] != first_branch
        )
        second_index[distinct] = candidate[distinct]
        second_distance[distinct] = distances[distinct, column]
    return first_index, first_distance, second_index, second_distance


def _seeded_component_map(
    lumen: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
) -> tuple[np.ndarray, set[int]]:
    components, _ = ndimage.label(lumen, structure=np.ones((3, 3, 3), dtype=bool))
    seeded_components: set[int] = set()
    for branch in branches:
        voxels = np.rint(_points_to_voxels(branch.points_ras_mm, affine)).astype(int)
        valid = np.all((voxels >= 0) & (voxels < np.asarray(lumen.shape)), axis=1)
        voxels = voxels[valid]
        if len(voxels):
            values = components[tuple(voxels.T)]
            seeded_components.update(int(value) for value in values if value > 0)
    return components, seeded_components


def nearest_centerline_labels(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
    *,
    radius_factor: float = 1.75,
    radius_margin_mm: float = 0.75,
    tie_tolerance_mm: float = 0.35,
    minimum_confidence: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Strategy A: nearest compatible labelled centreline.

    Compatibility requires a seeded lumen component.  Where a local radius is
    available, assignments farther than ``radius_factor * radius + margin`` are
    marked uncertain.  This strategy intentionally remains a comparator rather
    than the default authoritative result at bifurcations.
    """

    lumen, transform, checked = _validate_inputs(lumen_mask, affine, branches)
    labels_volume = np.zeros(lumen.shape, dtype=np.uint8)
    confidence_volume = np.zeros(lumen.shape, dtype=np.float32)
    if not lumen.any():
        return labels_volume, confidence_volume
    if radius_factor <= 0 or radius_margin_mm < 0 or tie_tolerance_mm < 0:
        raise ValueError("Radius and tie parameters must be non-negative, with radius_factor > 0.")

    points, point_branches, point_labels, point_radii, point_confidence = (
        _flatten_branch_points(checked)
    )
    tree = cKDTree(points)
    lumen_indices = np.argwhere(lumen)
    physical = _voxel_centres_ras(lumen_indices, transform)
    first, distance, second, second_distance = _distinct_branch_neighbours(
        tree, physical, point_branches
    )
    selected_labels = point_labels[first].copy()

    radius = point_radii[first]
    compatible = np.isnan(radius) | (
        distance <= radius_factor * radius + radius_margin_mm
    )
    margin = np.ones(len(distance), dtype=float)
    finite_second = np.isfinite(second_distance)
    margin[finite_second] = np.clip(
        (second_distance[finite_second] - distance[finite_second])
        / np.maximum(second_distance[finite_second], 1e-6),
        0,
        1,
    )
    scale = np.where(np.isfinite(radius), np.maximum(radius, 0.5), 2.0)
    confidence = point_confidence[first] * np.exp(-distance / scale) * (0.5 + 0.5 * margin)
    tied = np.isfinite(second_distance) & (
        np.abs(second_distance - distance) <= tie_tolerance_mm
    )

    components, seeded_components = _seeded_component_map(lumen, transform, checked)
    component_values = components[tuple(lumen_indices.T)]
    component_compatible = np.isin(component_values, list(seeded_components))
    uncertain = (~compatible) | tied | (~component_compatible) | (confidence < minimum_confidence)
    selected_labels[uncertain] = UNCERTAIN_LABEL
    confidence[uncertain] = np.minimum(confidence[uncertain], minimum_confidence)
    labels_volume[tuple(lumen_indices.T)] = selected_labels
    confidence_volume[tuple(lumen_indices.T)] = confidence.astype(np.float32)
    return labels_volume, confidence_volume


def bifurcation_exclusion_mask(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
    *,
    junction_detection_mm: float = 2.0,
    exclusion_radius_mm: float = 3.0,
) -> np.ndarray:
    """Create lumen-restricted physical balls around branch junctions."""

    lumen, transform, checked = _validate_inputs(lumen_mask, affine, branches)
    result = np.zeros(lumen.shape, dtype=bool)
    if not lumen.any():
        return result
    if junction_detection_mm < 0 or exclusion_radius_mm < 0:
        raise ValueError("Junction and exclusion radii must be non-negative.")
    branch_by_id = {branch.branch_id: branch for branch in checked}
    junctions: list[np.ndarray] = []
    for branch in checked:
        if branch.parent_branch_id in branch_by_id:
            parent = branch_by_id[branch.parent_branch_id]
            endpoints = np.vstack(
                [branch.points_ras_mm[[0, -1]], parent.points_ras_mm[[0, -1]]]
            )
            distances = np.linalg.norm(
                endpoints[:2, None, :] - endpoints[None, 2:, :], axis=2
            )
            first, second = np.unravel_index(np.argmin(distances), distances.shape)
            junctions.append((endpoints[first] + endpoints[second + 2]) / 2.0)
    for index, first_branch in enumerate(checked):
        first_endpoints = np.asarray(first_branch.points_ras_mm)[[0, -1]]
        for second_branch in checked[index + 1 :]:
            second_endpoints = np.asarray(second_branch.points_ras_mm)[[0, -1]]
            distances = np.linalg.norm(
                first_endpoints[:, None, :] - second_endpoints[None, :, :], axis=2
            )
            first, second = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[first, second] <= junction_detection_mm:
                junctions.append(
                    (first_endpoints[first] + second_endpoints[second]) / 2.0
                )
    if not junctions:
        return result
    lumen_indices = np.argwhere(lumen)
    physical = _voxel_centres_ras(lumen_indices, transform)
    tree = cKDTree(np.unique(np.round(np.asarray(junctions), decimals=6), axis=0))
    distances, _ = tree.query(physical, k=1, workers=-1)
    selected = lumen_indices[distances <= exclusion_radius_mm]
    if len(selected):
        result[tuple(selected.T)] = True
    return result


def _branch_markers(
    lumen: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
) -> tuple[np.ndarray, dict[int, int], int]:
    markers = np.zeros(lumen.shape, dtype=np.int32)
    marker_to_label: dict[int, int] = {}
    collision_count = 0
    lumen_indices = np.argwhere(lumen)
    lumen_tree = cKDTree(_voxel_centres_ras(lumen_indices, affine))
    for marker_id, branch in enumerate(branches, start=1):
        marker_to_label[marker_id] = int(branch.label_id)
        _, nearest = lumen_tree.query(branch.points_ras_mm, k=1, workers=-1)
        voxels = lumen_indices[np.unique(nearest)]
        existing = markers[tuple(voxels.T)]
        collision_count += int(np.count_nonzero((existing != 0) & (existing != marker_id)))
        available = existing == 0
        voxels = voxels[available]
        if len(voxels):
            markers[tuple(voxels.T)] = marker_id
    return markers, marker_to_label, collision_count


def branch_aware_connected_labels(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
) -> tuple[np.ndarray, dict[str, object]]:
    """Strategy B: branch-marker propagation constrained to the lumen mask.

    Marker-controlled watershed uses negative interior EDT as the topography.
    It is more expensive than nearest-centreline assignment but cannot cross a
    disconnected background gap.  The result is still not anatomical truth;
    graph labels and bifurcation QC remain mandatory.
    """

    lumen, transform, checked = _validate_inputs(lumen_mask, affine, branches)
    output = np.zeros(lumen.shape, dtype=np.uint8)
    if not lumen.any():
        return output, {"marker_count": 0, "marker_collisions": 0, "unseeded_voxels": 0}
    try:
        from skimage.segmentation import watershed
    except ImportError as exc:  # pragma: no cover - declared production dependency
        raise ImportError("scikit-image is required for branch-aware propagation.") from exc

    coordinates = np.argwhere(lumen)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(minimum, maximum))
    cropped_lumen = lumen[slices]
    cropped_affine = transform.copy()
    cropped_affine[:3, 3] = (
        transform @ np.append(minimum.astype(float), 1.0)
    )[:3]
    markers, marker_to_label, collisions = _branch_markers(
        cropped_lumen, cropped_affine, checked
    )
    spacing = np.linalg.norm(transform[:3, :3], axis=0)
    distance_inside = ndimage.distance_transform_edt(cropped_lumen, sampling=spacing)
    branch_regions = watershed(
        -distance_inside,
        markers=markers,
        mask=cropped_lumen,
        connectivity=np.ones((3, 3, 3), dtype=bool),
        watershed_line=False,
    )
    label_lookup = np.zeros(max(marker_to_label, default=0) + 1, dtype=np.uint8)
    for marker_id, label_id in marker_to_label.items():
        label_lookup[marker_id] = label_id
    cropped_output = label_lookup[branch_regions]
    cropped_output[cropped_lumen & (branch_regions == 0)] = UNCERTAIN_LABEL
    output[slices] = cropped_output
    return output, {
        "marker_count": int(np.count_nonzero(np.unique(markers))),
        "marker_collisions": collisions,
        "unseeded_voxels": int(np.count_nonzero(cropped_lumen & (branch_regions == 0))),
        "algorithm": "marker_controlled_watershed_on_negative_interior_edt",
        "crop_minimum_ijk": minimum.tolist(),
        "crop_shape": list(cropped_lumen.shape),
    }


def generate_anatomical_artery_volume(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    branches: Sequence[CenterlineSeed],
    *,
    bifurcation_radius_mm: float = 3.0,
    minimum_confidence: float = 0.05,
) -> LabelPropagationResult:
    """Run and compare both strategies, preserving conflicts as label 14."""

    lumen, transform, checked = _validate_inputs(lumen_mask, affine, branches)
    nearest, confidence = nearest_centerline_labels(
        lumen,
        transform,
        checked,
        minimum_confidence=minimum_confidence,
    )
    connected, connected_meta = branch_aware_connected_labels(lumen, transform, checked)
    bifurcation = bifurcation_exclusion_mask(
        lumen,
        transform,
        checked,
        exclusion_radius_mm=bifurcation_radius_mm,
    )
    disagreement = lumen & (nearest != connected)
    low_confidence = lumen & (confidence < minimum_confidence)
    uncertainty = lumen & (
        disagreement
        | bifurcation
        | low_confidence
        | (nearest == UNCERTAIN_LABEL)
        | (connected == UNCERTAIN_LABEL)
        | (connected == 0)
    )
    final = connected.copy()
    final[uncertainty] = UNCERTAIN_LABEL
    final[~lumen] = 0
    confidence = confidence.copy()
    confidence[uncertainty] = np.minimum(confidence[uncertainty], minimum_confidence)

    label_summary: dict[str, dict[str, object]] = {}
    voxel_volume = abs(float(np.linalg.det(transform[:3, :3])))
    for label_id in sorted(int(value) for value in np.unique(final) if value != 0):
        count = int(np.count_nonzero(final == label_id))
        names = sorted(
            {branch.label_name for branch in checked if branch.label_id == label_id}
        )
        label_summary[str(label_id)] = {
            "label_names": names if label_id != UNCERTAIN_LABEL else ["uncertain_or_conflicting"],
            "voxel_count": count,
            "volume_mm3": count * voxel_volume,
        }
    lumen_voxels = int(lumen.sum())
    summary: dict[str, object] = {
        "authoritative_source": "input_anatomical_centerline_labels",
        "final_strategy": "branch_aware_connected_with_disagreement_and_bifurcation_uncertainty",
        "comparator_strategy": "nearest_compatible_labelled_centerline",
        "lumen_voxels": lumen_voxels,
        "uncertain_voxels": int(uncertainty.sum()),
        "uncertain_fraction": float(uncertainty.sum() / lumen_voxels) if lumen_voxels else 0.0,
        "disagreement_voxels": int(disagreement.sum()),
        "bifurcation_exclusion_voxels": int(bifurcation.sum()),
        "connected_propagation": connected_meta,
        "labels": label_summary,
    }
    return LabelPropagationResult(
        labels=final,
        uncertainty_mask=uncertainty,
        confidence=confidence,
        bifurcation_exclusion_mask=bifurcation,
        nearest_labels=nearest,
        connected_labels=connected,
        disagreement_mask=disagreement,
        summary=summary,
    )


__all__ = [
    "CenterlineSeed",
    "LabelPropagationResult",
    "UNCERTAIN_LABEL",
    "bifurcation_exclusion_mask",
    "branch_aware_connected_labels",
    "generate_anatomical_artery_volume",
    "nearest_centerline_labels",
]
