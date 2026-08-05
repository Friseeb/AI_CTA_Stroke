"""Deterministic arterial surface-contact measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._spatial import indices_zyx_to_physical_xyz, validate_spacing
from .composition import TissueLabel
from .coordinates import VesselCoordinateSystem
from .sectors import sector_index_from_angle


DEFAULT_CONTACT_DISTANCES_MM = (0.5, 1.0, 2.0)
DEFAULT_CONTACT_CLASSES: dict[str, int] = {
    "adipose": int(TissueLabel.ADIPOSE),
    "skeletal_muscle": int(TissueLabel.SKELETAL_MUSCLE),
    "vein": int(TissueLabel.VEIN),
    "bone": int(TissueLabel.BONE),
    "thyroid_gland": int(TissueLabel.THYROID_GLAND),
    "other_soft_tissue": int(TissueLabel.OTHER_SOFT_TISSUE),
    "high_density": int(TissueLabel.HIGH_DENSITY),
    "uncertain": int(TissueLabel.UNCERTAIN),
}


@dataclass(frozen=True)
class ContactClassMetrics:
    tissue_name: str
    tissue_label: int
    distance_mm: float
    contact_area_mm2: float
    contact_fraction: float
    contiguous_contact_length_mm: float
    number_of_contact_regions: int
    longitudinal_profile: tuple[dict[str, float | int], ...]
    circumferential_profile: tuple[dict[str, float | int], ...]
    confidence: float
    uncertainty: float
    qc_flags: tuple[str, ...]


@dataclass(frozen=True)
class ContactDistanceResult:
    distance_mm: float
    total_surface_area_mm2: float
    assessable_surface_area_mm2: float
    boundary_missing_area_mm2: float
    boundary_truncation_fraction: float
    ambiguous_contact_area_mm2: float
    classes: dict[str, ContactClassMetrics]
    qc_flags: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceContactResult:
    vessel_id: str
    spacing_xyz: tuple[float, float, float]
    method: str
    distances: dict[float, ContactDistanceResult]
    qc_flags: tuple[str, ...]

    def at(self, distance_mm: float) -> ContactDistanceResult:
        for distance, result in self.distances.items():
            if np.isclose(distance, distance_mm):
                return result
        raise KeyError(f"No contact result for {distance_mm:g} mm.")


@dataclass(frozen=True)
class _SurfaceFaces:
    interior_zyx: np.ndarray
    exterior_zyx: np.ndarray
    exterior_valid: np.ndarray
    centres_zyx: np.ndarray
    area_mm2: np.ndarray


def _surface_faces(lumen: np.ndarray, spacing_xyz: tuple[float, float, float]) -> _SurfaceFaces:
    coords = np.argwhere(lumen)
    interiors: list[np.ndarray] = []
    exteriors: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    # NumPy axes z/y/x.  A face normal to each axis has the product of the
    # other two physical spacings.
    axis_areas = (
        spacing_xyz[0] * spacing_xyz[1],
        spacing_xyz[0] * spacing_xyz[2],
        spacing_xyz[1] * spacing_xyz[2],
    )
    shape = np.asarray(lumen.shape)
    for axis in range(3):
        for sign in (-1, 1):
            neighbor = coords.copy()
            neighbor[:, axis] += sign
            valid = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
            neighbor_is_lumen = np.zeros(len(coords), dtype=bool)
            if valid.any():
                neighbor_is_lumen[valid] = lumen[tuple(neighbor[valid].T)]
            exposed = ~neighbor_is_lumen
            count = int(exposed.sum())
            if not count:
                continue
            inside = coords[exposed]
            outside = neighbor[exposed]
            is_valid = valid[exposed]
            face_centres = inside.astype(float)
            face_centres[:, axis] += 0.5 * sign
            interiors.append(inside)
            exteriors.append(outside)
            validities.append(is_valid)
            centres.append(face_centres)
            areas.append(np.full(count, axis_areas[axis], dtype=float))
    if not interiors:
        empty_int = np.empty((0, 3), dtype=int)
        return _SurfaceFaces(
            interior_zyx=empty_int,
            exterior_zyx=empty_int.copy(),
            exterior_valid=np.empty(0, dtype=bool),
            centres_zyx=np.empty((0, 3), dtype=float),
            area_mm2=np.empty(0, dtype=float),
        )
    return _SurfaceFaces(
        interior_zyx=np.concatenate(interiors),
        exterior_zyx=np.concatenate(exteriors),
        exterior_valid=np.concatenate(validities),
        centres_zyx=np.concatenate(centres),
        area_mm2=np.concatenate(areas),
    )


def _number_of_regions(
    shape: tuple[int, ...],
    coords: np.ndarray,
) -> tuple[int, np.ndarray, tuple[int, int, int]]:
    """Label contact regions in the smallest topology-preserving crop.

    Contact voxels occupy only the arterial surface.  Allocating and labelling
    an entire CTA-sized array for every tissue class and contact distance is
    therefore unnecessary.  A one-voxel halo preserves the same 18-connected
    component topology as a full-grid label operation.
    """
    from scipy import ndimage as ndi

    if not coords.size:
        return 0, np.zeros((1, 1, 1), dtype=np.int32), (0, 0, 0)
    coordinates = np.asarray(coords, dtype=int).reshape((-1, 3))
    full_shape = np.asarray(shape, dtype=int)
    if np.any(coordinates < 0) or np.any(coordinates >= full_shape):
        raise ValueError("Contact-region coordinates fall outside the image grid.")
    minimum = np.maximum(coordinates.min(axis=0) - 1, 0)
    maximum = np.minimum(coordinates.max(axis=0) + 2, full_shape)
    local_shape = tuple(int(value) for value in maximum - minimum)
    mask = np.zeros(local_shape, dtype=bool)
    local_coordinates = coordinates - minimum
    mask[tuple(local_coordinates.T)] = True
    labelled, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 2))
    return int(count), labelled, tuple(int(value) for value in minimum)


def _contiguous_length(
    labelled: np.ndarray,
    count: int,
    offset_zyx: tuple[int, int, int],
    coordinate_system: VesselCoordinateSystem | None,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: Sequence[float],
    direction: Sequence[float] | np.ndarray | None,
) -> float:
    if coordinate_system is None or count == 0:
        return float("nan") if coordinate_system is None else 0.0
    total = 0.0
    resolution = min(float(coordinate_system.resample_interval_mm), min(spacing_xyz))
    for component in range(1, count + 1):
        coords = np.argwhere(labelled == component) + np.asarray(offset_zyx, dtype=int)
        physical = indices_zyx_to_physical_xyz(coords, spacing_xyz, origin_xyz, direction)
        path = coordinate_system.physical_to_vessel(physical).path_mm
        total += max(resolution, float(np.ptp(path)) + resolution)
    return total


def _bounded_distance_values_at_exterior(
    labels: np.ndarray,
    label_value: int,
    exterior_zyx: np.ndarray,
    exterior_valid: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    maximum_distance_mm: float,
) -> np.ndarray:
    """Return exact EDT values only where surface contact is queried.

    The crop has a two-voxel safety halo beyond the largest requested physical
    distance.  A class voxel outside that crop cannot change whether a sampled
    exterior voxel is within ``maximum_distance_mm``.  This preserves the
    thresholded full-grid EDT result while avoiding CTA-sized distance maps for
    every tissue class.
    """
    values = np.full(len(exterior_zyx), np.inf, dtype=float)
    valid_indices = np.flatnonzero(exterior_valid)
    if not valid_indices.size:
        return values
    sample_coordinates = np.asarray(exterior_zyx[valid_indices], dtype=int)
    spacing_zyx = np.asarray(
        (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), dtype=float
    )
    pad = np.ceil(float(maximum_distance_mm) / spacing_zyx).astype(int) + 2
    full_shape = np.asarray(labels.shape, dtype=int)
    minimum = np.maximum(sample_coordinates.min(axis=0) - pad, 0)
    maximum = np.minimum(sample_coordinates.max(axis=0) + pad + 1, full_shape)
    slices = tuple(
        slice(int(minimum[axis]), int(maximum[axis])) for axis in range(3)
    )
    class_crop = np.asarray(labels[slices] == int(label_value), dtype=bool)
    if not class_crop.any():
        return values

    from scipy import ndimage as ndi

    distance_crop = ndi.distance_transform_edt(~class_crop, sampling=tuple(spacing_zyx))
    local_coordinates = sample_coordinates - minimum
    values[valid_indices] = distance_crop[tuple(local_coordinates.T)]
    return values


def _profiles(
    contact: np.ndarray,
    valid_faces: np.ndarray,
    areas: np.ndarray,
    paths: np.ndarray | None,
    angles: np.ndarray | None,
    *,
    bin_length_mm: float,
    number_of_sectors: int,
) -> tuple[tuple[dict[str, float | int], ...], tuple[dict[str, float | int], ...]]:
    longitudinal: list[dict[str, float | int]] = []
    circumferential: list[dict[str, float | int]] = []
    if paths is not None and paths.size:
        maximum = float(np.nanmax(paths[valid_faces])) if valid_faces.any() else 0.0
        edges = np.arange(0.0, maximum + bin_length_mm, bin_length_mm)
        if len(edges) < 2:
            edges = np.asarray([0.0, bin_length_mm])
        for index in range(len(edges) - 1):
            lower, upper = float(edges[index]), float(edges[index + 1])
            within = valid_faces & (paths >= lower) & (
                (paths <= upper) if index == len(edges) - 2 else (paths < upper)
            )
            assessed_area = float(areas[within].sum())
            contact_area = float(areas[within & contact].sum())
            longitudinal.append(
                {
                    "bin": index,
                    "path_start_mm": lower,
                    "path_end_mm": upper,
                    "contact_area_mm2": contact_area,
                    "assessable_area_mm2": assessed_area,
                    "contact_fraction": contact_area / assessed_area if assessed_area else float("nan"),
                }
            )
    if angles is not None and angles.size:
        sectors = sector_index_from_angle(angles, number_of_sectors)
        for sector in range(1, number_of_sectors + 1):
            within = valid_faces & (sectors == sector)
            assessed_area = float(areas[within].sum())
            contact_area = float(areas[within & contact].sum())
            circumferential.append(
                {
                    "sector": sector,
                    "contact_area_mm2": contact_area,
                    "assessable_area_mm2": assessed_area,
                    "contact_fraction": contact_area / assessed_area if assessed_area else float("nan"),
                }
            )
    return tuple(longitudinal), tuple(circumferential)


def calculate_surface_contacts(
    lumen_mask: np.ndarray,
    tissue_labels: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    distances_mm: Sequence[float] = DEFAULT_CONTACT_DISTANCES_MM,
    class_labels: Mapping[str, int] | None = None,
    coordinate_system: VesselCoordinateSystem | None = None,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | np.ndarray | None = None,
    longitudinal_bin_mm: float = 1.0,
    number_of_sectors: int = 8,
    vessel_id: str | None = None,
) -> SurfaceContactResult:
    """Estimate face-area contact to each tissue class at physical distances.

    The arterial surface is the exposed, spacing-weighted voxel-face surface.
    A face contacts a class when that class is within the requested physical EDT
    distance of its immediately exterior voxel.  Fractions always use the
    observed in-image surface area, never an assumed complete surface.
    """
    lumen = np.asarray(lumen_mask, dtype=bool)
    labels = np.asarray(tissue_labels)
    if lumen.ndim != 3 or labels.shape != lumen.shape:
        raise ValueError("lumen_mask and tissue_labels must be same-shaped 3D arrays.")
    if not lumen.any():
        raise ValueError("lumen_mask is empty.")
    spacing = validate_spacing(spacing_xyz)
    distances = tuple(sorted({float(value) for value in distances_mm}))
    if not distances or not np.isfinite(distances).all() or any(value < 0 for value in distances):
        raise ValueError("distances_mm must contain finite non-negative values.")
    if longitudinal_bin_mm <= 0 or not np.isfinite(longitudinal_bin_mm):
        raise ValueError("longitudinal_bin_mm must be positive and finite.")
    # Also validates the sector count.
    sector_index_from_angle(np.asarray([0.0]), number_of_sectors)
    classes = {
        str(name): int(value)
        for name, value in (DEFAULT_CONTACT_CLASSES if class_labels is None else class_labels).items()
    }
    if len(set(classes.values())) != len(classes):
        raise ValueError("class_labels values must be unique.")
    faces = _surface_faces(lumen, spacing)
    total_area = float(faces.area_mm2.sum())
    assessable_area = float(faces.area_mm2[faces.exterior_valid].sum())
    boundary_missing_area = total_area - assessable_area
    boundary_fraction = boundary_missing_area / total_area if total_area else 0.0
    face_physical = indices_zyx_to_physical_xyz(
        faces.centres_zyx, spacing, origin_xyz, direction
    ) if len(faces.centres_zyx) else np.empty((0, 3), dtype=float)
    paths: np.ndarray | None = None
    angles: np.ndarray | None = None
    if coordinate_system is not None and len(face_physical):
        coordinates = coordinate_system.physical_to_vessel(face_physical)
        paths = coordinates.path_mm
        angles = coordinates.angle_rad

    maximum_distance = max(distances)
    distance_values: dict[str, np.ndarray] = {}
    for name, value in classes.items():
        distance_values[name] = _bounded_distance_values_at_exterior(
            labels,
            value,
            faces.exterior_zyx,
            faces.exterior_valid,
            spacing,
            maximum_distance,
        )
    results: dict[float, ContactDistanceResult] = {}
    global_flags: set[str] = set()
    if boundary_missing_area:
        global_flags.add("surface_truncated_by_image_boundary")
    if not assessable_area:
        global_flags.add("empty_assessable_surface")
    for distance in distances:
        contact_by_class: dict[str, np.ndarray] = {}
        for name, values in distance_values.items():
            contact_by_class[name] = faces.exterior_valid & (values <= distance + 1.0e-9)
        if contact_by_class:
            multiplicity = np.stack(list(contact_by_class.values()), axis=0).sum(axis=0)
        else:
            multiplicity = np.zeros(len(faces.area_mm2), dtype=int)
        ambiguous = faces.exterior_valid & (multiplicity > 1)
        ambiguous_area = float(faces.area_mm2[ambiguous].sum())
        distance_flags: set[str] = set()
        if ambiguous_area:
            distance_flags.add("multi_class_proximity_overlap")
        if distance < 0.5 * min(spacing):
            distance_flags.add("contact_distance_below_half_minimum_voxel_spacing")
        class_results: dict[str, ContactClassMetrics] = {}
        for name, label_value in classes.items():
            contact = contact_by_class[name]
            area = float(faces.area_mm2[contact].sum())
            fraction = area / assessable_area if assessable_area else float("nan")
            unique_interior = (
                np.unique(faces.interior_zyx[contact], axis=0)
                if contact.any()
                else np.empty((0, 3), int)
            )
            region_count, labelled_regions, region_offset = _number_of_regions(
                lumen.shape, unique_interior
            )
            contiguous_length = _contiguous_length(
                labelled_regions,
                region_count,
                region_offset,
                coordinate_system,
                spacing,
                origin_xyz,
                direction,
            )
            longitudinal, circumferential = _profiles(
                contact,
                faces.exterior_valid,
                faces.area_mm2,
                paths,
                angles,
                bin_length_mm=float(longitudinal_bin_mm),
                number_of_sectors=number_of_sectors,
            )
            ambiguous_class_area = float(faces.area_mm2[contact & ambiguous].sum())
            ambiguity_fraction = ambiguous_class_area / area if area else 0.0
            coverage = assessable_area / total_area if total_area else 0.0
            confidence = float(np.clip(coverage * (1.0 - ambiguity_fraction), 0.0, 1.0))
            class_flags: list[str] = []
            if ambiguous_class_area:
                class_flags.append("contact_class_ambiguous_within_distance")
            if boundary_missing_area:
                class_flags.append("surface_truncated_by_image_boundary")
            if not area:
                class_flags.append("no_contact_detected")
            class_results[name] = ContactClassMetrics(
                tissue_name=name,
                tissue_label=label_value,
                distance_mm=distance,
                contact_area_mm2=area,
                contact_fraction=fraction,
                contiguous_contact_length_mm=contiguous_length,
                number_of_contact_regions=region_count,
                longitudinal_profile=longitudinal,
                circumferential_profile=circumferential,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                qc_flags=tuple(class_flags),
            )
        if boundary_missing_area:
            distance_flags.add("surface_truncated_by_image_boundary")
        results[distance] = ContactDistanceResult(
            distance_mm=distance,
            total_surface_area_mm2=total_area,
            assessable_surface_area_mm2=assessable_area,
            boundary_missing_area_mm2=boundary_missing_area,
            boundary_truncation_fraction=boundary_fraction,
            ambiguous_contact_area_mm2=ambiguous_area,
            classes=class_results,
            qc_flags=tuple(sorted(distance_flags)),
        )
        global_flags.update(distance_flags)
    resolved_vessel_id = vessel_id or (
        coordinate_system.vessel_id if coordinate_system is not None else "unknown"
    )
    return SurfaceContactResult(
        vessel_id=resolved_vessel_id,
        spacing_xyz=spacing,
        method="spacing_weighted_exposed_voxel_faces_with_bounded_exterior_physical_edt",
        distances=results,
        qc_flags=tuple(sorted(global_flags)),
    )


# Concise alias.
surface_contact = calculate_surface_contacts


__all__ = [
    "ContactClassMetrics",
    "ContactDistanceResult",
    "DEFAULT_CONTACT_CLASSES",
    "DEFAULT_CONTACT_DISTANCES_MM",
    "SurfaceContactResult",
    "calculate_surface_contacts",
    "surface_contact",
]
