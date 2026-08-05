"""Safe loading and typed conversion of FLOWCAT centerline arrays."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .coordinate_systems import flowcat_relative_to_image_physical, validate_points_within_image
from .exceptions import FlowcatGeometryError, FlowcatSchemaError, UnsafeArtifactError
from .schemas import CenterlineBranch, CenterlinePoint, CoordinateReference, ImageGeometry


def load_numpy_array(
    path: str | Path,
    *,
    trusted_source: bool = False,
) -> np.ndarray:
    """Load a numeric NPY safely, or an object NPY only after explicit trust.

    Object arrays are pickle-backed and may execute arbitrary Python code while
    loading. Set ``trusted_source=True`` only for artifacts produced by the
    trusted local FLOWCAT pipeline, never for downloaded or user-supplied files.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"FLOWCAT NumPy artifact not found: {file_path}")
    try:
        return np.load(file_path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise FlowcatSchemaError(f"Could not read FLOWCAT NumPy artifact {file_path}: {exc}") from exc
        if trusted_source is not True:
            raise UnsafeArtifactError(
                f"Refusing to unpickle object NPY {file_path}. FLOWCAT centerline arrays "
                "must originate from a trusted local pipeline; pass trusted_source=True explicitly."
            ) from exc
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as exc:
        raise FlowcatSchemaError(f"Could not read trusted FLOWCAT object NPY {file_path}: {exc}") from exc


def load_centerline_segments(
    path: str | Path,
    *,
    trusted_source: bool = False,
    processing_version: str | None = None,
    geometry: ImageGeometry | None = None,
    convert_to_image_physical: bool = False,
    validate_in_image: bool = True,
) -> tuple[CenterlineBranch, ...]:
    """Load FLOWCAT's ``centerline_segments_array.npy`` as typed branches.

    FLOWCAT stores a shape ``(n_branches, 2)`` object array. Column zero holds
    LPI-corner-relative coordinates in millimetres and column one holds maximal
    inscribed-sphere radii in millimetres.
    """

    array = load_numpy_array(path, trusted_source=trusted_source)
    if array.ndim != 2 or array.shape[1] != 2:
        raise FlowcatSchemaError(
            "FLOWCAT centerline_segments_array must have shape (n_branches, 2); "
            f"got {array.shape!r}"
        )
    if array.shape[0] == 0:
        raise FlowcatSchemaError("FLOWCAT centerline_segments_array contains no branches")
    if convert_to_image_physical and geometry is None:
        raise FlowcatGeometryError("geometry is required when convert_to_image_physical=True")

    source_file = str(Path(path).expanduser().resolve())
    reference = CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM
    branches: list[CenterlineBranch] = []
    for cell_id in range(array.shape[0]):
        coordinates = np.asarray(array[cell_id, 0], dtype=float)
        radii = np.asarray(array[cell_id, 1], dtype=float)
        _validate_segment_arrays(cell_id, coordinates, radii)
        if convert_to_image_physical:
            assert geometry is not None
            coordinates = flowcat_relative_to_image_physical(coordinates, geometry)
            if validate_in_image:
                validate_points_within_image(coordinates, geometry)
            reference = CoordinateReference.NIFTI_RAS_PHYSICAL_MM

        branch_id = f"cell-{cell_id}"
        distances = _path_distances(coordinates)
        tangents = _unit_tangents(coordinates)
        points = tuple(
            CenterlinePoint(
                coordinates_mm=tuple(float(value) for value in coordinates[index]),
                path_distance_mm=float(distances[index]),
                local_radius_mm=float(radii[index]),
                branch_id=branch_id,
                coordinate_reference=reference,
                tangent=tuple(float(value) for value in tangents[index]),
                source_file=source_file,
                processing_version=processing_version,
            )
            for index in range(coordinates.shape[0])
        )
        branches.append(
            CenterlineBranch(
                branch_id=branch_id,
                cell_id=cell_id,
                points=points,
                coordinate_reference=reference,
                source_file=source_file,
                processing_version=processing_version,
            )
        )
    return tuple(branches)


def branches_to_image_physical(
    branches: tuple[CenterlineBranch, ...],
    geometry: ImageGeometry,
    *,
    validate_in_image: bool = True,
) -> tuple[CenterlineBranch, ...]:
    """Convert typed FLOWCAT-relative branches to absolute NIfTI RAS points."""

    converted: list[CenterlineBranch] = []
    for branch in branches:
        if branch.coordinate_reference is not CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM:
            raise FlowcatGeometryError(
                f"Branch {branch.branch_id!r} has coordinate reference "
                f"{branch.coordinate_reference!s}, expected FLOWCAT relative mm"
            )
        raw = np.asarray([point.coordinates_mm for point in branch.points], dtype=float)
        physical = flowcat_relative_to_image_physical(raw, geometry)
        if validate_in_image:
            validate_points_within_image(physical, geometry)
        points = tuple(
            replace(
                point,
                coordinates_mm=tuple(float(value) for value in physical[index]),
                coordinate_reference=CoordinateReference.NIFTI_RAS_PHYSICAL_MM,
            )
            for index, point in enumerate(branch.points)
        )
        converted.append(
            replace(
                branch,
                points=points,
                coordinate_reference=CoordinateReference.NIFTI_RAS_PHYSICAL_MM,
            )
        )
    return tuple(converted)


def _validate_segment_arrays(cell_id: int, coordinates: np.ndarray, radii: np.ndarray) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 2:
        raise FlowcatSchemaError(
            f"Centerline cell_id {cell_id} coordinates must have shape (n>=2, 3), got {coordinates.shape!r}"
        )
    if radii.ndim != 1 or radii.shape[0] != coordinates.shape[0]:
        raise FlowcatSchemaError(
            f"Centerline cell_id {cell_id} radii must have shape ({coordinates.shape[0]},), "
            f"got {radii.shape!r}"
        )
    if not np.isfinite(coordinates).all() or not np.isfinite(radii).all():
        raise FlowcatSchemaError(f"Centerline cell_id {cell_id} contains NaN or infinite values")
    if bool((radii < 0).any()):
        raise FlowcatSchemaError(f"Centerline cell_id {cell_id} contains a negative radius")
    steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    if bool((steps <= 1e-12).any()):
        raise FlowcatSchemaError(f"Centerline cell_id {cell_id} contains consecutive duplicate points")


def _path_distances(coordinates: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _unit_tangents(coordinates: np.ndarray) -> np.ndarray:
    derivatives = np.empty_like(coordinates, dtype=float)
    derivatives[0] = coordinates[1] - coordinates[0]
    derivatives[-1] = coordinates[-1] - coordinates[-2]
    if len(coordinates) > 2:
        derivatives[1:-1] = coordinates[2:] - coordinates[:-2]
    norms = np.linalg.norm(derivatives, axis=1)
    if bool((norms <= 1e-12).any()):
        raise FlowcatSchemaError("Cannot compute tangent for a zero-length centerline derivative")
    return derivatives / norms[:, None]
