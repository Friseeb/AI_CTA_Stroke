"""Explicit conversion of FLOWCAT and medical-image coordinate references."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

from .exceptions import FlowcatGeometryError
from .schemas import ImageGeometry


def validate_orientation(orientation: Iterable[str]) -> tuple[str, str, str]:
    """Validate an anatomical axis-code triplet (including permutations)."""

    codes = tuple(str(value).upper() for value in orientation)
    if len(codes) != 3 or any(code not in {"L", "R", "A", "P", "I", "S"} for code in codes):
        raise FlowcatGeometryError(f"Invalid anatomical orientation codes: {codes!r}")
    axis_groups = [{"L", "R"}, {"A", "P"}, {"I", "S"}]
    if any(sum(code in group for code in codes) != 1 for group in axis_groups):
        raise FlowcatGeometryError(f"Orientation does not contain one LR, AP, and IS axis: {codes!r}")
    return codes  # type: ignore[return-value]


def ras_to_lps(points_mm: np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert RAS physical coordinates to LPS by flipping x and y."""

    points = _points_array(points_mm)
    converted = points.copy()
    converted[..., 0:2] *= -1.0
    return converted


def lps_to_ras(points_mm: np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert LPS physical coordinates to RAS by flipping x and y."""

    return ras_to_lps(points_mm)


def lpi_corner_voxel(geometry: ImageGeometry) -> tuple[int, int, int]:
    """Return the voxel corner from which FLOWCAT removes translation.

    FLOWCAT calls this the LPI corner. For each voxel axis, index zero is used
    when its positive direction is R/A/S; otherwise the far edge is used. This
    generalizes the RAS, LAS, and LPS cases implemented in FLOWCAT itself.
    """

    orientation = validate_orientation(geometry.orientation)
    return tuple(
        0 if code in {"R", "A", "S"} else geometry.shape_ijk[axis] - 1
        for axis, code in enumerate(orientation)
    )  # type: ignore[return-value]


def lpi_corner_ras_mm(geometry: ImageGeometry) -> np.ndarray:
    affine = np.asarray(geometry.affine_ras_mm, dtype=float)
    corner_ijk = np.asarray((*lpi_corner_voxel(geometry), 1.0), dtype=float)
    return (affine @ corner_ijk)[:3]


def flowcat_relative_to_image_physical(
    points_mm: np.ndarray | Iterable[float],
    geometry: ImageGeometry,
) -> np.ndarray:
    """Restore NIfTI RAS physical coordinates from FLOWCAT relative-mm points."""

    return _points_array(points_mm) + lpi_corner_ras_mm(geometry)


def image_physical_to_flowcat_relative(
    points_ras_mm: np.ndarray | Iterable[float],
    geometry: ImageGeometry,
) -> np.ndarray:
    """Remove the NIfTI LPI-corner translation exactly as FLOWCAT does."""

    return _points_array(points_ras_mm) - lpi_corner_ras_mm(geometry)


def voxel_to_ras_physical(
    voxel_ijk: np.ndarray | Iterable[float],
    geometry: ImageGeometry,
) -> np.ndarray:
    points = _points_array(voxel_ijk)
    flattened = points.reshape(-1, 3)
    homogeneous = np.concatenate((flattened, np.ones((flattened.shape[0], 1))), axis=1)
    physical = (np.asarray(geometry.affine_ras_mm, dtype=float) @ homogeneous.T).T[:, :3]
    return physical.reshape(points.shape)


def assert_geometry_compatible(
    observed: ImageGeometry,
    reference: ImageGeometry,
    *,
    atol_mm: float = 1e-4,
) -> None:
    """Require matching dimensions, orientation, and RAS affine."""

    problems: list[str] = []
    if observed.shape_ijk != reference.shape_ijk:
        problems.append(f"shape {observed.shape_ijk} != {reference.shape_ijk}")
    if observed.orientation != reference.orientation:
        problems.append(f"orientation {observed.orientation} != {reference.orientation}")
    if not np.allclose(
        np.asarray(observed.affine_ras_mm),
        np.asarray(reference.affine_ras_mm),
        rtol=0.0,
        atol=atol_mm,
    ):
        problems.append("affine matrices differ")
    if problems:
        raise FlowcatGeometryError("Incompatible image geometry: " + "; ".join(problems))


def validate_points_within_image(
    points_ras_mm: np.ndarray | Iterable[float],
    geometry: ImageGeometry,
    *,
    tolerance_mm: float = 1.0,
) -> None:
    """Check physical points against the enclosing box of all image corners."""

    points = _points_array(points_ras_mm).reshape(-1, 3)
    affine = np.asarray(geometry.affine_ras_mm, dtype=float)
    corners = []
    for corner in product(*((0, size - 1) for size in geometry.shape_ijk)):
        corners.append((affine @ np.asarray((*corner, 1.0)))[:3])
    corners_array = np.asarray(corners)
    lower = corners_array.min(axis=0) - tolerance_mm
    upper = corners_array.max(axis=0) + tolerance_mm
    outside = np.logical_or(points < lower, points > upper).any(axis=1)
    if bool(outside.any()):
        first = points[np.flatnonzero(outside)[0]].tolist()
        raise FlowcatGeometryError(
            f"Physical point {first!r} lies outside image bounds "
            f"{lower.tolist()!r}..{upper.tolist()!r}"
        )


def _points_array(points: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.shape == (3,):
        pass
    elif array.ndim < 2 or array.shape[-1] != 3:
        raise FlowcatGeometryError(f"Coordinates must have trailing dimension 3, got {array.shape!r}")
    if not np.isfinite(array).all():
        raise FlowcatGeometryError("Coordinates contain NaN or infinite values")
    return array
