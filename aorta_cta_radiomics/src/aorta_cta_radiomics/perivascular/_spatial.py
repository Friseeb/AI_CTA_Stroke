"""Small physical-space helpers shared by perivascular modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_spacing(spacing_xyz: Sequence[float]) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in spacing_xyz)
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError("spacing_xyz must contain three positive finite values.")
    return spacing  # type: ignore[return-value]


def direction_matrix(direction: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if direction is None:
        return np.eye(3, dtype=float)
    matrix = np.asarray(direction, dtype=float)
    if matrix.size != 9:
        raise ValueError("direction must contain nine values.")
    matrix = matrix.reshape((3, 3))
    if not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-5):
        raise ValueError("direction must be a finite orthonormal 3x3 matrix.")
    return matrix


def indices_zyx_to_physical_xyz(
    indices_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | np.ndarray | None = None,
    *,
    voxel_centres: bool = False,
) -> np.ndarray:
    """Map NumPy z/y/x indices into x/y/z physical coordinates."""
    spacing = np.asarray(validate_spacing(spacing_xyz), dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_xyz must contain three finite values.")
    indices = np.asarray(indices_zyx, dtype=float)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices_zyx must have shape (n, 3).")
    xyz = indices[:, [2, 1, 0]]
    if voxel_centres:
        xyz = xyz + 0.5
    return origin + (direction_matrix(direction) @ (xyz * spacing).T).T


def physical_xyz_to_continuous_zyx(
    physical_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    spacing = np.asarray(validate_spacing(spacing_xyz), dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    points = np.asarray(physical_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("physical_xyz must have shape (n, 3).")
    xyz = (direction_matrix(direction).T @ (points - origin).T).T / spacing
    return xyz[:, [2, 1, 0]]


def voxel_volume_mm3(spacing_xyz: Sequence[float]) -> float:
    return float(np.prod(validate_spacing(spacing_xyz)))


__all__ = [
    "direction_matrix",
    "indices_zyx_to_physical_xyz",
    "physical_xyz_to_continuous_zyx",
    "validate_spacing",
    "voxel_volume_mm3",
]
