"""Deterministic vessel-centred coordinates in physical millimetres.

The coordinate backbone is a resampled physical-space polyline.  A
rotation-minimising (parallel-transport) frame is propagated along the line so
that circumferential angles do not acquire the arbitrary slice-to-slice twist
of independently fitted cross-sectional planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1.0e-12


def _as_points(points_xyz: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (n, 3).")
    if len(points) < 2:
        raise ValueError("A centreline requires at least two points.")
    if not np.isfinite(points).all():
        raise ValueError("Centreline coordinates must be finite.")
    return points


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > _EPS:
        return np.asarray(vector, dtype=float) / norm
    if fallback is not None:
        return _normalize(np.asarray(fallback, dtype=float))
    raise ValueError("Cannot normalize a zero-length vector.")


def _remove_consecutive_duplicates(points: np.ndarray, tolerance_mm: float = 1.0e-8) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > float(tolerance_mm)))
    result = points[keep]
    if len(result) < 2:
        raise ValueError("Centreline has no non-zero-length segment.")
    return result


def cumulative_path_length(points_xyz: np.ndarray) -> np.ndarray:
    """Return cumulative physical path length for an ``(n, 3)`` polyline."""
    points = _as_points(points_xyz)
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def resample_polyline(points_xyz: np.ndarray, interval_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at a physical interval while preserving both endpoints."""
    if not np.isfinite(interval_mm) or interval_mm <= 0:
        raise ValueError("interval_mm must be a positive finite value.")
    points = _remove_consecutive_duplicates(_as_points(points_xyz))
    original_path = cumulative_path_length(points)
    total = float(original_path[-1])
    targets = np.arange(0.0, total, float(interval_mm), dtype=float)
    if len(targets) == 0 or total - targets[-1] > 1.0e-9:
        targets = np.append(targets, total)
    else:
        targets[-1] = total
    resampled = np.column_stack(
        [np.interp(targets, original_path, points[:, axis]) for axis in range(3)]
    )
    return resampled, targets


def stable_tangents(points_xyz: np.ndarray, path_mm: np.ndarray) -> np.ndarray:
    """Estimate stable unit tangents using physical-distance finite differences."""
    points = _as_points(points_xyz)
    path = np.asarray(path_mm, dtype=float)
    if path.shape != (len(points),) or np.any(np.diff(path) <= 0):
        raise ValueError("path_mm must be strictly increasing and match points_xyz.")
    if len(points) == 2:
        tangent = _normalize(points[1] - points[0])
        return np.vstack((tangent, tangent))
    derivative = np.column_stack(
        [np.gradient(points[:, axis], path, edge_order=2) for axis in range(3)]
    )
    tangents = np.empty_like(derivative)
    fallback = _normalize(points[1] - points[0])
    for index, value in enumerate(derivative):
        tangents[index] = _normalize(value, fallback=fallback)
        fallback = tangents[index]
    return tangents


def _initial_normal(tangent: np.ndarray, preferred: np.ndarray | None = None) -> np.ndarray:
    if preferred is not None:
        candidate = np.asarray(preferred, dtype=float)
        if candidate.shape != (3,) or not np.isfinite(candidate).all():
            raise ValueError("initial_normal_xyz must contain three finite values.")
        projected = candidate - np.dot(candidate, tangent) * tangent
        if np.linalg.norm(projected) > 1.0e-8:
            return _normalize(projected)
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ tangent)))]
    return _normalize(reference - np.dot(reference, tangent) * tangent)


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def rotation_minimizing_frames(
    tangents_xyz: np.ndarray,
    initial_normal_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-transport a normal/binormal frame along unit tangents."""
    tangents = np.asarray(tangents_xyz, dtype=float)
    if tangents.ndim != 2 or tangents.shape[1] != 3 or len(tangents) < 2:
        raise ValueError("tangents_xyz must have shape (n, 3), n >= 2.")
    tangents = np.vstack([_normalize(value) for value in tangents])
    normals = np.empty_like(tangents)
    binormals = np.empty_like(tangents)
    normals[0] = _initial_normal(tangents[0], initial_normal_xyz)
    binormals[0] = _normalize(np.cross(tangents[0], normals[0]))

    for index in range(1, len(tangents)):
        previous = tangents[index - 1]
        current = tangents[index]
        cross = np.cross(previous, current)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
        if cross_norm <= 1.0e-10:
            transported = normals[index - 1].copy()
            if dot < 0.0:
                # A true 180-degree reversal is geometrically ambiguous.  Use
                # the previous binormal as a deterministic rotation axis.
                transported = _rodrigues(transported, binormals[index - 1], np.pi)
        else:
            transported = _rodrigues(normals[index - 1], cross / cross_norm, np.arctan2(cross_norm, dot))
        transported -= np.dot(transported, current) * current
        normals[index] = _normalize(transported, fallback=normals[index - 1])
        binormals[index] = _normalize(np.cross(current, normals[index]))
        # Recompute the first normal to remove accumulated numerical skew.
        normals[index] = _normalize(np.cross(binormals[index], current))
    return normals, binormals


def curvature_and_torsion(
    points_xyz: np.ndarray,
    path_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return curvature (1/mm) and geometric torsion (1/mm where stable)."""
    points = _as_points(points_xyz)
    path = np.asarray(path_mm, dtype=float)
    if path.shape != (len(points),):
        raise ValueError("path_mm must match points_xyz.")
    if len(points) < 3:
        return np.zeros(len(points), dtype=float), np.full(len(points), np.nan)
    edge_order = 2 if len(points) >= 3 else 1
    first = np.column_stack([np.gradient(points[:, a], path, edge_order=edge_order) for a in range(3)])
    second = np.column_stack([np.gradient(first[:, a], path, edge_order=edge_order) for a in range(3)])
    speed = np.linalg.norm(first, axis=1)
    cross = np.cross(first, second)
    cross_norm = np.linalg.norm(cross, axis=1)
    curvature = np.divide(
        cross_norm,
        np.maximum(speed, _EPS) ** 3,
        out=np.zeros_like(cross_norm),
        where=speed > _EPS,
    )
    torsion = np.full(len(points), np.nan, dtype=float)
    if len(points) >= 4:
        third = np.column_stack([np.gradient(second[:, a], path, edge_order=2) for a in range(3)])
        denominator = cross_norm**2
        stable = denominator > 1.0e-10
        torsion[stable] = np.einsum("ij,ij->i", cross[stable], third[stable]) / denominator[stable]
    return curvature, torsion


@dataclass(frozen=True)
class VesselCoordinates:
    """Vessel-space representation of physical points."""

    path_mm: np.ndarray
    radial_mm: np.ndarray
    angle_rad: np.ndarray
    axial_residual_mm: np.ndarray
    distance_to_centerline_mm: np.ndarray
    segment_index: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return columns ``path_mm, radial_mm, angle_rad``."""
        return np.column_stack((self.path_mm, self.radial_mm, self.angle_rad))


@dataclass(frozen=True)
class VesselCoordinateSystem:
    """A resampled centreline and its deterministic transported frames."""

    vessel_id: str
    points_xyz: np.ndarray
    path_mm: np.ndarray
    tangents_xyz: np.ndarray
    normals_xyz: np.ndarray
    binormals_xyz: np.ndarray
    curvature_per_mm: np.ndarray
    torsion_per_mm: np.ndarray
    resample_interval_mm: float
    branch_path_mm: np.ndarray
    anatomical_orientation: str | None = None

    @classmethod
    def from_polyline(
        cls,
        points_xyz: np.ndarray | Iterable[Iterable[float]],
        *,
        vessel_id: str = "unknown",
        interval_mm: float = 1.0,
        initial_normal_xyz: np.ndarray | None = None,
        branch_points_xyz: np.ndarray | None = None,
        anatomical_orientation: str | None = None,
    ) -> "VesselCoordinateSystem":
        points, path = resample_polyline(np.asarray(points_xyz, dtype=float), interval_mm)
        tangents = stable_tangents(points, path)
        normals, binormals = rotation_minimizing_frames(tangents, initial_normal_xyz)
        curvature, torsion = curvature_and_torsion(points, path)
        system = cls(
            vessel_id=str(vessel_id),
            points_xyz=points,
            path_mm=path,
            tangents_xyz=tangents,
            normals_xyz=normals,
            binormals_xyz=binormals,
            curvature_per_mm=curvature,
            torsion_per_mm=torsion,
            resample_interval_mm=float(interval_mm),
            branch_path_mm=np.empty(0, dtype=float),
            anatomical_orientation=anatomical_orientation,
        )
        if branch_points_xyz is None:
            return system
        branches = np.asarray(branch_points_xyz, dtype=float)
        if branches.size == 0:
            return system
        branches = branches.reshape((-1, 3))
        branch_path = np.unique(np.round(system.physical_to_vessel(branches).path_mm, decimals=8))
        return cls(**{**system.__dict__, "branch_path_mm": branch_path})

    @property
    def length_mm(self) -> float:
        return float(self.path_mm[-1])

    def frame_at(
        self, path_mm: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate centre point, tangent, first normal, and second normal."""
        requested = np.atleast_1d(np.asarray(path_mm, dtype=float))
        if not np.isfinite(requested).all():
            raise ValueError("path_mm must be finite.")
        clipped = np.clip(requested, 0.0, self.length_mm)
        upper = np.searchsorted(self.path_mm, clipped, side="right")
        upper = np.clip(upper, 1, len(self.path_mm) - 1)
        lower = upper - 1
        span = self.path_mm[upper] - self.path_mm[lower]
        weight = np.divide(clipped - self.path_mm[lower], span, out=np.zeros_like(clipped), where=span > 0)
        weight_col = weight[:, None]
        centres = self.points_xyz[lower] * (1.0 - weight_col) + self.points_xyz[upper] * weight_col
        tangents = self.tangents_xyz[lower] * (1.0 - weight_col) + self.tangents_xyz[upper] * weight_col
        normals = self.normals_xyz[lower] * (1.0 - weight_col) + self.normals_xyz[upper] * weight_col
        for index in range(len(clipped)):
            tangents[index] = _normalize(tangents[index])
            normals[index] -= np.dot(normals[index], tangents[index]) * tangents[index]
            normals[index] = _normalize(normals[index], fallback=self.normals_xyz[lower[index]])
        binormals = np.cross(tangents, normals)
        binormals = np.vstack([_normalize(value) for value in binormals])
        normals = np.cross(binormals, tangents)
        return centres, tangents, normals, binormals

    def vessel_to_physical(
        self,
        path_mm: float | np.ndarray,
        radial_mm: float | np.ndarray = 0.0,
        angle_rad: float | np.ndarray = 0.0,
        axial_residual_mm: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Map vessel coordinates back to physical x/y/z millimetres."""
        path, radius, angle, axial = np.broadcast_arrays(
            np.asarray(path_mm, dtype=float),
            np.asarray(radial_mm, dtype=float),
            np.asarray(angle_rad, dtype=float),
            np.asarray(axial_residual_mm, dtype=float),
        )
        flat_path = path.ravel()
        centres, tangents, normals, binormals = self.frame_at(flat_path)
        r = radius.ravel()[:, None]
        theta = angle.ravel()[:, None]
        physical = (
            centres
            + axial.ravel()[:, None] * tangents
            + r * np.cos(theta) * normals
            + r * np.sin(theta) * binormals
        )
        return physical.reshape(path.shape + (3,))

    def _nearest_segments(self, physical_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = self.points_xyz[:-1]
        vectors = self.points_xyz[1:] - starts
        lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
        chosen = np.empty(len(physical_xyz), dtype=int)
        fractions = np.empty(len(physical_xyz), dtype=float)
        projections = np.empty_like(physical_xyz)
        # A full vectorized search is exact and inexpensive for normal carotid
        # centrelines.  Chunking bounds memory for dense volume transforms.
        segment_count = len(starts)
        chunk_size = max(1, int(2_000_000 / max(segment_count, 1)))
        for begin in range(0, len(physical_xyz), chunk_size):
            end = min(begin + chunk_size, len(physical_xyz))
            delta = physical_xyz[begin:end, None, :] - starts[None, :, :]
            fraction = np.einsum("mij,ij->mi", delta, vectors) / lengths_sq[None, :]
            fraction = np.clip(fraction, 0.0, 1.0)
            candidate = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
            distance_sq = np.sum((physical_xyz[begin:end, None, :] - candidate) ** 2, axis=2)
            local = np.argmin(distance_sq, axis=1)
            rows = np.arange(end - begin)
            chosen[begin:end] = local
            fractions[begin:end] = fraction[rows, local]
            projections[begin:end] = candidate[rows, local]
        return chosen, fractions, projections

    def physical_to_vessel(self, physical_xyz: np.ndarray | Iterable[Iterable[float]]) -> VesselCoordinates:
        """Project physical points into longitudinal/radial/angular coordinates."""
        original = np.asarray(physical_xyz, dtype=float)
        if original.shape == (3,):
            points = original.reshape((1, 3))
        elif original.ndim >= 2 and original.shape[-1] == 3:
            points = original.reshape((-1, 3))
        else:
            raise ValueError("physical_xyz must end in an x/y/z dimension of length 3.")
        if not np.isfinite(points).all():
            raise ValueError("physical_xyz must be finite.")
        segment, fraction, projection = self._nearest_segments(points)
        segment_lengths = self.path_mm[segment + 1] - self.path_mm[segment]
        path = self.path_mm[segment] + fraction * segment_lengths
        centres, tangents, normals, binormals = self.frame_at(path)
        # Use the interpolated centre, equivalent to the segment projection for
        # the resampled piecewise-linear backbone.
        delta = points - centres
        first = np.einsum("ij,ij->i", delta, normals)
        second = np.einsum("ij,ij->i", delta, binormals)
        axial = np.einsum("ij,ij->i", delta, tangents)
        radial = np.hypot(first, second)
        angle = np.mod(np.arctan2(second, first), 2.0 * np.pi)
        distance = np.linalg.norm(points - projection, axis=1)
        return VesselCoordinates(
            path_mm=path,
            radial_mm=radial,
            angle_rad=angle,
            axial_residual_mm=axial,
            distance_to_centerline_mm=distance,
            segment_index=segment,
        )

    def distance_to_bifurcation_mm(self, path_mm: np.ndarray | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(path_mm, dtype=float))
        if self.branch_path_mm.size == 0:
            return np.full(values.shape, np.inf, dtype=float)
        return np.min(np.abs(values[:, None] - self.branch_path_mm[None, :]), axis=1)

    def bifurcation_samples(self, exclusion_half_length_mm: float) -> np.ndarray:
        if exclusion_half_length_mm < 0:
            raise ValueError("exclusion_half_length_mm must be non-negative.")
        if self.branch_path_mm.size == 0:
            return np.zeros(len(self.path_mm), dtype=bool)
        return self.distance_to_bifurcation_mm(self.path_mm) <= float(exclusion_half_length_mm)


def round_trip_error_mm(
    coordinate_system: VesselCoordinateSystem,
    physical_xyz: np.ndarray,
) -> np.ndarray:
    """Return per-point physical error after physical -> vessel -> physical."""
    coordinates = coordinate_system.physical_to_vessel(physical_xyz)
    recovered = coordinate_system.vessel_to_physical(
        coordinates.path_mm,
        coordinates.radial_mm,
        coordinates.angle_rad,
        coordinates.axial_residual_mm,
    )
    return np.linalg.norm(np.asarray(physical_xyz, dtype=float).reshape((-1, 3)) - recovered, axis=1)


__all__ = [
    "VesselCoordinateSystem",
    "VesselCoordinates",
    "cumulative_path_length",
    "curvature_and_torsion",
    "resample_polyline",
    "rotation_minimizing_frames",
    "round_trip_error_mm",
    "stable_tangents",
]
