"""Approximate centerline utilities for version 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import voxel_to_physical


def approximate_centerline_by_slices(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    reference_image: object | None = None,
    smooth_sigma_slices: float = 1.0,
) -> pd.DataFrame:
    """Estimate a coarse centerline from per-slice centers of mass.

    This is intentionally labelled approximate. It is useful for early QC and
    slice-wise geometry, and is designed to be replaced by graph/skeleton logic.
    """
    binary = np.asarray(mask, dtype=bool)
    points_zyx: list[tuple[float, float, float]] = []
    for z in np.where(binary.any(axis=(1, 2)))[0]:
        ys, xs = np.where(binary[z])
        if ys.size == 0:
            continue
        points_zyx.append((float(z), float(ys.mean()), float(xs.mean())))

    if not points_zyx:
        return pd.DataFrame(
            columns=[
                "case_id",
                "point_index",
                "x",
                "y",
                "z",
                "segment_label",
                "tangent_x",
                "tangent_y",
                "tangent_z",
                "curvature",
                "centerline_method",
            ]
        )

    zyx = np.asarray(points_zyx, dtype=float)
    if len(zyx) >= 3 and smooth_sigma_slices > 0:
        zyx[:, 1] = _smooth_1d(zyx[:, 1], smooth_sigma_slices)
        zyx[:, 2] = _smooth_1d(zyx[:, 2], smooth_sigma_slices)

    physical = voxel_to_physical(np.rint(zyx).astype(int), reference_image, spacing_xyz)
    tangent = _unit_vectors(np.gradient(physical, axis=0))
    curvature = _curvature(physical)

    rows = []
    for idx, xyz in enumerate(physical):
        rows.append(
            {
                "case_id": case_id,
                "point_index": idx,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "segment_label": "whole_aorta_approx",
                "tangent_x": float(tangent[idx, 0]),
                "tangent_y": float(tangent[idx, 1]),
                "tangent_z": float(tangent[idx, 2]),
                "curvature": float(curvature[idx]),
                "centerline_method": "slice_center_of_mass_approximate",
            }
        )
    return pd.DataFrame(rows)


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors, dtype=float)
    valid = norms > 0
    out[valid] = vectors[valid] / norms[valid, None]
    return out


def _curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(len(points), dtype=float)
    first = np.gradient(points, axis=0)
    second = np.gradient(first, axis=0)
    numerator = np.linalg.norm(np.cross(first, second), axis=1)
    denominator = np.linalg.norm(first, axis=1) ** 3
    curvature = np.zeros(len(points), dtype=float)
    valid = denominator > 0
    curvature[valid] = numerator[valid] / denominator[valid]
    return curvature


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(values, sigma, mode="nearest")
    except Exception:
        radius = max(1, int(round(sigma * 2)))
        padded = np.pad(values, radius, mode="edge")
        kernel = np.ones(radius * 2 + 1, dtype=float)
        kernel /= kernel.sum()
        return np.convolve(padded, kernel, mode="valid")
