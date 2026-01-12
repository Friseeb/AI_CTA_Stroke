"""Stage 5: Maximal inscribed sphere radius computation."""
from __future__ import annotations

import numpy as np


def compute_radii(
    centerlines: dict,
    distance_map: np.ndarray,
) -> dict:
    """Compute vessel radius at each centerline point.

    Radius = distance from centerline point to nearest wall.
    Uses trilinear interpolation of the distance map.

    Parameters
    ----------
    centerlines : dict
        Centerlines from Stage 4, each with 'path'
    distance_map : np.ndarray
        Distance transform from Stage 2

    Returns
    -------
    dict
        'centerlines_with_radii': updated centerlines with 'radii', 'mean_radius'
    """

    def trilinear_interp(distance_map: np.ndarray, point: np.ndarray) -> float:
        """Trilinear interpolation of distance map at point."""
        x, y, z = point

        x_floor = int(np.floor(x))
        y_floor = int(np.floor(y))
        z_floor = int(np.floor(z))

        # Clamp to bounds
        x_floor = np.clip(x_floor, 0, distance_map.shape[0] - 2)
        y_floor = np.clip(y_floor, 0, distance_map.shape[1] - 2)
        z_floor = np.clip(z_floor, 0, distance_map.shape[2] - 2)

        dx = x - x_floor
        dy = y - y_floor
        dz = z - z_floor

        # Corner values
        v000 = distance_map[x_floor, y_floor, z_floor]
        v001 = distance_map[x_floor, y_floor, z_floor + 1]
        v010 = distance_map[x_floor, y_floor + 1, z_floor]
        v011 = distance_map[x_floor, y_floor + 1, z_floor + 1]
        v100 = distance_map[x_floor + 1, y_floor, z_floor]
        v101 = distance_map[x_floor + 1, y_floor, z_floor + 1]
        v110 = distance_map[x_floor + 1, y_floor + 1, z_floor]
        v111 = distance_map[x_floor + 1, y_floor + 1, z_floor + 1]

        # Trilinear interpolation
        v00 = v000 * (1 - dx) + v100 * dx
        v01 = v001 * (1 - dx) + v101 * dx
        v10 = v010 * (1 - dx) + v110 * dx
        v11 = v011 * (1 - dx) + v111 * dx

        v0 = v00 * (1 - dy) + v10 * dy
        v1 = v01 * (1 - dy) + v11 * dy

        return float(v0 * (1 - dz) + v1 * dz)

    # Compute radii for each centerline
    centerlines_with_radii = {}

    for seg_id, seg_data in centerlines.items():
        path = seg_data['path']
        radii = []

        for point in path:
            radius = trilinear_interp(distance_map, point)
            radii.append(radius)

        radii = np.array(radii)
        seg_data['radii'] = radii.tolist()
        seg_data['mean_radius'] = float(np.mean(radii))
        seg_data['min_radius'] = float(np.min(radii))
        seg_data['max_radius'] = float(np.max(radii))

        centerlines_with_radii[seg_id] = seg_data

    return {
        'centerlines': centerlines_with_radii,
        'centerlines_with_radii': centerlines_with_radii,
    }
