"""Stage 4: Eikonal equation solver & shortest path tracing."""
from __future__ import annotations

import numpy as np


def extract_centerlines_via_eikonal(
    distance_map: np.ndarray,
    extremal_points: list[dict],
    step_size: float = 0.1,
    max_iterations: int = 5000,
) -> dict:
    """Extract centerlines via gradient descent on distance map.

    Uses the distance_map as a proxy for the Eikonal solution φ.
    Paths are extracted by following -∇φ from endpoints back to startpoints.

    Parameters
    ----------
    extremal_points : list[dict]
        List of extremal points with 'position' and 'id'
    distance_map : np.ndarray
        Precomputed distance transform
    step_size : float
        Gradient descent step size (mm)
    max_iterations : int
        Maximum iterations per path

    Returns
    -------
    dict
        'centerlines': dict mapping segment_id -> {path, length}
        'num_centerlines': count of extracted centerlines
    """

    def gradient_at_point(distance_map: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Compute gradient at point via central differences."""
        x, y, z = point
        xi, yi, zi = int(np.clip(x, 0, distance_map.shape[0] - 1)), \
                     int(np.clip(y, 0, distance_map.shape[1] - 1)), \
                     int(np.clip(z, 0, distance_map.shape[2] - 1))

        # Central differences
        grad_x = (distance_map[min(xi + 1, distance_map.shape[0] - 1), yi, zi] -
                  distance_map[max(xi - 1, 0), yi, zi]) / 2.0
        grad_y = (distance_map[xi, min(yi + 1, distance_map.shape[1] - 1), zi] -
                  distance_map[xi, max(yi - 1, 0), zi]) / 2.0
        grad_z = (distance_map[xi, yi, min(zi + 1, distance_map.shape[2] - 1)] -
                  distance_map[xi, yi, max(zi - 1, 0)]) / 2.0

        return np.array([grad_x, grad_y, grad_z])

    def extract_path(
        start_point: np.ndarray,
        end_point: np.ndarray,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        """Trace path from end to start via gradient descent."""
        path = [np.array(end_point, dtype=float)]
        current = np.array(end_point, dtype=float)

        for _ in range(max_iterations):
            if np.linalg.norm(current - start_point) < 1.0:
                break

            grad = gradient_at_point(distance_map, current)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < 1e-6:
                break

            grad_unit = grad / grad_norm
            current = current - step_size * grad_unit
            path.append(current.copy())

        if len(path) < 2:
            # Ensure at least two points for downstream processing/tests
            path.append(np.array(start_point, dtype=float))
        return np.array(path)

    # Extract paths for all extremal point pairs
    centerlines = {}
    count = 0

    for i, ep_i in enumerate(extremal_points):
        for j, ep_j in enumerate(extremal_points):
            if i >= j:
                continue

            start = ep_i['position']
            end = ep_j['position']

            path = extract_path(start, end, distance_map)
            length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

            seg_id = f'segment_{i}_{j}'
            centerlines[seg_id] = {
                'path': path,
                'length': float(length),
                'num_points': len(path),
                'start_point_id': i,
                'end_point_id': j,
            }
            count += 1

    return {
        'centerlines': centerlines,
        'num_centerlines': count,
    }
