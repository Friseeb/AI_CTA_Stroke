"""Stage 2: Automatic extremal point detection."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, label


def detect_extremal_points(
    binary_mask: np.ndarray,
    min_distance_value: float = 2.0,
    distance_map: np.ndarray | None = None,
) -> dict:
    """Detect start/endpoints at vessel terminations.

    Uses distance map maxima to identify endpoints automatically.

    Parameters
    ----------
    binary_mask : np.ndarray
        3D binary vessel mask
    min_distance_value : float
        Minimum distance value to consider as extremal point

    Returns
    -------
    dict
        'extremal_points': list of dicts with id, position, distance_value
        'distance_map': full distance transform
        'num_extremal_points': count of detected points
    """
    # Compute distance map if not provided
    if distance_map is None:
        distance_map = distance_transform_edt(binary_mask)

    # Find local maxima (peaks in distance map)
    local_max = (distance_map == maximum_filter(distance_map, size=5)) & binary_mask

    # Filter by minimum distance
    local_max = local_max & (distance_map >= min_distance_value)

    # Label connected components
    labeled_array, num_features = label(local_max)

    # Extract coordinates of peaks
    extremal_points = []
    for i in range(1, num_features + 1):
        coords = np.where(labeled_array == i)
        if len(coords[0]) == 0:
            continue

        centroid = np.array([np.mean(c) for c in coords])
        max_dist = distance_map[coords].max()

        extremal_points.append({
            'id': i,
            'position': centroid,
            'distance_value': float(max_dist),
            'voxel_count': len(coords[0]),
        })

    return {
        'extremal_points': extremal_points,
        'distance_map': distance_map,
        'num_extremal_points': len(extremal_points),
        'local_maxima_mask': local_max,
    }
