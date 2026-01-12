"""Stage 6: Bifurcation detection via tube containment."""
from __future__ import annotations

import numpy as np


def detect_bifurcations(
    centerlines: dict,
    contact_distance_threshold: float = 1.0,
) -> dict:
    """Detect bifurcation points where vessel segments intersect.

    Uses tube containment: bifurcation occurs where
    distance(centerline_1, centerline_2) < radius_1 + radius_2.

    Parameters
    ----------
    centerlines : dict
        Centerlines with radii (from Stage 5)
    contact_distance_threshold : float
        Maximum distance for contact (mm)

    Returns
    -------
    dict
        'bifurcations': list of detected bifurcations
        'num_bifurcations': count
    """
    bifurcations = []
    centerline_list = list(centerlines.items())

    for i, (seg_id_1, seg_1) in enumerate(centerline_list):
        for j, (seg_id_2, seg_2) in enumerate(centerline_list):
            if i >= j:
                continue

            path_1 = seg_1['path']
            path_2 = seg_2['path']
            radii_1 = np.array(seg_1['radii'])
            radii_2 = np.array(seg_2['radii'])

            # Find closest point pair between segments
            min_dist = np.inf
            best_pair = None

            for k, p1 in enumerate(path_1):
                for l, p2 in enumerate(path_2):
                    dist = np.linalg.norm(p1 - p2)

                    # Check tube containment condition
                    sum_radii = radii_1[k] + radii_2[l]
                    if dist < sum_radii and dist < contact_distance_threshold:
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (k, l, p1, p2)

            if best_pair is not None:
                k, l, p1, p2 = best_pair
                bifurcations.append({
                    'segment_1': seg_id_1,
                    'segment_2': seg_id_2,
                    'point_indices': (int(k), int(l)),
                    'location': ((p1 + p2) / 2.0).tolist(),
                    'contact_distance': float(min_dist),
                    'sum_radii': float(radii_1[k] + radii_2[l]),
                })

    return {
        'bifurcations': bifurcations,
        'num_bifurcations': len(bifurcations),
    }
