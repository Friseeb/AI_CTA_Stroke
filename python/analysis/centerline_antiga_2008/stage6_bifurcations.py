"""Stage 6: Bifurcation detection via tube containment (optimized with KD-tree)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False
    def tqdm(x, **kwargs):
        return x


def detect_bifurcations(
    centerlines: dict,
    contact_distance_threshold: float = 1.0,
    subsample_step: int = None,
) -> dict:
    """Detect bifurcation points where vessel segments intersect.

    Uses KD-tree for O(n log n) spatial queries instead of O(n²) brute force.
    Bifurcation occurs where distance(centerline_1, centerline_2) < radius_1 + radius_2.

    Parameters
    ----------
    centerlines : dict
        Centerlines with radii (from Stage 5)
    contact_distance_threshold : float
        Maximum distance for contact (mm)
    subsample_step : int, optional
        Subsample centerline points (take every Nth point) to reduce computation.
        If None, auto-determines based on total point count.

    Returns
    -------
    dict
        'bifurcations': list of detected bifurcations
        'num_bifurcations': count
    """
    if not centerlines:
        return {'bifurcations': [], 'num_bifurcations': 0}

    # Count total points to determine subsampling
    total_points = sum(len(seg_data['path']) for seg_data in centerlines.values())
    
    # Auto-determine subsampling: aim for ~500K points max for reasonable speed
    if subsample_step is None:
        if total_points > 2_000_000:
            subsample_step = max(2, total_points // 500_000)
        elif total_points > 500_000:
            subsample_step = 2
        else:
            subsample_step = 1
    
    if subsample_step > 1:
        print(f"  Subsampling centerlines by factor {subsample_step} ({total_points:,} -> ~{total_points//subsample_step:,} points)")

    # Build global point array with segment/index metadata
    all_points = []
    point_metadata = []  # (seg_id, point_idx, radius)
    
    for seg_id, seg_data in centerlines.items():
        path = np.array(seg_data['path'])
        radii = np.array(seg_data['radii'])
        # Subsample but always include endpoints for connectivity
        indices = list(range(0, len(path), subsample_step))
        if len(path) > 1 and (len(path) - 1) not in indices:
            indices.append(len(path) - 1)
        for idx in indices:
            all_points.append(path[idx])
            point_metadata.append((seg_id, idx, radii[idx]))
    
    if len(all_points) < 2:
        return {'bifurcations': [], 'num_bifurcations': 0}
    
    all_points = np.array(all_points)
    
    # Build KD-tree
    print(f"  Building KD-tree for {len(all_points):,} centerline points...")
    tree = cKDTree(all_points)
    
    # Find candidate pairs - use contact threshold as search radius
    # Points farther than this can't be in contact regardless of radii
    search_radius = contact_distance_threshold
    
    print(f"  Querying pairs within {search_radius:.1f} mm...")
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')
    print(f"  Found {len(pairs):,} candidate point pairs to check")
    
    # Check each candidate pair
    bifurcation_map = {}  # (seg1, seg2) -> best bifurcation
    
    if _TQDM_OK and len(pairs) > 1000:
        pairs_iter = tqdm(pairs, desc='Checking bifurcations', unit='pair')
    else:
        pairs_iter = pairs
    
    for i, j in pairs_iter:
        seg_id_1, idx_1, r1 = point_metadata[i]
        seg_id_2, idx_2, r2 = point_metadata[j]
        
        # Skip same segment
        if seg_id_1 == seg_id_2:
            continue
        
        # Normalize segment pair key
        if seg_id_1 > seg_id_2:
            seg_id_1, seg_id_2 = seg_id_2, seg_id_1
            idx_1, idx_2 = idx_2, idx_1
            r1, r2 = r2, r1
            i, j = j, i
        
        p1 = all_points[i]
        p2 = all_points[j]
        dist = np.linalg.norm(p1 - p2)
        
        # Check tube containment
        sum_radii = r1 + r2
        if dist < sum_radii and dist < contact_distance_threshold:
            key = (seg_id_1, seg_id_2)
            if key not in bifurcation_map or dist < bifurcation_map[key]['contact_distance']:
                bifurcation_map[key] = {
                    'segment_1': seg_id_1,
                    'segment_2': seg_id_2,
                    'point_indices': (int(idx_1), int(idx_2)),
                    'location': ((p1 + p2) / 2.0).tolist(),
                    'contact_distance': float(dist),
                    'sum_radii': float(sum_radii),
                }
    
    bifurcations = list(bifurcation_map.values())
    print(f"  Detected {len(bifurcations):,} bifurcations")

    return {
        'bifurcations': bifurcations,
        'num_bifurcations': len(bifurcations),
    }
