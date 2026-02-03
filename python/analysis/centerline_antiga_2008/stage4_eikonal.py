"""Stage 4: Eikonal equation solver & shortest path tracing."""
from __future__ import annotations

import numpy as np

try:
    from tqdm import tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False
    def tqdm(x, **kwargs):
        return x

try:
    import cupy as cp

    _CUPY_OK = True
except Exception:  # noqa: BLE001
    _CUPY_OK = False



def extract_centerlines_via_eikonal(
    distance_map: np.ndarray,
    extremal_points: list[dict],
    step_size: float = 0.1,
    max_iterations: int = 5000,
    gpu_backend: str | None = None,
    k_nearest: int | None = None,
    max_pair_distance: float | None = None,
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
    k_nearest : int | None
        Limit tracing to k nearest extremal points per point (reduces pair count)
    max_pair_distance : float | None
        Limit tracing to pairs within this distance (voxel units)

    Returns
    -------
    dict
        'centerlines': dict mapping segment_id -> {path, length}
        'num_centerlines': count of extracted centerlines
    """
    torch = None
    if gpu_backend in ('cuda', 'mps'):
        try:
            import torch as torch_mod

            torch = torch_mod
        except Exception:  # noqa: BLE001
            torch = None

    def gradient_np(dm: np.ndarray, point: np.ndarray) -> np.ndarray:
        x, y, z = point
        xi, yi, zi = int(np.clip(x, 0, dm.shape[0] - 1)), int(np.clip(y, 0, dm.shape[1] - 1)), int(np.clip(z, 0, dm.shape[2] - 1))
        grad_x = (dm[min(xi + 1, dm.shape[0] - 1), yi, zi] - dm[max(xi - 1, 0), yi, zi]) * 0.5
        grad_y = (dm[xi, min(yi + 1, dm.shape[1] - 1), zi] - dm[xi, max(yi - 1, 0), zi]) * 0.5
        grad_z = (dm[xi, yi, min(zi + 1, dm.shape[2] - 1)] - dm[xi, yi, max(zi - 1, 0)]) * 0.5
        return np.array([grad_x, grad_y, grad_z])

    def gradient_cp(dm: cp.ndarray, point: cp.ndarray) -> cp.ndarray:
        x, y, z = point
        xi = cp.clip(cp.floor(x), 0, dm.shape[0] - 1).astype(cp.int64)
        yi = cp.clip(cp.floor(y), 0, dm.shape[1] - 1).astype(cp.int64)
        zi = cp.clip(cp.floor(z), 0, dm.shape[2] - 1).astype(cp.int64)
        grad_x = (dm[cp.minimum(xi + 1, dm.shape[0] - 1), yi, zi] - dm[cp.maximum(xi - 1, 0), yi, zi]) * 0.5
        grad_y = (dm[xi, cp.minimum(yi + 1, dm.shape[1] - 1), zi] - dm[xi, cp.maximum(yi - 1, 0), zi]) * 0.5
        grad_z = (dm[xi, yi, cp.minimum(zi + 1, dm.shape[2] - 1)] - dm[xi, yi, cp.maximum(zi - 1, 0)]) * 0.5
        return cp.stack([grad_x, grad_y, grad_z])

    def extract_path_np(start_point: np.ndarray, end_point: np.ndarray, dm: np.ndarray) -> np.ndarray:
        """Trace path from end_point toward start_point along the distance map ridge.
        
        Uses gradient ascent (+grad) to follow the ridge of the distance transform,
        which corresponds to the vessel centerline.
        """
        bounds = np.array(dm.shape, dtype=float) - 1
        path = [np.array(end_point, dtype=float)]
        current = np.array(end_point, dtype=float)
        prev_dist = float('inf')
        
        for _ in range(max_iterations):
            # Check if we've reached the start point
            if np.linalg.norm(current - start_point) < 1.5:
                path.append(np.array(start_point, dtype=float))
                break
            
            # Check bounds before computing gradient
            if np.any(current < 0) or np.any(current > bounds):
                break
                
            grad = gradient_np(dm, current)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break
            
            # Move along the ridge: interpolate between gradient ascent and direct path to start
            direct = start_point - current
            direct_norm = np.linalg.norm(direct)
            if direct_norm > 1e-6:
                # Blend gradient direction with direct path to ensure convergence
                direction = 0.3 * (grad / grad_norm) + 0.7 * (direct / direct_norm)
                direction = direction / np.linalg.norm(direction)
            else:
                direction = grad / grad_norm
            
            current = current + step_size * direction
            current = np.clip(current, [0, 0, 0], bounds)
            
            # Detect if we're stuck
            dist_to_start = np.linalg.norm(current - start_point)
            if abs(dist_to_start - prev_dist) < 1e-4:
                break
            prev_dist = dist_to_start
            
            path.append(current.copy())
            
        if len(path) < 2:
            path.append(np.array(start_point, dtype=float))
        return np.array(path)

    def extract_path_cp(start_point: np.ndarray, end_point: np.ndarray, dm: cp.ndarray) -> np.ndarray:
        """Trace path from end_point toward start_point along the distance map ridge (CuPy)."""
        path = [np.array(end_point, dtype=float)]
        current = cp.asarray(end_point, dtype=cp.float32)
        start_cp = cp.asarray(start_point, dtype=cp.float32)
        bounds = cp.asarray(dm.shape, dtype=cp.float32) - 1
        prev_dist = float('inf')
        
        for _ in range(max_iterations):
            dist_to_start = float(cp.linalg.norm(current - start_cp))
            if dist_to_start < 1.5:
                path.append(np.array(start_point, dtype=float))
                break
            
            if bool(cp.any(current < 0)) or bool(cp.any(current > bounds)):
                break
                
            grad = gradient_cp(dm, current)
            grad_norm = float(cp.linalg.norm(grad))
            if grad_norm < 1e-6:
                break
            
            # Blend gradient direction with direct path to ensure convergence
            direct = start_cp - current
            direct_norm = float(cp.linalg.norm(direct))
            if direct_norm > 1e-6:
                grad_dir = grad / grad_norm
                direct_dir = direct / direct_norm
                direction = 0.3 * grad_dir + 0.7 * direct_dir
                direction = direction / cp.linalg.norm(direction)
            else:
                direction = grad / grad_norm
            
            current = current + step_size * direction
            current = cp.clip(current, cp.zeros(3, dtype=cp.float32), bounds)
            
            if abs(dist_to_start - prev_dist) < 1e-4:
                break
            prev_dist = dist_to_start
            
            path.append(cp.asnumpy(current))
            
        if len(path) < 2:
            path.append(np.array(start_point, dtype=float))
        return np.array(path)

    if torch is not None:
        def gradient_torch(dm: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
            x, y, z = point
            xi = torch.clamp(torch.floor(x), 0, dm.shape[0] - 1).long()
            yi = torch.clamp(torch.floor(y), 0, dm.shape[1] - 1).long()
            zi = torch.clamp(torch.floor(z), 0, dm.shape[2] - 1).long()
            grad_x = (dm[torch.minimum(xi + 1, torch.tensor(dm.shape[0] - 1, device=dm.device)), yi, zi] -
                      dm[torch.maximum(xi - 1, torch.tensor(0, device=dm.device)), yi, zi]) * 0.5
            grad_y = (dm[xi, torch.minimum(yi + 1, torch.tensor(dm.shape[1] - 1, device=dm.device)), zi] -
                      dm[xi, torch.maximum(yi - 1, torch.tensor(0, device=dm.device)), zi]) * 0.5
            grad_z = (dm[xi, yi, torch.minimum(zi + 1, torch.tensor(dm.shape[2] - 1, device=dm.device))] -
                      dm[xi, yi, torch.maximum(zi - 1, torch.tensor(0, device=dm.device))]) * 0.5
            return torch.stack([grad_x, grad_y, grad_z])

        def extract_path_torch(start_point: np.ndarray, end_point: np.ndarray, dm: torch.Tensor, device: torch.device) -> np.ndarray:
            """Trace path from end_point toward start_point along the distance map ridge (GPU)."""
            path = [np.array(end_point, dtype=float)]
            current = torch.as_tensor(end_point, device=device, dtype=torch.float32)
            start_t = torch.as_tensor(start_point, device=device, dtype=torch.float32)
            bounds = torch.tensor(dm.shape, device=device, dtype=torch.float32) - 1
            prev_dist = float('inf')
            
            for _ in range(max_iterations):
                dist_to_start = float(torch.linalg.norm(current - start_t))
                if dist_to_start < 1.5:
                    path.append(np.array(start_point, dtype=float))
                    break
                
                if bool(torch.any(current < 0)) or bool(torch.any(current > bounds)):
                    break
                    
                grad = gradient_torch(dm, current)
                grad_norm = float(torch.linalg.norm(grad))
                if grad_norm < 1e-6:
                    break
                
                # Blend gradient direction with direct path to ensure convergence
                direct = start_t - current
                direct_norm = float(torch.linalg.norm(direct))
                if direct_norm > 1e-6:
                    grad_dir = grad / grad_norm
                    direct_dir = direct / direct_norm
                    direction = 0.3 * grad_dir + 0.7 * direct_dir
                    direction = direction / torch.linalg.norm(direction)
                else:
                    direction = grad / grad_norm
                
                current = current + step_size * direction
                current = torch.clamp(current, min=0.0, max=None)
                current = torch.minimum(current, bounds)
                
                if abs(dist_to_start - prev_dist) < 1e-4:
                    break
                prev_dist = dist_to_start
                
                path.append(current.detach().cpu().numpy())
                
            if len(path) < 2:
                path.append(np.array(start_point, dtype=float))
            return np.array(path)

    def _build_pairs(
        pts: np.ndarray,
        k_limit: int | None,
        dist_limit: float | None,
    ) -> tuple[list[tuple[int, int]], str]:
        n = pts.shape[0]
        if n < 2:
            return [], "none"

        if k_limit is not None and k_limit < 1:
            k_limit = None
        if dist_limit is not None and dist_limit <= 0:
            dist_limit = None

        if k_limit is None and dist_limit is None:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return pairs, "all"

        mode_parts = []
        if k_limit is not None:
            mode_parts.append(f"knn{k_limit}")
        if dist_limit is not None:
            mode_parts.append(f"maxdist{dist_limit:g}")
        mode = "+".join(mode_parts) if mode_parts else "custom"

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pts)
            pairs: list[tuple[int, int]] = []

            if k_limit is not None:
                kq = min(k_limit + 1, n)
                if dist_limit is not None:
                    dists, idxs = tree.query(pts, k=kq, distance_upper_bound=dist_limit)
                else:
                    dists, idxs = tree.query(pts, k=kq)

                dists = np.asarray(dists)
                idxs = np.asarray(idxs)
                if dists.ndim == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]

                for i in range(n):
                    for dist, j in zip(dists[i], idxs[i]):
                        if j == i or j >= n:
                            continue
                        if dist_limit is not None and dist > dist_limit:
                            continue
                        if j > i:
                            pairs.append((i, int(j)))
                return pairs, mode

            if dist_limit is not None:
                neighbors = tree.query_ball_point(pts, r=dist_limit)
                for i, neigh in enumerate(neighbors):
                    for j in neigh:
                        if j > i:
                            pairs.append((i, int(j)))
                return pairs, mode

        except Exception:
            pass

        # Fallback: brute-force distances
        pairs = []
        for i in range(n - 1):
            dist = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
            idx = np.arange(i + 1, n)
            if dist_limit is not None:
                mask = dist <= dist_limit
                dist = dist[mask]
                idx = idx[mask]
            if k_limit is not None and dist.size > 0:
                order = np.argsort(dist)
                if k_limit < order.size:
                    order = order[:k_limit]
                idx = idx[order]
            for j in idx:
                pairs.append((i, int(j)))
        return pairs, mode

    # Extract paths for extremal point pairs
    centerlines = {}
    count = 0

    use_cp = gpu_backend == 'cuda' and _CUPY_OK
    use_torch = torch is not None
    torch_device = torch.device(gpu_backend) if use_torch else None
    dm_cp = cp.asarray(distance_map) if use_cp else None
    dm_torch = torch.as_tensor(distance_map, device=torch_device, dtype=torch.float32) if use_torch else None

    positions = np.array([ep["position"] for ep in extremal_points], dtype=np.float32)
    pairs, pair_mode = _build_pairs(positions, k_nearest, max_pair_distance)

    pair_iter = pairs
    if _TQDM_OK:
        pair_iter = tqdm(pair_iter, desc='Tracing centerlines', unit='pair', total=len(pairs))

    for i, j in pair_iter:
        ep_i = extremal_points[i]
        ep_j = extremal_points[j]
        start = ep_i['position']
        end = ep_j['position']

        if use_cp:
            path = extract_path_cp(start, end, dm_cp)
        elif use_torch:
            path = extract_path_torch(start, end, dm_torch, torch_device)
        else:
            path = extract_path_np(start, end, distance_map)
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
        'num_pairs': len(pairs),
        'pair_mode': pair_mode,
        'k_nearest': k_nearest,
        'max_pair_distance': max_pair_distance,
    }
