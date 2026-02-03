"""Stage 2: Automatic extremal point detection."""
from __future__ import annotations

import importlib.util
import platform
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, label

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt

    _CUPY_OK = True
except Exception:  # noqa: BLE001
    _CUPY_OK = False

_DT_SPEC = importlib.util.find_spec("distance_transforms") is not None

try:
    from .edt_julia_cli import distance_transform_julia_cli
except Exception:  # noqa: BLE001
    distance_transform_julia_cli = None


def _distance_transform_distance_transforms(mask: np.ndarray) -> np.ndarray | None:
    if not _DT_SPEC:
        return None
    try:
        from distance_transforms import transform as dt_transform

        m = mask.astype(np.uint8)
        dist_sq = dt_transform(m)
        if dist_sq is None:
            return None
        dist_sq = np.asarray(dist_sq)
        if np.max(dist_sq) < 1e-6:
            dist_sq = None

        if dist_sq is not None and np.any(m > 0):
            dist_inside_max = float(dist_sq[m > 0].max())
            if dist_inside_max < 1e-6:
                dist_sq = None

        if dist_sq is None:
            dist_sq = dt_transform(1 - m)
            if dist_sq is None:
                return None
            dist_sq = np.asarray(dist_sq)

        return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)
    except Exception:
        return None


def _distance_transform_distance_transforms_cuda(mask: np.ndarray) -> np.ndarray | None:
    if not _DT_SPEC:
        return None
    try:
        import torch
        from distance_transforms import transform_cuda

        tensor = torch.as_tensor(mask.astype(np.uint8), device='cuda')
        dist_sq = transform_cuda(tensor)
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0)).cpu().numpy()
        return dist.astype(np.float32)
    except Exception:
        return None


def _maximum_filter_torch_separable(
    distance_map: np.ndarray,
    filt_size: int,
    device: str,
) -> tuple[np.ndarray | None, str]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return None, "torch_unavailable"

    if device == "mps":
        if not torch.backends.mps.is_available():
            return None, "mps_unavailable"
    elif device == "cuda":
        if not torch.cuda.is_available():
            return None, "cuda_unavailable"
    else:
        return None, "unsupported_device"

    k = int(filt_size)
    if k <= 1:
        return distance_map.astype(np.float32, copy=False), f"torch_{device}"

    pad = k // 2
    shape = distance_map.shape
    if min(shape) <= pad:
        return None, "pad_too_large"

    t = torch.as_tensor(distance_map, dtype=torch.float32, device=device)

    def max_filter_axis(tensor: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == 0:
            perm = tensor.permute(1, 2, 0).contiguous()
            flat = perm.view(-1, 1, tensor.shape[0])
        elif axis == 1:
            perm = tensor.permute(0, 2, 1).contiguous()
            flat = perm.view(-1, 1, tensor.shape[1])
        else:
            flat = tensor.contiguous().view(-1, 1, tensor.shape[2])

        if pad > 0:
            flat = F.pad(flat, (pad, pad), mode="reflect")
        flat = F.max_pool1d(flat, kernel_size=k, stride=1, padding=0)

        if axis == 0:
            return flat.view(tensor.shape[1], tensor.shape[2], tensor.shape[0]).permute(2, 0, 1)
        if axis == 1:
            return flat.view(tensor.shape[0], tensor.shape[2], tensor.shape[1]).permute(0, 2, 1)
        return flat.view(tensor.shape[0], tensor.shape[1], tensor.shape[2])

    for axis in (0, 1, 2):
        t = max_filter_axis(t, axis)

    return t.cpu().numpy(), f"torch_{device}"


def detect_extremal_points(
    binary_mask: np.ndarray,
    min_distance_value: float = 2.0,
    distance_map: np.ndarray | None = None,
    edt_backend: str = 'auto',
    gpu_backend: str | None = None,
    max_filter_backend: str = 'cpu',
    retry_if_empty: bool = True,
    skeleton_fallback: bool = True,
    allow_cpu_edt: bool = True,
) -> dict:
    """Detect start/endpoints at vessel terminations.

    Uses distance map maxima to identify endpoints automatically.

    Parameters
    ----------
    binary_mask : np.ndarray
        3D binary vessel mask
    min_distance_value : float
        Minimum distance value to consider as extremal point
    max_filter_backend : str
        Backend for local-max filter: 'cpu', 'auto', 'mps', or 'cuda'
    allow_cpu_edt : bool
        Allow CPU EDT fallback when GPU backend is unavailable

    Returns
    -------
    dict
        'extremal_points': list of dicts with id, position, distance_value
        'distance_map': full distance transform
        'num_extremal_points': count of detected points
    """
    # Compute distance map if not provided
    if distance_map is None:
        backend = (edt_backend or 'auto').lower()

        if backend in ('auto', 'metal') and platform.system() == 'Darwin':
            if distance_transform_julia_cli is not None:
                distance_map = distance_transform_julia_cli(binary_mask, backend=backend)

        if distance_map is None and backend == 'cuda':
            distance_map = _distance_transform_distance_transforms_cuda(binary_mask)

        if distance_map is None and backend in ('auto', 'cuda') and gpu_backend == 'cuda' and _CUPY_OK:
            distance_map = cp_distance_transform_edt(cp.asarray(binary_mask))
            distance_map = cp.asnumpy(distance_map)

        if distance_map is None:
            if not allow_cpu_edt:
                raise RuntimeError(
                    "EDT CPU fallback disabled; no GPU EDT backend available. "
                    "Install distance_transforms for Metal on macOS or use CUDA/CuPy."
                )
            distance_map = distance_transform_edt(binary_mask)

    def _extract_points(
        dm: np.ndarray,
        mask: np.ndarray,
        min_val: float,
        filt_size: int = 5,
        max_backend: str = "cpu",
    ):
        backend_used = "scipy"
        dm_max = None
        dm_source = dm
        backend = (max_backend or "cpu").lower()

        if backend == "auto":
            for device in ("mps", "cuda"):
                dm_source = dm.astype(np.float32, copy=False)
                dm_max, backend_used = _maximum_filter_torch_separable(dm_source, filt_size, device)
                if dm_max is not None:
                    break
        elif backend in ("mps", "cuda"):
            dm_source = dm.astype(np.float32, copy=False)
            dm_max, backend_used = _maximum_filter_torch_separable(dm_source, filt_size, backend)

        if dm_max is None:
            dm_max = maximum_filter(dm, size=filt_size)
            dm_source = dm
            backend_used = "scipy"

        local_max = (dm_source == dm_max) & mask
        local_max = local_max & (dm >= min_val)
        labeled_array, num_features = label(local_max)
        points = []
        for i in range(1, num_features + 1):
            coords = np.where(labeled_array == i)
            if len(coords[0]) == 0:
                continue
            centroid = np.array([np.mean(c) for c in coords])
            max_dist = dm[coords].max()
            points.append({
                'id': i,
                'position': centroid,
                'distance_value': float(max_dist),
                'voxel_count': len(coords[0]),
            })
        return points, local_max, backend_used

    extremal_points, local_max, max_filter_backend_used = _extract_points(
        distance_map,
        binary_mask,
        min_distance_value,
        filt_size=5,
        max_backend=max_filter_backend,
    )

    # Retry with relaxed params if empty
    if retry_if_empty and len(extremal_points) == 0:
        extremal_points, local_max, max_filter_backend_used = _extract_points(
            distance_map,
            binary_mask,
            min_distance_value * 0.5,
            filt_size=7,
            max_backend=max_filter_backend,
        )

    # Skeleton-based fallback to generate seeds when distance map peaks vanish
    if skeleton_fallback and len(extremal_points) == 0:
        try:
            from skimage.morphology import skeletonize_3d

            skel = skeletonize_3d(binary_mask.astype(np.uint8) > 0)
            skel_pts = np.argwhere(skel > 0)
            for idx, pt in enumerate(skel_pts):
                extremal_points.append({
                    'id': idx + 1,
                    'position': pt.astype(float),
                    'distance_value': float(distance_map[tuple(pt)] if distance_map is not None else 0.0),
                    'voxel_count': 1,
                    'source': 'skeleton',
                })
            local_max = skel
        except Exception:
            pass

    return {
        'extremal_points': extremal_points,
        'distance_map': distance_map,
        'num_extremal_points': len(extremal_points),
        'local_maxima_mask': local_max,
        'max_filter_backend': max_filter_backend_used,
    }
