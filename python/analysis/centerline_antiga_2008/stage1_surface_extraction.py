"""Stage 1: Surface extraction and cleaning from binary segmentation."""
from __future__ import annotations

import importlib.util
import platform
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label,
    distance_transform_edt,
)

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
    """Compute EDT via distance_transforms (Julia backend, Metal)."""
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

        # distance_transforms computes distances for zeros in some builds.
        # If the foreground (mask==1) has zero distances, invert the mask.
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
    """Compute EDT via distance_transforms CUDA path (Julia + torch)."""
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


def _compute_distance_map(
    cleaned_mask: np.ndarray,
    edt_backend: str,
    gpu_backend: str | None,
    allow_cpu_fallback: bool = True,
) -> tuple[np.ndarray, str]:
    """Compute distance map using selected backend with fallbacks.

    Returns
    -------
    distance_map : np.ndarray
    backend_used : str
    """
    distance_map = None
    backend_used = 'scipy'

    backend = (edt_backend or 'auto').lower()

    # Metal / macOS path via Julia CLI (avoids juliacall crashes)
    if distance_map is None and backend in ('auto', 'metal'):
        if distance_transform_julia_cli is not None:
            distance_map = distance_transform_julia_cli(cleaned_mask, backend=backend)
            if distance_map is not None:
                backend_used = f'julia_cli_{backend}'

    # distance_transforms CUDA path (Julia CUDA) if requested explicitly
    if distance_map is None and backend == 'cuda':
        distance_map = _distance_transform_distance_transforms_cuda(cleaned_mask)
        if distance_map is not None:
            backend_used = 'distance_transforms_cuda'

    # CuPy CUDA path (Python-native)
    if distance_map is None and backend in ('auto', 'cuda') and gpu_backend == 'cuda' and _CUPY_OK:
        mask_gpu = cp.asarray(cleaned_mask)
        distance_map_gpu = cp_distance_transform_edt(mask_gpu)
        distance_map = cp.asnumpy(distance_map_gpu)
        backend_used = 'cupy'

    # Default SciPy CPU fallback
    if distance_map is None:
        if not allow_cpu_fallback:
            raise RuntimeError(
                "EDT CPU fallback disabled; no GPU EDT backend available. "
                "Install distance_transforms for Metal on macOS or use CUDA/CuPy."
            )
        distance_map = distance_transform_edt(cleaned_mask)
        backend_used = 'scipy'

    return distance_map, backend_used


def extract_surface(
    binary_mask: np.ndarray,
    min_component_size: int = 50,
    erosion_iterations: int = 1,
    dilation_iterations: int = 1,
    thick_component_max_radius: float | None = None,
    edt_backend: str = 'auto',
    gpu_backend: str | None = None,
    allow_cpu_edt: bool = True,
) -> dict:
    """Clean vessel mask, remove small components, and compute distance map.

    Parameters
    ----------
    binary_mask : np.ndarray
        3D binary vessel mask
    min_component_size : int
        Minimum voxel count to keep a connected component
    erosion_iterations : int
        Iterations for binary erosion (cleaning)
    dilation_iterations : int
        Iterations for binary dilation (smoothing)
    allow_cpu_edt : bool
        Allow CPU EDT fallback when GPU backend is unavailable

    Returns
    -------
    dict
        'cleaned_mask': np.ndarray, cleaned binary mask
        'distance_map': np.ndarray, distance transform of cleaned mask
        'num_components': int, number of kept components
    """
    # Ensure boolean array
    mask = binary_mask.astype(bool)

    # Fill small holes
    mask = binary_fill_holes(mask)

    # Morphological cleaning
    if erosion_iterations > 0:
        mask = binary_erosion(mask, iterations=erosion_iterations)
    if dilation_iterations > 0:
        mask = binary_dilation(mask, iterations=dilation_iterations)

    # Connected components
    labeled, num = label(mask)
    if num > 0:
        # Keep only components above size threshold
        sizes = np.bincount(labeled.ravel())
        keep = np.zeros_like(sizes, dtype=bool)
        keep[0] = False  # background
        keep[1:] = sizes[1:] >= min_component_size
        cleaned_mask = keep[labeled]
        num_components = int(np.count_nonzero(keep) - int(keep[0]))
    else:
        cleaned_mask = mask
        num_components = 0

    # Distance transform (backend preference: explicit edt_backend, else gpu_backend hints)
    distance_map, backend_used = _compute_distance_map(
        cleaned_mask,
        edt_backend,
        gpu_backend,
        allow_cpu_fallback=allow_cpu_edt,
    )

    # Optional removal of overly thick components (e.g., bone). Recompute EDT after removal.
    if thick_component_max_radius is not None and thick_component_max_radius > 0:
        labeled, num_lbl = label(cleaned_mask)
        if num_lbl > 0:
            max_per_component = np.zeros(num_lbl + 1, dtype=np.float32)
            # distance_map aligns with labeled; compute max per label
            flat_dm = distance_map.reshape(-1)
            flat_lbl = labeled.reshape(-1)
            for lbl_id in range(1, num_lbl + 1):
                mask_lbl = flat_lbl == lbl_id
                if mask_lbl.any():
                    max_per_component[lbl_id] = float(flat_dm[mask_lbl].max())
            keep = max_per_component <= float(thick_component_max_radius)
            keep[0] = False
            filtered_mask = keep[labeled]
            if not np.array_equal(filtered_mask, cleaned_mask):
                cleaned_mask = filtered_mask
                distance_map, backend_used = _compute_distance_map(
                    cleaned_mask,
                    edt_backend,
                    gpu_backend,
                    allow_cpu_fallback=allow_cpu_edt,
                )

    return {
        'cleaned_mask': cleaned_mask,
        'distance_map': distance_map,
        'num_components': num_components,
        'distance_backend': backend_used,
    }
