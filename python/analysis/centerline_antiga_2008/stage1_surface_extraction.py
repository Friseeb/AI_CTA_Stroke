"""Stage 1: Surface extraction and cleaning from binary segmentation."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label,
    distance_transform_edt,
)


def extract_surface(
    binary_mask: np.ndarray,
    min_component_size: int = 50,
    erosion_iterations: int = 1,
    dilation_iterations: int = 1,
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

    # Distance transform
    distance_map = distance_transform_edt(cleaned_mask)

    return {
        'cleaned_mask': cleaned_mask,
        'distance_map': distance_map,
        'num_components': num_components,
    }
