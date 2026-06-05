"""Mask preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskCleaningReport:
    components_before: int
    components_after: int
    voxels_before: int
    voxels_after: int
    kept_component_label: int | None


def clean_aorta_mask(
    mask: np.ndarray,
    keep_largest_component: bool = True,
    fill_holes: bool = True,
    min_component_voxels: int = 0,
) -> tuple[np.ndarray, MaskCleaningReport]:
    """Clean a binary aorta mask with conservative connected-component filtering."""
    binary = np.asarray(mask, dtype=bool)
    voxels_before = int(binary.sum())
    labels, n_components = _label(binary)
    cleaned = binary.copy()
    kept_label: int | None = None

    if n_components > 0:
        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        keep = np.ones(n_components + 1, dtype=bool)

        if min_component_voxels > 0:
            keep &= component_sizes >= min_component_voxels
            keep[0] = False

        if keep_largest_component:
            kept_label = int(component_sizes.argmax())
            keep[:] = False
            keep[kept_label] = True

        cleaned = keep[labels]

    if fill_holes and cleaned.any():
        cleaned = _binary_fill_holes(cleaned)

    _, n_after = _label(cleaned)
    report = MaskCleaningReport(
        components_before=int(n_components),
        components_after=int(n_after),
        voxels_before=voxels_before,
        voxels_after=int(cleaned.sum()),
        kept_component_label=kept_label,
    )
    return cleaned.astype(bool), report


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage as ndi

        return ndi.label(mask)
    except Exception as exc:
        if mask.size > 250_000:
            raise ImportError(
                "SciPy connected-component labeling is required for production mask cleaning. "
                "The current Python environment could not import SciPy correctly."
            ) from exc
        return _label_small(mask)


def _binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi

        return ndi.binary_fill_holes(mask)
    except Exception:
        return mask


def _label_small(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    labels = np.zeros(binary.shape, dtype=np.int32)
    label_id = 0
    neighbor_offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == dy == dx == 0)
    ]
    for start in np.argwhere(binary):
        start_tuple = tuple(int(v) for v in start)
        if labels[start_tuple] != 0:
            continue
        label_id += 1
        stack = [start_tuple]
        labels[start_tuple] = label_id
        while stack:
            z, y, x = stack.pop()
            for dz, dy, dx in neighbor_offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                if (
                    0 <= nz < binary.shape[0]
                    and 0 <= ny < binary.shape[1]
                    and 0 <= nx < binary.shape[2]
                    and binary[nz, ny, nx]
                    and labels[nz, ny, nx] == 0
                ):
                    labels[nz, ny, nx] = label_id
                    stack.append((nz, ny, nx))
    return labels, label_id
