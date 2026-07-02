"""Array crop/paste helpers for memory-bounded case stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropRegion:
    """A z, y, x crop region with enough metadata to paste outputs back."""

    slices: tuple[slice, slice, slice]
    full_shape: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s.stop - s.start) for s in self.slices)

    def crop(self, array: np.ndarray) -> np.ndarray:
        """Return a view/copy of ``array`` inside this crop."""
        if tuple(array.shape[:3]) != self.full_shape:
            raise ValueError(f"Array shape {array.shape[:3]} does not match crop full shape {self.full_shape}.")
        return array[self.slices]

    def paste(self, cropped: np.ndarray, fill_value: object = 0) -> np.ndarray:
        """Paste a cropped 3D array into a full-size array."""
        output = np.full(self.full_shape, fill_value, dtype=cropped.dtype)
        output[self.slices] = cropped
        return output


def crop_region_for_mask(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    margin_mm: float = 0.0,
) -> CropRegion:
    """Create a crop around non-zero mask voxels with a physical margin.

    Arrays in this project are z, y, x, while SimpleITK spacing is x, y, z.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("mask must be a 3D array.")
    coords = np.argwhere(binary)
    full_shape = tuple(int(v) for v in binary.shape)
    if coords.size == 0:
        return CropRegion(
            slices=tuple(slice(0, dim) for dim in full_shape),  # type: ignore[assignment]
            full_shape=full_shape,
        )

    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    pad = np.ceil(max(float(margin_mm), 0.0) / spacing_zyx).astype(int)
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(full_shape))
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    return CropRegion(slices=slices, full_shape=full_shape)  # type: ignore[arg-type]

