"""Physical-distance peri-aortic shell generation."""

from __future__ import annotations

import numpy as np


def _sampling_zyx(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))


def external_shell(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    inner_mm: float,
    outer_mm: float,
) -> np.ndarray:
    """Create an outside-only shell at physical distances from the mask boundary."""
    if outer_mm <= inner_mm:
        raise ValueError("outer_mm must be greater than inner_mm.")
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    cropped, slices = _crop_around_mask(binary, spacing_xyz, margin_mm=outer_mm)
    distance_outside = _distance_transform_edt(~cropped, sampling=_sampling_zyx(spacing_xyz))
    shell_crop = (~cropped) & (distance_outside > float(inner_mm)) & (distance_outside <= float(outer_mm))
    shell = np.zeros_like(binary, dtype=bool)
    shell[slices] = shell_crop
    return shell


def internal_boundary_shell(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    depth_mm: float,
) -> np.ndarray:
    """Create an inside-mask shell adjacent to the mask boundary."""
    binary = np.asarray(mask, dtype=bool)
    if depth_mm <= 0 or not binary.any():
        return np.zeros_like(binary, dtype=bool)
    cropped, slices = _crop_around_mask(binary, spacing_xyz, margin_mm=0.0)
    distance_inside = _distance_transform_edt(cropped, sampling=_sampling_zyx(spacing_xyz))
    shell_crop = cropped & (distance_inside <= float(depth_mm))
    shell = np.zeros_like(binary, dtype=bool)
    shell[slices] = shell_crop
    return shell


def combined_periaortic_shell(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    outer_mm: float,
    internal_mm: float = 1.0,
) -> np.ndarray:
    """Create a peri-aortic ROI that includes outside shell plus inner boundary voxels."""
    return external_shell(mask, spacing_xyz, 0.0, outer_mm) | internal_boundary_shell(
        mask, spacing_xyz, internal_mm
    )


def create_aorta_wall_band_masks(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    internal_mm: float,
    external_mm: float,
) -> dict[str, np.ndarray]:
    """Create physical-distance wall-band masks around the aorta mask boundary.

    The combined ``aorta_wall_band`` is intended for wall-focused calcium
    thresholding: it keeps only boundary-adjacent voxels and excludes the
    central contrast-filled lumen/core of the aorta mask.
    """
    binary = np.asarray(mask, dtype=bool)
    empty = np.zeros_like(binary, dtype=bool)
    internal = (
        internal_boundary_shell(binary, spacing_xyz, depth_mm=float(internal_mm))
        if internal_mm > 0
        else empty.copy()
    )
    external = (
        external_shell(binary, spacing_xyz, inner_mm=0.0, outer_mm=float(external_mm))
        if external_mm > 0
        else empty.copy()
    )
    return {
        "aorta_wall_internal": internal,
        "aorta_wall_external": external,
        "aorta_wall_band": internal | external,
    }


def create_base_shells(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    shell_specs: list[dict[str, float | str]],
) -> dict[str, np.ndarray]:
    """Create named external shells from config specs."""
    shells: dict[str, np.ndarray] = {}
    for spec in shell_specs:
        name = str(spec["name"])
        shells[name] = external_shell(
            mask,
            spacing_xyz,
            inner_mm=float(spec["inner_mm"]),
            outer_mm=float(spec["outer_mm"]),
        )
    return shells


def local_shell_around_mask(
    seed_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    outer_mm: float,
) -> np.ndarray:
    """Create an external local shell around seed voxels while excluding a base mask."""
    seeds = np.asarray(seed_mask, dtype=bool)
    excluded = np.asarray(exclusion_mask, dtype=bool)
    if outer_mm <= 0 or not seeds.any():
        return np.zeros_like(seeds, dtype=bool)
    cropped_seeds, slices = _crop_around_mask(seeds, spacing_xyz, margin_mm=outer_mm)
    cropped_excluded = excluded[slices]
    distance_to_seed = _distance_transform_edt(~cropped_seeds, sampling=_sampling_zyx(spacing_xyz))
    shell_crop = (~cropped_excluded) & (distance_to_seed > 0) & (distance_to_seed <= float(outer_mm))
    shell = np.zeros_like(seeds, dtype=bool)
    shell[slices] = shell_crop
    return shell


def _distance_transform_edt(mask: np.ndarray, sampling: tuple[float, float, float]) -> np.ndarray:
    """Use SciPy EDT, with a small-array fallback for broken local environments."""
    try:
        from scipy import ndimage as ndi

        return ndi.distance_transform_edt(mask, sampling=sampling)
    except Exception as exc:
        if mask.size > 250_000:
            raise ImportError(
                "SciPy distance_transform_edt is required for production shell generation. "
                "The current Python environment could not import SciPy correctly."
            ) from exc
        return _brute_force_distance_to_false(mask, sampling)


def _brute_force_distance_to_false(mask: np.ndarray, sampling: tuple[float, float, float]) -> np.ndarray:
    coords = np.argwhere(mask)
    zeros = np.argwhere(~mask)
    distances = np.zeros(mask.shape, dtype=float)
    if coords.size == 0 or zeros.size == 0:
        return distances
    scaled_zeros = zeros * np.asarray(sampling)
    for coord in coords:
        scaled = coord * np.asarray(sampling)
        distances[tuple(coord)] = float(np.sqrt(np.min(np.sum((scaled_zeros - scaled) ** 2, axis=1))))
    return distances


def _crop_around_mask(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    margin_mm: float,
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        empty_slices = tuple(slice(0, dim) for dim in mask.shape)
        return mask, empty_slices  # type: ignore[return-value]

    spacing_zyx = np.asarray(_sampling_zyx(spacing_xyz), dtype=float)
    pad = np.ceil(float(margin_mm) / spacing_zyx).astype(int) + 2
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(mask.shape))
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    return mask[slices], slices  # type: ignore[index, return-value]
