"""Deterministic anatomical overlap resolution and shell exclusions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._spatial import physical_xyz_to_continuous_zyx, validate_spacing


def _binary(mask: np.ndarray, shape: tuple[int, ...] | None = None, name: str = "mask") -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array.")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}.")
    return result


@dataclass(frozen=True)
class OverlapResolutionResult:
    """Result of first-priority-wins assignment of potentially overlapping masks."""

    labels: np.ndarray
    assigned_masks: dict[str, np.ndarray]
    conflict_mask: np.ndarray
    input_overlap_voxels: int
    priority: tuple[str, ...]
    label_values: dict[str, int]


def resolve_anatomical_overlaps(
    masks: Mapping[str, np.ndarray],
    *,
    priority: Sequence[str],
    label_values: Mapping[str, int] | None = None,
    domain_mask: np.ndarray | None = None,
) -> OverlapResolutionResult:
    """Resolve overlaps reproducibly with explicit, first-priority-wins semantics."""
    if not masks:
        raise ValueError("masks cannot be empty.")
    names = tuple(str(name) for name in priority)
    if len(names) != len(set(names)):
        raise ValueError("priority contains duplicate names.")
    missing = set(masks) - set(names)
    unknown = set(names) - set(masks)
    if missing or unknown:
        raise ValueError(f"priority/mask mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    first = _binary(next(iter(masks.values())), name="first mask")
    shape = first.shape
    binary = {name: _binary(masks[name], shape, name) for name in names}
    domain = np.ones(shape, dtype=bool) if domain_mask is None else _binary(domain_mask, shape, "domain_mask")
    values = (
        {name: index + 1 for index, name in enumerate(names)}
        if label_values is None
        else {name: int(label_values[name]) for name in names}
    )
    if len(set(values.values())) != len(values) or any(value <= 0 for value in values.values()):
        raise ValueError("label_values must be unique positive integers.")
    stack = np.stack([binary[name] & domain for name in names], axis=0)
    conflict = stack.sum(axis=0) > 1
    labels = np.zeros(shape, dtype=np.uint16)
    assigned: dict[str, np.ndarray] = {}
    unassigned = domain.copy()
    for name in names:
        current = binary[name] & unassigned
        labels[current] = values[name]
        assigned[name] = current
        unassigned &= ~current
    return OverlapResolutionResult(
        labels=labels,
        assigned_masks=assigned,
        conflict_mask=conflict,
        input_overlap_voxels=int(conflict.sum()),
        priority=names,
        label_values=values,
    )


@dataclass(frozen=True)
class ExclusionResult:
    raw_mask: np.ndarray
    filtered_mask: np.ndarray
    excluded_mask: np.ndarray
    exclusion_masks: dict[str, np.ndarray]
    exclusion_voxels: dict[str, int]
    exclusion_fractions: dict[str, float]
    overlap_mask: np.ndarray


def apply_shell_exclusions(
    raw_shell: np.ndarray,
    *,
    target_lumen_mask: np.ndarray | None = None,
    other_artery_mask: np.ndarray | None = None,
    bifurcation_mask: np.ndarray | None = None,
    partial_volume_mask: np.ndarray | None = None,
    additional_exclusions: Mapping[str, np.ndarray] | None = None,
) -> ExclusionResult:
    """Apply named exclusions and retain auditable per-reason fractions.

    Fractions use the raw in-image shell as denominator.  A voxel can belong to
    more than one reason; ``excluded_mask`` and the filtered result use their
    union while ``overlap_mask`` records this ambiguity.
    """
    raw = _binary(raw_shell, name="raw_shell")
    shape = raw.shape
    candidates: dict[str, np.ndarray] = {}
    for name, mask in (
        ("target_lumen", target_lumen_mask),
        ("other_artery", other_artery_mask),
        ("bifurcation", bifurcation_mask),
        ("partial_volume", partial_volume_mask),
    ):
        if mask is not None:
            candidates[name] = _binary(mask, shape, f"{name}_mask") & raw
    for name, mask in sorted((additional_exclusions or {}).items()):
        if name in candidates:
            raise ValueError(f"Duplicate exclusion name: {name}")
        candidates[str(name)] = _binary(mask, shape, str(name)) & raw
    if candidates:
        stack = np.stack(list(candidates.values()), axis=0)
        excluded = np.any(stack, axis=0)
        overlap = stack.sum(axis=0) > 1
    else:
        excluded = np.zeros(shape, dtype=bool)
        overlap = np.zeros(shape, dtype=bool)
    denominator = int(raw.sum())
    counts = {name: int(mask.sum()) for name, mask in candidates.items()}
    fractions = {
        name: (float(count) / denominator if denominator else 0.0) for name, count in counts.items()
    }
    return ExclusionResult(
        raw_mask=raw,
        filtered_mask=raw & ~excluded,
        excluded_mask=excluded,
        exclusion_masks=candidates,
        exclusion_voxels=counts,
        exclusion_fractions=fractions,
        overlap_mask=overlap,
    )


def physical_dilation(mask: np.ndarray, spacing_xyz: Sequence[float], radius_mm: float) -> np.ndarray:
    """Dilate a mask by Euclidean physical distance."""
    binary = _binary(mask)
    spacing = validate_spacing(spacing_xyz)
    if radius_mm < 0 or not np.isfinite(radius_mm):
        raise ValueError("radius_mm must be finite and non-negative.")
    if radius_mm == 0 or not binary.any():
        return binary.copy()
    from scipy import ndimage as ndi

    sampling_zyx = (spacing[2], spacing[1], spacing[0])
    return ndi.distance_transform_edt(~binary, sampling=sampling_zyx) <= float(radius_mm)


def bifurcation_exclusion_mask(
    shape_zyx: Sequence[int],
    branch_points_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
    radius_mm: float,
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize physical branch points and create spherical exclusion zones."""
    shape = tuple(int(value) for value in shape_zyx)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape_zyx must contain three positive dimensions.")
    points = np.asarray(branch_points_xyz, dtype=float)
    if points.size == 0:
        return np.zeros(shape, dtype=bool)
    points = points.reshape((-1, 3))
    continuous = physical_xyz_to_continuous_zyx(points, spacing_xyz, origin_xyz, direction)
    indices = np.rint(continuous).astype(int)
    inside = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    seeds = np.zeros(shape, dtype=bool)
    valid = indices[inside]
    if valid.size:
        seeds[tuple(valid.T)] = True
    return physical_dilation(seeds, spacing_xyz, radius_mm)


__all__ = [
    "ExclusionResult",
    "OverlapResolutionResult",
    "apply_shell_exclusions",
    "bifurcation_exclusion_mask",
    "physical_dilation",
    "resolve_anatomical_overlaps",
]
