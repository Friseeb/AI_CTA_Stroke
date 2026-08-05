"""Auditable physical-distance shells around an artery lumen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cta_common.shells import external_shell

from ..crop import crop_region_for_mask
from ._spatial import validate_spacing, voxel_volume_mm3
from .exclusions import ExclusionResult, apply_shell_exclusions


def _number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


@dataclass(frozen=True, order=True)
class RadialBand:
    inner_mm: float
    outer_mm: float
    name: str | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.inner_mm)
            or not np.isfinite(self.outer_mm)
            or self.inner_mm < 0
            or self.outer_mm <= self.inner_mm
        ):
            raise ValueError("A radial band requires 0 <= inner_mm < outer_mm.")

    @property
    def identifier(self) -> str:
        return self.name or f"shell_{_number(self.inner_mm)}_{_number(self.outer_mm)}mm"


DEFAULT_RADIAL_BANDS: tuple[RadialBand, ...] = (
    RadialBand(0.0, 1.0),
    RadialBand(1.0, 2.0),
    RadialBand(2.0, 3.0),
    RadialBand(3.0, 5.0),
    RadialBand(5.0, 10.0),
)

COMPATIBILITY_RADIAL_BANDS: tuple[RadialBand, ...] = (
    RadialBand(0.0, 2.0),
    RadialBand(2.0, 5.0),
)


@dataclass(frozen=True)
class AdaptiveShellLaw:
    """Capped diameter-dependent extent ``clip(alpha * diameter, min, max)``."""

    alpha: float = 1.0
    minimum_mm: float = 1.0
    maximum_mm: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite((self.alpha, self.minimum_mm, self.maximum_mm)).all():
            raise ValueError("Adaptive shell parameters must be finite.")
        if self.alpha < 0 or self.minimum_mm < 0 or self.maximum_mm < self.minimum_mm:
            raise ValueError("Require alpha >= 0 and 0 <= minimum_mm <= maximum_mm.")

    def extent_mm(self, diameter_mm: float | np.ndarray) -> float | np.ndarray:
        diameter = np.asarray(diameter_mm, dtype=float)
        if np.any(~np.isfinite(diameter)) or np.any(diameter < 0):
            raise ValueError("diameter_mm must be finite and non-negative.")
        result = np.clip(self.alpha * diameter, self.minimum_mm, self.maximum_mm)
        return float(result) if result.ndim == 0 else result


def capped_adaptive_extent_mm(
    diameter_mm: float | np.ndarray,
    *,
    alpha: float,
    minimum_mm: float,
    maximum_mm: float,
) -> float | np.ndarray:
    return AdaptiveShellLaw(alpha, minimum_mm, maximum_mm).extent_mm(diameter_mm)


@dataclass(frozen=True)
class ShellBandResult:
    band: RadialBand
    raw_mask: np.ndarray
    filtered_mask: np.ndarray
    valid_denominator_mask: np.ndarray
    boundary_missing_voxels: int
    expected_complete_voxels: int
    raw_voxels: int
    valid_voxels: int
    expected_complete_volume_mm3: float
    raw_volume_mm3: float
    valid_volume_mm3: float
    boundary_truncation_fraction: float
    exclusion_voxels: dict[str, int]
    exclusion_fractions: dict[str, float]
    flags: tuple[str, ...]

    @property
    def is_boundary_truncated(self) -> bool:
        return self.boundary_missing_voxels > 0


@dataclass(frozen=True)
class ShellSetResult:
    vessel_id: str
    spacing_xyz: tuple[float, float, float]
    interpolation_method: str
    bands: dict[str, ShellBandResult]
    parameters: dict[str, object]

    @property
    def raw_masks(self) -> dict[str, np.ndarray]:
        return {name: result.raw_mask for name, result in self.bands.items()}

    @property
    def filtered_masks(self) -> dict[str, np.ndarray]:
        return {name: result.filtered_mask for name, result in self.bands.items()}

    @property
    def valid_denominator_masks(self) -> dict[str, np.ndarray]:
        return {name: result.valid_denominator_mask for name, result in self.bands.items()}

    @property
    def qc_flags(self) -> tuple[str, ...]:
        return tuple(sorted({flag for band in self.bands.values() for flag in band.flags}))


def _coerce_bands(
    bands: Sequence[RadialBand | Sequence[float] | Mapping[str, object]],
) -> tuple[RadialBand, ...]:
    parsed: list[RadialBand] = []
    for item in bands:
        if isinstance(item, RadialBand):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(
                RadialBand(
                    float(item["inner_mm"]),
                    float(item["outer_mm"]),
                    str(item["name"]) if item.get("name") is not None else None,
                )
            )
        else:
            values = tuple(item)
            if len(values) not in (2, 3):
                raise ValueError("Band sequences must contain inner, outer[, name].")
            parsed.append(
                RadialBand(float(values[0]), float(values[1]), str(values[2]) if len(values) == 3 else None)
            )
    if not parsed:
        raise ValueError("At least one radial band is required.")
    identifiers = [band.identifier for band in parsed]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Radial band names must be unique.")
    return tuple(parsed)


def _pad_width(spacing_xyz: tuple[float, float, float], outer_mm: float) -> tuple[int, int, int]:
    spacing_zyx = np.asarray((spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), dtype=float)
    return tuple((np.ceil(float(outer_mm) / spacing_zyx).astype(int) + 2).tolist())


def _padded_geometry(mask: np.ndarray, spacing_xyz: tuple[float, float, float], outer_mm: float):
    pad = _pad_width(spacing_xyz, outer_mm)
    padded = np.pad(mask, tuple((value, value) for value in pad), mode="constant", constant_values=False)
    crop = tuple(slice(value, value + size) for value, size in zip(pad, mask.shape))
    return padded, crop


def _band_result(
    band: RadialBand,
    raw: np.ndarray,
    expected_complete_voxels: int,
    exclusions: ExclusionResult,
    voxel_volume: float,
) -> ShellBandResult:
    raw_count = int(raw.sum())
    valid_count = int(exclusions.filtered_mask.sum())
    missing = max(0, int(expected_complete_voxels) - raw_count)
    truncation_fraction = float(missing / expected_complete_voxels) if expected_complete_voxels else 0.0
    flags: list[str] = []
    if missing:
        flags.append("shell_truncated_by_image_boundary")
    if exclusions.exclusion_voxels.get("other_artery", 0):
        flags.append("shell_overlaps_other_artery")
    if exclusions.exclusion_voxels.get("bifurcation", 0):
        flags.append("shell_intersects_bifurcation")
    if valid_count == 0:
        flags.append("empty_valid_shell")
    return ShellBandResult(
        band=band,
        raw_mask=raw,
        filtered_mask=exclusions.filtered_mask,
        valid_denominator_mask=exclusions.filtered_mask.copy(),
        boundary_missing_voxels=missing,
        expected_complete_voxels=int(expected_complete_voxels),
        raw_voxels=raw_count,
        valid_voxels=valid_count,
        expected_complete_volume_mm3=float(expected_complete_voxels * voxel_volume),
        raw_volume_mm3=float(raw_count * voxel_volume),
        valid_volume_mm3=float(valid_count * voxel_volume),
        boundary_truncation_fraction=truncation_fraction,
        exclusion_voxels=dict(exclusions.exclusion_voxels),
        exclusion_fractions=dict(exclusions.exclusion_fractions),
        flags=tuple(flags),
    )


def generate_radial_shells(
    lumen_mask: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    vessel_id: str,
    bands: Sequence[RadialBand | Sequence[float] | Mapping[str, object]] = DEFAULT_RADIAL_BANDS,
    other_artery_mask: np.ndarray | None = None,
    bifurcation_exclusion_mask: np.ndarray | None = None,
    partial_volume_boundary_mm: float = 0.0,
    additional_exclusions: Mapping[str, np.ndarray] | None = None,
    interpolation_method: str = "native_grid_physical_edt",
) -> ShellSetResult:
    """Generate raw and filtered physical-mm shells around one vessel lumen."""
    lumen = np.asarray(lumen_mask, dtype=bool)
    if lumen.ndim != 3:
        raise ValueError("lumen_mask must be three-dimensional.")
    if not lumen.any():
        raise ValueError("lumen_mask is empty.")
    spacing = validate_spacing(spacing_xyz)
    parsed = _coerce_bands(bands)
    if partial_volume_boundary_mm < 0 or not np.isfinite(partial_volume_boundary_mm):
        raise ValueError("partial_volume_boundary_mm must be finite and non-negative.")
    max_outer = max(band.outer_mm for band in parsed)
    padded_lumen, crop = _padded_geometry(lumen, spacing, max_outer)
    partial_volume = (
        external_shell(lumen, spacing, 0.0, float(partial_volume_boundary_mm))
        if partial_volume_boundary_mm > 0
        else None
    )
    voxel_volume = voxel_volume_mm3(spacing)
    results: dict[str, ShellBandResult] = {}
    for band in parsed:
        padded_raw = external_shell(padded_lumen, spacing, band.inner_mm, band.outer_mm)
        raw = padded_raw[crop]
        exclusions = apply_shell_exclusions(
            raw,
            target_lumen_mask=lumen,
            other_artery_mask=other_artery_mask,
            bifurcation_mask=bifurcation_exclusion_mask,
            partial_volume_mask=partial_volume,
            additional_exclusions=additional_exclusions,
        )
        results[band.identifier] = _band_result(
            band, raw, int(padded_raw.sum()), exclusions, voxel_volume
        )
    parameters: dict[str, object] = {
        "distance_metric": "physical_euclidean",
        "units": "mm",
        "bands": [
            {"name": band.identifier, "inner_mm": band.inner_mm, "outer_mm": band.outer_mm}
            for band in parsed
        ],
        "partial_volume_boundary_mm": float(partial_volume_boundary_mm),
    }
    return ShellSetResult(
        vessel_id=str(vessel_id),
        spacing_xyz=spacing,
        interpolation_method=str(interpolation_method),
        bands=results,
        parameters=parameters,
    )


def _adaptive_padded_mask(
    lumen: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    diameter_mm: float | np.ndarray,
    law: AdaptiveShellLaw,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice], float]:
    from scipy import ndimage as ndi

    maximum = float(law.maximum_mm)
    padded_lumen, crop = _padded_geometry(lumen, spacing_xyz, maximum)
    sampling_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    distance, indices = ndi.distance_transform_edt(
        ~padded_lumen, sampling=sampling_zyx, return_indices=True
    )
    if np.asarray(diameter_mm).ndim == 0:
        extent = float(law.extent_mm(float(diameter_mm)))
        return (~padded_lumen) & (distance > 0) & (distance <= extent), distance, crop, extent
    diameter = np.asarray(diameter_mm, dtype=float)
    if diameter.shape != lumen.shape:
        raise ValueError("A diameter map must have the same shape as lumen_mask.")
    if np.any(~np.isfinite(diameter[lumen])) or np.any(diameter[lumen] < 0):
        raise ValueError("Diameter values on lumen voxels must be finite and non-negative.")
    pad = tuple(item.start for item in crop)
    padded_diameter = np.zeros_like(padded_lumen, dtype=float)
    padded_diameter[crop] = np.where(lumen, diameter, 0.0)
    nearest_diameter = padded_diameter[tuple(indices)]
    local_extent = np.asarray(law.extent_mm(nearest_diameter), dtype=float)
    adaptive = (~padded_lumen) & (distance > 0) & (distance <= local_extent)
    del pad
    representative = float(np.median(np.asarray(law.extent_mm(diameter[lumen]), dtype=float)))
    return adaptive, distance, crop, representative


def generate_adaptive_shell(
    lumen_mask: np.ndarray,
    spacing_xyz: Sequence[float],
    diameter_mm: float | np.ndarray,
    *,
    vessel_id: str,
    law: AdaptiveShellLaw = AdaptiveShellLaw(),
    other_artery_mask: np.ndarray | None = None,
    bifurcation_exclusion_mask: np.ndarray | None = None,
    partial_volume_boundary_mm: float = 0.0,
    additional_exclusions: Mapping[str, np.ndarray] | None = None,
    interpolation_method: str = "native_grid_physical_edt_nearest_lumen_diameter",
) -> ShellSetResult:
    """Generate a shell whose local maximum extent follows a capped diameter law."""
    lumen = np.asarray(lumen_mask, dtype=bool)
    if lumen.ndim != 3 or not lumen.any():
        raise ValueError("lumen_mask must be a non-empty three-dimensional mask.")
    spacing = validate_spacing(spacing_xyz)
    if partial_volume_boundary_mm < 0:
        raise ValueError("partial_volume_boundary_mm must be non-negative.")
    # Adaptive EDT needs only the lumen neighbourhood.  Include a two-voxel
    # physical safety halo beyond the maximum extent so artificial crop edges
    # cannot affect any accepted shell voxel.  _adaptive_padded_mask then pads
    # this local image as before, preserving expected out-of-FOV shell counts
    # when the true lumen touches an image boundary.
    region = crop_region_for_mask(
        lumen,
        spacing,
        margin_mm=float(law.maximum_mm) + 2.0 * max(spacing),
    )
    local_lumen = np.asarray(region.crop(lumen), dtype=bool)
    diameter_array = np.asarray(diameter_mm)
    local_diameter: float | np.ndarray
    if diameter_array.ndim == 0:
        local_diameter = float(diameter_array)
    else:
        if diameter_array.shape != lumen.shape:
            raise ValueError("A diameter map must have the same shape as lumen_mask.")
        local_diameter = np.asarray(region.crop(diameter_array), dtype=float)
    padded_raw, _, crop, representative = _adaptive_padded_mask(
        local_lumen, spacing, local_diameter, law
    )
    local_raw = padded_raw[crop]
    raw = region.paste(local_raw, fill_value=False)
    partial_volume = (
        external_shell(lumen, spacing, 0.0, float(partial_volume_boundary_mm))
        if partial_volume_boundary_mm > 0
        else None
    )
    exclusions = apply_shell_exclusions(
        raw,
        target_lumen_mask=lumen,
        other_artery_mask=other_artery_mask,
        bifurcation_mask=bifurcation_exclusion_mask,
        partial_volume_mask=partial_volume,
        additional_exclusions=additional_exclusions,
    )
    band = RadialBand(0.0, law.maximum_mm, "adaptive_shell")
    result = _band_result(
        band,
        raw,
        int(padded_raw.sum()),
        exclusions,
        voxel_volume_mm3(spacing),
    )
    parameters: dict[str, object] = {
        "distance_metric": "physical_euclidean",
        "units": "mm",
        "adaptive_law": {
            "formula": "clip(alpha * diameter_mm, minimum_mm, maximum_mm)",
            "alpha": law.alpha,
            "minimum_mm": law.minimum_mm,
            "maximum_mm": law.maximum_mm,
        },
        "diameter_source": "scalar" if np.asarray(diameter_mm).ndim == 0 else "nearest_lumen_voxel_map",
        "representative_extent_mm": representative,
        "partial_volume_boundary_mm": float(partial_volume_boundary_mm),
    }
    return ShellSetResult(
        vessel_id=str(vessel_id),
        spacing_xyz=spacing,
        interpolation_method=str(interpolation_method),
        bands={band.identifier: result},
        parameters=parameters,
    )


# A concise alias for callers that naturally use the noun phrase.
adaptive_shell = generate_adaptive_shell


__all__ = [
    "AdaptiveShellLaw",
    "COMPATIBILITY_RADIAL_BANDS",
    "DEFAULT_RADIAL_BANDS",
    "RadialBand",
    "ShellBandResult",
    "ShellSetResult",
    "adaptive_shell",
    "capped_adaptive_extent_mm",
    "generate_adaptive_shell",
    "generate_radial_shells",
]
