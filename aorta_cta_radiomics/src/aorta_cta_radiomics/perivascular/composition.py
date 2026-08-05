"""Deterministic, mutually exclusive perivascular tissue composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np

from ._spatial import validate_spacing, voxel_volume_mm3
from .exclusions import OverlapResolutionResult, resolve_anatomical_overlaps


class TissueLabel(IntEnum):
    OUTSIDE = 0
    ARTERIAL_LUMEN = 1
    OTHER_ARTERY = 2
    VEIN = 3
    BONE = 4
    AIRWAY = 5
    THYROID_GLAND = 6
    SKELETAL_MUSCLE = 7
    ADIPOSE = 8
    OTHER_SOFT_TISSUE = 9
    HIGH_DENSITY = 10
    UNCERTAIN = 11


TISSUE_NAMES: dict[int, str] = {
    int(label): label.name.lower() for label in TissueLabel
}

DEFAULT_OVERLAP_PRIORITY: tuple[str, ...] = (
    "arterial_lumen",
    "other_artery",
    "vein",
    "bone",
    "airway",
    "thyroid_gland",
    "skeletal_muscle",
    "adipose",
    "other_soft_tissue",
    "high_density",
    "uncertain",
)

_LABEL_BY_NAME: dict[str, TissueLabel] = {
    "arterial_lumen": TissueLabel.ARTERIAL_LUMEN,
    "other_artery": TissueLabel.OTHER_ARTERY,
    "vein": TissueLabel.VEIN,
    "bone": TissueLabel.BONE,
    "airway": TissueLabel.AIRWAY,
    "thyroid_gland": TissueLabel.THYROID_GLAND,
    "skeletal_muscle": TissueLabel.SKELETAL_MUSCLE,
    "adipose": TissueLabel.ADIPOSE,
    "other_soft_tissue": TissueLabel.OTHER_SOFT_TISSUE,
    "high_density": TissueLabel.HIGH_DENSITY,
    "uncertain": TissueLabel.UNCERTAIN,
}


def _validate_window(name: str, window: Sequence[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in window)
    if len(values) != 2 or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError(f"{name} must contain finite (lower, upper) with upper > lower.")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class TissueThresholds:
    """Stored HU configuration, including primary and sensitivity fat windows."""

    primary_fat_hu: tuple[float, float] = (-190.0, -30.0)
    narrow_fat_hu: tuple[float, float] = (-150.0, -50.0)
    wide_fat_hu: tuple[float, float] = (-200.0, -20.0)
    high_density_min_hu: float = 200.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_fat_hu", _validate_window("primary_fat_hu", self.primary_fat_hu))
        object.__setattr__(self, "narrow_fat_hu", _validate_window("narrow_fat_hu", self.narrow_fat_hu))
        object.__setattr__(self, "wide_fat_hu", _validate_window("wide_fat_hu", self.wide_fat_hu))
        if not np.isfinite(self.high_density_min_hu):
            raise ValueError("high_density_min_hu must be finite.")

    @property
    def fat_windows(self) -> dict[str, tuple[float, float]]:
        return {
            "primary": self.primary_fat_hu,
            "narrow": self.narrow_fat_hu,
            "wide": self.wide_fat_hu,
        }


@dataclass(frozen=True)
class CompositionResult:
    labels: np.ndarray
    roi_mask: np.ndarray
    valid_denominator_mask: np.ndarray
    counts: dict[str, int]
    volumes_mm3: dict[str, float]
    fractions: dict[str, float]
    fat_sensitivity_masks: dict[str, np.ndarray]
    overlap_conflict_mask: np.ndarray
    input_overlap_voxels: int
    overlap_priority: tuple[str, ...]
    thresholds: TissueThresholds
    spacing_xyz: tuple[float, float, float]
    acquisition_metadata: dict[str, object]
    qc_flags: tuple[str, ...]

    def mask(self, label: TissueLabel | int) -> np.ndarray:
        return self.labels == int(label)

    @property
    def fat_mask(self) -> np.ndarray:
        return self.mask(TissueLabel.ADIPOSE)


def _mask_or_empty(mask: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}.")
    return result


def composition_statistics(
    labels: np.ndarray,
    denominator_mask: np.ndarray,
    spacing_xyz: Sequence[float],
) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    """Summarize mutually exclusive labels against an explicit denominator."""
    label_array = np.asarray(labels)
    denominator = np.asarray(denominator_mask, dtype=bool)
    if label_array.shape != denominator.shape:
        raise ValueError("labels and denominator_mask must share a shape.")
    voxel_volume = voxel_volume_mm3(spacing_xyz)
    denominator_count = int(denominator.sum())
    counts: dict[str, int] = {}
    volumes: dict[str, float] = {}
    fractions: dict[str, float] = {}
    for label in TissueLabel:
        if label == TissueLabel.OUTSIDE:
            continue
        name = TISSUE_NAMES[int(label)]
        count = int(((label_array == int(label)) & denominator).sum())
        counts[name] = count
        volumes[name] = float(count * voxel_volume)
        fractions[name] = float(count / denominator_count) if denominator_count else float("nan")
    counts["valid_denominator"] = denominator_count
    volumes["valid_denominator"] = float(denominator_count * voxel_volume)
    fractions["valid_denominator"] = 1.0 if denominator_count else float("nan")
    return counts, volumes, fractions


def classify_tissue_composition(
    cta_hu: np.ndarray,
    roi_mask: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    target_artery_mask: np.ndarray | None = None,
    other_artery_mask: np.ndarray | None = None,
    vein_mask: np.ndarray | None = None,
    bone_mask: np.ndarray | None = None,
    airway_mask: np.ndarray | None = None,
    thyroid_mask: np.ndarray | None = None,
    muscle_mask: np.ndarray | None = None,
    uncertain_mask: np.ndarray | None = None,
    high_density_mask: np.ndarray | None = None,
    thresholds: TissueThresholds = TissueThresholds(),
    overlap_priority: Sequence[str] = DEFAULT_OVERLAP_PRIORITY,
    acquisition_metadata: Mapping[str, object] | None = None,
) -> CompositionResult:
    """Partition a shell into explicit, mutually exclusive tissue classes.

    Masks are resolved using the stored priority.  HU-derived residual tissue
    deliberately excludes high-density and uncertain voxels so those lower
    entries in the documented priority remain reachable.
    """
    image = np.asarray(cta_hu, dtype=float)
    roi = np.asarray(roi_mask, dtype=bool)
    if image.ndim != 3 or roi.shape != image.shape:
        raise ValueError("cta_hu and roi_mask must be same-shaped three-dimensional arrays.")
    spacing = validate_spacing(spacing_xyz)
    shape = image.shape
    finite = np.isfinite(image)
    target = _mask_or_empty(target_artery_mask, shape, "target_artery_mask")
    other = _mask_or_empty(other_artery_mask, shape, "other_artery_mask")
    vein = _mask_or_empty(vein_mask, shape, "vein_mask")
    bone = _mask_or_empty(bone_mask, shape, "bone_mask")
    airway = _mask_or_empty(airway_mask, shape, "airway_mask")
    thyroid = _mask_or_empty(thyroid_mask, shape, "thyroid_mask")
    muscle = _mask_or_empty(muscle_mask, shape, "muscle_mask")
    uncertain = _mask_or_empty(uncertain_mask, shape, "uncertain_mask") | ~finite
    high_density = (
        _mask_or_empty(high_density_mask, shape, "high_density_mask")
        if high_density_mask is not None
        else finite & (image >= float(thresholds.high_density_min_hu))
    )
    low, high = thresholds.primary_fat_hu
    adipose = finite & (image >= low) & (image <= high)
    residual = finite & ~adipose & ~high_density & ~uncertain
    candidates = {
        "arterial_lumen": target,
        "other_artery": other,
        "vein": vein,
        "bone": bone,
        "airway": airway,
        "thyroid_gland": thyroid,
        "skeletal_muscle": muscle,
        "adipose": adipose,
        "other_soft_tissue": residual,
        "high_density": high_density,
        "uncertain": uncertain,
    }
    label_values = {name: int(_LABEL_BY_NAME[name]) for name in candidates}
    resolved: OverlapResolutionResult = resolve_anatomical_overlaps(
        candidates,
        priority=tuple(overlap_priority),
        label_values=label_values,
        domain_mask=roi,
    )
    labels = resolved.labels.astype(np.uint8, copy=False)
    # Explicitly non-assessable compartments are not allowed to dilute tissue
    # fractions.  Vein, bone, airway, gland and high density remain valid,
    # interpretable composition classes.
    invalid = np.isin(
        labels,
        [
            int(TissueLabel.OUTSIDE),
            int(TissueLabel.ARTERIAL_LUMEN),
            int(TissueLabel.OTHER_ARTERY),
            int(TissueLabel.UNCERTAIN),
        ],
    )
    valid_denominator = roi & ~invalid
    counts, volumes, fractions = composition_statistics(labels, valid_denominator, spacing)
    blocked = (
        target | other | vein | bone | airway | thyroid | muscle | high_density | uncertain
    )
    sensitivity = {
        name: roi & finite & ~blocked & (image >= window[0]) & (image <= window[1])
        for name, window in thresholds.fat_windows.items()
    }
    flags: list[str] = []
    if not valid_denominator.any():
        flags.append("empty_valid_tissue_denominator")
    if not (labels == int(TissueLabel.ADIPOSE)).any():
        flags.append("empty_fat_compartment")
    raw_roi_count = int(roi.sum())
    uncertain_count = int(((labels == int(TissueLabel.UNCERTAIN)) & roi).sum())
    if raw_roi_count and uncertain_count / raw_roi_count > 0.25:
        flags.append("excessive_uncertain_fraction")
    metadata = dict(acquisition_metadata or {})
    metadata.setdefault("contrast_phase", "unknown")
    metadata.setdefault("reconstruction_kernel", "unknown")
    metadata.setdefault("slice_thickness_mm", "unknown")
    metadata.setdefault("scanner_manufacturer", "unknown")
    metadata.setdefault("tube_voltage_kvp", "unknown")
    metadata["metadata_missing"] = any(
        value is None or (isinstance(value, str) and value == "unknown")
        for value in metadata.values()
    )
    metadata["hu_threshold_configuration"] = asdict(thresholds)
    return CompositionResult(
        labels=labels,
        roi_mask=roi,
        valid_denominator_mask=valid_denominator,
        counts=counts,
        volumes_mm3=volumes,
        fractions=fractions,
        fat_sensitivity_masks=sensitivity,
        overlap_conflict_mask=resolved.conflict_mask,
        input_overlap_voxels=resolved.input_overlap_voxels,
        overlap_priority=resolved.priority,
        thresholds=thresholds,
        spacing_xyz=spacing,
        acquisition_metadata=metadata,
        qc_flags=tuple(flags),
    )


# Short, discoverable alias.
classify_composition = classify_tissue_composition


__all__ = [
    "CompositionResult",
    "DEFAULT_OVERLAP_PRIORITY",
    "TISSUE_NAMES",
    "TissueLabel",
    "TissueThresholds",
    "classify_composition",
    "classify_tissue_composition",
    "composition_statistics",
]
