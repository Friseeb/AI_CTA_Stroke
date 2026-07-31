"""Machine-readable quality control for perivascular measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .composition import CompositionResult
from .contact import SurfaceContactResult
from .coordinates import VesselCoordinateSystem
from .shells import ShellSetResult


class QCSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class QCFlag:
    code: str
    severity: QCSeverity
    message: str
    metric: float | int | str | None = None
    threshold: float | int | str | None = None


@dataclass(frozen=True)
class PerivascularQCReport:
    status: str
    flags: tuple[QCFlag, ...]
    metrics: dict[str, float | int | str]

    @property
    def requires_manual_review(self) -> bool:
        return any(flag.severity in (QCSeverity.WARNING, QCSeverity.ERROR) for flag in self.flags)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(flag.code for flag in self.flags)


def coordinate_qc(
    system: VesselCoordinateSystem,
    *,
    orthonormal_tolerance: float = 1.0e-5,
) -> tuple[list[QCFlag], dict[str, float | int | str]]:
    tangent_norm_error = float(np.max(np.abs(np.linalg.norm(system.tangents_xyz, axis=1) - 1.0)))
    normal_norm_error = float(np.max(np.abs(np.linalg.norm(system.normals_xyz, axis=1) - 1.0)))
    binormal_norm_error = float(np.max(np.abs(np.linalg.norm(system.binormals_xyz, axis=1) - 1.0)))
    orthogonality = max(
        float(np.max(np.abs(np.einsum("ij,ij->i", system.tangents_xyz, system.normals_xyz)))),
        float(np.max(np.abs(np.einsum("ij,ij->i", system.tangents_xyz, system.binormals_xyz)))),
        float(np.max(np.abs(np.einsum("ij,ij->i", system.normals_xyz, system.binormals_xyz)))),
    )
    maximum_error = max(tangent_norm_error, normal_norm_error, binormal_norm_error, orthogonality)
    flags: list[QCFlag] = []
    if maximum_error > orthonormal_tolerance:
        flags.append(
            QCFlag(
                "non_orthonormal_vessel_frame",
                QCSeverity.ERROR,
                "Vessel frame exceeds the orthonormality tolerance.",
                maximum_error,
                orthonormal_tolerance,
            )
        )
    if system.length_mm <= 0:
        flags.append(QCFlag("zero_vessel_length", QCSeverity.ERROR, "Centreline has zero length."))
    return flags, {
        "vessel_length_mm": system.length_mm,
        "frame_maximum_orthonormal_error": maximum_error,
        "centreline_samples": len(system.path_mm),
        "branch_points": len(system.branch_path_mm),
    }


def shell_qc(
    shells: ShellSetResult,
    *,
    maximum_boundary_truncation_fraction: float = 0.05,
    maximum_other_artery_fraction: float = 0.10,
) -> tuple[list[QCFlag], dict[str, float | int | str]]:
    flags: list[QCFlag] = []
    maximum_truncation = max(
        (band.boundary_truncation_fraction for band in shells.bands.values()), default=0.0
    )
    maximum_other = max(
        (band.exclusion_fractions.get("other_artery", 0.0) for band in shells.bands.values()), default=0.0
    )
    if maximum_truncation > maximum_boundary_truncation_fraction:
        flags.append(
            QCFlag(
                "shell_truncated_by_image_boundary",
                QCSeverity.WARNING,
                "A radial shell is materially truncated by the acquired field of view.",
                maximum_truncation,
                maximum_boundary_truncation_fraction,
            )
        )
    if maximum_other > maximum_other_artery_fraction:
        flags.append(
            QCFlag(
                "adjacent_artery_contamination",
                QCSeverity.WARNING,
                "A radial shell substantially overlaps another artery before filtering.",
                maximum_other,
                maximum_other_artery_fraction,
            )
        )
    empty = sum(band.valid_voxels == 0 for band in shells.bands.values())
    if empty:
        flags.append(
            QCFlag(
                "empty_valid_shell",
                QCSeverity.ERROR,
                "One or more shells have no valid voxels.",
                empty,
                0,
            )
        )
    masks = [band.raw_mask for band in shells.bands.values()]
    overlap_voxels = 0
    if len(masks) > 1:
        overlap_voxels = int((np.stack(masks, axis=0).sum(axis=0) > 1).sum())
        if overlap_voxels:
            flags.append(
                QCFlag(
                    "radial_band_overlap",
                    QCSeverity.ERROR,
                    "Configured radial bands overlap on the native grid.",
                    overlap_voxels,
                    0,
                )
            )
    return flags, {
        "shell_count": len(shells.bands),
        "maximum_boundary_truncation_fraction": maximum_truncation,
        "maximum_other_artery_exclusion_fraction": maximum_other,
        "empty_valid_shells": empty,
        "radial_band_overlap_voxels": overlap_voxels,
    }


def composition_qc(
    composition: CompositionResult,
    *,
    maximum_uncertain_fraction: float = 0.25,
    maximum_bone_fraction: float = 0.50,
    maximum_vein_fraction: float = 0.50,
    minimum_fat_voxels: int = 20,
) -> tuple[list[QCFlag], dict[str, float | int | str]]:
    labels = composition.labels
    roi_count = int(composition.roi_mask.sum())
    denominator = int(composition.valid_denominator_mask.sum())

    def raw_fraction(label: int) -> float:
        return (
            float(((labels == label) & composition.roi_mask).sum() / roi_count)
            if roi_count
            else float("nan")
        )

    uncertain = raw_fraction(11)
    bone = raw_fraction(4)
    vein = raw_fraction(3)
    fat_voxels = int(((labels == 8) & composition.valid_denominator_mask).sum())
    flags: list[QCFlag] = []
    if not denominator:
        flags.append(
            QCFlag(
                "empty_valid_tissue_denominator",
                QCSeverity.ERROR,
                "No assessable tissue remains.",
            )
        )
    if np.isfinite(uncertain) and uncertain > maximum_uncertain_fraction:
        flags.append(
            QCFlag(
                "excessive_uncertain_fraction",
                QCSeverity.WARNING,
                "The uncertain compartment exceeds its review threshold.",
                uncertain,
                maximum_uncertain_fraction,
            )
        )
    if np.isfinite(bone) and bone > maximum_bone_fraction:
        flags.append(
            QCFlag(
                "excessive_bone_contamination",
                QCSeverity.WARNING,
                "Bone dominates the shell.",
                bone,
                maximum_bone_fraction,
            )
        )
    if np.isfinite(vein) and vein > maximum_vein_fraction:
        flags.append(
            QCFlag(
                "excessive_venous_contamination",
                QCSeverity.WARNING,
                "Vein dominates the shell.",
                vein,
                maximum_vein_fraction,
            )
        )
    if fat_voxels < minimum_fat_voxels:
        flags.append(
            QCFlag(
                "insufficient_fat_voxels",
                QCSeverity.WARNING,
                "PVAT attenuation is unstable at this voxel count.",
                fat_voxels,
                minimum_fat_voxels,
            )
        )
    fraction_sum = float(
        sum(
            value
            for name, value in composition.fractions.items()
            if name != "valid_denominator" and np.isfinite(value)
        )
    )
    if denominator and not np.isclose(fraction_sum, 1.0, atol=1.0e-8):
        flags.append(
            QCFlag(
                "implausible_tissue_fraction_sum",
                QCSeverity.ERROR,
                "Mutually exclusive tissue fractions do not sum to one.",
                fraction_sum,
                1.0,
            )
        )
    return flags, {
        "raw_roi_voxels": roi_count,
        "valid_tissue_voxels": denominator,
        "uncertain_fraction": uncertain,
        "bone_fraction": bone,
        "vein_fraction": vein,
        "fat_voxels": fat_voxels,
        "valid_tissue_fraction_sum": fraction_sum,
        "input_overlap_voxels": composition.input_overlap_voxels,
    }


def contact_qc(
    contacts: SurfaceContactResult,
    *,
    maximum_boundary_truncation_fraction: float = 0.05,
    maximum_ambiguous_contact_fraction: float = 0.10,
) -> tuple[list[QCFlag], dict[str, float | int | str]]:
    flags: list[QCFlag] = []
    maximum_boundary = max(
        (result.boundary_truncation_fraction for result in contacts.distances.values()), default=0.0
    )
    maximum_ambiguity = 0.0
    for result in contacts.distances.values():
        if result.assessable_surface_area_mm2:
            maximum_ambiguity = max(
                maximum_ambiguity,
                result.ambiguous_contact_area_mm2 / result.assessable_surface_area_mm2,
            )
    if maximum_boundary > maximum_boundary_truncation_fraction:
        flags.append(
            QCFlag(
                "surface_truncated_by_image_boundary",
                QCSeverity.WARNING,
                "Arterial surface-contact denominator is truncated by the field of view.",
                maximum_boundary,
                maximum_boundary_truncation_fraction,
            )
        )
    if maximum_ambiguity > maximum_ambiguous_contact_fraction:
        flags.append(
            QCFlag(
                "ambiguous_multi_class_contact",
                QCSeverity.WARNING,
                "Multiple tissue classes fall within the requested contact distance.",
                maximum_ambiguity,
                maximum_ambiguous_contact_fraction,
            )
        )
    return flags, {
        "contact_distance_count": len(contacts.distances),
        "maximum_surface_boundary_truncation_fraction": maximum_boundary,
        "maximum_ambiguous_contact_fraction": maximum_ambiguity,
    }


def perivascular_qc(
    *,
    coordinate_system: VesselCoordinateSystem | None = None,
    shells: ShellSetResult | None = None,
    composition: CompositionResult | None = None,
    contacts: SurfaceContactResult | None = None,
) -> PerivascularQCReport:
    """Combine available module QC without requiring every analysis stage."""
    flags: list[QCFlag] = []
    metrics: dict[str, float | int | str] = {}
    for prefix, value, evaluator in (
        ("coordinate", coordinate_system, coordinate_qc),
        ("shell", shells, shell_qc),
        ("composition", composition, composition_qc),
        ("contact", contacts, contact_qc),
    ):
        if value is None:
            continue
        stage_flags, stage_metrics = evaluator(value)  # type: ignore[arg-type]
        flags.extend(stage_flags)
        metrics.update({f"{prefix}.{name}": metric for name, metric in stage_metrics.items()})
    severity_rank = {QCSeverity.INFO: 0, QCSeverity.WARNING: 1, QCSeverity.ERROR: 2}
    flags.sort(key=lambda flag: (-severity_rank[flag.severity], flag.code))
    status = "fail" if any(flag.severity == QCSeverity.ERROR for flag in flags) else (
        "review" if any(flag.severity == QCSeverity.WARNING for flag in flags) else "pass"
    )
    return PerivascularQCReport(status=status, flags=tuple(flags), metrics=metrics)


__all__ = [
    "PerivascularQCReport",
    "QCFlag",
    "QCSeverity",
    "composition_qc",
    "contact_qc",
    "coordinate_qc",
    "perivascular_qc",
    "shell_qc",
]
