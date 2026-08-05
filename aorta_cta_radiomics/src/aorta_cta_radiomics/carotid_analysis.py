"""Fail-closed first-milestone carotid perivascular analysis.

The module joins validated FLOWCAT outputs to the deterministic perivascular
primitives.  FLOWCAT/NIfTI arrays remain in nibabel ``(i, j, k)`` order until
an explicit transpose into the perivascular ``(k, j, i)`` (called ZYX) order.
Physical coordinates remain NIfTI RAS millimetres throughout.

No carotid-bulb label is inferred.  A bulb can only appear as a caller-supplied
and evidence-bearing longitudinal interval on an already labelled CCA or ICA
branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import platform
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from . import __version__
from .artery_volume import (
    CenterlineSeed,
    LabelPropagationResult,
    generate_anatomical_artery_volume,
)
from .flowcat_adapter import (
    CenterlineBranch,
    CoordinateReference,
    ImageGeometry,
    ValidatedFlowcatCase,
)
from .flowcat_adapter.labels import FLOWCAT_ARTERY_CLASSES
from .output_schema import (
    CaseLevelOutput,
    SamplePointLevelOutput,
    VesselLevelOutput,
    configuration_sha256,
)
from .perivascular.composition import (
    CompositionResult,
    TissueLabel,
    TissueThresholds,
    classify_tissue_composition,
)
from .perivascular.contact import SurfaceContactResult, calculate_surface_contacts
from .perivascular.coordinates import VesselCoordinateSystem
from .perivascular.exclusions import physical_dilation
from .perivascular.qc import PerivascularQCReport, perivascular_qc
from .perivascular.radial_profiles import (
    aggregate_longitudinal_profiles,
    aggregate_radial_profiles,
)
from .perivascular.sectors import SectorAssignment, assign_angular_sectors, summarize_sectors
from .perivascular.shells import (
    AdaptiveShellLaw,
    DEFAULT_RADIAL_BANDS,
    RadialBand,
    ShellSetResult,
    generate_adaptive_shell,
    generate_radial_shells,
)


CAROTID_FLOWCAT_CODES = frozenset({"RCCA", "LCCA", "RICA", "LICA", "RECA", "LECA"})
COARSE_ANATOMY_CONTEXT_NAMES = (
    "vein",
    "bone",
    "airway",
    "thyroid_gland",
    "skeletal_muscle",
)

# Derived volume 0 is background and 14 is uncertainty.  The documented
# project convention collapses FLOWCAT class 0 (other artery) and class 13
# (basilar artery) into derived label 13.  Their source graph codes remain on
# CenterlineSeed.label_name and the typed FLOWCAT branches, so the collapse is
# never presented as source-label equivalence.
FLOWCAT_CLASS_TO_DERIVED_VOLUME_LABEL: dict[int, int] = {
    **{class_id: class_id for class_id in range(1, 13)},
    0: 13,
    13: 13,
}


class CarotidAnalysisError(ValueError):
    """The validated inputs cannot support a non-speculative analysis."""


@dataclass(frozen=True)
class PerivascularArrayGeometry:
    """Geometry for KJI/ZYX arrays with physical coordinates in RAS mm."""

    spacing_xyz_mm: tuple[float, float, float]
    origin_ras_xyz_mm: tuple[float, float, float]
    direction_ijk_to_ras: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


def nibabel_ijk_to_perivascular_zyx(array_ijk: np.ndarray) -> np.ndarray:
    """Transpose a 3D nibabel IJK array to perivascular KJI/ZYX order."""
    array = np.asarray(array_ijk)
    if array.ndim != 3:
        raise CarotidAnalysisError("A nibabel IJK array must be three-dimensional.")
    return np.transpose(array, (2, 1, 0))


def perivascular_zyx_to_nibabel_ijk(array_zyx: np.ndarray) -> np.ndarray:
    """Transpose a perivascular KJI/ZYX array back to nibabel IJK order."""
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise CarotidAnalysisError("A perivascular ZYX array must be three-dimensional.")
    return np.transpose(array, (2, 1, 0))


def perivascular_geometry_from_nifti(geometry: ImageGeometry) -> PerivascularArrayGeometry:
    """Decompose a nibabel RAS affine into spacing, origin, and direction.

    Sheared affines fail closed because physical EDT is only valid for an
    orthogonal voxel lattice.  Rotations, reflections, and axis permutations
    remain supported.
    """
    affine = np.asarray(geometry.affine_ras_mm, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise CarotidAnalysisError("CTA affine must be a finite 4x4 matrix.")
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise CarotidAnalysisError("CTA affine contains an invalid voxel spacing.")
    direction = linear / spacing[None, :]
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1.0e-5, rtol=0.0):
        raise CarotidAnalysisError(
            "CTA affine contains shear; resample to an orthogonal grid before perivascular analysis."
        )
    return PerivascularArrayGeometry(
        spacing_xyz_mm=tuple(float(value) for value in spacing),
        origin_ras_xyz_mm=tuple(float(value) for value in affine[:3, 3]),
        direction_ijk_to_ras=tuple(
            tuple(float(value) for value in row) for row in direction
        ),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class CoarseAnatomyMasksIJK:
    """Optional coarse masks in the same nibabel IJK grid as the CTA."""

    vein: np.ndarray | None = None
    bone: np.ndarray | None = None
    airway: np.ndarray | None = None
    thyroid_gland: np.ndarray | None = None
    skeletal_muscle: np.ndarray | None = None
    uncertain: np.ndarray | None = None
    high_density: np.ndarray | None = None

    def availability(self) -> dict[str, bool]:
        """Record model/mask availability; an empty supplied mask is available."""
        return {
            "vein": self.vein is not None,
            "bone": self.bone is not None,
            "airway": self.airway is not None,
            "thyroid_gland": self.thyroid_gland is not None,
            "skeletal_muscle": self.skeletal_muscle is not None,
            "uncertain_mask": self.uncertain is not None,
            "high_density_mask": self.high_density is not None,
        }

    def validated_zyx(self, shape_ijk: tuple[int, int, int]) -> dict[str, np.ndarray | None]:
        converted: dict[str, np.ndarray | None] = {}
        for name, value in (
            ("vein_mask", self.vein),
            ("bone_mask", self.bone),
            ("airway_mask", self.airway),
            ("thyroid_mask", self.thyroid_gland),
            ("muscle_mask", self.skeletal_muscle),
            ("uncertain_mask", self.uncertain),
            ("high_density_mask", self.high_density),
        ):
            if value is None:
                converted[name] = None
                continue
            mask = np.asarray(value)
            if mask.shape != shape_ijk:
                raise CarotidAnalysisError(
                    f"Coarse anatomy {name} has IJK shape {mask.shape}, expected {shape_ijk}."
                )
            if mask.ndim != 3 or not np.isfinite(mask).all():
                raise CarotidAnalysisError(f"Coarse anatomy {name} must be a finite 3D mask.")
            converted[name] = nibabel_ijk_to_perivascular_zyx(mask != 0)
        return converted


@dataclass(frozen=True)
class SupportedBulbInterval:
    """Evidence-backed longitudinal annotation; never a generated vessel label."""

    flowcat_code: str
    start_path_mm: float
    end_path_mm: float
    support_source: str
    support_reference: str

    def __post_init__(self) -> None:
        code = self.flowcat_code.upper()
        object.__setattr__(self, "flowcat_code", code)
        if code not in {"RCCA", "LCCA", "RICA", "LICA"}:
            raise CarotidAnalysisError("A bulb interval must refer to a labelled CCA or ICA branch.")
        if (
            not np.isfinite(self.start_path_mm)
            or not np.isfinite(self.end_path_mm)
            or self.start_path_mm < 0
            or self.end_path_mm <= self.start_path_mm
        ):
            raise CarotidAnalysisError("A bulb interval requires 0 <= start_path_mm < end_path_mm.")
        if not self.support_source.strip() or not self.support_reference.strip():
            raise CarotidAnalysisError(
                "A bulb interval requires explicit support_source and support_reference."
            )

    def to_record(self) -> dict[str, object]:
        return {
            "name": "configured_carotid_bulb_interval",
            "flowcat_code": self.flowcat_code,
            "start_path_mm": self.start_path_mm,
            "end_path_mm": self.end_path_mm,
            "support_source": self.support_source,
            "support_reference": self.support_reference,
            "is_inferred": False,
        }


@dataclass(frozen=True)
class CarotidAnalysisConfig:
    required_flowcat_codes: tuple[str, ...] = ("RCCA", "LCCA", "RICA", "LICA")
    coordinate_interval_mm: float = 1.0
    fixed_radial_bands: tuple[RadialBand, ...] = DEFAULT_RADIAL_BANDS
    adaptive_law: AdaptiveShellLaw = AdaptiveShellLaw(
        alpha=0.75, minimum_mm=1.0, maximum_mm=5.0
    )
    partial_volume_boundary_mm: float = 0.0
    bifurcation_radius_mm: float = 3.0
    bifurcation_shell_exclusion_mm: float = 3.0
    propagation_minimum_confidence: float = 0.01
    sector_count: int = 8
    contact_distances_mm: tuple[float, ...] = (0.5, 1.0, 2.0)
    primary_contact_distance_mm: float = 0.5
    longitudinal_bin_mm: float = 1.0
    tissue_thresholds: TissueThresholds = TissueThresholds()
    bulb_intervals: tuple[SupportedBulbInterval, ...] = ()
    acquisition_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        codes = tuple(str(code).upper() for code in self.required_flowcat_codes)
        object.__setattr__(self, "required_flowcat_codes", codes)
        if not codes or len(codes) != len(set(codes)):
            raise CarotidAnalysisError("required_flowcat_codes must be non-empty and unique.")
        unsupported = set(codes) - CAROTID_FLOWCAT_CODES
        if unsupported:
            raise CarotidAnalysisError(f"Unsupported carotid FLOWCAT codes: {sorted(unsupported)}")
        if self.coordinate_interval_mm <= 0 or self.longitudinal_bin_mm <= 0:
            raise CarotidAnalysisError("Coordinate and longitudinal intervals must be positive.")
        if self.bifurcation_radius_mm < 0 or self.bifurcation_shell_exclusion_mm < 0:
            raise CarotidAnalysisError("Bifurcation exclusion radii must be non-negative.")
        if self.partial_volume_boundary_mm < 0:
            raise CarotidAnalysisError("partial_volume_boundary_mm must be non-negative.")
        if self.primary_contact_distance_mm not in self.contact_distances_mm:
            raise CarotidAnalysisError(
                "primary_contact_distance_mm must be one of contact_distances_mm."
            )
        interval_codes = [interval.flowcat_code for interval in self.bulb_intervals]
        if any(code not in codes for code in interval_codes):
            raise CarotidAnalysisError(
                "Every configured bulb interval must refer to a required analysis branch."
            )
        if len(interval_codes) != len(set(interval_codes)):
            raise CarotidAnalysisError("Only one configured bulb interval is allowed per branch.")


@dataclass(frozen=True)
class CarotidBranchAnalysis:
    flowcat_code: str
    branch: CenterlineBranch
    derived_volume_label: int
    target_lumen_zyx: np.ndarray
    coordinate_system: VesselCoordinateSystem
    fixed_shells: ShellSetResult
    adaptive_shell: ShellSetResult
    composition: CompositionResult
    sector_assignment: SectorAssignment
    radial_profiles: tuple[dict[str, object], ...]
    longitudinal_profiles: tuple[dict[str, object], ...]
    sector_summaries: tuple[dict[str, object], ...]
    contacts: SurfaceContactResult
    fixed_qc: PerivascularQCReport
    adaptive_qc: PerivascularQCReport
    coarse_anatomy_availability: dict[str, bool]
    configured_bulb_interval: SupportedBulbInterval | None
    vessel_output: VesselLevelOutput
    sample_outputs: tuple[SamplePointLevelOutput, ...]


@dataclass(frozen=True)
class CarotidCaseAnalysis:
    case_id: str
    anatomical_artery_volume_ijk: LabelPropagationResult
    branches: dict[str, CarotidBranchAnalysis]
    case_output: CaseLevelOutput
    vessel_outputs: tuple[VesselLevelOutput, ...]
    sample_outputs: tuple[SamplePointLevelOutput, ...]
    coarse_anatomy_availability: dict[str, bool]
    visual_review_required: bool = True
    array_order_contract: str = "FLOWCAT/nibabel IJK -> explicit transpose -> perivascular KJI/ZYX"


def flowcat_branch_to_centerline_seed(branch: CenterlineBranch) -> CenterlineSeed:
    """Convert one explicitly labelled physical-RAS FLOWCAT branch for propagation."""
    if branch.artery is None:
        raise CarotidAnalysisError(f"Branch {branch.branch_id!r} has no anatomical FLOWCAT label.")
    if branch.coordinate_reference is not CoordinateReference.NIFTI_RAS_PHYSICAL_MM:
        raise CarotidAnalysisError(
            f"Branch {branch.branch_id!r} is not in NIfTI RAS physical millimetres."
        )
    class_id = int(branch.artery.flowcat_class_id)
    if class_id not in FLOWCAT_CLASS_TO_DERIVED_VOLUME_LABEL:
        raise CarotidAnalysisError(
            f"Branch {branch.branch_id!r} has unsupported FLOWCAT class {class_id}."
        )
    expected_code = FLOWCAT_ARTERY_CLASSES[class_id][0]
    if branch.artery.flowcat_code != expected_code:
        raise CarotidAnalysisError(
            f"Branch {branch.branch_id!r} class/code mismatch: {class_id} is {expected_code}."
        )
    points = np.asarray([point.coordinates_mm for point in branch.points], dtype=float)
    radii = np.asarray([point.local_radius_mm for point in branch.points], dtype=object)
    if len(points) < 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise CarotidAnalysisError(f"Branch {branch.branch_id!r} has an invalid centreline.")
    if any(value is None for value in radii):
        raise CarotidAnalysisError(
            f"Branch {branch.branch_id!r} lacks local radii required for adaptive shells."
        )
    radii_float = radii.astype(float)
    if not np.isfinite(radii_float).all() or np.any(radii_float <= 0):
        raise CarotidAnalysisError(
            f"Branch {branch.branch_id!r} radii must be finite and strictly positive."
        )
    confidence = branch.artery.confidence if branch.artery.confidence is not None else 1.0
    return CenterlineSeed(
        branch_id=branch.branch_id,
        label_id=FLOWCAT_CLASS_TO_DERIVED_VOLUME_LABEL[class_id],
        label_name=branch.artery.flowcat_code,
        points_ras_mm=points,
        radii_mm=radii_float,
        label_confidence=float(confidence),
    )


def _validated_inputs(
    validated_case: ValidatedFlowcatCase,
    cta_hu_ijk: np.ndarray | None,
    config: CarotidAnalysisConfig,
) -> tuple[np.ndarray, ImageGeometry, dict[str, CenterlineBranch], tuple[CenterlineSeed, ...]]:
    if validated_case.cta_geometry is None:
        raise CarotidAnalysisError("Validated FLOWCAT case has no CTA geometry.")
    if cta_hu_ijk is None:
        raise CarotidAnalysisError("CTA voxel data are required; geometry alone is insufficient.")
    cta = np.asarray(cta_hu_ijk, dtype=float)
    geometry = validated_case.cta_geometry
    if cta.shape != geometry.shape_ijk:
        raise CarotidAnalysisError(
            f"CTA IJK shape {cta.shape} does not match validated geometry {geometry.shape_ijk}."
        )
    if not np.isfinite(cta).all():
        raise CarotidAnalysisError("CTA contains NaN or infinite values.")
    segmentation = np.asarray(validated_case.segmentation.data, dtype=bool)
    if segmentation.shape != geometry.shape_ijk:
        raise CarotidAnalysisError("FLOWCAT segmentation and CTA geometry have different IJK shapes.")
    if not segmentation.any():
        raise CarotidAnalysisError("FLOWCAT arterial segmentation is empty.")
    if not validated_case.branches:
        raise CarotidAnalysisError("Validated FLOWCAT case has no centreline branches.")
    seeds = tuple(flowcat_branch_to_centerline_seed(branch) for branch in validated_case.branches)
    by_code: dict[str, list[CenterlineBranch]] = {}
    for branch in validated_case.branches:
        assert branch.artery is not None  # enforced while constructing seeds
        by_code.setdefault(branch.artery.flowcat_code, []).append(branch)
    selected: dict[str, CenterlineBranch] = {}
    for code in config.required_flowcat_codes:
        matches = by_code.get(code, [])
        if not matches:
            raise CarotidAnalysisError(f"Required anatomical branch {code} is unavailable.")
        if len(matches) != 1:
            raise CarotidAnalysisError(
                f"Required anatomical branch {code} maps to {len(matches)} centreline branches; "
                "the first-milestone core will not guess how to merge them."
            )
        selected[code] = matches[0]
    return cta, geometry, selected, seeds


def _junction_points(
    branch: CenterlineBranch,
    all_branches: Sequence[CenterlineBranch],
    tolerance_mm: float = 2.0,
) -> np.ndarray:
    target = np.asarray([point.coordinates_mm for point in branch.points], dtype=float)
    endpoints = target[[0, -1]]
    junctions: list[np.ndarray] = []
    for other in all_branches:
        if other.branch_id == branch.branch_id:
            continue
        other_points = np.asarray([point.coordinates_mm for point in other.points], dtype=float)
        other_endpoints = other_points[[0, -1]]
        distances = np.linalg.norm(endpoints[:, None, :] - other_endpoints[None, :, :], axis=2)
        first, second = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[first, second] <= tolerance_mm:
            junctions.append(0.5 * (endpoints[first] + other_endpoints[second]))
    return (
        np.unique(np.round(np.asarray(junctions), decimals=6), axis=0)
        if junctions
        else np.empty((0, 3), dtype=float)
    )


def _diameter_map_ijk(
    target_lumen_ijk: np.ndarray,
    affine_ras_mm: np.ndarray,
    branch: CenterlineBranch,
) -> np.ndarray:
    indices = np.argwhere(target_lumen_ijk)
    if not len(indices):
        raise CarotidAnalysisError(f"Derived lumen for {branch.branch_id!r} is empty.")
    physical = (
        np.column_stack((indices.astype(float), np.ones(len(indices)))) @ affine_ras_mm.T
    )[:, :3]
    points = np.asarray([point.coordinates_mm for point in branch.points], dtype=float)
    radii = np.asarray([float(point.local_radius_mm) for point in branch.points], dtype=float)
    _, nearest = cKDTree(points).query(physical, k=1, workers=-1)
    diameter = np.zeros(target_lumen_ijk.shape, dtype=np.float32)
    diameter[tuple(indices.T)] = (2.0 * radii[nearest]).astype(np.float32)
    return diameter


def _union(masks: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for mask in masks.values():
        result |= np.asarray(mask, dtype=bool)
    return result


def _local_band_measurements(
    local_roi: np.ndarray,
    fixed_masks: Mapping[str, np.ndarray],
    valid_denominator: np.ndarray,
    tissue_labels: np.ndarray,
    cta_values: np.ndarray,
) -> dict[str, object]:
    records: dict[str, object] = {}
    for name, mask in fixed_masks.items():
        selected = local_roi & mask
        valid = selected & valid_denominator
        fat = valid & (tissue_labels == int(TissueLabel.ADIPOSE))
        fat_values = cta_values[fat]
        records[name] = {
            "voxels": int(selected.sum()),
            "valid_voxels": int(valid.sum()),
            "fat_fraction": float(fat.sum() / valid.sum()) if valid.any() else None,
            "mean_fat_attenuation_hu": float(np.mean(fat_values)) if fat_values.size else None,
        }
    return records


def _local_sector_measurements(
    path_mm: float,
    local_roi: np.ndarray,
    sector_labels: np.ndarray,
    valid_denominator: np.ndarray,
    tissue_labels: np.ndarray,
    assignment: SectorAssignment,
    interval: SupportedBulbInterval | None,
) -> dict[str, object]:
    sectors: dict[str, object] = {}
    for sector in range(1, assignment.number_of_sectors + 1):
        selected = local_roi & (sector_labels == sector) & valid_denominator
        fat = selected & (tissue_labels == int(TissueLabel.ADIPOSE))
        sectors[str(sector)] = {
            "valid_voxels": int(selected.sum()),
            "fat_fraction": float(fat.sum() / selected.sum()) if selected.any() else None,
        }
    intervals = []
    if interval is not None and interval.start_path_mm <= path_mm <= interval.end_path_mm:
        intervals.append("configured_carotid_bulb_interval")
    return {
        "number_of_sectors": assignment.number_of_sectors,
        "orientation_reference": assignment.orientation_reference,
        "sectors": sectors,
        "longitudinal_intervals": intervals,
    }


def _local_contact_measurements(
    path_mm: float,
    contacts: SurfaceContactResult,
    primary_distance_mm: float,
) -> dict[str, object]:
    distance = contacts.at(primary_distance_mm)
    result: dict[str, object] = {"distance_mm": primary_distance_mm, "classes": {}}
    classes = result["classes"]
    assert isinstance(classes, dict)
    for name, metrics in distance.classes.items():
        match = next(
            (
                record
                for record in metrics.longitudinal_profile
                if float(record["path_start_mm"]) <= path_mm <= float(record["path_end_mm"])
            ),
            None,
        )
        classes[name] = dict(match) if match is not None else {"contact_fraction": None}
    return result


def _radial_gradient(
    profiles: Sequence[Mapping[str, object]],
    bands: Sequence[RadialBand],
    key: str,
) -> float | None:
    x = np.asarray([0.5 * (band.inner_mm + band.outer_mm) for band in bands], dtype=float)
    y = np.asarray([float(record.get(key, np.nan)) for record in profiles], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return None
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def _assessable_length(system: VesselCoordinateSystem, valid_by_sample: np.ndarray) -> float:
    if len(system.path_mm) < 2:
        return 0.0
    pair_validity = 0.5 * (valid_by_sample[:-1].astype(float) + valid_by_sample[1:].astype(float))
    return float(np.sum(np.diff(system.path_mm) * pair_validity))


def _build_output_records(
    *,
    case_id: str,
    code: str,
    branch: CenterlineBranch,
    system: VesselCoordinateSystem,
    fixed_shells: ShellSetResult,
    composition: CompositionResult,
    assignment: SectorAssignment,
    radial_profiles: Sequence[dict[str, object]],
    longitudinal_profiles: Sequence[dict[str, object]],
    sector_summaries: Sequence[dict[str, object]],
    contacts: SurfaceContactResult,
    qc_reports: Sequence[PerivascularQCReport],
    cta_zyx: np.ndarray,
    config: CarotidAnalysisConfig,
    bulb_interval: SupportedBulbInterval | None,
    coarse_anatomy_availability: Mapping[str, bool],
) -> tuple[VesselLevelOutput, tuple[SamplePointLevelOutput, ...]]:
    source_path = np.asarray([point.path_distance_mm for point in branch.points], dtype=float)
    source_radius = np.asarray([float(point.local_radius_mm) for point in branch.points], dtype=float)
    local_radius = np.interp(system.path_mm, source_path, source_radius)
    half_width = max(0.5 * config.coordinate_interval_mm, 0.25)
    sample_outputs: list[SamplePointLevelOutput] = []
    sample_valid = np.zeros(len(system.path_mm), dtype=bool)
    sample_uncertain = np.full(len(system.path_mm), np.nan, dtype=float)
    missing_context_flags = [
        f"coarse_anatomy_{name}_unavailable"
        for name in COARSE_ANATOMY_CONTEXT_NAMES
        if not coarse_anatomy_availability[name]
    ]
    combined_qc_codes = sorted(
        {
            "manual_visual_review_pending",
            *missing_context_flags,
            *(code for report in qc_reports for code in report.codes),
        }
    )
    # Sample-level summaries only query voxels with a finite vessel coordinate.
    # Compact those arrays once instead of scanning the full CTA grid for every
    # 1-mm centreline sample and every radial band/sector.
    sample_domain = np.isfinite(assignment.path_mm)
    sample_paths = np.asarray(assignment.path_mm[sample_domain], dtype=float)
    sample_valid_denominator = np.asarray(
        composition.valid_denominator_mask[sample_domain], dtype=bool
    )
    sample_tissue_labels = np.asarray(composition.labels[sample_domain])
    sample_cta_values = np.asarray(cta_zyx[sample_domain])
    sample_fixed_masks = {
        name: np.asarray(mask[sample_domain], dtype=bool)
        for name, mask in fixed_shells.filtered_masks.items()
    }
    sample_sector_labels = np.asarray(assignment.labels[sample_domain])
    for index, path_mm in enumerate(system.path_mm):
        local_roi = np.abs(sample_paths - path_mm) <= half_width
        radial = _local_band_measurements(
            local_roi,
            sample_fixed_masks,
            sample_valid_denominator,
            sample_tissue_labels,
            sample_cta_values,
        )
        valid_count = int((local_roi & sample_valid_denominator).sum())
        sample_valid[index] = valid_count > 0
        uncertain_count = int(
            (local_roi & (sample_tissue_labels == int(TissueLabel.UNCERTAIN))).sum()
        )
        roi_count = int(local_roi.sum())
        uncertain_fraction = float(uncertain_count / roi_count) if roi_count else None
        if uncertain_fraction is not None:
            sample_uncertain[index] = uncertain_fraction
        sample_flags = list(combined_qc_codes)
        if not sample_valid[index]:
            sample_flags.append("missing_local_perivascular_voxels")
        distance_to_branch = system.distance_to_bifurcation_mm(np.asarray([path_mm]))[0]
        sample_outputs.append(
            SamplePointLevelOutput(
                case_id=case_id,
                vessel_name=code,
                laterality=branch.artery.laterality if branch.artery else None,
                branch_id=branch.branch_id,
                path_distance_mm=float(path_mm),
                physical_ras_x_mm=float(system.points_xyz[index, 0]),
                physical_ras_y_mm=float(system.points_xyz[index, 1]),
                physical_ras_z_mm=float(system.points_xyz[index, 2]),
                tangent_ras=tuple(float(value) for value in system.tangents_xyz[index]),
                first_normal_ras=tuple(float(value) for value in system.normals_xyz[index]),
                second_normal_ras=tuple(float(value) for value in system.binormals_xyz[index]),
                local_radius_mm=float(local_radius[index]),
                radius_method="FLOWCAT_maximal_inscribed_sphere",
                local_diameter_mm=float(2.0 * local_radius[index]),
                diameter_method="2_x_FLOWCAT_maximal_inscribed_sphere_radius",
                curvature_per_mm=float(system.curvature_per_mm[index]),
                torsion_per_mm=(
                    float(system.torsion_per_mm[index])
                    if np.isfinite(system.torsion_per_mm[index])
                    else None
                ),
                distance_to_bifurcation_mm=(
                    float(distance_to_branch) if np.isfinite(distance_to_branch) else None
                ),
                radial_band_measurements=radial,
                sector_measurements=_local_sector_measurements(
                    float(path_mm),
                    local_roi,
                    sample_sector_labels,
                    sample_valid_denominator,
                    sample_tissue_labels,
                    assignment,
                    bulb_interval,
                ),
                surface_contact_measurements=_local_contact_measurements(
                    float(path_mm), contacts, config.primary_contact_distance_mm
                ),
                tissue_confidence=(1.0 - uncertain_fraction if uncertain_fraction is not None else None),
                uncertain_fraction=uncertain_fraction,
                qc_flags=sorted(set(sample_flags)),
            )
        )

    fixed_raw = _union(fixed_shells.raw_masks, composition.labels.shape)
    fixed_valid = _union(fixed_shells.filtered_masks, composition.labels.shape)
    tissue_denominator = fixed_valid & composition.valid_denominator_mask
    denominator_count = int(tissue_denominator.sum())
    fat = tissue_denominator & (composition.labels == int(TissueLabel.ADIPOSE))
    soft = tissue_denominator & (composition.labels == int(TissueLabel.OTHER_SOFT_TISSUE))
    uncertain = fixed_raw & (composition.labels == int(TissueLabel.UNCERTAIN))
    voxel_volume = float(np.prod(composition.spacing_xyz))
    primary_contact = contacts.at(config.primary_contact_distance_mm)
    chord = float(np.linalg.norm(system.points_xyz[-1] - system.points_xyz[0]))
    assessable = min(system.length_mm, _assessable_length(system, sample_valid))
    interval_records = [bulb_interval.to_record()] if bulb_interval is not None else []
    vessel_flags = sorted(set(combined_qc_codes))
    vessel_output = VesselLevelOutput(
        case_id=case_id,
        vessel_name=code,
        laterality=branch.artery.laterality if branch.artery else None,
        branch_id=branch.branch_id,
        vessel_label_confidence=branch.artery.confidence if branch.artery else None,
        total_length_mm=system.length_mm,
        assessable_length_mm=assessable,
        nonassessable_length_mm=max(0.0, system.length_mm - assessable),
        mean_radius_mm=float(np.mean(local_radius)),
        minimum_radius_mm=float(np.min(local_radius)),
        maximum_radius_mm=float(np.max(local_radius)),
        radius_method="FLOWCAT_maximal_inscribed_sphere",
        diameter_method="2_x_FLOWCAT_maximal_inscribed_sphere_radius",
        tortuosity=float(system.length_mm / chord) if chord > 0 else None,
        mean_curvature_per_mm=float(np.mean(system.curvature_per_mm)),
        calcium_burden_mm3=None,
        total_shell_volume_mm3=float(fixed_raw.sum() * voxel_volume),
        valid_shell_volume_mm3=float(fixed_valid.sum() * voxel_volume),
        pvat_volume_mm3=float(fat.sum() * voxel_volume),
        pvat_fraction=float(fat.sum() / denominator_count) if denominator_count else None,
        mean_pvat_attenuation_hu=float(np.mean(cta_zyx[fat])) if fat.any() else None,
        perivascular_soft_tissue_volume_mm3=float(soft.sum() * voxel_volume),
        muscle_contact_fraction=(
            primary_contact.classes["skeletal_muscle"].contact_fraction
            if coarse_anatomy_availability["skeletal_muscle"]
            else None
        ),
        vein_contact_fraction=(
            primary_contact.classes["vein"].contact_fraction
            if coarse_anatomy_availability["vein"]
            else None
        ),
        bone_contact_fraction=(
            primary_contact.classes["bone"].contact_fraction
            if coarse_anatomy_availability["bone"]
            else None
        ),
        thyroid_contact_fraction=(
            primary_contact.classes["thyroid_gland"].contact_fraction
            if coarse_anatomy_availability["thyroid_gland"]
            else None
        ),
        other_soft_tissue_fraction=(
            float(soft.sum() / denominator_count) if denominator_count else None
        ),
        uncertain_fraction=float(uncertain.sum() / fixed_raw.sum()) if fixed_raw.any() else None,
        radial_gradients={
            "fat_fraction_per_mm": _radial_gradient(
                radial_profiles, config.fixed_radial_bands, "adipose_fraction"
            ),
            "fat_attenuation_hu_per_mm": _radial_gradient(
                radial_profiles, config.fixed_radial_bands, "adipose_mean_hu"
            ),
        },
        longitudinal_summaries={
            "profiles": list(longitudinal_profiles),
            "configured_intervals": interval_records,
            "coarse_anatomy_availability": dict(coarse_anatomy_availability),
        },
        circumferential_summaries={
            "number_of_sectors": assignment.number_of_sectors,
            "orientation_reference": assignment.orientation_reference,
            "profiles": list(sector_summaries),
            "coarse_anatomy_availability": dict(coarse_anatomy_availability),
        },
        qc_flags=vessel_flags,
        manual_review_required=True,
    )
    return vessel_output, tuple(sample_outputs)


def analyze_carotid_case(
    validated_case: ValidatedFlowcatCase,
    cta_hu_ijk: np.ndarray | None,
    *,
    config: CarotidAnalysisConfig = CarotidAnalysisConfig(),
    coarse_anatomy_ijk: CoarseAnatomyMasksIJK | None = None,
) -> CarotidCaseAnalysis:
    """Run the first carotid milestone from an already validated FLOWCAT case."""
    started = datetime.now(timezone.utc)
    cta_ijk, geometry, selected, seeds = _validated_inputs(validated_case, cta_hu_ijk, config)
    perivascular_geometry = perivascular_geometry_from_nifti(geometry)
    affine = np.asarray(geometry.affine_ras_mm, dtype=float)
    artery_volume = generate_anatomical_artery_volume(
        validated_case.segmentation.data,
        affine,
        seeds,
        bifurcation_radius_mm=config.bifurcation_radius_mm,
        minimum_confidence=config.propagation_minimum_confidence,
    )
    cta_zyx = nibabel_ijk_to_perivascular_zyx(cta_ijk)
    all_lumen_zyx = nibabel_ijk_to_perivascular_zyx(validated_case.segmentation.data)
    bifurcation_zyx = nibabel_ijk_to_perivascular_zyx(
        artery_volume.bifurcation_exclusion_mask
    )
    bifurcation_shell_zyx = physical_dilation(
        bifurcation_zyx,
        perivascular_geometry.spacing_xyz_mm,
        config.bifurcation_shell_exclusion_mm,
    )
    coarse_anatomy = coarse_anatomy_ijk or CoarseAnatomyMasksIJK()
    anatomy_availability = coarse_anatomy.availability()
    anatomy = coarse_anatomy.validated_zyx(geometry.shape_ijk)
    bulb_by_code = {interval.flowcat_code: interval for interval in config.bulb_intervals}
    branch_results: dict[str, CarotidBranchAnalysis] = {}
    vessel_outputs: list[VesselLevelOutput] = []
    sample_outputs: list[SamplePointLevelOutput] = []

    for code in config.required_flowcat_codes:
        branch = selected[code]
        assert branch.artery is not None
        label_id = FLOWCAT_CLASS_TO_DERIVED_VOLUME_LABEL[branch.artery.flowcat_class_id]
        target_ijk = artery_volume.labels == label_id
        if not target_ijk.any():
            raise CarotidAnalysisError(
                f"Anatomical artery-volume generation produced no lumen for required branch {code}."
            )
        target_zyx = nibabel_ijk_to_perivascular_zyx(target_ijk)
        other_artery_zyx = all_lumen_zyx & ~target_zyx
        branch_points = np.asarray([point.coordinates_mm for point in branch.points], dtype=float)
        junctions = _junction_points(branch, validated_case.branches)
        system = VesselCoordinateSystem.from_polyline(
            branch_points,
            vessel_id=code,
            interval_mm=config.coordinate_interval_mm,
            branch_points_xyz=junctions,
        )
        bulb = bulb_by_code.get(code)
        if bulb is not None and bulb.end_path_mm > system.length_mm + 1.0e-6:
            raise CarotidAnalysisError(
                f"Configured bulb interval for {code} ends at {bulb.end_path_mm:g} mm, "
                f"beyond branch length {system.length_mm:g} mm."
            )
        fixed = generate_radial_shells(
            target_zyx,
            perivascular_geometry.spacing_xyz_mm,
            vessel_id=code,
            bands=config.fixed_radial_bands,
            other_artery_mask=other_artery_zyx,
            bifurcation_exclusion_mask=bifurcation_shell_zyx,
            partial_volume_boundary_mm=config.partial_volume_boundary_mm,
        )
        diameter_ijk = _diameter_map_ijk(target_ijk, affine, branch)
        diameter_zyx = nibabel_ijk_to_perivascular_zyx(diameter_ijk)
        adaptive = generate_adaptive_shell(
            target_zyx,
            perivascular_geometry.spacing_xyz_mm,
            diameter_zyx,
            vessel_id=code,
            law=config.adaptive_law,
            other_artery_mask=other_artery_zyx,
            bifurcation_exclusion_mask=bifurcation_shell_zyx,
            partial_volume_boundary_mm=config.partial_volume_boundary_mm,
        )
        envelope = _union(fixed.filtered_masks, target_zyx.shape) | _union(
            adaptive.filtered_masks, target_zyx.shape
        )
        if not envelope.any():
            raise CarotidAnalysisError(f"No valid perivascular envelope remains for {code}.")
        composition = classify_tissue_composition(
            cta_zyx,
            envelope,
            perivascular_geometry.spacing_xyz_mm,
            target_artery_mask=target_zyx,
            other_artery_mask=other_artery_zyx,
            thresholds=config.tissue_thresholds,
            acquisition_metadata={
                **dict(config.acquisition_metadata),
                "coarse_anatomy_availability": anatomy_availability,
            },
            **anatomy,
        )
        assignment = assign_angular_sectors(
            envelope,
            system,
            perivascular_geometry.spacing_xyz_mm,
            number_of_sectors=config.sector_count,
            origin_xyz=perivascular_geometry.origin_ras_xyz_mm,
            direction=perivascular_geometry.direction_ijk_to_ras,
        )
        radial = tuple(
            aggregate_radial_profiles(
                cta_zyx,
                composition.labels,
                fixed.filtered_masks,
                perivascular_geometry.spacing_xyz_mm,
                valid_mask=composition.valid_denominator_mask,
            )
        )
        longitudinal = tuple(
            aggregate_longitudinal_profiles(
                cta_zyx,
                composition.labels,
                envelope,
                assignment.path_mm,
                perivascular_geometry.spacing_xyz_mm,
                bin_length_mm=config.longitudinal_bin_mm,
                valid_mask=composition.valid_denominator_mask,
                vessel_length_mm=system.length_mm,
            )
        )
        sector_summary = tuple(
            summarize_sectors(
                cta_zyx,
                composition.labels,
                assignment,
                perivascular_geometry.spacing_xyz_mm,
                radial_bands=fixed.filtered_masks,
                valid_mask=composition.valid_denominator_mask,
            )
        )
        contacts = calculate_surface_contacts(
            target_zyx,
            composition.labels,
            perivascular_geometry.spacing_xyz_mm,
            distances_mm=config.contact_distances_mm,
            coordinate_system=system,
            origin_xyz=perivascular_geometry.origin_ras_xyz_mm,
            direction=perivascular_geometry.direction_ijk_to_ras,
            longitudinal_bin_mm=config.longitudinal_bin_mm,
            number_of_sectors=config.sector_count,
            vessel_id=code,
        )
        fixed_qc = perivascular_qc(
            coordinate_system=system,
            shells=fixed,
            composition=composition,
            contacts=contacts,
        )
        adaptive_qc = perivascular_qc(shells=adaptive)
        vessel_output, branch_samples = _build_output_records(
            case_id=validated_case.manifest.case.case_id,
            code=code,
            branch=branch,
            system=system,
            fixed_shells=fixed,
            composition=composition,
            assignment=assignment,
            radial_profiles=radial,
            longitudinal_profiles=longitudinal,
            sector_summaries=sector_summary,
            contacts=contacts,
            qc_reports=(fixed_qc, adaptive_qc),
            cta_zyx=cta_zyx,
            config=config,
            bulb_interval=bulb,
            coarse_anatomy_availability=anatomy_availability,
        )
        branch_results[code] = CarotidBranchAnalysis(
            flowcat_code=code,
            branch=branch,
            derived_volume_label=label_id,
            target_lumen_zyx=target_zyx,
            coordinate_system=system,
            fixed_shells=fixed,
            adaptive_shell=adaptive,
            composition=composition,
            sector_assignment=assignment,
            radial_profiles=radial,
            longitudinal_profiles=longitudinal,
            sector_summaries=sector_summary,
            contacts=contacts,
            fixed_qc=fixed_qc,
            adaptive_qc=adaptive_qc,
            coarse_anatomy_availability=dict(anatomy_availability),
            configured_bulb_interval=bulb,
            vessel_output=vessel_output,
            sample_outputs=branch_samples,
        )
        vessel_outputs.append(vessel_output)
        sample_outputs.extend(branch_samples)

    ended = datetime.now(timezone.utc)
    provenance = validated_case.manifest.provenance
    global_flags = sorted(
        {
            flag
            for branch_result in branch_results.values()
            for flag in branch_result.vessel_output.qc_flags
        }
    )
    if provenance.commit is None:
        global_flags.append("flowcat_commit_unavailable")
    cta_artifact = next(
        (artifact for artifact in validated_case.manifest.artifacts if artifact.name == "cta"),
        None,
    )
    case_output = CaseLevelOutput(
        case_id=validated_case.manifest.case.case_id,
        cta_path=validated_case.manifest.case.cta_path or "<validated_in_memory_cta>",
        cta_metadata={
            **dict(config.acquisition_metadata),
            "array_order_input": "nibabel_ijk",
            "array_order_analysis": "perivascular_kji_zyx",
            "coordinate_reference": "nifti_ras_physical_mm",
            "coarse_anatomy_availability": dict(anatomy_availability),
        },
        original_spacing_xyz_mm=perivascular_geometry.spacing_xyz_mm,
        original_dimensions_ijk=geometry.shape_ijk,
        orientation=geometry.orientation,
        affine=affine.tolist(),
        available_vessel_segments=list(config.required_flowcat_codes),
        software_versions={"aorta_cta_radiomics": __version__, "numpy": np.__version__},
        model_versions={"flowcat": provenance.commit or "unrecorded"},
        flowcat_repository=provenance.repository_url,
        flowcat_branch=provenance.branch or "unrecorded",
        flowcat_commit=provenance.commit or "unrecorded",
        configuration_sha256=configuration_sha256(asdict(config)),
        processing_status=(
            "partial_missing_optional_context"
            if any(not anatomy_availability[name] for name in COARSE_ANATOMY_CONTEXT_NAMES)
            else "automated_pass_visual_pending"
        ),
        start_time=started.isoformat(),
        end_time=ended.isoformat(),
        processing_duration_seconds=(ended - started).total_seconds(),
        cpu_information=platform.processor() or platform.machine(),
        gpu_information=None,
        peak_ram_bytes=None,
        peak_gpu_memory_bytes=None,
        global_qc_flags=sorted(set(global_flags)),
        failed_modules=[],
        missing_outputs=list(validated_case.manifest.missing_required_outputs),
        input_sha256=cta_artifact.sha256 if cta_artifact is not None else None,
    )
    return CarotidCaseAnalysis(
        case_id=case_output.case_id,
        anatomical_artery_volume_ijk=artery_volume,
        branches=branch_results,
        case_output=case_output,
        vessel_outputs=tuple(vessel_outputs),
        sample_outputs=tuple(sample_outputs),
        coarse_anatomy_availability=dict(anatomy_availability),
        visual_review_required=True,
    )


analyze_validated_flowcat_carotids = analyze_carotid_case


__all__ = [
    "CAROTID_FLOWCAT_CODES",
    "COARSE_ANATOMY_CONTEXT_NAMES",
    "FLOWCAT_CLASS_TO_DERIVED_VOLUME_LABEL",
    "CarotidAnalysisConfig",
    "CarotidAnalysisError",
    "CarotidBranchAnalysis",
    "CarotidCaseAnalysis",
    "CoarseAnatomyMasksIJK",
    "PerivascularArrayGeometry",
    "SupportedBulbInterval",
    "analyze_carotid_case",
    "analyze_validated_flowcat_carotids",
    "flowcat_branch_to_centerline_seed",
    "nibabel_ijk_to_perivascular_zyx",
    "perivascular_geometry_from_nifti",
    "perivascular_zyx_to_nibabel_ijk",
]
