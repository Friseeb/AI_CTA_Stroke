"""Typed, dependency-light internal structures for FLOWCAT outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Mapping


class CoordinateReference(StrEnum):
    """Coordinate references used at the FLOWCAT boundary."""

    FLOWCAT_LPI_CORNER_RELATIVE_MM = "flowcat_lpi_corner_relative_mm"
    NIFTI_RAS_PHYSICAL_MM = "nifti_ras_physical_mm"
    NIFTI_LPS_PHYSICAL_MM = "nifti_lps_physical_mm"
    UNKNOWN = "unknown"


class QCSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CaseMetadata:
    case_id: str
    case_directory: str
    mode: str = "extracranial_vessels"
    cta_path: str | None = None
    deidentified: bool = True

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if self.mode not in {"extracranial_vessels", "intracranial_vessels"}:
            raise ValueError(f"Unsupported FLOWCAT mode: {self.mode!r}")


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """NIfTI geometry expressed in nibabel's RAS+ world convention."""

    shape_ijk: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    affine_ras_mm: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    orientation: tuple[str, str, str]
    coordinate_reference: CoordinateReference = CoordinateReference.NIFTI_RAS_PHYSICAL_MM
    units: str = "mm"

    def __post_init__(self) -> None:
        if len(self.shape_ijk) != 3 or any(int(value) <= 0 for value in self.shape_ijk):
            raise ValueError(f"Invalid image shape: {self.shape_ijk!r}")
        if len(self.spacing_mm) != 3 or any(float(value) <= 0 for value in self.spacing_mm):
            raise ValueError(f"Invalid spacing: {self.spacing_mm!r}")
        if len(self.orientation) != 3:
            raise ValueError(f"Invalid orientation: {self.orientation!r}")


@dataclass(frozen=True, slots=True)
class ArteryIdentity:
    """One of the 14 FLOWCAT extracranial vessel classes."""

    flowcat_class_id: int
    flowcat_code: str
    name: str
    laterality: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= int(self.flowcat_class_id) <= 13:
            raise ValueError(f"FLOWCAT class ID must be in [0, 13], got {self.flowcat_class_id}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CenterlinePoint:
    coordinates_mm: Vector3
    path_distance_mm: float
    local_radius_mm: float | None
    branch_id: str
    coordinate_reference: CoordinateReference
    tangent: Vector3 | None = None
    first_normal: Vector3 | None = None
    second_normal: Vector3 | None = None
    artery: ArteryIdentity | None = None
    vessel_label_confidence: float | None = None
    curvature_per_mm: float | None = None
    torsion_per_mm: float | None = None
    distance_to_bifurcation_mm: float | None = None
    source_file: str | None = None
    processing_version: str | None = None

    @property
    def x_mm(self) -> float:
        return self.coordinates_mm[0]

    @property
    def y_mm(self) -> float:
        return self.coordinates_mm[1]

    @property
    def z_mm(self) -> float:
        return self.coordinates_mm[2]


@dataclass(frozen=True, slots=True)
class CenterlineBranch:
    branch_id: str
    cell_id: int
    points: tuple[CenterlinePoint, ...]
    coordinate_reference: CoordinateReference
    artery: ArteryIdentity | None = None
    source_file: str | None = None
    processing_version: str | None = None


@dataclass(frozen=True, slots=True)
class Bifurcation:
    bifurcation_id: str
    coordinates_mm: Vector3
    connected_branch_ids: tuple[str, ...]
    coordinate_reference: CoordinateReference
    source_file: str | None = None


@dataclass(frozen=True, slots=True)
class VesselGraph:
    branches: tuple[CenterlineBranch, ...]
    bifurcations: tuple[Bifurcation, ...] = ()
    directed: bool = False
    source_file: str | None = None


@dataclass(frozen=True, slots=True)
class LocalGeometricFeature:
    branch_id: str
    path_distance_mm: float
    radius_mm: float | None = None
    diameter_mm: float | None = None
    curvature_per_mm: float | None = None
    torsion_per_mm: float | None = None
    tortuosity: float | None = None
    source_file: str | None = None


@dataclass(frozen=True, slots=True)
class FlowcatArtifact:
    name: str
    path: str | None
    format: str
    present: bool
    required: bool = False
    sha256: str | None = None
    size_bytes: int | None = None
    coordinate_reference: CoordinateReference = CoordinateReference.UNKNOWN
    units: str | None = None
    validation_status: str = "not_validated"
    validation_message: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingProvenance:
    repository_url: str = "https://github.com/FLOWCAT-CV/arterial"
    branch: str | None = None
    commit: str | None = None
    model_files: Mapping[str, str] = field(default_factory=dict)
    model_file_sha256: Mapping[str, str] = field(default_factory=dict)
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    command_line_parameters: Mapping[str, Any] = field(default_factory=dict)
    processing_status: str = "not_started"
    started_at: str | None = None
    ended_at: str | None = None


@dataclass(frozen=True, slots=True)
class QCFlag:
    code: str
    severity: QCSeverity
    message: str
    artifact_name: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FlowcatOutputManifest:
    schema_version: str
    case: CaseMetadata
    artifacts: tuple[FlowcatArtifact, ...]
    provenance: ProcessingProvenance
    qc_flags: tuple[QCFlag, ...] = ()
    missing_required_outputs: tuple[str, ...] = ()
    generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
