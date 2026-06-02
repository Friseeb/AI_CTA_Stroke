"""Pydantic report schema and JSON serialisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from . import DISCLAIMER


class FOVCompleteness(BaseModel):
    has_upper_dentition: Union[bool, str] = "unknown"
    has_lower_dentition: Union[bool, str] = "unknown"
    has_mandible: Union[bool, str] = "unknown"
    has_maxilla: Union[bool, str] = "unknown"
    partial_fov: Union[bool, str] = "unknown"
    left_right_coverage_estimate: str = "unknown"
    inferior_superior_coverage_estimate: str = "unknown"


class PreprocessingRecord(BaseModel):
    original_spacing_xyz_mm: Optional[list[float]] = None
    target_spacing_xyz_mm: Optional[list[float]] = None
    original_size_xyz: Optional[list[int]] = None
    resampled_size_xyz: Optional[list[int]] = None
    orientation: str = "RAS"
    hu_clip_min: float = -1000.0
    hu_clip_max: float = 3000.0
    series_instance_uid: Optional[str] = None
    convolution_kernel: Optional[str] = None
    protocol_name: Optional[str] = None
    modality: Optional[str] = None


class DentalReport(BaseModel):
    # Core identity
    case_id: str
    disclaimer: str = DISCLAIMER

    # Input
    input_type: Literal["dicom", "nifti", "unknown"] = "unknown"
    input_path: Optional[str] = None

    # Preprocessing
    original_spacing_xyz_mm: Optional[list[float]] = None
    target_spacing_xyz_mm: Optional[list[float]] = None
    preprocessing: Optional[PreprocessingRecord] = None
    age_status: Literal["adult", "pediatric", "unknown"] = "unknown"

    # ROI
    roi_method: str = "unknown"
    roi_quality: Literal["good", "fair", "poor", "failed", "unknown"] = "unknown"
    fov_completeness: Optional[FOVCompleteness] = None
    roi_bbox_voxel: Optional[dict] = None
    roi_bbox_physical: Optional[dict] = None

    # Segmentation
    segmentation_backend: str = "unknown"
    segmentation_status: Literal["success", "failed", "skipped", "unknown"] = "unknown"
    model_versions: dict[str, str] = Field(default_factory=dict)
    domain_warnings: list[str] = Field(default_factory=list)

    # Deface
    deface_mode: str = "none"
    deface_result: Optional[dict] = None

    # Outputs
    candidate_features_path: Optional[str] = None
    qc_paths: dict[str, str] = Field(default_factory=dict)

    # Status
    status: str = "unknown"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def write_report(report: DentalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2, exclude_none=False))


def load_report(path: Path) -> DentalReport:
    return DentalReport.model_validate_json(path.read_text())
