"""Typed, three-level outputs for vessel-centred CTA phenotyping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(configuration: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _clean_json(configuration), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CaseLevelOutput:
    case_id: str
    cta_path: str
    cta_metadata: dict[str, Any]
    original_spacing_xyz_mm: tuple[float, float, float]
    original_dimensions_ijk: tuple[int, int, int]
    orientation: tuple[str, str, str]
    affine: list[list[float]]
    available_vessel_segments: list[str]
    software_versions: dict[str, str]
    model_versions: dict[str, str]
    flowcat_repository: str
    flowcat_branch: str
    flowcat_commit: str
    configuration_sha256: str
    processing_status: str
    start_time: str
    end_time: str
    processing_duration_seconds: float
    cpu_information: str
    gpu_information: str | None
    peak_ram_bytes: int | None
    peak_gpu_memory_bytes: int | None
    global_qc_flags: list[str] = field(default_factory=list)
    failed_modules: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    input_sha256: str | None = None


@dataclass
class VesselLevelOutput:
    case_id: str
    vessel_name: str
    laterality: str | None
    branch_id: str
    vessel_label_confidence: float | None
    total_length_mm: float
    assessable_length_mm: float
    nonassessable_length_mm: float
    mean_radius_mm: float | None
    minimum_radius_mm: float | None
    maximum_radius_mm: float | None
    radius_method: str
    diameter_method: str
    tortuosity: float | None
    mean_curvature_per_mm: float | None
    calcium_burden_mm3: float | None
    total_shell_volume_mm3: float
    valid_shell_volume_mm3: float
    pvat_volume_mm3: float
    pvat_fraction: float | None
    mean_pvat_attenuation_hu: float | None
    perivascular_soft_tissue_volume_mm3: float
    muscle_contact_fraction: float | None
    vein_contact_fraction: float | None
    bone_contact_fraction: float | None
    thyroid_contact_fraction: float | None
    other_soft_tissue_fraction: float | None
    uncertain_fraction: float | None
    radial_gradients: dict[str, float | None] = field(default_factory=dict)
    longitudinal_summaries: dict[str, Any] = field(default_factory=dict)
    circumferential_summaries: dict[str, Any] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)
    manual_review_required: bool = True


@dataclass
class SamplePointLevelOutput:
    case_id: str
    vessel_name: str
    laterality: str | None
    branch_id: str
    path_distance_mm: float
    physical_ras_x_mm: float
    physical_ras_y_mm: float
    physical_ras_z_mm: float
    tangent_ras: tuple[float, float, float]
    first_normal_ras: tuple[float, float, float]
    second_normal_ras: tuple[float, float, float]
    local_radius_mm: float | None
    radius_method: str
    local_diameter_mm: float | None
    diameter_method: str
    curvature_per_mm: float | None
    torsion_per_mm: float | None
    distance_to_bifurcation_mm: float | None
    radial_band_measurements: dict[str, Any] = field(default_factory=dict)
    sector_measurements: dict[str, Any] = field(default_factory=dict)
    surface_contact_measurements: dict[str, Any] = field(default_factory=dict)
    calcium_measurements: dict[str, Any] = field(default_factory=dict)
    tissue_confidence: float | None = None
    uncertain_fraction: float | None = None
    qc_flags: list[str] = field(default_factory=list)


@dataclass
class OutputWriteResult:
    case_json: Path
    vessel_csv: Path
    sample_point_table: Path
    sample_point_format: str
    manifest_json: Path


def _records(items: Sequence[Any]) -> list[dict[str, Any]]:
    return [_clean_json(asdict(item)) for item in items]


def write_output_bundle(
    output_directory: str | Path,
    case: CaseLevelOutput,
    vessels: Sequence[VesselLevelOutput],
    sample_points: Sequence[SamplePointLevelOutput],
    *,
    prefer_parquet: bool = True,
) -> OutputWriteResult:
    """Write JSON/CSV/Parquet outputs without silently hiding a fallback.

    Parquet is used for sample-point data when a pandas engine is installed.
    If it is unavailable, a CSV is written and the manifest explicitly records
    ``csv_fallback_missing_parquet_engine``.
    """

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    case_json = destination / "case_summary.json"
    vessel_csv = destination / "vessel_level_features.csv"
    manifest_json = destination / "output_manifest.json"
    case_json.write_text(
        json.dumps(_clean_json(asdict(case)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_records(vessels)).to_csv(vessel_csv, index=False)

    sample_frame = pd.DataFrame(_records(sample_points))
    sample_format = "parquet"
    sample_path = destination / "sample_point_features.parquet"
    if prefer_parquet:
        try:
            sample_frame.to_parquet(sample_path, index=False)
        except (ImportError, ModuleNotFoundError):
            sample_format = "csv_fallback_missing_parquet_engine"
            sample_path = destination / "sample_point_features.csv"
            sample_frame.to_csv(sample_path, index=False)
    else:
        sample_format = "csv_requested"
        sample_path = destination / "sample_point_features.csv"
        sample_frame.to_csv(sample_path, index=False)

    manifest = {
        "schema_version": 1,
        "case_id": case.case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_point_format": sample_format,
        "files": {
            "case_summary": {
                "path": str(case_json.resolve()),
                "sha256": sha256_file(case_json),
            },
            "vessel_level": {
                "path": str(vessel_csv.resolve()),
                "sha256": sha256_file(vessel_csv),
                "rows": len(vessels),
            },
            "sample_point_level": {
                "path": str(sample_path.resolve()),
                "sha256": sha256_file(sample_path),
                "rows": len(sample_points),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return OutputWriteResult(
        case_json=case_json,
        vessel_csv=vessel_csv,
        sample_point_table=sample_path,
        sample_point_format=sample_format,
        manifest_json=manifest_json,
    )


__all__ = [
    "CaseLevelOutput",
    "OutputWriteResult",
    "SamplePointLevelOutput",
    "VesselLevelOutput",
    "configuration_sha256",
    "sha256_file",
    "write_output_bundle",
]
