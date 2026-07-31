"""Discovery, validation, hashing, and manifest creation for FLOWCAT cases."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .exceptions import FlowcatSchemaError, MissingFlowcatOutputError
from .schemas import (
    CaseMetadata,
    CoordinateReference,
    FlowcatArtifact,
    FlowcatOutputManifest,
    ProcessingProvenance,
    QCFlag,
    QCSeverity,
)


DEFAULT_REQUIRED_OUTPUTS: tuple[str, ...] = (
    "segmentation",
    "branch_model",
    "centerline_segments_array",
    "segments_graph_pred",
    "local_graph",
)


def _standard_candidates(mode: str) -> dict[str, tuple[str, ...]]:
    return {
        "cta": ("cta.nii.gz", "head_cta.nii.gz"),
        # FLOWCAT's current nested path comes first. The root-level name appears
        # in its public README and older case exports, so both are supported.
        "segmentation": (
            f"{mode}/segmentation.nii.gz",
            f"{mode}_segmentation.nii.gz",
            f"{mode}/{mode}_segmentation.nii.gz",
        ),
        "segmentation_probabilities": (f"{mode}/segmentation_probabilities.nii.gz",),
        "segmentation_vtk": (f"{mode}/segmentation.vtk",),
        "segmentation_stl": (f"{mode}/segmentation.stl",),
        "branch_model": (f"{mode}/branch_model.vtk", f"{mode}/branch_model.vtp"),
        "clipped_model": (f"{mode}/clipped_model.vtk", f"{mode}/clipped_model.vtp"),
        "centerline_segments_array": (f"{mode}/centerline_segments_array.npy",),
        # Both layouts exist in FLOWCAT documentation/source revisions.
        "landmarks": (f"{mode}/landmarks/landmarks.json", f"{mode}/landmarks.json"),
        "segments_graph": (f"{mode}/segments_graph.pickle",),
        "segments_graph_pred": (f"{mode}/segments_graph_pred.pickle",),
        "local_graph": (f"{mode}/local_graph.pickle", f"{mode}/graph.pickle"),
    }


_COLLECTION_GLOBS: tuple[tuple[str, str], ...] = (
    ("centerline_vtk", "{mode}/centerlines/*.vtk"),
    ("centerline_vtp", "{mode}/centerlines/*.vtp"),
    ("segmentation_component_vtk", "{mode}/segmentations/*.vtk"),
    ("branch_component_vtk", "{mode}/branch_models/*.vtk"),
    ("clipped_component_vtk", "{mode}/clipped_models/*.vtk"),
    ("individual_centerline_vtk", "{mode}/individual_centerlines/*.vtk"),
    ("surface_stl", "{mode}/**/*.stl"),
)


def discover_flowcat_files(
    case_directory: str | os.PathLike[str],
    *,
    mode: str = "extracranial_vessels",
) -> dict[str, Path]:
    """Discover standard outputs without opening pickle or object-NPY files."""

    root = Path(case_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FLOWCAT case directory does not exist: {root}")
    if mode not in {"extracranial_vessels", "intracranial_vessels"}:
        raise ValueError(f"Unsupported FLOWCAT mode: {mode!r}")

    found: dict[str, Path] = {}
    for name, candidates in _standard_candidates(mode).items():
        for relative in candidates:
            candidate = root / relative
            if candidate.is_file():
                _ensure_case_local(candidate, root)
                found[name] = candidate
                break

    for prefix, pattern in _COLLECTION_GLOBS:
        matches = sorted(path for path in root.glob(pattern.format(mode=mode)) if path.is_file())
        unique_matches = [path for path in matches if path not in found.values()]
        for index, path in enumerate(unique_matches):
            _ensure_case_local(path, root)
            # Do not duplicate a primary representation under a collection key.
            found[f"{prefix}[{index}]"] = path
    return found


def validate_required_outputs(
    discovered: Mapping[str, Path],
    required: Iterable[str] = DEFAULT_REQUIRED_OUTPUTS,
) -> None:
    """Raise one clear error listing every absent required artifact."""

    missing = sorted(set(required).difference(discovered))
    if missing:
        raise MissingFlowcatOutputError("Missing required FLOWCAT outputs: " + ", ".join(missing))


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing FLOWCAT artifact: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_output_manifest(
    case_directory: str | os.PathLike[str],
    *,
    case_id: str | None = None,
    mode: str = "extracranial_vessels",
    required: Sequence[str] = DEFAULT_REQUIRED_OUTPUTS,
    provenance: ProcessingProvenance | None = None,
    hash_files: bool = True,
) -> FlowcatOutputManifest:
    """Discover artifacts and create a JSON-ready manifest.

    Validation here is deliberately non-executing: files must be regular,
    non-empty, case-local, and use their expected extension. Pickle and object
    NPY contents are only opened by explicitly trusted loader functions.
    """

    root = Path(case_directory).expanduser().resolve()
    found = discover_flowcat_files(root, mode=mode)
    required_set = set(required)
    missing = tuple(sorted(required_set.difference(found)))
    artifacts: list[FlowcatArtifact] = []
    qc_flags: list[QCFlag] = []

    ordered_names = list(_standard_candidates(mode))
    ordered_names.extend(sorted(set(found).difference(ordered_names)))
    for name in ordered_names:
        path = found.get(name)
        if path is None:
            if name in required_set:
                artifacts.append(
                    FlowcatArtifact(
                        name=name,
                        path=None,
                        format=_expected_format(name),
                        present=False,
                        required=True,
                        validation_status="missing",
                        validation_message="required artifact not found",
                    )
                )
                qc_flags.append(
                    QCFlag(
                        code="FLOWCAT_REQUIRED_OUTPUT_MISSING",
                        severity=QCSeverity.ERROR,
                        message=f"Required FLOWCAT output {name!r} was not found",
                        artifact_name=name,
                    )
                )
            continue

        status, message = _validate_discovered_file(name, path)
        size_bytes = path.stat().st_size
        digest = sha256_file(path) if hash_files else None
        artifacts.append(
            FlowcatArtifact(
                name=name,
                path=str(path),
                format=_format_from_path(path),
                present=True,
                required=name in required_set,
                sha256=digest,
                size_bytes=size_bytes,
                coordinate_reference=_coordinate_reference(name),
                units=_units(name),
                validation_status=status,
                validation_message=message,
            )
        )
        if status == "invalid":
            qc_flags.append(
                QCFlag(
                    code="FLOWCAT_ARTIFACT_INVALID",
                    severity=QCSeverity.ERROR,
                    message=message or f"Invalid artifact {name!r}",
                    artifact_name=name,
                )
            )

    case = CaseMetadata(
        case_id=case_id or root.name,
        case_directory=str(root),
        mode=mode,
        cta_path=str(found["cta"]) if "cta" in found else None,
    )
    return FlowcatOutputManifest(
        schema_version="1.0.0",
        case=case,
        artifacts=tuple(artifacts),
        provenance=provenance or ProcessingProvenance(),
        qc_flags=tuple(qc_flags),
        missing_required_outputs=missing,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def write_output_manifest(
    manifest: FlowcatOutputManifest,
    output_path: str | os.PathLike[str],
) -> Path:
    """Atomically write a FLOWCAT manifest as UTF-8 JSON."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def create_and_write_output_manifest(
    case_directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    **kwargs: object,
) -> FlowcatOutputManifest:
    manifest = build_output_manifest(case_directory, **kwargs)
    write_output_manifest(manifest, output_path)
    return manifest


def _ensure_case_local(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise FlowcatSchemaError(f"FLOWCAT artifact escapes case directory: {path}") from exc


def _validate_discovered_file(name: str, path: Path) -> tuple[str, str | None]:
    if not path.is_file():
        return "invalid", "artifact is not a regular file"
    if path.stat().st_size <= 0:
        return "invalid", "artifact is empty"
    expected = _expected_format(name)
    actual = _format_from_path(path)
    allowed = _expected_formats(name)
    if allowed and actual not in allowed:
        return "invalid", f"expected {expected}, found {actual}"
    return "validated_file", None


def _format_from_path(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return "nifti"
    return {
        ".vtk": "vtk",
        ".vtp": "vtp",
        ".stl": "stl",
        ".npy": "numpy",
        ".pickle": "pickle",
        ".pkl": "pickle",
        ".json": "json",
    }.get(path.suffix.lower(), "unknown")


def _expected_format(name: str) -> str:
    base = name.split("[", maxsplit=1)[0]
    if base in {"cta", "segmentation", "segmentation_probabilities"}:
        return "nifti"
    vtk_names = {
        "segmentation_vtk",
        "branch_model",
        "clipped_model",
        "centerline_vtk",
        "segmentation_component_vtk",
        "branch_component_vtk",
        "clipped_component_vtk",
        "individual_centerline_vtk",
    }
    if base in vtk_names:
        return "vtk"
    if base in {"centerline_vtp"}:
        return "vtp"
    if base in {"segmentation_stl", "surface_stl"}:
        return "stl"
    if base == "centerline_segments_array":
        return "numpy"
    if base in {"segments_graph", "segments_graph_pred", "local_graph"}:
        return "pickle"
    if base == "landmarks":
        return "json"
    return "unknown"


def _expected_formats(name: str) -> set[str]:
    base = name.split("[", maxsplit=1)[0]
    if base in {"branch_model", "clipped_model"}:
        return {"vtk", "vtp"}
    expected = _expected_format(name)
    return set() if expected == "unknown" else {expected}


def _coordinate_reference(name: str) -> CoordinateReference:
    base = name.split("[", maxsplit=1)[0]
    if base in {"centerline_segments_array", "segments_graph", "segments_graph_pred", "local_graph"}:
        return CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM
    if base in {
        "cta",
        "segmentation",
        "segmentation_probabilities",
        "segmentation_vtk",
        "segmentation_stl",
        "branch_model",
        "clipped_model",
        "landmarks",
        "centerline_vtk",
        "centerline_vtp",
        "segmentation_component_vtk",
        "branch_component_vtk",
        "clipped_component_vtk",
        "individual_centerline_vtk",
        "surface_stl",
    }:
        return CoordinateReference.NIFTI_RAS_PHYSICAL_MM
    return CoordinateReference.UNKNOWN


def _units(name: str) -> str | None:
    base = name.split("[", maxsplit=1)[0]
    if base in {"segments_graph", "segments_graph_pred"}:
        return "mixed; positions and radii in mm"
    if base == "local_graph":
        return "mixed; positions and radii in mm"
    if _coordinate_reference(name) is not CoordinateReference.UNKNOWN:
        return "mm"
    return None
