"""Metadata eligibility filters for CTA cohort processing."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_CTA_TERMS = (
    "cta",
    "ct angiography",
    "ct angiogram",
    "computed tomography angiography",
    "angio",
    "angiography",
    "angiogram",
)

DEFAULT_NEURO_TERMS = (
    "brain",
    "head",
    "neck",
    "head neck",
    "head/neck",
    "head and neck",
    "carotid",
    "vertebral",
    "cerebral",
    "intracranial",
    "circle of willis",
    "willis",
    "arch to vertex",
    "stroke",
    "acute stroke",
    "hyperacute",
    "hypercute",
    "hyper acute",
    "code stroke",
)

DEFAULT_EXCLUDE_TERMS = (
    "coronary",
    "cardiac",
    "pulmonary embol",
    "pe protocol",
    "runoff",
    "lower extremity",
    "abdomen",
    "pelvis",
    "tavr",
)

DEFAULT_METADATA_PATH_COLUMNS = (
    "metadata_path",
    "json_path",
    "sidecar_path",
    "dicom_metadata_path",
    "metadata_json",
)

DEFAULT_METADATA_TEXT_COLUMNS = (
    "series_description",
    "seriesdescription",
    "protocol_name",
    "protocolname",
    "study_description",
    "studydescription",
    "requested_procedure_description",
    "requestedproceduredescription",
    "body_part_examined",
    "bodypartexamined",
    "modality",
    "acquisition",
    "acquisition_label",
    "scan_options",
    "scanoptions",
    "image_type",
    "imagetype",
)


@dataclass(frozen=True)
class MetadataEligibility:
    """Result of deciding whether one case should be processed."""

    case_id: str
    eligible: bool
    reason: str
    metadata_path: str
    matched_cta_terms: str
    matched_neuro_terms: str
    matched_exclude_terms: str
    metadata_source: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_neuro_cta_metadata(
    row: Mapping[str, object],
    manifest_base: str | Path,
    include_keywords: list[str] | tuple[str, ...] = (),
    exclude_keywords: list[str] | tuple[str, ...] = (),
    allow_missing_metadata: bool = False,
) -> MetadataEligibility:
    """Return whether a manifest row looks like a brain/neck stroke CTA case.

    The filter reads explicit manifest metadata columns plus a BIDS-style JSON
    sidecar. It does not use the image filename as evidence unless the metadata
    appears in a manifest text field.
    """
    case_id = _cell_as_str(row.get("case_id", ""))
    manifest_base = Path(manifest_base)
    metadata_path = resolve_metadata_path(row, manifest_base)
    sidecar_text = ""
    metadata_source = "manifest_columns"
    if metadata_path is not None and metadata_path.exists():
        sidecar_text = _json_text(metadata_path)
        metadata_source = "json_sidecar"

    manifest_text = manifest_metadata_text(row)
    text = _normalize_text(" ".join(part for part in [manifest_text, sidecar_text] if part))
    if not text:
        return MetadataEligibility(
            case_id=case_id,
            eligible=bool(allow_missing_metadata),
            reason="missing_metadata_allowed" if allow_missing_metadata else "no_metadata",
            metadata_path=str(metadata_path or ""),
            matched_cta_terms="",
            matched_neuro_terms="",
            matched_exclude_terms="",
            metadata_source="" if metadata_path is None else metadata_source,
        )

    cta_terms = _matched_terms(text, DEFAULT_CTA_TERMS)
    neuro_terms = _matched_terms(text, tuple(DEFAULT_NEURO_TERMS) + tuple(include_keywords))
    exclude_terms = _matched_terms(text, tuple(DEFAULT_EXCLUDE_TERMS) + tuple(exclude_keywords))

    if exclude_terms and not neuro_terms:
        return _eligibility(
            case_id,
            False,
            "excluded_non_neuro_protocol",
            metadata_path,
            cta_terms,
            neuro_terms,
            exclude_terms,
            metadata_source,
        )
    if not cta_terms:
        return _eligibility(
            case_id,
            False,
            "missing_cta_or_angiography_term",
            metadata_path,
            cta_terms,
            neuro_terms,
            exclude_terms,
            metadata_source,
        )
    if not neuro_terms:
        return _eligibility(
            case_id,
            False,
            "missing_brain_neck_stroke_term",
            metadata_path,
            cta_terms,
            neuro_terms,
            exclude_terms,
            metadata_source,
        )
    return _eligibility(
        case_id,
        True,
        "eligible_neuro_cta",
        metadata_path,
        cta_terms,
        neuro_terms,
        exclude_terms,
        metadata_source,
    )


def manifest_metadata_text(row: Mapping[str, object]) -> str:
    """Collect relevant free-text metadata values from a manifest row."""
    parts: list[str] = []
    metadata_column_names = set(DEFAULT_METADATA_TEXT_COLUMNS)
    for raw_key, value in row.items():
        key = _normalize_column_name(str(raw_key))
        if key in metadata_column_names or any(
            token in key
            for token in [
                "description",
                "protocol",
                "bodypart",
                "body_part",
                "modality",
                "acquisition",
                "procedure",
            ]
        ):
            text = _cell_as_str(value)
            if text:
                parts.append(text)
    return " ".join(parts)


def resolve_metadata_path(row: Mapping[str, object], manifest_base: str | Path) -> Path | None:
    """Resolve explicit or BIDS-style JSON sidecar metadata path."""
    base = Path(manifest_base)
    for column in DEFAULT_METADATA_PATH_COLUMNS:
        raw_value = row.get(column)
        text = _cell_as_str(raw_value)
        if text:
            return _resolve_path(text, base)

    image_text = _cell_as_str(row.get("image_path", ""))
    if not image_text:
        return None
    image_path = _resolve_path(image_text, base)
    candidates = _json_sidecar_candidates(image_path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0] if candidates else None)


def _json_sidecar_candidates(image_path: Path) -> list[Path]:
    name = image_path.name
    if name.endswith(".nii.gz"):
        return [image_path.with_name(name.removesuffix(".nii.gz") + ".json")]
    if name.endswith(".nii"):
        return [image_path.with_suffix(".json")]
    return [image_path.with_suffix(".json")]


def _json_text(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    values: list[str] = []
    _flatten_json_values(payload, values)
    return " ".join(values)


def _flatten_json_values(value: object, values: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _normalize_column_name(str(key)) in {
                "patientname",
                "patientid",
                "patientbirthdate",
                "accessionnumber",
            }:
                continue
            _flatten_json_values(item, values)
    elif isinstance(value, list):
        for item in value:
            _flatten_json_values(item, values)
    elif isinstance(value, (str, int, float, bool)):
        text = _cell_as_str(value)
        if text:
            values.append(text)


def _eligibility(
    case_id: str,
    eligible: bool,
    reason: str,
    metadata_path: Path | None,
    cta_terms: list[str],
    neuro_terms: list[str],
    exclude_terms: list[str],
    metadata_source: str,
) -> MetadataEligibility:
    return MetadataEligibility(
        case_id=case_id,
        eligible=eligible,
        reason=reason,
        metadata_path=str(metadata_path or ""),
        matched_cta_terms=";".join(cta_terms),
        matched_neuro_terms=";".join(neuro_terms),
        matched_exclude_terms=";".join(exclude_terms),
        metadata_source=metadata_source,
    )


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for term in terms:
        normalized = _normalize_text(term)
        if not normalized:
            continue
        if len(normalized) <= 3:
            if re.search(rf"\b{re.escape(normalized)}\b", text):
                matches.append(term)
        elif normalized in text:
            matches.append(term)
    return matches


def _normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _cell_as_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text
