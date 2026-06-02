"""DICOM utilities that strip identifiers before anything reaches a log file.

Why a separate module: this code is the *only* place that touches raw DICOM
tag values like PatientName / PatientBirthDate / IssuerOfPatientID. By keeping
it isolated we can audit PHI handling in one file. Other modules only ever
receive an already-scrubbed dict.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from .logging_utils import get_logger

log = get_logger("dicom_utils")


# Tags we keep (acquisition / geometry / safety) — *not* identifying.
_SAFE_TAGS = {
    "Modality", "Manufacturer", "ManufacturerModelName", "BodyPartExamined",
    "KVP", "XRayTubeCurrent", "ExposureTime", "SliceThickness", "PixelSpacing",
    "SpacingBetweenSlices", "ConvolutionKernel", "ReconstructionDiameter",
    "Rows", "Columns", "ImageOrientationPatient", "ImagePositionPatient",
    "ContrastBolusAgent", "ContrastBolusVolume", "ProtocolName",
    "ImageType", "PatientAge", "PatientSex",     # age/sex kept, name/MRN dropped
}

# Tags we explicitly redact (PHI / strong identifiers).
_PHI_TAGS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "EthnicGroup", "PatientComments",
    "OtherPatientIDs", "OtherPatientNames", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "StationName",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "AccessionNumber", "StudyID", "IssuerOfPatientID",
}


def safe_hash(value: str, length: int = 12) -> str:
    """SHA-1 truncated hash for opaque identifiers (study_id / scan_id)."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def scrub_dicom_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop PHI keys; keep only acquisition-relevant tags."""
    clean: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if k in _PHI_TAGS:
            continue
        if k in _SAFE_TAGS:
            clean[k] = v
        # everything else: only keep if numeric / boolean / short string
        elif isinstance(v, (int, float, bool)):
            clean[k] = v
        elif isinstance(v, str) and len(v) <= 64:
            clean[k] = v
    return clean


def derive_ids_from_dicom(
    raw_meta: dict[str, Any],
    fallback_path: Path,
) -> tuple[str, str]:
    """Return (study_id, scan_id) — both opaque hashes, never the raw MRN.

    Prefer the StudyInstanceUID / SeriesInstanceUID since they are stable
    across re-pulls, but hash them so the original UID never reaches a log
    or output file. Fall back to a path hash when no UID is available
    (e.g. converted NIfTI without sidecar).
    """
    study_uid = (raw_meta or {}).get("StudyInstanceUID")
    series_uid = (raw_meta or {}).get("SeriesInstanceUID")
    if study_uid:
        study_id = "stu_" + safe_hash(str(study_uid))
    else:
        study_id = "stu_" + safe_hash(str(fallback_path.resolve().parent))
    if series_uid:
        scan_id = "scn_" + safe_hash(str(series_uid))
    else:
        scan_id = "scn_" + safe_hash(str(fallback_path.resolve()))
    return study_id, scan_id


def check_adult(meta: dict[str, Any], min_age_years: int) -> Optional[str]:
    """Returns 'pediatric' if known-under-min, else None.

    DICOM `PatientAge` is a DA-style string like '054Y' / '012Y'. We refuse
    cases that are *known* to be pediatric; missing-age cases pass to QC.
    """
    age_str = (meta or {}).get("PatientAge")
    if not age_str:
        return None
    try:
        digits = "".join(c for c in str(age_str) if c.isdigit())
        if not digits:
            return None
        years = int(digits)
        return "pediatric" if years < min_age_years else None
    except Exception:
        return None


def infer_contrast(meta: dict[str, Any]) -> Optional[bool]:
    """Best-effort contrast inference. Returns None when unknown.

    `ContrastBolusAgent` is the most reliable field; `ImageType` containing
    'CTA' or 'ANGIO' is a secondary signal.
    """
    agent = (meta or {}).get("ContrastBolusAgent")
    if isinstance(agent, str) and agent.strip():
        return True
    image_type = (meta or {}).get("ImageType")
    if isinstance(image_type, (list, tuple)):
        joined = " ".join(str(x).upper() for x in image_type)
    else:
        joined = str(image_type or "").upper()
    if any(tok in joined for tok in ("CTA", "ANGIO", "ARTERIAL")):
        return True
    return None
