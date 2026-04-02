#!/usr/bin/env python3
"""Convert DICOM exports to SLAAOBIDS-style NIfTI structure.

Architecture
------------
Each patient directory (e.g. D:/224/) contains one or more Export_YYYY-*
subfolders, where EACH folder corresponds to exactly ONE CT study exported
from PACS (eCTA, Thorax CT, Abdomen CT, or Heart CT).

The script:
  1. Iterates over Export_* subfolders within each patient directory.
  2. Classifies the study type from StudyDescription / ProtocolName /
     BodyPartExamined using keyword matching.  Slice count alone never
     qualifies — at least one keyword or body-part match is required.
  3. Converts only the studies that are actually present; no empty folders
     are ever created.
  4. For multi-phase types (ctheart, ctabdomen) all qualifying source series
     inside the export are exported as _ph00, _ph01, ...
  5. For single-phase types (ecta, ctthorax) only the best series is kept.

Output layout (mirrors existing SLAAOBIDS dataset):
  <out-root>/sub-<id>/sub-<id>_acq-ecta_ct.nii.gz
  <out-root>/sub-<id>/sub-<id>_acq-ctheart_ph00_ct.nii.gz
  <out-root>/sub-<id>/sub-<id>_acq-ctthorax_ct.nii.gz
  <out-root>/sub-<id>/sub-<id>_acq-ctabdomen_ph00_ct.nii.gz
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import shutil

import pydicom
import SimpleITK as sitk
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="Expected implicit VR, but found explicit VR - using explicit VR for reading",
)

SKIP_DIRS = {"PLUGINS", "JRE", "HELP", "REPORT", "IHE_PDI", "XTR_CONT"}

# Export subfolders must match this prefix to be treated as a PACS study.
EXPORT_PREFIX = "Export_"

# Within a classified study, a source series must have at least this many
# slices to be considered a volumetric acquisition (not a scout/localiser).
MIN_SOURCE_SLICES = 100

# Bad series descriptors that disqualify a series as a source volume regardless
# of slice count (reformats, scouts, injection traces, etc.).
SERIES_BAD_KEYWORDS = (
    "mip", "render", "sagittal", "coronal", "scout", "localizer",
    "topogram", "summary", "reformatted", "medrad injection", "injection images", "medrad",
    "lcd", "bone", "stress", "perfusion", "rapid",
)


# ---------------------------------------------------------------------------
# Scan-type definitions
# ---------------------------------------------------------------------------

@dataclass
class ScanTypeConfig:
    name: str          # internal key used in --scan-types
    acq_label: str     # BIDS acquisition label
    multi_phase: bool  # True → export all source series as _ph00, _ph01 …
    # Keywords matched against StudyDescription only (highest weight: +150).
    study_desc_keywords: Tuple[str, ...]
    # Keywords matched against ProtocolName / SeriesDescription (lower weight: +80).
    protocol_keywords: Tuple[str, ...]
    # BodyPartExamined substrings (weight: +50).
    study_body_parts: Tuple[str, ...]
    # Hard coverage exclusion bounds (mm = z_max − z_min, scanner-independent).
    # If the measured coverage falls OUTSIDE [z_coverage_min_mm, z_coverage_max_mm],
    # the type is disqualified (score forced to 0) regardless of keywords.
    # None = no constraint on that side.
    z_coverage_min_mm: Optional[float]
    z_coverage_max_mm: Optional[float]
    # Size gate for the converted NIfTI.
    min_mb: float
    max_mb: float
    # If ANY of these substrings appear in StudyDescription, this type is
    # disqualified (score forced to 0) regardless of keyword matches.
    # Used to prevent broad keywords (e.g. "chest") from matching combined
    # chest/abdomen/pelvis studies when a more specific type is expected.
    exclude_study_desc_keywords: Tuple[str, ...] = ()
    # Per-type minimum source slices (overrides global MIN_SOURCE_SLICES).
    # CardiacCT uses 50 because gated phases are typically 54–57 slices.
    min_source_slices: int = MIN_SOURCE_SLICES


SCAN_TYPE_CONFIGS: List[ScanTypeConfig] = [
    ScanTypeConfig(
        name="CTA",
        acq_label="ecta",
        multi_phase=False,
        study_desc_keywords=(
            "cta", "angio", "angiograph", "ecta",
            "head/neck", "head neck", "tia", "stroke",
        ),
        protocol_keywords=(
            "cta", "angio", "head/neck", "tia",
        ),
        study_body_parts=("head", "neck", "brain", "cerebr", "carotid"),
        z_coverage_min_mm=100.0,
        z_coverage_max_mm=550.0,
        min_mb=200.0,
        max_mb=600.0,
    ),
    ScanTypeConfig(
        name="CardiacCT",
        acq_label="ctheart",
        multi_phase=True,
        study_desc_keywords=(
            "cardiac", "heart", "coronary", "ccta", "calcium", "cac",
            "cardio", "ctheart", "aortic", "tavi",
        ),
        protocol_keywords=(
            "cardiac", "heart", "coronary", "ccta", "cac", "cardio",
        ),
        study_body_parts=("heart", "cardiac", "coronary"),
        z_coverage_min_mm=70.0,
        z_coverage_max_mm=400.0,
        min_mb=10.0,
        max_mb=700.0,
        min_source_slices=50,
    ),
    ScanTypeConfig(
        # "chest" is safe because exclude_study_desc_keywords vetoes studies
        # that also mention "abdomen" or "pelvis" (combined CAP scans).
        # z_coverage_max_mm=750mm accommodates cardiac CAP protocols (e.g.
        # "CT Cardiac CAP") where the scan covers chest-to-pelvis (~666mm)
        # but the primary output of interest is the thorax/lung series.
        name="ThoraxCT",
        acq_label="ctthorax",
        multi_phase=False,
        study_desc_keywords=(
            "thorax", "chest", "lung", "pulmonar", "thoracic", "hrct", "hrtx", "ctthorax",
            "cardiac cap",
        ),
        protocol_keywords=(
            "thorax", "chest", "lung", "pulmonar", "hrct",
        ),
        study_body_parts=("thorax", "lung", "thorac", "chest"),
        z_coverage_min_mm=200.0,
        z_coverage_max_mm=750.0,
        min_mb=30.0,
        max_mb=900.0,
        exclude_study_desc_keywords=("abdomen", "pelvis"),
    ),
    ScanTypeConfig(
        # Combined chest + abdomen + pelvis studies (e.g. aortic dissection protocol,
        # whole-body CTA, trauma).  Must be checked BEFORE AbdomenCT so that
        # "CT Thorax/Abdomen/Pelvis" does not fall through to pure abdomen.
        name="TotalBodyCT",
        acq_label="ctbody",
        multi_phase=True,
        study_desc_keywords=(
            "thorax/abdomen", "chest/abdomen", "thorax abdomen", "chest abdomen",
            "body", "total body", "whole body", "totalbody",
            "dissection",
        ),
        protocol_keywords=(
            "dissection", "cap", "chest abdo pelvis", "whole body",
            "thorax abdo", "chest abdo",
        ),
        study_body_parts=(
            "chest_to_pelvis", "whole_body", "wholebody",
        ),
        z_coverage_min_mm=450.0,
        z_coverage_max_mm=None,
        min_mb=80.0,
        max_mb=1500.0,
    ),
    ScanTypeConfig(
        name="AbdomenCT",
        acq_label="ctabdomen",
        multi_phase=True,
        study_desc_keywords=(
            "abdomen", "abdomin", "liver", "pelvis", "abdopelv",
            "portal", "porto", "ctabdomen", "abdomen/pelvis",
        ),
        protocol_keywords=(
            "abdomen", "abdomin", "liver", "pelvis", "abdopelv", "portal",
        ),
        study_body_parts=(
            "abdomen", "abdo", "liver", "pelvis", "abdopelv",
        ),
        z_coverage_min_mm=200.0,
        z_coverage_max_mm=750.0,
        min_mb=30.0,
        max_mb=1200.0,
    ),
]

SCAN_TYPE_MAP: Dict[str, ScanTypeConfig] = {c.name: c for c in SCAN_TYPE_CONFIGS}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SeriesCandidate:
    uid: str
    files: List[str]
    meta: dict

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def series_number(self) -> int:
        try:
            return int(self.meta.get("series_number", ""))
        except (ValueError, TypeError):
            return 99999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_text(v) -> str:
    return str(v).strip() if v is not None else ""


def subject_id_from_name(name: str) -> str:
    if name.isdigit():
        return str(int(name))
    m = re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else name


def iter_candidate_files(export_dir: Path) -> Iterable[Path]:
    """Yield DICOM candidate files from one Export_* folder."""
    for p in export_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.upper() == "DICOMDIR":
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


# ---------------------------------------------------------------------------
# DICOM collection
# ---------------------------------------------------------------------------

def collect_series(export_dir: Path) -> Dict[str, SeriesCandidate]:
    """Group DICOM files in one Export_* folder by SeriesInstanceUID."""
    files_by_uid: Dict[str, List[str]] = defaultdict(list)
    meta_by_uid: Dict[str, dict] = {}

    for f in iter_candidate_files(export_dir):
        try:
            ds = pydicom.dcmread(
                str(f),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "Modality",
                    "SeriesInstanceUID",
                    "SeriesDescription",
                    "SeriesNumber",
                    "ProtocolName",
                    "StudyDescription",
                    "ImageComments",
                    "SliceThickness",
                    "Manufacturer",
                    "ManufacturerModelName",
                    "BodyPartExamined",
                    "PatientPosition",
                    "PatientSex",
                    "PatientBirthDate",
                    "StudyDate",
                ],
            )
        except Exception:
            continue

        if safe_text(getattr(ds, "Modality", "")).upper() != "CT":
            continue

        uid = safe_text(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            continue

        files_by_uid[uid].append(str(f))
        if uid not in meta_by_uid:
            meta_by_uid[uid] = {
                "series_description": safe_text(getattr(ds, "SeriesDescription", None)),
                "series_number":      safe_text(getattr(ds, "SeriesNumber", None)),
                "protocol_name":      safe_text(getattr(ds, "ProtocolName", None)),
                "study_description":  safe_text(getattr(ds, "StudyDescription", None)),
                "image_comments":     safe_text(getattr(ds, "ImageComments", None)),
                "slice_thickness":    safe_text(getattr(ds, "SliceThickness", None)),
                "manufacturer":       safe_text(getattr(ds, "Manufacturer", None)),
                "model":              safe_text(getattr(ds, "ManufacturerModelName", None)),
                "body_part":          safe_text(getattr(ds, "BodyPartExamined", None)),
                "patient_position":   safe_text(getattr(ds, "PatientPosition", None)),
                "patient_sex":        safe_text(getattr(ds, "PatientSex", None)),
                "patient_birth_date": safe_text(getattr(ds, "PatientBirthDate", None)),
                "study_date":         safe_text(getattr(ds, "StudyDate", None)),
            }

    return {
        uid: SeriesCandidate(uid=uid, files=files, meta=meta_by_uid.get(uid, {}))
        for uid, files in files_by_uid.items()
    }


# ---------------------------------------------------------------------------
# Study-level classification
# ---------------------------------------------------------------------------

def _sample_z_range(series_map: Dict[str, SeriesCandidate]) -> Optional[float]:
    """Estimate anatomical coverage (mm) from CT series in the export.

    Samples ~20 evenly-spaced files per eligible series and returns the
    MAXIMUM z-spread found across all series.  This is important because
    exports often contain sagittal/coronal reformats alongside the axial
    source series: the reformats have near-zero z-spread (their slices vary
    in x or y, not z), so picking the largest-by-file-count series can yield
    coverage ≈ 0 mm and incorrectly disqualify the export.  Taking the max
    across all series ensures the axial acquisition (which has the real
    head-to-foot coverage) drives the estimate.
    """
    ct_series = [c for c in series_map.values() if c.n_files >= MIN_SOURCE_SLICES]
    if not ct_series:
        return None

    best_coverage: Optional[float] = None

    for candidate in ct_series:
        files = sorted(candidate.files)
        step = max(1, len(files) // 20)
        z_vals = []
        for fp in files[::step]:
            try:
                ds = pydicom.dcmread(
                    str(fp), stop_before_pixels=True, force=True,
                    specific_tags=["ImagePositionPatient"],
                )
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp and len(ipp) >= 3:
                    z_vals.append(float(ipp[2]))
            except Exception:
                continue

        if len(z_vals) < 2:
            continue
        coverage = abs(max(z_vals) - min(z_vals))
        if best_coverage is None or coverage > best_coverage:
            best_coverage = coverage

    return best_coverage


def classify_export_study(
    series_map: Dict[str, SeriesCandidate],
    allowed_types: List[str],
) -> Optional[str]:
    """Classify the CT type of an export folder.

    Scoring (higher = more confident):
      +150  StudyDescription keyword match   (most reliable PACS label)
      +80   ProtocolName keyword match
      +50   BodyPartExamined match

    Coverage hard exclusion (scanner-independent, coverage_mm = z_max − z_min):
      If measured coverage falls outside a type's [z_coverage_min_mm, z_coverage_max_mm],
      that type is disqualified entirely (score forced to 0).
      If coverage is unavailable, exclusion is skipped (fall back to keywords only).

    At least one keyword or body-part signal is required (score ≥ 50).
    Returns None if nothing qualifies.
    """
    # Aggregate text per DICOM field separately to allow differential weighting.
    study_descs = " ".join(
        c.meta.get("study_description", "") for c in series_map.values()
    ).lower()
    proto_names = " ".join(
        c.meta.get("protocol_name", "") for c in series_map.values()
    ).lower()
    body_parts = " ".join(
        c.meta.get("body_part", "") for c in series_map.values()
    ).lower()

    # Sample coverage once (lightweight — reads ~20 files).
    coverage_mm = _sample_z_range(series_map)

    scores: Dict[str, int] = {}

    for type_name in allowed_types:
        cfg = SCAN_TYPE_MAP[type_name]

        # Hard coverage exclusion — disqualify if outside anatomical bounds.
        if coverage_mm is not None:
            if cfg.z_coverage_min_mm is not None and coverage_mm < cfg.z_coverage_min_mm:
                scores[type_name] = 0
                continue
            # TAVI protocols cover heart + full aorta + iliofemoral vessels,
            # so z can exceed CardiacCT's normal upper bound — skip z_max for
            # CardiacCT when "tavi" appears in either study description or
            # protocol name (sub-194 has "tavi" only in ProtocolName).
            skip_z_max = type_name == "CardiacCT" and (
                "tavi" in study_descs or "tavi" in proto_names
            )
            if not skip_z_max and cfg.z_coverage_max_mm is not None and coverage_mm > cfg.z_coverage_max_mm:
                scores[type_name] = 0
                continue

        # Study-description veto — disqualify if an exclusion keyword is present.
        if any(kw in study_descs for kw in cfg.exclude_study_desc_keywords):
            scores[type_name] = 0
            continue

        s = 0

        # StudyDescription — highest weight, most reliable.
        for kw in cfg.study_desc_keywords:
            if kw in study_descs:
                s += 150
                break

        # ProtocolName — medium weight.
        for kw in cfg.protocol_keywords:
            if kw in proto_names:
                s += 80
                break

        # BodyPartExamined.
        for bp in cfg.study_body_parts:
            if bp in body_parts:
                s += 50
                break

        scores[type_name] = s

    # Must have at least one text signal (keyword or body_part).
    TEXT_THRESHOLD = 50
    valid = {t: s for t, s in scores.items() if s >= TEXT_THRESHOLD}
    if not valid:
        return None

    best = max(valid, key=lambda t: valid[t])

    # Log classification details for transparency.
    cov_str = f"{coverage_mm:.0f}mm" if coverage_mm is not None else "unknown"
    print(f"      coverage≈{cov_str}  scores={dict(sorted(scores.items(), key=lambda x: -x[1]))}")

    return best


# ---------------------------------------------------------------------------
# Source series selection within a classified study
# ---------------------------------------------------------------------------

def _is_source_series(c: SeriesCandidate, min_slices: int = MIN_SOURCE_SLICES) -> bool:
    """Return True if the series looks like a volumetric source acquisition."""
    txt = " ".join([
        c.meta.get("series_description", ""),
        c.meta.get("protocol_name", ""),
    ]).lower()

    if any(bad in txt for bad in SERIES_BAD_KEYWORDS):
        return False
    if c.n_files < min_slices:
        return False
    return True


def select_source_series(
    series_map: Dict[str, SeriesCandidate],
    cfg: ScanTypeConfig,
) -> List[SeriesCandidate]:
    """Return the source series to convert for a classified study.

    For multi-phase types: all qualifying series ordered by SeriesNumber.
    For single-phase types: the single series with the most slices.
    """
    candidates = [c for c in series_map.values()
                  if _is_source_series(c, min_slices=cfg.min_source_slices)]

    if not candidates:
        return []

    # Sort by SeriesNumber (acquisition order).
    candidates.sort(key=lambda c: (c.series_number, -c.n_files))

    if cfg.multi_phase:
        return candidates
    else:
        # Best = most slices (primary CTA/thorax volume).
        return [max(candidates, key=lambda c: c.n_files)]


# ---------------------------------------------------------------------------
# Slice ordering & deduplication
# ---------------------------------------------------------------------------

def _ordered_dedup_file_list(file_list: List[str]) -> Tuple[List[str], dict]:
    """Deduplicate by SOPInstanceUID, filter to dominant image dimensions,
    and sort slices by physical z-position.

    Some series mix axial slices (512×512) with reformatted coronal/sagittal
    images (512×707, 664×512, …) under the same SeriesInstanceUID.  SimpleITK
    refuses to stack files of different sizes.  We keep only the files that
    share the most-common (Rows, Columns) pair, discarding the rest.
    """
    records = []
    for fp in file_list:
        try:
            ds = pydicom.dcmread(
                str(fp),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "SOPInstanceUID", "ImagePositionPatient",
                    "InstanceNumber", "Rows", "Columns",
                ],
            )
        except Exception:
            continue

        sop = safe_text(getattr(ds, "SOPInstanceUID", ""))
        inst_raw = getattr(ds, "InstanceNumber", None)
        try:
            inst = int(inst_raw) if inst_raw is not None else None
        except Exception:
            inst = None

        z = None
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                z = float(ipp[2])
            except Exception:
                z = None

        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)

        records.append((sop, z, inst, str(fp), rows, cols))

    if not records:
        uniq = sorted(set(file_list))
        return uniq, {"raw_rows": len(file_list), "used_rows": len(uniq),
                      "dedup_removed": len(file_list) - len(uniq), "ordered_by": "path"}

    # Deduplicate by SOPInstanceUID.
    dedup: Dict[str, tuple] = {}
    for sop, z, inst, fp, rows, cols in records:
        key = sop if sop else f"__PATH__::{fp}"
        prev = dedup.get(key)
        cur = (sop, z, inst, fp, rows, cols)
        if prev is None:
            dedup[key] = cur
            continue
        prev_s = int(prev[1] is not None) + int(prev[2] is not None)
        cur_s  = int(cur[1]  is not None) + int(cur[2]  is not None)
        if cur_s > prev_s or (cur_s == prev_s and len(cur[3]) < len(prev[3])):
            dedup[key] = cur

    uniq = list(dedup.values())

    # Filter to dominant image dimensions (Rows × Columns).
    # Series exported from PACS sometimes mix axial source slices (e.g. 512×512)
    # with reformatted coronal/sagittal images (512×707, 664×512, …) under the
    # same SeriesInstanceUID.  SimpleITK will raise a size-mismatch error when
    # trying to stack them — we keep only the most common dimension pair.
    from collections import Counter as _Counter
    dim_counts = _Counter(
        (r[4], r[5]) for r in uniq if r[4] is not None and r[5] is not None
    )
    if dim_counts:
        dominant = dim_counts.most_common(1)[0][0]
        uniq = [r for r in uniq if (r[4], r[5]) == dominant or r[4] is None]

    n = len(uniq)
    n_z    = sum(1 for r in uniq if r[1] is not None)
    n_inst = sum(1 for r in uniq if r[2] is not None)

    if n_z >= max(1, int(0.7 * n)):
        uniq.sort(key=lambda r: (r[1] if r[1] is not None else 1e18,
                                  r[2] if r[2] is not None else 1e18, r[3]))
        ordered_by = "z"
    elif n_inst > 0:
        uniq.sort(key=lambda r: (r[2] if r[2] is not None else 1e18, r[3]))
        ordered_by = "instance"
    else:
        uniq.sort(key=lambda r: r[3])
        ordered_by = "path"

    ordered = [r[3] for r in uniq]
    return ordered, {"raw_rows": len(file_list), "used_rows": len(ordered),
                     "dedup_removed": len(file_list) - len(ordered), "ordered_by": ordered_by}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def output_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _convert_to_nifti(file_list: List[str], out_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(file_list)
    sitk.WriteImage(reader.Execute(), str(out_path), useCompression=True)


@dataclass
class ConvertedSeries:
    phase_idx: int
    nii_path: Path
    n_slices: int
    series_desc: str
    series_uid: str
    status: str          # "converted" | "size_rejected" | "error" | "skip_exists"
    error_message: str = ""
    size_mb: float = 0.0


def _convert_one_series(
    candidate: SeriesCandidate,
    out_path: Path,
    cfg: ScanTypeConfig,
    phase_idx: Optional[int],
    min_mb: float,
    max_mb: float,
    rejected_root: Optional[Path] = None,
) -> ConvertedSeries:
    tmp = out_path.parent / f".tmp_{out_path.name}"
    if tmp.exists():
        tmp.unlink()
    try:
        ordered, order_info = _ordered_dedup_file_list(candidate.files)
        _convert_to_nifti(ordered, tmp)
        mb = output_size_mb(tmp)
        if mb < min_mb or mb > max_mb:
            rej_note = ""
            if rejected_root is not None:
                rej_dir = rejected_root / out_path.parent.name
                rej_dir.mkdir(parents=True, exist_ok=True)
                rej_path = rej_dir / out_path.name
                shutil.move(str(tmp), str(rej_path))
                rej_note = f" → {rej_path}"
            else:
                tmp.unlink(missing_ok=True)
            return ConvertedSeries(
                phase_idx=phase_idx or 0, nii_path=out_path,
                n_slices=order_info["used_rows"],
                series_desc=candidate.meta.get("series_description", ""),
                series_uid=candidate.uid,
                status="size_rejected",
                error_message=f"{mb:.1f} MB outside [{min_mb:.0f}, {max_mb:.0f}] MB{rej_note}",
                size_mb=round(mb, 1),
            )

        tmp.replace(out_path)

        sidecar = {
            "Modality": "CT",
            "ScanType": cfg.name,
            "AcqLabel": cfg.acq_label,
            "PhaseIndex": phase_idx,
            "SeriesDescription":      candidate.meta.get("series_description", ""),
            "SeriesNumber":           candidate.meta.get("series_number", ""),
            "ProtocolName":           candidate.meta.get("protocol_name", ""),
            "StudyDescription":       candidate.meta.get("study_description", ""),
            "BodyPartExamined":       candidate.meta.get("body_part", ""),
            "PatientPosition":        candidate.meta.get("patient_position", ""),
            "Manufacturer":           candidate.meta.get("manufacturer", ""),
            "ManufacturerModelName":  candidate.meta.get("model", ""),
            "PatientSex":             candidate.meta.get("patient_sex", ""),
            "PatientBirthDate":       candidate.meta.get("patient_birth_date", ""),
            "StudyDate":              candidate.meta.get("study_date", ""),
            "SeriesInstanceUID":      candidate.uid,
            "SourceFileCountRaw":     order_info["raw_rows"],
            "SourceFileCountUsed":    order_info["used_rows"],
            "SourceFilesDedupRemoved": order_info["dedup_removed"],
            "SourceFileOrder":        order_info["ordered_by"],
            "ConversionSoftware":     "SimpleITK",
            "ConversionSoftwareVersion": sitk.Version_VersionString(),
            "OutputSizeMB":           round(mb, 3),
        }
        json_path = out_path.with_suffix("").with_suffix(".json")
        json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        return ConvertedSeries(
            phase_idx=phase_idx or 0, nii_path=out_path,
            n_slices=order_info["used_rows"],
            series_desc=candidate.meta.get("series_description", ""),
            series_uid=candidate.uid,
            status="converted",
            size_mb=round(mb, 1),
        )

    except Exception as e:
        tmp.unlink(missing_ok=True)
        return ConvertedSeries(
            phase_idx=phase_idx or 0, nii_path=out_path,
            n_slices=0, series_desc="", series_uid=candidate.uid,
            status="error",
            error_message=f"{type(e).__name__}: {e}",
        )


def convert_export(
    sid: str,
    export_dir: Path,
    sub_dir: Path,
    cfg: ScanTypeConfig,
    min_mb: float,
    max_mb: float,
    series_map: Dict[str, SeriesCandidate],
    rejected_root: Optional[Path] = None,
) -> List[ConvertedSeries]:
    """Convert one Export_* folder (already classified) into NIfTI(s)."""

    # Skip if this type already exists for this subject.
    if cfg.multi_phase:
        existing = sorted(sub_dir.glob(f"sub-{sid}_acq-{cfg.acq_label}_ph*_ct.nii.gz"))
    else:
        existing = list(sub_dir.glob(f"sub-{sid}_acq-{cfg.acq_label}_ct.nii.gz"))

    if existing:
        return [ConvertedSeries(
            phase_idx=0, nii_path=existing[0], n_slices=0,
            series_desc="", series_uid="", status="skip_exists",
            size_mb=round(sum(output_size_mb(p) for p in existing), 1),
        )]

    source_series = select_source_series(series_map, cfg)
    if not source_series:
        return [ConvertedSeries(
            phase_idx=0,
            nii_path=sub_dir / f"sub-{sid}_acq-{cfg.acq_label}_ct.nii.gz",
            n_slices=0, series_desc="", series_uid="",
            status="error",
            error_message="no source series found after filtering",
        )]

    results: List[ConvertedSeries] = []
    for phase_idx, candidate in enumerate(source_series):
        if cfg.multi_phase:
            stem = f"sub-{sid}_acq-{cfg.acq_label}_ph{phase_idx:02d}_ct"
        else:
            stem = f"sub-{sid}_acq-{cfg.acq_label}_ct"
        out_path = sub_dir / f"{stem}.nii.gz"
        r = _convert_one_series(
            candidate, out_path, cfg,
            phase_idx if cfg.multi_phase else None,
            min_mb, max_mb,
            rejected_root=rejected_root,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Subject processing
# ---------------------------------------------------------------------------

def find_export_dirs(subject_dir: Path) -> List[Path]:
    """Return all Export_* subdirectories within a patient folder."""
    return sorted([
        p for p in subject_dir.iterdir()
        if p.is_dir() and p.name.startswith(EXPORT_PREFIX)
    ])


def process_subject(
    sid: str,
    subject_dir: Path,
    out_root: Path,
    allowed_types: List[str],
    min_mb_override: Optional[float],
    max_mb_override: Optional[float],
    rejected_root: Optional[Path] = None,
) -> Tuple[List[ConvertedSeries], dict]:
    """Process all Export_* folders for one patient."""
    export_dirs = find_export_dirs(subject_dir)
    if not export_dirs:
        # Fallback: treat the subject dir itself as the source (flat structure).
        export_dirs = [subject_dir]

    sub_dir = out_root / f"sub-{sid}"
    all_results: List[ConvertedSeries] = []
    manifest_row: dict = {
        "subject_id": sid,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dicom_dir": str(subject_dir),
    }

    # ── Phase 1: classify every export, keep the best (most source files) per type ──
    # "best" = export whose series_map has the highest total file count.
    # This avoids converting incomplete partial exports when a fuller one exists.
    best_per_type: Dict[str, Tuple[Path, Dict]] = {}   # type_name → (export_dir, series_map)
    best_n_files:  Dict[str, int]               = {}   # type_name → total files in best export

    _n_no_dicom     = 0  # Export_* dirs containing no DICOM files
    _n_unclassified = 0  # Export_* dirs with DICOM files that failed keyword/z scoring
    _n_with_dicom   = 0  # Export_* dirs with DICOM files (classified or unclassified)

    for export_dir in export_dirs:
        series_map = collect_series(export_dir)
        if not series_map:
            _n_no_dicom += 1
            continue
        _n_with_dicom += 1

        type_name = classify_export_study(series_map, allowed_types)
        if type_name is None:
            _n_unclassified += 1
            print(f"    [UNCLASSIFIED] {export_dir.name} — skipped")
            continue

        n_files = sum(c.n_files for c in series_map.values())
        prev_best = best_per_type.get(type_name)

        if prev_best is None:
            best_per_type[type_name] = (export_dir, series_map)
            best_n_files[type_name]  = n_files
            print(f"    {export_dir.name}  →  {type_name} ({SCAN_TYPE_MAP[type_name].acq_label})  [best so far: {n_files} files]")
        elif n_files > best_n_files[type_name]:
            prev_name = best_per_type[type_name][0].name
            print(f"    {export_dir.name}  →  {type_name} ({SCAN_TYPE_MAP[type_name].acq_label})  [replaces {prev_name}: {n_files} > {best_n_files[type_name]} files]")
            best_per_type[type_name] = (export_dir, series_map)
            best_n_files[type_name]  = n_files
        else:
            print(f"    [DUPLICATE {type_name}] {export_dir.name} skipped — {n_files} files ≤ best {best_n_files[type_name]} files")

    manifest_row["n_exports_found"] = _n_with_dicom
    manifest_row["n_no_dicom"]      = _n_no_dicom
    manifest_row["n_unclassified"]  = _n_unclassified

    # ── Phase 2: convert the best export for each classified type ──
    for type_name, (export_dir, series_map) in best_per_type.items():
        cfg = SCAN_TYPE_MAP[type_name]
        min_mb = min_mb_override if min_mb_override is not None else cfg.min_mb
        max_mb = max_mb_override if max_mb_override is not None else cfg.max_mb

        # Only create the subfolder if we have something to write.
        sub_dir.mkdir(parents=True, exist_ok=True)

        results = convert_export(sid, export_dir, sub_dir, cfg, min_mb, max_mb, series_map,
                                  rejected_root=rejected_root)
        all_results.extend(results)

        # Accumulate manifest columns for this type.
        label = cfg.acq_label
        converted = [r for r in results if r.status == "converted"]
        skipped   = [r for r in results if r.status == "skip_exists"]

        # StudyDate from DICOM (YYYYMMDD) — same for all series in the export.
        first_meta = next(iter(series_map.values())).meta if series_map else {}
        manifest_row[f"{label}_study_date"] = first_meta.get("study_date", "")

        if skipped and not converted:
            existing_files = (
                sorted(sub_dir.glob(f"sub-{sid}_acq-{label}_ph*_ct.nii.gz"))
                if cfg.multi_phase
                else list(sub_dir.glob(f"sub-{sid}_acq-{label}_ct.nii.gz"))
            )
            manifest_row[f"{label}_status"]      = "skip_exists"
            manifest_row[f"{label}_nifti"]       = "; ".join(str(p) for p in existing_files)
            manifest_row[f"{label}_n_slices"]    = ""
            manifest_row[f"{label}_series_desc"] = ""
            manifest_row[f"{label}_size_mb"]     = str(skipped[0].size_mb)
            if cfg.multi_phase:
                manifest_row[f"{label}_n_phases"] = str(len(existing_files))
        elif converted:
            manifest_row[f"{label}_status"]      = "converted"
            manifest_row[f"{label}_nifti"]       = "; ".join(str(r.nii_path) for r in converted)
            manifest_row[f"{label}_n_slices"]    = str(sum(r.n_slices for r in converted))
            manifest_row[f"{label}_series_desc"] = "; ".join(r.series_desc for r in converted)
            manifest_row[f"{label}_size_mb"]     = str(round(sum(r.size_mb for r in converted), 1))
            if cfg.multi_phase:
                manifest_row[f"{label}_n_phases"] = str(len(converted))
        else:
            errs = "; ".join(r.error_message for r in results if r.error_message)
            manifest_row[f"{label}_status"]      = results[0].status if results else "error"
            manifest_row[f"{label}_nifti"]       = ""
            manifest_row[f"{label}_n_slices"]    = ""
            manifest_row[f"{label}_series_desc"] = ""
            manifest_row[f"{label}_size_mb"]     = ""
            manifest_row[f"{label}_error"]       = errs
            if cfg.multi_phase:
                manifest_row[f"{label}_n_phases"] = ""

    return all_results, manifest_row


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _write_html_report(
    manifest_rows: List[dict],
    scan_type_configs: List[ScanTypeConfig],
    allowed_types: List[str],
    src_root: str,
    out_root: str,
    report_path: "Path",
    summary_mode: bool = False,
) -> None:
    """Write a self-contained HTML summary of the conversion run.

    summary_mode: when True, both "converted" and "skip_exists" badges are
    shown as green "available" so non-technical readers find the report
    more intuitive.  All other content (filters, details, JS) is identical.
    """

    active_cfgs = [c for c in scan_type_configs if c.name in allowed_types]
    labels = [c.acq_label for c in active_cfgs]

    # Human-readable display names and HTML data-attribute keys per acq_label.
    DISPLAY_NAMES: Dict[str, str] = {
        "ecta":      "eCTA",
        "ctheart":   "CT_heart",
        "ctthorax":  "CT_thorax",
        "ctbody":    "CT_totalbody",
        "ctabdomen": "CT_abdomen",
    }
    DATA_KEYS: Dict[str, str] = {
        "ecta":      "ecta",
        "ctheart":   "ctheart",
        "ctthorax":  "ctthorax",
        "ctbody":    "ctbody",
        "ctabdomen": "ctabdomen",
    }

    # Separate special annotation rows (already_exists, duplicate_source) from
    # the main manifest so they are rendered in a dedicated notes section below
    # the table and do not interfere with JS filtering.
    normal_rows  = [r for r in manifest_rows if not r.get("_row_type")]
    special_rows = [r for r in manifest_rows if r.get("_row_type")]

    # Tally summary counts (normal rows only).
    totals: Dict[str, Counter] = {lbl: Counter() for lbl in labels}
    for row in normal_rows:
        for lbl in labels:
            st = row.get(f"{lbl}_status", "")
            if st:
                totals[lbl][st] += 1

    badge_css = {
        "converted":    ("badge-ok",   "converted"),
        "skip_exists":  ("badge-skip", "already exists"),
        "size_rejected":("badge-warn", "size rejected"),
        "error":        ("badge-err",  "error"),
    }
    if summary_mode:
        badge_css["converted"]  = ("badge-ok", "available")
        badge_css["skip_exists"] = ("badge-ok", "available")

    def badge(status: str, detail: str = "") -> str:
        css, label = badge_css.get(status, ("badge-unk", status or "—"))
        tip = f' title="{detail}"' if detail else ""
        return f'<span class="{css}"{tip}>{label}</span>'

    def esc(s) -> str:
        return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Build table rows (normal rows only — special rows rendered separately below).
    rows_html = []
    row_type_counts: Counter = Counter()
    for row in normal_rows:
        sid = row.get("subject_id", "")
        cells = [f"<td><b>sub-{esc(sid)}</b></td>"]
        _n_exp         = row.get("n_exports_found", "")
        _n_no_dicom    = int(row.get("n_no_dicom",    0) or 0)
        _n_unclassif   = int(row.get("n_unclassified", 0) or 0)
        _exp_cell      = f"<td>{esc(str(_n_exp))}"
        if _n_no_dicom > 0:
            _exp_cell += f'<br><span class="badge-err">error</span><br><small class="warn-detail">no DICOM ({_n_no_dicom})</small>'
        if _n_unclassif > 0:
            _exp_cell += f'<br><span class="badge-err">error</span><br><small class="warn-detail">unclassified ({_n_unclassif})</small>'
        _exp_cell += "</td>"
        cells.append(_exp_cell)

        # Track whether at least one type was successfully obtained for this subject.
        any_success = False
        any_rejected = False
        # Per-type state keyed by data attribute name (absent|converted|size_rejected|error).
        type_states: Dict[str, str] = {DATA_KEYS.get(cfg.acq_label, cfg.acq_label): "absent" for cfg in active_cfgs}

        for cfg in active_cfgs:
            lbl = cfg.acq_label
            dk  = DATA_KEYS.get(lbl, lbl)
            st  = row.get(f"{lbl}_status", "")
            if not st:
                cells.append("<td>—</td>")
                continue
            if st in ("converted", "skip_exists"):
                any_success = True
                type_states[dk] = "converted"
            elif st == "size_rejected":
                any_rejected = True
                type_states[dk] = "size_rejected"
            elif st == "error":
                type_states[dk] = "error"
            nifti = row.get(f"{lbl}_nifti", "")
            # Show only filenames, not full paths.
            fnames = "; ".join(Path(p.strip()).name for p in nifti.split(";") if p.strip())
            err = row.get(f"{lbl}_error", "")
            slices = row.get(f"{lbl}_n_slices", "")
            phases = row.get(f"{lbl}_n_phases", "")
            size_mb = row.get(f"{lbl}_size_mb", "")
            detail = err or fnames
            b = badge(st, detail)
            extra = ""
            if st in ("converted", "skip_exists") and not summary_mode:
                parts = []
                if phases:
                    parts.append(f"{phases} ph")
                if slices:
                    parts.append(f"{slices} sl")
                if size_mb:
                    parts.append(f"{size_mb} MB")
                if parts:
                    extra = f'<br><small>{", ".join(parts)}</small>'
                if fnames:
                    extra += f'<br><small class="fname">{esc(fnames)}</small>'
            elif st == "size_rejected" and err:
                extra = f'<br><small class="warn-detail">{esc(err)}</small>'
            elif st == "error":
                extra = '<br><small class="warn-detail">convert</small>'
            cells.append(f"<td>{b}{extra}</td>")

        if any_success:
            row_type = "converted"
        elif any_rejected:
            row_type = "rejected"
        else:
            row_type = "missing"
        row_type_counts[row_type] += 1

        # Build per-type data attributes for JS filtering.
        _type_attrs = " ".join(
            f'data-{DATA_KEYS.get(cfg.acq_label, cfg.acq_label)}="{type_states.get(DATA_KEYS.get(cfg.acq_label, cfg.acq_label), "absent")}"'
            for cfg in active_cfgs
        )
        data_attrs = f' data-rowtype="{row_type}" {_type_attrs}'
        row_cls = "" if any_success else ' class="row-unprocessed"'
        rows_html.append(f"<tr{row_cls}{data_attrs}>" + "".join(cells) + "</tr>")

    # Header columns.
    header_cols = (
        ["Subject", "Exports"]
        + [f'<abbr title="{c.name}">{DISPLAY_NAMES.get(c.acq_label, c.acq_label)}</abbr>' for c in active_cfgs]
    )
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in header_cols) + "</tr>"

    # Summary cards.
    cards = []
    for cfg in active_cfgs:
        lbl = cfg.acq_label
        t = totals[lbl]
        ok   = t.get("converted", 0)
        skip = t.get("skip_exists", 0)
        warn = t.get("size_rejected", 0)
        err  = t.get("error", 0)
        if summary_mode:
            available = ok + skip
            cards.append(
                f'<div class="card"><b>{DISPLAY_NAMES.get(lbl, lbl)}</b>'
                f'<br><span class="badge-ok">{available} available</span>'
                f'<br><span class="badge-warn">{warn} size-rej</span>'
                f'<br><span class="badge-err">{err} errors</span>'
                f'</div>'
            )
        else:
            cards.append(
                f'<div class="card"><b>{DISPLAY_NAMES.get(lbl, lbl)}</b>'
                f'<br><span class="badge-ok">{ok} converted</span>'
                f'<br><span class="badge-skip">{skip} skipped</span>'
                f'<br><span class="badge-warn">{warn} size-rej</span>'
                f'<br><span class="badge-err">{err} errors</span>'
                f'</div>'
            )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_subj = len(normal_rows)

    # Build checkbox filter rows — one per CT type, four state checkboxes + visible count cell.
    # Checkboxes start CHECKED; unchecking EXCLUDES patients with that CT type + state.
    _filter_rows = []
    for cfg in active_cfgs:
        dk  = DATA_KEYS.get(cfg.acq_label, cfg.acq_label)
        lbl = DISPLAY_NAMES.get(cfg.acq_label, cfg.acq_label)
        _filter_rows.append(
            f'<tr>'
            f'<td class="ftype-label"><label class="row-hdr-lbl">'
            f'{lbl}'
            f'</label></td>'
            f'<td class="fc-converted"><input type="checkbox" class="cell-cb" data-key="{dk}" data-state="converted" onchange="applyFilters()" checked></td>'
            f'<td class="fc-size_rejected"><input type="checkbox" class="cell-cb" data-key="{dk}" data-state="size_rejected" onchange="applyFilters()" checked></td>'
            f'<td class="fc-error"><input type="checkbox" class="cell-cb" data-key="{dk}" data-state="error" onchange="applyFilters()" checked></td>'
            f'<td class="fc-absent"><input type="checkbox" class="cell-cb" data-key="{dk}" data-state="absent" onchange="applyFilters()" checked></td>'
            f'<td class="frow-count" id="frc-{dk}">—</td>'
            f'</tr>'
        )
    filter_rows_html = "\n".join(_filter_rows)

    # Static TOTAL row for the filter table (display-only, no checkboxes; values filled by JS).
    total_row_html = (
        '<tr class="ftotal-row">'
        '<td class="ftype-label ftotal-label">TOTAL</td>'
        '<td class="fc-converted ftotal-cell" id="ftotal-converted">—</td>'
        '<td class="fc-size_rejected ftotal-cell" id="ftotal-size_rejected">—</td>'
        '<td class="fc-error ftotal-cell" id="ftotal-error">—</td>'
        '<td class="fc-absent ftotal-cell" id="ftotal-absent">—</td>'
        '<td class="frow-count ftotal-cell" id="ftotal-visible">—</td>'
        '</tr>'
    )

    # JS array of {{key, label}} for all CT types (injected from Python so it stays in sync).
    ct_types_js = "[" + ",".join(
        f'{{"key":"{DATA_KEYS.get(cfg.acq_label, cfg.acq_label)}",'
        f'"label":"{DISPLAY_NAMES.get(cfg.acq_label, cfg.acq_label)}"}}'
        for cfg in active_cfgs
    ) + "]"

    # ── Build notes section for special rows ────────────────────────────────────
    already_exists_rows = [r for r in special_rows if r.get("_row_type") == "already_exists"]
    dup_source_rows     = [r for r in special_rows if r.get("_row_type") == "duplicate_source"]

    _notes_parts = []
    if already_exists_rows:
        _ae_rows = "".join(
            f'<tr class="note-already-exists">'
            f'<td><b>sub-{esc(r.get("subject_id",""))}</b></td>'
            f'<td><span class="badge-skip">already exists</span></td>'
            f'<td><small>{esc(r.get("_note",""))}</small></td>'
            f'</tr>'
            for r in sorted(already_exists_rows, key=lambda r: int(r.get("subject_id") or 0))
        )
        _notes_parts.append(
            f'<h3 style="font-size:13px;color:#555;margin:12px 0 6px 0;">'
            f'Already converted ({len(already_exists_rows)}) — skipped</h3>'
            f'<table class="notes-table">'
            f'<thead><tr><th>Subject</th><th>Status</th><th>Path</th></tr></thead>'
            f'<tbody>{_ae_rows}</tbody></table>'
        )
    if dup_source_rows:
        _ds_rows = "".join(
            f'<tr class="note-dup-source">'
            f'<td><b>sub-{esc(r.get("subject_id",""))}</b></td>'
            f'<td><span class="badge-warn">duplicate source</span></td>'
            f'<td><small>{esc(r.get("_note",""))}</small></td>'
            f'</tr>'
            for r in sorted(dup_source_rows, key=lambda r: int(r.get("subject_id") or 0))
        )
        _notes_parts.append(
            f'<h3 style="font-size:13px;color:#555;margin:12px 0 6px 0;">'
            f'Duplicate source folders ({len(dup_source_rows)}) — auto-resolved by file count</h3>'
            f'<table class="notes-table">'
            f'<thead><tr><th>Subject</th><th>Status</th><th>Details</th></tr></thead>'
            f'<tbody>{_ds_rows}</tbody></table>'
        )

    notes_section_html = (
        f'<div class="notes-section"><h2>Notes</h2>{"".join(_notes_parts)}</div>'
        if _notes_parts else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Conversion Report</title>
<style>
  body {{font-family:Arial,sans-serif;font-size:13px;margin:24px;color:#222;}}
  h1 {{font-size:18px;margin-bottom:4px;}}
  .meta {{color:#666;font-size:12px;margin-bottom:16px;}}
  .cards {{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;}}
  .card {{background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:10px 16px;min-width:120px;line-height:1.8;}}
  /* ── Summary panel ── */
  .summary-panel {{background:#f0f4ff;border:1px solid #c5cae9;border-radius:8px;padding:14px 18px;margin-bottom:16px;}}
  .summary-section {{margin-bottom:14px;}}
  .summary-section:last-child {{margin-bottom:0;}}
  .summary-section h3 {{font-size:13px;font-weight:bold;color:#283593;margin:0 0 8px 0;border-bottom:1px solid #c5cae9;padding-bottom:4px;}}
  .sa-row {{display:flex;gap:0;margin-bottom:8px;flex-wrap:wrap;}}
  .sa-stat {{display:flex;flex-direction:column;align-items:center;background:#fff;border:1px solid #c5cae9;padding:6px 16px;min-width:100px;}}
  .sa-stat:first-child {{border-radius:6px 0 0 6px;}}
  .sa-stat:last-child {{border-radius:0 6px 6px 0;}}
  .sa-stat+.sa-stat {{border-left:none;}}
  .sa-num {{font-size:20px;font-weight:bold;line-height:1.2;}}
  .sa-lbl {{font-size:11px;color:#555;text-align:center;white-space:nowrap;}}
  .stacked-bar {{display:flex;height:18px;border-radius:4px;overflow:hidden;margin-bottom:6px;background:#e0e0e0;}}
  .bar-0 {{background:#e0e0e0;}}
  .bar-1 {{background:#81c784;}}
  .bar-2 {{background:#4fc3f7;}}
  .bar-3 {{background:#7986cb;}}
  .bar-legend {{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#555;}}
  .bar-legend span {{display:flex;align-items:center;gap:4px;}}
  .leg-dot {{width:10px;height:10px;border-radius:2px;display:inline-block;flex-shrink:0;}}
  /* ── Filter panel ── */
  .filter-panel {{background:#f8f9ff;border:1px solid #c5cae9;border-radius:8px;padding:12px 16px;margin-bottom:16px;}}
  .filter-header {{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap;}}
  .filter-title {{font-weight:bold;font-size:13px;color:#283593;}}
  .reset-btn {{padding:4px 12px;border:1px solid #bbb;border-radius:12px;background:#fff;font-size:12px;cursor:pointer;}}
  .reset-btn:hover {{background:#e8eaf6;border-color:#7986cb;}}
  #row-count {{font-size:12px;color:#666;margin-left:auto;}}
  .filter-table {{border-collapse:collapse;font-size:12px;}}
  .filter-table th,.filter-table td {{padding:4px 14px;text-align:center;border:none;}}
  .ftype-hdr {{text-align:left;color:#444;font-weight:bold;min-width:110px;padding-left:0;}}
  .ftype-label {{text-align:left;font-weight:bold;color:#333;padding:5px 14px 5px 0;white-space:nowrap;}}
  .fstate-hdr {{font-weight:bold;}}
  .row-hdr-lbl {{cursor:pointer;font-weight:bold;color:#333;}}
  .col-hdr-lbl {{cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:2px;}}
  .frow-count {{text-align:right;font-size:12px;font-weight:bold;color:#283593;padding:4px 8px 4px 14px;white-space:nowrap;}}
  .ftotal-row td {{background:#eef0fa !important;border-top:2px solid #c5cae9;}}
  .ftotal-label {{color:#283593;font-weight:bold;}}
  .ftotal-cell {{font-size:12px;font-weight:bold;text-align:center;}}
  .fc-converted {{color:#2e7d32;}} .fstate-hdr.fc-converted {{color:#2e7d32;}}
  .fc-size_rejected {{color:#e65100;}} .fstate-hdr.fc-size_rejected {{color:#e65100;}}
  .fc-error {{color:#b71c1c;}} .fstate-hdr.fc-error {{color:#b71c1c;}}
  .fc-absent {{color:#757575;}} .fstate-hdr.fc-absent {{color:#757575;}}
  /* ── Table ── */
  table {{border-collapse:collapse;width:100%;table-layout:fixed;}}
  th,td {{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top;}}
  #report-tbody td,thead th {{width:16%;min-width:16%;max-width:16%;white-space:normal;word-break:break-word;vertical-align:top;}}
  th {{background:#e8eaf6;position:sticky;top:0;z-index:1;}}
  tr:nth-child(even){{background:#fafafa;}}
  .badge-ok   {{background:#c8e6c9;color:#1b5e20;padding:1px 6px;border-radius:4px;font-size:11px;}}
  .badge-skip {{background:#fff9c4;color:#795548;padding:1px 6px;border-radius:4px;font-size:11px;}}
  .badge-warn {{background:#ffe0b2;color:#bf360c;padding:1px 6px;border-radius:4px;font-size:11px;}}
  .badge-err  {{background:#ffcdd2;color:#b71c1c;padding:1px 6px;border-radius:4px;font-size:11px;}}
  .badge-unk  {{background:#e0e0e0;color:#555;padding:1px 6px;border-radius:4px;font-size:11px;}}
  .fname {{color:#555;word-break:break-all;}}
  .warn-detail {{color:#bf360c;}}
  .row-unprocessed td {{background:#fff3e0 !important;}}
  .row-unprocessed td:first-child {{border-left:4px solid #e65100;}}
  small {{font-size:11px;}}
  tr.hidden {{display:none;}}
  /* ── Notes section (already_exists / duplicate_source) ── */
  .notes-section {{margin-top:24px;}}
  .notes-section h2 {{font-size:14px;color:#444;margin-bottom:8px;}}
  .notes-table {{border-collapse:collapse;width:100%;font-size:12px;}}
  .notes-table th {{background:#eee;padding:5px 10px;text-align:left;border:1px solid #ccc;}}
  .notes-table td {{padding:4px 10px;border:1px solid #ccc;vertical-align:top;}}
  .notes-table tr.note-already-exists td {{background:#f5f5f5;color:#666;}}
  .notes-table tr.note-dup-source td {{background:#fff8e1;}}
</style>
</head>
<body>
<h1>DICOM → NIfTI Conversion Report</h1>
<div class="meta">
  Generated: {now} &nbsp;|&nbsp;
  Subjects: {n_subj} &nbsp;|&nbsp;
  Source: <code>{esc(src_root)}</code> &nbsp;|&nbsp;
  Output: <code>{esc(out_root)}</code>
</div>
<div class="cards">{''.join(cards)}</div>
<div class="summary-panel">
  <div class="summary-section">
    <h3>Patients by number of available CT types</h3>
    <div id="sa-bar"></div>
    <div id="sa-counts"></div>
    <div class="bar-legend">
      <span><i class="leg-dot" style="background:#e0e0e0"></i>0 CT — no usable NIfTI</span>
      <span><i class="leg-dot" style="background:#81c784"></i>1 CT</span>
      <span><i class="leg-dot" style="background:#4fc3f7"></i>2 CT</span>
      <span><i class="leg-dot" style="background:#7986cb"></i>3+ CT</span>
    </div>
  </div>
  </div>
</div>
<div class="filter-panel">
  <div class="filter-header">
    <span class="filter-title">Filter patients</span>
    <label style="font-size:13px;cursor:pointer;"><input type="checkbox" id="select-all-cb" onchange="toggleSelectAll(this)" checked> Select all</label>
    <button class="reset-btn" onclick="resetFilters()">Reset all filters</button>
    <span id="row-count">Showing {n_subj} of {n_subj} patients</span>
  </div>
  <table class="filter-table">
    <thead><tr>
      <th class="ftype-hdr">all</th>
      <th class="fstate-hdr fc-converted">converted</th>
      <th class="fstate-hdr fc-size_rejected">size rejected</th>
      <th class="fstate-hdr fc-error">error</th>
      <th class="fstate-hdr fc-absent">absent</th>
      <th class="ftype-hdr" style="text-align:right">Visible</th>
    </tr></thead>
    <tbody>{total_row_html}
{filter_rows_html}
    </tbody>
  </table>
</div>
<div id="no-filter-msg" style="display:none;padding:20px 0;color:#888;font-style:italic;">
  No filters active — select at least one state to display patients, or click Reset to restore all filters.
</div>
<table id="report-table">
<thead>{thead}</thead>
<tbody id="report-tbody">
{''.join(rows_html)}
</tbody>
</table>
<script>
  var CT_TYPES = {ct_types_js};
  var CT_KEYS  = CT_TYPES.map(function(t) {{ return t.key; }});
  var TOTAL    = {n_subj};

  // Collect checked states per CT-type key.
  // Keys with zero checked boxes are "unconstrained" (no filter applied for that type).
  function _getChecked() {{
    var checked = {{}};
    CT_KEYS.forEach(function(k) {{ checked[k] = []; }});
    document.querySelectorAll('.filter-panel .cell-cb:checked').forEach(function(cb) {{
      checked[cb.dataset.key].push(cb.dataset.state);
    }});
    return checked;
  }}

  // Recount visible rows and push numbers into TOTAL row + per-row count cells.
  function _updateCounts() {{
    var rows = document.querySelectorAll('#report-tbody tr');
    var visible = 0;
    var rowCounts = {{}};
    var stateTotals = {{converted:0, size_rejected:0, error:0, absent:0}};
    CT_KEYS.forEach(function(k) {{ rowCounts[k] = 0; }});
    rows.forEach(function(row) {{
      if (row.classList.contains('hidden')) return;
      visible++;
      CT_KEYS.forEach(function(k) {{
        var s = row.dataset[k] || 'absent';
        if (s !== 'absent') rowCounts[k]++;
        if (stateTotals[s] !== undefined) stateTotals[s]++;
      }});
    }});
    document.getElementById('row-count').textContent = 'Showing ' + visible + ' of ' + TOTAL + ' patients';
    CT_KEYS.forEach(function(k) {{
      var el = document.getElementById('frc-' + k);
      if (el) el.textContent = rowCounts[k];
    }});
    ['converted','size_rejected','error','absent'].forEach(function(s) {{
      var el = document.getElementById('ftotal-' + s);
      if (el) el.textContent = stateTotals[s];
    }});
    var tv = document.getElementById('ftotal-visible');
    if (tv) tv.textContent = visible;
  }}

  // Apply inclusion logic: a CT type is constrained only when ≥1 checkbox is checked for it.
  // Patient is visible iff for every constrained CT type its state is in the checked set.
  // CT types with NO checked boxes impose no constraint at all.
  function applyFilters() {{
    var checked = _getChecked();
    var rows = document.querySelectorAll('#report-tbody tr');
    rows.forEach(function(row) {{
      var hide = false;
      CT_KEYS.forEach(function(key) {{
        if (hide || checked[key].length === 0) return;
        if (checked[key].indexOf(row.dataset[key] || 'absent') === -1) hide = true;
      }});
      if (hide) row.classList.add('hidden');
      else      row.classList.remove('hidden');
    }});
    _updateCounts();
    _updateEmptyState();
  }}

  function _updateEmptyState() {{
    var anyChecked = document.querySelector('.filter-panel .cell-cb:checked') !== null;
    var msg = document.getElementById('no-filter-msg');
    var tbl = document.getElementById('report-table');
    var countEl = document.getElementById('row-count');
    if (!anyChecked) {{
      if (msg) msg.style.display = 'block';
      if (tbl) tbl.style.display = 'none';
      if (countEl) countEl.textContent = 'Showing 0 of ' + TOTAL + ' patients';
    }} else {{
      if (msg) msg.style.display = 'none';
      if (tbl) tbl.style.display = '';
    }}
  }}

  function toggleSelectAll(cb) {{
    document.querySelectorAll('.filter-panel .cell-cb').forEach(function(c) {{ c.checked = cb.checked; }});
    applyFilters();
  }}

  // Re-check everything and restore all rows.
  function resetFilters() {{
    document.querySelectorAll('.filter-panel input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
    document.querySelectorAll('#report-tbody tr').forEach(function(row) {{ row.classList.remove('hidden'); }});
    _updateCounts();
    _updateEmptyState();
  }}

  function initCounts() {{
    // All checkboxes start checked → all rows visible. Compute initial counts.
    _updateCounts();

    // Build stacked bar from all patient rows.
    var rows = document.querySelectorAll('#report-tbody tr');
    var nWith0 = 0, nWith1 = 0, nWith2 = 0, nWith3plus = 0;
    rows.forEach(function(row) {{
      var present = CT_KEYS.filter(function(k) {{ return row.dataset[k] === 'converted'; }});
      var n = present.length;
      if      (n === 0) nWith0++;
      else if (n === 1) nWith1++;
      else if (n === 2) nWith2++;
      else              nWith3plus++;
    }});

    var total = nWith0 + nWith1 + nWith2 + nWith3plus;
    var pct   = function(n) {{ return total > 0 ? (n / total * 100).toFixed(1) : '0'; }};
    document.getElementById('sa-bar').innerHTML =
      '<div class="stacked-bar">' +
      (nWith0     > 0 ? '<div class="bar-0" style="width:' + pct(nWith0)     + '%" title="0 CT: ' + nWith0     + '"></div>' : '') +
      (nWith1     > 0 ? '<div class="bar-1" style="width:' + pct(nWith1)     + '%" title="1 CT: ' + nWith1     + '"></div>' : '') +
      (nWith2     > 0 ? '<div class="bar-2" style="width:' + pct(nWith2)     + '%" title="2 CT: ' + nWith2     + '"></div>' : '') +
      (nWith3plus > 0 ? '<div class="bar-3" style="width:' + pct(nWith3plus) + '%" title="3+ CT: ' + nWith3plus + '"></div>' : '') +
      '</div>';

    document.getElementById('sa-counts').innerHTML =
      '<div class="sa-row">' +
      '<div class="sa-stat"><span class="sa-num" style="color:#283593">'  + total     + '</span><span class="sa-lbl">Total</span></div>' +
      '<div class="sa-stat"><span class="sa-num" style="color:#555">'     + nWith0    + '</span><span class="sa-lbl">No usable NIfTI</span></div>' +
      '<div class="sa-stat"><span class="sa-num" style="color:#2e7d32">'  + nWith1    + '</span><span class="sa-lbl">1 CT type</span></div>' +
      '<div class="sa-stat"><span class="sa-num" style="color:#0277bd">'  + nWith2    + '</span><span class="sa-lbl">2 CT types</span></div>' +
      '<div class="sa-stat"><span class="sa-num" style="color:#4527a0">'  + nWith3plus+ '</span><span class="sa-lbl">3+ CT types</span></div>' +
      '</div>';
  }}

  document.addEventListener('DOMContentLoaded', initCounts);
</script>
{notes_section_html}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    print(f"Report   : {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_names = [c.name for c in SCAN_TYPE_CONFIGS]
    p = argparse.ArgumentParser(
        description="Convert DICOM exports (one Export_* folder = one study) to SLAAOBIDS NIfTI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--src-root", default="D:/",
        help="Root containing per-patient numbered directories (default: D:/)",
    )
    p.add_argument(
        "--out-root",
        default=r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS",
        help="Output SLAAOBIDS root",
    )
    p.add_argument(
        "--scan-types", nargs="+", choices=all_names, default=all_names,
        metavar="TYPE",
        help=f"Scan types to process. Choices: {all_names}",
    )
    p.add_argument("--min-mb", type=float, default=None)
    p.add_argument("--max-mb", type=float, default=None)
    p.add_argument(
        "--rejected-root",
        default=r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS\REJECTED",
        help="Directory where size-rejected NIfTI files are saved as sub-XXX/<filename>.nii.gz",
    )
    p.add_argument(
        "--subject", action="append", default=[],
        help="Restrict to subject id(s); repeatable, e.g. --subject 224",
    )
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel worker processes (default: half of CPU count)",
    )
    return p.parse_args()


def _process_subject_worker(
    sid: str,
    subject_dir: Path,
    out_root: Path,
    allowed_types: List[str],
    min_mb_override: Optional[float],
    max_mb_override: Optional[float],
    rejected_root: Optional[Path] = None,
) -> Tuple[List[ConvertedSeries], dict, str]:
    """Top-level worker for ProcessPoolExecutor (must be picklable on Windows)."""
    buf = io.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        results, manifest_row = process_subject(
            sid=sid,
            subject_dir=subject_dir,
            out_root=out_root,
            allowed_types=allowed_types,
            min_mb_override=min_mb_override,
            max_mb_override=max_mb_override,
            rejected_root=rejected_root,
        )
    finally:
        sys.stdout = old_stdout
    return results, manifest_row, buf.getvalue()


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    rejected_root = Path(args.rejected_root) if args.rejected_root else None

    subject_dirs = sorted(
        [p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith(("$", "."))],
        # Primary: numeric subject id; secondary: plain numeric names first so
        # "557" beats "Folder 557" if both resolve to the same id.
        key=lambda p: (
            int(subject_id_from_name(p.name)) if subject_id_from_name(p.name).isdigit() else float("inf"),
            0 if p.name.isdigit() else 1,
        ),
    )
    wanted = {str(int(s)) for s in args.subject if str(s).isdigit()}

    print(f"Scan types : {args.scan_types}")
    print(f"Source     : {src_root}")
    print(f"Output     : {out_root}")
    print(f"Rejected   : {rejected_root or '(discard)'}")
    print(f"Subjects   : {len(subject_dirs)} directories found")
    print(f"Workers    : {args.workers}")
    print()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = out_root / "conversion Report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / f"conversion_manifest_{run_ts}.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    all_manifest_rows: List[dict] = []
    counts: Counter = Counter()

    # ── Resolve duplicate source folders by file count ──────────────────────────
    # Group all source dirs by numeric sid, then pick the dir with the most files.
    sid_to_dirs: Dict[str, List[Path]] = defaultdict(list)
    for subj in subject_dirs:
        sid = subject_id_from_name(subj.name)
        if not sid.isdigit():
            continue
        if wanted and sid not in wanted:
            continue
        sid_to_dirs[sid].append(subj)

    def _count_files_fast(d: Path) -> int:
        """Quick recursive file count (no DICOM parsing) for duplicate ranking."""
        return sum(
            1 for p in d.rglob("*")
            if p.is_file() and p.name.upper() != "DICOMDIR"
            and not any(part in SKIP_DIRS for part in p.parts)
        )

    work_items: List[Tuple[str, Path]] = []
    for sid in sorted(sid_to_dirs, key=lambda s: int(s)):
        dirs = sid_to_dirs[sid]
        if len(dirs) == 1:
            work_items.append((sid, dirs[0]))
        else:
            counted = [(d, _count_files_fast(d)) for d in dirs]
            counted.sort(key=lambda x: -x[1])
            best_dir, best_count = counted[0]
            work_items.append((sid, best_dir))
            for dup_dir, dup_count in counted[1:]:
                note = (f"Dropped: {dup_dir.name} ({dup_count} files); "
                        f"Kept: {best_dir.name} ({best_count} files)")
                print(f"[dup_source] sub-{sid}: {note}")
                all_manifest_rows.append({
                    "subject_id": sid,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "_row_type": "duplicate_source",
                    "_note": note,
                })

    # ── Skip already-converted subjects ─────────────────────────────────────────
    # If sub-{sid} already exists in out_root with at least one .nii.gz, skip.
    final_work_items: List[Tuple[str, Path]] = []
    for sid, subj in work_items:
        sub_dir = out_root / f"sub-{sid}"
        if sub_dir.exists() and any(f for f in sub_dir.glob("*.nii.gz") if not f.name.startswith(".")):
            print(f"[already_exists] sub-{sid} — already in {out_root}")
            all_manifest_rows.append({
                "subject_id": sid,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "_row_type": "already_exists",
                "_note": str(sub_dir),
            })
        else:
            final_work_items.append((sid, subj))

    futures_map: dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sid, subj in final_work_items:
            export_dirs = find_export_dirs(subj)
            print(f"[sub-{sid}]  {len(export_dirs)} export folder(s)  → queued")
            fut = executor.submit(
                _process_subject_worker,
                sid, subj, out_root, args.scan_types, args.min_mb, args.max_mb,
                rejected_root,
            )
            futures_map[fut] = sid

        for fut in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Converting", unit="subject", dynamic_ncols=True):
            sid = futures_map[fut]
            try:
                results, manifest_row, log_output = fut.result()
            except Exception as e:
                traceback.print_exc()
                print(f"  [ERROR sub-{sid}] {e}")
                continue

            if log_output:
                print(log_output, end="")

            for r in results:
                counts[r.status] += 1
                if r.status == "converted":
                    print(f"    [OK]  {r.nii_path.name}")
                elif r.status == "skip_exists":
                    print(f"    [SKIP] {r.nii_path.name} already exists")
                else:
                    print(f"    [{r.status.upper()}]  {r.nii_path.name}: {r.error_message}")

            all_manifest_rows.append(manifest_row)

    # Collect all column names seen across rows (wide format).
    all_keys: List[str] = ["subject_id", "timestamp", "n_exports_found", "n_no_dicom", "n_unclassified"]
    for cfg in SCAN_TYPE_CONFIGS:
        if cfg.name not in args.scan_types:
            continue
        label = cfg.acq_label
        if cfg.multi_phase:
            all_keys += [f"{label}_n_phases", f"{label}_n_slices", f"{label}_size_mb",
                         f"{label}_nifti", f"{label}_series_desc", f"{label}_status"]
        else:
            all_keys += [f"{label}_n_slices", f"{label}_size_mb", f"{label}_nifti",
                         f"{label}_series_desc", f"{label}_status"]
    all_keys.append("dicom_dir")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    qc_report_path = report_dir / f"conversion_report_QC_{run_ts}.html"
    _write_html_report(
        manifest_rows=all_manifest_rows,
        scan_type_configs=SCAN_TYPE_CONFIGS,
        allowed_types=args.scan_types,
        src_root=str(src_root),
        out_root=str(out_root),
        report_path=qc_report_path,
    )

    summary_report_path = report_dir / f"conversion_report_summary_{run_ts}.html"
    _write_html_report(
        manifest_rows=all_manifest_rows,
        scan_type_configs=SCAN_TYPE_CONFIGS,
        allowed_types=args.scan_types,
        src_root=str(src_root),
        out_root=str(out_root),
        report_path=summary_report_path,
        summary_mode=True,
    )

    print()
    print("---")
    print(f"Manifest     : {manifest_path}")
    print(f"Report QC    : {qc_report_path}")
    print(f"Report summ  : {summary_report_path}")
    print("Counts   :", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
