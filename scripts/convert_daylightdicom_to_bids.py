#!/usr/bin/env python3
"""Convert DAYLIGHT DICOM exports into CTA NIfTI files with strict QC gates.

Default behavior:
- Skip subjects that already have `sub-<id>_acq-CTA_ct.nii.gz`.
- Convert only likely CTA source series (not MIP/reformats/perfusion maps).
- Accept outputs only if compressed NIfTI size is within [300, 520] MB.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pydicom
import SimpleITK as sitk

# Common warning for odd but readable DICOM transfer syntax inconsistencies.
warnings.filterwarnings(
    "ignore",
    message="Expected implicit VR, but found explicit VR - using explicit VR for reading",
)

SKIP_DIRS = {"PLUGINS", "JRE", "HELP", "REPORT", "IHE_PDI", "XTR_CONT"}

POS_KEYWORDS = ("cta", "angi")
BAD_KEYWORDS = (
    "mip",
    "render",
    "summary",
    "rapid",
    "perfusion",
    "ctp",
    "sagittal",
    "coronal",
    "reformatted",
    "scout",
    "localizer",
    "topogram",
)


@dataclass
class SeriesCandidate:
    uid: str
    files: List[str]
    meta: dict
    score: int

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.meta.get("series_description", ""),
                self.meta.get("protocol_name", ""),
                self.meta.get("study_description", ""),
                self.meta.get("image_comments", ""),
            ]
        ).lower()


def safe_text(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def subject_id_from_name(name: str) -> str:
    if name.isdigit():
        return str(int(name))
    m = re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else name


def iter_candidate_files(subject_dir: Path) -> Iterable[Path]:
    for p in subject_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.upper() == "DICOMDIR":
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def collect_series(subject_dir: Path) -> Dict[str, SeriesCandidate]:
    files_by_uid: Dict[str, List[str]] = defaultdict(list)
    meta_by_uid: Dict[str, dict] = {}

    for f in iter_candidate_files(subject_dir):
        try:
            ds = pydicom.dcmread(
                str(f),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "Modality",
                    "SeriesInstanceUID",
                    "SeriesDescription",
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
                "protocol_name": safe_text(getattr(ds, "ProtocolName", None)),
                "study_description": safe_text(getattr(ds, "StudyDescription", None)),
                "image_comments": safe_text(getattr(ds, "ImageComments", None)),
                "slice_thickness": safe_text(getattr(ds, "SliceThickness", None)),
                "manufacturer": safe_text(getattr(ds, "Manufacturer", None)),
                "model": safe_text(getattr(ds, "ManufacturerModelName", None)),
                "body_part": safe_text(getattr(ds, "BodyPartExamined", None)),
                "patient_position": safe_text(getattr(ds, "PatientPosition", None)),
                "patient_sex": safe_text(getattr(ds, "PatientSex", None)),
                "patient_birth_date": safe_text(getattr(ds, "PatientBirthDate", None)),
            }

    out: Dict[str, SeriesCandidate] = {}
    for uid, files in files_by_uid.items():
        meta = meta_by_uid.get(uid, {})
        score = score_series(meta, len(files))
        out[uid] = SeriesCandidate(uid=uid, files=files, meta=meta, score=score)
    return out


def score_series(meta: dict, n_files: int) -> int:
    txt = " ".join(
        [
            meta.get("series_description", ""),
            meta.get("protocol_name", ""),
            meta.get("study_description", ""),
            meta.get("image_comments", ""),
        ]
    ).lower()
    s = 0
    if "cta" in txt:
        s += 100
    if "angi" in txt:
        s += 50
    if "stroke" in txt:
        s += 10
    if "head" in txt:
        s += 10
    if "neck" in txt:
        s += 10
    if "ce" in txt:
        s += 5
    s += min(n_files, 2000) // 20

    for bad in BAD_KEYWORDS:
        if bad in txt:
            s -= 80
    return s


def is_likely_cta_source(c: SeriesCandidate, min_files: int) -> bool:
    txt = c.text
    if not any(k in txt for k in POS_KEYWORDS):
        return False
    if any(k in txt for k in BAD_KEYWORDS):
        return False
    if c.n_files < min_files:
        return False
    return True


def convert_series(file_list: List[str], out_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(file_list)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path), useCompression=True)


def _ordered_dedup_file_list(file_list: List[str]) -> Tuple[List[str], dict]:
    """Deduplicate by SOPInstanceUID and sort slices by geometry.

    DAYLIGHT exports may contain duplicate slices across repeated exports for the
    same SeriesInstanceUID. We remove duplicates and enforce deterministic order
    to avoid malformed volumes.
    """
    records = []
    for fp in file_list:
        try:
            ds = pydicom.dcmread(
                str(fp),
                stop_before_pixels=True,
                force=True,
                specific_tags=["SOPInstanceUID", "ImagePositionPatient", "InstanceNumber"],
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

        records.append((sop, z, inst, str(fp)))

    # Fallback if metadata could not be parsed for ordering.
    if not records:
        uniq = sorted(set(file_list))
        return uniq, {
            "raw_rows": len(file_list),
            "used_rows": len(uniq),
            "dedup_removed": len(file_list) - len(uniq),
            "ordered_by": "path",
        }

    # Deduplicate by SOP UID when available; otherwise keep by path.
    dedup: Dict[str, tuple] = {}
    for sop, z, inst, fp in records:
        key = sop if sop else f"__PATH__::{fp}"
        prev = dedup.get(key)
        cur = (sop, z, inst, fp)
        if prev is None:
            dedup[key] = cur
            continue

        # Prefer rows with richer ordering metadata; tie-break by shorter path.
        prev_score = int(prev[1] is not None) + int(prev[2] is not None)
        cur_score = int(cur[1] is not None) + int(cur[2] is not None)
        if cur_score > prev_score or (cur_score == prev_score and len(cur[3]) < len(prev[3])):
            dedup[key] = cur

    uniq = list(dedup.values())
    n = len(uniq)
    n_with_z = sum(1 for _, z, _, _ in uniq if z is not None)
    n_with_inst = sum(1 for _, _, inst, _ in uniq if inst is not None)

    # Prefer physical z ordering if widely available, otherwise instance number.
    if n_with_z >= max(1, int(0.7 * n)):
        uniq.sort(
            key=lambda r: (
                r[1] if r[1] is not None else 1e18,
                r[2] if r[2] is not None else 1e18,
                r[3],
            )
        )
        ordered_by = "z"
    elif n_with_inst > 0:
        uniq.sort(key=lambda r: (r[2] if r[2] is not None else 1e18, r[3]))
        ordered_by = "instance"
    else:
        uniq.sort(key=lambda r: r[3])
        ordered_by = "path"

    ordered_files = [r[3] for r in uniq]
    return ordered_files, {
        "raw_rows": len(file_list),
        "used_rows": len(ordered_files),
        "dedup_removed": len(file_list) - len(ordered_files),
        "ordered_by": ordered_by,
    }


def output_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def convert_subject(
    sid: str,
    subject_dir: Path,
    out_root: Path,
    min_files: int,
    min_mb: float,
    max_mb: float,
) -> dict:
    out_nii = out_root / f"sub-{sid}_acq-CTA_ct.nii.gz"
    out_json = out_root / f"sub-{sid}_acq-CTA_ct.json"

    series_map = collect_series(subject_dir)
    if not series_map:
        return {
            "subject_id": sid,
            "status": "failure",
            "cta_count": 0,
            "output_mb": "",
            "chosen_uid": "",
            "error_message": "No CT DICOM series found",
        }

    candidates = sorted(series_map.values(), key=lambda x: (x.score, x.n_files), reverse=True)
    valid_candidates = [c for c in candidates if is_likely_cta_source(c, min_files=min_files)]
    if not valid_candidates:
        return {
            "subject_id": sid,
            "status": "success_no_cta",
            "cta_count": 0,
            "output_mb": "",
            "chosen_uid": "",
            "error_message": "No CTA source series passed filters",
        }

    last_error = ""
    for c in valid_candidates:
        tmp_out = out_root / f".tmp_sub-{sid}_acq-CTA_ct.nii.gz"
        if tmp_out.exists():
            tmp_out.unlink()
        try:
            ordered_files, order_info = _ordered_dedup_file_list(c.files)
            convert_series(ordered_files, tmp_out)
            mb = output_size_mb(tmp_out)
            if mb < min_mb or mb > max_mb:
                tmp_out.unlink(missing_ok=True)
                last_error = (
                    f"Rejected {c.uid} by size gate: {mb:.1f}MB "
                    f"(expected {min_mb:.0f}-{max_mb:.0f}MB)"
                )
                continue

            tmp_out.replace(out_nii)
            sidecar = {
                "Modality": "CT",
                "SeriesDescription": c.meta.get("series_description", ""),
                "ProtocolName": c.meta.get("protocol_name", ""),
                "StudyDescription": c.meta.get("study_description", ""),
                "ImageComments": c.meta.get("image_comments", ""),
                "BodyPartExamined": c.meta.get("body_part", ""),
                "PatientPosition": c.meta.get("patient_position", ""),
                "Manufacturer": c.meta.get("manufacturer", ""),
                "ManufacturerModelName": c.meta.get("model", ""),
                "PatientSex": c.meta.get("patient_sex", ""),
                "PatientBirthDate": c.meta.get("patient_birth_date", ""),
                "SeriesInstanceUID": c.uid,
                "SourceFileCountRaw": order_info["raw_rows"],
                "SourceFileCountUsed": order_info["used_rows"],
                "SourceFilesDedupRemoved": order_info["dedup_removed"],
                "SourceFileOrder": order_info["ordered_by"],
                "ConversionSoftware": "SimpleITK",
                "ConversionSoftwareVersion": sitk.Version_VersionString(),
                "OutputSizeMB": round(mb, 3),
            }
            out_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            return {
                "subject_id": sid,
                "status": "success",
                "cta_count": 1,
                "output_mb": f"{mb:.3f}",
                "chosen_uid": c.uid,
                "error_message": "",
            }
        except Exception as e:
            tmp_out.unlink(missing_ok=True)
            last_error = f"{type(e).__name__}: {e}"

    return {
        "subject_id": sid,
        "status": "failure",
        "cta_count": 0,
        "output_mb": "",
        "chosen_uid": "",
        "error_message": last_error or "All CTA candidates failed",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DAYLIGHT DICOM to CTA BIDS-style NIfTI")
    p.add_argument(
        "--src-root",
        default="./data/DAYLIGHTDICOM",
        help="Source root containing subject folders",
    )
    p.add_argument(
        "--out-root",
        default="./data/daylightbids",
        help="Destination root for sub-*_acq-CTA_ct.nii.gz",
    )
    p.add_argument("--min-files", type=int, default=300, help="Minimum DICOM count for CTA source series")
    p.add_argument("--min-mb", type=float, default=300.0, help="Minimum accepted output size (MB)")
    p.add_argument("--max-mb", type=float, default=520.0, help="Maximum accepted output size (MB)")
    p.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Optional subject id(s) to process; repeatable, e.g. --subject 631",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    existing = set()
    for p in out_root.glob("sub-*_acq-CTA_ct.nii.gz"):
        sid = p.name.split("_")[0].replace("sub-", "")
        if sid.isdigit():
            existing.add(str(int(sid)))

    subject_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    wanted = {str(int(s)) for s in args.subject if str(s).isdigit()}

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_root / f"conversion_log_{run_ts}.csv"
    rows = []
    counts = Counter()

    for subj in subject_dirs:
        sid = subject_id_from_name(subj.name)
        if not sid.isdigit():
            continue
        if wanted and sid not in wanted:
            continue

        out_nii = out_root / f"sub-{sid}_acq-CTA_ct.nii.gz"
        if sid in existing or out_nii.exists():
            rows.append(
                {
                    "subject_id": sid,
                    "status": "skip_exists",
                    "cta_count": 1,
                    "output_mb": "",
                    "chosen_uid": "",
                    "error_message": "",
                }
            )
            counts["skip_exists"] += 1
            continue

        try:
            row = convert_subject(
                sid=sid,
                subject_dir=subj,
                out_root=out_root,
                min_files=args.min_files,
                min_mb=args.min_mb,
                max_mb=args.max_mb,
            )
        except Exception as e:
            row = {
                "subject_id": sid,
                "status": "failure",
                "cta_count": 0,
                "output_mb": "",
                "chosen_uid": "",
                "error_message": f"{type(e).__name__}: {e}",
            }
            traceback.print_exc()

        rows.append(row)
        counts[row["status"]] += 1
        if row["status"] == "success":
            print(f"[OK] sub-{sid} -> {row['output_mb']} MB")
        elif row["status"] == "skip_exists":
            pass
        else:
            print(f"[{row['status'].upper()}] sub-{sid}: {row['error_message']}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "status",
                "cta_count",
                "output_mb",
                "chosen_uid",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("---")
    print(f"Log: {log_path}")
    print("Counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
