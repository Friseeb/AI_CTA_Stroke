#!/usr/bin/env python3
"""
Extract reconstruction kernel (DICOM tag 0018,1210 ConvolutionKernel) from
original DICOM files for all SLAAOBIDS subjects.

For each subject, extracts the kernel per CT type and per phase (ph00, ph01 …
for multi-phase studies; phase="" for single-phase studies).

Output: SLAAOBIDS/derivatives/dicom_kernel.csv
Columns: subject_id, ct_type, phase, kernel

Usage:
    conda run -n cardiac-ct-explorer python scripts/extract_dicom_kernels.py
    conda run -n cardiac-ct-explorer python scripts/extract_dicom_kernels.py --subjects 45 46 47
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import pydicom
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SLAAOBIDS   = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
OUTPUT_CSV  = SLAAOBIDS / "derivatives" / "dicom_kernel.csv"
DICOM_ROOTS = [Path("D:/"), Path("E:/")]

# ── Classification keywords (study-level, lower-cased) ────────────────────────
# Order matters: earlier entries win if a study would score equally.
ACQ_RULES = [
    # (acq_label, multi_phase, must_contain_any, must_not_contain_any)
    ("ctbody",    True,  ("thorax/abdomen", "chest/abdomen", "thorax abdomen",
                          "chest abdomen", "whole body", "total body", "totalbody",
                          "dissection"), ()),
    ("ecta",      False, ("cta", "angio", "angiograph", "ecta",
                          "head/neck", "head neck", "tia", "stroke"), ()),
    ("ctheart",   True,  ("cardiac", "heart", "coronary", "ccta", "calcium",
                          "cac", "cardio", "ctheart", "aortic", "tavi"), ()),
    ("ctthorax",  False, ("thorax", "chest", "lung", "pulmonar", "thoracic",
                          "hrct", "hrtx", "ctthorax", "cardiac cap"),
                         ("abdomen", "pelvis")),
    ("ctabdomen", True,  ("abdomen", "abdomin", "liver", "pelvis", "abdopelv",
                          "portal", "ctabdomen"), ()),
]

# Series descriptors that disqualify a series as a real acquisition
BAD_SERIES_KW = (
    "mip", "render", "sagittal", "coronal", "scout", "localizer",
    "topogram", "summary", "reformatted", "medrad", "injection",
    "lcd", "bone", "stress", "perfusion", "rapid",
)

MIN_SLICES   = 50   # minimum DICOM files for a volumetric series
SKIP_DIRS    = {"PLUGINS", "JRE", "HELP", "REPORT", "IHE_PDI", "XTR_CONT"}
EXPORT_PREFIX = "Export_"


# ── DICOM helpers ─────────────────────────────────────────────────────────────

def _read_tags(path: Path, tags: list[str]) -> dict:
    """Read specific DICOM tags from a file; return {} on failure."""
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=tags,
        )
        return {t: str(getattr(ds, t, "")).strip() for t in tags}
    except Exception:
        return {}


def _iter_dicom_files(export_dir: Path):
    """Yield DICOM file paths from one Export_* folder."""
    for p in export_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.upper() == "DICOMDIR":
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


# ── Classification ────────────────────────────────────────────────────────────

def _classify_study(study_desc: str) -> str | None:
    """Map StudyDescription to an acq_label using ACQ_RULES; None if unclassified."""
    s = study_desc.lower()
    for acq_label, _, must_have, must_not in ACQ_RULES:
        if any(kw in s for kw in must_have):
            if not any(kw in s for kw in must_not):
                return acq_label
    return None


def _is_source_series(series_desc: str, protocol: str, n_files: int) -> bool:
    txt = f"{series_desc} {protocol}".lower()
    if any(bad in txt for bad in BAD_SERIES_KW):
        return False
    return n_files >= MIN_SLICES


# ── Per-Export processing ─────────────────────────────────────────────────────

def extract_kernels_from_export(export_dir: Path) -> list[dict]:
    """
    Scan one Export_* directory and return rows:
      [{ct_type, phase, kernel, series_desc, n_files}, ...]
    Returns [] if export cannot be classified or has no qualifying series.
    """
    # Group files by SeriesInstanceUID
    series_files: dict[str, list[Path]]       = defaultdict(list)
    series_meta:  dict[str, dict]             = {}

    for f in _iter_dicom_files(export_dir):
        tags = _read_tags(f, [
            "Modality", "SeriesInstanceUID", "StudyDescription",
            "SeriesDescription", "ProtocolName", "SeriesNumber",
            "ConvolutionKernel",
        ])
        if tags.get("Modality", "").upper() != "CT":
            continue
        uid = tags.get("SeriesInstanceUID", "")
        if not uid:
            continue
        series_files[uid].append(f)
        if uid not in series_meta:
            series_meta[uid] = tags

    if not series_files:
        return []

    # Classify study from any series' StudyDescription
    study_desc = next(
        (m.get("StudyDescription", "") for m in series_meta.values()
         if m.get("StudyDescription", "")),
        "",
    )
    acq_label = _classify_study(study_desc)
    if acq_label is None:
        return []

    multi_phase = next(
        (mp for lbl, mp, _, _ in ACQ_RULES if lbl == acq_label), False
    )

    # Filter to source series and sort by SeriesNumber
    source = []
    for uid, files in series_files.items():
        meta = series_meta[uid]
        if _is_source_series(
            meta.get("SeriesDescription", ""),
            meta.get("ProtocolName", ""),
            len(files),
        ):
            try:
                snum = int(meta.get("SeriesNumber", "99999"))
            except ValueError:
                snum = 99999
            source.append((snum, uid, files, meta))

    source.sort(key=lambda x: x[0])

    if not source:
        return []

    # For single-phase: keep only the largest (most files) series
    if not multi_phase:
        source = [max(source, key=lambda x: len(x[2]))]

    rows = []
    for ph_idx, (_, uid, files, meta) in enumerate(source):
        # Read ConvolutionKernel from the first available file
        kernel = meta.get("ConvolutionKernel", "")
        if not kernel:
            # Tag not in cached read — retry on the first file explicitly
            for f in sorted(files)[:5]:
                extra = _read_tags(f, ["ConvolutionKernel"])
                k = extra.get("ConvolutionKernel", "")
                if k:
                    kernel = k
                    break

        phase = f"ph{ph_idx:02d}" if multi_phase else ""
        rows.append({
            "ct_type":     acq_label,
            "phase":       phase,
            "kernel":      kernel,
            "series_desc": meta.get("SeriesDescription", ""),
            "n_files":     len(files),
        })

    return rows


# ── Subject processing ────────────────────────────────────────────────────────

def find_dicom_root(subject_id: str) -> Path | None:
    """Find the patient DICOM folder on D:\ or E:\."""
    for root in DICOM_ROOTS:
        candidate = root / subject_id
        if candidate.is_dir():
            return candidate
    return None


def process_subject(subject_id: str) -> list[dict]:
    """Return kernel rows for one subject (all CT types/phases)."""
    dicom_root = find_dicom_root(subject_id)
    if dicom_root is None:
        return []

    export_dirs = sorted(dicom_root.glob(f"{EXPORT_PREFIX}*"))
    rows = []
    for exp_dir in export_dirs:
        if not exp_dir.is_dir():
            continue
        for row in extract_kernels_from_export(exp_dir):
            rows.append({
                "subject_id": f"sub-{subject_id}",
                "ct_type":    row["ct_type"],
                "phase":      row["phase"],
                "kernel":     row["kernel"],
            })

    # Deduplicate: keep first occurrence of (ct_type, phase)
    seen: set[tuple] = set()
    deduped = []
    for r in rows:
        key = (r["ct_type"], r["phase"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract DICOM reconstruction kernels for all SLAAOBIDS subjects"
    )
    parser.add_argument(
        "--slaaobids", default=str(SLAAOBIDS),
        help="SLAAOBIDS root directory",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_CSV),
        help="Output CSV path",
    )
    parser.add_argument(
        "--subjects", nargs="+", metavar="N",
        help="Limit to specific subject IDs, e.g. --subjects 1 2 3",
    )
    parser.add_argument(
        "--workers", type=int,
        default=min(8, max(1, (os.cpu_count() or 4) // 2)),
        help="Parallel workers (default: min(8, cpu_count//2))",
    )
    args = parser.parse_args()

    slaaobids  = Path(args.slaaobids)
    output_csv = Path(args.output)

    # Collect subject IDs from SLAAOBIDS directory
    all_subs = sorted(
        [p.name.replace("sub-", "") for p in slaaobids.glob("sub-*") if p.is_dir()],
        key=lambda x: int(x) if x.isdigit() else float("inf"),
    )

    if args.subjects:
        wanted = {str(int(s)) for s in args.subjects}
        all_subs = [s for s in all_subs if s in wanted]

    print(f"Subjects to process : {len(all_subs)}")
    print(f"DICOM search roots  : {[str(r) for r in DICOM_ROOTS]}")
    print(f"Workers             : {args.workers}")
    print(f"Output CSV          : {output_csv}")
    print()

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    missing_dicom: list[str] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_map = {ex.submit(process_subject, sid): sid for sid in all_subs}
        with tqdm(total=len(all_subs), desc="Extracting kernels", unit="sub", dynamic_ncols=True) as pbar:
            for fut in concurrent.futures.as_completed(future_map):
                sid = future_map[fut]
                try:
                    rows = fut.result()
                except Exception as exc:
                    rows = []
                    tqdm.write(f"  ERROR sub-{sid}: {exc}")
                if rows:
                    all_rows.extend(rows)
                else:
                    missing_dicom.append(f"sub-{sid}")
                pbar.update(1)

    # Sort rows by subject_id numerically for clean output
    all_rows.sort(key=lambda r: (
        int(r["subject_id"].replace("sub-", ""))
        if r["subject_id"].replace("sub-", "").isdigit() else float("inf"),
        r["ct_type"], r["phase"],
    ))

    # Write CSV
    fieldnames = ["subject_id", "ct_type", "phase", "kernel"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows for {len(all_subs) - len(missing_dicom)} subjects")
    if missing_dicom:
        print(f"No DICOM source found for {len(missing_dicom)} subject(s):")
        for s in sorted(missing_dicom)[:20]:
            print(f"  {s}")
        if len(missing_dicom) > 20:
            print(f"  ... and {len(missing_dicom) - 20} more")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
