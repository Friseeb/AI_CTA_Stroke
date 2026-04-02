"""
anonymize_dicom_export.py

Anonymizes all DICOM files in a source folder (recursively) and writes
de-identified copies to an output folder, preserving the original directory
structure. UIDs are replaced consistently (same original UID → same new UID)
so the DICOM hierarchy (study/series/instance) remains valid.

Usage:
    python scripts/anonymize_dicom_export.py \
        --input  "E:/1401001_Export_2026-03-30_10-11-46_1" \
        --output "E:/1401001_Export_2026-03-30_10-11-46_1_anon" \
        [--limit N]
"""

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PHI tags to blank / remove
# ---------------------------------------------------------------------------
TAGS_TO_BLANK = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "PatientSize",
    "PatientAddress",
    "PatientMotherBirthName",
    "PatientTelephoneNumbers",
    "PatientComments",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PhysiciansOfRecord",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    "RequestedProcedureDescription",
    "PerformedProcedureStepDescription",
    "StudyDescription",
    "SeriesDescription",
    "AccessionNumber",
    "StudyID",
]

# UID tags that must be remapped consistently
UID_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "ReferencedSOPInstanceUID",
    "FrameOfReferenceUID",
    "SynchronizationFrameOfReferenceUID",
]

# Time tags — blanked to remove potential identifying info
TIME_TAGS = [
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
]


def deterministic_uid(original_uid: str) -> str:
    """Map an original UID to a new unique but deterministic UID."""
    h = hashlib.sha256(original_uid.encode()).hexdigest()
    # Build a valid DICOM UID from the hash (max 64 chars, digits + dots)
    numeric = str(int(h[:16], 16))
    return f"2.25.{numeric}"


def anonymize_dataset(ds: pydicom.Dataset, uid_map: dict) -> pydicom.Dataset:
    """Anonymize a pydicom Dataset in-place."""
    # --- blank PHI text tags ---
    for tag_name in TAGS_TO_BLANK:
        if hasattr(ds, tag_name):
            elem = ds[tag_name]
            # Replace with empty string or ANON depending on VR
            if elem.VR in ("LO", "LT", "PN", "SH", "ST", "UC", "UR", "UT", "CS"):
                elem.value = "ANON" if elem.VR == "CS" else ""
            else:
                del ds[tag_name]

    # --- remap UIDs consistently ---
    for tag_name in UID_TAGS:
        if hasattr(ds, tag_name):
            orig = str(getattr(ds, tag_name))
            if orig not in uid_map:
                uid_map[orig] = deterministic_uid(orig)
            ds[tag_name].value = uid_map[orig]

    # --- keep full study dates intact ---
    # (DATE_TAGS are left as-is; PatientBirthDate is blanked via TAGS_TO_BLANK)

    # --- blank times ---
    for tag_name in TIME_TAGS:
        if hasattr(ds, tag_name):
            ds[tag_name].value = "000000"

    # --- recurse into sequences ---
    for elem in ds:
        if elem.VR == "SQ":
            for item in elem.value:
                anonymize_dataset(item, uid_map)

    return ds


def collect_dicom_files(root: Path) -> list[Path]:
    """Return all files under root (no extension filter — DICOM often has none)."""
    return [p for p in root.rglob("*") if p.is_file()]


def main():
    parser = argparse.ArgumentParser(description="Anonymize DICOM export folder")
    parser.add_argument("--input",  required=True, help="Source DICOM folder")
    parser.add_argument("--output", required=True, help="Destination folder for anonymized files")
    parser.add_argument("--limit",  type=int, default=None, help="Process only first N files (for testing)")
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    dst_root.mkdir(parents=True, exist_ok=True)

    log_path = dst_root / "anonymization_log.csv"

    # --- collect files ---
    print(f"Scanning {src_root} ...")
    all_files = collect_dicom_files(src_root)
    if args.limit:
        all_files = all_files[: args.limit]
    print(f"Found {len(all_files)} files to process.")

    # --- load existing log for skip-if-done ---
    done_files: set[str] = set()
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "done":
                    done_files.add(row["source_file"])

    # Shared UID mapping across the whole export so UIDs stay consistent
    uid_map: dict[str, str] = {}

    log_rows = []

    with tqdm(all_files, desc="Anonymizing", unit="file", dynamic_ncols=True) as pbar:
        for src_file in pbar:
            rel = src_file.relative_to(src_root)
            dst_file = dst_root / rel
            src_str = str(src_file)

            # skip-if-done
            if src_str in done_files and dst_file.exists():
                log_rows.append({
                    "source_file": src_str,
                    "dest_file": str(dst_file),
                    "status": "skipped",
                    "timestamp": datetime.now().isoformat(),
                    "error": "",
                })
                pbar.set_postfix({"last": rel.name, "status": "skip"})
                continue

            dst_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                ds = pydicom.dcmread(str(src_file), force=True)
                anonymize_dataset(ds, uid_map)
                ds.save_as(str(dst_file), write_like_original=False)
                status = "done"
                error = ""
            except InvalidDicomError:
                # Not a DICOM file — copy as-is (e.g. DICOMDIR index files)
                shutil.copy2(src_file, dst_file)
                status = "copied_non_dicom"
                error = ""
            except Exception as exc:
                status = "failed"
                error = str(exc)

            log_rows.append({
                "source_file": src_str,
                "dest_file": str(dst_file),
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "error": error,
            })
            pbar.set_postfix({"last": rel.name, "status": status})

    # --- write log ---
    fieldnames = ["source_file", "dest_file", "status", "timestamp", "error"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # --- summary ---
    counts = {}
    for r in log_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nDone.")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\nLog saved to: {log_path}")
    print(f"Anonymized files saved to: {dst_root}")

    failed = [r for r in log_rows if r["status"] == "failed"]
    if failed:
        print(f"\nWARNING: {len(failed)} files failed. Check log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
