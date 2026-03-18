"""
Debug + forced NIfTI conversion for the 8 subjects that got
'no source series found after filtering' in the batch run.

Outputs
-------
  CSV  : C:\\Users\\spost\\Desktop\\CT_image\\REJECTED\\debug_series_filter.csv
  NIfTI: C:\\Users\\spost\\Desktop\\CT_image\\REJECTED\\sub-<id>\\
         sub-<id>_s<NNN>_<series_desc>.nii.gz   (any series >= 10 slices)
"""
import sys, csv, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import (
    collect_series, classify_export_study, SCAN_TYPE_CONFIGS,
    SERIES_BAD_KEYWORDS, MIN_SOURCE_SLICES,
    _ordered_dedup_file_list, _convert_to_nifti,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]
OUT_ROOT  = Path(r"C:\Users\spost\Desktop\CT_image\REJECTED")
CSV_PATH  = OUT_ROOT / "debug_series_filter.csv"
MIN_SLICES_FOR_CONVERSION = 10   # attempt NIfTI for anything >= 10 slices

# The 8 error subjects: (subject_id, error_export_folder)
ERROR_SUBJECTS = [
    ("122", "Export_2026-03-07_18-07-07_1"),   # CardiacCT, 90  files
    ("125", "Export_2026-03-03_11-57-12_1"),   # CardiacCT, 157 files
    ("240", "Export_2026-03-09_14-09-19_1"),   # CardiacCT, 149 files
    ("148", "Export_2026-03-03_10-12-56_1"),   # ThoraxCT,  1053 files
    ("130", "Export_2026-03-04_13-24-18_1"),   # ThoraxCT,  1116 files
    ("220", "Export_2026-03-09_13-50-49_1"),   # ThoraxCT,  928  files
    ("250", "Export_2026-03-09_14-11-44_1"),   # ThoraxCT,  2123 files
    ("222", "Export_2026-03-04_15-47-40_1"),   # CTA,       1938 files
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_stem(txt: str, maxlen: int = 40) -> str:
    """Turn a series description into a safe filename fragment."""
    txt = re.sub(r"[^\w\s\-]", "", txt).strip()
    txt = re.sub(r"\s+", "_", txt)
    return txt[:maxlen] or "unknown"


def check_series(c) -> tuple[str, str]:
    """Return (rejected_by, matched_keyword) for one SeriesCandidate."""
    txt = " ".join([
        c.meta.get("series_description", ""),
        c.meta.get("protocol_name", ""),
    ]).lower()
    for bad in SERIES_BAD_KEYWORDS:
        if bad in txt:
            return "SERIES_BAD_KEYWORDS", bad
    if c.n_files < MIN_SOURCE_SLICES:
        return "MIN_SOURCE_SLICES", f"{c.n_files} < {MIN_SOURCE_SLICES}"
    return "passed", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
OUT_ROOT.mkdir(parents=True, exist_ok=True)

csv_rows = []
conv_summary = []   # (sid, series_desc, status, note)

for sid, export_name in ERROR_SUBJECTS:
    export_dir = Path(f"D:/{sid}/{export_name}")
    print(f"\n{'='*70}")
    print(f"sub-{sid}  /  {export_name}")

    if not export_dir.exists():
        print(f"  !! DICOM dir not found: {export_dir}")
        continue

    series_map = collect_series(export_dir)
    print(f"  {len(series_map)} series collected")

    sub_out = OUT_ROOT / f"sub-{sid}"
    sub_out.mkdir(exist_ok=True)

    for c in sorted(series_map.values(), key=lambda x: x.series_number):
        series_desc = c.meta.get("series_description", "").strip()
        rejected_by, matched_kw = check_series(c)

        csv_rows.append({
            "subject_id":         f"sub-{sid}",
            "export_folder":      export_name,
            "series_description": series_desc,
            "slice_count":        c.n_files,
            "rejected_by":        rejected_by,
            "matched_keyword":    matched_kw,
        })

        status_tag = f"[{'PASS' if rejected_by == 'passed' else 'REJECT'}]"
        print(f"  {status_tag} n={c.n_files:>4}  {series_desc!r}  "
              f"reason={rejected_by} {matched_kw}")

        # Attempt NIfTI conversion for any series >= MIN_SLICES_FOR_CONVERSION
        if c.n_files < MIN_SLICES_FOR_CONVERSION:
            conv_summary.append((sid, series_desc, "skipped", f"only {c.n_files} slices"))
            continue

        stem = f"sub-{sid}_s{c.series_number:03d}_{safe_stem(series_desc)}"
        out_path = sub_out / f"{stem}.nii.gz"

        if out_path.exists():
            print(f"    -> already exists, skipping")
            conv_summary.append((sid, series_desc, "skip_exists", str(out_path.name)))
            continue

        tmp = sub_out / f".tmp_{stem}.nii.gz"
        try:
            ordered, info = _ordered_dedup_file_list(c.files)
            _convert_to_nifti(ordered, tmp)
            tmp.replace(out_path)
            mb = out_path.stat().st_size / 1024 / 1024
            print(f"    -> converted  {out_path.name}  ({mb:.1f} MB, {info['used_rows']} slices)")
            conv_summary.append((sid, series_desc, "converted", f"{mb:.1f} MB"))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"    -> FAILED: {e}")
            conv_summary.append((sid, series_desc, "error", str(e)[:80]))

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
fieldnames = ["subject_id", "export_folder", "series_description",
              "slice_count", "rejected_by", "matched_keyword"]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(csv_rows)
print(f"\nCSV written → {CSV_PATH}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
from collections import Counter
status_counts = Counter(s for _, _, s, _ in conv_summary)
print(f"\n{'='*70}")
print("CONVERSION SUMMARY")
print(f"  converted   : {status_counts.get('converted', 0)}")
print(f"  skip_exists : {status_counts.get('skip_exists', 0)}")
print(f"  skipped     : {status_counts.get('skipped', 0)}  (< {MIN_SLICES_FOR_CONVERSION} slices)")
print(f"  error       : {status_counts.get('error', 0)}")
print(f"\nCSV rows     : {len(csv_rows)}")
print(f"Output folder: {OUT_ROOT}")
