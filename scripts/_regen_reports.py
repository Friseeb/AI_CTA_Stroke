"""Regenerate QC and summary HTML reports from the last batch manifest."""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import _write_html_report, SCAN_TYPE_CONFIGS

MANIFEST  = Path(r"C:/Users/spost/Desktop/CT_image/SLAAOBIDS/conversion_manifest_20260318_145741.csv")
OUT_ROOT  = MANIFEST.parent
TS        = "20260318_145741"
SRC_ROOT  = "D:/"

with MANIFEST.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} manifest rows")
allowed = [c.name for c in SCAN_TYPE_CONFIGS]

qc_path = OUT_ROOT / f"conversion_report_QC_{TS}.html"
_write_html_report(rows, SCAN_TYPE_CONFIGS, allowed, SRC_ROOT, str(OUT_ROOT), qc_path)

summary_path = OUT_ROOT / f"conversion_report_summary_{TS}.html"
_write_html_report(rows, SCAN_TYPE_CONFIGS, allowed, SRC_ROOT, str(OUT_ROOT), summary_path, summary_mode=True)
