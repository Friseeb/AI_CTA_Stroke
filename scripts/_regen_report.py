import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import _write_html_report, SCAN_TYPE_CONFIGS

csv_path = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS\conversion_manifest_20260317_123031.csv")
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
out = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
report = out / "conversion_report_updated.html"
_write_html_report(
    manifest_rows=rows,
    scan_type_configs=SCAN_TYPE_CONFIGS,
    allowed_types=[c.name for c in SCAN_TYPE_CONFIGS],
    src_root="D:/",
    out_root=str(out),
    report_path=report,
)
print("Done:", report)
