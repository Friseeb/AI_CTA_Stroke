"""Quick check that sub-238 still classifies as TotalBodyCT after the fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import (
    collect_series, classify_export_study, SCAN_TYPE_CONFIGS,
)
ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]
subject_dir = Path(r'D:\238')
for export_dir in sorted(subject_dir.glob('Export_*')):
    series_map = collect_series(export_dir)
    if not series_map:
        continue
    t = classify_export_study(series_map, ALLOWED)
    print(f"{export_dir.name} -> {t}")
