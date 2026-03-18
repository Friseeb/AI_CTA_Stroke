"""Debug sub-14 classification and size_rejected diagnosis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from convert_daylightdicom_to_bids import (
    collect_series, classify_export_study, _sample_z_range,
    find_export_dirs, SCAN_TYPE_CONFIGS, _is_source_series,
    MIN_SOURCE_SLICES, SERIES_BAD_KEYWORDS,
)

BIDS_ROOT = Path(r'C:\Users\spost\Desktop\CT_image\SLAAOBIDS')
SUBJECT_DIR = Path(r'D:\14')
ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]

export_dirs = sorted(SUBJECT_DIR.glob('Export_*'))
print(f"Found {len(export_dirs)} export dirs\n")

for export_dir in export_dirs:
    print(f"{'='*70}")
    print(f"Export: {export_dir.name}")
    series_map = collect_series(export_dir)
    print(f"  {len(series_map)} series found")

    # z-coverage
    cov = _sample_z_range(series_map)
    print(f"  z_coverage = {cov:.1f} mm" if cov else "  z_coverage = unknown")

    # Source series that would qualify
    print(f"\n  Source series (after bad-keyword + slice-count filter):")
    qualified = []
    for uid, c in series_map.items():
        desc = c.meta.get("series_description", "")
        n = c.n_files
        is_src = _is_source_series(c)
        flag = "✓" if is_src else "✗"
        # show why filtered
        reason = ""
        if not is_src:
            txt = (desc + " " + c.meta.get("protocol_name", "")).lower()
            bad_found = [b for b in SERIES_BAD_KEYWORDS if b in txt]
            if bad_found:
                reason = f"  (bad_kw: {bad_found})"
            elif n < MIN_SOURCE_SLICES:
                reason = f"  (only {n} files < {MIN_SOURCE_SLICES})"
            else:
                reason = "  (other)"
        else:
            qualified.append(c)
        print(f"    {flag} {n:4d}f  {desc!r}{reason}")

    if qualified:
        winner = max(qualified, key=lambda c: c.n_files)
        print(f"\n  Winner for single-phase: {winner.n_files} files, "
              f"{winner.meta.get('series_description')!r}")

    # Classification
    print(f"\n  Classification:")
    type_name = classify_export_study(series_map, ALLOWED)
    print(f"  → Classified as: {type_name}")
