"""Debug sub-116 classification — why CT thorax was not detected."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pydicom

from convert_daylightdicom_to_bids import (
    collect_series, classify_export_study, _sample_z_range,
    SCAN_TYPE_CONFIGS, _is_source_series,
    MIN_SOURCE_SLICES, SERIES_BAD_KEYWORDS,
)

SUBJECT_DIR = Path(r'D:\116')
ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]

export_dirs = sorted(SUBJECT_DIR.glob('Export_*'))
print(f"Found {len(export_dirs)} export dirs\n")

for export_dir in export_dirs:
    print(f"{'='*70}")
    print(f"Export: {export_dir.name}")
    series_map = collect_series(export_dir)
    print(f"  {len(series_map)} series found")

    # Print study-level tags from first available file
    for uid, c in series_map.items():
        if c.files:
            try:
                ds = pydicom.dcmread(str(c.files[0]), stop_before_pixels=True)
                print(f"  StudyDescription : {getattr(ds, 'StudyDescription', 'N/A')!r}")
                print(f"  ProtocolName     : {getattr(ds, 'ProtocolName', 'N/A')!r}")
                print(f"  BodyPartExamined : {getattr(ds, 'BodyPartExamined', 'N/A')!r}")
            except Exception:
                pass
            break

    # z-coverage
    cov = _sample_z_range(series_map)
    print(f"  z_coverage = {cov:.1f} mm" if cov else "  z_coverage = unknown")

    # Source series filter
    print(f"\n  Source series (after bad-keyword + slice-count filter):")
    qualified = []
    for uid, c in series_map.items():
        desc = c.meta.get("series_description", "")
        n = c.n_files
        is_src = _is_source_series(c)
        flag = "v" if is_src else "x"
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

    # Classification scoring
    print(f"\n  Classification scores per scan type:")
    for cfg in SCAN_TYPE_CONFIGS:
        # Re-run scoring manually to show results
        pass

    type_name = classify_export_study(series_map, ALLOWED)
    print(f"  -> Classified as: {type_name}")

    # Show z-range vs each config's z_coverage bounds
    cov_str = f"{cov:.1f}" if cov else "unknown"
    print(f"\n  Z-coverage check per scan type (z_coverage={cov_str}):")
    for cfg in SCAN_TYPE_CONFIGS:
        if cov is not None:
            ok_z = cfg.z_coverage_min_mm <= cov <= cfg.z_coverage_max_mm
            print(f"    {cfg.name:15s}  z_min={cfg.z_coverage_min_mm:.0f}  z_max={cfg.z_coverage_max_mm:.0f}  "
                  f"{'PASS' if ok_z else 'FAIL (z out of range)'}")
        else:
            print(f"    {cfg.name:15s}  z unknown")
    print()
