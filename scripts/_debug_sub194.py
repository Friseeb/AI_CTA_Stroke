"""Debug sub-194 classification with current code."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import (
    collect_series, _sample_z_range, find_export_dirs,
    classify_export_study, SCAN_TYPE_CONFIGS, SCAN_TYPE_MAP,
)

sid = "194"
subject_dir = Path("D:/194")
export_dirs = find_export_dirs(subject_dir)
print(f"Export dirs: {[e.name for e in export_dirs]}")

for export_dir in export_dirs:
    print(f"\n=== {export_dir.name} ===")
    series_map = collect_series(export_dir)
    study_descs = " ".join(c.meta.get("study_description","") for c in series_map.values()).lower()
    proto_names = " ".join(c.meta.get("protocol_name","") for c in series_map.values()).lower()
    body_parts  = " ".join(c.meta.get("body_part","") for c in series_map.values()).lower()
    print(f"study_descs: {study_descs[:200]!r}")
    print(f"proto_names: {proto_names[:200]!r}")
    print(f"body_parts:  {body_parts[:100]!r}")
    cov = _sample_z_range(series_map)
    print(f"coverage_mm: {cov}")
    print(f"'tavi' in study_descs: {'tavi' in study_descs}")
    cfg = SCAN_TYPE_MAP["CardiacCT"]
    print(f"CardiacCT study_desc_keywords: {cfg.study_desc_keywords}")
    print(f"CardiacCT z_min/z_max: {cfg.z_coverage_min_mm}/{cfg.z_coverage_max_mm}")

    ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]
    result = classify_export_study(series_map, ALLOWED)
    print(f"\nclassify_export_study -> {result}")

    # Manual scoring breakdown
    print("\nManual scoring:")
    for type_name in ALLOWED:
        cfg2 = SCAN_TYPE_MAP[type_name]
        if cov is not None:
            if cfg2.z_coverage_min_mm is not None and cov < cfg2.z_coverage_min_mm:
                print(f"  {type_name}: GATE_FAIL z={cov:.0f} < z_min={cfg2.z_coverage_min_mm:.0f}")
                continue
            skip_z_max = type_name == "CardiacCT" and "tavi" in study_descs
            if not skip_z_max and cfg2.z_coverage_max_mm is not None and cov > cfg2.z_coverage_max_mm:
                print(f"  {type_name}: GATE_FAIL z={cov:.0f} > z_max={cfg2.z_coverage_max_mm:.0f}")
                continue
        if any(kw in study_descs for kw in cfg2.exclude_study_desc_keywords):
            matched = [kw for kw in cfg2.exclude_study_desc_keywords if kw in study_descs]
            print(f"  {type_name}: VETO exclude_study_desc {matched}")
            continue
        s = 0
        md = [kw for kw in cfg2.study_desc_keywords if kw in study_descs]
        mp = [kw for kw in cfg2.protocol_keywords if kw in proto_names]
        mb = [bp for bp in cfg2.study_body_parts if bp in body_parts]
        if md: s += 150
        if mp: s += 80
        if mb: s += 50
        print(f"  {type_name}: score={s}  desc={md}  proto={mp}  body={mb}")
