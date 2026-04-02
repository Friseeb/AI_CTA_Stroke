"""
Diagnose 5 unclassified subjects: sub-4, sub-66, sub-118, sub-194, sub-226.

For each export prints:
  - All SeriesDescriptions (with slice count)
  - z-spread
  - Per-type keyword scoring and coverage gate result
  - Classification outcome

Writes CSV → C:/Users/spost/Desktop/CT_image/REJECTED/unclassified_debug.csv
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_daylightdicom_to_bids import (
    collect_series, _sample_z_range, find_export_dirs,
    SCAN_TYPE_CONFIGS, SCAN_TYPE_MAP,
)

SUBJECTS = ["4", "66", "118", "194", "226"]
SRC_ROOT  = Path("D:/")
CSV_PATH  = Path(r"C:/Users/spost/Desktop/CT_image/REJECTED/unclassified_debug.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

ALLOWED = [c.name for c in SCAN_TYPE_CONFIGS]

csv_rows = []

for sid in SUBJECTS:
    subject_dir = SRC_ROOT / sid
    print(f"\n{'='*70}")
    print(f"sub-{sid}  ({subject_dir})")

    export_dirs = find_export_dirs(subject_dir)
    if not export_dirs:
        print("  !! No Export_ dirs found")
        csv_rows.append({
            "subject_id": f"sub-{sid}", "export_folder": "NONE",
            "series_description": "", "slice_count": "",
            "study_description": "", "protocol_name": "", "body_part": "",
            "z_coverage_mm": "", "classification": "no_export_dirs",
            "scores": "", "note": "no Export_ dirs",
        })
        continue

    for export_dir in export_dirs:
        print(f"\n  Export: {export_dir.name}")
        series_map = collect_series(export_dir)
        print(f"  {len(series_map)} series found")

        # --- Aggregate metadata (mirrors classify_export_study) ---
        study_descs = " ".join(
            c.meta.get("study_description", "") for c in series_map.values()
        ).lower()
        proto_names = " ".join(
            c.meta.get("protocol_name", "") for c in series_map.values()
        ).lower()
        body_parts = " ".join(
            c.meta.get("body_part", "") for c in series_map.values()
        ).lower()

        # Pull first non-empty values for CSV
        first_study_desc = next(
            (c.meta.get("study_description","") for c in series_map.values()
             if c.meta.get("study_description","")), "")
        first_proto = next(
            (c.meta.get("protocol_name","") for c in series_map.values()
             if c.meta.get("protocol_name","")), "")
        first_body = next(
            (c.meta.get("body_part","") for c in series_map.values()
             if c.meta.get("body_part","")), "")

        coverage_mm = _sample_z_range(series_map)
        cov_str = f"{coverage_mm:.1f}" if coverage_mm is not None else "unknown"
        print(f"  z_coverage = {cov_str} mm")

        # --- Series list ---
        print(f"\n  Series (sorted by series_number):")
        for c in sorted(series_map.values(), key=lambda x: x.series_number):
            sd = c.meta.get("series_description","").strip()
            pn = c.meta.get("protocol_name","").strip()
            print(f"    n={c.n_files:>4}  desc={sd!r}  proto={pn!r}")

        # --- Per-type scoring ---
        print(f"\n  Scoring breakdown:")
        scores = {}
        type_notes = {}
        for type_name in ALLOWED:
            cfg = SCAN_TYPE_MAP[type_name]
            note_parts = []

            # Coverage gate
            if coverage_mm is not None:
                if cfg.z_coverage_min_mm is not None and coverage_mm < cfg.z_coverage_min_mm:
                    scores[type_name] = 0
                    note_parts.append(f"z={cov_str} < z_min={cfg.z_coverage_min_mm:.0f}")
                    type_notes[type_name] = "; ".join(note_parts)
                    print(f"    {type_name:15s}  score=0  GATE_FAIL: {note_parts[-1]}")
                    continue
                if cfg.z_coverage_max_mm is not None and coverage_mm > cfg.z_coverage_max_mm:
                    scores[type_name] = 0
                    note_parts.append(f"z={cov_str} > z_max={cfg.z_coverage_max_mm:.0f}")
                    type_notes[type_name] = "; ".join(note_parts)
                    print(f"    {type_name:15s}  score=0  GATE_FAIL: {note_parts[-1]}")
                    continue

            # Study-desc veto
            if any(kw in study_descs for kw in cfg.exclude_study_desc_keywords):
                matched = [kw for kw in cfg.exclude_study_desc_keywords if kw in study_descs]
                scores[type_name] = 0
                note_parts.append(f"study_desc_veto={matched}")
                type_notes[type_name] = "; ".join(note_parts)
                print(f"    {type_name:15s}  score=0  VETO: {matched}")
                continue

            s = 0
            matched_desc = [kw for kw in cfg.study_desc_keywords if kw in study_descs]
            matched_proto = [kw for kw in cfg.protocol_keywords if kw in proto_names]
            matched_body  = [bp for bp in cfg.study_body_parts  if bp in body_parts]

            if matched_desc:  s += 150
            if matched_proto: s += 80
            if matched_body:  s += 50

            scores[type_name] = s

            kw_detail = (
                f"desc_match={matched_desc or 'none'}  "
                f"proto_match={matched_proto or 'none'}  "
                f"body_match={matched_body or 'none'}"
            )
            gate_str = "NO_KW_SIGNAL" if s < 50 else "OK"
            note_parts.append(kw_detail)
            type_notes[type_name] = "; ".join(note_parts)
            print(f"    {type_name:15s}  score={s:>3}  {gate_str}  {kw_detail}")

        best_type = max(scores, key=lambda t: scores[t]) if scores else None
        best_score = scores.get(best_type, 0) if best_type else 0
        classification = best_type if best_score >= 50 else "UNCLASSIFIED"
        print(f"\n  => Classification: {classification}  (best_score={best_score})")
        print(f"     StudyDescription aggregate: {study_descs[:120]!r}")
        print(f"     ProtocolName    aggregate:  {proto_names[:120]!r}")
        print(f"     BodyPart        aggregate:  {body_parts[:80]!r}")

        # --- CSV rows: one per series ---
        for c in sorted(series_map.values(), key=lambda x: x.series_number):
            csv_rows.append({
                "subject_id":        f"sub-{sid}",
                "export_folder":     export_dir.name,
                "series_description": c.meta.get("series_description","").strip(),
                "slice_count":       c.n_files,
                "study_description": first_study_desc,
                "protocol_name":     first_proto,
                "body_part":         first_body,
                "z_coverage_mm":     cov_str,
                "classification":    classification,
                "scores":            str({t: s for t, s in scores.items()}),
                "note":              "; ".join(
                    f"{t}:{type_notes[t]}" for t in ALLOWED if type_notes.get(t)
                ),
            })

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
fieldnames = [
    "subject_id", "export_folder", "series_description", "slice_count",
    "study_description", "protocol_name", "body_part",
    "z_coverage_mm", "classification", "scores", "note",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(csv_rows)

print(f"\nCSV written → {CSV_PATH}  ({len(csv_rows)} rows)")
