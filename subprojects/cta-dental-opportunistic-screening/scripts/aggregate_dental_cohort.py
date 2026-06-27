#!/usr/bin/env python
"""Aggregate per-case dental outputs into one cohort CSV + summary.

Reads every <outdir>/<case>/report.json and candidate_features.json and writes a
flat per-case table plus a printed cohort summary. Implant/crown/bridge candidates
are counted only above --min-candidate-mm3 (TotalSegmentator writes an empty label
file per class, which would otherwise inflate prevalence to ~100%), so this gives
corrected numbers even on outputs produced before that fix.

Note: as of the crown HU calibration, crowns/bridges are gated at the detector
(features.py) by both volume (crown_min_volume_mm3) AND a supra-enamel median HU
(crown_min_median_hu ~3000) to separate metal/ceramic restorations from dense
natural enamel. On regenerated candidate_features.json the --min-candidate-mm3
gate here is therefore only a backstop for older outputs.

Example:
  python scripts/aggregate_dental_cohort.py --root outputs/dental_slaobids \
      --out outputs/dental_cohort_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

VOL_MARKERS = ("implants_candidate", "crowns_or_bridges_candidate")
COUNT_MARKERS = ("teeth_present", "teeth_missing_candidate", "root_remnant_candidate",
                 "periapical_lucency_candidate", "severe_periodontal_bone_loss_candidate")


def _real_entries(entries, min_vol=None):
    """Count meaningful marker entries (drop status placeholders; apply vol gate)."""
    if not isinstance(entries, list):
        return 0
    n = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("status"):                      # e.g. {"status": "not_assessable"}
            continue
        if min_vol is not None and float(e.get("volume_mm3") or 0) < min_vol:
            continue
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Batch output dir (contains <case>/ subdirs).")
    ap.add_argument("--out", required=True, help="Cohort CSV path.")
    ap.add_argument("--min-candidate-mm3", type=float, default=20.0)
    args = ap.parse_args()

    root = Path(args.root)
    fields = ["case_id", "status", "roi_quality", "dentition_fov", "segmentation_quality",
              "n_teeth_present", "n_teeth_missing", "n_implants", "n_crowns_bridges",
              "n_root_remnant", "n_periapical_lucency", "n_periodontal_bone_loss"]
    rows, summ = [], Counter()
    cases = 0

    for case_dir in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        rep = case_dir / "report.json"
        if not rep.is_file():
            continue
        cases += 1
        try:
            d = json.loads(rep.read_text())
        except Exception:
            continue
        a = d.get("assessability", {}) or {}
        feat = {}
        fp = case_dir / "candidate_features.json"
        if fp.is_file():
            try:
                feat = (json.loads(fp.read_text()).get("candidate_markers") or {})
            except Exception:
                feat = {}
        row = {
            "case_id": case_dir.name,
            "status": d.get("status"),
            "roi_quality": d.get("roi_quality"),
            "dentition_fov": a.get("dentition_fov"),
            "segmentation_quality": a.get("segmentation_quality"),
            "n_teeth_present": _real_entries(feat.get("teeth_present")),
            "n_teeth_missing": _real_entries(feat.get("teeth_missing_candidate")),
            "n_implants": _real_entries(feat.get("implants_candidate"), args.min_candidate_mm3),
            "n_crowns_bridges": _real_entries(feat.get("crowns_or_bridges_candidate"), args.min_candidate_mm3),
            "n_root_remnant": _real_entries(feat.get("root_remnant_candidate")),
            "n_periapical_lucency": _real_entries(feat.get("periapical_lucency_candidate")),
            "n_periodontal_bone_loss": _real_entries(feat.get("severe_periodontal_bone_loss_candidate")),
        }
        rows.append(row)
        summ[f"roi_{row['roi_quality']}"] += 1
        for m in ("n_implants", "n_crowns_bridges", "n_periapical_lucency", "n_periodontal_bone_loss"):
            if row[m] > 0:
                summ[f"cases_with_{m}"] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    good = summ.get("roi_good", 0)
    print(f"cohort: {cases} cases | good ROI: {good} | failed_roi: {summ.get('roi_failed', 0)}")
    print("--- prevalence among good-ROI cases (candidate markers, experimental) ---")
    for m in ("n_implants", "n_crowns_bridges", "n_periapical_lucency", "n_periodontal_bone_loss"):
        c = summ.get(f"cases_with_{m}", 0)
        pct = f"{c/good*100:.0f}%" if good else "-"
        print(f"  {m.replace('n_',''):24s}: {c} cases ({pct})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
