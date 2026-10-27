#!/usr/bin/env python3
"""Summarise the confirmed-CCE cohort run against the report-derived reference sites.

  python evaluate_cce_cohort.py --root outputs/calcified_emboli_cohort [--out summary.csv]

Per case and modality: number of distal / proximal candidates, candidates on the reported side, sizes / HU of the
top candidates, and the class counts, so a reader can judge sensitivity case by case (no voxel-level truth exists).
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    ref = json.loads((a.root / "reference_sites.json").read_text())["cases"] if (a.root / "reference_sites.json").is_file() else {}
    rows = []
    for case_dir in sorted(p for p in a.root.iterdir() if p.is_dir() and p.name.startswith("sub-cce")):
        for mod in ("ncct", "cta"):
            csv = case_dir / mod / "search" / f"{case_dir.name}_intracranial_calcium.csv"
            if not csv.is_file():
                continue
            d = pd.read_csv(csv); r = ref.get(case_dir.name, {})
            cand = d[d.category.isin(["distal_candidate", "proximal_intraluminal"])].sort_values("max_hu", ascending=False)
            side = r.get("side", "?"); on_side = cand[cand.side.isin(list(side.replace("/", "")))] if side != "?" else cand
            top = "; ".join(f"{c.side} {c.equiv_diameter_mm:.1f}mm {c.max_hu:.0f}HU T{c.tier}" for c in cand.head(4).itertuples())
            rows.append({"case_id": case_dir.name, "modality": mod, "reported_side": side, "reported_sites": " | ".join(r.get("sites", [])),
                         "foci_total": len(d), "distal_candidates": int((d.category == "distal_candidate").sum()), "proximal": int((d.category == "proximal_intraluminal").sum()),
                         "candidates_on_reported_side": len(on_side), "tier1_candidates": int((cand.tier == 1).sum()), "tier1_on_side": int((on_side.tier == 1).sum()), "top_candidates": top,
                         "skull_dural": int(d.category.str.startswith("skull").sum()), "choroid": int((d.category == "choroid_plexus").sum()),
                         "pineal": int((d.category == "pineal_habenula").sum()), "basal_ganglia": int((d.category == "basal_ganglia").sum()),
                         "symmetric": int((d.category == "symmetric_bilateral").sum()), "other": int((d.category == "parenchymal_other").sum()),
                         "low_conf": int((d.category == "distal_candidate_low").sum())})
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(df.drop(columns=["reported_sites"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
