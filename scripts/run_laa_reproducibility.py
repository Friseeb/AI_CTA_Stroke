#!/usr/bin/env python3
"""Compute interobserver reproducibility for LAA annotations.

Reads a manifest of per-reader finalized LAA masks, groups them by case, and
computes pairwise Dice / Surface Dice / HD95 across readers. Writes per-case and
aggregate reports for the Phase-0 reproducibility study.

Manifest columns (CSV):
  case_id, reader_id, mask_path
  (case_id may be blank; it is inferred from mask_path)

Example:
  python scripts/run_laa_reproducibility.py \\
    --manifest derivatives/laa_pilot/repro_manifest.csv \\
    --out-dir derivatives/laa_pilot/metrics \\
    --tolerance-mm 1.0

Outputs in <out-dir>:
  interrater_report.json   <- per-case + aggregate metrics
  interrater_report.csv    <- one row per reader pair per case
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Make the slicer_module core importable without installing it.
CORE_DIR = Path(__file__).resolve().parent.parent / "subprojects" / "la_laa" / "slicer_module"
sys.path.insert(0, str(CORE_DIR))

from laa_annotation_core import (  # noqa: E402
    interrater_report,
    read_repro_manifest,
)

PAIR_CSV_FIELDS = (
    "case_id",
    "reader_a",
    "reader_b",
    "dice",
    "surface_dice",
    "hd95_mm",
)


def _load_mask(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms) if len(zooms) == 3 else (1.0, 1.0, 1.0)
    return data, spacing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV: case_id,reader_id,mask_path")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=1.0, help="Surface Dice tolerance (mm)")
    args = parser.parse_args(argv)

    rows = read_repro_manifest(args.manifest)

    # group mask paths by case
    cases: dict[str, dict[str, str]] = {}
    for row in rows:
        cases.setdefault(row["case_id"], {})[row["reader_id"]] = row["mask_path"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_case = []
    pair_rows = []
    for case_id, reader_paths in sorted(cases.items()):
        if len(reader_paths) < 2:
            print(f"[skip] {case_id}: only {len(reader_paths)} reader(s)", file=sys.stderr)
            continue
        masks = {}
        spacing = (1.0, 1.0, 1.0)
        for reader_id, path in reader_paths.items():
            data, spacing = _load_mask(path)
            masks[reader_id] = data
        report = interrater_report(masks, spacing=spacing, tolerance_mm=args.tolerance_mm)
        report["case_id"] = case_id
        per_case.append(report)
        for pair in report["pairs"]:
            pair_rows.append({
                "case_id": case_id,
                "reader_a": pair["reader_a"],
                "reader_b": pair["reader_b"],
                "dice": pair["dice"],
                "surface_dice": pair["surface_dice"],
                "hd95_mm": pair["hd95_mm"],
            })
        print(
            f"[ok] {case_id}: readers={report['readers']} "
            f"mean_dice={report['mean_dice']:.3f} mean_hd95={report['mean_hd95_mm']}"
        )

    def _agg(key: str):
        vals = [c[key] for c in per_case if c.get(key) is not None and np.isfinite(c[key])]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "n_cases": len(per_case),
        "tolerance_mm": args.tolerance_mm,
        "mean_dice": _agg("mean_dice"),
        "mean_surface_dice": _agg("mean_surface_dice"),
        "mean_hd95_mm": _agg("mean_hd95_mm"),
    }

    report_json = args.out_dir / "interrater_report.json"
    report_json.write_text(json.dumps({"aggregate": aggregate, "cases": per_case}, indent=2))

    report_csv = args.out_dir / "interrater_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PAIR_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"\nWrote {report_json}")
    print(f"Wrote {report_csv}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
