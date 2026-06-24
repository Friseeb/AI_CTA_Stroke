#!/usr/bin/env python
"""Scan CTA cases for metal/beam-hardening artifact burden (within the aorta).

Uses cta_common.artifacts to detect metal+bloom+streak, scores burden against the
cleaned aorta mask, and writes a per-case CSV with a flag (none/low/moderate/high).
This is the 'flag + exclude' deliverable — no HU is fabricated. Optionally writes
per-case artifact-mask NIfTIs that downstream HU features can subtract.

Examples:
  # specific cases
  python scripts/scan_artifact_burden.py --cases sub-529 sub-86 --out outputs/artifact_burden
  # whole batch
  python scripts/scan_artifact_burden.py --batch-dir outputs/aorta_batch_run/cases \
      --out outputs/artifact_burden --save-masks
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cta_common" / "src"))

from cta_common.artifacts import artifact_burden, artifact_masks, classify_burden  # noqa: E402
from cta_common.geometry import compute_spacing_from_sitk  # noqa: E402
from cta_common.io import read_volume, write_mask_like  # noqa: E402

BATCH = REPO / "aorta_cta_radiomics" / "outputs" / "aorta_batch_run" / "cases"


def resolve_cta(case: str) -> Path | None:
    candidates = [REPO / "data" / f"{case}_acq-CTA_ct.nii.gz"]
    slaobids = os.environ.get("SLAOBIDS_DIR")
    if slaobids:
        candidates.append(Path(slaobids) / f"{case}_acq-CTA_ct.nii.gz")
    for p in candidates:
        if p.exists():
            return p
    return None


def aorta_mask_path(case: str) -> Path:
    return BATCH / case / "masks" / case / f"{case}_aorta_mask_cleaned.nii.gz"


FIELDS = ["case_id", "flag", "has_metal", "metal_ml", "core_ml", "artifact_ml",
          "n_metal_components", "roi_artifact_fraction", "roi_core_fraction", "status"]


def scan_case(case: str, outdir: Path, save_masks: bool) -> dict:
    cta = resolve_cta(case)
    if cta is None:
        return {"case_id": case, "status": "no_cta"}
    vol = read_volume(cta)
    spacing = compute_spacing_from_sitk(vol.image)  # array order (z,y,x)
    masks = artifact_masks(vol.array.astype("float32"), spacing)

    roi = None
    amp = aorta_mask_path(case)
    if amp.exists():
        am = read_volume(amp).array
        if am.shape == vol.array.shape:
            roi = am > 0
    burden = artifact_burden(masks, spacing, roi_mask=roi)
    flag = classify_burden(burden)

    if save_masks and masks.metal.any():
        cdir = outdir / case
        cdir.mkdir(parents=True, exist_ok=True)
        write_mask_like(masks.artifact, vol.image, cdir / f"{case}_artifact_mask.nii.gz")

    row = {"case_id": case, "flag": flag, "status": "ok"}
    row.update({k: round(v, 4) if isinstance(v, float) else v
                for k, v in burden.items() if k in FIELDS})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", nargs="+")
    src.add_argument("--batch-dir", help="dir of sub-* case folders to scan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    if args.cases:
        cases = args.cases
    else:
        cases = sorted(p.name for p in Path(args.batch_dir).glob("sub-*") if p.is_dir())
    if args.limit:
        cases = cases[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "artifact_burden.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, case in enumerate(cases, 1):
            try:
                row = scan_case(case, outdir, args.save_masks)
            except Exception as exc:
                row = {"case_id": case, "status": f"error: {exc}"}
            w.writerow({k: row.get(k, "") for k in FIELDS})
            fh.flush()
            print(f"[{i}/{len(cases)}] {case}: {row.get('flag', row.get('status'))}"
                  f" (artifact_ml={row.get('artifact_ml', '')})", flush=True)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
