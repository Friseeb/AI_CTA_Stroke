#!/usr/bin/env python
"""Batch-run the dental opportunistic-screening pipeline over many CTAs.

Wraps the single-case ``cta-dental run`` CLI: for each input NIfTI it runs the
full pipeline into ``<outdir>/<case_id>/`` and records per-case status + wall
time to ``<outdir>/dental_batch_status.csv``. Resumable via --skip-existing
(forwarded to the per-case run, which reuses a completed segmentation).

Example:
  python scripts/run_dental_batch.py \
      --glob '/Volumes/DICOM5/slaobids/*_acq-CTA_ct.nii.gz' \
      --outdir outputs/dental_slaobids_pilot \
      --config configs/pilot_mps.yaml \
      --limit 3 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import re
import subprocess
import sys
import time
from pathlib import Path

# Default to the dental CLI in the totalseg-mac env (has TotalSegmentator).
DEFAULT_CTA_DENTAL = "/opt/anaconda3/envs/totalseg-mac/bin/cta-dental"


def case_id_from_path(p: Path) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+)", p.name)
    return m.group(1) if m else p.name.split(".")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--glob", help="Glob of input NIfTI CTAs.")
    src.add_argument("--manifest", help="Text file with one input path per line.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default=None, help="Pipeline config yaml (e.g. device: mps).")
    ap.add_argument("--segmenter", default="totalseg_teeth")
    ap.add_argument("--roi-method", default="totalseg_teeth")
    ap.add_argument("--deface-mode", default="mask_only")
    ap.add_argument("--target-spacing", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N inputs.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--reuse-roi-seg", action="store_true",
                    help="Reuse ROI-detection labels as the final segmentation (~2x faster).")
    ap.add_argument("--cta-dental", default=DEFAULT_CTA_DENTAL, help="Path to the cta-dental CLI.")
    args = ap.parse_args()

    if args.glob:
        inputs = [Path(p) for p in sorted(globmod.glob(args.glob))]
    else:
        inputs = [Path(line.strip()) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit("No inputs matched.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_csv = outdir / "dental_batch_status.csv"
    write_header = not status_csv.exists()

    print(f"Dental batch: {len(inputs)} case(s) -> {outdir}")
    with status_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["case_id", "input", "status", "returncode", "seconds", "out_dir"])

        for i, cta in enumerate(inputs, 1):
            case_id = case_id_from_path(cta)
            case_out = outdir / case_id
            if not cta.exists():
                print(f"[{i}/{len(inputs)}] {case_id}: MISSING input {cta}")
                writer.writerow([case_id, str(cta), "missing_input", "", "", str(case_out)])
                fh.flush()
                continue

            cmd = [
                args.cta_dental, "run", str(cta),
                "--out", str(case_out),
                "--case-id", case_id,
                "--segmenter", args.segmenter,
                "--roi-method", args.roi_method,
                "--deface-mode", args.deface_mode,
                "--target-spacing", str(args.target_spacing),
            ]
            if args.config:
                cmd += ["--config", args.config]
            if args.skip_existing:
                cmd += ["--skip-existing"]
            if args.reuse_roi_seg:
                cmd += ["--reuse-roi-seg"]

            print(f"[{i}/{len(inputs)}] {case_id}: running …", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            status = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
                print(f"    FAILED (rc={proc.returncode}, {dt:.0f}s)\n{tail}")
            else:
                print(f"    {status} in {dt:.0f}s")
            writer.writerow([case_id, str(cta), status, proc.returncode, f"{dt:.1f}", str(case_out)])
            fh.flush()

    print(f"Done. Status: {status_csv}")


if __name__ == "__main__":
    main()
