#!/usr/bin/env python3
"""Standalone peri-LAA fat shell extraction.

Use when the SLAAO annotation package already exists and you only need to
(re-)compute fat shells without rebuilding the whole annotation package; or
when you want to run with a *corrected* expert LAA mask instead of the
consensus prior.

Single-case example:

    python scripts/run_peri_laa_fat.py \\
        --case-id sub-547_acq-CTA_ct \\
        --ct-path /data/sub-547_acq-CTA_ct.nii.gz \\
        --laa-mask outputs/test/prior_fusion_547/sub-547_acq-CTA_ct_consensus_laa.nii.gz \\
        --negative-prior outputs/test/prior_fusion_547/sub-547_acq-CTA_ct_negative_prior.nii.gz \\
        --out-dir outputs/peri_laa_fat/sub-547 \\
        --shells 0-2,2-5,5-10 \\
        --write-per-shell-masks

Batch example (reads a CSV with case_id, ct_path, laa_mask, negative_prior columns):

    python scripts/run_peri_laa_fat.py \\
        --batch-csv cases.csv \\
        --out-root outputs/peri_laa_fat \\
        --shells 0-2,2-5,5-10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.laa_slaao.peri_laa_fat import (
    DEFAULT_SHELLS_MM, run_peri_laa_fat_from_paths,
)


def _parse_shells(spec: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for band in spec.split(","):
        band = band.strip()
        if not band:
            continue
        lo, hi = band.split("-")
        out.append((float(lo), float(hi)))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute peri-LAA fat shell features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--case-id", default=None)
    g.add_argument("--batch-csv", default=None,
                   help="CSV with columns: case_id, ct_path, laa_mask, "
                        "[negative_prior]")

    p.add_argument("--ct-path", default=None)
    p.add_argument("--laa-mask", default=None)
    p.add_argument("--negative-prior", default=None)
    p.add_argument("--out-dir", default=None,
                   help="Single-case output dir.")
    p.add_argument("--out-root", default=None,
                   help="Batch output root; writes to <out-root>/<case_id>/.")
    p.add_argument(
        "--shells", default=",".join(f"{lo:g}-{hi:g}" for lo, hi in DEFAULT_SHELLS_MM),
        help="Comma-separated radial bands in mm.",
    )
    p.add_argument("--fat-hu-min", type=float, default=-190.0)
    p.add_argument("--fat-hu-max", type=float, default=-30.0)
    p.add_argument("--write-per-shell-masks", action="store_true",
                   help="In addition to the multi-label NIfTI, write one binary "
                        "NIfTI per shell — useful for direct Slicer overlay.")
    p.add_argument("--extend-laa", action="store_true",
                   help="Also emit an extended-LAA mask: close the perifat "
                        "into a 3D boundary (using only the fat's own "
                        "geometry — no ROI), then fill the enclosed pocket "
                        "and union with the LAA. Captures the LAA wall + "
                        "any thrombus / filling defect.")
    p.add_argument("--extend-laa-closing-mm", type=float, default=4.0,
                   help="Closing radius (mm) used to seal gaps in the fat "
                        "shell before fill-holes.")
    p.add_argument("--extend-laa-max-added-ml", type=float, default=25.0,
                   help="Safety cap: refuse extension if the added pocket "
                        "would exceed this volume.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _one(case_id, ct_path, laa_mask, neg_prior, out_dir, shells, hu_min, hu_max,
         write_per_shell, extend_laa, extend_laa_closing_mm,
         extend_laa_max_added_ml, log):
    log(f"[{case_id}] CT={ct_path}")
    log(f"[{case_id}] LAA={laa_mask}")
    log(f"[{case_id}] neg-prior={neg_prior or '∅'}")
    log(f"[{case_id}] shells={shells}  HU=[{hu_min},{hu_max}]")
    result, paths = run_peri_laa_fat_from_paths(
        ct_path=Path(ct_path),
        laa_mask_path=Path(laa_mask),
        out_dir=Path(out_dir),
        case_id=case_id,
        fat_hu_min=hu_min,
        fat_hu_max=hu_max,
        shells_mm=shells,
        negative_prior_path=Path(neg_prior) if neg_prior else None,
        write_per_shell_masks=write_per_shell,
        extend_laa=extend_laa,
        extend_laa_closing_mm=extend_laa_closing_mm,
        extend_laa_max_added_volume_ml=extend_laa_max_added_ml,
    )
    log(f"[{case_id}] total_volume_ml="
        f"{result.features.get('peri_laa_fat_total_volume_ml')}  "
        f"mean_HU={result.features.get('peri_laa_fat_total_mean_hu')}")
    for k, p in paths.items():
        log(f"  {k:10s}  {p}")
    return result, paths


def main() -> None:
    args = _parse_args()

    def log(msg: str):
        if not args.quiet:
            print(msg, flush=True)

    shells = _parse_shells(args.shells)

    if args.case_id:
        for required in ("ct_path", "laa_mask", "out_dir"):
            if getattr(args, required) is None:
                raise SystemExit(f"--{required.replace('_','-')} is required in single-case mode")
        _one(
            args.case_id, args.ct_path, args.laa_mask, args.negative_prior,
            args.out_dir, shells, args.fat_hu_min, args.fat_hu_max,
            args.write_per_shell_masks,
            args.extend_laa, args.extend_laa_closing_mm,
            args.extend_laa_max_added_ml,
            log,
        )
        return

    # Batch mode
    if args.out_root is None:
        raise SystemExit("--out-root is required in batch mode")
    df = pd.read_csv(args.batch_csv)
    for required in ("case_id", "ct_path", "laa_mask"):
        if required not in df.columns:
            raise SystemExit(f"batch CSV missing column: {required}")
    has_neg = "negative_prior" in df.columns
    out_root = Path(args.out_root)

    ok, failed = 0, []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        try:
            _one(
                case_id=case_id,
                ct_path=row["ct_path"],
                laa_mask=row["laa_mask"],
                neg_prior=row["negative_prior"] if has_neg and pd.notna(
                    row.get("negative_prior")) else None,
                out_dir=out_root / case_id,
                shells=shells, hu_min=args.fat_hu_min, hu_max=args.fat_hu_max,
                write_per_shell=args.write_per_shell_masks,
                extend_laa=args.extend_laa,
                extend_laa_closing_mm=args.extend_laa_closing_mm,
                extend_laa_max_added_ml=args.extend_laa_max_added_ml,
                log=log,
            )
            ok += 1
        except Exception as exc:
            log(f"  [fail] {case_id}: {exc}")
            failed.append(case_id)

    log(f"[peri_laa_fat] done: {ok} ok, {len(failed)} failed")
    if failed:
        log(f"  failed: {failed}")


if __name__ == "__main__":
    main()
