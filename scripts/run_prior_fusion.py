#!/usr/bin/env python3
"""Run prior fusion for a single LAA case.

Combines NUDF, VISTA3D (nv_segment_ct), and TotalSegmentator LAA masks into:
  - consensus / union / intersection LAA masks
  - disagreement map
  - positive anatomical priors (LA + LAA cavity region)
  - negative anatomical priors with high-res source priority:
      heartchambers_highres > total  (myocardium, aorta, pulmonary_artery)
      coronary_arteries task > total  (coronaries)
      VISTA3D > total                 (lungs, pulmonary_vein)
  - distance transform from negative prior boundary

With --run-missing-tasks, any missing TotalSegmentator high-res task outputs
(heartchambers_highres, coronary_arteries) will be run automatically.

Examples
--------
All three LAA sources + high-res tasks already run:
  python scripts/run_prior_fusion.py \\
    --case-id sub-001_acq-CTA_ct \\
    --input /data/sub-001_defaced.nii.gz \\
    --nudf-laa derivatives/nudf_la/sub-001/sub-001_laa_nudf.nii.gz \\
    --totalseg-total-dir derivatives/totalseg/sub-001/total \\
    --totalseg-heart-dir derivatives/totalseg/sub-001/heartchambers_highres \\
    --totalseg-coronary-dir derivatives/totalseg/sub-001/coronary_arteries \\
    --vista3d-combined derivatives/vista3d/sub-001/sub-001_vista3d.nii.gz \\
    --out-dir derivatives/prior_fusion/sub-001

Auto-run any missing high-res TotalSegmentator tasks:
  python scripts/run_prior_fusion.py \\
    --case-id sub-001_acq-CTA_ct \\
    --input /data/sub-001_defaced.nii.gz \\
    --nudf-laa derivatives/nudf_la/sub-001/sub-001_laa_nudf.nii.gz \\
    --totalseg-total-dir derivatives/totalseg/sub-001/total \\
    --totalseg-heart-dir derivatives/totalseg/sub-001/heartchambers_highres \\
    --totalseg-coronary-dir derivatives/totalseg/sub-001/coronary_arteries \\
    --run-missing-tasks --device gpu \\
    --out-dir derivatives/prior_fusion/sub-001
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.laa_slaao.prior_fusion import fuse_priors, save_fusion_outputs


# Sentinel files that indicate a task has already been run
_HC_SENTINEL = "heart_atrium_left.nii.gz"    # heartchambers_highres
_COR_SENTINEL = "coronary_arteries.nii.gz"   # coronary_arteries


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse LAA priors and build high-res anatomical prior maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case-id", required=True, help="Case/subject identifier string")
    p.add_argument("--input", default=None, help="CT NIfTI path (required for --run-missing-tasks)")

    # --- LAA source masks ---
    p.add_argument("--nudf-laa", default=None, help="Binary LAA mask from NUDF (.nii.gz)")
    p.add_argument("--vista3d-laa", default=None,
                   help="Binary LAA mask from VISTA3D/nv_segment_ct (.nii.gz)")
    p.add_argument("--vista3d-label-id", type=int, default=108,
                   help="Label ID for LAA in --vista3d-laa if multi-label")
    p.add_argument("--vista3d-combined", default=None,
                   help="VISTA3D full 133-class multi-label output (.nii.gz) "
                        "— used for both LAA (108) and negative priors")
    p.add_argument("--totalseg-laa", default=None,
                   help="Binary or multi-label LAA mask from TotalSegmentator total task")
    p.add_argument("--totalseg-laa-label-id", type=int, default=None,
                   help="Label ID if --totalseg-laa is a multi-label file")

    # --- TotalSegmentator task dirs (per-structure NIfTI files) ---
    p.add_argument("--totalseg-total-dir", default=None,
                   help="TotalSegmentator 'total' task output dir")
    p.add_argument("--totalseg-heart-dir", default=None,
                   help="TotalSegmentator 'heartchambers_highres' task output dir "
                        "(preferred for myocardium, aorta, pulmonary_artery, LA)")
    p.add_argument("--totalseg-coronary-dir", default=None,
                   help="TotalSegmentator 'coronary_arteries' task output dir "
                        "(preferred for coronary_arteries)")

    # --- auto-run missing tasks ---
    p.add_argument("--run-missing-tasks", action="store_true",
                   help="Run heartchambers_highres and/or coronary_arteries tasks "
                        "if their output dirs are missing or empty (requires --input)")
    p.add_argument("--device", default="gpu", help="TotalSegmentator device: gpu|cpu|mps")
    p.add_argument("--totalseg-fast", action="store_true",
                   help="Use TotalSegmentator fast mode (not available for highres/coronary tasks)")

    p.add_argument("--out-dir", required=True, help="Output directory for fusion results")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def _task_missing(task_dir: Optional[Path], sentinel: str) -> bool:
    """Return True if the task output dir is absent or sentinel file is missing."""
    if task_dir is None:
        return True
    return not (task_dir / sentinel).exists()


def _run_totalseg_task(
    input_path: Path,
    out_dir: Path,
    task: str,
    device: str,
    quiet: bool,
) -> None:
    """Run a TotalSegmentator task via subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "totalsegmentator",
        "-i", str(input_path),
        "-o", str(out_dir),
        "--task", task,
        "--device", device,
    ]
    if not quiet:
        print(f"  [totalseg] running task={task} -> {out_dir}", flush=True)
    result = subprocess.run(cmd, capture_output=quiet)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace") if result.stderr else ""
        raise RuntimeError(f"TotalSegmentator {task} failed (rc={result.returncode}):\n{err}")


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    def log(msg: str):
        if not args.quiet:
            print(msg, flush=True)

    log(f"[prior_fusion] case: {args.case_id}")

    input_path = Path(args.input) if args.input else None
    heart_dir = Path(args.totalseg_heart_dir) if args.totalseg_heart_dir else None
    coronary_dir = Path(args.totalseg_coronary_dir) if args.totalseg_coronary_dir else None

    # --- auto-run missing TotalSegmentator tasks ---
    if args.run_missing_tasks:
        if input_path is None:
            raise SystemExit("--input is required with --run-missing-tasks")

        if _task_missing(heart_dir, _HC_SENTINEL):
            if heart_dir is None:
                raise SystemExit("--totalseg-heart-dir is required with --run-missing-tasks")
            log(f"[prior_fusion] heartchambers_highres output missing — running task")
            _run_totalseg_task(input_path, heart_dir, "heartchambers_highres", args.device, args.quiet)

        if _task_missing(coronary_dir, _COR_SENTINEL):
            if coronary_dir is None:
                raise SystemExit("--totalseg-coronary-dir is required with --run-missing-tasks")
            log(f"[prior_fusion] coronary_arteries output missing — running task")
            _run_totalseg_task(input_path, coronary_dir, "coronary_arteries", args.device, args.quiet)

    # --- report what sources are available ---
    log(
        f"[prior_fusion] sources: "
        f"nudf={'yes' if args.nudf_laa else 'no'}, "
        f"vista3d={'yes' if (args.vista3d_laa or args.vista3d_combined) else 'no'}, "
        f"totalseg={'yes' if args.totalseg_laa else 'no'} | "
        f"heart_highres={'yes' if (heart_dir and (heart_dir / _HC_SENTINEL).exists()) else 'no'}, "
        f"coronary_highres={'yes' if (coronary_dir and (coronary_dir / _COR_SENTINEL).exists()) else 'no'}"
    )

    result = fuse_priors(
        case_id=args.case_id,
        nudf_laa_path=Path(args.nudf_laa) if args.nudf_laa else None,
        vista3d_laa_path=Path(args.vista3d_laa) if args.vista3d_laa else None,
        vista3d_label_id=args.vista3d_label_id,
        totalseg_laa_path=Path(args.totalseg_laa) if args.totalseg_laa else None,
        totalseg_la_label_id=args.totalseg_laa_label_id,
        totalseg_total_dir=Path(args.totalseg_total_dir) if args.totalseg_total_dir else None,
        totalseg_heart_dir=heart_dir,
        totalseg_coronary_dir=coronary_dir,
        vista3d_combined_path=Path(args.vista3d_combined) if args.vista3d_combined else None,
        ref_image_path=input_path,
    )

    out_dir = Path(args.out_dir)
    saved = save_fusion_outputs(result, out_dir)

    elapsed = time.time() - t0
    log(f"[prior_fusion] saved {len(saved)} files to {out_dir} in {elapsed:.1f}s")
    log(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
