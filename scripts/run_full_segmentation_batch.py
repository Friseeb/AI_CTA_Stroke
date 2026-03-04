#!/usr/bin/env python3
"""
Batch wrapper for full CTA segmentation pipeline.

Reads a CSV manifest with at least:
  - case_id
  - dicom_dir OR input_nifti

For each case, runs:
  DICOM -> NIfTI (optional) -> deface -> segment -> merge labels

Example:
  python -u scripts/run_full_segmentation_batch.py \
    --manifest data/manifests/cta_inputs.csv \
    --output-root outputs/full_seg_batch \
    --run-totalseg --run-topcow --run-nv --run-nudf --merge-labels
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_case_cmd(args: argparse.Namespace, row: Dict[str, str]) -> list[str]:
    case_id = row.get(args.id_field, "").strip()
    if not case_id:
        raise ValueError(f"Missing case_id in manifest row: {row}")

    dicom_dir = row.get(args.dicom_field, "").strip()
    input_nifti = row.get(args.nifti_field, "").strip()
    if not dicom_dir and not input_nifti:
        raise ValueError(f"Row must contain dicom_dir or input_nifti: {row}")

    case_out = Path(args.output_root) / case_id
    cmd = [
        "python",
        str(Path(__file__).parent / "run_full_segmentation_pipeline.py"),
        "--output-dir",
        str(case_out),
        "--case-id",
        case_id,
    ]

    if dicom_dir:
        cmd += ["--dicom-dir", dicom_dir]
    else:
        cmd += ["--input-nifti", input_nifti]

    if args.deface:
        cmd.append("--deface")
    if args.skip_deface:
        cmd.append("--skip-deface")

    if args.run_totalseg:
        cmd.append("--run-totalseg")
    if args.skip_totalseg:
        cmd.append("--skip-totalseg")

    if args.run_topcow:
        cmd.append("--run-topcow")
        if args.topcow_yolo_model:
            cmd += ["--topcow-yolo-model", args.topcow_yolo_model]
        if args.topcow_nnunet_model_dir:
            cmd += ["--topcow-nnunet-model-dir", args.topcow_nnunet_model_dir]
    if args.skip_topcow:
        cmd.append("--skip-topcow")

    if args.run_nv:
        cmd.append("--run-nv")
    if args.skip_nv:
        cmd.append("--skip-nv")
    if args.run_nv_aorta:
        cmd.append("--run-nv-aorta")
    if args.skip_nv_aorta:
        cmd.append("--skip-nv-aorta")

    if args.run_nudf:
        cmd.append("--run-nudf")
    if args.skip_nudf:
        cmd.append("--skip-nudf")

    if args.merge_labels:
        cmd.append("--merge-labels")
    if args.skip_merge:
        cmd.append("--skip-merge")

    # Env routing
    cmd += ["--totalseg-env", args.totalseg_env]
    cmd += ["--topcow-env", args.topcow_env]
    cmd += ["--nv-env", args.nv_env]
    cmd += ["--nudf-env", args.nudf_env]
    cmd += ["--merge-env", args.merge_env]

    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch CTA segmentation wrapper")
    p.add_argument("--manifest", required=True, help="CSV manifest (case_id + dicom_dir or input_nifti)")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--id-field", default="case_id", help="Manifest column for case id")
    p.add_argument("--dicom-field", default="dicom_dir", help="Manifest column for DICOM directory")
    p.add_argument("--nifti-field", default="input_nifti", help="Manifest column for input NIfTI")
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")

    p.add_argument("--deface", action="store_true", help="Run defacing")
    p.add_argument("--skip-deface", action="store_true", help="Skip defacing")

    p.add_argument("--run-totalseg", action="store_true", help="Run TotalSegmentator tasks")
    p.add_argument("--skip-totalseg", action="store_true", help="Skip TotalSegmentator tasks")

    p.add_argument("--run-topcow", action="store_true", help="Run TopCoW (Circle of Willis)")
    p.add_argument("--skip-topcow", action="store_true", help="Skip TopCoW")
    p.add_argument("--topcow-yolo-model", default=None, help="YOLO model path for TopCoW")
    p.add_argument("--topcow-nnunet-model-dir", default=None, help="nnUNet model dir for TopCoW")

    p.add_argument("--run-nv", action="store_true", help="Run NV-Segment-CT LAA")
    p.add_argument("--skip-nv", action="store_true", help="Skip NV-Segment-CT LAA")
    p.add_argument("--run-nv-aorta", action="store_true", help="Run NV-Segment-CT aorta (label 6)")
    p.add_argument("--skip-nv-aorta", action="store_true", help="Skip NV-Segment-CT aorta")

    p.add_argument("--run-nudf", action="store_true", help="Run NUDF LAA (CardiacCTExplorer)")
    p.add_argument("--skip-nudf", action="store_true", help="Skip NUDF LAA")

    p.add_argument("--merge-labels", action="store_true", help="Build merged label map")
    p.add_argument("--skip-merge", action="store_true", help="Skip merged label map")

    # Env routing
    p.add_argument("--totalseg-env", default="totalseg-mac", help="Conda env for TotalSegmentator")
    p.add_argument("--topcow-env", default="topcow_claim", help="Conda env for TopCoW")
    p.add_argument("--nv-env", default="nv-segment-ct", help="Conda env for NV-Segment-CT")
    p.add_argument("--nudf-env", default="cardiac-ct-explorer", help="Conda env for NUDF")
    p.add_argument("--merge-env", default="cardiac-ct-explorer", help="Conda env for merge step")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            cmd = _build_case_cmd(args, row)
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
                continue
            _run(cmd)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
