#!/usr/bin/env python3
"""
Batch pipeline for DAYLIGHTBIDS:
  - (optional) deface CTA
  - run NUDF LAA segmentation (CardiacCTExplorer) with TotalSegmentator

Outputs under derivatives/:
  - defaced/*.nii.gz
  - laa_nudf/*.nii.gz
  - laa_nudf/qc_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch NUDF LAA pipeline for DAYLIGHTBIDS")
    p.add_argument("--root", default="./data/daylightbids", help="DAYLIGHTBIDS root")
    p.add_argument("--derivatives", default=None, help="Derivatives folder (default: root/derivatives)")
    p.add_argument("--cta-glob", default="sub-*_acq-CTA_ct.nii.gz", help="CTA glob under root")
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")

    p.add_argument("--skip-deface", action="store_true", help="Skip defacing")
    p.add_argument("--skip-nudf", action="store_true", help="Skip NUDF LAA")

    p.add_argument("--deface-env", default="cta-deface", help="Conda env for CTA-DEFACE")
    p.add_argument("--deface-device", default="cpu", help="CTA-DEFACE device: cpu|mps|cuda")
    p.add_argument("--nudf-env", default="cardiac-ct-explorer", help="Conda env for NUDF")
    p.add_argument("--device", default="auto", help="NUDF device: auto|cpu|gpu")
    p.add_argument("--totalseg-device", default=None, help="TotalSegmentator device override")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    derivatives = Path(args.derivatives) if args.derivatives else root / "derivatives"
    defaced_dir = derivatives / "defaced"
    nudf_dir = derivatives / "laa_nudf"
    defaced_dir.mkdir(parents=True, exist_ok=True)
    nudf_dir.mkdir(parents=True, exist_ok=True)

    cta_files = sorted(root.glob(args.cta_glob))
    if not cta_files:
        raise RuntimeError(f"No CTA files found with pattern: {root}/{args.cta_glob}")

    summary_path = nudf_dir / "qc_summary.csv"
    summary_rows = []

    for idx, cta_path in enumerate(cta_files):
        if args.limit is not None and idx >= args.limit:
            break

        base = cta_path.name.replace(".nii.gz", "")
        defaced_path = defaced_dir / f"{base}_defaced.nii.gz"
        laa_out = nudf_dir / f"{base}_defaced_laa8.nii.gz"

        if not args.skip_deface:
            if defaced_path.exists():
                print(f"Defaced exists, skipping: {defaced_path}")
            else:
                # CTA-DEFACE (nnUNet) expects *_0000.nii.gz in input folder
                case_id = cta_path.name.replace(".nii.gz", "")
                tmp_root = derivatives / "defaced_cta_deface_tmp" / case_id
                input_dir = tmp_root / "input"
                output_dir = tmp_root / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                input_path = input_dir / f"{case_id}_0000.nii.gz"
                if not input_path.exists():
                    input_path.write_bytes(cta_path.read_bytes())

                cmd = [
                    "conda",
                    "run",
                    "-n",
                    args.deface_env,
                    "python",
                    str(Path(__file__).parent.parent / "external" / "CTA-DEFACE" / "run_CTA-DEFACE_mac.py"),
                    "-i",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--device",
                    args.deface_device,
                ]

                if args.dry_run:
                    print("DRY RUN:", " ".join(cmd))
                else:
                    _run(cmd)

                defaced_candidate = output_dir / f"{case_id}_defaced.nii.gz"
                if defaced_candidate.exists():
                    defaced_path.parent.mkdir(parents=True, exist_ok=True)
                    defaced_path.write_bytes(defaced_candidate.read_bytes())
                else:
                    raise FileNotFoundError(f"CTA-DEFACE output missing: {defaced_candidate}")
        else:
            defaced_path = cta_path

        if not args.skip_nudf:
            cmd = [
                "conda",
                "run",
                "-n",
                args.nudf_env,
                "python",
                str(Path(__file__).parent / "run_cardiac_ct_explorer_nudf_only.py"),
                "--input",
                str(defaced_path),
                "--output-dir",
                str(nudf_dir / base),
                "--laa-output",
                str(laa_out),
                "--run-totalseg",
                "--device",
                args.device,
            ]
            if args.totalseg_device:
                cmd += ["--totalseg-device", args.totalseg_device]
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
            else:
                _run(cmd)

        summary_rows.append(
            {
                "case_id": base,
                "cta_path": str(cta_path),
                "defaced_path": str(defaced_path),
                "laa_nudf_path": str(laa_out),
            }
        )

    with summary_path.open("w", newline="") as f:
        fieldnames = ["case_id", "cta_path", "defaced_path", "laa_nudf_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
