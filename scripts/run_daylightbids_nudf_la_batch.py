#!/usr/bin/env python3
"""
Batch pipeline (defaced inputs):
  - use defaced CTA NIfTIs from derivatives/defaced
  - run NUDF LAA (CardiacCTExplorer) + TotalSegmentator heartchambers_highres
  - extract Left Atrium (heart_atrium_left) from heartchambers_highres

Outputs under derivatives/:
  - nudf_la/<case_id>/<case_id>_laa_nudf.nii.gz
  - nudf_la/<case_id>/<case_id>_left_atrium_highres.nii.gz
  - nudf_la/qc_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


LA_LABEL_ID = 2  # TotalSegmentator heartchambers_highres: heart_atrium_left


def _run(cmd: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch NUDF LAA + LA highres from defaced CTAs")
    p.add_argument("--root", default="/Volumes/DICOM3/DAYLIGHTBIDS", help="DAYLIGHTBIDS root")
    p.add_argument("--defaced-dir", default=None, help="Defaced input dir (default: root/derivatives/defaced)")
    p.add_argument(
        "--defaced-glob",
        default="sub-*_acq-CTA_ct_defaced.nii.gz",
        help="Glob for defaced inputs",
    )
    p.add_argument("--out-dir", default=None, help="Output base dir (default: root/derivatives/nudf_la)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")

    p.add_argument("--nudf-env", default="cardiac-ct-explorer", help="Conda env for NUDF")
    p.add_argument("--device", default="auto", help="NUDF device: auto|cpu|gpu")
    p.add_argument("--totalseg-device", default=None, help="TotalSegmentator device override")
    p.add_argument(
        "--roi-subset-total",
        default="atrial_appendage_left,pulmonary_vein",
        help="Comma-separated ROI subset for TotalSegmentator total task",
    )
    p.add_argument(
        "--roi-subset-heartchambers",
        default="heart_atrium_left,aorta",
        help="Comma-separated ROI subset for TotalSegmentator heartchambers_highres task",
    )
    p.add_argument("--skip-coronary", action="store_true", default=True, help="Skip coronary_arteries task (faster)")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    return p.parse_args()


def _scan_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    # remove trailing _defaced if present
    if name.endswith("_defaced"):
        name = name[: -len("_defaced")]
    return name


def _find_heartchambers(output_dir: Path) -> Path | None:
    # CardiacCTExplorer may place TotalSegmentator outputs in different subfolders.
    matches = list(output_dir.glob("**/heartchambers_highres.nii.gz"))
    if not matches:
        return None
    # Prefer the shallowest path
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _extract_la(heartchambers_path: Path, out_path: Path) -> None:
    img = nib.load(str(heartchambers_path))
    data = img.get_fdata()
    la = (data == LA_LABEL_ID).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(la, img.affine, img.header), str(out_path))


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    defaced_dir = Path(args.defaced_dir) if args.defaced_dir else root / "derivatives" / "defaced"
    if not defaced_dir.exists():
        raise FileNotFoundError(f"Defaced dir not found: {defaced_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else root / "derivatives" / "nudf_la"
    out_dir.mkdir(parents=True, exist_ok=True)

    defaced_files = sorted(defaced_dir.glob(args.defaced_glob))
    if not defaced_files:
        raise RuntimeError(f"No defaced files found with pattern: {defaced_dir}/{args.defaced_glob}")

    summary_path = out_dir / "qc_summary.csv"
    summary_rows = []

    for idx, defaced_path in enumerate(defaced_files):
        if args.limit is not None and idx >= args.limit:
            break

        case_id = _scan_id(defaced_path)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        laa_out = case_dir / f"{case_id}_laa_nudf.nii.gz"
        la_out = case_dir / f"{case_id}_left_atrium_highres.nii.gz"

        if not args.force and laa_out.exists() and la_out.exists():
            print(f"Outputs exist, skipping: {case_id}")
            summary_rows.append(
                {
                    "case_id": case_id,
                    "defaced_path": str(defaced_path),
                    "laa_nudf_path": str(laa_out),
                    "la_highres_path": str(la_out),
                }
            )
            continue

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
            str(case_dir / "cardiac_ct_explorer"),
            "--laa-output",
            str(laa_out),
            "--run-totalseg",
            "--device",
            args.device,
        ]
        if args.totalseg_device:
            cmd += ["--totalseg-device", args.totalseg_device]
        if args.roi_subset_total:
            cmd += ["--roi-subset-total", args.roi_subset_total]
        if args.roi_subset_heartchambers:
            cmd += ["--roi-subset-heartchambers", args.roi_subset_heartchambers]
        if args.skip_coronary:
            cmd += ["--skip-coronary"]

        _run(cmd, args.dry_run)

        if args.dry_run:
            continue

        heartchambers_path = _find_heartchambers(case_dir)
        if heartchambers_path is None:
            raise FileNotFoundError(f"heartchambers_highres.nii.gz not found under {case_dir}")

        _extract_la(heartchambers_path, la_out)
        print(f"Saved LA: {la_out}")

        summary_rows.append(
            {
                "case_id": case_id,
                "defaced_path": str(defaced_path),
                "laa_nudf_path": str(laa_out),
                "la_highres_path": str(la_out),
            }
        )

    with summary_path.open("w", newline="") as f:
        fieldnames = ["case_id", "defaced_path", "laa_nudf_path", "la_highres_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
