#!/usr/bin/env python3
"""Batch CTA-DEFACE for all subjects in DAYLIGHTBIDS.

Usage (run from terminal):
    cd <PROJECT_ROOT>
    source .venv_dt/bin/activate
    python scripts/batch_deface.py --device mps
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# nnUNet env vars
SCRIPT_DIR = Path(__file__).parent.parent / "external" / "CTA-DEFACE"
os.environ["nnUNet_results"] = str(SCRIPT_DIR / "model")
os.environ["nnUNet_preprocessed"] = str(SCRIPT_DIR / "model")
os.environ["nnUNet_raw"] = str(SCRIPT_DIR / "model")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BIDS_DIR = Path(os.environ.get("DAYLIGHTBIDS_ROOT", "./data/daylightbids"))
OUTPUT_DIR = BIDS_DIR / "derivatives" / "defaced"


def prepare_input(bids_dir: Path, input_dir: Path, output_dir: Path) -> list[tuple[str, Path]]:
    """Symlink CTA files into nnUNet input folder, skipping already defaced."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for f in input_dir.glob("*_0000.nii.gz"):
        f.unlink()

    subjects = []
    skipped = 0
    for cta_file in sorted(bids_dir.glob("sub-*_acq-CTA_ct.nii.gz")):
        sub_id = cta_file.name.replace("_acq-CTA_ct.nii.gz", "")
        defaced_path = output_dir / f"{sub_id}_acq-CTA_ct_defaced.nii.gz"
        if defaced_path.exists():
            skipped += 1
            continue
        link_name = input_dir / f"{sub_id}_0000.nii.gz"
        link_name.symlink_to(cta_file.resolve())
        subjects.append((sub_id, cta_file))

    if skipped:
        print(f"  Skipped {skipped} already defaced subjects")
    return subjects


def run_inference(input_dir: Path, mask_dir: Path, device: str):
    """Run nnUNet inference on all subjects at once."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    n_subjects = len(list(input_dir.glob("*_0000.nii.gz")))
    if n_subjects == 0:
        print("  No subjects to process!")
        return

    command = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(mask_dir),
        "-d", "001",
        "-c", "3d_fullres",
        "-f", "all",
        "--disable_tta",
        "-npp", "1",
        "-nps", "1",
        "-device", device,
    ]

    import subprocess
    print(f"  nnUNet predicting {n_subjects} subjects on {device}...")
    print(f"  Command: {' '.join(command)}\n")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(f"\nnnUNet failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def apply_masks(subjects: list[tuple[str, Path]], mask_dir: Path, output_dir: Path):
    """Apply face masks to create defaced images with progress bar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = []

    for sub_id, original_path in tqdm(subjects, desc="Applying masks", unit="sub"):
        mask_path = mask_dir / f"{sub_id}.nii.gz"
        if not mask_path.exists():
            failed.append(sub_id)
            continue

        defaced_path = output_dir / f"{sub_id}_acq-CTA_ct_defaced.nii.gz"
        mask_out_path = output_dir / f"{sub_id}_acq-CTA_ct_facemask.nii.gz"

        if defaced_path.exists():
            done += 1
            continue

        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata().astype(np.uint8)

        img = nib.load(str(original_path))
        data = img.get_fdata()
        fill_val = np.percentile(data, 10)
        defaced = np.where(mask == 1, fill_val, data)

        nib.save(nib.Nifti1Image(defaced.astype(np.float32), img.affine, img.header), str(defaced_path))
        nib.save(nib.Nifti1Image(mask, mask_img.affine), str(mask_out_path))
        done += 1

    if failed:
        print(f"\n  WARNING: {len(failed)} subjects had no mask: {failed}")
    return done


def main():
    parser = argparse.ArgumentParser(description="Batch CTA-DEFACE")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--bids-dir", default=str(BIDS_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    output_dir = Path(args.output_dir)
    input_dir = SCRIPT_DIR / "batch_input"
    mask_dir = SCRIPT_DIR / "batch_masks"

    print("=" * 60)
    print("BATCH CTA-DEFACE")
    print(f"  Source:  {bids_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Device:  {args.device}")
    print("=" * 60)

    t0 = time.time()

    # 1. Prepare input symlinks (skip already processed)
    print("\n[1/3] Preparing input...")
    subjects = prepare_input(bids_dir, input_dir, output_dir)
    print(f"  {len(subjects)} subjects to process")

    if not subjects:
        print("\nAll subjects already defaced!")
        return

    # 2. Run nnUNet inference (nnUNet has its own progress bar)
    print("\n[2/3] Running nnUNet inference...")
    run_inference(input_dir, mask_dir, args.device)

    # 3. Apply masks with progress bar
    print("\n[3/3] Applying face masks...")
    done = apply_masks(subjects, mask_dir, output_dir)

    elapsed = time.time() - t0
    minutes = elapsed / 60
    per_sub = elapsed / max(len(subjects), 1)
    print(f"\nDone! {done}/{len(subjects)} defaced in {minutes:.1f} min ({per_sub:.0f}s/subject)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
