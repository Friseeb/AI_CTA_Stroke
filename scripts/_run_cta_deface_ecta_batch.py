#!/usr/bin/env python3
"""
Batch CTA-DEFACE for SLAAOBIDS eCTA files using the nnUNet CTA-DEFACE model.

Replaces the TotalSegmentator-based defacing with the purpose-built CTA-DEFACE
model (Dataset001_DEFACE), which produces a complete face mask covering both
soft tissue and bone.

Workflow:
  1. Collect sub-*/sub-*_acq-ecta_ct.nii.gz files from SLAAOBIDS
  2. Hardlink them into a temp nnUNet input folder as <scan>_0000.nii.gz
  3. Run nnUNetv2_predict (Dataset001, 3d_fullres, fold_all) on the full batch
  4. Apply each predicted mask: fill face region with 10th-percentile of the volume
  5. Save defaced volumes to derivatives/defaced/

Usage:
    python scripts/_run_cta_deface_ecta_batch.py --subjects 1        # test single
    python scripts/_run_cta_deface_ecta_batch.py                     # full batch
    python scripts/_run_cta_deface_ecta_batch.py --force             # re-deface all

Model location expected:
    AI_CTA_Stroke-main/external/CTA-DEFACE/model/Dataset001_DEFACE/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CTA_DEFACE_DIR = PROJECT_ROOT / "external" / "CTA-DEFACE"
MODEL_DIR    = CTA_DEFACE_DIR / "model"

SLAAOBIDS    = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
DEFACED_DIR  = SLAAOBIDS / "derivatives" / "defaced"

# ── nnUNet settings ──────────────────────────────────────────────────────────
DATASET_ID   = "001"
CONFIG       = "3d_fullres"
FOLD         = "all"


def _set_nnunet_env() -> dict:
    env = os.environ.copy()
    env["nnUNet_results"]      = str(MODEL_DIR)
    env["nnUNet_preprocessed"] = str(MODEL_DIR)
    env["nnUNet_raw"]          = str(MODEL_DIR)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def collect_ecta_files(slaaobids: Path, wanted: list[str] | None = None) -> list[tuple[str, Path]]:
    """Return (scan_id, path) pairs for all eCTA files."""
    results = []
    for ecta_path in sorted(slaaobids.glob("sub-*/sub-*_acq-ecta_ct.nii.gz")):
        sid = ecta_path.parent.name          # sub-1
        scan_id = ecta_path.stem.replace(".nii", "")  # sub-1_acq-ecta_ct
        if wanted and sid not in wanted:
            continue
        results.append((scan_id, ecta_path))
    return results


def prepare_input_dir(subjects: list[tuple[str, Path]], input_dir: Path,
                      output_dir: Path, force: bool) -> list[tuple[str, Path]]:
    """
    Hardlink (or copy) each eCTA file into input_dir as <scan_id>_0000.nii.gz.
    Skips scans whose defaced output already exists (unless --force).
    Returns the list of subjects that will actually be processed.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale links from previous run
    for f in input_dir.glob("*_0000.nii.gz"):
        f.unlink()

    to_process = []
    skipped = 0
    for scan_id, ecta_path in subjects:
        defaced_path = output_dir / f"{scan_id}_defaced.nii.gz"
        if defaced_path.exists() and not force:
            skipped += 1
            continue
        link = input_dir / f"{scan_id}_0000.nii.gz"
        try:
            os.link(ecta_path, link)           # hardlink — fast, no extra disk
        except OSError:
            shutil.copy2(ecta_path, link)      # fallback: copy
        to_process.append((scan_id, ecta_path))

    if skipped:
        print(f"  Skipped {skipped} already-defaced subjects (use --force to redo)")
    return to_process


def run_nnunet(input_dir: Path, mask_dir: Path, device: str, env: dict) -> bool:
    """Run nnUNetv2_predict on the input directory. Returns True on success."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(input_dir.glob("*_0000.nii.gz")))
    if n == 0:
        print("  No subjects to predict.")
        return True

    # Locate nnUNetv2_predict executable (conda puts it in Scripts/ on Windows)
    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / "Scripts" / "nnUNetv2_predict.exe",  # conda Windows
        python_dir / "nnUNetv2_predict.exe",              # venv / Linux
        python_dir / "Scripts" / "nnUNetv2_predict",
        python_dir / "nnUNetv2_predict",
    ]
    predict_exe = next((c for c in candidates if c.exists()), None)
    if predict_exe is None:
        predict_exe = "nnUNetv2_predict"   # rely on PATH

    cmd = [
        str(predict_exe),
        "-i", str(input_dir),
        "-o", str(mask_dir),
        "-d", DATASET_ID,
        "-c", CONFIG,
        "-f", FOLD,
        "--disable_tta",
        "-npp", "1",
        "-nps", "2",
        "-device", device,
    ]
    log_path = mask_dir.parent / "nnunet_inference.log"
    print(f"  Running nnUNetv2_predict on {n} subjects ({device}) ...")
    print(f"  nnUNet log: {log_path}\n")

    already_done = len(list(mask_dir.glob("*.nii.gz")))
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    with tqdm(total=n, desc="  nnUNet inference", unit="sub", dynamic_ncols=True) as pbar:
        seen = already_done
        while proc.poll() is None:
            now = len(list(mask_dir.glob("*.nii.gz")))
            new = now - seen
            if new > 0:
                pbar.update(new)
                seen = now
            time.sleep(3)
        # pick up any final files written after process exited
        now = len(list(mask_dir.glob("*.nii.gz")))
        pbar.update(now - seen)

    return proc.returncode == 0


def apply_masks(subjects: list[tuple[str, Path]], mask_dir: Path,
                output_dir: Path) -> tuple[int, list[str]]:
    """
    For each subject: load original + predicted mask, fill face with p10, save.
    Returns (n_done, failed_scan_ids).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = []

    for scan_id, original_path in tqdm(subjects, desc="Applying masks", unit="sub", dynamic_ncols=True):
        mask_path = mask_dir / f"{scan_id}.nii.gz"
        defaced_path = output_dir / f"{scan_id}_defaced.nii.gz"

        if not mask_path.exists():
            print(f"  WARNING: mask missing for {scan_id}: {mask_path}")
            failed.append(scan_id)
            continue

        try:
            mask_img = nib.load(str(mask_path))
            mask = np.asanyarray(mask_img.dataobj).astype(np.uint8)

            img = nib.load(str(original_path))
            data = img.get_fdata(dtype=np.float32)

            fill_val = float(np.percentile(data, 10))
            defaced = np.where(mask == 1, fill_val, data)

            nib.save(
                nib.Nifti1Image(defaced.astype(np.float32), img.affine, img.header),
                str(defaced_path),
            )
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR applying mask for {scan_id}: {exc}")
            failed.append(scan_id)

    return done, failed


def save_summary(summary_path: Path, subjects: list[tuple[str, Path]],
                 output_dir: Path, failed: list[str], elapsed: float) -> None:
    rows = []
    for scan_id, orig in subjects:
        out = output_dir / f"{scan_id}_defaced.nii.gz"
        rows.append({
            "scan_id": scan_id,
            "original": str(orig),
            "defaced": str(out) if out.exists() else "",
            "status": "failed" if scan_id in failed else "ok",
        })
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": len(subjects),
        "done": len(subjects) - len(failed),
        "failed": len(failed),
        "elapsed_sec": round(elapsed, 1),
        "results": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Batch CTA-DEFACE for SLAAOBIDS eCTA")
    p.add_argument("--slaaobids", default=str(SLAAOBIDS),
                   help="SLAAOBIDS root directory")
    p.add_argument("--output-dir", default=str(DEFACED_DIR),
                   help="Defaced output directory")
    p.add_argument("--subjects", nargs="+", metavar="N",
                   help="Limit to specific subject IDs, e.g. --subjects 1 2 3")
    p.add_argument("--force", action="store_true",
                   help="Re-deface even if output already exists")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="nnUNet device (default: cpu)")
    p.add_argument("--keep-tmp", action="store_true",
                   help="Keep temporary nnUNet input/mask directories after run")
    args = p.parse_args()

    slaaobids  = Path(args.slaaobids)
    output_dir = Path(args.output_dir)
    wanted     = [f"sub-{s}" for s in args.subjects] if args.subjects else None

    # Validate model
    model_check = MODEL_DIR / "Dataset001_DEFACE" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_all" / "checkpoint_final.pth"
    if not model_check.exists():
        print(f"ERROR: CTA-DEFACE model not found at:\n  {model_check}")
        return 2

    print("=" * 70)
    print("CTA-DEFACE BATCH  (nnUNet Dataset001_DEFACE)")
    print("=" * 70)
    print(f"  SLAAOBIDS : {slaaobids}")
    print(f"  Output    : {output_dir}")
    print(f"  Model     : {MODEL_DIR}")
    print(f"  Device    : {args.device}")

    # Temp dirs inside the output tree (same drive → hardlinks work)
    tmp_root  = output_dir / ".tmp_cta_deface"
    input_dir = tmp_root / "input"
    mask_dir  = tmp_root / "masks"

    t0 = time.time()

    # 1. Collect
    print("\n[1/4] Collecting eCTA files...")
    all_subjects = collect_ecta_files(slaaobids, wanted)
    print(f"  Found {len(all_subjects)} eCTA file(s)")
    if not all_subjects:
        print("  Nothing to do.")
        return 0

    # 2. Prepare input dir (hardlinks, skip already-defaced)
    print("\n[2/4] Preparing nnUNet input...")
    to_process = prepare_input_dir(all_subjects, input_dir, output_dir, args.force)
    print(f"  {len(to_process)} subject(s) queued for prediction")
    if not to_process:
        print("  All subjects already defaced.")
        return 0

    # 3. nnUNet inference
    print("\n[3/4] Running nnUNet inference...")
    env = _set_nnunet_env()
    ok = run_nnunet(input_dir, mask_dir, args.device, env)
    if not ok:
        print("ERROR: nnUNetv2_predict failed — check output above.")
        return 1

    # 4. Apply masks
    print("\n[4/4] Applying face masks...")
    done, failed = apply_masks(to_process, mask_dir, output_dir)

    elapsed = time.time() - t0

    # Summary
    summary_path = output_dir / f"cta_deface_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_summary(summary_path, to_process, output_dir, failed, elapsed)

    print("\n" + "=" * 70)
    print(f"  Success : {done}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        for s in failed:
            print(f"    ✗ {s}")
    print(f"  Elapsed : {elapsed/60:.1f} min ({elapsed/max(done,1):.0f}s/subject)")
    print("=" * 70)

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
