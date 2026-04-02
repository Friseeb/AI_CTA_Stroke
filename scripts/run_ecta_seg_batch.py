#!/usr/bin/env python3
"""
Parallel TotalSegmentator heartchambers_highres batch segmentation
for all *_defaced.nii.gz files in a defaced derivatives folder.

Outputs per case under <out-dir>/<case_id>/:
  - <case_id>_left_atrium_highres.nii.gz (LA label 2 extracted from heartchambers)
  - <case_id>_aorta_highres_ts.nii.gz    (Aorta label 6 extracted from heartchambers)
  - totalseg_heartchambers/               (TotalSegmentator raw multi-label output)
  - <case_id>_laa_vista3d.nii.gz         (optional, with --run-vista3d-laa)
  - <case_id>_aorta_highres_vista3d.nii.gz (optional, with --run-vista3d-aorta)

Summary CSV: <out-dir>/seg_summary.csv
Per-case logs: <out-dir>/_logs/<case_id>.log

Workers guidance (--workers flag):
  --workers 1   Single GPU  — safe, one case at a time (DEFAULT)
  --workers 2   Dual GPU    — use with --device-list gpu:0,gpu:1 --vista3d-device-list cuda:0,cuda:1
  --workers N   CPU-only    — set --device cpu

Example (single GPU):
  conda run -n cardiac-ct-explorer python scripts/run_ecta_seg_batch.py

Example (dual GPU):
  conda run -n cardiac-ct-explorer python scripts/run_ecta_seg_batch.py ^
    --workers 2 --device-list gpu:0,gpu:1 --vista3d-device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

_VISTA3D_SCRIPT = "run_nv_segment_ct_laa.py"
_LA_LABEL_ID     = 2
_AORTA_LABEL_ID  = 6
_MIN_LA_VOXELS   = 1_000

# ── Per-case worker (must be top-level for ProcessPoolExecutor pickling) ──────

def _process_case(job: dict) -> dict:
    import time
    import nibabel as nib
    import numpy as np

    started = time.time()
    case_id: str = job["case_id"]
    input_path   = Path(job["input_path"])
    case_dir     = Path(job["case_dir"])
    la_out       = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
    aorta_ts_out = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"
    case_dir.mkdir(parents=True, exist_ok=True)

    status       = "ok"
    message      = ""
    la_vox       = ""
    aorta_ts_vox = ""

    try:
        ts_done = la_out.exists() and aorta_ts_out.exists() and not job.get("force")
        if ts_done:
            # Outputs already present — read existing voxel counts
            try:
                la_vox = int((nib.load(str(la_out)).get_fdata(dtype="float32") > 0).sum())
            except Exception:  # noqa: BLE001
                la_vox = ""
            try:
                aorta_ts_vox = int(
                    (nib.load(str(aorta_ts_out)).get_fdata(dtype="float32") > 0).sum()
                )
            except Exception:  # noqa: BLE001
                pass
        else:
            from totalsegmentator.python_api import totalsegmentator

            ts_dir = case_dir / "totalseg_heartchambers"
            ts_dir.mkdir(parents=True, exist_ok=True)
            totalsegmentator(
                input=str(input_path),
                output=str(ts_dir),
                task="heartchambers_highres",
                device=job.get("totalseg_device") or job["device"],
                ml=True,
            )

            hc_path = ts_dir / "heartchambers_highres.nii.gz"
            if not hc_path.exists():
                hc_matches = sorted(case_dir.glob("**/heartchambers_highres.nii.gz"),
                                    key=lambda p: len(p.parts))
                if not hc_matches:
                    raise FileNotFoundError("heartchambers_highres.nii.gz not found under case dir")
                hc_path = hc_matches[0]

            hc_img = nib.load(str(hc_path))
            data   = hc_img.get_fdata(dtype="float32")

            la_mask    = (data == _LA_LABEL_ID).astype("uint8")
            la_vox     = int(la_mask.sum())
            aorta_mask = (data == _AORTA_LABEL_ID).astype("uint8")
            aorta_ts_vox = int(aorta_mask.sum())
            nib.save(nib.Nifti1Image(aorta_mask, hc_img.affine, hc_img.header), str(aorta_ts_out))

            if la_vox < _MIN_LA_VOXELS:
                status  = "skip_la_fov"
                message = f"LA voxels={la_vox} < threshold {_MIN_LA_VOXELS} (heart outside FOV)"
            else:
                nib.save(nib.Nifti1Image(la_mask, hc_img.affine, hc_img.header), str(la_out))

    except Exception as exc:  # noqa: BLE001
        status  = "failed"
        message = str(exc)

    # ── Optional VISTA3D LAA step (label 108) ─────────────────────────────────
    vista3d_path    = ""
    vista3d_status  = "skipped"
    vista3d_message = ""

    if job.get("run_vista3d"):
        vista3d_out = case_dir / f"{case_id}_laa_vista3d.nii.gz"
        if not job.get("force") and vista3d_out.exists():
            vista3d_path    = str(vista3d_out)
            vista3d_status  = "skipped"
            vista3d_message = "output already exists"
        else:
            vista3d_runner = Path(job["scripts_dir"]) / _VISTA3D_SCRIPT
            if job.get("python"):
                v3d_cmd = [job["python"], str(vista3d_runner)]
            else:
                v3d_cmd = ["conda", "run", "-n", job["conda_env"], "python", str(vista3d_runner)]
            v3d_cmd += [
                "--input",    str(input_path),
                "--output",   str(vista3d_out),
                "--label-id", "108",
                "--device",   job.get("vista3d_device") or job["device"],
            ]
            if job.get("vista3d_model_dir"):
                v3d_cmd += ["--model-dir", job["vista3d_model_dir"]]

            try:
                v3d_proc = subprocess.run(v3d_cmd)
                if v3d_proc.returncode != 0:
                    raise RuntimeError(f"VISTA3D exit code {v3d_proc.returncode}")
                if not vista3d_out.exists():
                    raise RuntimeError("VISTA3D finished but output file not created")
                vista3d_path   = str(vista3d_out)
                vista3d_status = "ok"
            except Exception as v3d_exc:  # noqa: BLE001
                vista3d_status  = "failed"
                vista3d_message = str(v3d_exc)

    # ── Optional VISTA3D Aorta step (label 6) ────────────────────────────────
    vista3d_aorta_path    = ""
    vista3d_aorta_status  = "skipped"
    vista3d_aorta_message = ""

    if job.get("run_vista3d_aorta"):
        vista3d_aorta_out = case_dir / f"{case_id}_aorta_highres_vista3d.nii.gz"
        if not job.get("force") and vista3d_aorta_out.exists():
            vista3d_aorta_path    = str(vista3d_aorta_out)
            vista3d_aorta_status  = "skipped"
            vista3d_aorta_message = "output already exists"
        else:
            vista3d_runner = Path(job["scripts_dir"]) / _VISTA3D_SCRIPT
            if job.get("python"):
                va_cmd = [job["python"], str(vista3d_runner)]
            else:
                va_cmd = ["conda", "run", "-n", job["conda_env"], "python", str(vista3d_runner)]
            va_cmd += [
                "--input",    str(input_path),
                "--output",   str(vista3d_aorta_out),
                "--label-id", "6",
                "--device",   job.get("vista3d_device") or job["device"],
            ]
            if job.get("vista3d_model_dir"):
                va_cmd += ["--model-dir", job["vista3d_model_dir"]]
            try:
                va_proc = subprocess.run(va_cmd)
                if va_proc.returncode != 0:
                    raise RuntimeError(f"VISTA3D aorta exit code {va_proc.returncode}")
                if not vista3d_aorta_out.exists():
                    raise RuntimeError("VISTA3D aorta finished but output file not created")
                vista3d_aorta_path   = str(vista3d_aorta_out)
                vista3d_aorta_status = "ok"
            except Exception as va_exc:  # noqa: BLE001
                vista3d_aorta_status  = "failed"
                vista3d_aorta_message = str(va_exc)

    return {
        "case_id":               case_id,
        "input_path":            str(input_path),
        "la_path":               str(la_out)        if la_out.exists()        else "",
        "aorta_ts_path":         str(aorta_ts_out)  if aorta_ts_out.exists()  else "",
        "la_voxels":             str(la_vox),
        "aorta_ts_voxels":       str(aorta_ts_vox),
        "device":                job["device"],
        "elapsed_sec":           f"{time.time() - started:.1f}",
        "status":                status,
        "message":               message,
        "vista3d_path":          vista3d_path,
        "vista3d_status":        vista3d_status,
        "vista3d_message":       vista3d_message,
        "vista3d_aorta_path":    vista3d_aorta_path,
        "vista3d_aorta_status":  vista3d_aorta_status,
        "vista3d_aorta_message": vista3d_aorta_message,
    }


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel TotalSegmentator heartchambers batch for eCTA defaced files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS/derivatives/defaced",
        help="Directory containing *_defaced.nii.gz files",
    )
    p.add_argument(
        "--out-dir",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS/derivatives/nudf_la_eCTA",
        help="Output base directory",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers. Use 1 for single GPU. Use N with --device-list for N GPUs.",
    )
    p.add_argument(
        "--device", default="gpu",
        help="Device for TotalSegmentator (gpu|cpu|gpu:X). Overridden per-worker if --device-list is set.",
    )
    p.add_argument(
        "--totalseg-device", default=None,
        help="TotalSegmentator device override (default: same as --device)",
    )
    p.add_argument(
        "--device-list", default=None,
        help="Comma-separated TotalSegmentator device list for multi-GPU, e.g. gpu:0,gpu:1. "
             "Workers are assigned round-robin.",
    )
    p.add_argument(
        "--conda-env", default="cardiac-ct-explorer",
        help="Conda environment for VISTA3D subprocess",
    )
    p.add_argument(
        "--python", default=None,
        help="Direct Python executable (skips conda run)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Reprocess cases even if outputs already exist",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N cases (useful for testing)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List jobs without executing",
    )
    # VISTA3D LAA step
    p.add_argument(
        "--run-vista3d-laa", action="store_true",
        help="Run VISTA3D (label 108) for LAA, saving <case_id>_laa_vista3d.nii.gz.",
    )
    p.add_argument(
        "--run-vista3d-aorta", action="store_true",
        help="Run VISTA3D (label 6) for aorta, saving <case_id>_aorta_highres_vista3d.nii.gz.",
    )
    p.add_argument(
        "--vista3d-device", default="cuda:0",
        help="Device for VISTA3D steps in PyTorch format (cuda:0|cpu|auto). Default: cuda:0",
    )
    p.add_argument(
        "--vista3d-model-dir", default=None,
        help="NV-Segment-CT model dir for VISTA3D (default: repo/external/nv_segment_ct)",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_defaced"):
        name = name[: -len("_defaced")]
    return name


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*_defaced.nii.gz"))
    if not input_files:
        print(f"ERROR: no *_defaced.nii.gz files in {input_dir}", file=sys.stderr)
        return 1

    if args.limit:
        input_files = input_files[: args.limit]

    device_list = (
        [d.strip() for d in args.device_list.split(",")]
        if args.device_list else None
    )
    scripts_dir = str(Path(__file__).resolve().parent)

    # ── Build job list, skip already done ────────────────────────────────────
    jobs: list[dict] = []
    skipped: list[str] = []

    for i, fp in enumerate(input_files):
        case_id      = _scan_id(fp)
        case_dir     = out_dir / case_id
        la_out       = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_ts_out = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"

        vista3d_laa_done   = (not args.run_vista3d_laa)   or (case_dir / f"{case_id}_laa_vista3d.nii.gz").exists()
        vista3d_aorta_done = (not args.run_vista3d_aorta) or (case_dir / f"{case_id}_aorta_highres_vista3d.nii.gz").exists()
        if not args.force and la_out.exists() and aorta_ts_out.exists() and vista3d_laa_done and vista3d_aorta_done:
            skipped.append(case_id)
            continue

        device = device_list[i % len(device_list)] if device_list else args.device
        jobs.append({
            "case_id":          case_id,
            "input_path":       str(fp),
            "case_dir":         str(case_dir),
            "scripts_dir":      scripts_dir,
            "conda_env":        args.conda_env,
            "python":           args.python,
            "device":           device,
            "totalseg_device":  args.totalseg_device,
            "force":              args.force,
            "run_vista3d":        args.run_vista3d_laa,
            "run_vista3d_aorta":  args.run_vista3d_aorta,
            "vista3d_device":     args.vista3d_device,
            "vista3d_model_dir":  args.vista3d_model_dir,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Total files : {len(input_files)}")
    print(f"Already done: {len(skipped)}  (skipping)")
    print(f"To process  : {len(jobs)}")
    print(f"Workers     : {args.workers}")
    if device_list:
        print(f"Device list : {device_list}")
    else:
        print(f"Device      : {args.device}")
    print()

    if args.dry_run:
        for j in jobs:
            print(f"  [dry-run] {j['case_id']}  device={j['device']}")
        print("\nDry run — nothing executed.")
        return 0

    if not jobs:
        print("Nothing to process.")
        return 0

    # ── Run ───────────────────────────────────────────────────────────────────
    fieldnames = [
        "case_id", "input_path", "la_path", "aorta_ts_path",
        "la_voxels", "aorta_ts_voxels", "device", "elapsed_sec", "status", "message",
        "vista3d_path", "vista3d_status", "vista3d_message",
        "vista3d_aorta_path", "vista3d_aorta_status", "vista3d_aorta_message",
    ]
    summary_rows: list[dict] = []

    # Pre-populate skipped rows
    for cid in skipped:
        case_dir = out_dir / cid
        summary_rows.append({
            "case_id":               cid,
            "input_path":            "",
            "la_path":               str(case_dir / f"{cid}_left_atrium_highres.nii.gz"),
            "aorta_ts_path":         str(case_dir / f"{cid}_aorta_highres_ts.nii.gz"),
            "la_voxels":             "",
            "aorta_ts_voxels":       "",
            "device":                "",
            "elapsed_sec":           "0.0",
            "status":                "skipped",
            "message":               "outputs already exist",
            "vista3d_path":          "",
            "vista3d_status":        "",
            "vista3d_message":       "",
            "vista3d_aorta_path":    "",
            "vista3d_aorta_status":  "",
            "vista3d_aorta_message": "",
        })

    use_tqdm = tqdm is not None
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_case, job): job["case_id"]
            for job in jobs
        }
        iter_futures = as_completed(futures)
        if use_tqdm:
            iter_futures = tqdm(
                iter_futures,
                total=len(futures),
                desc="Segmentation",
                unit="case",
            )

        for future in iter_futures:
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "case_id":               case_id,
                    "input_path":            "",
                    "la_path":               "",
                    "aorta_ts_path":         "",
                    "la_voxels":             "",
                    "aorta_ts_voxels":       "",
                    "device":                "",
                    "elapsed_sec":           "0.0",
                    "status":                "failed",
                    "message":               str(exc),
                    "vista3d_path":          "",
                    "vista3d_status":        "",
                    "vista3d_message":       "",
                    "vista3d_aorta_path":    "",
                    "vista3d_aorta_status":  "",
                    "vista3d_aorta_message": "",
                }
            summary_rows.append(result)
            status = result["status"]
            suffix = f"  {result['message']}" if result["message"] else ""
            tqdm.write(f"[{status.upper():12s}] {case_id}{suffix}") if use_tqdm else print(
                f"[{status.upper():12s}] {case_id}{suffix}"
            )

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = out_dir / "seg_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    ok        = sum(1 for r in summary_rows if r["status"] == "ok")
    skipped_n = sum(1 for r in summary_rows if r["status"] == "skipped")
    fov       = sum(1 for r in summary_rows if r["status"] == "skip_la_fov")
    failed    = sum(1 for r in summary_rows if r["status"] == "failed")

    print(f"\nDone.  ok={ok}  skipped={skipped_n}  la_fov={fov}  failed={failed}")
    print(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
