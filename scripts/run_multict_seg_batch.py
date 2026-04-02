#!/usr/bin/env python3
"""
Batch TotalSegmentator + VISTA3D segmentation for non-eCTA CT types
found directly in SLAAOBIDS sub-* folders (no defacing).

CT types handled:
  ctthorax  : sub-XXX_acq-ctthorax_ct.nii.gz             (single file)
  ctabdomen : sub-XXX_acq-ctabdomen_ph00_ct.nii.gz  ...  (ALL phases processed)
  ctbody    : sub-XXX_acq-ctbody_ph00_ct.nii.gz     ...  (ALL phases processed)
  ctheart   : sub-XXX_acq-ctheart_ph00_ct.nii.gz    ...  (ALL phases processed)

Each phase gets its own output folder named after the full case_id
(e.g. sub-15_acq-ctheart_ph03_ct/).

Outputs under: <out-dir>/<case_id>/
  <case_id>_left_atrium_highres.nii.gz
  <case_id>_aorta_highres_ts.nii.gz
  totalseg_heartchambers/
  <case_id>_laa_vista3d.nii.gz          (with --run-vista3d-laa)
  <case_id>_aorta_highres_vista3d.nii.gz (with --run-vista3d-aorta)

Summary: <out-dir>/seg_summary_multict.csv

Example — pilot (1 subject per CT type):
  conda run -n cardiac-ct-explorer python scripts/run_multict_seg_batch.py ^
    --run-vista3d-laa --run-vista3d-aorta --limit-per-type 1

Example — all subjects, specific types:
  conda run -n cardiac-ct-explorer python scripts/run_multict_seg_batch.py ^
    --acq-types ctheart,ctthorax --run-vista3d-laa --run-vista3d-aorta
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
_LA_LABEL_ID    = 2
_AORTA_LABEL_ID = 6
_MIN_LA_VOXELS  = 1_000

_ALL_ACQ_TYPES = ["ctthorax", "ctabdomen", "ctbody", "ctheart"]

# ── Per-case worker ───────────────────────────────────────────────────────────

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

            la_mask      = (data == _LA_LABEL_ID).astype("uint8")
            la_vox       = int(la_mask.sum())
            aorta_mask   = (data == _AORTA_LABEL_ID).astype("uint8")
            aorta_ts_vox = int(aorta_mask.sum())
            nib.save(nib.Nifti1Image(aorta_mask, hc_img.affine, hc_img.header), str(aorta_ts_out))

            if la_vox < _MIN_LA_VOXELS:
                status  = "skip_la_fov"
                message = f"LA voxels={la_vox} < {_MIN_LA_VOXELS} (heart outside FOV)"
            else:
                nib.save(nib.Nifti1Image(la_mask, hc_img.affine, hc_img.header), str(la_out))

    except Exception as exc:  # noqa: BLE001
        status  = "failed"
        message = str(exc)

    # ── Optional VISTA3D LAA (label 108) ─────────────────────────────────────
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
            v3d_cmd = (
                [job["python"], str(vista3d_runner)] if job.get("python")
                else ["conda", "run", "-n", job["conda_env"], "python", str(vista3d_runner)]
            )
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
                    raise RuntimeError(f"VISTA3D LAA exit code {v3d_proc.returncode}")
                if not vista3d_out.exists():
                    raise RuntimeError("VISTA3D LAA output not created")
                vista3d_path   = str(vista3d_out)
                vista3d_status = "ok"
            except Exception as exc:  # noqa: BLE001
                vista3d_status  = "failed"
                vista3d_message = str(exc)

    # ── Optional VISTA3D Aorta (label 6) ─────────────────────────────────────
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
            va_cmd = (
                [job["python"], str(vista3d_runner)] if job.get("python")
                else ["conda", "run", "-n", job["conda_env"], "python", str(vista3d_runner)]
            )
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
                    raise RuntimeError("VISTA3D aorta output not created")
                vista3d_aorta_path   = str(vista3d_aorta_out)
                vista3d_aorta_status = "ok"
            except Exception as exc:  # noqa: BLE001
                vista3d_aorta_status  = "failed"
                vista3d_aorta_message = str(exc)

    return {
        "acq_type":              job["acq_type"],
        "case_id":               case_id,
        "input_path":            str(input_path),
        "la_path":               str(la_out)       if la_out.exists()        else "",
        "aorta_ts_path":         str(aorta_ts_out) if aorta_ts_out.exists()  else "",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def _find_all_files(sub_dir: Path, acq_type: str) -> list[Path]:
    """Return all NIfTI files for the given acq_type in sub_dir.

    Single-phase types (ctthorax): returns the one file (or empty list).
    Multi-phase types (ctabdomen, ctbody, ctheart): returns all phases sorted.
    """
    if acq_type in ("ecta", "ctthorax"):
        return sorted(sub_dir.glob(f"*_acq-{acq_type}_ct.nii.gz"))
    return sorted(sub_dir.glob(f"*_acq-{acq_type}_ph*_ct.nii.gz"))


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch segmentation for non-eCTA CT types in SLAAOBIDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS",
        help="SLAAOBIDS root directory",
    )
    p.add_argument(
        "--out-dir",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS/derivatives/nudf_la_multict",
        help="Output base directory",
    )
    p.add_argument(
        "--acq-types",
        default=",".join(_ALL_ACQ_TYPES),
        help="Comma-separated CT types to process",
    )
    p.add_argument(
        "--limit-per-type", type=int, default=None,
        help="Process at most N files per CT type across all subjects (useful for pilot runs)",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (1 = safe for single GPU)",
    )
    p.add_argument(
        "--device", default="gpu",
        help="Device for TotalSegmentator (gpu|cpu|gpu:X)",
    )
    p.add_argument(
        "--totalseg-device", default=None,
        help="TotalSegmentator device override",
    )
    p.add_argument(
        "--device-list", default=None,
        help="Comma-separated TotalSegmentator device list for multi-GPU, e.g. gpu:0,gpu:1",
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
        "--run-vista3d-laa", action="store_true",
        help="Run VISTA3D (label 108) for LAA",
    )
    p.add_argument(
        "--run-vista3d-aorta", action="store_true",
        help="Run VISTA3D (label 6) for aorta",
    )
    p.add_argument(
        "--vista3d-device", default="cuda:0",
        help="Device for VISTA3D in PyTorch format (cuda:0|cpu|auto). Default: cuda:0",
    )
    p.add_argument(
        "--vista3d-model-dir", default=None,
        help="NV-Segment-CT model dir (default: repo/external/nv_segment_ct)",
    )
    p.add_argument("--force",   action="store_true", help="Reprocess even if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="List jobs without executing")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args    = _parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out_dir)
    acq_types = [t.strip() for t in args.acq_types.split(",") if t.strip()]

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    device_list = (
        [d.strip() for d in args.device_list.split(",")]
        if args.device_list else None
    )
    scripts_dir = str(Path(__file__).resolve().parent)

    # ── Collect input files ───────────────────────────────────────────────────
    sub_dirs = sorted(p for p in root.iterdir()
                      if p.is_dir() and p.name.startswith("sub-"))

    jobs: list[dict]    = []
    skipped: list[dict] = []

    job_idx = 0
    for acq_type in acq_types:
        count = 0
        for sub_dir in sub_dirs:
            if args.limit_per_type is not None and count >= args.limit_per_type:
                break

            for fp in _find_all_files(sub_dir, acq_type):
                if args.limit_per_type is not None and count >= args.limit_per_type:
                    break

                case_id      = _scan_id(fp)
                case_dir     = out_dir / case_id
                la_out       = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
                aorta_ts_out = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"

                vista3d_laa_done   = (not args.run_vista3d_laa)   or (case_dir / f"{case_id}_laa_vista3d.nii.gz").exists()
                vista3d_aorta_done = (not args.run_vista3d_aorta) or (case_dir / f"{case_id}_aorta_highres_vista3d.nii.gz").exists()

                if not args.force and la_out.exists() and aorta_ts_out.exists() and vista3d_laa_done and vista3d_aorta_done:
                    skipped.append({"acq_type": acq_type, "case_id": case_id})
                    count += 1
                    continue

                device = device_list[job_idx % len(device_list)] if device_list else args.device
                jobs.append({
                    "acq_type":          acq_type,
                    "case_id":           case_id,
                    "input_path":        str(fp),
                    "case_dir":          str(case_dir),
                    "scripts_dir":       scripts_dir,
                    "conda_env":         args.conda_env,
                    "python":            args.python,
                    "device":            device,
                    "totalseg_device":   args.totalseg_device,
                    "force":             args.force,
                    "run_vista3d":       args.run_vista3d_laa,
                    "run_vista3d_aorta": args.run_vista3d_aorta,
                    "vista3d_device":    args.vista3d_device,
                    "vista3d_model_dir": args.vista3d_model_dir,
                })
                job_idx += 1
                count   += 1

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Root      : {root}")
    print(f"Out dir   : {out_dir}")
    print(f"CT types  : {acq_types}")
    print(f"Phases    : all (each phase gets its own output folder)")
    print(f"To process: {len(jobs)}")
    print(f"Skipped   : {len(skipped)}  (already done)")
    print(f"Workers   : {args.workers}")
    print(f"Device    : {args.device}")
    print()

    # Per-type breakdown
    for acq_type in acq_types:
        n_proc = sum(1 for j in jobs    if j["acq_type"] == acq_type)
        n_skip = sum(1 for s in skipped if s["acq_type"] == acq_type)
        print(f"  {acq_type:12s}  to_process={n_proc}  skipped={n_skip}")
    print()

    if args.dry_run:
        for j in jobs:
            print(f"  [dry-run] [{j['acq_type']:12s}] {j['case_id']}  device={j['device']}")
        print("\nDry run — nothing executed.")
        return 0

    if not jobs:
        print("Nothing to process.")
        return 0

    # ── Run ───────────────────────────────────────────────────────────────────
    fieldnames = [
        "acq_type", "case_id", "input_path",
        "la_path", "aorta_ts_path",
        "la_voxels", "aorta_ts_voxels",
        "device", "elapsed_sec", "status", "message",
        "vista3d_path", "vista3d_status", "vista3d_message",
        "vista3d_aorta_path", "vista3d_aorta_status", "vista3d_aorta_message",
    ]
    summary_rows: list[dict] = []

    use_tqdm = tqdm is not None
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_case, job): job["case_id"] for job in jobs}
        iter_futures = as_completed(futures)
        if use_tqdm:
            iter_futures = tqdm(iter_futures, total=len(futures),
                                desc="Segmentation", unit="case")

        for future in iter_futures:
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {k: "" for k in fieldnames}
                result.update({"case_id": case_id, "status": "failed", "message": str(exc)})
            summary_rows.append(result)
            st  = result["status"]
            msg = f"  {result['message']}" if result["message"] else ""
            acq = result.get("acq_type", "")
            line = f"[{st.upper():12s}] [{acq:12s}] {case_id}{msg}"
            tqdm.write(line) if use_tqdm else print(line)

    # ── Write summary ─────────────────────────────────────────────────────────
    summary_path = out_dir / "seg_summary_multict.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    ok     = sum(1 for r in summary_rows if r["status"] == "ok")
    fov    = sum(1 for r in summary_rows if r["status"] == "skip_la_fov")
    failed = sum(1 for r in summary_rows if r["status"] == "failed")

    print(f"\nDone.  ok={ok}  la_fov={fov}  failed={failed}  skipped={len(skipped)}")
    print(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
