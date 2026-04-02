#!/usr/bin/env python3
"""
Master segmentation batch — TotalSegmentator + VISTA3D for all SLAAOBIDS CTs.

Produces three radiomics-ready masks per case (by default, no flags needed):
  {case_id}_left_atrium_highres.nii.gz   ← TotalSegmentator heartchambers, label 2
  {case_id}_aorta_highres_ts.nii.gz      ← TotalSegmentator heartchambers, label 6
  {case_id}_laa_vista3d.nii.gz           ← VISTA3D label 108

CT types covered:
  ecta      : derivatives/defaced/*_defaced.nii.gz           (251 cases)
  ctthorax  : sub-*/sub-*_acq-ctthorax_ct.nii.gz             (single phase)
  ctabdomen : sub-*/sub-*_acq-ctabdomen_ph*_ct.nii.gz        (all phases)
  ctbody    : sub-*/sub-*_acq-ctbody_ph*_ct.nii.gz           (all phases)
  ctheart   : sub-*/sub-*_acq-ctheart_ph*_ct.nii.gz          (all phases)

Cases that already have all three masks are automatically skipped.

Output directories (matching existing pipeline):
  eCTA    → <root>/derivatives/nudf_la_eCTA/<case_id>/
  non-eCTA → <root>/derivatives/nudf_la_multict/<case_id>/

Summary CSV → <root>/derivatives/seg_summary_full.csv

Usage (single GPU, everything default):
  conda run -n cardiac-ct-explorer python scripts/run_full_seg_batch.py \\
      --root "C:/Users/spost/Desktop/CT_image/SLAAOBIDS"

Usage (dual GPU):
  conda run -n cardiac-ct-explorer python scripts/run_full_seg_batch.py \\
      --root "C:/Users/spost/Desktop/CT_image/SLAAOBIDS" \\
      --workers 2 --device-list gpu:0,gpu:1 --vista3d-device-list cuda:0,cuda:1

Usage (dry run to preview):
  conda run -n cardiac-ct-explorer python scripts/run_full_seg_batch.py \\
      --root "C:/Users/spost/Desktop/CT_image/SLAAOBIDS" --dry-run
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
_ALL_ACQ_TYPES  = ["ctthorax", "ctabdomen", "ctbody", "ctheart"]


# ── Per-case worker (top-level for ProcessPoolExecutor pickling) ──────────────

def _process_case(job: dict) -> dict:
    import time
    import nibabel as nib
    import numpy as np

    started     = time.time()
    case_id     = job["case_id"]
    input_path  = Path(job["input_path"])
    case_dir    = Path(job["case_dir"])
    la_out      = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
    aorta_out   = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"
    laa_v3d_out = case_dir / f"{case_id}_laa_vista3d.nii.gz"
    case_dir.mkdir(parents=True, exist_ok=True)

    ts_status  = "ok"
    ts_message = ""
    la_vox     = ""
    ao_vox     = ""

    # ── TotalSegmentator heartchambers_highres ────────────────────────────────
    try:
        ts_done = la_out.exists() and aorta_out.exists() and not job.get("force")
        if ts_done:
            try:
                la_vox = int((nib.load(str(la_out)).get_fdata(dtype="float32") > 0).sum())
            except Exception:  # noqa: BLE001
                la_vox = ""
            try:
                ao_vox = int((nib.load(str(aorta_out)).get_fdata(dtype="float32") > 0).sum())
            except Exception:  # noqa: BLE001
                pass
        else:
            ts_dir = case_dir / "totalseg_heartchambers"
            # NOTE: do NOT pre-create ts_dir as a directory.
            # TotalSegmentator ml=True treats `output` as a FILE path base.
            # Pre-creating the directory causes TS to write totalseg_heartchambers.nii
            # one level up in case_dir instead of inside ts_dir.
            ts_device = job.get("totalseg_device") or job["device"]
            _in  = str(input_path).replace("\\", "/")
            _out = str(ts_dir).replace("\\", "/")
            ts_snippet = (
                "from totalsegmentator.python_api import totalsegmentator; "
                f"totalsegmentator(input='{_in}', output='{_out}', "
                f"task='heartchambers_highres', device='{ts_device}', ml=True)"
            )
            if job.get("python"):
                ts_cmd = [job["python"], "-c", ts_snippet]
            else:
                ts_cmd = ["conda", "run", "-n", job["conda_env"], "python", "-c", ts_snippet]
            ts_proc = subprocess.run(ts_cmd)
            if ts_proc.returncode != 0:
                raise RuntimeError(f"TotalSegmentator subprocess exit code {ts_proc.returncode}")

            # TotalSegmentator ml=True writes a single file whose base name matches
            # the last component of the output path we passed (ts_dir.name =
            # "totalseg_heartchambers"), placed in case_dir (ts_dir's parent).
            hc_candidates = [
                case_dir / (ts_dir.name + ".nii.gz"),  # totalseg_heartchambers.nii.gz
                case_dir / (ts_dir.name + ".nii"),      # totalseg_heartchambers.nii
                ts_dir / "heartchambers_highres.nii.gz",  # fallback: old TS directory mode
            ]
            hc_path = next((p for p in hc_candidates if p.exists()), None)
            if hc_path is None:
                raise FileNotFoundError(
                    f"TotalSegmentator output not found; searched: "
                    f"{[str(p) for p in hc_candidates]}"
                )

            hc_img = nib.load(str(hc_path))
            data   = hc_img.get_fdata(dtype="float32")

            la_mask = (data == _LA_LABEL_ID).astype("uint8")
            la_vox  = int(la_mask.sum())
            ao_mask = (data == _AORTA_LABEL_ID).astype("uint8")
            ao_vox  = int(ao_mask.sum())
            nib.save(nib.Nifti1Image(ao_mask, hc_img.affine, hc_img.header), str(aorta_out))

            if la_vox < _MIN_LA_VOXELS:
                ts_status  = "skip_la_fov"
                ts_message = f"LA voxels={la_vox} < {_MIN_LA_VOXELS} (heart outside FOV)"
            else:
                nib.save(nib.Nifti1Image(la_mask, hc_img.affine, hc_img.header), str(la_out))

    except Exception as exc:  # noqa: BLE001
        ts_status  = "failed"
        ts_message = str(exc)
        print(f"  [TS-ERROR] {case_id}: {exc}", flush=True)

    # ── VISTA3D LAA (label 108) — runs independently of TotalSegmentator ──────
    v3d_status  = "skipped"
    v3d_message = ""

    if not job.get("force") and laa_v3d_out.exists():
        v3d_status  = "skipped"
        v3d_message = "output already exists"
    else:
        vista3d_runner = Path(job["scripts_dir"]) / _VISTA3D_SCRIPT
        v3d_cmd = (
            [job["python"], str(vista3d_runner)] if job.get("python")
            else ["conda", "run", "-n", job["conda_env"], "python", str(vista3d_runner)]
        )
        v3d_cmd += [
            "--input",    str(input_path),
            "--output",   str(laa_v3d_out),
            "--label-id", "108",
            "--device",   job["vista3d_device"],
        ]
        if job.get("vista3d_model_dir"):
            v3d_cmd += ["--model-dir", job["vista3d_model_dir"]]

        try:
            v3d_proc = subprocess.run(v3d_cmd)
            if v3d_proc.returncode != 0:
                raise RuntimeError(f"VISTA3D exit code {v3d_proc.returncode}")
            if not laa_v3d_out.exists():
                raise RuntimeError("VISTA3D finished but output file not created")
            v3d_status = "ok"
        except Exception as exc:  # noqa: BLE001
            v3d_status  = "failed"
            v3d_message = str(exc)

    # Overall status: worst of the two steps
    if ts_status == "failed" or v3d_status == "failed":
        overall = "failed"
    elif ts_status == "skip_la_fov":
        overall = "skip_la_fov"
    else:
        overall = "ok"

    combined_message = "  |  ".join(
        m for m in [ts_message, v3d_message] if m
    )

    return {
        "ct_type":     job["ct_type"],
        "case_id":     case_id,
        "input_path":  str(input_path),
        "la_path":     str(la_out)      if la_out.exists()      else "",
        "aorta_path":  str(aorta_out)   if aorta_out.exists()   else "",
        "laa_v3d_path": str(laa_v3d_out) if laa_v3d_out.exists() else "",
        "la_voxels":   str(la_vox),
        "aorta_voxels": str(ao_vox),
        "ts_device":   job.get("totalseg_device") or job["device"],
        "v3d_device":  job["vista3d_device"],
        "elapsed_sec": f"{time.time() - started:.1f}",
        "status":      overall,
        "ts_status":   ts_status,
        "v3d_status":  v3d_status,
        "message":     combined_message,
    }


# ── TotalSegmentator smoke test ───────────────────────────────────────────────

def _run_totalseg_smoke_test(job: dict) -> bool:
    """Run TotalSegmentator on one case in the main process, printing all output.

    Returns True if the test passed, False if it failed (caller should abort).
    """
    import tempfile

    input_path = job["input_path"]
    ts_device  = job.get("totalseg_device") or job["device"]

    with tempfile.TemporaryDirectory() as _tmp_parent:
        # Pass a non-existent path inside the temp dir so TS ml=True can create
        # the file freely (e.g. heartchambers_smoke.nii.gz) rather than finding
        # a pre-existing directory and writing next to it.
        tmp_out = _tmp_parent  # used for listing below
        _in  = str(input_path).replace("\\", "/")
        _out = (Path(_tmp_parent) / "heartchambers_smoke").as_posix()
        ts_snippet = (
            "from totalsegmentator.python_api import totalsegmentator; "
            f"totalsegmentator(input='{_in}', output='{_out}', "
            f"task='heartchambers_highres', device='{ts_device}', ml=True)"
        )
        if job.get("python"):
            ts_cmd = [job["python"], "-c", ts_snippet]
        else:
            ts_cmd = ["conda", "run", "-n", job["conda_env"], "python", "-c", ts_snippet]

        sep = "=" * 70
        print(sep, flush=True)
        print("[SMOKE TEST] TotalSegmentator heartchambers_highres", flush=True)
        print(f"  case   : {job['case_id']}", flush=True)
        print(f"  input  : {input_path}", flush=True)
        print(f"  output : {tmp_out}  (temp — deleted after test)", flush=True)
        print(f"  device : {ts_device}", flush=True)
        print(f"  cmd    : {' '.join(ts_cmd)}", flush=True)
        print(sep, flush=True)

        # stdout/stderr NOT captured — flow directly to terminal so errors are visible
        proc = subprocess.run(ts_cmd)

        print(sep, flush=True)
        print(f"[SMOKE TEST] Exit code: {proc.returncode}", flush=True)

        # Full recursive listing — shows exactly what TotalSegmentator created and where
        all_files = sorted(Path(tmp_out).rglob("*")) if Path(tmp_out).exists() else []
        nii_files = [f for f in all_files if f.name.endswith(".nii.gz") or f.name.endswith(".nii")]
        print(f"[SMOKE TEST] All items under output dir ({len(all_files)} total):", flush=True)
        for f in all_files:
            rel = f.relative_to(tmp_out)
            tag = "DIR " if f.is_dir() else "FILE"
            print(f"  {tag}  {rel}", flush=True)
        if not all_files:
            print("  (empty — nothing was written to the temp output dir)", flush=True)
        print(sep, flush=True)

        if proc.returncode != 0:
            print(
                "[SMOKE TEST] FAILED — TotalSegmentator returned a non-zero exit code.\n"
                "  Fix the error above, then re-run the batch.",
                flush=True,
            )
            return False

        if not nii_files:
            print(
                "[SMOKE TEST] FAILED — subprocess exited 0 but produced no .nii/.nii.gz files.\n"
                "  See the listing above for what was actually created.",
                flush=True,
            )
            return False

        print(f"[SMOKE TEST] PASSED — {len(nii_files)} NIfTI file(s) found: {[f.name for f in nii_files]}. Starting batch ...\n", flush=True)
        return True


# ── Input discovery ───────────────────────────────────────────────────────────

def _case_id_from_path(path: Path, strip_suffix: str = "") -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if strip_suffix and name.endswith(strip_suffix):
        name = name[: -len(strip_suffix)]
    return name


def _collect_jobs(
    root: Path,
    ecta_out_dir: Path,
    multict_out_dir: Path,
    scripts_dir: str,
    device_list: list[str] | None,
    vista3d_device_list: list[str] | None,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    jobs: list[dict]    = []
    skipped: list[dict] = []
    job_idx = 0

    def _register(fp: Path, ct_type: str, case_dir: Path) -> None:
        nonlocal job_idx
        suffix  = "_defaced" if ct_type == "ecta" else ""
        case_id = _case_id_from_path(fp, strip_suffix=suffix)
        la_out      = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_out   = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"
        laa_v3d_out = case_dir / f"{case_id}_laa_vista3d.nii.gz"

        if not args.force and la_out.exists() and aorta_out.exists() and laa_v3d_out.exists():
            skipped.append({"ct_type": ct_type, "case_id": case_id})
            return

        device      = device_list[job_idx % len(device_list)] if device_list else args.device
        v3d_device  = (
            vista3d_device_list[job_idx % len(vista3d_device_list)]
            if vista3d_device_list else args.vista3d_device
        )

        jobs.append({
            "ct_type":           ct_type,
            "case_id":           case_id,
            "input_path":        str(fp),
            "case_dir":          str(case_dir),
            "scripts_dir":       scripts_dir,
            "conda_env":         args.conda_env,
            "python":            getattr(args, "python", None),
            "device":            device,
            "totalseg_device":   args.totalseg_device,
            "vista3d_device":    v3d_device,
            "vista3d_model_dir": args.vista3d_model_dir,
            "force":             args.force,
        })
        job_idx += 1

    # ── eCTA (from derivatives/defaced/) ─────────────────────────────────────
    defaced_dir = root / "derivatives" / "defaced"
    if defaced_dir.exists():
        for fp in sorted(defaced_dir.glob("*_defaced.nii.gz")):
            case_id = _case_id_from_path(fp, strip_suffix="_defaced")
            _register(fp, "ecta", ecta_out_dir / case_id)
    else:
        print(f"[WARN] eCTA defaced dir not found: {defaced_dir}", file=sys.stderr)

    # ── non-eCTA (from sub-* directories) ────────────────────────────────────
    sub_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    for acq_type in _ALL_ACQ_TYPES:
        for sub_dir in sub_dirs:
            pattern = (
                f"*_acq-{acq_type}_ct.nii.gz"
                if acq_type == "ctthorax"
                else f"*_acq-{acq_type}_ph*_ct.nii.gz"
            )
            for fp in sorted(sub_dir.glob(pattern)):
                case_id = _case_id_from_path(fp)
                _register(fp, acq_type, multict_out_dir / case_id)

    return jobs, skipped


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Master segmentation batch: TotalSegmentator + VISTA3D for all SLAAOBIDS CTs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS",
        help="SLAAOBIDS root directory",
    )
    p.add_argument(
        "--ecta-out-dir", default=None,
        help="Output dir for eCTA cases (default: <root>/derivatives/nudf_la_eCTA)",
    )
    p.add_argument(
        "--multict-out-dir", default=None,
        help="Output dir for non-eCTA cases (default: <root>/derivatives/nudf_la_multict)",
    )
    # TotalSegmentator device
    p.add_argument(
        "--device", default="gpu",
        help="TotalSegmentator device (gpu|cpu|gpu:0)",
    )
    p.add_argument(
        "--totalseg-device", default=None,
        help="TotalSegmentator device override (default: same as --device)",
    )
    p.add_argument(
        "--device-list", default=None,
        help="Multi-GPU: comma-separated TotalSegmentator devices, e.g. gpu:0,gpu:1 (assigned round-robin)",
    )
    # VISTA3D device
    p.add_argument(
        "--vista3d-device", default="cuda:0",
        help="VISTA3D device in PyTorch format (cuda:0|cpu|auto)",
    )
    p.add_argument(
        "--vista3d-device-list", default=None,
        help="Multi-GPU: comma-separated VISTA3D devices, e.g. cuda:0,cuda:1 (assigned round-robin)",
    )
    p.add_argument(
        "--vista3d-model-dir", default=None,
        help="NV-Segment-CT model dir (auto-downloaded from HuggingFace if omitted)",
    )
    # Runtime
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (1 = safe for single GPU)",
    )
    p.add_argument(
        "--conda-env", default="cardiac-ct-explorer",
        help="Conda environment for VISTA3D subprocess",
    )
    p.add_argument(
        "--python", default=None,
        help="Direct Python executable for VISTA3D subprocess (skips conda run)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Reprocess cases even if all three output masks already exist",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N cases total (for quick testing)",
    )
    p.add_argument(
        "--acq-types", default=None,
        help=f"Restrict non-eCTA types, comma-separated (default: all — {','.join(_ALL_ACQ_TYPES)})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print job list without executing anything",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    ecta_out_dir    = Path(args.ecta_out_dir)    if args.ecta_out_dir    else root / "derivatives" / "nudf_la_eCTA"
    multict_out_dir = Path(args.multict_out_dir) if args.multict_out_dir else root / "derivatives" / "nudf_la_multict"
    ecta_out_dir.mkdir(parents=True, exist_ok=True)
    multict_out_dir.mkdir(parents=True, exist_ok=True)

    if args.acq_types:
        global _ALL_ACQ_TYPES
        _ALL_ACQ_TYPES = [t.strip() for t in args.acq_types.split(",") if t.strip()]

    device_list = (
        [d.strip() for d in args.device_list.split(",")]
        if args.device_list else None
    )
    vista3d_device_list = (
        [d.strip() for d in args.vista3d_device_list.split(",")]
        if args.vista3d_device_list else None
    )
    scripts_dir = str(Path(__file__).resolve().parent)

    jobs, skipped = _collect_jobs(
        root, ecta_out_dir, multict_out_dir,
        scripts_dir, device_list, vista3d_device_list, args,
    )

    if args.limit:
        jobs = jobs[: args.limit]

    # ── Summary ───────────────────────────────────────────────────────────────
    all_types = ["ecta"] + _ALL_ACQ_TYPES
    print(f"Root         : {root}")
    print(f"Total CTs    : {len(jobs) + len(skipped)}")
    print(f"Already done : {len(skipped)}  (all 3 masks present — skipping)")
    print(f"To process   : {len(jobs)}")
    print(f"Workers      : {args.workers}")
    print(f"TS device    : {args.totalseg_device or (device_list[0] if device_list else args.device)}")
    print(f"V3D device   : {vista3d_device_list[0] if vista3d_device_list else args.vista3d_device}")
    print()
    for ct_type in all_types:
        n_proc = sum(1 for j in jobs    if j["ct_type"] == ct_type)
        n_skip = sum(1 for s in skipped if s["ct_type"] == ct_type)
        if n_proc + n_skip:
            print(f"  {ct_type:12s}  to_process={n_proc:4d}  skipped={n_skip:4d}")
    print()

    if args.dry_run:
        print("Dry run — nothing executed.")
        return 0

    if not jobs:
        print("Nothing to process.")
        return 0

    # ── Smoke test: verify TotalSegmentator works before committing to the batch
    if not _run_totalseg_smoke_test(jobs[0]):
        return 1

    # ── Run ───────────────────────────────────────────────────────────────────
    fieldnames = [
        "ct_type", "case_id", "input_path",
        "la_path", "aorta_path", "laa_v3d_path",
        "la_voxels", "aorta_voxels",
        "ts_device", "v3d_device", "elapsed_sec",
        "status", "ts_status", "v3d_status", "message",
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
                result.update({
                    "case_id": case_id,
                    "status":  "failed",
                    "message": str(exc),
                })
            summary_rows.append(result)
            st  = result["status"]
            ct  = result.get("ct_type", "")
            v3d = result.get("v3d_status", "")
            msg = f"  {result['message']}" if result.get("message") else ""
            line = f"[{st.upper():12s}] [{ct:12s}] {case_id}  ts={result.get('ts_status','')}  v3d={v3d}{msg}"
            tqdm.write(line) if use_tqdm else print(line)

            # incremental live CSV — readable while the batch is still running
            live_csv = root / "derivatives" / "seg_summary_full_live.csv"
            live_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not live_csv.exists() or live_csv.stat().st_size == 0
            with live_csv.open("a", newline="") as _f:
                _w = csv.DictWriter(_f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    _w.writeheader()
                _w.writerow(result)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = root / "derivatives" / "seg_summary_full.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
