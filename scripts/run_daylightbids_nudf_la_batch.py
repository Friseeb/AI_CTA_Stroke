#!/usr/bin/env python3
"""
Batch pipeline for DAYLIGHTBIDS CTA segmentation.

Per case, this runner can:
  1) Run NUDF LAA + TotalSegmentator heartchambers_highres bootstrap.
  2) Export LA highres from TotalSegmentator heartchambers_highres (label 2).
  3) Export aorta highres from TotalSegmentator heartchambers_highres (label 6).
  4) Export aorta highres from MONAI/VISTA3D (label 6), with optional CPU fallback.
  5) Write canonical aorta_highres from selected source.

Outputs under derivatives/nudf_la/<case_id>/:
  - <case_id>_laa_nudf.nii.gz
  - <case_id>_left_atrium_highres.nii.gz
  - <case_id>_aorta_highres_ts.nii.gz
  - <case_id>_aorta_highres_monai.nii.gz
  - <case_id>_aorta_highres.nii.gz   (canonical output)
  - qc_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


LA_LABEL_ID = 2
AORTA_LABEL_ID = 6

# Minimum LA voxel count to consider the heart within the CT field of view.
# eCTA scans focused on the head/neck may not cover the heart at all; TotalSegmentator
# will then return an empty or near-empty label 2, which causes silent failures
# downstream (empty mesh, zero-volume metrics).  Any case below this threshold is
# flagged as "skip_la_fov" in qc_summary.csv and its empty LA file is removed.
# Rule of thumb: a normal LA at typical eCTA resolution is >10,000 voxels.
# 1 000 is a deliberately conservative floor that catches truly empty masks while
# tolerating edge cases where only a small inferior portion of the LA is captured.
MIN_LA_VOXELS = 1_000


class _LAFOVExcludedError(RuntimeError):
    """Raised when the LA label is present but too small to be a real segmentation."""


def _run(
    cmd: list[str],
    dry_run: bool,
    check: bool = True,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    quiet_subprocess: bool = False,
) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    if quiet_subprocess:
        if log_path is None:
            raise ValueError("log_path is required when quiet_subprocess=True")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write("=== Running ===\n")
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, env=env)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch NUDF/TotalSegmentator/MONAI pipeline for DAYLIGHTBIDS CTAs")
    p.add_argument("--root", default="./data/daylightbids", help="DAYLIGHTBIDS root")
    p.add_argument("--use-nondefaced", action="store_true", help="Use root CTA files instead of derivatives/defaced")
    p.add_argument("--input-dir", default=None, help="Override input dir (default depends on --use-nondefaced)")
    p.add_argument("--input-glob", default=None, help="Override input glob pattern")
    p.add_argument("--out-dir", default=None, help="Output base dir (default: root/derivatives/nudf_la)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    p.add_argument("--progress", dest="progress", action="store_true", default=None, help="Show tqdm progress bar")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bar")
    p.add_argument("--quiet-subprocess", action="store_true", help="Redirect per-case subprocess stdout/stderr to log files")
    p.add_argument("--subprocess-log-dir", default=None, help="Directory for per-case logs (default: <out-dir>/_logs)")

    # NUDF / TotalSegmentator stage
    p.add_argument("--nudf-env", default="cardiac-ct-explorer", help="Conda env for NUDF stage when --nudf-python is not set")
    p.add_argument("--nudf-python", default=None, help="Python executable for NUDF stage (recommended)")
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
    p.add_argument("--skip-coronary", dest="skip_coronary", action="store_true", default=True, help="Skip coronary_arteries task")
    p.add_argument("--no-skip-coronary", dest="skip_coronary", action="store_false", help="Run coronary_arteries task")
    p.add_argument("--allow-missing-laa", dest="allow_missing_laa", action="store_true", default=True, help="Write empty LAA if NUDF fails")
    p.add_argument("--no-allow-missing-laa", dest="allow_missing_laa", action="store_false", help="Fail if NUDF LAA is missing")
    p.add_argument("--totalseg-fast", action="store_true", help="Use TotalSegmentator fast mode (lower memory, lower fidelity)")
    p.add_argument("--retry-nudf-on-fail-cpu", action="store_true", help="Retry failed NUDF/TotalSegmentator case on CPU")
    p.add_argument("--large-z-threshold", type=int, default=None, help="If set, cases with z-dim >= threshold use --large-z-totalseg-device")
    p.add_argument("--large-z-totalseg-device", default="cpu", help="TotalSegmentator device for large-z cases (default: cpu)")
    p.add_argument("--incremental-summary", dest="incremental_summary", action="store_true", default=True, help="Append per-case rows to qc_summary_live.csv")
    p.add_argument("--no-incremental-summary", dest="incremental_summary", action="store_false", help="Disable incremental summary writes")

    # Aorta outputs
    p.add_argument("--save-ts-aorta", action="store_true", help="Save TS aorta as <case>_aorta_highres_ts.nii.gz")
    p.add_argument("--run-monai-aorta", action="store_true", help="Run MONAI/VISTA3D aorta (label 6)")
    p.add_argument("--monai-script", default=str(Path(__file__).parent / "run_nv_segment_ct_laa.py"), help="Path to MONAI aorta runner")
    p.add_argument("--monai-python", default=None, help="Python executable for MONAI stage (default: --nudf-python or current python)")
    p.add_argument("--monai-model-dir", default=str(Path(__file__).resolve().parents[1] / "external" / "nv_segment_ct"), help="MONAI/NV-Segment-CT model dir")
    p.add_argument("--monai-device", default="cuda:0", help="MONAI device, e.g. cuda:0 or cpu")
    p.add_argument("--monai-fallback-cpu", dest="monai_fallback_cpu", action="store_true", default=True, help="Retry MONAI on CPU when GPU run fails")
    p.add_argument("--no-monai-fallback-cpu", dest="monai_fallback_cpu", action="store_false", help="Do not retry MONAI on CPU")
    p.add_argument("--monai-cuda-visible-devices", default=None, help="Set CUDA_VISIBLE_DEVICES for MONAI stage")
    p.add_argument("--main-aorta-source", choices=["monai", "totalseg", "none"], default="monai", help="Source for canonical <case>_aorta_highres.nii.gz")
    return p.parse_args()


def _scan_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_defaced"):
        name = name[: -len("_defaced")]
    return name


def _find_heartchambers(output_dir: Path) -> Path | None:
    matches = list(output_dir.glob("**/heartchambers_highres.nii.gz"))
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _extract_binary_label(img: nib.Nifti1Image, label_id: int, out_path: Path) -> int:
    data = np.asanyarray(img.dataobj)
    mask = (data == label_id).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))
    return int(mask.sum())


def _binarize_inplace(mask_path: Path) -> int:
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    mask = (data > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(mask_path))
    return int(mask.sum())


def _build_nudf_cmd(
    args: argparse.Namespace,
    input_path: Path,
    case_dir: Path,
    laa_out: Path,
    device_override: str | None = None,
    totalseg_device_override: str | None = None,
    disable_totalseg_fast: bool = False,
) -> list[str]:
    runner = str(Path(__file__).parent / "run_cardiac_ct_explorer_nudf_only.py")
    if args.nudf_python:
        cmd = [args.nudf_python, runner]
    else:
        cmd = ["conda", "run", "-n", args.nudf_env, "python", runner]
    nudf_device = device_override or args.device
    totalseg_device = totalseg_device_override or args.totalseg_device
    cmd += [
        "--input",
        str(input_path),
        "--output-dir",
        str(case_dir / "cardiac_ct_explorer"),
        "--laa-output",
        str(laa_out),
        "--run-totalseg",
        "--device",
        nudf_device,
    ]
    if totalseg_device:
        cmd += ["--totalseg-device", totalseg_device]
    if args.totalseg_fast and not disable_totalseg_fast:
        cmd += ["--totalseg-fast"]
    if args.roi_subset_total:
        cmd += ["--roi-subset-total", args.roi_subset_total]
    if args.roi_subset_heartchambers:
        cmd += ["--roi-subset-heartchambers", args.roi_subset_heartchambers]
    if args.skip_coronary:
        cmd += ["--skip-coronary"]
    if args.allow_missing_laa:
        cmd += ["--allow-missing-laa"]
    return cmd


def _append_summary_row(summary_path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_z_dim(path: Path) -> int | None:
    try:
        return int(nib.load(str(path)).shape[2])
    except Exception:  # noqa: BLE001
        return None


def _read_log_tail(log_path: Path, max_lines: int = 120) -> str:
    if not log_path.exists():
        return ""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:  # noqa: BLE001
        return ""
    return "".join(lines[-max_lines:])


def _classify_case_log_failure(log_path: Path) -> tuple[str, bool]:
    tail = _read_log_tail(log_path).lower()
    if not tail:
        return "", False
    if "heartchambers_highres does not work with option --fast" in tail:
        return "heartchambers_highres is incompatible with --totalseg-fast", False
    if "compressed file ended before the end-of-stream marker was reached" in tail:
        return "input NIfTI appears corrupted (truncated .nii.gz stream)", True
    if "not a gzipped file" in tail:
        return "input NIfTI appears corrupted (invalid .nii.gz stream)", True
    return "", False


def _is_likely_corrupt_input_error(message: str) -> bool:
    lower = message.lower()
    patterns = (
        "compressed file ended before the end-of-stream marker was reached",
        "not a gzipped file",
        "input nifti appears corrupted",
    )
    return any(p in lower for p in patterns)


def _build_monai_cmd(
    monai_python: str,
    monai_script: str,
    input_path: Path,
    output_path: Path,
    model_dir: str,
    device: str,
) -> list[str]:
    return [
        monai_python,
        monai_script,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--label-id",
        str(AORTA_LABEL_ID),
        "--model-dir",
        model_dir,
        "--device",
        device,
    ]


def _choose_main_aorta_source(
    preferred: str,
    ts_path: Path,
    monai_path: Path,
) -> Path | None:
    if preferred == "none":
        return None
    if preferred == "monai":
        if monai_path.exists():
            return monai_path
        if ts_path.exists():
            return ts_path
        return None
    if preferred == "totalseg":
        if ts_path.exists():
            return ts_path
        if monai_path.exists():
            return monai_path
        return None
    return None


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = root if args.use_nondefaced else root / "derivatives" / "defaced"
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.input_glob:
        input_glob = args.input_glob
    else:
        input_glob = "sub-*_acq-CTA_ct.nii.gz" if args.use_nondefaced else "sub-*_acq-CTA_ct_defaced.nii.gz"

    out_dir = Path(args.out_dir) if args.out_dir else root / "derivatives" / "nudf_la"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(input_glob))
    if not input_files:
        raise RuntimeError(f"No input files found with pattern: {input_dir}/{input_glob}")

    monai_python = args.monai_python or args.nudf_python or sys.executable
    monai_env = os.environ.copy()
    if args.monai_cuda_visible_devices is not None:
        monai_env["CUDA_VISIBLE_DEVICES"] = args.monai_cuda_visible_devices

    summary_path = out_dir / "qc_summary.csv"
    summary_live_path = out_dir / "qc_summary_live.csv"
    fieldnames = [
        "case_id",
        "input_path",
        "laa_nudf_path",
        "la_highres_path",
        "aorta_ts_path",
        "aorta_monai_path",
        "aorta_highres_path",
        "la_voxels",
        "aorta_ts_voxels",
        "aorta_monai_voxels",
        "case_z",
        "totalseg_device_used",
        "retried_cpu",
        "elapsed_sec",
        "status",
        "message",
    ]
    summary_rows: list[dict[str, str]] = []
    progress_enabled = (sys.stdout.isatty() if args.progress is None else args.progress) and tqdm is not None
    subprocess_log_dir = Path(args.subprocess_log_dir) if args.subprocess_log_dir else (out_dir / "_logs")
    if args.quiet_subprocess and not args.dry_run:
        subprocess_log_dir.mkdir(parents=True, exist_ok=True)

    loop_iter = enumerate(input_files)
    if progress_enabled:
        loop_iter = enumerate(tqdm(input_files, total=len(input_files), desc="CTA cases", unit="case"))

    for idx, input_path in loop_iter:
        if args.limit is not None and idx >= args.limit:
            break

        started_at = time.time()
        case_id = _scan_id(input_path)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_log = subprocess_log_dir / f"{case_id}.log"
        case_z = _safe_z_dim(input_path)
        case_totalseg_device = args.totalseg_device
        if (
            args.large_z_threshold is not None
            and case_z is not None
            and case_z >= args.large_z_threshold
        ):
            case_totalseg_device = args.large_z_totalseg_device
            print(
                f"[MEM-SAFE] {case_id}: z={case_z} >= {args.large_z_threshold}; "
                f"using totalseg-device={case_totalseg_device}"
            )

        laa_out = case_dir / f"{case_id}_laa_nudf.nii.gz"
        la_out = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_ts_out = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"
        aorta_monai_out = case_dir / f"{case_id}_aorta_highres_monai.nii.gz"
        aorta_main_out = case_dir / f"{case_id}_aorta_highres.nii.gz"
        corrupt_marker = case_dir / "_corrupt_input.txt"

        required_outputs = [laa_out, la_out]
        if args.save_ts_aorta:
            required_outputs.append(aorta_ts_out)
        if args.run_monai_aorta:
            required_outputs.append(aorta_monai_out)
        if args.main_aorta_source != "none" and (args.save_ts_aorta or args.run_monai_aorta):
            required_outputs.append(aorta_main_out)

        if not args.force and required_outputs and all(p.exists() for p in required_outputs):
            print(f"Outputs exist, skipping: {case_id}")
            row = {
                "case_id": case_id,
                "input_path": str(input_path),
                "laa_nudf_path": str(laa_out),
                "la_highres_path": str(la_out),
                "aorta_ts_path": str(aorta_ts_out if aorta_ts_out.exists() else ""),
                "aorta_monai_path": str(aorta_monai_out if aorta_monai_out.exists() else ""),
                "aorta_highres_path": str(aorta_main_out if aorta_main_out.exists() else ""),
                "la_voxels": "",
                "aorta_ts_voxels": "",
                "aorta_monai_voxels": "",
                "case_z": str(case_z if case_z is not None else ""),
                "totalseg_device_used": str(case_totalseg_device or ""),
                "retried_cpu": "no",
                "elapsed_sec": f"{time.time() - started_at:.1f}",
                "status": "skipped",
                "message": "all requested outputs already present",
            }
            summary_rows.append(row)
            if args.incremental_summary and not args.dry_run:
                _append_summary_row(summary_live_path, row, fieldnames)
            continue

        if not args.force and corrupt_marker.exists():
            marker_reason = "input previously marked as corrupt"
            try:
                marker_reason = corrupt_marker.read_text(encoding="utf-8").strip() or marker_reason
            except Exception:  # noqa: BLE001
                pass
            print(f"[SKIP-CORRUPT] {case_id}: {marker_reason}")
            row = {
                "case_id": case_id,
                "input_path": str(input_path),
                "laa_nudf_path": "",
                "la_highres_path": "",
                "aorta_ts_path": "",
                "aorta_monai_path": "",
                "aorta_highres_path": "",
                "la_voxels": "",
                "aorta_ts_voxels": "",
                "aorta_monai_voxels": "",
                "case_z": str(case_z if case_z is not None else ""),
                "totalseg_device_used": str(case_totalseg_device or ""),
                "retried_cpu": "no",
                "elapsed_sec": f"{time.time() - started_at:.1f}",
                "status": "skipped",
                "message": marker_reason,
            }
            summary_rows.append(row)
            if args.incremental_summary and not args.dry_run:
                _append_summary_row(summary_live_path, row, fieldnames)
            continue

        status = "ok"
        message = ""
        la_vox = ""
        aorta_ts_vox = ""
        aorta_monai_vox = ""
        retried_cpu = "no"
        mark_corrupt_input = False

        try:
            disable_totalseg_fast = False
            nudf_cmd = _build_nudf_cmd(
                args=args,
                input_path=input_path,
                case_dir=case_dir,
                laa_out=laa_out,
                totalseg_device_override=case_totalseg_device,
                disable_totalseg_fast=disable_totalseg_fast,
            )
            rc = _run(
                nudf_cmd,
                args.dry_run,
                check=False,
                log_path=case_log,
                quiet_subprocess=args.quiet_subprocess,
            )
            if rc != 0 and args.totalseg_fast:
                disable_totalseg_fast = True
                print(
                    f"[RETRY-NOFAST] {case_id}: initial NUDF/TS exit={rc}; "
                    "retrying without --totalseg-fast"
                )
                nudf_cmd_nofast = _build_nudf_cmd(
                    args=args,
                    input_path=input_path,
                    case_dir=case_dir,
                    laa_out=laa_out,
                    totalseg_device_override=case_totalseg_device,
                    disable_totalseg_fast=disable_totalseg_fast,
                )
                rc = _run(
                    nudf_cmd_nofast,
                    args.dry_run,
                    check=False,
                    log_path=case_log,
                    quiet_subprocess=args.quiet_subprocess,
                )
            if rc != 0 and args.retry_nudf_on_fail_cpu:
                retried_cpu = "yes"
                print(f"[RETRY-CPU] {case_id}: initial NUDF/TS exit={rc}, retrying with CPU")
                nudf_cmd_cpu = _build_nudf_cmd(
                    args=args,
                    input_path=input_path,
                    case_dir=case_dir,
                    laa_out=laa_out,
                    device_override="cpu",
                    totalseg_device_override="cpu",
                    disable_totalseg_fast=disable_totalseg_fast,
                )
                rc = _run(
                    nudf_cmd_cpu,
                    args.dry_run,
                    check=False,
                    log_path=case_log,
                    quiet_subprocess=args.quiet_subprocess,
                )
            if rc != 0:
                hint, hint_corrupt = _classify_case_log_failure(case_log)
                mark_corrupt_input = hint_corrupt
                if hint:
                    raise RuntimeError(f"NUDF/TotalSegmentator failed for {case_id} (exit={rc}): {hint}")
                raise RuntimeError(f"NUDF/TotalSegmentator failed for {case_id} (exit={rc})")

            if not args.dry_run:
                heartchambers_path = _find_heartchambers(case_dir)
                if heartchambers_path is None:
                    raise FileNotFoundError(f"heartchambers_highres.nii.gz not found under {case_dir}")

                hc_img = nib.load(str(heartchambers_path))
                la_vox = str(_extract_binary_label(hc_img, LA_LABEL_ID, la_out))
                print(f"Saved LA: {la_out} voxels={la_vox}")
                if int(la_vox) < MIN_LA_VOXELS:
                    la_out.unlink(missing_ok=True)
                    raise _LAFOVExcludedError(
                        f"LA label 2 contains only {la_vox} voxels "
                        f"(threshold={MIN_LA_VOXELS}); heart likely outside CT field of view"
                    )

                if args.save_ts_aorta:
                    aorta_ts_vox = str(_extract_binary_label(hc_img, AORTA_LABEL_ID, aorta_ts_out))
                    print(f"Saved TS aorta: {aorta_ts_out} voxels={aorta_ts_vox}")

                if args.run_monai_aorta:
                    monai_cmd = _build_monai_cmd(
                        monai_python=monai_python,
                        monai_script=args.monai_script,
                        input_path=input_path,
                        output_path=aorta_monai_out,
                        model_dir=args.monai_model_dir,
                        device=args.monai_device,
                    )
                    rc = _run(
                        monai_cmd,
                        args.dry_run,
                        check=False,
                        env=monai_env,
                        log_path=case_log,
                        quiet_subprocess=args.quiet_subprocess,
                    )
                    if rc != 0 and args.monai_fallback_cpu:
                        print(f"MONAI failed on {args.monai_device}; retrying on CPU for {case_id}")
                        monai_cmd_cpu = _build_monai_cmd(
                            monai_python=monai_python,
                            monai_script=args.monai_script,
                            input_path=input_path,
                            output_path=aorta_monai_out,
                            model_dir=args.monai_model_dir,
                            device="cpu",
                        )
                        rc = _run(
                            monai_cmd_cpu,
                            args.dry_run,
                            check=False,
                            env=monai_env,
                            log_path=case_log,
                            quiet_subprocess=args.quiet_subprocess,
                        )
                    if rc != 0:
                        raise RuntimeError(f"MONAI aorta failed for {case_id} (exit={rc})")
                    aorta_monai_vox = str(_binarize_inplace(aorta_monai_out))
                    print(f"Saved MONAI aorta: {aorta_monai_out} voxels={aorta_monai_vox}")

                if args.main_aorta_source != "none" and (args.save_ts_aorta or args.run_monai_aorta):
                    source = _choose_main_aorta_source(
                        preferred=args.main_aorta_source,
                        ts_path=aorta_ts_out,
                        monai_path=aorta_monai_out,
                    )
                    if source is not None:
                        shutil.copy2(source, aorta_main_out)
                        print(f"Saved canonical aorta: {aorta_main_out} (from {source.name})")

        except _LAFOVExcludedError as exc:
            status = "skip_la_fov"
            message = str(exc)
            print(f"[SKIP-LA-FOV] {case_id}: {message}")
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = str(exc)
            if _is_likely_corrupt_input_error(message):
                mark_corrupt_input = True
            if mark_corrupt_input and not args.dry_run:
                marker_text = (
                    "input marked as corrupt; use --force after replacing the source file.\n"
                    f"input={input_path}\n"
                    f"reason={message}\n"
                )
                corrupt_marker.write_text(marker_text, encoding="utf-8")
                print(f"[MARK-CORRUPT] {case_id}: {corrupt_marker}")
            print(f"[FAIL] {case_id}: {message}")

        row = {
            "case_id": case_id,
            "input_path": str(input_path),
            "laa_nudf_path": str(laa_out if laa_out.exists() or args.dry_run else ""),
            "la_highres_path": str(la_out if la_out.exists() or args.dry_run else ""),
            "aorta_ts_path": str(aorta_ts_out if aorta_ts_out.exists() or args.dry_run else ""),
            "aorta_monai_path": str(aorta_monai_out if aorta_monai_out.exists() or args.dry_run else ""),
            "aorta_highres_path": str(aorta_main_out if aorta_main_out.exists() or args.dry_run else ""),
            "la_voxels": la_vox,
            "aorta_ts_voxels": aorta_ts_vox,
            "aorta_monai_voxels": aorta_monai_vox,
            "case_z": str(case_z if case_z is not None else ""),
            "totalseg_device_used": str(case_totalseg_device or ""),
            "retried_cpu": retried_cpu,
            "elapsed_sec": f"{time.time() - started_at:.1f}",
            "status": status,
            "message": message,
        }
        summary_rows.append(row)
        if args.incremental_summary and not args.dry_run:
            _append_summary_row(summary_live_path, row, fieldnames)

    if args.dry_run:
        print("Dry-run complete. No summary files written.")
        return 0

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Summary: {summary_path}")
    if args.incremental_summary:
        print(f"Incremental summary: {summary_live_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
