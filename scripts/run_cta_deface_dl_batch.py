#!/usr/bin/env python3
"""Run CTA-DEFACE (nnUNet Dataset001_DEFACE) on a CTA folder using 2+ GPUs.

This script:
- Uses the official CTA-DEFACE nnUNet model (Dataset001_DEFACE)
- Splits pending cases across visible GPU IDs
- Runs one nnUNet inference worker per GPU
- Writes:
  - defaced volumes: <case>_defaced.nii.gz
  - face masks:      <case>_mask.nii.gz
- Skips cases that already have defaced output
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


@dataclass
class Worker:
    gpu: str
    input_dir: Path
    pred_dir: Path
    log_path: Path
    cases: list[Path]
    proc: subprocess.Popen | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch CTA-DEFACE inference on multi-GPU")
    p.add_argument(
        "--backend",
        choices=["cta-deface", "pydeface"],
        default="cta-deface",
        help="Defacing backend: CTA-DEFACE nnUNet or pydeface CLI",
    )
    p.add_argument(
        "--input-dir",
        default="/media/fridmans/b202ad4e-785a-49f0-a418-ec73cd117466/datasets/daylightbids",
        help="Input CTA folder (sub-*_acq-CTA_ct.nii.gz)",
    )
    p.add_argument(
        "--output-dir",
        default="/media/fridmans/b202ad4e-785a-49f0-a418-ec73cd117466/datasets/daylightbids/derivatives/defaced",
        help="Output folder for defaced NIfTI files",
    )
    p.add_argument(
        "--mask-dir",
        default="/media/fridmans/b202ad4e-785a-49f0-a418-ec73cd117466/datasets/daylightbids/derivatives/deface_masks",
        help="Output folder for predicted face masks",
    )
    p.add_argument(
        "--model-dir",
        default="/home/fridmans/Documents/pwd/AI_CTA_Stroke/external/CTA-DEFACE/model",
        help="CTA-DEFACE model root containing Dataset001_DEFACE",
    )
    p.add_argument(
        "--nnunet-bin",
        default="/home/fridmans/AI/ai-env/bin/nnUNetv2_predict",
        help="Path to nnUNetv2_predict binary",
    )
    p.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU ids to use (example: 0,1)",
    )
    p.add_argument(
        "--glob",
        default="sub-*_acq-CTA_ct.nii.gz",
        help="Input filename glob",
    )
    p.add_argument("--subject", action="append", default=[], help="Optional subject id(s), repeatable")
    p.add_argument("--limit", type=int, default=None, help="Optional limit of pending cases")
    p.add_argument("--force", action="store_true", help="Recompute even if output exists")
    p.add_argument("--pydeface-bin", default="pydeface", help="pydeface CLI binary for --backend pydeface")
    p.add_argument(
        "--pydeface-extra-arg",
        action="append",
        default=[],
        help="Extra argument(s) passed to pydeface (repeatable)",
    )
    p.add_argument(
        "--tmp-root",
        default="/home/fridmans/Documents/pwd/AI_CTA_Stroke/tmp/cta_deface_batch",
        help="Temporary workspace root",
    )
    return p.parse_args()


def build_pending(input_dir: Path, output_dir: Path, pattern: str, force: bool) -> list[Path]:
    inputs = sorted(input_dir.glob(pattern))
    pending = []
    for p in inputs:
        case = strip_nii_suffix(p.name)
        out = output_dir / f"{case}_defaced.nii.gz"
        if force or not out.exists():
            pending.append(p)
    return pending


def filter_pending_by_subject(pending: list[Path], subjects: list[str]) -> list[Path]:
    if not subjects:
        return pending
    wanted = {str(int(s)) for s in subjects if str(s).isdigit()}
    if not wanted:
        return pending
    out: list[Path] = []
    for p in pending:
        case = strip_nii_suffix(p.name)
        sid = case.split("_")[0].replace("sub-", "")
        if sid.isdigit() and str(int(sid)) in wanted:
            out.append(p)
    return out


def split_round_robin(items: list[Path], n: int) -> list[list[Path]]:
    groups: list[list[Path]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        groups[i % n].append(it)
    return groups


def prepare_worker_dirs(tmp_root: Path, gpu: str) -> tuple[Path, Path]:
    inp = tmp_root / f"gpu{gpu}" / "input"
    out = tmp_root / f"gpu{gpu}" / "pred"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for d in [inp, out]:
        for f in d.glob("*"):
            if f.is_file() or f.is_symlink():
                f.unlink()
    return inp, out


def launch_workers(
    workers: list[Worker],
    nnunet_bin: Path,
    model_dir: Path,
) -> None:
    for w in workers:
        if not w.cases:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = w.gpu
        env["nnUNet_results"] = str(model_dir)
        env["nnUNet_preprocessed"] = str(model_dir)
        env["nnUNet_raw"] = str(model_dir)

        cmd = [
            str(nnunet_bin),
            "-i",
            str(w.input_dir),
            "-o",
            str(w.pred_dir),
            "-d",
            "001",
            "-c",
            "3d_fullres",
            "-f",
            "all",
            "--disable_tta",
            "-device",
            "cuda",
            "--continue_prediction",
        ]

        logf = w.log_path.open("w", encoding="utf-8")
        w.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        print(f"[LAUNCHED] GPU {w.gpu}: {len(w.cases)} cases -> {w.log_path}")


def wait_workers(workers: list[Worker]) -> dict[str, int]:
    codes: dict[str, int] = {}
    for w in workers:
        if w.proc is None:
            continue
        rc = w.proc.wait()
        codes[w.gpu] = rc
        print(f"[DONE] GPU {w.gpu} exit={rc}")
    return codes


def create_defaced_and_masks(
    workers: list[Worker],
    output_dir: Path,
    mask_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for w in workers:
        if not w.cases:
            continue
        case_map = {strip_nii_suffix(p.name): p for p in w.cases}
        for pred in sorted(w.pred_dir.glob("*.nii.gz")):
            case_id = strip_nii_suffix(pred.name)
            if case_id not in case_map:
                continue

            src = case_map[case_id]
            mask_img = nib.load(str(pred))
            mask = mask_img.get_fdata() > 0.5

            src_img = nib.load(str(src))
            src_data = src_img.get_fdata()
            p10 = float(np.percentile(src_data, 10))
            defaced = np.where(mask, p10, src_data)

            out_mask = mask_dir / f"{case_id}_mask.nii.gz"
            out_defaced = output_dir / f"{case_id}_defaced.nii.gz"

            nib.save(nib.Nifti1Image(mask.astype(np.uint8), src_img.affine, src_img.header), str(out_mask))
            nib.save(nib.Nifti1Image(defaced.astype(np.float32), src_img.affine, src_img.header), str(out_defaced))

            rows.append(
                {
                    "case_id": case_id,
                    "input": str(src),
                    "mask": str(out_mask),
                    "defaced": str(out_defaced),
                    "fill_value": f"{p10:.6f}",
                    "gpu_worker": w.gpu,
                    "status": "success",
                }
            )

    return rows


def run_pydeface_backend(
    pending: list[Path],
    output_dir: Path,
    tmp_root: Path,
    run_ts: str,
    pydeface_bin: str,
    force: bool,
    extra_args: list[str],
) -> int:
    resolved = shutil.which(pydeface_bin) if "/" not in pydeface_bin else pydeface_bin
    if not resolved or not Path(resolved).exists():
        raise FileNotFoundError(
            f"pydeface binary not found: {pydeface_bin}. "
            "Install pydeface or set --pydeface-bin."
        )

    rows: list[dict] = []
    for src in pending:
        case_id = strip_nii_suffix(src.name)
        out_defaced = output_dir / f"{case_id}_defaced.nii.gz"
        log_path = tmp_root / f"pydeface_{case_id}_{run_ts}.log"
        cmd = [resolved, str(src), "--outfile", str(out_defaced)]
        if force:
            cmd.append("--force")
        if extra_args:
            cmd.extend(extra_args)

        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, check=False)

        status = "success" if (proc.returncode == 0 and out_defaced.exists()) else "failure"
        rows.append(
            {
                "case_id": case_id,
                "input": str(src),
                "mask": "",
                "defaced": str(out_defaced) if out_defaced.exists() else "",
                "fill_value": "",
                "gpu_worker": "",
                "status": status,
                "return_code": proc.returncode,
                "command": " ".join(cmd),
                "log_path": str(log_path),
                "error_message": "" if status == "success" else f"pydeface exit={proc.returncode}",
            }
        )

    summary = output_dir / f"pydeface_summary_{run_ts}.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "input",
                "mask",
                "defaced",
                "fill_value",
                "gpu_worker",
                "status",
                "return_code",
                "command",
                "log_path",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_success = sum(1 for r in rows if r["status"] == "success")
    n_fail = len(rows) - n_success
    print("---")
    print(f"PyDeface success: {n_success}")
    print(f"PyDeface failures: {n_fail}")
    print(f"Summary: {summary}")
    return 0 if n_fail == 0 else 2


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir)
    model_dir = Path(args.model_dir)
    nnunet_bin = Path(args.nnunet_bin)
    tmp_root = Path(args.tmp_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    pending = build_pending(input_dir, output_dir, args.glob, args.force)
    pending = filter_pending_by_subject(pending, args.subject)
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Total pending CTA files: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return 0

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.backend == "pydeface":
        return run_pydeface_backend(
            pending=pending,
            output_dir=output_dir,
            tmp_root=tmp_root,
            run_ts=run_ts,
            pydeface_bin=args.pydeface_bin,
            force=args.force,
            extra_args=args.pydeface_extra_arg,
        )

    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if not nnunet_bin.exists():
        raise FileNotFoundError(f"nnUNet binary not found: {nnunet_bin}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPU IDs provided in --gpus")

    groups = split_round_robin(pending, len(gpus))

    workers: list[Worker] = []
    for gpu, group in zip(gpus, groups):
        inp_dir, pred_dir = prepare_worker_dirs(tmp_root, gpu)
        for src in group:
            case_id = strip_nii_suffix(src.name)
            link = inp_dir / f"{case_id}_0000.nii.gz"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src)
        workers.append(
            Worker(
                gpu=gpu,
                input_dir=inp_dir,
                pred_dir=pred_dir,
                log_path=tmp_root / f"cta_deface_gpu{gpu}_{run_ts}.log",
                cases=group,
            )
        )

    launch_workers(workers, nnunet_bin, model_dir)
    rc_map = wait_workers(workers)

    any_fail = any(code != 0 for code in rc_map.values())
    if any_fail:
        print("One or more GPU workers failed. Check logs under:", tmp_root)
        return 1

    rows = create_defaced_and_masks(workers, output_dir, mask_dir)
    ok_cases = {r["case_id"] for r in rows}
    for src in pending:
        cid = strip_nii_suffix(src.name)
        if cid not in ok_cases:
            rows.append(
                {
                    "case_id": cid,
                    "input": str(src),
                    "mask": "",
                    "defaced": "",
                    "fill_value": "",
                    "gpu_worker": "",
                    "status": "missing_prediction",
                }
            )

    summary = output_dir / f"cta_deface_summary_{run_ts}.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "input", "mask", "defaced", "fill_value", "gpu_worker", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_success = sum(1 for r in rows if r["status"] == "success")
    n_missing = sum(1 for r in rows if r["status"] != "success")
    print("---")
    print(f"Defaced success: {n_success}")
    print(f"Missing predictions: {n_missing}")
    print(f"Summary: {summary}")
    return 0 if n_missing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
