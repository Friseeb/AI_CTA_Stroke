#!/usr/bin/env python3
"""
Batch VISTA3D LAA segmentation (label 108) for DAYLIGHTBIDS CTA volumes.

For each *_defaced.nii.gz in derivatives/defaced/ it runs:
    run_nv_segment_ct_laa.py --label-id 108
and saves:
    derivatives/nudf_la_eCTA/<case_id>/<case_id>_laa_vista3d.nii.gz

Skips cases where the output already exists.
Writes a summary CSV at derivatives/nudf_la_eCTA/vista3d_laa_summary.csv.

Example:
  conda run -n cardiac-ct-explorer python scripts/run_vista3d_laa_batch.py \\
    --root "C:/Users/spost/Desktop/CT_image/daylightbids" \\
    --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch VISTA3D LAA (label 108) segmentation for DAYLIGHTBIDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default=".", help="BIDS root directory")
    p.add_argument("--out-dir", default=None, help="Output base dir (default: root/derivatives/nudf_la_eCTA)")
    p.add_argument("--python", default=None, help="Python executable; if omitted, uses conda run -n <conda-env>")
    p.add_argument("--conda-env", default="cardiac-ct-explorer", help="Conda env to use when --python is not set")
    p.add_argument("--model-dir", default=None, help="NV-Segment-CT model dir (default: repo/external/nv_segment_ct)")
    p.add_argument("--device", default="auto", help="Device: auto|cpu|cuda:0")
    p.add_argument("--force", action="store_true", help="Recompute even if output already exists")
    p.add_argument("--limit", type=int, default=None, help="Process at most N cases")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return p.parse_args()


def _scan_id(path: Path) -> str:
    """Strip .nii.gz and _defaced suffix to get the canonical case ID."""
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_defaced"):
        name = name[: -len("_defaced")]
    return name


def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: Root not found: {root}", file=sys.stderr)
        return 1

    input_dir = root / "derivatives" / "defaced"
    if not input_dir.exists():
        print(f"ERROR: Input dir not found: {input_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "derivatives" / "nudf_la_eCTA"
    out_dir.mkdir(parents=True, exist_ok=True)

    laa_script = Path(__file__).resolve().parent / "run_nv_segment_ct_laa.py"
    if not laa_script.exists():
        print(f"ERROR: Runner script not found: {laa_script}", file=sys.stderr)
        return 1

    model_dir = args.model_dir or str(Path(__file__).resolve().parents[1] / "external" / "nv_segment_ct")

    if args.python:
        python_prefix = [args.python]
    else:
        python_prefix = ["conda", "run", "-n", args.conda_env, "python"]

    input_files = sorted(input_dir.glob("*_defaced.nii.gz"))
    if not input_files:
        print(f"ERROR: No *_defaced.nii.gz files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(input_files)} input file(s) in {input_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Model dir:  {model_dir}")
    print(f"Device:     {args.device}")
    if args.dry_run:
        print("DRY RUN — no commands will be executed\n")

    summary_path = out_dir / "vista3d_laa_summary.csv"
    fieldnames = ["case_id", "input_path", "output_path", "elapsed_sec", "status", "message"]
    summary_rows: list[dict] = []

    progress_enabled = tqdm is not None
    loop_iter: enumerate = enumerate(input_files)
    if progress_enabled:
        loop_iter = enumerate(tqdm(input_files, total=len(input_files), desc="VISTA3D LAA", unit="case"))

    for idx, input_path in loop_iter:
        if args.limit is not None and idx >= args.limit:
            break

        started_at = time.time()
        case_id = _scan_id(input_path)
        case_dir = out_dir / case_id
        output_path = case_dir / f"{case_id}_laa_vista3d.nii.gz"

        # ── Skip already processed ────────────────────────────────────────────
        if not args.force and output_path.exists():
            print(f"[SKIP] {case_id} — output exists")
            summary_rows.append({
                "case_id": case_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "elapsed_sec": f"{time.time() - started_at:.1f}",
                "status": "skipped",
                "message": "output already exists",
            })
            continue

        case_dir.mkdir(parents=True, exist_ok=True)

        cmd = python_prefix + [
            str(laa_script),
            "--input", str(input_path),
            "--output", str(output_path),
            "--label-id", "108",
            "--model-dir", model_dir,
            "--device", args.device,
        ]
        print(f"[RUN] {case_id}")
        print("  " + " ".join(cmd))

        if args.dry_run:
            summary_rows.append({
                "case_id": case_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "elapsed_sec": "0.0",
                "status": "dry_run",
                "message": "",
            })
            continue

        status = "ok"
        message = ""
        try:
            proc = subprocess.run(cmd)

            if proc.returncode != 0:
                status = "failed"
                message = f"exit code {proc.returncode}"
                print(f"[FAIL] {case_id}: exit code {proc.returncode}")
            elif not output_path.exists():
                status = "failed"
                message = "output file not created despite exit 0"
                print(f"[FAIL] {case_id}: output file not created")
            else:
                print(f"[OK]   {case_id}: {output_path}")

        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = str(exc)
            print(f"[FAIL] {case_id}: {message}")

        summary_rows.append({
            "case_id": case_id,
            "input_path": str(input_path),
            "output_path": str(output_path) if output_path.exists() else "",
            "elapsed_sec": f"{time.time() - started_at:.1f}",
            "status": status,
            "message": message,
        })

    # ── Write summary CSV ─────────────────────────────────────────────────────
    if not args.dry_run:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        ok = sum(1 for r in summary_rows if r["status"] == "ok")
        skipped = sum(1 for r in summary_rows if r["status"] == "skipped")
        failed = sum(1 for r in summary_rows if r["status"] == "failed")
        print(f"\nDone. ok={ok}  skipped={skipped}  failed={failed}")
        print(f"Summary: {summary_path}")
    else:
        n = sum(1 for r in summary_rows if r["status"] == "dry_run")
        print(f"\nDry run complete. Would process {n} case(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
