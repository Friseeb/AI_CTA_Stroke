#!/usr/bin/env python3
"""Serial, resumable batch over locally staged CTAs (never point this at an external drive).

  python run_heart_segmentation_batch.py --input-dir outputs/daylight_evt_extended \
      --out-root outputs/heart_segmentation \
      --crop-mask 'outputs/valve_calcium/{case_id}/total_fast/heart.nii.gz'   # reuse existing heart crops

Writes <out-root>/batch_status.csv and <out-root>/holorama_heart_volumes_all.csv after every case.
Every other flag is passed through to run_heart_segmentation_case.py.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASSTHROUGH_FLAGS = ("no_crop", "no_totalseg", "no_structures", "no_qc", "allow_large", "force")
PASSTHROUGH_VALUES = ("weights_dir", "crop_mask", "ts_bin", "ts_device", "crop_margin_mm", "device", "precision", "threads", "tile_step_size", "max_memory_fraction")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--glob", default="sub-*_acq-CTA_ct.nii.gz")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--redo-ok", action="store_true", help="re-run cases already marked ok")
    for name in PASSTHROUGH_VALUES:
        ap.add_argument("--" + name.replace("_", "-"), default=None)
    for name in PASSTHROUGH_FLAGS:
        ap.add_argument("--" + name.replace("_", "-"), action="store_true")
    a = ap.parse_args(argv)

    ctas = sorted(a.input_dir.glob(a.glob))[: a.limit]
    if not ctas:
        raise SystemExit(f"no files match {a.input_dir / a.glob}")
    a.out_root.mkdir(parents=True, exist_ok=True)
    status_path = a.out_root / "batch_status.csv"
    done: set[str] = set()
    if status_path.is_file() and not a.redo_ok:
        done = {r["case_id"] for r in csv.DictReader(status_path.open()) if r["status"] == "ok"}
    for cta in ctas:
        case_id = cta.name.replace(".nii.gz", "")
        if case_id in done:
            continue
        t0 = time.time()
        cmd = [a.python, str(HERE / "run_heart_segmentation_case.py"), "--cta", str(cta), "--out-root", str(a.out_root)]
        for name in PASSTHROUGH_VALUES:
            v = getattr(a, name)
            if v is not None:
                cmd += ["--" + name.replace("_", "-"), str(v)]
        for name in PASSTHROUGH_FLAGS:
            if getattr(a, name):
                cmd.append("--" + name.replace("_", "-"))
        r = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "failed"
        err = ""
        if r.returncode != 0:
            tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
            err = tail[-1][:300] if tail else "unknown"
        new = not status_path.is_file()
        with status_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["case_id", "status", "seconds", "error"])
            w.writerow([case_id, status, f"{time.time() - t0:.0f}", err])
        print(f"{case_id}: {status} ({time.time() - t0:.0f}s) {err}", flush=True)
        aggregate_volumes(a.out_root)
    return 0


def aggregate_volumes(out_root: Path) -> None:
    files = sorted(out_root.glob("*/*_holorama_heart_volumes.csv"))
    if not files:
        return
    rows, header = [], None
    for f in files:
        with f.open() as fh:
            reader = csv.reader(fh)
            h = next(reader)
            header = header or h
            rows.extend(reader)
    with (out_root / "holorama_heart_volumes_all.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
