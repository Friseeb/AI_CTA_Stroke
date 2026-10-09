#!/usr/bin/env python3
"""Serial, resumable batch over staged CTAs (never point this at an external drive).

  python run_valve_calcium_batch.py --input-dir outputs/daylight_evt_extended --glob 'sub-*_acq-CTA_ct.nii.gz' \
      --out-root outputs/valve_calcium [--limit N] [--no-coronary]

Writes <out-root>/batch_status.csv and <out-root>/valve_calcium_features_all.csv after every case.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, time, traceback
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--glob", default="sub-*_acq-CTA_ct.nii.gz")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no-coronary", action="store_true")
    ap.add_argument("--no-stopstorm", action="store_true")
    ap.add_argument("--force-analysis", action="store_true")
    ap.add_argument("--redo-ok", action="store_true", help="re-run cases already marked ok (e.g. to add STOPSTORM regions)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    ctas = sorted(a.input_dir.glob(a.glob))[: a.limit]
    a.out_root.mkdir(parents=True, exist_ok=True)
    status_path = a.out_root / "batch_status.csv"
    done = set()
    if status_path.is_file() and not a.redo_ok:
        done = {r["case_id"] for r in csv.DictReader(status_path.open()) if r["status"] == "ok"}
    for cta in ctas:
        case_id = cta.name.replace(".nii.gz", "")
        if case_id in done:
            continue
        t0 = time.time()
        cmd = [a.python, str(HERE / "run_valve_calcium_case.py"), "--cta", str(cta), "--out-root", str(a.out_root), "--device", a.device]
        for flag in ("no_coronary", "no_stopstorm", "force_analysis"):
            if getattr(a, flag):
                cmd.append("--" + flag.replace("_", "-"))
        r = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "failed"
        err = "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else "unknown"
        new = not status_path.is_file()
        with status_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["case_id", "status", "seconds", "error"])
            w.writerow([case_id, status, f"{time.time() - t0:.0f}", err])
        print(f"{case_id}: {status} ({time.time() - t0:.0f}s) {err}", flush=True)
        feats = list(a.out_root.glob("*/analysis/*_valve_calcium_features.csv"))
        if feats:
            pd.concat([pd.read_csv(f) for f in feats]).to_csv(a.out_root / "valve_calcium_features_all.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
