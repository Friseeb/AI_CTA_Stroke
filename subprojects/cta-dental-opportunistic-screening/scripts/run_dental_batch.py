#!/usr/bin/env python
"""Batch-run the dental opportunistic-screening pipeline over many CTAs.

Wraps the single-case ``cta-dental run`` CLI: for each input NIfTI it runs the
full pipeline into ``<outdir>/<case_id>/`` and records per-case status + wall
time to ``<outdir>/dental_batch_status.csv``. Resumable via --skip-existing
(forwarded to the per-case run, which reuses a completed segmentation).

Example:
  python scripts/run_dental_batch.py \
      --glob '/path/to/slaobids/*_acq-CTA_ct.nii.gz' \
      --outdir outputs/dental_slaobids_pilot \
      --config configs/pilot_mps.yaml \
      --limit 3 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Dental CLI: CTA_DENTAL_BIN env override, else resolve `cta-dental` on PATH
# (run inside the env that has TotalSegmentator, or pass --cta-dental).
DEFAULT_CTA_DENTAL = os.environ.get("CTA_DENTAL_BIN") or "cta-dental"


def case_id_from_path(p: Path) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+)", p.name)
    return m.group(1) if m else p.name.split(".")[0]


def _thread_capped_env(threads: int) -> dict:
    """Limit each worker's math/nnU-Net threads so N workers don't oversubscribe."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env["nnUNet_n_proc_DA"] = str(threads)
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--glob", help="Glob of input NIfTI CTAs.")
    src.add_argument("--manifest", help="Text file with one input path per line.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default=None, help="Pipeline config yaml (e.g. device: mps).")
    ap.add_argument("--segmenter", default="totalseg_teeth")
    ap.add_argument("--roi-method", default="totalseg_teeth")
    ap.add_argument("--deface-mode", default="mask_only")
    ap.add_argument("--target-spacing", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N inputs.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--reuse-roi-seg", action="store_true",
                    help="Reuse ROI-detection labels as the final segmentation (~2x faster).")
    ap.add_argument("--slim", action="store_true",
                    help="After each successful case, delete the redundant intermediates "
                         "(preprocessed.nii.gz + roi/_roi_input.nii.gz) to save disk.")
    ap.add_argument("--cta-dental", default=DEFAULT_CTA_DENTAL, help="Path to the cta-dental CLI.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Run this many cases concurrently (CPU device recommended; MPS is a "
                         "single GPU and won't parallelize TotalSegmentator).")
    ap.add_argument("--threads-per-worker", type=int, default=None,
                    help="Math/nnU-Net threads per worker (default: 12 // workers, min 1).")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-case timeout (s). A stuck case (e.g. a dropped /Volumes mount) "
                         "is killed and marked 'timeout' so it can't hang the batch; "
                         "--skip-existing resumes it later. 0 disables.")
    args = ap.parse_args()
    threads = args.threads_per_worker or max(1, 12 // max(1, args.workers))

    if args.glob:
        inputs = [Path(p) for p in sorted(globmod.glob(args.glob))]
    else:
        inputs = [Path(line.strip()) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit("No inputs matched.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status_csv = outdir / "dental_batch_status.csv"
    write_header = not status_csv.exists()

    n = len(inputs)
    env = _thread_capped_env(threads)
    print(f"Dental batch: {n} case(s) -> {outdir} | workers={args.workers} "
          f"threads/worker={threads}", flush=True)

    done = {"i": 0}
    lock = threading.Lock()
    fh = status_csv.open("a", newline="")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["case_id", "input", "status", "returncode", "seconds", "out_dir"])
        fh.flush()

    def run_one(cta: Path):
        case_id = case_id_from_path(cta)
        case_out = outdir / case_id
        if not cta.exists():
            return [case_id, str(cta), "missing_input", "", "", str(case_out)]
        cmd = [
            args.cta_dental, "run", str(cta), "--out", str(case_out), "--case-id", case_id,
            "--segmenter", args.segmenter, "--roi-method", args.roi_method,
            "--deface-mode", args.deface_mode, "--target-spacing", str(args.target_spacing),
        ]
        if args.config:
            cmd += ["--config", args.config]
        if args.skip_existing:
            cmd += ["--skip-existing"]
        if args.reuse_roi_seg:
            cmd += ["--reuse-roi-seg"]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  timeout=args.timeout or None)
            rc, stderr, stdout = proc.returncode, proc.stderr, proc.stdout
        except subprocess.TimeoutExpired as exc:
            # A single stuck case (e.g. a dropped /Volumes mount) must not hang the
            # whole batch. Mark it timeout and move on; --skip-existing resumes it.
            rc, stderr, stdout = 124, f"timeout after {args.timeout}s", (exc.stdout or "")
        dt = time.time() - t0
        status = "ok" if rc == 0 else ("timeout" if rc == 124 else "failed")
        if args.slim and rc == 0:
            for redundant in (case_out / "preprocessed.nii.gz", case_out / "roi" / "_roi_input.nii.gz"):
                try:
                    redundant.unlink(missing_ok=True)
                except OSError:
                    pass
        with lock:
            done["i"] += 1
            prefix = f"[{done['i']}/{n}] {case_id}"
            if rc != 0:
                tail = "\n".join((stderr or stdout or "").strip().splitlines()[-8:])
                print(f"{prefix}: {status.upper()} (rc={rc}, {dt:.0f}s)\n{tail}", flush=True)
            else:
                print(f"{prefix}: ok in {dt:.0f}s", flush=True)
        return [case_id, str(cta), status, rc, f"{dt:.1f}", str(case_out)]

    try:
        if args.workers <= 1:
            for cta in inputs:
                writer.writerow(run_one(cta)); fh.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(run_one, cta) for cta in inputs]
                for fut in as_completed(futures):
                    with lock:
                        writer.writerow(fut.result()); fh.flush()
    finally:
        fh.close()

    print(f"Done. Status: {status_csv}")


if __name__ == "__main__":
    main()
