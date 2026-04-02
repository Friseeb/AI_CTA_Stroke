#!/usr/bin/env python3
"""
Batch defacing of all eCTA patients in SLAAOBIDS.

Uses TotalSegmentator face ROI subset (fast, lightweight) via deface_cta.py.
Runs subjects in parallel with ProcessPoolExecutor.

Output:
  SLAAOBIDS/derivatives/defaced/sub-XXX_acq-ecta_ct_defaced.nii.gz

Usage:
  PYTHONUTF8=1 /path/to/python.exe scripts/_run_deface_ecta_batch.py [--workers N]
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

SLAAOBIDS   = Path(r"C:/Users/spost/Desktop/CT_image/SLAAOBIDS")
DERIVATIVES = SLAAOBIDS / "derivatives"
OUTPUT_DIR  = DERIVATIVES / "defaced"
SCRIPT_DIR  = Path(__file__).parent


def _deface_one(args: tuple) -> dict:
    """Worker function: deface a single eCTA volume in a subprocess."""
    input_nii, out_path = args
    input_nii = Path(input_nii)
    out_path  = Path(out_path)

    sys.path.insert(0, str(SCRIPT_DIR))
    from deface_cta import deface_volume

    t0 = time.time()
    try:
        stats = deface_volume(
            input_path=input_nii,
            output_path=out_path,
            run_totalseg=True,
            fast_totalseg=True,
            fill_value=-1024.0,
        )
        elapsed = time.time() - t0
        return {
            "subject":     input_nii.parent.name,
            "input":       str(input_nii),
            "output":      str(out_path),
            "status":      "success",
            "elapsed_s":   round(elapsed, 1),
            "face_voxels": stats.get("defaced_voxels", 0),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "subject":   input_nii.parent.name,
            "input":     str(input_nii),
            "output":    str(out_path),
            "status":    "failed",
            "elapsed_s": round(elapsed, 1),
            "error":     str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel batch defacing for all eCTA NIfTI files in SLAAOBIDS"
    )
    parser.add_argument(
        "--workers", type=int,
        default=min(4, max(1, (os.cpu_count() or 4) // 2)),
        help="Parallel workers (default: min(4, cpu_count//2))",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess files that already have a defaced output",
    )
    args = parser.parse_args()

    # Collect all converted eCTA files
    ecta_files = sorted(SLAAOBIDS.glob("sub-*/sub-*_acq-ecta_ct.nii.gz"))
    print(f"Found {len(ecta_files)} eCTA NIfTI files in SLAAOBIDS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build work list, skip already-defaced
    work:   list[tuple[Path, Path]] = []
    n_skip: int = 0
    for inp in ecta_files:
        stem = inp.name[:-7]  # strip .nii.gz
        out  = OUTPUT_DIR / f"{stem}_defaced.nii.gz"
        if out.exists() and not args.force:
            n_skip += 1
        else:
            work.append((inp, out))

    if n_skip:
        print(f"Skipping {n_skip} already-defaced files")
    print(f"Processing {len(work)} files  |  workers={args.workers}")
    print(f"Output dir : {OUTPUT_DIR}")
    print()

    if not work:
        print("Nothing to do.")
        return

    t_start = time.time()
    results: list[dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        # Pass as tuple (picklable on Windows)
        future_map = {
            ex.submit(_deface_one, (str(inp), str(out))): inp.parent.name
            for inp, out in work
        }
        n_done = 0
        for fut in concurrent.futures.as_completed(future_map):
            r = fut.result()
            n_done += 1
            sym = "OK " if r["status"] == "success" else "ERR"
            print(f"  [{n_done:3d}/{len(work)}] {sym}  {r['subject']:12s}  {r['elapsed_s']:6.1f}s")
            results.append(r)

    elapsed_total = time.time() - t_start
    n_ok   = sum(1 for r in results if r["status"] == "success")
    n_fail = len(results) - n_ok

    # Save summary JSON
    summary_path = OUTPUT_DIR / "deface_batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("=" * 52)
    print(f"  Completed in {elapsed_total / 60:.1f} min ({elapsed_total:.0f}s)")
    print(f"  Success : {n_ok}")
    print(f"  Failed  : {n_fail}")
    print(f"  Skipped : {n_skip}")
    print(f"  Summary : {summary_path}")
    print("=" * 52)

    if n_fail:
        print("\nFailed subjects:")
        for r in results:
            if r["status"] != "success":
                print(f"  {r['subject']:12s}  {r.get('error', '?')}")


if __name__ == "__main__":
    main()
