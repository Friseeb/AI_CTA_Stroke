"""
Batch LAA Segmentation Script
SLAAO Project - AI_CTA_Stroke
Processes all defaced CTA scans in the defaced folder
"""

import os
import subprocess
import glob
import concurrent.futures
from pathlib import Path
from datetime import datetime

# ── PATHS ──────────────────────────────────────────────────────────────────────
BIDS_ROOT     = r"C:\Users\spost\Desktop\CT_image\daylightbids"
PROJECT_ROOT  = r"C:\Users\spost\Desktop\AI_CTA_Stroke-main"
DEFACED_DIR   = os.path.join(BIDS_ROOT, "derivatives", "defaced")
OUTPUT_BASE   = os.path.join(BIDS_ROOT, "derivatives", "nudf_la")
SCRIPT        = os.path.join(PROJECT_ROOT, "scripts", "run_cardiac_ct_explorer_nudf_only.py")
LOG_FILE      = os.path.join(PROJECT_ROOT, "batch_log.txt")
WORKERS       = max(1, (os.cpu_count() or 4) // 2)

# ── FIND ALL PATIENTS ──────────────────────────────────────────────────────────
input_files = sorted(glob.glob(os.path.join(DEFACED_DIR, "sub-*_acq-CTA_ct_defaced.nii.gz")))
total = len(input_files)
print(f"\n{'='*60}")
print(f"  SLAAO Batch LAA Segmentation")
print(f"  Found {total} patients to process")
print(f"  Workers: {WORKERS}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")


def _run_case(input_path: str) -> tuple:
    filename   = Path(input_path).name
    case_id    = filename.replace("_acq-CTA_ct_defaced.nii.gz", "")
    output_dir = os.path.join(OUTPUT_BASE, case_id, "cardiac_ct_explorer")
    laa_output = os.path.join(OUTPUT_BASE, case_id, f"{case_id}_laa_nudf.nii.gz")

    cmd = [
        "python", SCRIPT,
        "--input",      input_path,
        "--output-dir", output_dir,
        "--laa-output", laa_output,
        "--run-totalseg"
    ]

    start = datetime.now()
    try:
        result   = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        duration = (datetime.now() - start).seconds
        if result.returncode == 0 and os.path.exists(laa_output):
            return (case_id, True, f"done in {duration}s → {laa_output}", "")
        else:
            err = result.stderr[-300:] if result.stderr else "No error message"
            return (case_id, False, "FAILED", err)
    except subprocess.TimeoutExpired:
        return (case_id, False, "TIMEOUT — exceeded 30 minutes", "")
    except Exception as e:
        return (case_id, False, f"ERROR — {e}", "")


success_list = []
failed_list  = []

with open(LOG_FILE, "w") as log:
    log.write(f"Batch started: {datetime.now()}\n")
    log.write(f"Total patients: {total}\n")
    log.write(f"Workers: {WORKERS}\n\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_run_case, p): p for p in input_files}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            case_id, ok, msg, err = fut.result()
            if ok:
                print(f"[{done}/{total}] ✅ {case_id}: {msg}")
                success_list.append(case_id)
                log.write(f"SUCCESS: {case_id} — {msg}\n")
            else:
                print(f"[{done}/{total}] ❌ {case_id}: {msg}")
                if err:
                    print(f"     {err}")
                failed_list.append(case_id)
                log.write(f"FAILED: {case_id} — {msg}\n{err}\n\n")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BATCH COMPLETE")
print(f"  ✅ Success: {len(success_list)}/{total}")
print(f"  ❌ Failed:  {len(failed_list)}/{total}")
if failed_list:
    print(f"\n  Failed cases:")
    for case in failed_list:
        print(f"    - {case}")
print(f"\n  Log saved to: {LOG_FILE}")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

with open(LOG_FILE, "a") as log:
    log.write(f"\nBatch finished: {datetime.now()}\n")
    log.write(f"Success: {len(success_list)}/{total}\n")
    log.write(f"Failed: {failed_list}\n")
