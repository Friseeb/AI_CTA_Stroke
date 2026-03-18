"""
Batch LAA Shape Descriptors Script
SLAAO Project - AI_CTA_Stroke
Processes only the 83 successful eCTA cases
"""

import subprocess
import sys
import os
import concurrent.futures
from datetime import datetime

# ── YOUR 83 SUCCESSFUL CASES ───────────────────────────────────────────────────
SUBJECTS = [
    "sub-101", "sub-547", "sub-657", "sub-658", "sub-659",
    "sub-663", "sub-664", "sub-666", "sub-669", "sub-670",
    "sub-673", "sub-674", "sub-676", "sub-677", "sub-685",
    "sub-686", "sub-690", "sub-694", "sub-695", "sub-696",
    "sub-697", "sub-698", "sub-700", "sub-701", "sub-702",
    "sub-706", "sub-707", "sub-710", "sub-713", "sub-714",
    "sub-716", "sub-719", "sub-723", "sub-725", "sub-727",
    "sub-729", "sub-730", "sub-731", "sub-732", "sub-734",
    "sub-736", "sub-737", "sub-740", "sub-741", "sub-743",
    "sub-744", "sub-747", "sub-748", "sub-751", "sub-752",
    "sub-753", "sub-754", "sub-756", "sub-758", "sub-766",
    "sub-769", "sub-771", "sub-778", "sub-780", "sub-781",
    "sub-783", "sub-788", "sub-791", "sub-792", "sub-793",
    "sub-794", "sub-796", "sub-797", "sub-799", "sub-800",
    "sub-801", "sub-804", "sub-806", "sub-807", "sub-809",
    "sub-811", "sub-813", "sub-814", "sub-819", "sub-821",
    "sub-822", "sub-825", "sub-830"
]

# ── PATHS ──────────────────────────────────────────────────────────────────────
BIDS_ROOT    = r"C:\Users\spost\Desktop\CT_image\daylightbids"
PROJECT_ROOT = r"C:\Users\spost\Desktop\AI_CTA_Stroke-main"
MASK_ROOT    = rf"{BIDS_ROOT}\derivatives\nudf_la_eCTA"
OUTPUT_DIR   = rf"{BIDS_ROOT}\derivatives\shape_meshes_eCTA"
SCRIPT       = rf"{PROJECT_ROOT}\scripts\run_laa_shape_descriptors.py"
WORKERS      = max(1, (os.cpu_count() or 4) // 2)

print(f"\n{'='*60}")
print(f"  SLAAO Batch Shape Descriptors - eCTA Cohort")
print(f"  Processing {len(SUBJECTS)} cases")
print(f"  Workers: {WORKERS}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")


def _run_shape(subject: str) -> tuple:
    input_file = os.path.join(MASK_ROOT, subject, f"{subject}_laa_nudf.nii.gz")
    output_dir = os.path.join(OUTPUT_DIR, subject)

    if not os.path.exists(input_file):
        return (subject, False, "file not found")

    cmd = [
        "python", SCRIPT,
        "--input",      input_file,
        "--output-dir", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return (subject, True, "Done")
    else:
        return (subject, False, result.stderr[-200:])


success_list = []
failed_list  = []

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(_run_shape, s): s for s in SUBJECTS}
    done = 0
    for fut in concurrent.futures.as_completed(futures):
        done += 1
        subject, ok, msg = fut.result()
        if ok:
            print(f"[{done}/{len(SUBJECTS)}] ✅ {subject}: {msg}")
            success_list.append(subject)
        else:
            print(f"[{done}/{len(SUBJECTS)}] ❌ {subject}: {msg}")
            failed_list.append(subject)

print(f"\n{'='*60}")
print(f"  BATCH COMPLETE")
print(f"  ✅ Success: {len(success_list)}/{len(SUBJECTS)}")
print(f"  ❌ Failed:  {len(failed_list)}/{len(SUBJECTS)}")
if failed_list:
    print(f"  Failed: {failed_list}")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")
