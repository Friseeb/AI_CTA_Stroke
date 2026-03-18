"""
SLAAO Full LA/LAA Metrics Batch Pipeline
==========================================
For each patient this script:
  1. Extracts LA mask (label 2) from heartchambers_highres.nii.gz
  2. Generates LA surface mesh
  3. Copies both LA and LAA vtk files to the case root (required by metrics script)
  4. Runs LA/LAA relational metrics for all patients

HOW TO ADAPT FOR NEW COHORTS:
- Update BIDS_ROOT and PROJECT_ROOT if paths change
- Update NUDF_ROOT to point to your segmentation folder
- Update MESH_ROOT to point to your mesh output folder
- Update SUBJECTS list with your patient IDs
- LA_LABEL: label number for LA in heartchambers_highres.nii.gz (2 = LA, verified in Slicer)

Author: SLAAO Project
"""

import subprocess
import sys
import shutil
import time
import os
import concurrent.futures
import nibabel as nib
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _fmt(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"

# ── CONFIGURATION (update these for new cohorts) ───────────────────────────────
BIDS_ROOT    = r"C:\Users\spost\Desktop\CT_image\daylightbids"
PROJECT_ROOT = r"C:\Users\spost\Desktop\AI_CTA_Stroke-main"
NUDF_ROOT    = rf"{BIDS_ROOT}\derivatives\nudf_la_eCTA"       # segmentation folder
MESH_ROOT    = rf"{BIDS_ROOT}\derivatives\shape_meshes_eCTA"  # mesh output folder
LA_LABEL     = 2  # label 2 = LA in heartchambers_highres.nii.gz (verified in Slicer)
# Cases where the eCTA FOV does not cover the heart produce an empty or near-empty
# label 2.  Any result below this voxel floor is treated as FOV-excluded and skipped.
MIN_LA_VOXELS = 1_000
WORKERS       = max(1, (os.cpu_count() or 4) // 2)

SHAPE_SCRIPT   = rf"{PROJECT_ROOT}\scripts\run_laa_shape_descriptors.py"
METRICS_SCRIPT = rf"{PROJECT_ROOT}\scripts\run_la_laa_metrics_batch.py"
OUT_CSV        = rf"{MESH_ROOT}\la_laa_metrics_batch.csv"

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

# ── HELPERS ────────────────────────────────────────────────────────────────────
def extract_la_mask(subject: str) -> Path | None:
    """Extract LA mask (label 2) from heartchambers_highres.nii.gz."""
    hc_path  = Path(NUDF_ROOT) / subject / "cardiac_ct_explorer" / "TotalSegmentator" / f"{subject}_acq-CTA_ct_defaced" / "heartchambers_highres.nii.gz"
    out_path = Path(NUDF_ROOT) / subject / f"{subject}_left_atrium_highres.nii.gz"

    if not hc_path.exists():
        return None

    if out_path.exists():
        return out_path

    img  = nib.load(str(hc_path))
    data = img.get_fdata()
    la   = (data == LA_LABEL).astype(np.uint8)
    if int(la.sum()) < MIN_LA_VOXELS:
        return None
    nib.save(nib.Nifti1Image(la, img.affine, img.header), str(out_path))
    return out_path


def generate_mesh(input_nii: Path, output_dir: Path) -> Path | None:
    """Generate surface mesh from NIfTI mask."""
    result = subprocess.run(
        ["python", SHAPE_SCRIPT, "--input", str(input_nii), "--output-dir", str(output_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    stem = input_nii.stem.replace(".nii", "")
    vtk  = output_dir / "surfaces" / f"{stem}_laa_surface.vtk"
    return vtk if vtk.exists() else None


def copy_to_root(vtk_path: Path, case_root: Path) -> Path:
    """Copy vtk file from surfaces/ subfolder to case root (required by metrics script)."""
    dest = case_root / vtk_path.name
    if not dest.exists():
        shutil.copy2(str(vtk_path), str(dest))
    return dest


def _prep_subject(subject: str) -> tuple:
    """Run steps 1–4 for one subject. Returns (subject, ok, log_lines)."""
    lines = []
    mesh_case_dir = Path(MESH_ROOT) / subject

    # Step 1: Extract LA mask
    t = time.perf_counter()
    la_nii = extract_la_mask(subject)
    lines.append(f"  [1/4] LA mask: {'done' if la_nii else 'SKIP/FAIL'}  ({time.perf_counter()-t:.1f}s)")
    if la_nii is None:
        return (subject, False, "\n".join(lines))

    # Step 2: Generate LA mesh
    t = time.perf_counter()
    la_vtk = generate_mesh(la_nii, mesh_case_dir)
    lines.append(f"  [2/4] LA mesh: {'done' if la_vtk else 'FAIL'}  ({time.perf_counter()-t:.1f}s)")
    if la_vtk is None:
        return (subject, False, "\n".join(lines))

    # Step 3: Copy LA vtk to case root
    copy_to_root(la_vtk, mesh_case_dir)
    lines.append(f"  [3/4] LA vtk copied")

    # Step 4: Copy LAA vtk to case root
    laa_vtk_src = mesh_case_dir / "surfaces" / f"{subject}_laa_nudf_laa_surface.vtk"
    if laa_vtk_src.exists():
        copy_to_root(laa_vtk_src, mesh_case_dir)
        lines.append(f"  [4/4] LAA vtk copied")
    else:
        lines.append(f"  [4/4] LAA vtk NOT FOUND  ({laa_vtk_src.name})")
        return (subject, False, "\n".join(lines))

    return (subject, True, "\n".join(lines))


# ── MAIN ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SLAAO Full LA/LAA Metrics Batch Pipeline")
print(f"  Processing {len(SUBJECTS)} cases")
print(f"  Workers: {WORKERS}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

success_la     = []
failed_la      = []
pipeline_start = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(_prep_subject, s): s for s in SUBJECTS}
    done = 0
    for fut in concurrent.futures.as_completed(futures):
        done += 1
        subject, ok, log_output = fut.result()
        elapsed_total = time.perf_counter() - pipeline_start
        print(f"\n[{done}/{len(SUBJECTS)}] {subject}  |  elapsed {_fmt(elapsed_total)}")
        print(log_output)
        if ok:
            print(f"  ✅ case done")
            success_la.append(subject)
        else:
            print(f"  ❌ case FAILED")
            failed_la.append(subject)

# ── Step 5: Run metrics for all patients (must run after all prep is complete) ──
print(f"\n{'='*60}")
print(f"  Running LA/LAA metrics for all patients...")
print(f"{'='*60}\n")

cmd = [
    "python", METRICS_SCRIPT,
    "--mesh-root",  MESH_ROOT,
    "--out-csv",    OUT_CSV,
    "--case-glob",  "sub-*",
    "--la-suffix",  "left_atrium_highres_laa_surface",
    "--laa-suffix", "laa_nudf_laa_surface",
]
subprocess.run(cmd)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  ✅ LA prepared: {len(success_la)}/{len(SUBJECTS)}")
print(f"  ❌ Failed:      {len(failed_la)}/{len(SUBJECTS)}")
if failed_la:
    print(f"  Failed cases: {failed_la}")
print(f"  📄 Metrics CSV: {OUT_CSV}")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")
