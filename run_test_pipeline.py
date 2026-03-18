"""
LA/LAA Full Pipeline — Test Run
================================
Runs all 5 pipeline stages on a small test cohort:

  Step 0  Deface CTA          → derivatives/defaced/
  Step 1  LAA segmentation    → derivatives/nudf_la/
  Step 2  Mesh generation     → derivatives/shape_meshes/
  Step 3  Relational metrics  → la_laa_metrics_batch.csv
  Step 4  HTML report         → la_laa_shape_report.html

HOW TO USE
----------
1. Set BIDS_ROOT and SUBJECTS below.
2. Run with the base conda environment (or any env that has nibabel/numpy):
       python run_test_pipeline.py
   The script invokes the correct sub-environment for each step automatically.

ENVIRONMENTS
------------
  cardiac-ct-explorer  →  Steps 0 and 1  (TotalSegmentator + CardiacCTExplorer)
  laa-shape            →  Steps 2, 3, 4  (trimesh / VTK / matplotlib)
"""

from __future__ import annotations

import subprocess
import sys
import time
import csv
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BIDS_ROOT    = Path(r"C:\Users\spost\Desktop\CT_IMAGE_new\daylightbids")
PROJECT_ROOT = Path(r"C:\Users\spost\Desktop\AI_CTA_Stroke-main")

DEFACED_DIR  = BIDS_ROOT / "derivatives" / "defaced"
NUDF_DIR     = BIDS_ROOT / "derivatives" / "nudf_la"
MESH_DIR     = BIDS_ROOT / "derivatives" / "shape_meshes"
OUT_CSV      = MESH_DIR / "la_laa_metrics_batch.csv"
OUT_HTML     = MESH_DIR / "la_laa_shape_report.html"

# Python executables for each conda environment
PYTHON_CCE   = r"C:\Users\spost\miniconda3\envs\cardiac-ct-explorer\python.exe"
PYTHON_SHAPE = r"C:\Users\spost\miniconda3\envs\laa-shape\python.exe"

# Label IDs in heartchambers_highres.nii.gz (TotalSegmentator)
LA_LABEL_ID    = 2
MIN_LA_VOXELS  = 1_000  # skip cases where heart is outside FOV

# Test subjects — auto-discovered if left empty
SUBJECTS: list[str] = []   # e.g. ["sub-6", "sub-12"] or leave [] for auto-discovery

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def _run(label: str, cmd: list[str], log_path: Path) -> bool:
    """Run a subprocess, tee stdout/stderr to a log file. Returns True on success."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n{'='*60}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"    FAIL (rc={proc.returncode}) — see {log_path.name}")
    return proc.returncode == 0


def _find_heartchambers(case_dir: Path) -> Path | None:
    matches = list(case_dir.glob("**/heartchambers_highres.nii.gz"))
    if not matches:
        return None
    return sorted(matches, key=lambda p: len(p.parts))[0]


def _extract_la_mask(hc_path: Path, out_path: Path) -> int:
    """Extract label 2 from heartchambers_highres → binary LA mask. Returns voxel count."""
    img  = nib.load(str(hc_path))
    data = np.asanyarray(img.dataobj)
    la   = (data == LA_LABEL_ID).astype(np.uint8)
    vox  = int(la.sum())
    if vox >= MIN_LA_VOXELS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(la, img.affine, img.header), str(out_path))
    return vox


# ── DISCOVER SUBJECTS ─────────────────────────────────────────────────────────
if not SUBJECTS:
    SUBJECTS = sorted(
        p.name.replace("_acq-CTA_ct.nii.gz", "")
        for p in BIDS_ROOT.glob("sub-*_acq-CTA_ct.nii.gz")
    )

if not SUBJECTS:
    print("ERROR: No sub-*_acq-CTA_ct.nii.gz files found under", BIDS_ROOT)
    sys.exit(1)

# ── BANNER ────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  LA/LAA Test Pipeline")
print(f"  {len(SUBJECTS)} cases: {', '.join(SUBJECTS)}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*62}\n")

log_dir      = BIDS_ROOT / "derivatives" / "_test_logs"
pipeline_t0  = time.perf_counter()
results: list[dict] = []

use_tqdm  = _tqdm is not None and sys.stdout.isatty()
case_iter = _tqdm(SUBJECTS, unit="case", desc="Pipeline") if use_tqdm else SUBJECTS

# ══════════════════════════════════════════════════════════════════════════════
# PER-CASE STEPS  (0 → 2)
# ══════════════════════════════════════════════════════════════════════════════
for i, subject in enumerate(case_iter if use_tqdm else SUBJECTS, 1):
    case_t0      = time.perf_counter()
    elapsed      = time.perf_counter() - pipeline_t0
    done         = i - 1
    eta_str      = _fmt(elapsed / done * (len(SUBJECTS) - done)) if done > 0 else "?"
    row          = {"subject": subject, "status": "ok", "note": ""}

    print(f"\n{'─'*62}")
    print(f"  [{i}/{len(SUBJECTS)}] {subject}  |  elapsed {_fmt(elapsed)}  ETA {eta_str}")
    print(f"{'─'*62}")

    raw_nii      = BIDS_ROOT / f"{subject}_acq-CTA_ct.nii.gz"
    defaced_nii  = DEFACED_DIR / f"{subject}_acq-CTA_ct_defaced.nii.gz"
    case_nudf    = NUDF_DIR / subject
    laa_nii      = case_nudf / f"{subject}_laa_nudf.nii.gz"
    la_nii       = case_nudf / f"{subject}_left_atrium_highres.nii.gz"
    mesh_case    = MESH_DIR / subject

    # ── Step 0: Deface ────────────────────────────────────────────────────────
    t = time.perf_counter()
    print(f"  [0/4] Defacing...", end="  ", flush=True)
    if defaced_nii.exists():
        print(f"already done  ({_fmt(time.perf_counter()-t)})")
    else:
        ok = _run(
            "deface",
            [PYTHON_CCE, str(PROJECT_ROOT / "scripts" / "deface_cta.py"),
             "--input",  str(raw_nii),
             "--output", str(defaced_nii),
             "--run-totalseg"],
            log_dir / f"{subject}_step0_deface.log",
        )
        print(f"{'done' if ok else 'FAIL'}  ({_fmt(time.perf_counter()-t)})")
        if not ok:
            row.update(status="failed", note="deface failed")
            results.append(row)
            continue

    # ── Step 1: LAA segmentation (NUDF + TotalSegmentator heartchambers) ─────
    t = time.perf_counter()
    print(f"  [1/4] NUDF segmentation...", end="  ", flush=True)
    if laa_nii.exists():
        print(f"already done  ({_fmt(time.perf_counter()-t)})")
    else:
        ok = _run(
            "nudf",
            [PYTHON_CCE, str(PROJECT_ROOT / "scripts" / "run_cardiac_ct_explorer_nudf_only.py"),
             "--input",      str(defaced_nii),
             "--output-dir", str(case_nudf / "cardiac_ct_explorer"),
             "--laa-output", str(laa_nii),
             "--run-totalseg"],
            log_dir / f"{subject}_step1_nudf.log",
        )
        print(f"{'done' if ok else 'FAIL'}  ({_fmt(time.perf_counter()-t)})")
        if not ok:
            row.update(status="failed", note="NUDF segmentation failed")
            results.append(row)
            continue

    # ── Step 1b: Extract LA mask (label 2) from heartchambers_highres ─────────
    t = time.perf_counter()
    print(f"  [1b/4] Extracting LA mask...", end="  ", flush=True)
    if la_nii.exists():
        print(f"already done  ({_fmt(time.perf_counter()-t)})")
    else:
        hc_path = _find_heartchambers(case_nudf)
        if hc_path is None:
            print(f"FAIL — heartchambers_highres.nii.gz not found")
            row.update(status="failed", note="heartchambers not found")
            results.append(row)
            continue
        vox = _extract_la_mask(hc_path, la_nii)
        if vox < MIN_LA_VOXELS:
            print(f"SKIP — only {vox} voxels (heart outside FOV)")
            row.update(status="skip_la_fov", note=f"la_voxels={vox}")
            results.append(row)
            continue
        print(f"done  ({vox:,} voxels, {_fmt(time.perf_counter()-t)})")

    # ── Step 2: Mesh generation ───────────────────────────────────────────────
    for mask_tag, mask_path in [("LA", la_nii), ("LAA", laa_nii)]:
        t = time.perf_counter()
        print(f"  [2/4] {mask_tag} mesh...", end="  ", flush=True)
        # Check if mesh already exists (in surfaces/ or case root)
        stem     = mask_path.stem.replace(".nii", "")
        vtk_surf = mesh_case / "surfaces" / f"{stem}_laa_surface.vtk"
        vtk_root = mesh_case / f"{stem}_laa_surface.vtk"
        if vtk_surf.exists() or vtk_root.exists():
            print(f"already done  ({_fmt(time.perf_counter()-t)})")
            continue
        ok = _run(
            f"mesh_{mask_tag}",
            [PYTHON_SHAPE, str(PROJECT_ROOT / "scripts" / "run_laa_shape_descriptors.py"),
             "--input",      str(mask_path),
             "--output-dir", str(mesh_case)],
            log_dir / f"{subject}_step2_mesh_{mask_tag.lower()}.log",
        )
        print(f"{'done' if ok else 'FAIL'}  ({_fmt(time.perf_counter()-t)})")
        if not ok:
            row.update(status="failed", note=f"{mask_tag} mesh failed")

    if row["status"] == "failed":
        results.append(row)
        continue

    case_elapsed = time.perf_counter() - case_t0
    print(f"  ✅ case done  ({_fmt(case_elapsed)})")
    results.append(row)
    if use_tqdm:
        ok_n   = sum(1 for r in results if r["status"] == "ok")
        fail_n = sum(1 for r in results if r["status"] not in ("ok", "skip_la_fov"))
        case_iter.set_postfix(ok=ok_n, fail=fail_n)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: BATCH METRICS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*62}")
print(f"  [3] Batch LA/LAA metrics...")
print(f"{'─'*62}")
t = time.perf_counter()
ok = _run(
    "metrics",
    [PYTHON_SHAPE, str(PROJECT_ROOT / "scripts" / "run_la_laa_metrics_batch.py"),
     "--mesh-root",  str(MESH_DIR),
     "--out-csv",    str(OUT_CSV),
     "--case-glob",  "sub-*",
     "--la-suffix",  "left_atrium_highres_laa_surface",
     "--laa-suffix", "laa_nudf_laa_surface"],
    log_dir / "step3_metrics.log",
)
print(f"  {'✅ done' if ok else '❌ FAIL'}  ({_fmt(time.perf_counter()-t)})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*62}")
print(f"  [4] Generating HTML report...")
print(f"{'─'*62}")
t = time.perf_counter()
ok = _run(
    "report",
    [PYTHON_SHAPE, str(PROJECT_ROOT / "scripts" / "generate_la_laa_shape_report.py"),
     "--metrics-csv",  str(OUT_CSV),
     "--mesh-root",    str(MESH_DIR),
     "--output-html",  str(OUT_HTML)],
    log_dir / "step4_report.log",
)
print(f"  {'✅ done' if ok else '❌ FAIL'}  ({_fmt(time.perf_counter()-t)})")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total_elapsed = time.perf_counter() - pipeline_t0
ok_cases      = [r["subject"] for r in results if r["status"] == "ok"]
fov_skip      = [r["subject"] for r in results if r["status"] == "skip_la_fov"]
failed_cases  = [r["subject"] for r in results if r["status"] == "failed"]

# Write per-case summary CSV
summary_csv = BIDS_ROOT / "derivatives" / "test_pipeline_summary.csv"
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["subject", "status", "note"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n{'='*62}")
print(f"  TEST PIPELINE COMPLETE  ({_fmt(total_elapsed)})")
print(f"  ✅ Success:          {len(ok_cases)}/{len(SUBJECTS)}  {ok_cases}")
if fov_skip:
    print(f"  ⚠️  FOV excluded:    {len(fov_skip)}/{len(SUBJECTS)}  {fov_skip}")
if failed_cases:
    print(f"  ❌ Failed:          {len(failed_cases)}/{len(SUBJECTS)}  {failed_cases}")
print(f"  📄 Metrics CSV:      {OUT_CSV}")
print(f"  🌐 HTML report:      {OUT_HTML}")
print(f"  📋 Case summary:     {summary_csv}")
print(f"  📁 Step logs:        {log_dir}")
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*62}\n")
