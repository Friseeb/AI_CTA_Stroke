#!/usr/bin/env python3
"""
Pilot pipeline: Steps 1+2 only (TotalSegmentator + NUDF LAA) on 9 patients
across CT_heart, CT_totalbody, and CT_abdomen.

Step 3 (LAA shape descriptors) is intentionally excluded — run visual QC in
3D Slicer first.

Usage:
  PYTHONUTF8=1 /path/to/cardiac-ct-explorer/python.exe scripts/_run_multict_pilot.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

SLAAOBIDS   = Path(r"C:/Users/spost/Desktop/CT_image/SLAAOBIDS")
DERIVATIVES = SLAAOBIDS / "derivatives"
SCRIPT_DIR  = Path(__file__).parent
PYTHON      = sys.executable

SUBJECTS = [
    # CT_heart
    {"sid": "71",  "ct_type": "CT_heart",     "filename": "sub-71_acq-ctheart_ph00_ct.nii.gz"},
    {"sid": "113", "ct_type": "CT_heart",     "filename": "sub-113_acq-ctheart_ph00_ct.nii.gz"},
    {"sid": "194", "ct_type": "CT_heart",     "filename": "sub-194_acq-ctheart_ph00_ct.nii.gz"},
    # CT_totalbody
    {"sid": "9",   "ct_type": "CT_totalbody", "filename": "sub-9_acq-ctbody_ph00_ct.nii.gz"},
    {"sid": "288", "ct_type": "CT_totalbody", "filename": "sub-288_acq-ctbody_ph00_ct.nii.gz"},
    {"sid": "238", "ct_type": "CT_totalbody", "filename": "sub-238_acq-ctbody_ph02_ct.nii.gz"},
    # CT_abdomen
    {"sid": "84",  "ct_type": "CT_abdomen",   "filename": "sub-84_acq-ctabdomen_ph00_ct.nii.gz"},
    {"sid": "173", "ct_type": "CT_abdomen",   "filename": "sub-173_acq-ctabdomen_ph00_ct.nii.gz"},
    {"sid": "163", "ct_type": "CT_abdomen",   "filename": "sub-163_acq-ctabdomen_ph00_ct.nii.gz"},
]


def run(cmd: list, label: str) -> tuple:
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    combined = result.stdout + result.stderr
    print(f"\n{'='*60}")
    print(f"  {label}  [rc={result.returncode}  {elapsed:.0f}s]")
    print(combined[-4000:] if len(combined) > 4000 else combined)
    print("="*60)
    return result.returncode, elapsed, combined


def compute_stats(input_nii: Path, step12_out: Path, laa_out: Path) -> dict:
    """Compute size_mb, la_hu_mean, laa_volume_mm3 from output files."""
    stats: dict = {"size_mb": None, "la_hu_mean": None, "laa_volume_mm3": None}

    # File size in MB
    if input_nii.exists():
        stats["size_mb"] = round(input_nii.stat().st_size / 1024 / 1024, 1)

    # LA HU mean — label 3 of heartchambers_highres
    scan_id = input_nii.name[:-7] if input_nii.name.endswith(".nii.gz") else input_nii.stem
    hc_path = step12_out / "TotalSegmentator" / scan_id / "heartchambers_highres.nii.gz"
    if hc_path.exists() and input_nii.exists():
        try:
            ct_data = nib.load(str(input_nii)).get_fdata(dtype=np.float32)
            hc_data = nib.load(str(hc_path)).get_fdata(dtype=np.float32)
            la_mask = hc_data == 3
            if la_mask.any():
                stats["la_hu_mean"] = round(float(ct_data[la_mask].mean()), 1)
        except Exception as e:
            print(f"  WARNING: LA HU computation failed: {e}")

    # LAA volume mm³
    if laa_out.exists():
        try:
            img = nib.load(str(laa_out))
            data = img.get_fdata(dtype=np.float32)
            zooms = img.header.get_zooms()
            voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
            n_vox = int((data > 0).sum())
            stats["laa_volume_mm3"] = round(n_vox * voxel_vol, 1) if n_vox > 0 else 0.0
        except Exception as e:
            print(f"  WARNING: LAA volume computation failed: {e}")

    return stats


def main() -> None:
    results = []

    for subj in SUBJECTS:
        sid      = subj["sid"]
        ct_type  = subj["ct_type"]
        filename = subj["filename"]
        sub_label = f"sub-{sid}"

        input_nii  = SLAAOBIDS / sub_label / filename
        step12_out = DERIVATIVES / f"cardiac_ct_explorer_{sub_label}"
        laa_out    = step12_out / filename.replace(".nii.gz", "_laa8.nii.gz")

        print(f"\n\n{'#'*60}")
        print(f"  SUBJECT : {sub_label}  [{ct_type}]")
        print(f"  Input   : {input_nii}")
        print(f"  Out dir : {step12_out}")
        print(f"  LAA out : {laa_out}")
        print(f"{'#'*60}")

        if not input_nii.exists():
            print(f"  ERROR: input file missing — skipping {sub_label}")
            results.append({
                "subject": sub_label, "ct_type": ct_type, "filename": filename,
                "status": "input_missing",
                "size_mb": None, "la_hu_mean": None, "laa_volume_mm3": None,
            })
            continue

        rc, elapsed, _ = run(
            [
                PYTHON,
                str(SCRIPT_DIR / "run_cardiac_ct_explorer_nudf_only.py"),
                "--input",      str(input_nii),
                "--output-dir", str(step12_out),
                "--laa-output", str(laa_out),
                "--run-totalseg",
                "--device",     "auto",
                "--allow-missing-laa",
            ],
            label=f"[{sub_label}] Step 1+2 (TotalSeg + NUDF)",
        )

        stats = compute_stats(input_nii, step12_out, laa_out)

        results.append({
            "subject":        sub_label,
            "ct_type":        ct_type,
            "filename":       filename,
            "status":         "success" if rc == 0 else "step12_failed",
            "step12_rc":      rc,
            "elapsed_s":      round(elapsed, 1),
            **stats,
        })

    # Save summary JSON
    summary_path = DERIVATIVES / "multict_pilot_summary.json"
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print report table
    print(f"\n\n{'='*70}")
    print("  PILOT REPORT")
    print("="*70)
    print(f"{'Subject':<12} {'CT type':<15} {'MB':>7} {'LA HU':>7} {'LAA mm³':>10}  Status")
    print("-"*70)
    for r in results:
        mb    = str(r["size_mb"])        if r.get("size_mb")        is not None else "—"
        la_hu = str(r["la_hu_mean"])     if r.get("la_hu_mean")     is not None else "—"
        vol   = str(r["laa_volume_mm3"]) if r.get("laa_volume_mm3") is not None else "—"
        print(f"{r['subject']:<12} {r['ct_type']:<15} {mb:>7} {la_hu:>7} {vol:>10}  {r['status']}")
    print(f"\nSummary JSON: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()
