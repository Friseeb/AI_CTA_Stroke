#!/usr/bin/env python3
"""
Full LAA segmentation pipeline on 3 defaced eCTA cases.
  Step 1+2 : TotalSegmentator + NUDF LAA
  Step 3   : LAA shape descriptors

Cases chosen for good isotropic spacing from defaced eCTA pool:
  sub-100   0.488 × 0.250 mm  (ratio 0.51)
  sub-224   0.507 × 0.300 mm  (ratio 0.59)
  sub-1072  0.488 × 0.250 mm  (ratio 0.51)

Usage:
    & "C:/Users/spost/miniconda3/envs/cardiac-ct-explorer/python.exe" scripts/_run_ecta_pilot.py
"""
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

SLAAOBIDS   = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
DEFACED_DIR = SLAAOBIDS / "derivatives" / "defaced"
DERIVATIVES = SLAAOBIDS / "derivatives"
SCRIPT_DIR  = Path(__file__).parent

SUBJECTS = [
    {"subject": "sub-100",  "spacing_z": 0.250,
     "input": DEFACED_DIR / "sub-100_acq-ecta_ct_defaced.nii.gz"},
    {"subject": "sub-224",  "spacing_z": 0.300,
     "input": DEFACED_DIR / "sub-224_acq-ecta_ct_defaced.nii.gz"},
    {"subject": "sub-1072", "spacing_z": 0.250,
     "input": DEFACED_DIR / "sub-1072_acq-ecta_ct_defaced.nii.gz"},
]

_inproc_counter = 0


def _scan_id(path: Path) -> str:
    n = path.name
    return n[:-7] if n.endswith(".nii.gz") else (n[:-4] if n.endswith(".nii") else path.stem)


def _run_inproc(script_path: Path, args: list[str], label: str) -> tuple[int, float]:
    global _inproc_counter
    _inproc_counter += 1
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + args
    t0 = time.time()
    rc = 0
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    try:
        spec = importlib.util.spec_from_file_location(
            f"_p_{script_path.stem}_{_inproc_counter}", str(script_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.main()
        rc = result if isinstance(result, int) else 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except Exception:
        traceback.print_exc()
        rc = 1
    finally:
        sys.argv = old_argv
    elapsed = time.time() - t0
    print(f"\n  → {label}  [rc={rc}  {elapsed:.0f}s]")
    return rc, elapsed


def compute_stats(input_nii: Path, step12_out: Path, laa_out: Path) -> dict:
    stats: dict = {"la_hu_mean": None, "laa_volume_mm3": None}
    sid = _scan_id(input_nii)
    hc_path = step12_out / "TotalSegmentator" / sid / "heartchambers_highres.nii.gz"
    if hc_path.exists() and input_nii.exists():
        try:
            ct = nib.load(str(input_nii)).get_fdata(dtype=np.float32)
            hc = nib.load(str(hc_path)).get_fdata(dtype=np.float32)
            la = hc == 3
            if la.any():
                stats["la_hu_mean"] = round(float(ct[la].mean()), 1)
        except Exception as e:
            print(f"  WARNING: LA HU failed: {e}")
    if laa_out.exists():
        try:
            img     = nib.load(str(laa_out))
            data    = img.get_fdata(dtype=np.float32)
            zooms   = img.header.get_zooms()
            vox_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
            n_vox   = int((data > 0).sum())
            stats["laa_volume_mm3"] = round(n_vox * vox_vol, 1)
        except Exception as e:
            print(f"  WARNING: LAA volume failed: {e}")
    return stats


def main() -> None:
    t_total = time.time()
    results = []

    print("=" * 70)
    print("  eCTA PILOT — 3 defaced cases (TotalSeg + NUDF + Shape descriptors)")
    print("=" * 70)

    for subj in tqdm(SUBJECTS, desc="eCTA subjects", unit="sub"):
        subject   = subj["subject"]
        spacing_z = subj["spacing_z"]
        input_nii = subj["input"]
        sid       = _scan_id(input_nii)

        step12_out = DERIVATIVES / f"cardiac_ct_explorer_{sid}"
        laa_out    = step12_out / f"{sid}_laa8.nii.gz"
        step3_out  = DERIVATIVES / f"laa_shape_{sid}"

        print(f"\n\n{'#'*60}")
        print(f"  {subject}  [eCTA  spacing_z={spacing_z}mm]")
        print(f"  Input   : {input_nii}")
        print(f"  Step1+2 : {step12_out}")
        print(f"  LAA     : {laa_out}")
        print(f"  Step3   : {step3_out}")
        print(f"{'#'*60}")

        if not input_nii.exists():
            print(f"  ERROR: input missing — {input_nii}")
            results.append({"subject": subject, "status": "INPUT_MISSING", "flags": ["INPUT_MISSING"]})
            continue

        # Step 1+2
        rc12, t12 = _run_inproc(
            SCRIPT_DIR / "run_cardiac_ct_explorer_nudf_only.py",
            ["--input", str(input_nii), "--output-dir", str(step12_out),
             "--laa-output", str(laa_out), "--run-totalseg",
             "--device", "auto", "--allow-missing-laa"],
            label=f"[{subject}] Step 1+2 TotalSeg + NUDF",
        )

        if rc12 != 0:
            results.append({
                "subject": subject, "ct_type": "eCTA", "spacing_z": spacing_z,
                "status": "STEP12_FAILED", "nudf_status": "FAILED",
                "la_hu_mean": None, "laa_vol_mm3": None,
                "time_step12_s": round(t12, 1), "time_step3_s": None,
                "flags": ["STEP12_FAILED"],
            })
            continue

        # Step 3
        rc3, t3 = _run_inproc(
            SCRIPT_DIR / "run_laa_shape_descriptors.py",
            ["--input", str(laa_out), "--output-dir", str(step3_out), "--label-id", "1"],
            label=f"[{subject}] Step 3 Shape descriptors",
        )

        stats   = compute_stats(input_nii, step12_out, laa_out)
        laa_vol = stats["laa_volume_mm3"]

        # Prefer volume from descriptor JSON
        desc_jsons = sorted((step3_out / "descriptors").glob("*_shape_descriptors.json")) \
                     if (step3_out / "descriptors").exists() else []
        if desc_jsons:
            try:
                d = json.loads(desc_jsons[0].read_text())
                if d.get("volume") is not None:
                    laa_vol = round(float(d["volume"]), 1)
            except Exception:
                pass

        flags = []
        if not laa_out.exists():
            flags.append("EMPTY_MASK")
        elif laa_vol is None or laa_vol == 0.0:
            flags.append("LAA_VOL_ZERO")
        if rc3 != 0:
            flags.append("STEP3_FAILED")

        results.append({
            "subject":       subject,
            "ct_type":       "eCTA",
            "spacing_z":     spacing_z,
            "status":        "OK" if not flags else " | ".join(flags),
            "nudf_status":   "OK" if laa_out.exists() else "FAILED",
            "la_hu_mean":    stats["la_hu_mean"],
            "laa_vol_mm3":   laa_vol,
            "time_step12_s": round(t12, 1),
            "time_step3_s":  round(t3, 1),
            "flags":         flags,
            "laa_path":      str(laa_out),
        })

    # Save JSON
    summary_path = DERIVATIVES / "ecta_pilot_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print report
    elapsed = time.time() - t_total
    print(f"\n\n{'='*70}")
    print("  eCTA PILOT — FINAL REPORT")
    print("="*70)
    print(f"{'Subject':<12} {'sz_z':>5} {'LA HU':>7} {'LAA mm³':>10} {'NUDF':>6}  {'t12':>6}  {'t3':>6}  Flags")
    print("-"*70)
    for r in results:
        la_hu = f"{r['la_hu_mean']:.0f}" if r.get("la_hu_mean") is not None else "—"
        vol   = f"{r['laa_vol_mm3']:.1f}" if r.get("laa_vol_mm3") is not None else "—"
        t12   = f"{r['time_step12_s']:.0f}s" if r.get("time_step12_s") is not None else "—"
        t3    = f"{r.get('time_step3_s', 0) or 0:.0f}s"
        flags = "  *** " + " ".join(r.get("flags", [])) if r.get("flags") else ""
        print(
            f"{r['subject']:<12} {r.get('spacing_z', 0):>5.3f} {la_hu:>7} {vol:>10} "
            f"{r.get('nudf_status','—'):>6}  {t12:>6}  {t3:>6}{flags}"
        )
    print(f"\n  Total wall time : {elapsed/60:.1f} min")
    print(f"  Summary JSON    : {summary_path}")
    print("="*70)

    flagged = [r for r in results if r.get("flags")]
    if flagged:
        print("\n  *** FLAGGED:")
        for r in flagged:
            print(f"      {r['subject']}  flags={r['flags']}")


if __name__ == "__main__":
    main()
