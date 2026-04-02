#!/usr/bin/env python3
"""
Full LAA segmentation pilot: 5 patients, one per CT type.
Steps run for every subject:
  Step 1+2 : TotalSegmentator + NUDF LAA  (run_cardiac_ct_explorer_nudf_only.py)
  Step 3   : LAA shape descriptors         (run_laa_shape_descriptors.py)

Subjects / phases selected for best isotropic spacing:
  1. eCTA        sub-1    — monophase  0.25 mm  (defaced file)
  2. CT_thorax   sub-18   — monophase  0.30 mm
  3. CT_heart    sub-107  — ph01       0.25 mm
  4. CT_totalbody sub-1070 — ph02      0.30 mm
  5. CT_abdomen  sub-107  — ph01       0.50 mm

Usage:
    & "C:/Users/spost/miniconda3/envs/cardiac-ct-explorer/python.exe" scripts/_run_5ct_pilot.py
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
DERIVATIVES = SLAAOBIDS / "derivatives"
SCRIPT_DIR  = Path(__file__).parent

_inproc_counter = 0  # ensures unique module names across calls

# ── Pilot subjects ────────────────────────────────────────────────────────────
# For eCTA: try the flat defaced path first (output of the deface batch),
# then the sub-folder variant the user named.
_ecta_flat   = DERIVATIVES / "defaced" / "sub-1_acq-ecta_ct_defaced.nii.gz"
_ecta_nested = DERIVATIVES / "defaced" / "sub-1" / "sub-1_acq-ecta_ct_defaced.nii.gz"
_ecta_input  = _ecta_flat if _ecta_flat.exists() else _ecta_nested

SUBJECTS = [
    {
        "subject":    "sub-1",
        "ct_type":    "eCTA",
        "phase":      "—",
        "spacing_z":  0.25,
        "input":      _ecta_input,
    },
    {
        "subject":    "sub-18",
        "ct_type":    "CT_thorax",
        "phase":      "—",
        "spacing_z":  0.30,
        "input":      SLAAOBIDS / "sub-18" / "sub-18_acq-ctthorax_ct.nii.gz",
    },
    {
        "subject":    "sub-107",
        "ct_type":    "CT_heart",
        "phase":      "ph01",
        "spacing_z":  0.25,
        "input":      SLAAOBIDS / "sub-107" / "sub-107_acq-ctheart_ph01_ct.nii.gz",
    },
    {
        "subject":    "sub-1070",
        "ct_type":    "CT_totalbody",
        "phase":      "ph02",
        "spacing_z":  0.30,
        "input":      SLAAOBIDS / "sub-1070" / "sub-1070_acq-ctbody_ph02_ct.nii.gz",
    },
    {
        "subject":    "sub-107",
        "ct_type":    "CT_abdomen",
        "phase":      "ph01",
        "spacing_z":  0.50,
        "input":      SLAAOBIDS / "sub-107" / "sub-107_acq-ctabdomen_ph01_ct.nii.gz",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_id(path: Path) -> str:
    n = path.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return path.stem


def _run_inproc(script_path: Path, args: list[str], label: str) -> tuple[int, float]:
    """
    Load and call script's main() in the current Python process.

    This avoids Windows Application Control blocking VTK/cardiacctexplorer DLLs
    in child subprocesses. The parent process (launched directly by the user) has
    the required trust level; subprocesses may not.
    """
    global _inproc_counter
    _inproc_counter += 1
    mod_name = f"_pilot_{script_path.stem}_{_inproc_counter}"

    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + args
    t0 = time.time()
    rc = 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Running in-process: {script_path.name}")
    print("=" * 60)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # defines all functions & top-level imports
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
    print(f"\n  {label}  [rc={rc}  {elapsed:.0f}s]")
    return rc, elapsed


def compute_stats(input_nii: Path, step12_out: Path, laa_out: Path) -> dict:
    """Compute size_mb, la_hu_mean, laa_volume_mm3."""
    stats: dict = {"size_mb": None, "la_hu_mean": None, "laa_volume_mm3": None}

    if input_nii.exists():
        stats["size_mb"] = round(input_nii.stat().st_size / 1024 / 1024, 1)

    # LA HU — label 3 of heartchambers_highres (CardiacCTExplorer convention)
    sid = _scan_id(input_nii)
    hc_path = step12_out / "TotalSegmentator" / sid / "heartchambers_highres.nii.gz"
    if hc_path.exists() and input_nii.exists():
        try:
            ct   = nib.load(str(input_nii)).get_fdata(dtype=np.float32)
            hc   = nib.load(str(hc_path)).get_fdata(dtype=np.float32)
            la   = hc == 3
            if la.any():
                stats["la_hu_mean"] = round(float(ct[la].mean()), 1)
        except Exception as e:
            print(f"  WARNING: LA HU computation failed: {e}")

    # LAA volume — from mask voxels
    if laa_out.exists():
        try:
            img      = nib.load(str(laa_out))
            data     = img.get_fdata(dtype=np.float32)
            zooms    = img.header.get_zooms()
            vox_vol  = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
            n_vox    = int((data > 0).sum())
            stats["laa_volume_mm3"] = round(n_vox * vox_vol, 1)
        except Exception as e:
            print(f"  WARNING: LAA volume computation failed: {e}")

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    results = []

    print("=" * 70)
    print("  5-CT PILOT  — Full pipeline (TotalSeg + NUDF + Shape descriptors)")
    print("=" * 70)

    for subj in tqdm(SUBJECTS, desc="Subjects", unit="sub"):
        subject  = subj["subject"]
        ct_type  = subj["ct_type"]
        phase    = subj["phase"]
        spacing_z = subj["spacing_z"]
        input_nii = subj["input"]

        sid       = _scan_id(input_nii)
        # Use full scan_id for output dirs to avoid sub-107 collision
        step12_out = DERIVATIVES / f"cardiac_ct_explorer_{sid}"
        laa_out    = step12_out / f"{sid}_laa8.nii.gz"
        step3_out  = DERIVATIVES / f"laa_shape_{sid}"

        print(f"\n\n{'#'*60}")
        print(f"  {subject}  [{ct_type}  phase={phase}  spacing_z={spacing_z}mm]")
        print(f"  Input   : {input_nii}")
        print(f"  Step1+2 : {step12_out}")
        print(f"  LAA     : {laa_out}")
        print(f"  Step3   : {step3_out}")
        print(f"{'#'*60}")

        if not input_nii.exists():
            print(f"  ERROR: input file missing — {input_nii}")
            results.append({
                "subject":       subject,
                "ct_type":       ct_type,
                "phase":         phase,
                "spacing_z":     spacing_z,
                "status":        "INPUT_MISSING",
                "nudf_status":   "—",
                "la_hu_mean":    None,
                "laa_vol_mm3":   None,
                "time_step12_s": None,
                "time_step3_s":  None,
                "flags":         ["INPUT_MISSING"],
            })
            continue

        # ── Step 1+2: TotalSegmentator + NUDF ────────────────────────────────
        rc12, t12 = _run_inproc(
            SCRIPT_DIR / "run_cardiac_ct_explorer_nudf_only.py",
            [
                "--input",      str(input_nii),
                "--output-dir", str(step12_out),
                "--laa-output", str(laa_out),
                "--run-totalseg",
                "--device",     "auto",
                "--allow-missing-laa",
            ],
            label=f"[{subject}] Step 1+2 TotalSeg + NUDF",
        )

        if rc12 != 0:
            results.append({
                "subject":       subject,
                "ct_type":       ct_type,
                "phase":         phase,
                "spacing_z":     spacing_z,
                "status":        "STEP12_FAILED",
                "nudf_status":   "FAILED",
                "la_hu_mean":    None,
                "laa_vol_mm3":   None,
                "time_step12_s": round(t12, 1),
                "time_step3_s":  None,
                "flags":         ["STEP12_FAILED"],
            })
            continue

        # ── Step 3: LAA shape descriptors ────────────────────────────────────
        rc3, t3 = _run_inproc(
            SCRIPT_DIR / "run_laa_shape_descriptors.py",
            [
                "--input",      str(laa_out),
                "--output-dir", str(step3_out),
                "--label-id",   "1",
            ],
            label=f"[{subject}] Step 3 Shape descriptors",
        )

        # ── Compute stats ─────────────────────────────────────────────────────
        stats = compute_stats(input_nii, step12_out, laa_out)
        laa_vol = stats["laa_volume_mm3"]

        # Try to read volume from descriptor JSON (more precise)
        desc_dir = step3_out / "descriptors"
        desc_jsons = sorted(desc_dir.glob("*_shape_descriptors.json")) if desc_dir.exists() else []
        if desc_jsons:
            try:
                d = json.loads(desc_jsons[0].read_text())
                if d.get("volume") is not None:
                    laa_vol = round(float(d["volume"]), 1)
            except Exception:
                pass

        # ── Flags ─────────────────────────────────────────────────────────────
        flags = []
        if not laa_out.exists():
            flags.append("EMPTY_MASK")
        elif laa_vol is None or laa_vol == 0.0:
            flags.append("LAA_VOL_ZERO")
        if rc3 != 0:
            flags.append("STEP3_FAILED")

        nudf_ok = "OK" if rc12 == 0 and laa_out.exists() else "FAILED"

        results.append({
            "subject":       subject,
            "ct_type":       ct_type,
            "phase":         phase,
            "spacing_z":     spacing_z,
            "status":        "OK" if not flags else " | ".join(flags),
            "nudf_status":   nudf_ok,
            "la_hu_mean":    stats["la_hu_mean"],
            "laa_vol_mm3":   laa_vol,
            "time_step12_s": round(t12, 1),
            "time_step3_s":  round(t3, 1),
            "flags":         flags,
            "laa_path":      str(laa_out),
            "step3_path":    str(step3_out),
        })

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_path = DERIVATIVES / "5ct_pilot_summary.json"
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Print report ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    print(f"\n\n{'='*80}")
    print("  5-CT PILOT — FINAL REPORT")
    print("="*80)
    hdr = f"{'Subject':<12} {'CT type':<14} {'Phase':<6} {'sz_z':>5} {'LA HU':>7} {'LAA mm³':>10} {'NUDF':>6}  {'Time':>7}  Flags"
    print(hdr)
    print("-"*80)
    for r in results:
        la_hu = f"{r['la_hu_mean']:.0f}" if r["la_hu_mean"] is not None else "—"
        vol   = f"{r['laa_vol_mm3']:.1f}" if r["laa_vol_mm3"] is not None else "—"
        t12   = f"{r['time_step12_s']:.0f}s" if r.get("time_step12_s") is not None else "—"
        flags = "  *** " + " ".join(r["flags"]) if r.get("flags") else ""
        print(
            f"{r['subject']:<12} {r['ct_type']:<14} {r['phase']:<6} "
            f"{r['spacing_z']:>5.2f} {la_hu:>7} {vol:>10} {r['nudf_status']:>6}  "
            f"{t12:>7}{flags}"
        )
    print(f"\n  Total wall time: {total_elapsed/60:.1f} min")
    print(f"  Summary JSON   : {summary_path}")
    print("="*80)

    # Flag if any problem
    flagged = [r for r in results if r.get("flags")]
    if flagged:
        print("\n  *** FLAGGED SUBJECTS (empty mask or zero volume):")
        for r in flagged:
            print(f"      {r['subject']:12s} [{r['ct_type']}]  flags={r['flags']}")


if __name__ == "__main__":
    main()
