#!/usr/bin/env python3
"""
Pilot pipeline: run CardiacCTExplorer NUDF + TotalSegmentator (Steps 1+2)
then LAA shape descriptors (Step 3) on CT_thorax subjects.

Usage:
  conda run -n cardiac-ct-explorer python scripts/_run_ctthorax_pilot.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SLAAOBIDS = Path(r"C:/Users/spost/Desktop/CT_image/SLAAOBIDS")
DERIVATIVES = SLAAOBIDS / "derivatives"
SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable

SUBJECTS = ["54", "190", "148"]


def run(cmd: list[str], label: str) -> tuple[int, float, str]:
    """Run a command, return (returncode, elapsed_s, combined_output)."""
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    combined = result.stdout + result.stderr
    print(f"\n{'='*60}")
    print(f"  {label}  [rc={result.returncode}  {elapsed:.0f}s]")
    print(combined[-4000:] if len(combined) > 4000 else combined)
    print("="*60)
    return result.returncode, elapsed, combined


def main() -> None:
    results = []

    for sid in SUBJECTS:
        sub_label = f"sub-{sid}"
        input_nii = SLAAOBIDS / sub_label / f"{sub_label}_acq-ctthorax_ct.nii.gz"
        step12_out = DERIVATIVES / f"cardiac_ct_explorer_{sub_label}"
        laa_out    = step12_out / f"{sub_label}_acq-ctthorax_ct_laa8.nii.gz"
        step3_out  = DERIVATIVES / f"laa_shape_{sub_label}"

        print(f"\n\n{'#'*60}")
        print(f"  SUBJECT: {sub_label}")
        print(f"  Input:   {input_nii}")
        print(f"  Step12:  {step12_out}")
        print(f"  LAA:     {laa_out}")
        print(f"  Step3:   {step3_out}")
        print(f"{'#'*60}")

        if not input_nii.exists():
            print(f"  ERROR: input file missing — skipping {sub_label}")
            results.append({"subject": sub_label, "status": "input_missing"})
            continue

        # ── Step 1+2: TotalSegmentator + NUDF LAA ──────────────────────────
        rc12, t12, _ = run(
            [
                PYTHON,
                str(SCRIPT_DIR / "run_cardiac_ct_explorer_nudf_only.py"),
                "--input",      str(input_nii),
                "--output-dir", str(step12_out),
                "--laa-output", str(laa_out),
                "--run-totalseg",
                "--device", "auto",
                "--allow-missing-laa",
            ],
            label=f"[{sub_label}] Step 1+2 (TotalSeg + NUDF)",
        )

        if rc12 != 0:
            results.append({
                "subject": sub_label,
                "status": "step12_failed",
                "step12_rc": rc12,
                "step12_time_s": round(t12, 1),
            })
            continue

        # ── Step 3: LAA shape descriptors ──────────────────────────────────
        rc3, t3, _ = run(
            [
                PYTHON,
                str(SCRIPT_DIR / "run_laa_shape_descriptors.py"),
                "--input",      str(laa_out),
                "--output-dir", str(step3_out),
                "--label-id",   "1",
            ],
            label=f"[{sub_label}] Step 3 (LAA shape descriptors)",
        )

        # Try to read volume from the descriptor JSON
        laa_volume_mm3 = None
        desc_json = step3_out / "descriptors" / f"{sub_label}_acq-ctthorax_ct_laa8_shape_descriptors.json"
        if not desc_json.exists():
            # Fallback: find any descriptor JSON in the descriptors dir
            candidates = list((step3_out / "descriptors").glob("*_shape_descriptors.json")) if (step3_out / "descriptors").exists() else []
            if candidates:
                desc_json = sorted(candidates)[0]
        if desc_json.exists():
            try:
                d = json.loads(desc_json.read_text())
                laa_volume_mm3 = d.get("volume")
            except Exception as e:
                print(f"  WARNING: could not parse descriptor JSON: {e}")

        results.append({
            "subject":        sub_label,
            "status":         "success" if rc3 == 0 else "step3_failed",
            "step12_rc":      rc12,
            "step3_rc":       rc3,
            "step12_time_s":  round(t12, 1),
            "step3_time_s":   round(t3, 1),
            "total_time_s":   round(t12 + t3, 1),
            "laa_volume_mm3": round(laa_volume_mm3, 1) if laa_volume_mm3 is not None else None,
            "laa_output":     str(laa_out),
            "descriptor_json": str(desc_json) if desc_json.exists() else "NOT_FOUND",
        })

    # ── Final summary ────────────────────────────────────────────────────────
    summary_path = DERIVATIVES / "ctthorax_pilot_summary.json"
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n\n{'='*60}")
    print("  PILOT SUMMARY")
    print("="*60)
    for r in results:
        vol = f"{r['laa_volume_mm3']} mm³" if r.get("laa_volume_mm3") is not None else "n/a"
        total_t = f"{r.get('total_time_s', '?')}s" if r.get("total_time_s") is not None else "?"
        print(f"  {r['subject']:10s}  status={r['status']:20s}  LAA_vol={vol:>14s}  total_time={total_t}")
    print(f"\nSummary JSON: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
