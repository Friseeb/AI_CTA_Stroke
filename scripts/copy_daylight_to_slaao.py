"""
copy_daylight_to_slaao.py

Copies DAYLIGHT NIfTIs (original + defaced) into SLAAOBIDS, renaming them
to use the corresponding BS-SLAAO subject ID and _acq-ecta_ label.

Run without --execute for a dry-run summary.
Run with --execute to actually copy files.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
EXCEL_PATH  = Path(r"C:\Users\spost\Desktop\CT_image\SUB-ID_20260213a_FR.xlsx")
SRC_ORIG    = Path(r"C:\Users\spost\Desktop\CT_image\daylightbids")
SRC_DEFACED = Path(r"C:\Users\spost\Desktop\CT_image\daylightbids\derivatives\defaced")
DST_ROOT    = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
DST_DEFACED = DST_ROOT / "derivatives" / "defaced"


def load_pairs(excel_path: Path) -> list[tuple[int, int]]:
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["DAYLIGHT"].notna() & df["BS-SLAAO"].notna()].copy()
    df["DAYLIGHT"] = df["DAYLIGHT"].astype(int)
    df["BS-SLAAO"] = df["BS-SLAAO"].astype(int)
    return list(zip(df["DAYLIGHT"], df["BS-SLAAO"]))


def plan_copies(pairs: list[tuple[int, int]]) -> list[dict]:
    """Build a list of copy operations without touching the filesystem."""
    ops = []
    for day_id, slaao_id in pairs:
        day_str   = str(day_id)
        slaao_str = str(slaao_id)

        # ── Original NIfTI + JSON ──────────────────────────────────────────────
        src_nii  = SRC_ORIG / f"sub-{day_str}_acq-CTA_ct.nii.gz"
        src_json = SRC_ORIG / f"sub-{day_str}_acq-CTA_ct.json"
        dst_dir  = DST_ROOT / f"sub-{slaao_str}"
        dst_nii  = dst_dir  / f"sub-{slaao_str}_acq-ecta_ct.nii.gz"
        dst_json = dst_dir  / f"sub-{slaao_str}_acq-ecta_ct.json"

        ops.append({
            "daylight": day_id,
            "slaao":    slaao_id,
            "kind":     "original_nii",
            "src":      src_nii,
            "dst":      dst_nii,
            "dst_dir":  dst_dir,
            "exists":   src_nii.exists(),
        })
        if src_json.exists():
            ops.append({
                "daylight": day_id,
                "slaao":    slaao_id,
                "kind":     "original_json",
                "src":      src_json,
                "dst":      dst_json,
                "dst_dir":  dst_dir,
                "exists":   True,
            })

        # ── Defaced NIfTI ──────────────────────────────────────────────────────
        src_def = SRC_DEFACED / f"sub-{day_str}_acq-CTA_ct_defaced.nii.gz"
        dst_def_dir = DST_DEFACED / f"sub-{slaao_str}"
        dst_def = dst_def_dir / f"sub-{slaao_str}_acq-ecta_ct_defaced.nii.gz"

        ops.append({
            "daylight": day_id,
            "slaao":    slaao_id,
            "kind":     "defaced_nii",
            "src":      src_def,
            "dst":      dst_def,
            "dst_dir":  dst_def_dir,
            "exists":   src_def.exists(),
        })

    return ops


def dry_run(pairs: list[tuple[int, int]], ops: list[dict]) -> None:
    # Group by patient for the high-level summary
    by_patient: dict[int, dict] = {}
    for day_id, slaao_id in pairs:
        by_patient[day_id] = {"slaao": slaao_id, "orig": False, "defaced": False, "json": False}

    for op in ops:
        d = by_patient[op["daylight"]]
        if op["kind"] == "original_nii" and op["exists"]:
            d["orig"] = True
        if op["kind"] == "defaced_nii" and op["exists"]:
            d["defaced"] = True
        if op["kind"] == "original_json" and op["exists"]:
            d["json"] = True

    will_copy_orig    = sum(1 for d in by_patient.values() if d["orig"])
    will_copy_defaced = sum(1 for d in by_patient.values() if d["defaced"])
    will_copy_json    = sum(1 for d in by_patient.values() if d["json"])
    missing_both      = [
        (day_id, d["slaao"])
        for day_id, d in by_patient.items()
        if not d["orig"] and not d["defaced"]
    ]
    orig_only         = [
        (day_id, d["slaao"])
        for day_id, d in by_patient.items()
        if d["orig"] and not d["defaced"]
    ]
    defaced_only      = [
        (day_id, d["slaao"])
        for day_id, d in by_patient.items()
        if not d["orig"] and d["defaced"]
    ]

    print("=" * 60)
    print("DRY-RUN SUMMARY")
    print("=" * 60)
    print(f"Total pairs in Excel:              {len(pairs)}")
    print(f"Will copy original NIfTI:          {will_copy_orig}")
    print(f"Will copy JSON sidecar:            {will_copy_json}")
    print(f"Will copy defaced NIfTI:           {will_copy_defaced}")
    print(f"Missing BOTH original + defaced:   {len(missing_both)}")
    print()

    # Per-patient detail table
    print(f"{'DAYLIGHT':>10}  {'SLAAO':>6}  {'orig':>6}  {'json':>6}  {'defaced':>8}")
    print("-" * 46)
    for day_id, d in sorted(by_patient.items()):
        orig_mark    = "OK"  if d["orig"]    else "MISSING"
        json_mark    = "OK"  if d["json"]    else "-"
        defaced_mark = "OK"  if d["defaced"] else "MISSING"
        print(f"{day_id:>10}  {d['slaao']:>6}  {orig_mark:>6}  {json_mark:>6}  {defaced_mark:>8}")

    if missing_both:
        print()
        print("⚠  MISSING BOTH files (no copy will happen for these):")
        for day_id, slaao_id in missing_both:
            print(f"   DAYLIGHT {day_id} → SLAAO {slaao_id}")

    if orig_only:
        print()
        print("ℹ  Original only (no defaced available):")
        for day_id, slaao_id in orig_only:
            print(f"   DAYLIGHT {day_id} → SLAAO {slaao_id}")

    if defaced_only:
        print()
        print("ℹ  Defaced only (no original NIfTI found):")
        for day_id, slaao_id in defaced_only:
            print(f"   DAYLIGHT {day_id} → SLAAO {slaao_id}")

    # Skip-if-exists check
    already_done = [op for op in ops if op["exists"] and op["dst"].exists()]
    if already_done:
        print()
        print(f"ℹ  {len(already_done)} destination file(s) already exist and will be SKIPPED.")

    print()
    print("Run with --execute to perform the actual copy.")


def execute_copies(ops: list[dict]) -> None:
    to_copy = [op for op in ops if op["exists"] and not op["dst"].exists()]
    skipped = [op for op in ops if op["exists"] and op["dst"].exists()]
    missing = [op for op in ops if not op["exists"]]

    print(f"Files to copy:   {len(to_copy)}")
    print(f"Already exists (skip): {len(skipped)}")
    print(f"Source missing (skip): {len(missing)}")
    print()

    for op in tqdm(to_copy, desc="Copying", unit="file", dynamic_ncols=True):
        op["dst_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(op["src"], op["dst"])

    print()
    print(f"Done. Copied {len(to_copy)} file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} already-existing file(s).")
    if missing:
        print(f"Skipped {len(missing)} missing source file(s).")


def main():
    parser = argparse.ArgumentParser(description="Copy DAYLIGHT NIfTIs into SLAAOBIDS.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually copy files (default: dry-run only)")
    args = parser.parse_args()

    print(f"Loading pairs from: {EXCEL_PATH}")
    pairs = load_pairs(EXCEL_PATH)
    print(f"Found {len(pairs)} valid DAYLIGHT→SLAAO pairs.\n")

    ops = plan_copies(pairs)

    if args.execute:
        execute_copies(ops)
    else:
        dry_run(pairs, ops)


if __name__ == "__main__":
    main()
