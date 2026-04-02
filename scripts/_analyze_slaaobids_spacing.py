#!/usr/bin/env python3
"""
Analyze spacing of all NIfTI files in SLAAOBIDS dataset.

Outputs:
  1. Full table (one row per file):        slaaobids_spacing_full.csv
  2. Per-subject summary:                  slaaobids_spacing_subject_summary.csv
  3. Per-CT-type counts:                   slaaobids_spacing_cttype_counts.csv
  4. Flagged subjects (all non-isotropic): slaaobids_spacing_flagged.csv

Usage:
    python scripts/_analyze_slaaobids_spacing.py
"""
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

SLAAOBIDS = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
OUT_DIR    = SLAAOBIDS / "derivatives" / "spacing_analysis"

# Regex to parse: sub-<N>_acq-<type>[_ph<NN>]_ct.nii.gz
FNAME_RE = re.compile(
    r"^sub-(?P<sub_id>\d+)_acq-(?P<ct_type>[^_]+?)(?:_ph(?P<phase>\d+))?_ct\.nii\.gz$"
)


def parse_filename(fname: str) -> dict | None:
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return {
        "subject":  f"sub-{m.group('sub_id')}",
        "sub_num":  int(m.group("sub_id")),
        "ct_type":  m.group("ct_type"),
        "phase":    int(m.group("phase")) if m.group("phase") is not None else None,
        "phase_label": f"ph{int(m.group('phase')):02d}" if m.group("phase") is not None else None,
    }


def get_spacing(nii_path: Path) -> tuple[float, float, float]:
    """Return (spacing_xy, spacing_z, n_slices) without loading voxel data."""
    img = nib.load(str(nii_path))
    zooms = img.header.get_zooms()
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    spacing_xy = (sx + sy) / 2          # mean in-plane spacing
    n_slices   = img.shape[2] if img.ndim >= 3 else 1
    return spacing_xy, float(sz), n_slices


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all subject-level NIfTI files ────────────────────────────
    nii_files = sorted(SLAAOBIDS.glob("sub-*/*.nii.gz"))
    print(f"Found {len(nii_files)} NIfTI files. Loading headers...")

    rows = []
    errors = []
    for path in tqdm(nii_files, desc="Reading headers", unit="file"):
        meta = parse_filename(path.name)
        if meta is None:
            errors.append(str(path))
            continue
        try:
            sxy, sz, n_slices = get_spacing(path)
        except Exception as e:
            errors.append(f"{path}: {e}")
            continue

        ratio      = sz / sxy if sxy > 0 else float("nan")
        isotropic  = ratio <= 1.2

        rows.append({
            "subject":     meta["subject"],
            "sub_num":     meta["sub_num"],
            "ct_type":     meta["ct_type"],
            "phase":       meta["phase"],
            "phase_label": meta["phase_label"],
            "spacing_xy":  round(sxy, 4),
            "spacing_z":   round(sz,  4),
            "n_slices":    n_slices,
            "ratio":       round(ratio, 3),
            "isotropic":   isotropic,
            "path":        str(path),
        })

    if errors:
        print(f"\nWarning: {len(errors)} file(s) could not be parsed:")
        for e in errors[:10]:
            print(f"  {e}")

    df = pd.DataFrame(rows).sort_values(["sub_num", "ct_type", "phase"])

    # ── 2. Save full table ──────────────────────────────────────────────────
    full_path = OUT_DIR / "slaaobids_spacing_full.csv"
    df.to_csv(full_path, index=False)
    print(f"\n[1/3] Full table → {full_path}  ({len(df)} rows)")

    # ── 3. Per-subject summary ──────────────────────────────────────────────
    subject_rows = []
    for (subject, ct_type), grp in df.groupby(["subject", "ct_type"], sort=False):
        n_phases  = len(grp)
        n_iso     = grp["isotropic"].sum()

        # Best phase = smallest spacing_z (most isotropic)
        best_idx  = grp["spacing_z"].idxmin()
        best_row  = grp.loc[best_idx]

        subject_rows.append({
            "subject":          subject,
            "ct_type":          ct_type,
            "n_phases":         n_phases,
            "n_isotropic":      int(n_iso),
            "best_phase":       best_row["phase_label"] if best_row["phase_label"] else "—",
            "spacing_z_best":   best_row["spacing_z"],
            "spacing_xy_best":  best_row["spacing_xy"],
            "ratio_best":       best_row["ratio"],
            "isotropic_best":   best_row["isotropic"],
        })

    subj_df = pd.DataFrame(subject_rows).sort_values(["subject", "ct_type"])
    subj_path = OUT_DIR / "slaaobids_spacing_subject_summary.csv"
    subj_df.to_csv(subj_path, index=False)
    print(f"[2/3] Subject summary → {subj_path}  ({len(subj_df)} rows)")

    # ── 4. Per-CT-type counts ───────────────────────────────────────────────
    ct_rows = []
    for ct_type, grp in df.groupby("ct_type"):
        n_subjects_multi_phase = (
            grp.groupby("subject")["phase"].count() > 1
        ).sum()
        ct_rows.append({
            "ct_type":                ct_type,
            "n_files":                len(grp),
            "n_isotropic":            int(grp["isotropic"].sum()),
            "n_non_isotropic":        int((~grp["isotropic"]).sum()),
            "n_subjects":             grp["subject"].nunique(),
            "n_subjects_multi_phase": int(n_subjects_multi_phase),
            "median_spacing_z":       round(grp["spacing_z"].median(), 3),
            "median_spacing_xy":      round(grp["spacing_xy"].median(), 3),
            "median_ratio":           round(grp["ratio"].median(), 3),
        })
    ct_df = pd.DataFrame(ct_rows).sort_values("ct_type")
    ct_path = OUT_DIR / "slaaobids_spacing_cttype_counts.csv"
    ct_df.to_csv(ct_path, index=False)
    print(f"[3/3] CT-type counts → {ct_path}")

    # ── 5. Flag subjects where ALL phases are non-isotropic ────────────────
    flagged = (
        subj_df[~subj_df["isotropic_best"]]
        [["subject", "ct_type", "n_phases", "spacing_z_best", "ratio_best"]]
        .sort_values(["subject", "ct_type"])
    )
    flag_path = OUT_DIR / "slaaobids_spacing_flagged.csv"
    flagged.to_csv(flag_path, index=False)

    # ── 6. Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPACING ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Total NIfTI files analysed : {len(df)}")
    print(f"  Unique subjects            : {df['subject'].nunique()}")
    print(f"  Isotropic (ratio ≤ 1.2)    : {df['isotropic'].sum()}")
    print(f"  Non-isotropic              : {(~df['isotropic']).sum()}")
    print()
    print("Per CT type:")
    print(ct_df[["ct_type","n_files","n_isotropic","n_non_isotropic",
                  "n_subjects","n_subjects_multi_phase",
                  "median_spacing_z","median_ratio"]].to_string(index=False))
    print()
    print(f"Subjects where BEST phase is non-isotropic: {len(flagged)}")
    if len(flagged):
        print(flagged.to_string(index=False))
    print("=" * 70)
    print(f"\nOutputs written to: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
