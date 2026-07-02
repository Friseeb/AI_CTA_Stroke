#!/usr/bin/env python3
"""Correct the Z (slice) spacing of pilot CTAs/candidates from the source DICOM.

The SLAObids NIfTIs were converted with SimpleITK's ImageSeriesReader, which
derives Z spacing from only the first slice interval. For series with
non-uniform / overlapping slices this yields a wrong uniform Z spacing, so the
volume looks deformed along Z (axial fine). This tool re-derives the true
spacing as the MEDIAN of consecutive ImagePositionPatient distances and rewrites
the NIfTI Z spacing for the CTA and its candidate masks (arrays untouched, so
masks stay aligned).

A corrected CTA copy is written into the case folder and the session.json is
pointed at it; candidate masks are corrected in place.

Usage:
  # dry-run (report current vs true Z per case):
  python scripts/fix_laa_z_spacing_from_dicom.py \
    --dicom-root /Volumes/Research13T/datasets/SLAODICOM \
    --pilot-root outputs/laa_pilot
  # apply:
  python scripts/fix_laa_z_spacing_from_dicom.py --dicom-root ... --pilot-root ... --apply

Run in an env with pydicom + nibabel (e.g. the monailabel env).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def true_z_spacing(dicom_dir: Path) -> tuple[float, int]:
    """Median consecutive ImagePositionPatient distance (mm) over a series."""
    positions = []
    for p in dicom_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "ImageOrientationPatient"])
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                positions.append([float(x) for x in ipp])
        except Exception:
            continue
    if len(positions) < 2:
        raise ValueError(f"Too few slices with IPP in {dicom_dir}")
    pos = np.array(positions)
    # project onto the slice-normal axis (use the axis of max variance = Z stack)
    var = pos.var(axis=0)
    axis = int(np.argmax(var))
    zs = np.sort(pos[:, axis])
    deltas = np.diff(zs)
    deltas = deltas[deltas > 1e-4]  # drop duplicates
    return float(np.median(deltas)), len(zs)


def rescale_z(nii_path: Path, correct_z: float, out_path: Path) -> tuple[float, float]:
    img = nib.load(str(nii_path))
    cur_z = float(img.header.get_zooms()[2])
    aff = img.affine.copy()
    factor = correct_z / cur_z
    aff[:3, 2] = aff[:3, 2] * factor
    new = nib.Nifti1Image(np.asarray(img.dataobj), aff)
    new.header.set_qform(aff, code=1)
    new.header.set_sform(aff, code=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new, str(out_path))
    return cur_z, float(nib.load(str(out_path)).header.get_zooms()[2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", required=True, type=Path, help="dir containing <subject-id>/ DICOM folders")
    ap.add_argument("--pilot-root", type=Path, default=Path("outputs/laa_pilot"))
    ap.add_argument("--reader", default="readerA")
    ap.add_argument("--tol", type=float, default=0.01, help="mm difference to consider wrong")
    ap.add_argument("--apply", action="store_true", help="write fixes (default: dry-run)")
    args = ap.parse_args(argv)

    cases = sorted(d.name for d in args.pilot_root.glob("sub-*") if d.is_dir())
    print(f"{'case':10s} {'cur_z':>7s} {'true_z':>7s} {'status'}")
    for case_id in cases:
        num = case_id.replace("sub-", "")
        dicom_dir = args.dicom_root / num
        if not dicom_dir.exists():
            print(f"{case_id:10s} {'--':>7s} {'--':>7s} DICOM dir not found ({dicom_dir})")
            continue
        try:
            tz, n = true_z_spacing(dicom_dir)
        except Exception as e:
            print(f"{case_id:10s} ERROR: {e}")
            continue
        sess_path = args.pilot_root / case_id / "laa_annotation" / args.reader / "logs" / f"{case_id}_session.json"
        session = json.loads(sess_path.read_text()) if sess_path.exists() else {}
        cta = Path(session.get("cta_path", ""))
        cur_z = float(nib.load(str(cta)).header.get_zooms()[2]) if cta.exists() else float("nan")
        wrong = abs(cur_z - tz) > args.tol
        status = "WRONG -> fix" if wrong else "ok"
        print(f"{case_id:10s} {cur_z:7.3f} {tz:7.3f} {status} (n={n})")
        if not (args.apply and wrong and cta.exists()):
            continue
        # corrected CTA copy in the case folder
        fixed_cta = args.pilot_root / case_id / f"{case_id}_ct_zfixed.nii.gz"
        rescale_z(cta, tz, fixed_cta)
        session["cta_path"] = str(fixed_cta)
        session["z_spacing_corrected_from"] = cur_z
        session["z_spacing_true"] = tz
        sess_path.write_text(json.dumps(session, indent=2))
        # candidate masks (share the grid) corrected in place
        cand_dir = args.pilot_root / case_id / "laa_annotation" / args.reader / "candidate_masks"
        for m in cand_dir.glob("*.nii.gz"):
            rescale_z(m, tz, m)
        print(f"           -> wrote {fixed_cta.name} + fixed {len(list(cand_dir.glob('*.nii.gz')))} candidate(s)")
    if not args.apply:
        print("\n(dry-run; re-run with --apply to write fixes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
