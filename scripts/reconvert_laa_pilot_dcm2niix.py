#!/usr/bin/env python3
"""Re-convert pilot CTAs from source DICOM with dcm2niix (correct Z spacing).

The original SLAObids NIfTIs were made with SimpleITK's ImageSeriesReader, which
takes Z spacing from only the first slice gap -> wrong/deformed Z on series with
non-uniform or overlapping slices (e.g. sub-142). dcm2niix derives geometry
correctly (and handles gantry tilt), so we re-convert each pilot case from its
DICOM folder and repoint the case session at the corrected CTA.

The largest-volume series in each subject's DICOM folder is taken as the CTA.

After this, regenerate candidates (run_laa_pilot_candidates.py) on the corrected
CTAs, since the old candidates share the wrong geometry.

Usage:
  python scripts/reconvert_laa_pilot_dcm2niix.py \
    --dicom-root /Volumes/Research13T/datasets/SLAODICOM --pilot-root outputs/laa_pilot      # dry-run
  # then add --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_DCM2NIIX = "/opt/anaconda3/bin/dcm2niix"


def reconvert_one(dcm2niix: str, dicom_dir: Path) -> Path | None:
    """Run dcm2niix on a subject DICOM folder; return the largest output NIfTI."""
    work = Path(tempfile.mkdtemp(prefix="dcm2niix_"))
    cmd = [
        dcm2niix, "-z", "y", "-m", "y", "-b", "n",
        "-f", "%s_%d_%r", "-o", str(work), str(dicom_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    niis = sorted(work.glob("*.nii.gz"))
    if not niis:
        print(f"      dcm2niix produced no NIfTI (rc={proc.returncode}); tail:\n{proc.stderr[-400:]}")
        return None
    # pick the volume with the most voxels (the CTA, vs scouts/derived)
    best = max(niis, key=lambda p: int(np.prod(nib.load(str(p)).shape)))
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", required=True, type=Path)
    ap.add_argument("--pilot-root", type=Path, default=Path("outputs/laa_pilot"))
    ap.add_argument("--reader", default="readerA")
    ap.add_argument("--dcm2niix", default=DEFAULT_DCM2NIIX)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    cases = sorted(d.name for d in args.pilot_root.glob("sub-*") if d.is_dir())
    print(f"{'case':10s} {'old_zooms (mm)':>22s}   {'new_zooms (mm)':>22s}  status")
    for case_id in cases:
        num = case_id.replace("sub-", "")
        dicom_dir = args.dicom_root / num
        sess_path = args.pilot_root / case_id / "laa_annotation" / args.reader / "logs" / f"{case_id}_session.json"
        session = json.loads(sess_path.read_text()) if sess_path.exists() else {}
        old = Path(session.get("cta_path", ""))
        old_z = tuple(round(float(x), 3) for x in nib.load(str(old)).header.get_zooms()[:3]) if old.exists() else None
        if not dicom_dir.exists():
            print(f"{case_id:10s} {str(old_z):>22s}   {'--':>22s}  DICOM not found ({dicom_dir})")
            continue
        best = reconvert_one(args.dcm2niix, dicom_dir)
        if best is None:
            print(f"{case_id:10s} {str(old_z):>22s}   {'--':>22s}  dcm2niix FAILED")
            continue
        new_z = tuple(round(float(x), 3) for x in nib.load(str(best)).header.get_zooms()[:3])
        changed = old_z is None or abs(new_z[2] - old_z[2]) > 0.01
        status = "Z CHANGED" if changed else "same"
        print(f"{case_id:10s} {str(old_z):>22s}   {str(new_z):>22s}  {status}")
        if args.apply:
            dst = args.pilot_root / case_id / f"{case_id}_acq-CTA_ct_dcm2niix.nii.gz"
            shutil.copy2(best, dst)
            session["cta_path"] = str(dst)
            session["cta_source"] = "dcm2niix"
            session["cta_zooms"] = list(new_z)
            session["candidate_stale"] = changed  # candidates need regen if geometry changed
            sess_path.write_text(json.dumps(session, indent=2))
            print(f"           -> {dst.name}" + ("  (candidates now STALE: regenerate)" if changed else ""))
        shutil.rmtree(best.parent, ignore_errors=True)
    if not args.apply:
        print("\n(dry-run; re-run with --apply to write corrected CTAs)")
    print("\nAfter --apply with Z changes: regenerate candidates on the corrected CTAs:")
    print("  conda run -n nv-segment-ct python scripts/run_laa_pilot_candidates.py --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
