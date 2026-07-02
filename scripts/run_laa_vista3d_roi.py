#!/usr/bin/env python3
"""Fast 3D VISTA3D LAA segmentation inside a sub-selected ROI.

Crops the CTA to an ROI box around the LAA, runs VISTA3D (label 108 = left
atrial appendage) on the small crop (seconds on CPU instead of minutes on the
full volume), then pastes the result back onto the full CTA grid.

The ROI box may be given explicitly in voxel indices, or derived from an
existing candidate mask's robust (percentile) bounding box plus a margin.

Run in the env that has monai + transformers<5 (VISTA3D), e.g.:
  conda run -n nv-segment-ct python scripts/run_laa_vista3d_roi.py \
    --ct /Volumes/DICOM5/slaobids/sub-138_acq-CTA_ct.nii.gz \
    --out outputs/laa_pilot/sub-138/laa_annotation/readerA/candidate_masks/vista3d_roi.nii.gz \
    --from-mask outputs/laa_pilot/sub-138/laa_annotation/readerA/candidate_masks/vista3d_laa.nii.gz \
    --margin 24 24 12

  # or an explicit voxel ROI (i0 j0 k0 i1 j1 k1):
    --roi-ijk 245 200 0 375 330 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
VISTA_SCRIPT = REPO / "scripts" / "run_nv_segment_ct_laa.py"
MODEL_DIR = REPO / "external" / "nv_segment_ct"


def robust_bbox(mask_path: Path, pct: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) voxel bbox of a mask using [pct, 100-pct] percentiles.

    Percentiles drop stray voxels that would otherwise inflate the box across
    the whole volume.
    """
    arr = np.asarray(nib.load(str(mask_path)).dataobj)
    ijk = np.argwhere(arr > 0)
    if ijk.size == 0:
        raise SystemExit(f"Mask is empty: {mask_path}")
    lo = np.percentile(ijk, pct, axis=0)
    hi = np.percentile(ijk, 100 - pct, axis=0)
    return np.floor(lo).astype(int), np.ceil(hi).astype(int) + 1


def crop_affine(affine: np.ndarray, lo: np.ndarray) -> np.ndarray:
    out = affine.copy()
    out[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-mask", type=Path, help="derive ROI from this mask's robust bbox")
    src.add_argument("--roi-ijk", type=int, nargs=6, metavar=("i0", "j0", "k0", "i1", "j1", "k1"))
    ap.add_argument("--margin", type=int, nargs=3, default=[24, 24, 12], metavar=("di", "dj", "dk"))
    ap.add_argument("--label-id", default="108")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-crop", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    ct_img = nib.load(str(args.ct))
    ct = np.asarray(ct_img.dataobj)
    shape = np.array(ct.shape)

    if args.roi_ijk:
        lo = np.array(args.roi_ijk[:3]); hi = np.array(args.roi_ijk[3:])
    else:
        lo, hi = robust_bbox(args.from_mask)
    margin = np.array(args.margin)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, shape)
    if np.any(hi <= lo):
        raise SystemExit(f"Invalid ROI lo={lo.tolist()} hi={hi.tolist()}")
    print(f"[roi] crop ijk lo={lo.tolist()} hi={hi.tolist()} "
          f"size={(hi - lo).tolist()} (full={shape.tolist()})", flush=True)

    crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    work = Path(tempfile.mkdtemp(prefix="laa_roi_"))
    crop_path = work / "crop.nii.gz"
    crop_mask_path = work / "crop_laa.nii.gz"
    nib.save(nib.Nifti1Image(crop, crop_affine(ct_img.affine, lo), ct_img.header), str(crop_path))

    print(f"[vista3d] running on crop {crop.shape} ...", flush=True)
    t1 = time.time()
    rc = subprocess.run(
        [sys.executable, str(VISTA_SCRIPT), "--input", str(crop_path),
         "--output", str(crop_mask_path), "--label-id", str(args.label_id),
         "--model-dir", str(MODEL_DIR), "--device", args.device],
    ).returncode
    infer_s = time.time() - t1
    if rc != 0 or not crop_mask_path.exists():
        raise SystemExit(f"VISTA3D failed (rc={rc}); crop kept at {crop_path}")

    crop_mask = np.asarray(nib.load(str(crop_mask_path)).dataobj)
    full = np.zeros(ct.shape, dtype=np.uint8)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (crop_mask > 0).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, ct_img.affine), str(args.out))

    vox = int(full.sum())
    vol = vox * float(np.prod(ct_img.header.get_zooms()[:3])) / 1000.0
    if not args.keep_crop:
        for p in (crop_path, crop_mask_path):
            p.unlink(missing_ok=True)
    print(f"[done] {args.out}  voxels={vox} vol_ml={vol:.2f}  "
          f"infer={infer_s:.0f}s total={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
