#!/usr/bin/env python3
"""Interactive LAA completion: crop -> SAM-3D (point prompts) -> paste back.

SAM-3D on a full CTA flattens all ~1100 slices (minutes on CPU). Cropping to an
ROI around the LAA first makes 3D interactive segmentation fast: we send only
the small crop to the MONAILabel server's `sam_3d` model, with the reader's
foreground/background points (converted to crop voxel coordinates), then paste
the returned mask back onto the full CTA grid.

Unlike VISTA3D label-108 (which reproduces the truncated LAA), SAM is driven by
the points, so a foreground click on the missed distal lobe *extends* the mask.

Example:
  python scripts/run_laa_sam3d_roi.py \
    --ct /Volumes/DICOM5/slaobids/sub-138_acq-CTA_ct.nii.gz \
    --out .../candidate_masks/sam3d_roi.nii.gz \
    --roi-ijk 230 150 0 370 350 40 \
    --fg "329,245,0;320,250,10" --bg "" \
    --server http://localhost:8000 --model sam_3d
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import requests


def _parse_points(text: str) -> list[list[int]]:
    pts = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pts.append([int(round(float(v))) for v in chunk.split(",")])
    return pts


def crop_affine(affine: np.ndarray, lo: np.ndarray) -> np.ndarray:
    out = affine.copy()
    out[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roi-ijk", type=int, nargs=6, required=True,
                    metavar=("i0", "j0", "k0", "i1", "j1", "k1"))
    ap.add_argument("--fg", default="", help='foreground pts (full-volume ijk): "i,j,k;i,j,k"')
    ap.add_argument("--bg", default="", help="background pts (full-volume ijk)")
    ap.add_argument("--prior-mask", type=Path, default=None,
                    help="existing VISTA3D candidate; the output is (prior UNION sam-from-points)")
    ap.add_argument("--margin", type=int, nargs=3, default=[8, 8, 4])
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--model", default="sam_3d")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)

    t0 = time.time()
    ct_img = nib.load(str(args.ct))
    ct = np.asarray(ct_img.dataobj)
    shape = np.array(ct.shape)

    lo = np.maximum(np.array(args.roi_ijk[:3]) - np.array(args.margin), 0)
    hi = np.minimum(np.array(args.roi_ijk[3:]) + np.array(args.margin), shape)
    if np.any(hi <= lo):
        raise SystemExit(f"Invalid ROI lo={lo.tolist()} hi={hi.tolist()}")
    print(f"[roi] crop lo={lo.tolist()} hi={hi.tolist()} size={(hi - lo).tolist()} full={shape.tolist()}", flush=True)

    lo_i = [int(x) for x in lo]
    fg = [[int(p[0] - lo_i[0]), int(p[1] - lo_i[1]), int(p[2] - lo_i[2])] for p in _parse_points(args.fg)]
    bg = [[int(p[0] - lo_i[0]), int(p[1] - lo_i[1]), int(p[2] - lo_i[2])] for p in _parse_points(args.bg)]
    if not fg:
        raise SystemExit("At least one foreground point (--fg) is required for SAM.")
    print(f"[pts] fg(crop)={fg} bg(crop)={bg}", flush=True)

    crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    work = Path(tempfile.mkdtemp(prefix="laa_sam_"))
    crop_path = work / "crop.nii.gz"
    nib.save(nib.Nifti1Image(crop, crop_affine(ct_img.affine, lo), ct_img.header), str(crop_path))

    params = {"device": args.device, "foreground": fg, "background": bg}
    url = f"{args.server.rstrip('/')}/infer/{args.model}?output=image"
    print(f"[sam] POST {url} crop={crop.shape} ...", flush=True)
    t1 = time.time()
    with crop_path.open("rb") as fh:
        import json
        resp = requests.post(
            url, files={"file": (crop_path.name, fh, "application/gzip")},
            data={"params": json.dumps(params)}, timeout=args.timeout,
        )
    infer_s = time.time() - t1
    if resp.status_code != 200:
        raise SystemExit(f"SAM infer failed HTTP {resp.status_code}: {resp.text[:500]}")

    mask_path = work / "crop_mask.nii.gz"
    mask_path.write_bytes(resp.content)
    crop_mask = np.asarray(nib.load(str(mask_path)).dataobj)
    if crop_mask.shape != tuple(crop.shape):
        raise SystemExit(f"Mask shape {crop_mask.shape} != crop {crop.shape}")

    full = np.zeros(ct.shape, dtype=np.uint8)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (crop_mask > 0).astype(np.uint8)
    sam_vox = int(full.sum())

    # Keep ALL the existing VISTA3D segmentation; SAM-from-points only ADDS.
    if args.prior_mask is not None and Path(args.prior_mask).exists():
        prior = np.asarray(nib.load(str(args.prior_mask)).dataobj) > 0
        if prior.shape == full.shape:
            full = (full.astype(bool) | prior).astype(np.uint8)
            print(f"[union] prior_vox={int(prior.sum())} sam_added={sam_vox} "
                  f"union={int(full.sum())}", flush=True)
        else:
            print(f"[warn] prior shape {prior.shape} != {full.shape}; skipping union", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, ct_img.affine), str(args.out))

    vox = int(full.sum())
    vol = vox * float(np.prod(ct_img.header.get_zooms()[:3])) / 1000.0
    print(f"[done] {args.out} voxels={vox} vol_ml={vol:.2f} infer={infer_s:.0f}s total={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
