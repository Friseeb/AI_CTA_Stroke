#!/usr/bin/env python3
"""VISTA3D promptable LAA segmentation: class 108 + foreground/background points.

Drives the VISTA3D bundle's own inference workflow (configs/inference.json) so
points are mapped through the image affine and `use_point_window` auto-crops a
window around the points (fast on CPU, no manual cropping needed).

Unlike a label-only run (which reproduces the truncated LAA), the foreground
points tell VISTA3D to *extend* class 108 (LAA) into the clicked region; it stays
anatomy-aware so it doesn't flood the whole blood pool like SAM.

Points are given in WORLD/RAS coordinates (what 3D Slicer markups store).

Example:
  python scripts/run_laa_vista3d_prompt.py \
    --ct /Volumes/DICOM5/slaobids/sub-138_acq-CTA_ct.nii.gz \
    --out .../candidate_masks/vista3d_prompt.nii.gz \
    --fg "-35.5,5.0,-254.0" --bg "" \
    --prior-mask .../candidate_masks/vista3d_laa.nii.gz
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = REPO / "outputs" / "laa_pilot" / "monai_apps" / "monaibundle" / "model" / "vista3d"


def _parse_points(text: str) -> list[list[float]]:
    pts = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if chunk:
            pts.append([float(v) for v in chunk.split(",")])
    return pts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fg", default="", help='foreground pts in RAS: "r,a,s;r,a,s"')
    ap.add_argument("--bg", default="", help="background pts in RAS")
    ap.add_argument("--label-id", type=int, default=108, help="VISTA3D class (108 = LAA)")
    ap.add_argument("--prior-mask", type=Path, default=None,
                    help="existing candidate; output = prior UNION vista3d-prompt")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = ap.parse_args(argv)

    fg = _parse_points(args.fg)
    bg = _parse_points(args.bg)
    point_labels = [1] * len(fg) + [0] * len(bg)

    ct_img = nib.load(str(args.ct))
    # The VISTA3D bundle expects points in ORIGINAL IMAGE VOXEL (i,j,k) space
    # (it maps voxel->preprocessed internally via the affines). Our prompts are
    # in RAS world coords, so convert: ijk = inv(affine) @ [r,a,s,1].
    aff_inv = np.linalg.inv(ct_img.affine)

    def _ras_to_vox(p):
        v = aff_inv @ np.array([p[0], p[1], p[2], 1.0])
        return [float(v[0]), float(v[1]), float(v[2])]

    points = [_ras_to_vox(p) for p in (fg + bg)]

    input_dict = {"image": str(args.ct), "label_prompt": [args.label_id]}
    if points:
        input_dict["points"] = points
        input_dict["point_labels"] = point_labels
    print(f"[vista3d-prompt] label_prompt=[{args.label_id}] fg={len(fg)} bg={len(bg)} "
          f"points(voxel)={[[round(c, 1) for c in p] for p in points]}", flush=True)

    work = Path(tempfile.mkdtemp(prefix="laa_v3dp_"))
    t0 = time.time()
    # The bundle config references `scripts.inferer.*` relative to the bundle
    # root, so the bundle dir must be importable.
    sys.path.insert(0, str(args.bundle))
    from monai.bundle import run

    run(
        run_id="run",
        init_id="initialize",
        config_file=str(args.bundle / "configs" / "inference.json"),
        meta_file=str(args.bundle / "configs" / "metadata.json"),
        logging_file=None,
        bundle_root=str(args.bundle),
        input_dict=input_dict,
        output_dir=str(work),
    )
    infer_s = time.time() - t0

    results = sorted(work.rglob("*.nii.gz"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise SystemExit(f"No VISTA3D output written under {work}")
    pred = np.asarray(nib.load(str(results[-1])).dataobj)

    # VISTA3D fills non-target voxels with 255 ("ignore"); the requested class is
    # labelled with its own id (108 = LAA). Extract exactly that class.
    full = (pred == args.label_id).astype(np.uint8)
    if full.shape != tuple(ct_img.shape):
        raise SystemExit(f"Pred shape {full.shape} != CT {ct_img.shape}")
    sam_vox = int(full.sum())

    if args.prior_mask is not None and Path(args.prior_mask).exists():
        prior = np.asarray(nib.load(str(args.prior_mask)).dataobj) > 0
        if prior.shape == full.shape:
            full = (full.astype(bool) | prior).astype(np.uint8)
            print(f"[union] prior_vox={int(prior.sum())} prompt_vox={sam_vox} "
                  f"union={int(full.sum())}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, ct_img.affine), str(args.out))
    vox = int(full.sum())
    vol = vox * float(np.prod(ct_img.header.get_zooms()[:3])) / 1000.0
    print(f"[done] {args.out} voxels={vox} vol_ml={vol:.2f} infer={infer_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
