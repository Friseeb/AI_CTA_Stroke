#!/usr/bin/env python3
"""Crop a head-neck-chest CTA to the thorax for NV-Reason-CT (chest/abdomen model, 192 mm cube at 2 mm).

The model auto-crops with an anatomy heuristic that expects a chest or abdomen study; a vertex-to-abdomen CTA
confuses it, so we hand it a thorax-only NIfTI: bounding box of the TotalSegmentator lung lobes (+ margin),
HU int16, original orientation.  Runs on the Mac; the model itself runs on the DGX Spark.

  python prepare_chest_crop.py --cta <cta.nii.gz> --ts-total <ts total dir> --out <case>_chest.nii.gz [--margin-mm 25]
"""
import argparse, json, sys
from pathlib import Path
import nibabel as nib, numpy as np

LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True); ap.add_argument("--ts-total", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--margin-mm", type=float, default=25.0)
    ap.add_argument("--downsample-mm", type=float, default=0.0, help="resample the crop to this isotropic spacing (e.g. 1.0) to shrink the file; 0 = native")
    a = ap.parse_args(argv)
    img = nib.load(str(a.cta)); sp = img.header.get_zooms()[:3]
    lungs = np.zeros(img.shape, bool); found = []
    for n in LOBES:
        p = a.ts_total / f"{n}.nii.gz"
        if p.is_file():
            lungs |= np.asarray(nib.load(str(p)).dataobj) > 0; found.append(n)
    if not lungs.any():
        raise SystemExit("no lung lobe masks found; run TotalSegmentator total with the lung lobes first")
    idx = np.where(lungs); pad = [int(a.margin_mm / s) for s in sp]
    sl = tuple(slice(max(idx[i].min() - pad[i], 0), min(idx[i].max() + pad[i] + 1, img.shape[i])) for i in range(3))
    data = np.asarray(img.dataobj)[sl].astype(np.int16)
    aff = img.affine.copy(); aff[:3, 3] = (img.affine @ np.array([sl[0].start, sl[1].start, sl[2].start, 1]))[:3]
    out_sp = sp
    if a.downsample_mm > 0:
        from scipy import ndimage
        zoom = [s / a.downsample_mm for s in sp]
        data = ndimage.zoom(data.astype(np.float32), zoom, order=1).round().astype(np.int16)
        aff = aff @ np.diag([1 / zoom[0], 1 / zoom[1], 1 / zoom[2], 1.0]); out_sp = (a.downsample_mm,) * 3
    out = nib.Nifti1Image(data, aff); out.header.set_zooms(out_sp)
    a.out.parent.mkdir(parents=True, exist_ok=True); nib.save(out, str(a.out))
    ext_mm = [float((s.stop - s.start) * z) for s, z in zip(sl, sp)]
    meta = {"source": str(a.cta), "crop_start_ijk": [int(s.start) for s in sl], "crop_shape": list(map(int, data.shape)),
            "extent_mm": ext_mm, "lobes_used": found, "margin_mm": a.margin_mm, "output_spacing_mm": list(map(float, out_sp)),
            "note": "NV-Reason-CT resamples to 2 mm and takes a 192 mm cube; extent above 192 mm in any axis will be cropped by the model"}
    a.out.with_suffix("").with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta)); return 0


if __name__ == "__main__":
    sys.exit(main())
