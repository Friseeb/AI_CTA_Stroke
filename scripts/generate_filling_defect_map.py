#!/usr/bin/env python3
"""Generate a voxel-level filling-defect map from a CT volume and LAA cavity mask.

Classifies each voxel within the LAA cavity mask into filling-state categories
based on Hounsfield unit thresholds:

  0 = background (outside LAA mask)
  1 = normal contrast lumen  (HU >= --lumen-min)
  2 = stagnation / low contrast  (--stagnation-min <= HU < --lumen-min)
  3 = thrombus-like dark defect  (HU < --stagnation-min)
  4 = mixed / uncertain  (within --mixed-std-thresh of a category boundary)

The map and a per-region statistics JSON are written to the output directory.

Example:
  python scripts/generate_filling_defect_map.py \\
    --ct <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \\
    --laa-mask <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID>/<CASE_ID>_laa_corrected.nii.gz \\
    --output-dir <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID> \\
    --case-id <CASE_ID>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label


# Filling-state label values written to the output NIfTI.
LABEL_BACKGROUND = 0
LABEL_NORMAL_LUMEN = 1
LABEL_STAGNATION = 2
LABEL_THROMBUS = 3
LABEL_MIXED = 4

LABEL_NAMES = {
    LABEL_BACKGROUND: "background",
    LABEL_NORMAL_LUMEN: "normal_lumen",
    LABEL_STAGNATION: "stagnation",
    LABEL_THROMBUS: "thrombus_like",
    LABEL_MIXED: "mixed_uncertain",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a voxel-level filling-defect map from CT + LAA mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ct", required=True, help="CT volume NIfTI (.nii.gz)")
    p.add_argument("--laa-mask", required=True, help="LAA cavity mask NIfTI (.nii.gz)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--case-id", required=True, help="Case identifier for output filenames")

    p.add_argument(
        "--lumen-min",
        type=float,
        default=200.0,
        help="HU threshold above which voxels are classified as normal contrast lumen",
    )
    p.add_argument(
        "--stagnation-min",
        type=float,
        default=50.0,
        help="HU threshold above which voxels are classified as stagnation (below lumen-min)",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.5,
        help="Gaussian smoothing sigma (voxels) applied to CT before HU classification. 0 = off.",
    )
    p.add_argument(
        "--mixed-band-hu",
        type=float,
        default=30.0,
        help="Half-width of HU band around category boundaries that is reclassified as mixed/uncertain",
    )
    p.add_argument(
        "--min-component-voxels",
        type=int,
        default=10,
        help="Remove connected components smaller than this (noise suppression)",
    )
    p.add_argument("--check-env", action="store_true", help="Check required packages and exit")
    return p.parse_args()


def _check_env() -> None:
    import sys
    details: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("nibabel", "numpy", "scipy"):
        try:
            m = __import__(pkg)
            details[pkg] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            details[f"{pkg}_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _remove_small_components(binary: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1:
        return binary
    labeled, n = cc_label(binary)
    if n == 0:
        return binary
    keep = np.zeros_like(binary)
    for lab in range(1, n + 1):
        component = labeled == lab
        if component.sum() >= min_voxels:
            keep |= component
    return keep.astype(binary.dtype)


def _voxel_stats(ct_vals: np.ndarray) -> dict[str, float]:
    if ct_vals.size == 0:
        return {"n_voxels": 0, "mean_hu": float("nan"), "median_hu": float("nan"),
                "std_hu": float("nan"), "min_hu": float("nan"), "max_hu": float("nan")}
    return {
        "n_voxels": int(ct_vals.size),
        "mean_hu": float(np.mean(ct_vals)),
        "median_hu": float(np.median(ct_vals)),
        "std_hu": float(np.std(ct_vals)),
        "min_hu": float(np.min(ct_vals)),
        "max_hu": float(np.max(ct_vals)),
    }


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_img = nib.load(args.ct)
    mask_img = nib.load(args.laa_mask)

    ct = np.asarray(ct_img.get_fdata(), dtype=np.float32)
    mask = (np.asarray(mask_img.get_fdata()) > 0)

    # Resample mask to CT space if shapes differ
    if ct.shape != mask.shape:
        try:
            import SimpleITK as sitk
            ct_sitk = sitk.ReadImage(args.ct)
            mask_sitk = sitk.ReadImage(args.laa_mask)
            rs = sitk.ResampleImageFilter()
            rs.SetReferenceImage(ct_sitk)
            rs.SetInterpolator(sitk.sitkNearestNeighbor)
            rs.SetDefaultPixelValue(0)
            mask_sitk = sitk.Cast(rs.Execute(mask_sitk), sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(mask_sitk).astype(bool)
        except Exception:
            raise RuntimeError(
                f"CT shape {ct.shape} != mask shape {mask.shape}. "
                "Install SimpleITK for automatic resampling, or pre-register the inputs."
            )

    if args.smooth_sigma > 0:
        ct_smooth = gaussian_filter(ct, sigma=args.smooth_sigma)
    else:
        ct_smooth = ct

    # --- HU-threshold classification ---
    filling_map = np.zeros(ct.shape, dtype=np.uint8)
    lumen_min = args.lumen_min
    stag_min = args.stagnation_min
    band = args.mixed_band_hu

    roi = mask
    hu = ct_smooth

    # Primary classification
    filling_map[roi & (hu >= lumen_min)] = LABEL_NORMAL_LUMEN
    filling_map[roi & (hu >= stag_min) & (hu < lumen_min)] = LABEL_STAGNATION
    filling_map[roi & (hu < stag_min)] = LABEL_THROMBUS

    # Mixed band: voxels near a category boundary are reclassified as mixed/uncertain
    if band > 0:
        near_lumen_boundary = roi & (hu >= (lumen_min - band)) & (hu < (lumen_min + band))
        near_stag_boundary = roi & (hu >= (stag_min - band)) & (hu < (stag_min + band))
        filling_map[near_lumen_boundary | near_stag_boundary] = LABEL_MIXED

    # Remove small components per category to suppress noise
    min_v = args.min_component_voxels
    for lab in (LABEL_THROMBUS, LABEL_STAGNATION, LABEL_MIXED):
        binary = (filling_map == lab)
        if binary.sum() == 0:
            continue
        cleaned = _remove_small_components(binary, min_v)
        # Removed voxels fall back to their HU-based primary class
        removed = binary & ~cleaned
        filling_map[removed & (hu >= lumen_min)] = LABEL_NORMAL_LUMEN
        filling_map[removed & (hu >= stag_min) & (hu < lumen_min)] = LABEL_STAGNATION
        filling_map[removed & (hu < stag_min)] = LABEL_THROMBUS
        filling_map[cleaned] = lab

    # Save filling-defect map
    out_map = out_dir / f"{args.case_id}_filling_defect_map.nii.gz"
    nib.save(nib.Nifti1Image(filling_map, ct_img.affine, ct_img.header), str(out_map))

    # --- Per-region statistics ---
    stats: dict[str, object] = {
        "case_id": args.case_id,
        "thresholds": {
            "lumen_min_hu": lumen_min,
            "stagnation_min_hu": stag_min,
            "mixed_band_hu": band,
        },
        "roi_voxels": int(mask.sum()),
        "regions": {},
    }
    for lab, name in LABEL_NAMES.items():
        if lab == LABEL_BACKGROUND:
            continue
        region_vox = (filling_map == lab) & roi
        ct_vals = ct[region_vox]
        region_stats = _voxel_stats(ct_vals)
        region_stats["fraction_of_roi"] = (
            float(region_vox.sum() / mask.sum()) if mask.sum() > 0 else 0.0
        )
        stats["regions"][name] = region_stats

    stats_path = out_dir / f"{args.case_id}_filling_defect_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    for lab, name in LABEL_NAMES.items():
        if lab == LABEL_BACKGROUND:
            continue
        n = int((filling_map == lab).sum())
        frac = n / max(1, int(mask.sum()))
        print(f"  {name}: {n} voxels ({frac:.1%})")

    print(f"Saved: {out_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
