#!/usr/bin/env python3
"""Fuse multiple LAA anatomical priors and generate positive/negative prior maps.

Takes NUDF, VISTA3D, and TotalSegmentator LAA masks as inputs and produces:
  - consensus mask (majority vote)
  - union mask
  - intersection mask
  - voxel-level disagreement map (0–N raters disagree)
  - distance transform from negative prior exclusion zones
  - positive prior mask (LA/LAA/ostium structures)
  - negative prior mask (coronary, lung, aorta, myocardium, etc.)

Example:
  python scripts/fuse_laa_priors.py \\
    --nudf <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \\
    --vista3d <BIDS_ROOT>/derivatives/vista3d/<CASE_ID>/<CASE_ID>_laa_vista3d.nii.gz \\
    --totalseg-dir <BIDS_ROOT>/derivatives/totalseg/<CASE_ID> \\
    --output-dir <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID>/prior_fusion \\
    --case-id <CASE_ID>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


# TotalSegmentator per-structure filenames that form the positive LAA anatomical prior.
_POSITIVE_STRUCTURES: list[str] = [
    "left_atrium_appendage.nii.gz",
    "heart_atrium_left.nii.gz",
    "heart_myocardium.nii.gz",  # included for ostium boundary; masked out in negative later
]

# TotalSegmentator per-structure filenames that form the negative anatomical prior.
# These structures should NOT overlap with the true LAA cavity.
_NEGATIVE_STRUCTURES: list[str] = [
    "aorta.nii.gz",
    "pulmonary_artery.nii.gz",
    "pulmonary_vein.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_lower_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "heart_myocardium.nii.gz",
    "coronary_arteries.nii.gz",
    "pericardium.nii.gz",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse LAA anatomical priors and generate positive/negative prior maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nudf", default=None, help="NUDF LAA mask (.nii.gz)")
    p.add_argument("--vista3d", default=None, help="VISTA3D LAA mask (.nii.gz)")
    p.add_argument("--totalseg-laa", default=None, help="TotalSegmentator LAA mask (.nii.gz)")
    p.add_argument(
        "--totalseg-dir",
        default=None,
        help="TotalSegmentator per-structure output directory (for positive/negative priors)",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for fusion outputs")
    p.add_argument("--case-id", required=True, help="Case identifier used in output filenames")
    p.add_argument(
        "--majority-threshold",
        type=float,
        default=0.5,
        help="Fraction of non-None priors that must agree for majority-vote consensus (0–1)",
    )
    p.add_argument(
        "--neg-distance-mm",
        type=float,
        default=5.0,
        help="Distance in mm used to dilate the negative prior exclusion zone",
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


def _load_binary(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    img = nib.load(str(path))
    return (np.asarray(img.get_fdata()) > 0).astype(np.uint8)


def _load_structure(ts_dir: Path, filename: str, shape: tuple[int, ...]) -> np.ndarray:
    path = ts_dir / filename
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    arr = _load_binary(path)
    if arr is None or arr.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return arr


def _save(arr: np.ndarray, ref_img: nib.Nifti1Image, path: Path, dtype=np.uint8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(arr.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(out, str(path))


def _voxel_size_mm(img: nib.Nifti1Image) -> np.ndarray:
    return np.sqrt((np.array(img.affine[:3, :3]) ** 2).sum(axis=0))


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load reference geometry from the first available prior ---
    ref_img: nib.Nifti1Image | None = None
    for candidate in (args.nudf, args.vista3d, args.totalseg_laa):
        if candidate and Path(candidate).exists():
            ref_img = nib.load(candidate)
            break
    if ref_img is None:
        raise RuntimeError("No prior mask found. Provide at least one of --nudf/--vista3d/--totalseg-laa.")

    shape: tuple[int, ...] = ref_img.shape[:3]
    voxel_mm = _voxel_size_mm(ref_img)

    # --- Load LAA prior masks ---
    prior_masks: list[np.ndarray] = []
    prior_names: list[str] = []
    for flag, name in ((args.nudf, "nudf"), (args.vista3d, "vista3d"), (args.totalseg_laa, "totalseg")):
        arr = _load_binary(Path(flag) if flag else None)
        if arr is not None:
            if arr.shape != shape:
                print(f"WARNING: {name} mask shape {arr.shape} != reference {shape}; skipping.")
                continue
            prior_masks.append(arr)
            prior_names.append(name)

    if not prior_masks:
        raise RuntimeError("No valid LAA prior masks loaded.")

    print(f"Loaded {len(prior_masks)} LAA prior(s): {prior_names}")

    n = len(prior_masks)
    vote_stack = np.stack(prior_masks, axis=0).astype(np.float32)  # (N, X, Y, Z)

    # Disagreement: how many priors differ from the vote mean (values 0..N)
    vote_sum = vote_stack.sum(axis=0)  # agrees-positive voxels
    disagree_map = np.minimum(vote_sum, n - vote_sum).astype(np.uint8)  # 0=consensus, N//2=max disagreement

    union_mask = (vote_sum >= 1).astype(np.uint8)
    intersection_mask = (vote_sum == n).astype(np.uint8)
    threshold = max(1, round(args.majority_threshold * n))
    consensus_mask = (vote_sum >= threshold).astype(np.uint8)

    _save(union_mask, ref_img, out_dir / f"{args.case_id}_laa_prior_union.nii.gz")
    _save(intersection_mask, ref_img, out_dir / f"{args.case_id}_laa_prior_intersection.nii.gz")
    _save(consensus_mask, ref_img, out_dir / f"{args.case_id}_laa_prior_consensus.nii.gz")
    _save(disagree_map, ref_img, out_dir / f"{args.case_id}_laa_prior_disagreement.nii.gz")

    print(f"Union voxels: {union_mask.sum()} | Intersection: {intersection_mask.sum()} | Consensus: {consensus_mask.sum()}")

    # --- Positive and negative anatomical priors from TotalSegmentator ---
    positive_mask = np.zeros(shape, dtype=np.uint8)
    negative_mask = np.zeros(shape, dtype=np.uint8)

    if args.totalseg_dir:
        ts_dir = Path(args.totalseg_dir)
        for fname in _POSITIVE_STRUCTURES:
            positive_mask |= _load_structure(ts_dir, fname, shape)
        for fname in _NEGATIVE_STRUCTURES:
            negative_mask |= _load_structure(ts_dir, fname, shape)

        # Positive prior: subtract negative structures to avoid overlap
        positive_mask = np.clip(positive_mask.astype(np.int16) - negative_mask.astype(np.int16), 0, 1).astype(np.uint8)
        # Negative prior: subtract the union LAA cavity so we don't exclude the target
        negative_mask = np.clip(negative_mask.astype(np.int16) - union_mask.astype(np.int16), 0, 1).astype(np.uint8)

        _save(positive_mask, ref_img, out_dir / f"{args.case_id}_laa_positive_prior.nii.gz")
        _save(negative_mask, ref_img, out_dir / f"{args.case_id}_laa_negative_prior.nii.gz")
        print(f"Positive prior voxels: {positive_mask.sum()} | Negative prior voxels: {negative_mask.sum()}")

        # Distance transform from negative prior surface (in mm)
        # Voxels inside negative zone get distance 0; outside increases with distance
        if negative_mask.sum() > 0:
            neg_inv = 1 - negative_mask  # EDT wants background=0
            neg_dist = distance_transform_edt(neg_inv, sampling=voxel_mm.tolist()).astype(np.float32)
            _save(neg_dist, ref_img, out_dir / f"{args.case_id}_laa_negative_prior_dist.nii.gz", dtype=np.float32)
    else:
        print("No --totalseg-dir provided; skipping positive/negative prior and distance transform outputs.")

    # --- Distance transform of the disagreement map (for uncertainty-aware training) ---
    # EDT of regions where all priors agree on foreground (distance from edge of confident region)
    confident_fg = (vote_sum == n).astype(np.uint8)
    if confident_fg.sum() > 0:
        conf_dist = distance_transform_edt(1 - confident_fg, sampling=voxel_mm.tolist()).astype(np.float32)
    else:
        conf_dist = np.zeros(shape, dtype=np.float32)
    _save(conf_dist, ref_img, out_dir / f"{args.case_id}_laa_confidence_dist.nii.gz", dtype=np.float32)

    # --- Metadata JSON ---
    meta = {
        "case_id": args.case_id,
        "priors_loaded": prior_names,
        "n_priors": n,
        "majority_threshold": args.majority_threshold,
        "neg_distance_mm": args.neg_distance_mm,
        "union_voxels": int(union_mask.sum()),
        "intersection_voxels": int(intersection_mask.sum()),
        "consensus_voxels": int(consensus_mask.sum()),
        "positive_prior_voxels": int(positive_mask.sum()),
        "negative_prior_voxels": int(negative_mask.sum()),
        "outputs": {
            "union": f"{args.case_id}_laa_prior_union.nii.gz",
            "intersection": f"{args.case_id}_laa_prior_intersection.nii.gz",
            "consensus": f"{args.case_id}_laa_prior_consensus.nii.gz",
            "disagreement": f"{args.case_id}_laa_prior_disagreement.nii.gz",
            "positive_prior": f"{args.case_id}_laa_positive_prior.nii.gz",
            "negative_prior": f"{args.case_id}_laa_negative_prior.nii.gz",
            "negative_prior_dist": f"{args.case_id}_laa_negative_prior_dist.nii.gz",
            "confidence_dist": f"{args.case_id}_laa_confidence_dist.nii.gz",
        },
    }
    meta_path = out_dir / f"{args.case_id}_prior_fusion_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved prior fusion outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
