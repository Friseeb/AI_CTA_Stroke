#!/usr/bin/env python3
"""
Step-by-step vessel mask creation with intermediate outputs.

This script creates a vessel mask from CTA using HU thresholding
and proper bone-adjacent voxel removal.

Steps:
1. HU threshold (150-700 HU) - captures contrast-enhanced vessels
2. Remove bone-adjacent voxels - dilate bone mask and subtract
3. Filter small components
4. Final vessel mask after morphological cleaning

Usage:
    python scripts/create_vessel_mask_stepwise.py \
        --cta data/sub-547_acq-CTA_ct.nii.gz \
        --output outputs/test_547_steps
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, label


def create_vessel_mask_stepwise(
    cta_path: Path,
    output_dir: Path,
    hu_low: int = 150,
    hu_high: int = 700,
    bone_hu: int = 400,
    bone_dilation: int = 2,
    min_component_size: int = 500,
):
    """
    Create vessel mask with intermediate outputs at each step.

    Parameters
    ----------
    cta_path : Path
        Input CTA NIfTI file
    output_dir : Path
        Output directory for intermediate and final masks
    hu_low : int
        Lower HU threshold for vessels (default: 150)
    hu_high : int
        Upper HU threshold for vessels (default: 700)
    bone_hu : int
        HU threshold for bone (default: 400 - conservative to catch calcium)
    bone_dilation : int
        Dilation iterations for bone mask (default: 2)
    min_component_size : int
        Minimum connected component size (default: 500)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {cta.header.get_zooms()}")
    print(f"  HU range: {data.min():.0f} to {data.max():.0f}")

    # Step 1: HU threshold
    print(f"\n=== STEP 1: HU Threshold ({hu_low}-{hu_high} HU) ===")
    vessel_mask = (data >= hu_low) & (data <= hu_high)
    vessel_mask = vessel_mask.astype(np.uint8)
    step1_count = vessel_mask.sum()
    print(f"  Vessel voxels: {step1_count:,}")

    step1_path = output_dir / "step1_hu_threshold.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step1_path))
    print(f"  Saved: {step1_path}")

    # Step 2: Remove bone-adjacent voxels
    # Key insight: we need to identify bone (high HU) and remove vessels NEAR bone
    print(f"\n=== STEP 2: Remove Bone-Adjacent Voxels ===")
    print(f"  Bone threshold: >= {bone_hu} HU")
    print(f"  Dilation: {bone_dilation} voxels")

    # Create bone mask - use a lower threshold to catch calcium too
    bone_mask = (data >= bone_hu).astype(np.uint8)
    bone_voxels = bone_mask.sum()
    print(f"  Bone voxels (>= {bone_hu} HU): {bone_voxels:,}")

    # Dilate bone mask to create exclusion zone
    if bone_dilation > 0:
        bone_dilated = binary_dilation(bone_mask > 0, iterations=bone_dilation)
    else:
        bone_dilated = bone_mask > 0
    dilated_voxels = bone_dilated.sum()
    print(f"  Dilated bone zone: {dilated_voxels:,} voxels")

    # Remove vessels that overlap with dilated bone zone
    vessel_before = vessel_mask.sum()
    vessel_mask = vessel_mask & ~bone_dilated.astype(np.uint8)
    vessel_after = vessel_mask.sum()
    removed = vessel_before - vessel_after
    print(f"  Removed {removed:,} bone-adjacent voxels ({100*removed/vessel_before:.1f}%)")
    print(f"  Remaining: {vessel_after:,} voxels")

    step2_path = output_dir / "step2_no_bone.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step2_path))
    print(f"  Saved: {step2_path}")

    # Also save the bone mask for reference
    bone_path = output_dir / "bone_mask.nii.gz"
    nib.save(nib.Nifti1Image(bone_mask, cta.affine, cta.header), str(bone_path))
    bone_dilated_path = output_dir / "bone_dilated.nii.gz"
    nib.save(nib.Nifti1Image(bone_dilated.astype(np.uint8), cta.affine, cta.header), str(bone_dilated_path))
    print(f"  Saved bone masks: {bone_path}, {bone_dilated_path}")

    # Step 3: Filter small components
    print(f"\n=== STEP 3: Filter Small Components (< {min_component_size} voxels) ===")
    labeled, num_features = label(vessel_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        # Find components larger than threshold
        large_components = np.where(component_sizes >= min_component_size)[0]
        large_components = large_components[large_components > 0]  # Exclude background
        vessel_mask = np.isin(labeled, large_components).astype(np.uint8)
        print(f"  Components: {num_features} -> {len(large_components)} (kept)")
        print(f"  Voxels: {vessel_after:,} -> {vessel_mask.sum():,}")

    step3_path = output_dir / "step3_filtered.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step3_path))
    print(f"  Saved: {step3_path}")

    # Step 4: Morphological cleaning
    print(f"\n=== STEP 4: Morphological Cleaning ===")
    vessel_before = vessel_mask.sum()
    vessel_mask = binary_erosion(vessel_mask > 0, iterations=1)
    vessel_mask = binary_dilation(vessel_mask, iterations=1).astype(np.uint8)
    vessel_after = vessel_mask.sum()
    print(f"  Voxels: {vessel_before:,} -> {vessel_after:,}")

    step4_path = output_dir / "step4_vessel_mask.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask, cta.affine, cta.header), str(step4_path))
    print(f"  Saved: {step4_path}")

    # Final summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Step 1 (HU threshold):     {step1_count:>12,} voxels")
    print(f"  Step 2 (bone removal):     {output_dir / 'step2_no_bone.nii.gz'}")
    print(f"  Step 3 (filtered):         {output_dir / 'step3_filtered.nii.gz'}")
    print(f"  Step 4 (final):            {vessel_after:>12,} voxels")
    print(f"\nOutputs saved to: {output_dir}")

    return vessel_mask


def main():
    parser = argparse.ArgumentParser(description="Step-by-step vessel mask creation")
    parser.add_argument("--cta", required=True, help="Input CTA NIfTI file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--hu-low", type=int, default=150, help="Lower HU threshold (default: 150)")
    parser.add_argument("--hu-high", type=int, default=700, help="Upper HU threshold (default: 700)")
    parser.add_argument("--bone-hu", type=int, default=400, help="Bone HU threshold (default: 400)")
    parser.add_argument("--bone-dilation", type=int, default=2, help="Bone dilation iterations (default: 2)")
    parser.add_argument("--min-component", type=int, default=500, help="Min component size (default: 500)")

    args = parser.parse_args()

    create_vessel_mask_stepwise(
        cta_path=Path(args.cta),
        output_dir=Path(args.output),
        hu_low=args.hu_low,
        hu_high=args.hu_high,
        bone_hu=args.bone_hu,
        bone_dilation=args.bone_dilation,
        min_component_size=args.min_component,
    )


if __name__ == "__main__":
    main()
