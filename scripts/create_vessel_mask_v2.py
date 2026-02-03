#!/usr/bin/env python3
"""
Improved vessel mask creation using TotalSegmentator bone masks + HU thresholding.

Uses:
1. TotalSegmentator vertebrae for precise spine exclusion
2. HU threshold for skull/other bone (no segmentation available)
3. Face mask for optional defacing
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, label


def load_vertebrae_mask(totalseg_dir: Path, reference_shape: tuple) -> np.ndarray:
    """Load and combine all vertebrae masks from TotalSegmentator."""
    vertebrae = [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
        "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
        "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8",
        "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
        "sacrum",
    ]

    combined = np.zeros(reference_shape, dtype=np.uint8)
    found = []

    for name in vertebrae:
        path = totalseg_dir / f"{name}.nii.gz"
        if path.exists():
            img = nib.load(str(path))
            data = img.get_fdata() > 0
            if data.shape == reference_shape:
                combined |= data.astype(np.uint8)
                found.append(name)

    print(f"  Loaded {len(found)} vertebrae masks")
    return combined


def create_vessel_mask_v2(
    cta_path: Path,
    output_dir: Path,
    totalseg_dir: Path | None = None,
    hu_low: int = 150,
    hu_high: int = 700,
    skull_hu: int = 600,
    skull_dilation: int = 2,
    spine_dilation: int = 3,
    min_component_size: int = 500,
    deface: bool = False,
):
    """
    Create vessel mask with better bone removal.

    Uses TotalSegmentator vertebrae for spine, HU threshold for skull.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  HU range: {data.min():.0f} to {data.max():.0f}")

    # Step 0: Defacing (optional)
    if deface and totalseg_dir:
        face_path = totalseg_dir / "face.nii.gz"
        if face_path.exists():
            print(f"\n=== STEP 0: DEFACING ===")
            face_img = nib.load(str(face_path))
            face_mask = face_img.get_fdata() > 0
            if face_mask.shape == data.shape:
                face_dilated = binary_dilation(face_mask, iterations=3)
                data = data.copy()
                data[face_dilated] = -1024  # Air
                print(f"  Defaced {face_dilated.sum():,} voxels")

                # Save defaced CTA
                defaced_path = output_dir / "defaced_cta.nii.gz"
                nib.save(nib.Nifti1Image(data, cta.affine, cta.header), str(defaced_path))
                print(f"  Saved: {defaced_path}")
        else:
            print(f"  ⚠ No face.nii.gz found in {totalseg_dir}")

    # Step 1: HU threshold
    print(f"\n=== STEP 1: HU Threshold ({hu_low}-{hu_high} HU) ===")
    vessel_mask = (data >= hu_low) & (data <= hu_high)
    vessel_mask = vessel_mask.astype(np.uint8)
    print(f"  Vessel voxels: {vessel_mask.sum():,}")

    step1_path = output_dir / "step1_hu_threshold.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step1_path))

    # Step 2a: Remove spine using TotalSegmentator vertebrae
    if totalseg_dir and totalseg_dir.exists():
        print(f"\n=== STEP 2a: Remove Spine (TotalSegmentator) ===")
        spine_mask = load_vertebrae_mask(totalseg_dir, data.shape)

        if spine_mask.sum() > 0:
            spine_dilated = binary_dilation(spine_mask > 0, iterations=spine_dilation)
            before = vessel_mask.sum()
            vessel_mask = vessel_mask & ~spine_dilated.astype(np.uint8)
            removed = before - vessel_mask.sum()
            print(f"  Spine dilation: {spine_dilation} voxels")
            print(f"  Removed {removed:,} spine-adjacent voxels")

            # Save spine mask
            spine_path = output_dir / "spine_mask.nii.gz"
            nib.save(nib.Nifti1Image(spine_dilated.astype(np.uint8), cta.affine, cta.header), str(spine_path))

    # Step 2b: Remove skull using HU threshold (no segmentation available)
    print(f"\n=== STEP 2b: Remove Skull (HU >= {skull_hu}) ===")
    skull_mask = (data >= skull_hu).astype(np.uint8)

    # Exclude spine from skull mask (already handled above)
    if totalseg_dir and totalseg_dir.exists():
        spine_mask = load_vertebrae_mask(totalseg_dir, data.shape)
        spine_dilated = binary_dilation(spine_mask > 0, iterations=spine_dilation + 2)
        skull_mask = skull_mask & ~spine_dilated.astype(np.uint8)

    skull_dilated = binary_dilation(skull_mask > 0, iterations=skull_dilation)
    before = vessel_mask.sum()
    vessel_mask = vessel_mask & ~skull_dilated.astype(np.uint8)
    removed = before - vessel_mask.sum()
    print(f"  Skull dilation: {skull_dilation} voxels")
    print(f"  Removed {removed:,} skull-adjacent voxels")

    step2_path = output_dir / "step2_no_bone.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step2_path))

    # Step 3: Filter small components
    print(f"\n=== STEP 3: Filter Components (>= {min_component_size} voxels) ===")
    labeled, num_features = label(vessel_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        large_components = np.where(component_sizes >= min_component_size)[0]
        large_components = large_components[large_components > 0]
        vessel_mask = np.isin(labeled, large_components).astype(np.uint8)
        print(f"  Components: {num_features} -> {len(large_components)}")
        print(f"  Voxels: {vessel_mask.sum():,}")

    step3_path = output_dir / "step3_filtered.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask.copy(), cta.affine, cta.header), str(step3_path))

    # Step 4: Morphological cleaning
    print(f"\n=== STEP 4: Morphological Cleaning ===")
    vessel_mask = binary_erosion(vessel_mask > 0, iterations=1)
    vessel_mask = binary_dilation(vessel_mask, iterations=1).astype(np.uint8)
    print(f"  Final voxels: {vessel_mask.sum():,}")

    final_path = output_dir / "vessel_mask.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask, cta.affine, cta.header), str(final_path))
    print(f"\n✓ Final vessel mask: {final_path}")

    return vessel_mask


def main():
    parser = argparse.ArgumentParser(description="Improved vessel mask creation")
    parser.add_argument("--cta", required=True, help="Input CTA NIfTI file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--totalseg-dir", help="TotalSegmentator output directory")
    parser.add_argument("--hu-low", type=int, default=150)
    parser.add_argument("--hu-high", type=int, default=700)
    parser.add_argument("--skull-hu", type=int, default=600, help="HU threshold for skull")
    parser.add_argument("--skull-dilation", type=int, default=2)
    parser.add_argument("--spine-dilation", type=int, default=3)
    parser.add_argument("--min-component", type=int, default=500)
    parser.add_argument("--deface", action="store_true", help="Apply defacing")

    args = parser.parse_args()

    create_vessel_mask_v2(
        cta_path=Path(args.cta),
        output_dir=Path(args.output),
        totalseg_dir=Path(args.totalseg_dir) if args.totalseg_dir else None,
        hu_low=args.hu_low,
        hu_high=args.hu_high,
        skull_hu=args.skull_hu,
        skull_dilation=args.skull_dilation,
        spine_dilation=args.spine_dilation,
        min_component_size=args.min_component,
        deface=args.deface,
    )


if __name__ == "__main__":
    main()
