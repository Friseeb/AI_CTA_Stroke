#!/usr/bin/env python3
"""
Create vessel mask from CTA scan (one-time preprocessing step).

Usage:
  python scripts/create_vessel_mask.py --input data/sub-547_acq-CTA_ct.nii.gz --output outputs/vessel_masks/sub547_mask.nii.gz

This saves a binary vessel mask that can be reused for multiple centerline runs.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, label

def create_vessel_mask(
    cta_path: Path,
    output_path: Path,
    threshold_hu: int = 150,
    max_hu: int | None = 700,
    bone_hu: int = 900,
    strip_boundary_bone: bool = True,
    boundary_margin_mm: float = 6.0,
    min_component_size: int = 500,
):
    """Create vessel mask from CTA with bone stripping."""
    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {tuple(round(v, 3) for v in cta.header.get_zooms())}")
    print(f"  Intensity range: {data.min():.1f} to {data.max():.1f} HU")

    # Check if already a binary mask
    unique_vals = np.unique(data)
    if unique_vals.size <= 3 and unique_vals.min() >= 0 and unique_vals.max() <= 1:
        print("\n  Input is already a binary mask; saving directly")
        mask = (data > 0).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(mask, cta.affine, cta.header), str(output_path))
        print(f"✓ Saved: {output_path}")
        print(f"  Vessel voxels: {mask.sum():,}")
        return

    upper_desc = f" and < {max_hu}" if max_hu is not None else ""
    print(f"\nCreating vessel mask (HU > {threshold_hu}{upper_desc})...")
    
    bandpass = data > threshold_hu
    if max_hu is not None:
        bandpass &= data < max_hu

    print(f"  Initial vessel voxels: {bandpass.sum():,}")

    if strip_boundary_bone:
        print(f"  Stripping boundary bone (>= {bone_hu} HU)...")
        bone_mask = (data >= bone_hu).astype(np.uint8)
        labeled_bone, _ = label(bone_mask)

        boundary_ids = set()
        faces = [
            labeled_bone[0, :, :], labeled_bone[-1, :, :],
            labeled_bone[:, 0, :], labeled_bone[:, -1, :],
            labeled_bone[:, :, 0], labeled_bone[:, :, -1],
        ]
        for face in faces:
            boundary_ids.update(np.unique(face))
        boundary_ids.discard(0)

        if boundary_margin_mm > 0:
            vx, vy, vz = cta.header.get_zooms()
            margin_vox = [int(np.ceil(boundary_margin_mm / s)) for s in (vx, vy, vz)]
            nx, ny, nz = data.shape
            shell_mask = np.zeros_like(bone_mask, dtype=bool)
            shell_mask[:margin_vox[0], :, :] = True
            shell_mask[-margin_vox[0]:, :, :] = True
            shell_mask[:, :margin_vox[1], :] = True
            shell_mask[:, -margin_vox[1]:, :] = True
            shell_mask[:, :, :margin_vox[2]] = True
            shell_mask[:, :, -margin_vox[2]:] = True
            shell_bone_labels = np.unique(labeled_bone[shell_mask & (labeled_bone > 0)])
            boundary_ids.update(shell_bone_labels)

        if boundary_ids:
            boundary_bone = np.isin(labeled_bone, list(boundary_ids))
            bandpass &= ~boundary_bone
            print(f"  After bone removal: {bandpass.sum():,} voxels")

    print(f"  Filtering small components (< {min_component_size} voxels)...")
    mask = bandpass.astype(np.uint8)
    labeled, _ = label(mask)
    component_sizes = np.bincount(labeled.ravel())
    large_components = np.where(component_sizes >= min_component_size)[0]
    large_components = large_components[large_components > 0]
    mask_filtered = np.isin(labeled, large_components).astype(np.uint8)
    print(f"  After filtering: {mask_filtered.sum():,} voxels ({len(large_components)} components)")

    print("  Morphological cleaning...")
    mask_clean = binary_erosion(mask_filtered, iterations=1)
    mask_clean = binary_dilation(mask_clean, iterations=1).astype(np.uint8)
    print(f"  Final vessel voxels: {mask_clean.sum():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_clean, cta.affine, cta.header), str(output_path))
    print(f"\n✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create vessel mask from CTA')
    parser.add_argument('--input', required=True, help='Input CTA NIfTI')
    parser.add_argument('--output', required=True, help='Output mask NIfTI')
    parser.add_argument('--threshold', type=int, default=150, help='Lower HU threshold')
    parser.add_argument('--max-hu', type=int, default=700, help='Upper HU threshold (None to disable)')
    parser.add_argument('--bone-hu', type=int, default=900, help='Bone threshold for stripping')
    parser.add_argument('--no-strip-bone', action='store_true', help='Disable bone stripping')
    parser.add_argument('--boundary-margin-mm', type=float, default=6.0, help='Boundary margin for bone')
    parser.add_argument('--min-component-size', type=int, default=500, help='Min component size')
    args = parser.parse_args()

    create_vessel_mask(
        cta_path=Path(args.input),
        output_path=Path(args.output),
        threshold_hu=args.threshold,
        max_hu=args.max_hu if args.max_hu > 0 else None,
        bone_hu=args.bone_hu,
        strip_boundary_bone=not args.no_strip_bone,
        boundary_margin_mm=args.boundary_margin_mm,
        min_component_size=args.min_component_size,
    )


if __name__ == '__main__':
    main()
