#!/usr/bin/env python3
"""
Manually build bone and vessel masks from partial TotalSegmentator outputs.
Handles individual vertebrae files and missing structures gracefully.
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Bone structures (flexible patterns)
BONE_PATTERNS = [
    "vertebrae_*.nii.gz",  # Individual vertebrae
    "rib_*.nii.gz",        # Ribs
    "sternum.nii.gz",
    "skull.nii.gz",
]

# Vessel structures
VESSEL_PATTERNS = [
    "aorta.nii.gz",
    "*carotid*.nii.gz",
    "brachiocephalic*.nii.gz",
    "*vena_cava*.nii.gz",
    "subclavian*.nii.gz",
    "iliac*.nii.gz",
    "pulmonary*.nii.gz",
    "portal*.nii.gz",
]


def combine_masks_from_patterns(seg_dir: Path, patterns: list):
    """Combine all matching files into a single binary mask."""
    combined = None
    affine = None
    header = None
    count = 0
    
    for pattern in patterns:
        for file in seg_dir.glob(pattern):
            try:
                img = nib.load(str(file))
                data = img.get_fdata() > 0
                
                if combined is None:
                    combined = data.astype(np.uint8)
                    affine = img.affine
                    header = img.header
                else:
                    combined |= data.astype(np.uint8)
                count += 1
                print(f"  Added: {file.name}")
            except Exception as e:
                print(f"  Skipped {file.name}: {e}")
    
    print(f"  Total files combined: {count}")
    return combined, affine, header


def main():
    seg_dir = Path(sys.argv[1])
    bone_out = Path(sys.argv[2])
    vessel_out = Path(sys.argv[3])
    
    print("Building bone mask...")
    bone_mask, bone_aff, bone_hdr = combine_masks_from_patterns(seg_dir, BONE_PATTERNS)
    
    print("\nBuilding vessel mask...")
    vessel_mask, vessel_aff, vessel_hdr = combine_masks_from_patterns(seg_dir, VESSEL_PATTERNS)
    
    if bone_mask is not None:
        nib.save(nib.Nifti1Image(bone_mask, bone_aff, bone_hdr), str(bone_out))
        print(f"\n✓ Bone mask: {bone_out}")
    else:
        print("\n✗ No bone structures found")
    
    if vessel_mask is not None:
        nib.save(nib.Nifti1Image(vessel_mask, vessel_aff, vessel_hdr), str(vessel_out))
        print(f"✓ Vessel mask: {vessel_out}")
    else:
        print("✗ No vessel structures found")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python build_masks_manual.py <segmentator_dir> <bone_mask_out> <vessel_mask_out>")
        sys.exit(1)
    main()
