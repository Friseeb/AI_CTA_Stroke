#!/usr/bin/env python3
"""
Build bone and vessel masks from a TotalSegmentator output folder (per-structure NIfTI files).

Usage:
  python -u scripts/build_masks_from_totalseg.py \
    --totalseg-dir /path/to/ts_output \
    --output-bone /path/to/bone_mask.nii.gz \
    --output-vessel /path/to/vessel_mask.nii.gz

Notes:
- Expects per-structure NIfTI files produced by TotalSegmentator (default behavior).
- Bone mask combines skull and cervical vertebrae (C1-C7) for head/neck CTA only.
  - Ignores thoracic, lumbar, and sacral vertebrae (outside field of view).
- Vessel mask combines available vascular structures (keeps vessels; does not subtract bones).
- If a structure file is missing, it is skipped; the script remains robust and logs found/missing structures.
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


BONE_STRUCTURES = [
    "skull",
    "vertebrae_C1.nii.gz",  # Cervical vertebrae only (head/neck CTA)
    "vertebrae_C2.nii.gz",
    "vertebrae_C3.nii.gz",
    "vertebrae_C4.nii.gz",
    "vertebrae_C5.nii.gz",
    "vertebrae_C6.nii.gz",
    "vertebrae_C7.nii.gz",
]

VESSEL_STRUCTURES = [
    "aorta",
    "pulmonary_artery",
    "pulmonary_vein",
    "vena_cava_inferior",
    "vena_cava_superior",
    "portal_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vein_left",
    "iliac_vein_right",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "subclavian_vein_left",
    "subclavian_vein_right",
    "brachiocephalic_trunk",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
    "internal_carotid_artery_left",
    "internal_carotid_artery_right",
]


def load_structure_mask(base_dir: Path, name: str, shape_ref):
    """Load a structure mask, supporting glob patterns and .nii.gz extension."""
    import glob
    
    # Check if it's a glob pattern
    if '*' in name:
        # Find all matching files
        matches = list(base_dir.glob(name))
        if not matches:
            return None
        # Combine all matching files
        combined_data = None
        affine, header = None, None
        for path in sorted(matches):  # Sort for consistent ordering
            img = nib.load(str(path))
            data = img.get_fdata() > 0
            if shape_ref is not None and data.shape != shape_ref:
                raise ValueError(f"Shape mismatch for {path.name}: {data.shape} vs {shape_ref}")
            if combined_data is None:
                combined_data = data.astype(np.uint8)
                affine, header = img.affine, img.header
            else:
                combined_data |= data
        return combined_data, affine, header
    else:
        # Single file - try with .nii.gz if no extension provided
        if not name.endswith('.nii.gz'):
            path = base_dir / f"{name}.nii.gz"
        else:
            path = base_dir / name
        if not path.exists():
            return None
        img = nib.load(str(path))
        data = img.get_fdata() > 0
        if shape_ref is not None and data.shape != shape_ref:
            raise ValueError(f"Shape mismatch for {path.name}: {data.shape} vs {shape_ref}")
        return data, img.affine, img.header


def combine_masks(base_dir: Path, structures):
    combined = None
    affine = None
    header = None
    found_structures = []
    missing_structures = []
    
    for name in structures:
        loaded = load_structure_mask(base_dir, name, combined.shape if combined is not None else None)
        if loaded is None:
            missing_structures.append(name)
            continue
        found_structures.append(name)
        data, affine, header = loaded if affine is None else (loaded[0], affine, header)
        if combined is None:
            combined = loaded[0].astype(np.uint8)
        else:
            combined |= loaded[0]
    
    if found_structures:
        print(f"  ✓ Found structures: {', '.join(found_structures)}")
    if missing_structures:
        print(f"  ⚠ Missing structures (skipped): {', '.join(missing_structures)}")
    
    return combined, affine, header


def main():
    ap = argparse.ArgumentParser(description="Build bone and vessel masks from TotalSegmentator outputs")
    ap.add_argument("--totalseg-dir", required=True, help="Path to TotalSegmentator output folder (per-structure NIfTI files)")
    ap.add_argument("--output-bone", required=True, help="Output path for bone mask NIfTI")
    ap.add_argument("--output-vessel", required=True, help="Output path for vessel mask NIfTI")
    args = ap.parse_args()

    ts_dir = Path(args.totalseg_dir)
    if not ts_dir.exists():
        raise FileNotFoundError(f"TotalSegmentator directory not found: {ts_dir}")

    bone_mask, affine_b, header_b = combine_masks(ts_dir, BONE_STRUCTURES)
    vessel_mask, affine_v, header_v = combine_masks(ts_dir, VESSEL_STRUCTURES)

    if bone_mask is None and vessel_mask is None:
        raise RuntimeError("No masks were built; check that expected structure files exist")

    if bone_mask is not None:
        nib.save(nib.Nifti1Image(bone_mask.astype(np.uint8), affine_b, header_b), str(args.output_bone))
        print(f"✓ Bone mask written: {args.output_bone}")
    if vessel_mask is not None:
        nib.save(nib.Nifti1Image(vessel_mask.astype(np.uint8), affine_v, header_v), str(args.output_vessel))
        print(f"✓ Vessel mask written: {args.output_vessel}")


if __name__ == "__main__":
    main()
