#!/usr/bin/env python3
"""
Simple CTA defacing by splitting head from body, defacing head, and restitching.

This approach:
1. Detects head region (superior portion with skull)
2. Extracts head volume
3. Defaceses head using anterior masking (face = anterior soft tissue)
4. Restitches with body

For more robust defacing, consider CTA-DEFACE: https://github.com/CCI-Bonn/CTA-DEFACE

Usage:
    python scripts/deface_cta_simple.py \
        --input data/sub-547_acq-CTA_ct.nii.gz \
        --output outputs/test/defaced_cta.nii.gz
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, center_of_mass


def find_head_region(data: np.ndarray, skull_hu: int = 400, voxel_size_z: float = 1.0) -> tuple[int, int]:
    """
    Find z-range containing the head based on skull detection.

    Head is identified as the superior region with high bone density (skull)
    vs lower bone density (spine). The skull forms a shell around the brain.

    Returns (z_start, z_end) of head region.
    """
    # Find skull voxels (high HU = bone)
    bone = data >= skull_hu

    # Sum bone voxels per axial slice
    bone_per_slice = bone.sum(axis=(0, 1))

    # Skull has much higher bone density than spine
    # Find the "jump" in bone density that indicates skull vs body
    # Skull typically has 3-5x more bone voxels per slice than spine

    max_bone = bone_per_slice.max()
    if max_bone == 0:
        return 0, data.shape[2]

    # Skull threshold: slices with >50% of max bone density
    skull_threshold = max_bone * 0.5
    is_skull = bone_per_slice > skull_threshold

    if not is_skull.any():
        # Fall back to top portion of volume
        return data.shape[2] - int(200 / voxel_size_z), data.shape[2]

    # Find contiguous skull region
    z_indices = np.where(is_skull)[0]

    # The skull is typically in the upper portion - find largest contiguous block
    # in the upper half of the volume
    mid_z = data.shape[2] // 2
    upper_indices = z_indices[z_indices > mid_z]

    if len(upper_indices) > 0:
        z_start = int(upper_indices[0])
        z_end = int(upper_indices[-1])
    else:
        z_start = int(z_indices[0])
        z_end = int(z_indices[-1])

    # Extend a bit below for neck
    neck_extension = int(50 / voxel_size_z)  # ~50mm for neck
    z_start = max(0, z_start - neck_extension)

    return z_start, z_end


def create_face_mask(
    data: np.ndarray,
    z_start: int,
    z_end: int,
    affine: np.ndarray,
    skull_hu: int = 400,
    anterior_fraction: float = 0.35,
) -> np.ndarray:
    """
    Create face mask for head region.

    Face is defined as anterior soft tissue in head slices.
    Detects anterior direction from image orientation.
    """
    import nibabel as nib

    face_mask = np.zeros(data.shape, dtype=bool)

    # Detect which Y direction is anterior from orientation
    orient = nib.aff2axcodes(affine)
    # 'A' in orient[1] means Y+ is anterior
    # 'P' in orient[1] means Y+ is posterior (Y- is anterior)
    y_is_anterior = orient[1] == 'A'
    print(f"  Orientation: {orient}, Y+ is {'anterior' if y_is_anterior else 'posterior'}")

    for z in range(z_start, z_end):
        slice_data = data[:, :, z]

        # Find skull in this slice
        skull_slice = slice_data >= skull_hu
        if skull_slice.sum() < 50:
            continue

        # Get skull bounding box
        skull_coords = np.where(skull_slice)
        if len(skull_coords[0]) == 0:
            continue

        y_min, y_max = skull_coords[1].min(), skull_coords[1].max()
        y_range = y_max - y_min
        if y_range < 10:
            continue

        # Determine anterior cutoff based on orientation
        if y_is_anterior:
            # Y+ is anterior, so face is at HIGH Y values
            anterior_limit = int(y_max - y_range * anterior_fraction)
            anterior = np.zeros_like(skull_slice)
            anterior[:, anterior_limit:] = True
        else:
            # Y- is anterior, so face is at LOW Y values
            anterior_limit = int(y_min + y_range * anterior_fraction)
            anterior = np.zeros_like(skull_slice)
            anterior[:, :anterior_limit] = True

        # Face = tissue in anterior region (not air, not dense bone)
        tissue = (slice_data > -200) & (slice_data < skull_hu)
        face_mask[:, :, z] = tissue & anterior

    return face_mask


def deface_cta(
    input_path: Path,
    output_path: Path,
    skull_hu: int = 400,
    anterior_fraction: float = 0.35,
    dilation_voxels: int = 3,
    fill_value: float = -1024.0,
    save_mask: bool = False,
) -> dict:
    """
    Deface CTA by masking anterior head region (face).

    Parameters
    ----------
    input_path : Path
        Input CTA NIfTI
    output_path : Path
        Output defaced NIfTI
    skull_hu : int
        HU threshold for skull detection
    anterior_fraction : float
        Fraction of head considered "anterior" (face)
    dilation_voxels : int
        Dilate face mask for more aggressive defacing
    fill_value : float
        Value to fill face region (-1024 = air)
    save_mask : bool
        Save face mask for verification

    Returns
    -------
    dict
        Defacing statistics
    """
    print(f"Loading: {input_path}")
    img = nib.load(str(input_path))
    data = img.get_fdata(dtype=np.float32)
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {img.header.get_zooms()}")

    # Find head region
    print("  Detecting head region...")
    voxel_z = float(img.header.get_zooms()[2])
    z_start, z_end = find_head_region(data, skull_hu, voxel_z)
    print(f"  Head z-range: {z_start} to {z_end} ({z_end - z_start} slices)")

    # Create face mask
    print(f"  Creating face mask (anterior {anterior_fraction*100:.0f}%)...")
    face_mask = create_face_mask(
        data,
        z_start,
        z_end,
        img.affine,
        skull_hu=skull_hu,
        anterior_fraction=anterior_fraction,
    )
    print(f"  Initial face mask: {face_mask.sum():,} voxels")

    # Dilate for safety margin
    if dilation_voxels > 0:
        print(f"  Dilating mask by {dilation_voxels} voxels...")
        face_mask = binary_dilation(face_mask, iterations=dilation_voxels)
        print(f"  Dilated face mask: {face_mask.sum():,} voxels")

    # Apply defacing
    print(f"  Applying defacing (fill={fill_value})...")
    data[face_mask] = fill_value

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(output_path))
    print(f"✓ Saved: {output_path}")

    if save_mask:
        mask_path = output_path.parent / f"{output_path.stem}_face_mask.nii.gz"
        nib.save(
            nib.Nifti1Image(face_mask.astype(np.uint8), img.affine, img.header),
            str(mask_path),
        )
        print(f"✓ Saved mask: {mask_path}")

    return {
        "head_z_range": (z_start, z_end),
        "face_voxels": int(face_mask.sum()),
        "fill_value": fill_value,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple CTA defacing")
    parser.add_argument("--input", required=True, help="Input CTA NIfTI")
    parser.add_argument("--output", required=True, help="Output defaced NIfTI")
    parser.add_argument("--skull-hu", type=int, default=400, help="Skull HU threshold")
    parser.add_argument(
        "--anterior-fraction",
        type=float,
        default=0.35,
        help="Fraction of head to deface (0.35 = front 35%%)",
    )
    parser.add_argument(
        "--dilation", type=int, default=3, help="Dilate face mask (voxels)"
    )
    parser.add_argument(
        "--fill", type=float, default=-1024.0, help="Fill value (-1024=air)"
    )
    parser.add_argument("--save-mask", action="store_true", help="Save face mask")

    args = parser.parse_args()

    result = deface_cta(
        input_path=Path(args.input),
        output_path=Path(args.output),
        skull_hu=args.skull_hu,
        anterior_fraction=args.anterior_fraction,
        dilation_voxels=args.dilation,
        fill_value=args.fill,
        save_mask=args.save_mask,
    )

    print(f"\nDefacing complete:")
    print(f"  Head range: z={result['head_z_range'][0]} to {result['head_z_range'][1]}")
    print(f"  Face voxels removed: {result['face_voxels']:,}")


if __name__ == "__main__":
    main()
