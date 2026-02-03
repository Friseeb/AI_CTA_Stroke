#!/usr/bin/env python3
"""
Deface CTA scans by removing facial features using TotalSegmentator's face mask.

This script anonymizes CT/CTA scans by zeroing out voxels in the face region,
making re-identification impossible while preserving brain and vascular structures.

Usage:
  # Using pre-computed TotalSegmentator output:
  python -u scripts/deface_cta.py \
    --input /path/to/cta.nii.gz \
    --totalseg-dir /path/to/totalseg_output \
    --output /path/to/defaced_cta.nii.gz

  # Run TotalSegmentator automatically:
  python -u scripts/deface_cta.py \
    --input /path/to/cta.nii.gz \
    --output /path/to/defaced_cta.nii.gz \
    --run-totalseg

  # Batch processing:
  python -u scripts/deface_cta.py \
    --input /path/to/folder \
    --output /path/to/output_folder \
    --run-totalseg --recursive

Notes:
- Uses TotalSegmentator's 'face' structure for precise face localization
- Optionally dilates the face mask for more aggressive defacing
- Fill value can be set to air (-1024 HU), zero, or custom value
- Preserves all non-face anatomy including vessels and brain
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation


def load_face_mask(totalseg_dir: Path, reference_shape: tuple) -> np.ndarray | None:
    """Load the face mask from TotalSegmentator output."""
    face_path = totalseg_dir / "face.nii.gz"
    if not face_path.exists():
        print(f"  ⚠ Face mask not found: {face_path}")
        return None

    face_img = nib.load(str(face_path))
    face_data = np.asanyarray(face_img.dataobj) > 0

    if face_data.shape != reference_shape:
        raise ValueError(
            f"Face mask shape mismatch: {face_data.shape} vs {reference_shape}"
        )

    return face_data.astype(bool)


def create_anatomical_face_mask(
    data: np.ndarray,
    voxel_sizes: tuple,
    skull_hu: int = 400,
    face_anterior_fraction: float = 0.4,
) -> np.ndarray:
    """
    Create a face mask based on anatomical heuristics (no TotalSegmentator needed).

    This identifies the face as the anterior portion of the head where bone/soft tissue
    exists. It's less precise than TotalSegmentator but works as a fallback.

    Parameters
    ----------
    data : np.ndarray
        CTA volume data
    voxel_sizes : tuple
        Voxel dimensions (x, y, z)
    skull_hu : int
        HU threshold for skull detection
    face_anterior_fraction : float
        Fraction of head to consider as "anterior" (face region)

    Returns
    -------
    np.ndarray
        Boolean face mask
    """
    print("  Creating anatomical face mask (no TotalSegmentator)...")

    # Find head region by detecting skull (high HU)
    skull_mask = data >= skull_hu

    # Find the bounding box of the skull in each slice
    # We'll identify face as anterior portion where there's tissue
    face_mask = np.zeros(data.shape, dtype=bool)

    # For each axial slice, find the centroid and mark anterior region as face
    for z in range(data.shape[2]):
        slice_skull = skull_mask[:, :, z]
        if slice_skull.sum() < 100:  # Skip slices with little skull
            continue

        # Find centroid of skull in this slice
        coords = np.where(slice_skull)
        if len(coords[0]) == 0:
            continue

        centroid_x = coords[0].mean()
        centroid_y = coords[1].mean()

        # Find extent
        y_min, y_max = coords[1].min(), coords[1].max()
        y_range = y_max - y_min

        # Mark anterior portion (lower y values typically = anterior in radiological convention)
        # Adjust based on image orientation - we mark the front 40% of the head
        anterior_cutoff = y_min + y_range * face_anterior_fraction

        # Create face region: anterior to centroid, where there's tissue
        tissue_slice = data[:, :, z] > -500  # Non-air
        anterior_mask = np.zeros_like(slice_skull)
        anterior_mask[:, :int(anterior_cutoff)] = True

        face_mask[:, :, z] = tissue_slice & anterior_mask

    print(f"  Anatomical face mask: {face_mask.sum():,} voxels")
    return face_mask


def run_totalsegmentator(
    input_path: Path,
    output_dir: Path,
    fast: bool = True,
    roi_subset: list[str] | None = None,
    task: str = "total",
) -> Path:
    """Run TotalSegmentator to generate requested masks."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        raise ImportError(
            "TotalSegmentator not installed. Install with: pip install TotalSegmentator"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if roi_subset is None:
        roi_subset = ["face"]

    print(f"  Running TotalSegmentator (fast={fast})...")
    totalsegmentator(
        input=str(input_path),
        output=str(output_dir),
        task=task,
        fast=fast,
        ml=False,  # Individual files per structure
        roi_subset=roi_subset,  # Limit to requested structures for speed
    )

    return output_dir


def deface_volume(
    input_path: Path,
    output_path: Path,
    totalseg_dir: Path | None = None,
    run_totalseg: bool = False,
    fast_totalseg: bool = True,
    dilation_mm: float = 0.0,
    fill_value: float = -1024.0,
    save_mask: bool = False,
) -> dict:
    """
    Deface a single CTA volume.

    Parameters
    ----------
    input_path : Path
        Input CTA NIfTI file
    output_path : Path
        Output defaced NIfTI file
    totalseg_dir : Path, optional
        Pre-computed TotalSegmentator output directory
    run_totalseg : bool
        Run TotalSegmentator if totalseg_dir not provided
    fast_totalseg : bool
        Use fast mode for TotalSegmentator (lower resolution)
    dilation_mm : float
        Dilate face mask by this amount (mm) for more aggressive defacing
    fill_value : float
        Value to fill defaced region (-1024 = air, 0 = water)
    save_mask : bool
        Save the face mask alongside the defaced volume

    Returns
    -------
    dict
        Statistics about the defacing operation
    """
    print(f"Loading: {input_path}")
    img = nib.load(str(input_path))
    data = img.get_fdata(dtype=np.float32)
    voxel_sizes = img.header.get_zooms()[:3]

    print(f"  Shape: {data.shape}, Voxels: {voxel_sizes}")

    # Get or create TotalSegmentator output
    if totalseg_dir is None:
        if not run_totalseg:
            raise ValueError(
                "Either --totalseg-dir or --run-totalseg must be specified"
            )
        totalseg_dir = output_path.parent / f"{output_path.stem}_totalseg"
        run_totalsegmentator(input_path, totalseg_dir, fast=fast_totalseg)

    totalseg_dir = Path(totalseg_dir)

    # Load face mask - try TotalSegmentator first, fall back to anatomical
    face_mask = load_face_mask(totalseg_dir, data.shape)
    if face_mask is None:
        print("  Using anatomical fallback for face detection...")
        face_mask = create_anatomical_face_mask(data, voxel_sizes)

    original_face_voxels = int(np.sum(face_mask))
    print(f"  Face mask voxels: {original_face_voxels:,}")

    # Optionally dilate the mask
    if dilation_mm > 0:
        min_voxel = min(voxel_sizes)
        dilation_voxels = max(1, int(round(dilation_mm / min_voxel)))
        print(f"  Dilating face mask by {dilation_mm}mm ({dilation_voxels} voxels)...")
        face_mask = binary_dilation(face_mask, iterations=dilation_voxels)
        print(f"  Dilated mask voxels: {int(np.sum(face_mask)):,}")

    # Apply defacing
    print(f"  Applying defacing (fill={fill_value})...")
    defaced_voxels = int(np.sum(face_mask))
    data[face_mask] = fill_value

    # Save defaced volume
    output_path.parent.mkdir(parents=True, exist_ok=True)
    defaced_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    nib.save(defaced_img, str(output_path))
    print(f"✓ Saved defaced volume: {output_path}")

    # Optionally save the mask
    if save_mask:
        mask_path = output_path.parent / f"{output_path.stem}_face_mask.nii.gz"
        mask_img = nib.Nifti1Image(face_mask.astype(np.uint8), img.affine, img.header)
        nib.save(mask_img, str(mask_path))
        print(f"✓ Saved face mask: {mask_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "original_face_voxels": original_face_voxels,
        "defaced_voxels": defaced_voxels,
        "fill_value": fill_value,
        "dilation_mm": dilation_mm,
    }


def collect_inputs(input_path: Path, recursive: bool = False) -> list[Path]:
    """Collect input NIfTI files from path."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        patterns = ["**/*.nii.gz", "**/*.nii"] if recursive else ["*.nii.gz", "*.nii"]
        files = []
        for pattern in patterns:
            files.extend(sorted(input_path.glob(pattern)))
        # Deduplicate
        seen = set()
        unique = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique
    raise FileNotFoundError(f"Input not found: {input_path}")


def strip_nii_suffix(path: Path) -> str:
    """Return filename without .nii or .nii.gz extension."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def main():
    parser = argparse.ArgumentParser(
        description="Deface CTA scans using TotalSegmentator face mask"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CTA NIfTI file or directory for batch processing"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output defaced NIfTI file or directory"
    )
    parser.add_argument(
        "--totalseg-dir",
        help="Pre-computed TotalSegmentator output directory"
    )
    parser.add_argument(
        "--run-totalseg", action="store_true",
        help="Run TotalSegmentator automatically if --totalseg-dir not provided"
    )
    parser.add_argument(
        "--fast", action="store_true", default=True,
        help="Use fast mode for TotalSegmentator (default: True)"
    )
    parser.add_argument(
        "--no-fast", dest="fast", action="store_false",
        help="Use full resolution TotalSegmentator"
    )
    parser.add_argument(
        "--dilation-mm", type=float, default=0.0,
        help="Dilate face mask by this amount (mm) for more aggressive defacing"
    )
    parser.add_argument(
        "--fill-value", type=float, default=-1024.0,
        help="Fill value for defaced region (-1024=air, 0=water, default: -1024)"
    )
    parser.add_argument(
        "--save-mask", action="store_true",
        help="Save the face mask alongside the defaced volume"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search recursively for NIfTI files in input directory"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        inputs = collect_inputs(input_path, recursive=args.recursive)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2

    if not inputs:
        print(f"ERROR: No NIfTI files found in {input_path}")
        return 2

    batch_mode = len(inputs) > 1

    print("=" * 70)
    print("CTA DEFACING")
    print("=" * 70)
    if batch_mode:
        print(f"Batch mode: {len(inputs)} files")

    failures = []
    for idx, cta_path in enumerate(inputs, start=1):
        print(f"\n[{idx}/{len(inputs)}] {cta_path.name}")
        print("-" * 50)

        # Determine output path
        if batch_mode:
            case_name = strip_nii_suffix(cta_path)
            case_output = output_path / f"{case_name}_defaced.nii.gz"
            totalseg_dir = Path(args.totalseg_dir) / case_name if args.totalseg_dir else None
        else:
            case_output = output_path
            totalseg_dir = Path(args.totalseg_dir) if args.totalseg_dir else None

        try:
            deface_volume(
                input_path=cta_path,
                output_path=case_output,
                totalseg_dir=totalseg_dir,
                run_totalseg=args.run_totalseg,
                fast_totalseg=args.fast,
                dilation_mm=args.dilation_mm,
                fill_value=args.fill_value,
                save_mask=args.save_mask,
            )
        except Exception as e:
            failures.append((cta_path, str(e)))
            print(f"ERROR: {e}")

    print("\n" + "=" * 70)
    if failures:
        print(f"COMPLETED WITH {len(failures)} ERROR(S)")
        for path, err in failures:
            print(f"  {path}: {err}")
        return 1
    else:
        print("DEFACING COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
