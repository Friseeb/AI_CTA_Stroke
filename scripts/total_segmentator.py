"""Run TotalSegmentator and build bone/vessel masks in one call.

Example:
  python -u scripts/total_segmentator.py \
    --input /path/to/cta.nii.gz \
    --out-dir outputs/segmentator \
    --bone-mask outputs/bone_mask.nii.gz \
    --vessel-mask outputs/vessel_mask.nii.gz
"""

import argparse
import sys
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_masks_from_totalseg import combine_masks


BONE_STRUCTURES = [
    "skull",
    "vertebrae",
    "rib_left",
    "rib_right",
    "sternum",
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


def run_totalseg(input_cta: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    totalsegmentator(
        input=str(input_cta),
        output=str(out_dir),
        task="total",
        fast=True,
        ml=True,
    )


def write_mask(mask, affine, header, output_path: Path, label: str):
    import nibabel as nib

    if mask is None:
        return False
    nib.save(nib.Nifti1Image(mask.astype("uint8"), affine, header), str(output_path))
    print(f"✓ {label} mask written: {output_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Run TotalSegmentator and emit bone/vessel masks")
    ap.add_argument("--input", required=True, help="Path to CTA NIfTI")
    ap.add_argument("--out-dir", required=True, help="Directory for TotalSegmentator outputs")
    ap.add_argument("--bone-mask", required=True, help="Output path for combined bone mask")
    ap.add_argument("--vessel-mask", required=True, help="Output path for combined vessel mask")
    args = ap.parse_args()

    input_cta = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_cta.exists():
        raise FileNotFoundError(f"CTA not found: {input_cta}")

    print(f"Running TotalSegmentator on {input_cta} -> {out_dir}")
    run_totalseg(input_cta, out_dir)

    print("Combining structures into bone/vessel masks")
    bone_mask, affine_b, header_b = combine_masks(out_dir, BONE_STRUCTURES)
    vessel_mask, affine_v, header_v = combine_masks(out_dir, VESSEL_STRUCTURES)

    wrote_bone = write_mask(bone_mask, affine_b, header_b, Path(args.bone_mask), "Bone")
    wrote_vessel = write_mask(vessel_mask, affine_v, header_v, Path(args.vessel_mask), "Vessel")

    if not (wrote_bone or wrote_vessel):
        raise RuntimeError("No masks written; check TotalSegmentator outputs")


if __name__ == "__main__":
    main()