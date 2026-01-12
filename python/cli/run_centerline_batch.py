"""Batch runner for vessel segmentation and QC.

Example:
python run_centerline_batch.py \
  --manifest data/manifests/cta_inputs.csv \
  --method vmtk \
  --output-root outputs/predictions/centerlines
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import nibabel as nib

# Add repository root to path for imports
REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.analysis.segmentation_methods import (
    skeleton_based_method,
    synthetic_cylinder,
    vmtk_eikonal_centerline,
)
from python.analysis.segmentation_runner import VesselSegmentationRunner


METHODS = {
    'skeleton': skeleton_based_method,
    'vmtk': vmtk_eikonal_centerline,
}


def _run_self_test(output_root: Path, method_key: str) -> Path:
    runner = VesselSegmentationRunner(output_root)
    phantom = synthetic_cylinder()
    tmp_dir = Path(tempfile.mkdtemp())
    phantom_path = tmp_dir / "phantom_cta.nii.gz"
    nib.save(phantom, phantom_path)
    result = runner.run_case("self_test", phantom_path, method_key, METHODS[method_key])
    return result.mask_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch vessel segmentation runner")
    parser.add_argument("--manifest", type=Path, help="CSV with case_id,nifti_path")
    parser.add_argument("--method", choices=METHODS.keys(), default="skeleton")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/predictions/centerlines"))
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    parser.add_argument("--dry-run", action="store_true", help="Only validate manifest paths")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic phantom test instead of manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = VesselSegmentationRunner(args.output_root)

    if args.self_test:
        path = _run_self_test(args.output_root, args.method)
        print(f"Self-test completed. Mask saved to {path}")
        return

    if args.dry_run:
        stats = runner.dry_run_manifest(args.manifest)
        print(f"Dry run: {stats['missing']} missing of {stats['total']} entries")
        return

    summary_path = runner.run_manifest(
        manifest_csv=args.manifest,
        method_name=args.method,
        segmentation_fn=METHODS[args.method],
        limit=args.limit,
    )
    print(f"Finished. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
