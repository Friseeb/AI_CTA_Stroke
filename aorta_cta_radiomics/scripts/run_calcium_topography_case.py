#!/usr/bin/env python
"""Split saved wall calcium into luminal-side vs adventitial-side components.

Reads the aorta mask, lumen-core mask, and wall calcium mask already written
by the main pipeline (or ``run_calcium_case.py``) for one case, classifies
each connected calcium component by its position between the lumen and the
outer wall, and writes luminal/adventitial/indeterminate masks plus a
``calcium_topography_features.csv`` row set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aorta_cta_radiomics import __version__
from aorta_cta_radiomics.calcium_topography import (
    classify_calcium_topography,
    summarize_calcium_topography,
)
from aorta_cta_radiomics.config import load_config
from aorta_cta_radiomics.features import write_csv
from aorta_cta_radiomics.io import read_mask, read_volume, write_mask_like
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--aorta-mask", default=None, type=Path, help="Optional cleaned aorta mask override.")
    parser.add_argument(
        "--calcium-mask",
        default=None,
        type=Path,
        help="Optional wall calcium mask override (defaults to the saved dynamic wall calcium mask).",
    )
    parser.add_argument("--seed-threshold-hu", default=500, type=int, help="Must match the dynamic wall config used to build the calcium mask.")
    parser.add_argument("--wall-thickness-mm", default=2.0, type=float, help="Assumed normal wall thickness used to build the inner lumen reference surface.")
    parser.add_argument("--luminal-max-ratio", default=0.4, type=float)
    parser.add_argument("--adventitial-min-ratio", default=0.6, type=float)
    parser.add_argument("--dominance-margin-fraction", default=0.2, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outdir = args.outdir
    masks_dir = outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    image = read_volume(args.image)

    aorta_mask_path = args.aorta_mask or masks_dir / f"{args.case_id}_aorta_mask_cleaned.nii.gz"
    dynamic_mask_name = f"aorta_wall_dynamic_seed{int(args.seed_threshold_hu)}HU"
    calcium_mask_path = args.calcium_mask or masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}.nii.gz"

    for label, path in [
        ("aorta mask", aorta_mask_path),
        ("wall calcium mask", calcium_mask_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label} at {path}. Run the main pipeline or run_calcium_case.py "
                "for this case first (with calcification.dynamic_wall.enabled: true)."
            )

    aorta_mask = read_mask(aorta_mask_path).array.astype(bool)
    calcium_mask = read_mask(calcium_mask_path).array.astype(bool)
    for name, arr in [("aorta mask", aorta_mask), ("wall calcium mask", calcium_mask)]:
        if arr.shape != image.array.shape:
            raise ValueError(f"Image and {name} must have the same shape.")

    result = classify_calcium_topography(
        calcium_mask=calcium_mask,
        aorta_mask=aorta_mask,
        spacing_xyz=image.spacing_xyz,
        image=image.array,
        wall_thickness_mm=float(args.wall_thickness_mm),
        luminal_max_ratio=float(args.luminal_max_ratio),
        adventitial_min_ratio=float(args.adventitial_min_ratio),
    )

    write_mask_like(result.luminal_mask, image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_luminal.nii.gz")
    write_mask_like(result.adventitial_mask, image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_adventitial.nii.gz")
    write_mask_like(result.indeterminate_mask, image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_indeterminate.nii.gz")
    write_csv(result.component_table, features_dir / f"{args.case_id}_calcium_topography_components.csv")

    software_version = str(config["outputs"].get("software_version", __version__))
    topography_frame = summarize_calcium_topography(
        image=image.array,
        result=result,
        spacing_xyz=image.spacing_xyz,
        case_id=args.case_id,
        mask_name=dynamic_mask_name,
        threshold_label=f"dynamic_lumen_referenced_seed{int(args.seed_threshold_hu)}HU",
        software_version=software_version,
        dominance_margin_fraction=float(args.dominance_margin_fraction),
    )
    write_csv(topography_frame, features_dir / "calcium_topography_features.csv")
    rebuild_modeling_wide(features_dir)

    dominant_row = topography_frame[topography_frame["feature_name"] == "dominant_topography"]
    dominant = dominant_row["feature_value"].iloc[0] if not dominant_row.empty else "unknown"
    print(f"Saved calcium topography outputs to {outdir.resolve()}. dominant_topography={dominant}")


if __name__ == "__main__":
    main()
