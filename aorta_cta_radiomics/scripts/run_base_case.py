#!/usr/bin/env python
"""Run base aorta case preparation: cleaned mask, QC, shells, segments, centerline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aorta_cta_radiomics import __version__
from aorta_cta_radiomics.aorta_segments import segment_summary, whole_aorta_segment_mask
from aorta_cta_radiomics.centerline import approximate_centerline_by_slices
from aorta_cta_radiomics.cli import _qc_to_feature_rows, _save_figures
from aorta_cta_radiomics.config import load_config
from aorta_cta_radiomics.features import write_csv
from aorta_cta_radiomics.io import load_image_and_mask, write_label_like, write_mask_like
from aorta_cta_radiomics.lumen_geometry import slice_geometry_features
from aorta_cta_radiomics.preprocess import clean_aorta_mask
from aorta_cta_radiomics.segmentation_qc import calculate_qc_metrics, qc_metrics_to_frame
from aorta_cta_radiomics.shells import create_aorta_wall_band_masks, create_base_shells
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--aorta-mask", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outdir = args.outdir
    masks_dir = outdir / "masks" / args.case_id
    figures_dir = outdir / "figures" / args.case_id
    qc_dir = outdir / "qc"
    features_dir = outdir / "features"
    for directory in [masks_dir, figures_dir, qc_dir, features_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    image, raw_mask, mask_resampled = load_image_and_mask(
        image_path=args.image,
        mask_path=args.aorta_mask,
        resample_mask_if_needed=bool(config["image"]["resample_mask_if_needed"]),
    )
    cleaned_mask, cleaning_report = clean_aorta_mask(
        raw_mask.array,
        keep_largest_component=bool(config["mask_cleaning"]["keep_largest_component"]),
        fill_holes=bool(config["mask_cleaning"]["fill_holes"]),
        min_component_voxels=int(config["mask_cleaning"]["min_component_voxels"]),
    )
    cleaned_mask_path = masks_dir / f"{args.case_id}_aorta_mask_cleaned.nii.gz"
    write_mask_like(cleaned_mask, image.image, cleaned_mask_path)

    qc_metrics = calculate_qc_metrics(
        image=image.array,
        mask=cleaned_mask,
        spacing_xyz=image.spacing_xyz,
        case_id=args.case_id,
        components_before_cleaning=cleaning_report.components_before,
        mask_resampled=mask_resampled,
        small_mask_volume_mm3=float(config["mask_cleaning"]["small_mask_volume_mm3"]),
        large_mask_volume_mm3=float(config["mask_cleaning"]["large_mask_volume_mm3"]),
    )
    write_csv(qc_metrics_to_frame(qc_metrics), qc_dir / "qc_summary.csv")
    write_csv(
        _qc_to_feature_rows(qc_metrics, str(config["outputs"].get("software_version", __version__))),
        features_dir / "case_level_features.csv",
    )

    shell_masks = create_base_shells(cleaned_mask, image.spacing_xyz, list(config["shells"].get("base", [])))
    shell_masks.update(
        create_aorta_wall_band_masks(
            cleaned_mask,
            image.spacing_xyz,
            internal_mm=float(config["shells"].get("aorta_wall_internal_mm", 2.0)),
            external_mm=float(config["shells"].get("aorta_wall_external_mm", 2.0)),
        )
    )
    if bool(config["outputs"].get("save_masks", True)):
        for name, mask in shell_masks.items():
            write_mask_like(mask, image.image, masks_dir / f"{args.case_id}_{name}.nii.gz")

    centerline_frame = approximate_centerline_by_slices(
        cleaned_mask,
        spacing_xyz=image.spacing_xyz,
        case_id=args.case_id,
        reference_image=image.image,
    )
    geometry_frame = (
        slice_geometry_features(
            cleaned_mask,
            spacing_xyz=image.spacing_xyz,
            case_id=args.case_id,
            min_slice_voxels=int(config["geometry"]["min_slice_voxels"]),
            max_branch_link_distance_mm=float(config["geometry"].get("max_branch_link_distance_mm", 20.0)),
            max_components_per_slice=int(config["geometry"].get("max_components_per_slice", 4)),
        )
        if bool(config["geometry"]["enabled"])
        else centerline_frame.iloc[0:0].copy()
    )
    segment_labels = whole_aorta_segment_mask(cleaned_mask)
    write_label_like(segment_labels, image.image, masks_dir / f"{args.case_id}_aorta_segments_v1.nii.gz")
    write_csv(segment_summary(segment_labels, image.spacing_xyz, args.case_id), features_dir / "segment_level_features.csv")
    write_csv(centerline_frame, features_dir / "centerline_points.csv")
    write_csv(geometry_frame, features_dir / "centerline_point_features.csv")

    if bool(config["outputs"].get("save_figures", False)):
        _save_figures(image.array, cleaned_mask, cleaned_mask & False, args.case_id, figures_dir)

    rebuild_modeling_wide(features_dir)
    print(f"Saved base outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
