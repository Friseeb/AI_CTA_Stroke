#!/usr/bin/env python
"""Run periaortic fat-omics and fat-closed wall/lumen masks from saved base masks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aorta_cta_radiomics import __version__
from aorta_cta_radiomics.aorta_segments import SEGMENT_LABELS, whole_aorta_segment_mask
from aorta_cta_radiomics.cli import _optional_float, _parse_named_ranges, _parse_ranges
from aorta_cta_radiomics.config import load_config
from aorta_cta_radiomics.crop import crop_region_for_mask
from aorta_cta_radiomics.fat_omics import extract_periaortic_fat_omics
from aorta_cta_radiomics.fat_wall import extract_fat_closed_aortic_wall
from aorta_cta_radiomics.features import write_csv
from aorta_cta_radiomics.io import read_mask, read_volume, write_label_like, write_mask_like
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--cleaned-aorta-mask", default=None, type=Path)
    parser.add_argument("--crop-margin-mm", default=8.0, type=float)
    parser.add_argument(
        "--input-aorta-mask",
        default=None,
        type=Path,
        help="Original VISTA/input aorta mask used as optional lumen floor.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outdir = args.outdir
    masks_dir = outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    image = read_volume(args.image)
    cleaned_mask_path = args.cleaned_aorta_mask or masks_dir / f"{args.case_id}_aorta_mask_cleaned.nii.gz"
    cleaned_mask = read_mask(cleaned_mask_path).array.astype(bool)
    if cleaned_mask.shape != image.array.shape:
        raise ValueError("Image and cleaned aorta mask must have the same shape.")
    fat_config = config.get("fat_omics", {})
    wall_config = config.get("wall_from_fat", {})
    crop_margin_mm = max(
        float(args.crop_margin_mm),
        float(fat_config.get("external_radius_mm", 5.0)) + 2.0,
        float(wall_config.get("outer_limit_mm", 5.0)) + float(wall_config.get("close_radius_mm", 3.0)) + 2.0,
    )
    crop_region = crop_region_for_mask(cleaned_mask, image.spacing_xyz, margin_mm=crop_margin_mm)
    image_array = crop_region.crop(image.array)
    cleaned_crop = crop_region.crop(cleaned_mask)

    centerline_frame = _read_csv_or_none(features_dir / "centerline_points.csv")
    segment_labels = crop_region.crop(_read_segment_labels(masks_dir, args.case_id, cleaned_mask))
    software_version = str(config["outputs"].get("software_version", __version__))

    fat_result = None
    fat_frame = pd.DataFrame()
    if bool(fat_config.get("enabled", True)):
        fat_result = extract_periaortic_fat_omics(
            image=image_array,
            aorta_mask=cleaned_crop,
            spacing_xyz=image.spacing_xyz,
            case_id=args.case_id,
            centerline_points=centerline_frame,
            segment_labels=segment_labels,
            segment_names=SEGMENT_LABELS,
            external_radius_mm=float(fat_config.get("external_radius_mm", 10.0)),
            adipose_hu_min=float(fat_config.get("adipose_hu_min", -190.0)),
            adipose_hu_max=float(fat_config.get("adipose_hu_max", -30.0)),
            high_hu_bins=_parse_named_ranges(fat_config.get("high_hu_bins", {})),
            radial_bins_mm=_parse_ranges(fat_config.get("radial_bins_mm", [])),
            angle_bins=int(fat_config.get("angle_bins", 12)),
            texture_levels=int(fat_config.get("texture_levels", 16)),
            software_version=software_version,
        )
        fat_frame = fat_result.features
        if bool(fat_config.get("save_mask", True)) and bool(config["outputs"].get("save_masks", True)):
            for stale_layer_path in masks_dir.glob(f"{args.case_id}_periaortic_fat_*mm.nii.gz"):
                stale_layer_path.unlink()
            write_mask_like(crop_region.paste(fat_result.periaortic_roi_mask), image.image, masks_dir / f"{args.case_id}_periaortic_fat_roi.nii.gz")
            write_mask_like(crop_region.paste(fat_result.fat_mask), image.image, masks_dir / f"{args.case_id}_periaortic_fat.nii.gz")
            for layer_name, layer_mask in fat_result.fat_layer_masks.items():
                write_mask_like(crop_region.paste(layer_mask), image.image, masks_dir / f"{args.case_id}_{layer_name}.nii.gz")

    wall_frame = pd.DataFrame()
    if bool(wall_config.get("enabled", False)):
        if fat_result is None:
            raise ValueError("wall_from_fat requires fat_omics.enabled=true.")
        lumen_floor_mask = None
        if bool(wall_config.get("preserve_raw_input_aorta_in_lumen_floor", False)) and args.input_aorta_mask:
            lumen_floor_mask = crop_region.crop(read_mask(args.input_aorta_mask).array.astype(bool))
            if lumen_floor_mask.shape != cleaned_crop.shape:
                raise ValueError("Input aorta floor mask crop must have the same shape as the aorta crop.")
        wall_result = extract_fat_closed_aortic_wall(
            image=image_array,
            aorta_mask=cleaned_crop,
            fat_mask=fat_result.fat_mask,
            spacing_xyz=image.spacing_xyz,
            case_id=args.case_id,
            outer_limit_mm=float(wall_config.get("outer_limit_mm", 5.0)),
            close_radius_mm=float(wall_config.get("close_radius_mm", 3.0)),
            lumen_core_distance_mm=float(wall_config.get("lumen_core_distance_mm", 5.0)),
            centerline_core_radius_mm=float(wall_config.get("centerline_core_radius_mm", 2.0)),
            contrast_lower_margin_hu=float(wall_config.get("contrast_lower_margin_hu", 120.0)),
            min_lumen_hu=float(wall_config.get("min_lumen_hu", 150.0)),
            max_lumen_hu_above_reference=_optional_float(wall_config.get("max_lumen_hu_above_reference", 300.0)),
            lumen_reference_lower_fraction=_optional_float(wall_config.get("lumen_reference_lower_fraction", None)),
            lumen_reference_upper_fraction=_optional_float(wall_config.get("lumen_reference_upper_fraction", None)),
            lumen_reference_statistic=str(wall_config.get("lumen_reference_statistic", "median")),
            require_lumen_seed_connectivity=bool(wall_config.get("require_lumen_seed_connectivity", False)),
            use_input_aorta_as_lumen_floor=bool(wall_config.get("use_input_aorta_as_lumen_floor", False)),
            lumen_floor_mask=lumen_floor_mask,
            smooth_lumen_profile_mm=float(wall_config.get("smooth_lumen_profile_mm", 10.0)),
            min_core_voxels_per_slice=int(wall_config.get("min_core_voxels_per_slice", 20)),
            wall_hu_min=float(wall_config.get("wall_hu_min", -30.0)),
            wall_hu_max=float(wall_config.get("wall_hu_max", 1200.0)),
            exclude_fat_from_wall=bool(wall_config.get("exclude_fat_from_wall", True)),
            exclude_calcification_hu=_optional_float(wall_config.get("exclude_calcification_hu", None)),
            include_calcification_in_wall=bool(wall_config.get("include_calcification_in_wall", True)),
            lumen_correction_enabled=bool(wall_config.get("lumen_correction_enabled", False)),
            lumen_correction_outer_mm=float(wall_config.get("lumen_correction_outer_mm", 2.0)),
            lumen_correction_close_radius_mm=float(wall_config.get("lumen_correction_close_radius_mm", 1.0)),
            lumen_correction_lower_margin_hu=_optional_float(wall_config.get("lumen_correction_lower_margin_hu", None)),
            lumen_correction_min_hu=_optional_float(wall_config.get("lumen_correction_min_hu", None)),
            lumen_correction_max_above_reference_hu=_optional_float(
                wall_config.get("lumen_correction_max_above_reference_hu", None)
            ),
            software_version=software_version,
        )
        wall_frame = wall_result.features
        if bool(wall_config.get("save_masks", True)) and bool(config["outputs"].get("save_masks", True)):
            for stale_wall_path in masks_dir.glob(f"{args.case_id}_aortic_wall_*from_fat*.nii.gz"):
                stale_wall_path.unlink()
            write_mask_like(crop_region.paste(wall_result.contrast_lumen_mask), image.image, masks_dir / f"{args.case_id}_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz")
            write_mask_like(crop_region.paste(wall_result.fat_support_mask), image.image, masks_dir / f"{args.case_id}_aortic_wall_fat_support_0_5mm.nii.gz")
            write_mask_like(crop_region.paste(wall_result.closed_outer_envelope_mask), image.image, masks_dir / f"{args.case_id}_aortic_wall_outer_closed_from_fat_5mm.nii.gz")
            write_mask_like(crop_region.paste(wall_result.wall_candidate_mask), image.image, masks_dir / f"{args.case_id}_aortic_wall_candidate_from_fat_lumen.nii.gz")
            write_mask_like(crop_region.paste(wall_result.hu_refined_aorta_mask), image.image, masks_dir / f"{args.case_id}_aortic_wall_hu_refined_aorta_trace.nii.gz")
            write_label_like(crop_region.paste(wall_result.labelmap), image.image, masks_dir / f"{args.case_id}_aortic_wall_from_fat_lumen_labels.nii.gz")

    write_csv(fat_frame, features_dir / "fat_omics_features.csv")
    write_csv(wall_frame, features_dir / "wall_from_fat_features.csv")
    rebuild_modeling_wide(features_dir)
    print(f"Saved fat/wall outputs to {outdir.resolve()}")


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _read_segment_labels(masks_dir: Path, case_id: str, cleaned_mask: np.ndarray) -> np.ndarray:
    path = masks_dir / f"{case_id}_aorta_segments_v1.nii.gz"
    if not path.exists():
        return whole_aorta_segment_mask(cleaned_mask)
    return read_volume(path).array.astype(np.uint8)


if __name__ == "__main__":
    main()
