#!/usr/bin/env python
"""Run calcification and calcium-omics stages from saved base masks."""

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
from aorta_cta_radiomics.calcification import (
    extract_calcification_masks,
    extract_dynamic_wall_calcification,
    summarize_calcification,
    summarize_dynamic_wall_calcification,
)
from aorta_cta_radiomics.calcium_omics import summarize_calcium_omics
from aorta_cta_radiomics.config import load_config
from aorta_cta_radiomics.crop import crop_region_for_mask
from aorta_cta_radiomics.features import write_csv
from aorta_cta_radiomics.io import read_mask, read_volume, write_mask_like
from aorta_cta_radiomics.shells import create_aorta_wall_band_masks, local_shell_around_mask
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--aorta-mask", default=None, type=Path, help="Optional cleaned aorta mask override.")
    parser.add_argument("--crop-margin-mm", default=8.0, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outdir = args.outdir
    masks_dir = outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    image = read_volume(args.image)
    cleaned_mask_path = args.aorta_mask or masks_dir / f"{args.case_id}_aorta_mask_cleaned.nii.gz"
    cleaned_mask = read_mask(cleaned_mask_path).array.astype(bool)
    if cleaned_mask.shape != image.array.shape:
        raise ValueError("Image and cleaned aorta mask must have the same shape.")
    crop_region = crop_region_for_mask(cleaned_mask, image.spacing_xyz, margin_mm=float(args.crop_margin_mm))
    image_array = crop_region.crop(image.array)
    cleaned_crop = crop_region.crop(cleaned_mask)

    calc_config = config.get("calcification", {})
    if not bool(calc_config.get("enabled", True)):
        write_csv(pd.DataFrame(), features_dir / "calcification_features.csv")
        write_csv(pd.DataFrame(), features_dir / "calcium_omics_features.csv")
        rebuild_modeling_wide(features_dir)
        return

    wall_band_masks = _load_or_create_wall_band_masks(
        cleaned_mask=cleaned_mask,
        spacing_xyz=image.spacing_xyz,
        masks_dir=masks_dir,
        case_id=args.case_id,
        image_ref=image.image,
        config=config,
    )

    calc_roi_name = str(calc_config.get("roi", "aorta_wall_band"))
    calc_roi = cleaned_mask if calc_roi_name == "aorta_mask" else wall_band_masks.get(calc_roi_name)
    if calc_roi is None:
        raise ValueError(f"Configured calcification ROI '{calc_roi_name}' was not found.")
    calc_roi_crop = crop_region.crop(calc_roi)

    calcium_masks = extract_calcification_masks(
        image=image_array,
        roi_mask=calc_roi_crop,
        thresholds_hu=list(calc_config.get("thresholds_hu", [130, 300, 500, 600])),
    )
    calcium_masks_full = {threshold: crop_region.paste(mask) for threshold, mask in calcium_masks.items()}
    if bool(calc_config.get("save_masks", True)):
        for threshold, calcium_mask in calcium_masks_full.items():
            write_mask_like(
                calcium_mask,
                image.image,
                masks_dir / _calcification_mask_filename(args.case_id, calc_roi_name, threshold),
            )

    software_version = str(config["outputs"].get("software_version", __version__))
    calcification_frame = summarize_calcification(
        image=image_array,
        calcium_masks=calcium_masks,
        spacing_xyz=image.spacing_xyz,
        case_id=args.case_id,
        region=calc_roi_name,
        mask_name=calc_roi_name,
        software_version=software_version,
    )

    dynamic_calcification = None
    dynamic_mask_name = ""
    dynamic_config = calc_config.get("dynamic_wall", {})
    if bool(dynamic_config.get("enabled", False)):
        dynamic_calcification = extract_dynamic_wall_calcification(
            image=image_array,
            aorta_mask=cleaned_crop,
            spacing_xyz=image.spacing_xyz,
            seed_threshold_hu=float(dynamic_config.get("seed_threshold_hu", 500.0)),
            lumen_margin_hu=float(dynamic_config.get("lumen_margin_hu", 75.0)),
            min_candidate_hu=float(dynamic_config.get("min_candidate_hu", 300.0)),
            lumen_core_distance_mm=float(dynamic_config.get("lumen_core_distance_mm", 5.0)),
            search_internal_mm=float(dynamic_config.get("search_internal_mm", 5.0)),
            search_external_mm=float(dynamic_config.get("search_external_mm", 2.0)),
            smooth_lumen_profile_mm=float(dynamic_config.get("smooth_lumen_profile_mm", 10.0)),
            min_core_voxels_per_slice=int(dynamic_config.get("min_core_voxels_per_slice", 20)),
            exclude_external_contrast_touching=bool(dynamic_config.get("exclude_external_contrast_touching", True)),
            external_contrast_tolerance_hu=float(dynamic_config.get("external_contrast_tolerance_hu", 75.0)),
        )
        seed_threshold = int(float(dynamic_config.get("seed_threshold_hu", 500.0)))
        dynamic_mask_name = f"aorta_wall_dynamic_seed{seed_threshold}HU"
        dynamic_mask_full = crop_region.paste(dynamic_calcification.mask)
        if bool(calc_config.get("save_masks", True)):
            write_mask_like(crop_region.paste(dynamic_calcification.lumen_core_mask), image.image, masks_dir / f"{args.case_id}_aorta_lumen_core_for_dynamic_threshold.nii.gz")
            write_mask_like(crop_region.paste(dynamic_calcification.search_roi_mask), image.image, masks_dir / f"{args.case_id}_aorta_wall_calcium_search_band.nii.gz")
            write_mask_like(crop_region.paste(dynamic_calcification.external_contrast_like_mask), image.image, masks_dir / f"{args.case_id}_aorta_external_contrast_like_for_dynamic_threshold.nii.gz")
            write_mask_like(crop_region.paste(dynamic_calcification.external_contrast_rejected_mask), image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_rejected_external_contrast_touching.nii.gz")
            write_mask_like(crop_region.paste(dynamic_calcification.high_confidence_seed_mask), image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_high_confidence_seed.nii.gz")
            write_mask_like(crop_region.paste(dynamic_calcification.candidate_mask), image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}_candidate.nii.gz")
            write_mask_like(dynamic_mask_full, image.image, masks_dir / f"{args.case_id}_calcification_{dynamic_mask_name}.nii.gz")
        dynamic_burden = summarize_calcification(
            image=image_array,
            calcium_masks={f"dynamic_lumen_referenced_seed{seed_threshold}HU": dynamic_calcification.mask},
            spacing_xyz=image.spacing_xyz,
            case_id=args.case_id,
            region="aorta_wall_dynamic",
            mask_name=dynamic_mask_name,
            software_version=software_version,
        )
        calcification_frame = pd.concat(
            [
                calcification_frame,
                dynamic_burden,
                summarize_dynamic_wall_calcification(
                    dynamic_calcification,
                    case_id=args.case_id,
                    mask_name=dynamic_mask_name,
                    software_version=software_version,
                ),
            ],
            ignore_index=True,
        )

    calcium_seed = (
        dynamic_mask_full
        if dynamic_calcification is not None and dynamic_calcification.mask.any()
        else (_highest_nonempty_mask(calcium_masks_full) if calcium_masks_full else np.zeros_like(cleaned_mask, dtype=bool))
    )
    calcification_local_shell = local_shell_around_mask(
        seed_mask=calcium_seed,
        exclusion_mask=cleaned_mask,
        spacing_xyz=image.spacing_xyz,
        outer_mm=float(config["shells"].get("calcification_local_outer_mm", 5.0)),
    )
    write_mask_like(calcification_local_shell, image.image, masks_dir / f"{args.case_id}_shell_calcification_local.nii.gz")

    calcium_omics_frame = pd.DataFrame()
    if calcium_seed.any():
        centerline_frame = _read_csv_or_none(features_dir / "centerline_points.csv")
        segment_labels = _read_segment_labels(masks_dir, args.case_id, cleaned_mask)
        if dynamic_calcification is not None and dynamic_calcification.mask.any():
            threshold_label = f"dynamic_lumen_referenced_seed{int(float(dynamic_config.get('seed_threshold_hu', 500.0)))}HU"
            mask_name = dynamic_mask_name
        else:
            threshold_label = f"{_highest_nonempty_threshold(calcium_masks_full)}HU"
            mask_name = calc_roi_name
        calcium_omics_frame = summarize_calcium_omics(
            image=image.array,
            calcium_mask=calcium_seed,
            aorta_mask=cleaned_mask,
            spacing_xyz=image.spacing_xyz,
            case_id=args.case_id,
            mask_name=mask_name,
            threshold_label=threshold_label,
            centerline_points=centerline_frame,
            segment_labels=segment_labels,
            segment_names=SEGMENT_LABELS,
            software_version=software_version,
        )

    write_csv(calcification_frame, features_dir / "calcification_features.csv")
    write_csv(calcium_omics_frame, features_dir / "calcium_omics_features.csv")
    rebuild_modeling_wide(features_dir)
    print(f"Saved calcium outputs to {outdir.resolve()}")


def _load_or_create_wall_band_masks(
    cleaned_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    masks_dir: Path,
    case_id: str,
    image_ref: object,
    config: dict,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name in ["aorta_wall_internal", "aorta_wall_external", "aorta_wall_band"]:
        path = masks_dir / f"{case_id}_{name}.nii.gz"
        if path.exists():
            masks[name] = read_mask(path).array.astype(bool)
    missing = {"aorta_wall_internal", "aorta_wall_external", "aorta_wall_band"} - set(masks)
    if missing:
        masks.update(
            create_aorta_wall_band_masks(
                cleaned_mask,
                spacing_xyz,
                internal_mm=float(config["shells"].get("aorta_wall_internal_mm", 2.0)),
                external_mm=float(config["shells"].get("aorta_wall_external_mm", 2.0)),
            )
        )
        for name, mask in masks.items():
            write_mask_like(mask, image_ref, masks_dir / f"{case_id}_{name}.nii.gz")
    return masks


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


def _highest_nonempty_mask(masks: dict[int, np.ndarray]) -> np.ndarray:
    for _, mask in sorted(masks.items(), reverse=True):
        if np.asarray(mask).any():
            return mask
    first = next(iter(masks.values()))
    return np.zeros_like(first, dtype=bool)


def _highest_nonempty_threshold(masks: dict[int, np.ndarray]) -> int:
    for threshold, mask in sorted(masks.items(), reverse=True):
        if np.asarray(mask).any():
            return int(threshold)
    return int(next(iter(masks.keys())))


def _calcification_mask_filename(case_id: str, roi_name: str, threshold: int | float) -> str:
    threshold_text = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
    if roi_name == "aorta_mask":
        return f"{case_id}_calcification_thr{threshold_text}HU.nii.gz"
    safe_roi = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in roi_name)
    return f"{case_id}_calcification_{safe_roi}_thr{threshold_text}HU.nii.gz"


if __name__ == "__main__":
    main()
