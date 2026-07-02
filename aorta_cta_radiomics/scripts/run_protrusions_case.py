#!/usr/bin/env python
"""Run lumen/wall protrusion candidate detection from saved masks."""

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
from aorta_cta_radiomics.cli import (
    _optional_float,
    _parse_float_list,
    _write_thresholded_lumen_protrusion_qc_masks,
)
from aorta_cta_radiomics.config import load_config
from aorta_cta_radiomics.features import write_csv
from aorta_cta_radiomics.io import read_mask, read_volume, write_label_like, write_mask_like
from aorta_cta_radiomics.lumen_protrusions import detect_lumen_protrusions
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    outdir = args.outdir
    masks_dir = outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    image = read_volume(args.image)
    cleaned_mask = read_mask(masks_dir / f"{args.case_id}_aorta_mask_cleaned.nii.gz").array.astype(bool)
    segment_labels = _read_segment_labels(masks_dir, args.case_id, cleaned_mask)
    software_version = str(config["outputs"].get("software_version", __version__))

    summaries: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    points: list[pd.DataFrame] = []

    wall_config = config.get("wall_from_fat", {})
    wall_protrusion_config = wall_config.get("protrusions", {})
    legacy_protrusion_config = config.get("lumen_protrusions", {})
    if bool(legacy_protrusion_config.get("enabled", False)):
        if not bool(wall_protrusion_config.get("enabled", False)):
            raise ValueError(
                "The staged protrusion detector now requires wall_from_fat.protrusions.enabled=true. "
                "Run the fat-wall stage first and configure protrusions under wall_from_fat."
            )
        logger.warning("Ignoring legacy top-level lumen_protrusions; using wall_from_fat.protrusions instead.")

    if bool(wall_protrusion_config.get("enabled", False)):
        lumen_path = _require_mask(
            masks_dir / f"{args.case_id}_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz",
            "wall-from-fat lumen",
        )
        wall_path = _require_mask(
            masks_dir / f"{args.case_id}_aortic_wall_candidate_from_fat_lumen.nii.gz",
            "wall-from-fat wall",
        )
        lumen = read_mask(lumen_path).array.astype(bool)
        wall = read_mask(wall_path).array.astype(bool)
        calcium = _read_calcium_mask_if_available(masks_dir, args.case_id, lumen.shape)
        if calcium is not None and calcium.any():
            lumen = lumen & ~calcium
            wall = wall | calcium
            if bool(wall_protrusion_config.get("save_masks", True)) and bool(config["outputs"].get("save_masks", True)):
                write_mask_like(
                    lumen,
                    image.image,
                    masks_dir / f"{args.case_id}_protrusion_lumen_no_calcium.nii.gz",
                )
                write_mask_like(
                    wall,
                    image.image,
                    masks_dir / f"{args.case_id}_protrusion_wall_plus_calcium.nii.gz",
                )
        result = _run_detection(
            image=image,
            lumen_mask=lumen,
            analysis_mask_override=lumen | wall,
            contrast_exclude_mask=calcium,
            segment_labels=segment_labels,
            config=wall_protrusion_config,
            case_id=args.case_id,
            software_version=software_version,
            default_radial_sample_step_mm=0.25,
            default_angular_bins=144,
            default_detect_outward=True,
        )
        summaries.append(result.summary_features.assign(analysis_source="wall_from_fat_lumen_wall"))
        candidates.append(result.candidates.assign(analysis_source="wall_from_fat_lumen_wall"))
        points.append(result.point_features.assign(analysis_source="wall_from_fat_lumen_wall"))
        if bool(wall_protrusion_config.get("save_masks", True)) and bool(config["outputs"].get("save_masks", True)):
            _remove_stale_wall_lumen_protrusion_masks(masks_dir, args.case_id)
            _write_thresholded_lumen_protrusion_qc_masks(
                candidates=result.candidates,
                case_id=args.case_id,
                reference_image=image.image,
                masks_dir=masks_dir,
                write_label_like=write_label_like,
                inward_core_labelmap=result.inward_aorta_surface_core_labelmap,
                outward_core_labelmap=result.outward_aorta_surface_core_labelmap,
                inward_projection_labelmap=result.inward_aorta_surface_projection_labelmap,
                outward_projection_labelmap=result.outward_aorta_surface_projection_labelmap,
                inward_native_labelmap=result.inward_aorta_surface_native_labelmap,
                outward_native_labelmap=result.outward_aorta_surface_native_labelmap,
                inward_thresholds_mm=_parse_float_list(
                    wall_protrusion_config.get("inward_qc_depth_thresholds_mm", [2, 3, 4])
                ),
                outward_thresholds_mm=_parse_float_list(
                    wall_protrusion_config.get("outward_qc_depth_thresholds_mm", [1.5, 2, 3, 4])
                ),
                sources=list(
                    wall_protrusion_config.get(
                        "thresholded_qc_sources", ["aorta_surface_native", "aorta_surface_core"]
                    )
                ),
                output_prefix="wall_lumen_protrusion",
            )

    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    candidates_frame = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    points_frame = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    write_csv(summary_frame, features_dir / "lumen_protrusion_summary_features.csv")
    write_csv(candidates_frame, features_dir / "lumen_protrusion_candidates.csv")
    write_csv(points_frame, features_dir / "lumen_protrusion_point_features.csv")
    rebuild_modeling_wide(features_dir)
    print(f"Saved protrusion outputs to {outdir.resolve()}")


def _run_detection(
    image: object,
    lumen_mask: np.ndarray,
    analysis_mask_override: np.ndarray | None,
    contrast_exclude_mask: np.ndarray | None,
    segment_labels: np.ndarray,
    config: dict,
    case_id: str,
    software_version: str,
    default_radial_sample_step_mm: float = 0.5,
    default_angular_bins: int = 72,
    default_detect_outward: bool = False,
):
    return detect_lumen_protrusions(
        lumen_mask=lumen_mask,
        spacing_xyz=image.spacing_xyz,
        case_id=case_id,
        image_hu=image.array,
        segment_labels=segment_labels,
        segment_names=SEGMENT_LABELS,
        analysis_mask_override=analysis_mask_override,
        contrast_exclude_mask=contrast_exclude_mask,
        centerline_interval_mm=float(config.get("centerline_interval_mm", 2.0)),
        centerline_smoothing_mm=float(config.get("centerline_smoothing_mm", 6.0)),
        plane_spacing_mm=float(config.get("plane_spacing_mm", 0.75)),
        radial_sample_step_mm=float(config.get("radial_sample_step_mm", default_radial_sample_step_mm)),
        max_radius_mm=float(config.get("max_radius_mm", 35.0)),
        angular_bins=int(config.get("angular_bins", default_angular_bins)),
        angular_median_window_deg=float(config.get("angular_median_window_deg", 50.0)),
        inward_angular_median_window_deg=_optional_float(config.get("inward_angular_median_window_deg", None)),
        outward_angular_median_window_deg=_optional_float(config.get("outward_angular_median_window_deg", None)),
        longitudinal_smoothing_mm=float(config.get("longitudinal_smoothing_mm", 12.0)),
        inward_longitudinal_smoothing_mm=_optional_float(config.get("inward_longitudinal_smoothing_mm", None)),
        outward_longitudinal_smoothing_mm=_optional_float(config.get("outward_longitudinal_smoothing_mm", None)),
        min_depth_mm=float(config.get("min_depth_mm", 2.0)),
        outward_min_depth_mm=_optional_float(config.get("outward_min_depth_mm", None)),
        high_risk_depth_mm=float(config.get("high_risk_depth_mm", 4.0)),
        min_angular_width_deg=float(config.get("min_angular_width_deg", 5.0)),
        max_angular_width_deg=float(config.get("max_angular_width_deg", 90.0)),
        outward_min_angular_width_deg=_optional_float(config.get("outward_min_angular_width_deg", None)),
        outward_max_angular_width_deg=_optional_float(config.get("outward_max_angular_width_deg", None)),
        min_length_mm=float(config.get("min_length_mm", 1.0)),
        max_length_mm=float(config.get("max_length_mm", 25.0)),
        outward_min_length_mm=_optional_float(config.get("outward_min_length_mm", None)),
        outward_max_length_mm=_optional_float(config.get("outward_max_length_mm", None)),
        min_peak_prominence_mm=_optional_float(config.get("min_peak_prominence_mm", None)),
        outward_min_peak_prominence_mm=_optional_float(config.get("outward_min_peak_prominence_mm", None)),
        max_median_depth_fraction=_optional_float(config.get("max_median_depth_fraction", None)),
        outward_max_median_depth_fraction=_optional_float(config.get("outward_max_median_depth_fraction", None)),
        min_focality_ratio=_optional_float(config.get("min_focality_ratio", None)),
        outward_min_focality_ratio=_optional_float(config.get("outward_min_focality_ratio", None)),
        end_margin_mm=float(config.get("end_margin_mm", 10.0)),
        analysis_inner_layer_mm=float(config.get("analysis_inner_layer_mm", 0.0)),
        analysis_outer_layer_mm=float(config.get("analysis_outer_layer_mm", 0.0)),
        patch_longitudinal_padding_mm=float(config.get("patch_longitudinal_padding_mm", 2.0)),
        patch_angular_padding_deg=float(config.get("patch_angular_padding_deg", 10.0)),
        surface_projection_depth_mm=float(config.get("surface_projection_depth_mm", 1.0)),
        surface_core_relative_threshold=float(config.get("surface_core_relative_threshold", 0.75)),
        surface_core_depth_window_mm=float(config.get("surface_core_depth_window_mm", 1.0)),
        surface_core_longitudinal_padding_mm=float(config.get("surface_core_longitudinal_padding_mm", 0.0)),
        surface_core_angular_padding_deg=float(config.get("surface_core_angular_padding_deg", 2.5)),
        detect_inward=bool(config.get("detect_inward", True)),
        detect_outward=bool(config.get("detect_outward", default_detect_outward)),
        intensity_gate_enabled=bool(config.get("intensity_gate_enabled", True)),
        centerline_core_radius_mm=float(config.get("centerline_core_radius_mm", 2.0)),
        contrast_lower_margin_hu=float(config.get("contrast_lower_margin_hu", 120.0)),
        min_contrast_hu=float(config.get("min_contrast_hu", 150.0)),
        max_contrast_hu_above_reference=_optional_float(config.get("max_contrast_hu_above_reference", None)),
        contrast_reference_lower_fraction=_optional_float(config.get("contrast_reference_lower_fraction", None)),
        contrast_reference_upper_fraction=_optional_float(config.get("contrast_reference_upper_fraction", None)),
        max_external_contrast_component_volume_mm3=_optional_float(
            config.get("max_external_contrast_component_volume_mm3", None)
        ),
        max_candidate_outside_aorta_fraction=_optional_float(
            config.get("max_candidate_outside_aorta_fraction", None)
        ),
        clip_candidate_masks_to_analysis_mask=bool(config.get("clip_candidate_masks_to_analysis_mask", True)),
        software_version=software_version,
    )


def _write_core_outputs(result: object, case_id: str, reference_image: object, masks_dir: Path, prefix: str) -> None:
    write_mask_like(result.analysis_mask, reference_image, masks_dir / f"{case_id}_{prefix}_analysis_surface_band.nii.gz")
    write_mask_like(result.contrast_like_mask, reference_image, masks_dir / f"{case_id}_{prefix}_contrast_like_from_centerline_hu.nii.gz")
    write_label_like(result.candidate_labelmap, reference_image, masks_dir / f"{case_id}_{prefix}_candidate_labels.nii.gz")
    write_label_like(result.inward_candidate_labelmap, reference_image, masks_dir / f"{case_id}_{prefix}_inward_candidate_labels.nii.gz")
    write_label_like(result.outward_candidate_labelmap, reference_image, masks_dir / f"{case_id}_{prefix}_outward_ulcer_like_candidate_labels.nii.gz")
    write_label_like(result.aorta_surface_core_labelmap, reference_image, masks_dir / f"{case_id}_{prefix}_aorta_surface_core_labels_3d.nii.gz")
    write_label_like(result.aorta_surface_native_labelmap, reference_image, masks_dir / f"{case_id}_{prefix}_aorta_surface_native_labels_3d.nii.gz")


def _require_mask(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description} mask: {path}. Run the fat-wall stage before protrusions.")
    return path


def _read_calcium_mask_if_available(masks_dir: Path, case_id: str, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    candidates = [
        masks_dir / f"{case_id}_calcification_aorta_wall_dynamic_seed500HU.nii.gz",
        masks_dir / f"{case_id}_calcification_aorta_wall_dynamic_seed500HU_candidate.nii.gz",
        masks_dir / f"{case_id}_calcification_aorta_wall_band_thr500HU.nii.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        calcium = read_mask(path).array.astype(bool)
        if calcium.shape != expected_shape:
            raise ValueError(f"Calcium mask shape does not match wall/lumen masks: {path}")
        logger.info("Using calcium as wall and excluding it from contrast sampling: %s", path)
        return calcium
    logger.info("No calcium mask found; protrusion detector will use wall-from-fat lumen/wall without calcium remapping.")
    return None


def _remove_stale_wall_lumen_protrusion_masks(masks_dir: Path, case_id: str) -> None:
    for path in masks_dir.glob(f"{case_id}_wall_lumen_protrusion_*depth_ge*labels_3d.nii.gz"):
        path.unlink()


def _read_segment_labels(masks_dir: Path, case_id: str, cleaned_mask: np.ndarray) -> np.ndarray:
    path = masks_dir / f"{case_id}_aorta_segments_v1.nii.gz"
    if not path.exists():
        return whole_aorta_segment_mask(cleaned_mask)
    return read_volume(path).array.astype(np.uint8)


if __name__ == "__main__":
    main()
