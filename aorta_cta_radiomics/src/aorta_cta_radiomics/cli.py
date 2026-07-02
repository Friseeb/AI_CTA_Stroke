"""Command-line interface for the aorta CTA radiomics pipeline."""

from __future__ import annotations

import logging
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

from . import __version__

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    qc: pd.DataFrame
    calcification: pd.DataFrame
    calcium_omics: pd.DataFrame
    fat_omics: pd.DataFrame
    lumen_protrusion_summary: pd.DataFrame
    lumen_protrusion_candidates: pd.DataFrame
    lumen_protrusion_point_features: pd.DataFrame
    radiomics: pd.DataFrame
    case_level_features: pd.DataFrame
    centerline_points: pd.DataFrame
    centerline_point_features: pd.DataFrame
    segment_level_features: pd.DataFrame
    wall_from_fat_features: pd.DataFrame
    wide_features: pd.DataFrame


def run_pipeline_case(
    image_path: str | Path,
    aorta_mask_path: str | Path,
    case_id: str,
    outdir: str | Path,
    config_path: str | Path | None = None,
) -> CaseResult:
    """Run the version-1 pipeline for one CTA/aorta-mask pair."""
    import numpy as np
    import pandas as pd

    from .aorta_segments import SEGMENT_LABELS, segment_summary, whole_aorta_segment_mask
    from .calcium_omics import summarize_calcium_omics
    from .calcification import (
        extract_calcification_masks,
        extract_dynamic_wall_calcification,
        summarize_calcification,
        summarize_dynamic_wall_calcification,
    )
    from .centerline import approximate_centerline_by_slices
    from .config import load_config
    from .fat_omics import extract_periaortic_fat_omics
    from .fat_wall import extract_fat_closed_aortic_wall
    from .features import ensure_feature_columns, long_to_wide_features, write_csv
    from .io import load_image_and_mask, write_label_like, write_mask_like
    from .lumen_protrusions import detect_lumen_protrusions
    from .lumen_geometry import slice_geometry_features
    from .preprocess import clean_aorta_mask
    from .segmentation_qc import calculate_qc_metrics, qc_metrics_to_frame
    from .shells import create_aorta_wall_band_masks, create_base_shells, local_shell_around_mask

    config = load_config(config_path)
    outdir = Path(outdir)
    project_root = Path(__file__).resolve().parents[2]

    masks_dir = outdir / "masks" / case_id
    figures_dir = outdir / "figures" / case_id
    qc_dir = outdir / "qc"
    features_dir = outdir / "features"
    for directory in [masks_dir, figures_dir, qc_dir, features_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    image, raw_mask, mask_resampled = load_image_and_mask(
        image_path=image_path,
        mask_path=aorta_mask_path,
        resample_mask_if_needed=bool(config["image"]["resample_mask_if_needed"]),
    )
    spacing_xyz = image.spacing_xyz
    software_version = str(config["outputs"].get("software_version", __version__))

    cleaned_mask, cleaning_report = clean_aorta_mask(
        raw_mask.array,
        keep_largest_component=bool(config["mask_cleaning"]["keep_largest_component"]),
        fill_holes=bool(config["mask_cleaning"]["fill_holes"]),
        min_component_voxels=int(config["mask_cleaning"]["min_component_voxels"]),
    )
    cleaned_mask_path = masks_dir / f"{case_id}_aorta_mask_cleaned.nii.gz"
    write_mask_like(cleaned_mask, image.image, cleaned_mask_path)

    qc_metrics = calculate_qc_metrics(
        image=image.array,
        mask=cleaned_mask,
        spacing_xyz=spacing_xyz,
        case_id=case_id,
        components_before_cleaning=cleaning_report.components_before,
        mask_resampled=mask_resampled,
        small_mask_volume_mm3=float(config["mask_cleaning"]["small_mask_volume_mm3"]),
        large_mask_volume_mm3=float(config["mask_cleaning"]["large_mask_volume_mm3"]),
    )
    qc_frame = qc_metrics_to_frame(qc_metrics)

    calcification_enabled = bool(config["calcification"].get("enabled", True))
    radiomics_regions = (
        set(config.get("radiomics", {}).get("regions", []))
        if bool(config.get("radiomics", {}).get("enabled", False))
        else set()
    )
    shell_specs = list(config["shells"].get("base", []))
    shell_masks = create_base_shells(cleaned_mask, spacing_xyz, shell_specs) if shell_specs else {}
    if calcification_enabled or "aorta_wall_band" in radiomics_regions:
        wall_band_masks = create_aorta_wall_band_masks(
            cleaned_mask,
            spacing_xyz,
            internal_mm=float(config["shells"].get("aorta_wall_internal_mm", 2.0)),
            external_mm=float(config["shells"].get("aorta_wall_external_mm", 2.0)),
        )
        shell_masks.update(wall_band_masks)
    for name, shell_mask in shell_masks.items():
        write_mask_like(shell_mask, image.image, masks_dir / f"{case_id}_{name}.nii.gz")

    calcium_masks = {}
    calcification_frame = pd.DataFrame()
    calc_roi_name = ""
    if calcification_enabled:
        calc_roi_name = str(config["calcification"]["roi"])
        calc_roi = cleaned_mask if calc_roi_name == "aorta_mask" else shell_masks.get(calc_roi_name)
        if calc_roi is None:
            raise ValueError(f"Configured calcification ROI '{calc_roi_name}' was not found.")

        calcium_masks = extract_calcification_masks(
            image=image.array,
            roi_mask=calc_roi,
            thresholds_hu=list(config["calcification"]["thresholds_hu"]),
        )
        if bool(config["calcification"]["save_masks"]):
            for threshold, calcium_mask in calcium_masks.items():
                write_mask_like(
                    calcium_mask,
                    image.image,
                    masks_dir / _calcification_mask_filename(case_id, calc_roi_name, threshold),
                )

        calcification_frame = summarize_calcification(
            image=image.array,
            calcium_masks=calcium_masks,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            region=calc_roi_name,
            mask_name=calc_roi_name,
            software_version=software_version,
        )

    dynamic_calcification = None
    dynamic_mask_name = ""
    dynamic_config = config["calcification"].get("dynamic_wall", {}) if calcification_enabled else {}
    if bool(dynamic_config.get("enabled", False)):
        dynamic_calcification = extract_dynamic_wall_calcification(
            image=image.array,
            aorta_mask=cleaned_mask,
            spacing_xyz=spacing_xyz,
            seed_threshold_hu=float(dynamic_config.get("seed_threshold_hu", 500.0)),
            lumen_margin_hu=float(dynamic_config.get("lumen_margin_hu", 75.0)),
            min_candidate_hu=float(dynamic_config.get("min_candidate_hu", 300.0)),
            lumen_core_distance_mm=float(dynamic_config.get("lumen_core_distance_mm", 5.0)),
            search_internal_mm=float(dynamic_config.get("search_internal_mm", 5.0)),
            search_external_mm=float(dynamic_config.get("search_external_mm", 2.0)),
            smooth_lumen_profile_mm=float(dynamic_config.get("smooth_lumen_profile_mm", 10.0)),
            min_core_voxels_per_slice=int(dynamic_config.get("min_core_voxels_per_slice", 20)),
            exclude_external_contrast_touching=bool(
                dynamic_config.get("exclude_external_contrast_touching", True)
            ),
            external_contrast_tolerance_hu=float(dynamic_config.get("external_contrast_tolerance_hu", 75.0)),
        )
        seed_threshold = int(float(dynamic_config.get("seed_threshold_hu", 500.0)))
        dynamic_mask_name = f"aorta_wall_dynamic_seed{seed_threshold}HU"
        if bool(config["calcification"]["save_masks"]):
            write_mask_like(
                dynamic_calcification.lumen_core_mask,
                image.image,
                masks_dir / f"{case_id}_aorta_lumen_core_for_dynamic_threshold.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.search_roi_mask,
                image.image,
                masks_dir / f"{case_id}_aorta_wall_calcium_search_band.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.external_contrast_like_mask,
                image.image,
                masks_dir / f"{case_id}_aorta_external_contrast_like_for_dynamic_threshold.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.external_contrast_rejected_mask,
                image.image,
                masks_dir / f"{case_id}_calcification_{dynamic_mask_name}_rejected_external_contrast_touching.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.high_confidence_seed_mask,
                image.image,
                masks_dir / f"{case_id}_calcification_{dynamic_mask_name}_high_confidence_seed.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.candidate_mask,
                image.image,
                masks_dir / f"{case_id}_calcification_{dynamic_mask_name}_candidate.nii.gz",
            )
            write_mask_like(
                dynamic_calcification.mask,
                image.image,
                masks_dir / f"{case_id}_calcification_{dynamic_mask_name}.nii.gz",
            )
        dynamic_burden = summarize_calcification(
            image=image.array,
            calcium_masks={f"dynamic_lumen_referenced_seed{seed_threshold}HU": dynamic_calcification.mask},
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            region="aorta_wall_dynamic",
            mask_name=dynamic_mask_name,
            software_version=software_version,
        )
        dynamic_threshold_summary = summarize_dynamic_wall_calcification(
            dynamic_calcification,
            case_id=case_id,
            mask_name=dynamic_mask_name,
            software_version=software_version,
        )
        calcification_frame = pd.concat(
            [calcification_frame, dynamic_burden, dynamic_threshold_summary],
            ignore_index=True,
        )

    centerline_frame = approximate_centerline_by_slices(
        cleaned_mask,
        spacing_xyz=spacing_xyz,
        case_id=case_id,
        reference_image=image.image,
    )
    geometry_frame = (
        slice_geometry_features(
            cleaned_mask,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            min_slice_voxels=int(config["geometry"]["min_slice_voxels"]),
            max_branch_link_distance_mm=float(config["geometry"].get("max_branch_link_distance_mm", 20.0)),
            max_components_per_slice=int(config["geometry"].get("max_components_per_slice", 4)),
        )
        if bool(config["geometry"]["enabled"])
        else pd.DataFrame()
    )

    calcium_seed = (
        dynamic_calcification.mask
        if dynamic_calcification is not None and dynamic_calcification.mask.any()
        else (_highest_nonempty_mask(calcium_masks) if calcium_masks else np.zeros_like(cleaned_mask, dtype=bool))
    )
    if calcification_enabled:
        calcification_local_shell = local_shell_around_mask(
            seed_mask=calcium_seed,
            exclusion_mask=cleaned_mask,
            spacing_xyz=spacing_xyz,
            outer_mm=float(config["shells"]["calcification_local_outer_mm"]),
        )
        shell_masks["shell_calcification_local"] = calcification_local_shell
        write_mask_like(
            calcification_local_shell,
            image.image,
            masks_dir / f"{case_id}_shell_calcification_local.nii.gz",
        )

    segment_labels = whole_aorta_segment_mask(cleaned_mask)
    segment_path = masks_dir / f"{case_id}_aorta_segments_v1.nii.gz"
    write_mask_like(segment_labels, image.image, segment_path)
    segment_frame = segment_summary(segment_labels, spacing_xyz, case_id)
    if calcification_enabled and calcium_seed.any():
        if dynamic_calcification is not None and dynamic_calcification.mask.any():
            calcium_omics_threshold = (
                f"dynamic_lumen_referenced_seed{int(float(dynamic_config.get('seed_threshold_hu', 500.0)))}HU"
            )
            calcium_omics_mask_name = dynamic_mask_name
        else:
            calcium_omics_threshold = f"{_highest_nonempty_threshold(calcium_masks)}HU"
            calcium_omics_mask_name = calc_roi_name
        calcium_omics_frame = summarize_calcium_omics(
            image=image.array,
            calcium_mask=calcium_seed,
            aorta_mask=cleaned_mask,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            mask_name=calcium_omics_mask_name,
            threshold_label=calcium_omics_threshold,
            centerline_points=centerline_frame,
            segment_labels=segment_labels,
            segment_names=SEGMENT_LABELS,
            software_version=software_version,
        )
    else:
        calcium_omics_frame = pd.DataFrame()

    protrusion_config = config.get("lumen_protrusions", {})
    if bool(protrusion_config.get("enabled", False)):
        protrusion_result = detect_lumen_protrusions(
            lumen_mask=cleaned_mask,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            image_hu=image.array,
            segment_labels=segment_labels,
            segment_names=SEGMENT_LABELS,
            centerline_interval_mm=float(protrusion_config.get("centerline_interval_mm", 2.0)),
            centerline_smoothing_mm=float(protrusion_config.get("centerline_smoothing_mm", 6.0)),
            plane_spacing_mm=float(protrusion_config.get("plane_spacing_mm", 0.75)),
            radial_sample_step_mm=float(protrusion_config.get("radial_sample_step_mm", 0.5)),
            max_radius_mm=float(protrusion_config.get("max_radius_mm", 35.0)),
            angular_bins=int(protrusion_config.get("angular_bins", 72)),
            angular_median_window_deg=float(protrusion_config.get("angular_median_window_deg", 50.0)),
            inward_angular_median_window_deg=_optional_float(
                protrusion_config.get("inward_angular_median_window_deg", None)
            ),
            outward_angular_median_window_deg=_optional_float(
                protrusion_config.get("outward_angular_median_window_deg", None)
            ),
            longitudinal_smoothing_mm=float(protrusion_config.get("longitudinal_smoothing_mm", 12.0)),
            inward_longitudinal_smoothing_mm=_optional_float(
                protrusion_config.get("inward_longitudinal_smoothing_mm", None)
            ),
            outward_longitudinal_smoothing_mm=_optional_float(
                protrusion_config.get("outward_longitudinal_smoothing_mm", None)
            ),
            min_depth_mm=float(protrusion_config.get("min_depth_mm", 2.0)),
            outward_min_depth_mm=_optional_float(protrusion_config.get("outward_min_depth_mm", None)),
            high_risk_depth_mm=float(protrusion_config.get("high_risk_depth_mm", 4.0)),
            min_angular_width_deg=float(protrusion_config.get("min_angular_width_deg", 5.0)),
            max_angular_width_deg=float(protrusion_config.get("max_angular_width_deg", 90.0)),
            outward_min_angular_width_deg=_optional_float(
                protrusion_config.get("outward_min_angular_width_deg", None)
            ),
            outward_max_angular_width_deg=_optional_float(
                protrusion_config.get("outward_max_angular_width_deg", None)
            ),
            min_length_mm=float(protrusion_config.get("min_length_mm", 1.0)),
            max_length_mm=float(protrusion_config.get("max_length_mm", 25.0)),
            outward_min_length_mm=_optional_float(protrusion_config.get("outward_min_length_mm", None)),
            outward_max_length_mm=_optional_float(protrusion_config.get("outward_max_length_mm", None)),
            min_peak_prominence_mm=_optional_float(protrusion_config.get("min_peak_prominence_mm", None)),
            outward_min_peak_prominence_mm=_optional_float(
                protrusion_config.get("outward_min_peak_prominence_mm", None)
            ),
            max_median_depth_fraction=_optional_float(protrusion_config.get("max_median_depth_fraction", None)),
            outward_max_median_depth_fraction=_optional_float(
                protrusion_config.get("outward_max_median_depth_fraction", None)
            ),
            min_focality_ratio=_optional_float(protrusion_config.get("min_focality_ratio", None)),
            outward_min_focality_ratio=_optional_float(protrusion_config.get("outward_min_focality_ratio", None)),
            end_margin_mm=float(protrusion_config.get("end_margin_mm", 10.0)),
            analysis_inner_layer_mm=float(protrusion_config.get("analysis_inner_layer_mm", 0.0)),
            analysis_outer_layer_mm=float(protrusion_config.get("analysis_outer_layer_mm", 0.0)),
            patch_longitudinal_padding_mm=float(protrusion_config.get("patch_longitudinal_padding_mm", 2.0)),
            patch_angular_padding_deg=float(protrusion_config.get("patch_angular_padding_deg", 10.0)),
            surface_sheet_thickness_mm=float(protrusion_config.get("surface_sheet_thickness_mm", 1.0)),
            surface_projection_depth_mm=float(protrusion_config.get("surface_projection_depth_mm", 1.0)),
            surface_core_relative_threshold=float(protrusion_config.get("surface_core_relative_threshold", 0.75)),
            surface_core_depth_window_mm=float(protrusion_config.get("surface_core_depth_window_mm", 1.0)),
            surface_core_longitudinal_padding_mm=float(
                protrusion_config.get("surface_core_longitudinal_padding_mm", 0.0)
            ),
            surface_core_angular_padding_deg=float(protrusion_config.get("surface_core_angular_padding_deg", 2.5)),
            detect_inward=bool(protrusion_config.get("detect_inward", True)),
            detect_outward=bool(protrusion_config.get("detect_outward", False)),
            intensity_gate_enabled=bool(protrusion_config.get("intensity_gate_enabled", True)),
            centerline_core_radius_mm=float(protrusion_config.get("centerline_core_radius_mm", 2.0)),
            contrast_lower_margin_hu=float(protrusion_config.get("contrast_lower_margin_hu", 120.0)),
            min_contrast_hu=float(protrusion_config.get("min_contrast_hu", 150.0)),
            max_contrast_hu_above_reference=_optional_float(
                protrusion_config.get("max_contrast_hu_above_reference", None)
            ),
            contrast_reference_lower_fraction=_optional_float(
                protrusion_config.get("contrast_reference_lower_fraction", None)
            ),
            contrast_reference_upper_fraction=_optional_float(
                protrusion_config.get("contrast_reference_upper_fraction", None)
            ),
            max_external_contrast_component_volume_mm3=_optional_float(
                protrusion_config.get("max_external_contrast_component_volume_mm3", None)
            ),
            max_candidate_outside_aorta_fraction=_optional_float(
                protrusion_config.get("max_candidate_outside_aorta_fraction", None)
            ),
            clip_candidate_masks_to_analysis_mask=bool(
                protrusion_config.get(
                    "clip_candidate_masks_to_analysis_mask",
                    protrusion_config.get("clip_candidate_masks_to_lumen", False),
                )
            ),
            software_version=software_version,
        )
        lumen_protrusion_summary = protrusion_result.summary_features
        lumen_protrusion_candidates = protrusion_result.candidates
        lumen_protrusion_point_features = protrusion_result.point_features
        shell_masks["lumen_protrusion_analysis_mask"] = protrusion_result.analysis_mask
        shell_masks["lumen_protrusion_contrast_like_mask"] = protrusion_result.contrast_like_mask
        shell_masks["lumen_protrusion_candidate_mask"] = protrusion_result.candidate_mask
        shell_masks["lumen_protrusion_candidate_boundary"] = protrusion_result.boundary_mask
        shell_masks["lumen_protrusion_inward_candidate_mask"] = protrusion_result.inward_candidate_mask
        shell_masks["lumen_protrusion_outward_ulcer_like_candidate_mask"] = protrusion_result.outward_candidate_mask
        shell_masks["lumen_protrusion_patch_roi"] = protrusion_result.patch_mask
        shell_masks["lumen_protrusion_inward_patch_roi"] = protrusion_result.inward_patch_mask
        shell_masks["lumen_protrusion_outward_ulcer_like_patch_roi"] = protrusion_result.outward_patch_mask
        shell_masks["lumen_protrusion_surface_sheet"] = protrusion_result.surface_sheet_mask
        shell_masks["lumen_protrusion_inward_surface_sheet"] = protrusion_result.inward_surface_sheet_mask
        shell_masks["lumen_protrusion_outward_ulcer_like_surface_sheet"] = protrusion_result.outward_surface_sheet_mask
        shell_masks["lumen_protrusion_aorta_surface_projection"] = protrusion_result.aorta_surface_projection_mask
        shell_masks["lumen_protrusion_inward_aorta_surface_projection"] = (
            protrusion_result.inward_aorta_surface_projection_mask
        )
        shell_masks["lumen_protrusion_outward_ulcer_like_aorta_surface_projection"] = (
            protrusion_result.outward_aorta_surface_projection_mask
        )
        shell_masks["lumen_protrusion_aorta_surface_core"] = protrusion_result.aorta_surface_core_mask
        shell_masks["lumen_protrusion_inward_aorta_surface_core"] = (
            protrusion_result.inward_aorta_surface_core_mask
        )
        shell_masks["lumen_protrusion_outward_ulcer_like_aorta_surface_core"] = (
            protrusion_result.outward_aorta_surface_core_mask
        )
        if bool(protrusion_config.get("save_masks", True)) and bool(config["outputs"]["save_masks"]):
            for stale_path in masks_dir.glob(f"{case_id}_lumen_protrusion*.nii.gz"):
                stale_path.unlink()
            write_mask_like(
                protrusion_result.analysis_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_analysis_surface_band.nii.gz",
            )
            write_mask_like(
                protrusion_result.contrast_like_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_contrast_like_from_centerline_hu.nii.gz",
            )
            write_mask_like(
                protrusion_result.candidate_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_candidate_mask.nii.gz",
            )
            write_label_like(
                protrusion_result.candidate_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_candidate_labels.nii.gz",
            )
            write_mask_like(
                protrusion_result.boundary_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_candidate_boundary.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_candidate_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_candidate_mask.nii.gz",
            )
            write_label_like(
                protrusion_result.inward_candidate_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_candidate_labels.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_boundary_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_candidate_boundary.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_candidate_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_candidate_mask.nii.gz",
            )
            write_label_like(
                protrusion_result.outward_candidate_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_candidate_labels.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_boundary_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_candidate_boundary.nii.gz",
            )
            write_mask_like(
                protrusion_result.patch_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_patch_roi_4mm_in_4mm_out.nii.gz",
            )
            write_label_like(
                protrusion_result.patch_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_patch_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_patch_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_patch_roi_4mm_in_4mm_out.nii.gz",
            )
            write_label_like(
                protrusion_result.inward_patch_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_patch_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_patch_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_patch_roi_4mm_in_4mm_out.nii.gz",
            )
            write_label_like(
                protrusion_result.outward_patch_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_patch_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.surface_sheet_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_surface_sheet_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.surface_sheet_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_surface_sheet_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_surface_sheet_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_surface_sheet_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.inward_surface_sheet_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_surface_sheet_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_surface_sheet_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_surface_sheet_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.outward_surface_sheet_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_surface_sheet_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.aorta_surface_projection_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_aorta_surface_projection_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.aorta_surface_projection_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_aorta_surface_projection_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_aorta_surface_projection_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_aorta_surface_projection_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.inward_aorta_surface_projection_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_aorta_surface_projection_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_aorta_surface_projection_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_aorta_surface_projection_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.outward_aorta_surface_projection_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_aorta_surface_projection_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.aorta_surface_core_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_aorta_surface_core_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.aorta_surface_core_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_aorta_surface_core_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.inward_aorta_surface_core_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_aorta_surface_core_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.inward_aorta_surface_core_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_inward_aorta_surface_core_labels_3d.nii.gz",
            )
            write_mask_like(
                protrusion_result.outward_aorta_surface_core_mask,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_aorta_surface_core_1mm.nii.gz",
            )
            write_label_like(
                protrusion_result.outward_aorta_surface_core_labelmap,
                image.image,
                masks_dir / f"{case_id}_lumen_protrusion_outward_ulcer_like_aorta_surface_core_labels_3d.nii.gz",
            )
            _write_thresholded_lumen_protrusion_qc_masks(
                candidates=protrusion_result.candidates,
                case_id=case_id,
                reference_image=image.image,
                masks_dir=masks_dir,
                write_label_like=write_label_like,
                inward_core_labelmap=protrusion_result.inward_aorta_surface_core_labelmap,
                outward_core_labelmap=protrusion_result.outward_aorta_surface_core_labelmap,
                inward_projection_labelmap=protrusion_result.inward_aorta_surface_projection_labelmap,
                outward_projection_labelmap=protrusion_result.outward_aorta_surface_projection_labelmap,
                inward_native_labelmap=protrusion_result.inward_aorta_surface_native_labelmap,
                outward_native_labelmap=protrusion_result.outward_aorta_surface_native_labelmap,
                inward_thresholds_mm=_parse_float_list(
                    protrusion_config.get("inward_qc_depth_thresholds_mm", [])
                ),
                outward_thresholds_mm=_parse_float_list(
                    protrusion_config.get("outward_qc_depth_thresholds_mm", [])
                ),
                sources=list(protrusion_config.get("thresholded_qc_sources", ["aorta_surface_core"])),
            )
    else:
        lumen_protrusion_summary = pd.DataFrame()
        lumen_protrusion_candidates = pd.DataFrame()
        lumen_protrusion_point_features = pd.DataFrame()

    fat_config = config.get("fat_omics", {})
    if bool(fat_config.get("enabled", True)):
        fat_result = extract_periaortic_fat_omics(
            image=image.array,
            aorta_mask=cleaned_mask,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
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
        fat_omics_frame = fat_result.features
        shell_masks["periaortic_fat"] = fat_result.fat_mask
        shell_masks["periaortic_fat_roi"] = fat_result.periaortic_roi_mask
        shell_masks.update(fat_result.fat_layer_masks)
        if bool(fat_config.get("save_mask", True)) and bool(config["outputs"]["save_masks"]):
            for stale_layer_path in masks_dir.glob(f"{case_id}_periaortic_fat_*mm.nii.gz"):
                stale_layer_path.unlink()
            write_mask_like(
                fat_result.periaortic_roi_mask,
                image.image,
                masks_dir / f"{case_id}_periaortic_fat_roi.nii.gz",
            )
            write_mask_like(
                fat_result.fat_mask,
                image.image,
                masks_dir / f"{case_id}_periaortic_fat.nii.gz",
            )
            for layer_name, layer_mask in fat_result.fat_layer_masks.items():
                write_mask_like(
                    layer_mask,
                    image.image,
                    masks_dir / f"{case_id}_{layer_name}.nii.gz",
                )
    else:
        fat_omics_frame = pd.DataFrame()
        fat_result = None

    wall_from_fat_config = config.get("wall_from_fat", {})
    if bool(wall_from_fat_config.get("enabled", False)):
        if fat_result is None:
            raise ValueError("wall_from_fat requires fat_omics.enabled=true so fat support can be estimated.")
        wall_from_fat_result = extract_fat_closed_aortic_wall(
            image=image.array,
            aorta_mask=cleaned_mask,
            fat_mask=fat_result.fat_mask,
            spacing_xyz=spacing_xyz,
            case_id=case_id,
            outer_limit_mm=float(wall_from_fat_config.get("outer_limit_mm", 5.0)),
            close_radius_mm=float(wall_from_fat_config.get("close_radius_mm", 3.0)),
            lumen_core_distance_mm=float(wall_from_fat_config.get("lumen_core_distance_mm", 5.0)),
            centerline_core_radius_mm=float(wall_from_fat_config.get("centerline_core_radius_mm", 2.0)),
            contrast_lower_margin_hu=float(wall_from_fat_config.get("contrast_lower_margin_hu", 120.0)),
            min_lumen_hu=float(wall_from_fat_config.get("min_lumen_hu", 150.0)),
            max_lumen_hu_above_reference=_optional_float(
                wall_from_fat_config.get("max_lumen_hu_above_reference", 300.0)
            ),
            lumen_reference_lower_fraction=_optional_float(
                wall_from_fat_config.get("lumen_reference_lower_fraction", None)
            ),
            lumen_reference_upper_fraction=_optional_float(
                wall_from_fat_config.get("lumen_reference_upper_fraction", None)
            ),
            lumen_reference_statistic=str(
                wall_from_fat_config.get("lumen_reference_statistic", "median")
            ),
            require_lumen_seed_connectivity=bool(
                wall_from_fat_config.get("require_lumen_seed_connectivity", False)
            ),
            use_input_aorta_as_lumen_floor=bool(
                wall_from_fat_config.get("use_input_aorta_as_lumen_floor", False)
            ),
            lumen_floor_mask=(
                raw_mask.array
                if bool(wall_from_fat_config.get("preserve_raw_input_aorta_in_lumen_floor", False))
                else None
            ),
            smooth_lumen_profile_mm=float(wall_from_fat_config.get("smooth_lumen_profile_mm", 10.0)),
            min_core_voxels_per_slice=int(wall_from_fat_config.get("min_core_voxels_per_slice", 20)),
            wall_hu_min=float(wall_from_fat_config.get("wall_hu_min", -30.0)),
            wall_hu_max=float(wall_from_fat_config.get("wall_hu_max", 1200.0)),
            exclude_fat_from_wall=bool(wall_from_fat_config.get("exclude_fat_from_wall", True)),
            exclude_calcification_hu=_optional_float(wall_from_fat_config.get("exclude_calcification_hu", None)),
            include_calcification_in_wall=bool(wall_from_fat_config.get("include_calcification_in_wall", True)),
            lumen_correction_enabled=bool(wall_from_fat_config.get("lumen_correction_enabled", False)),
            lumen_correction_outer_mm=float(wall_from_fat_config.get("lumen_correction_outer_mm", 2.0)),
            lumen_correction_close_radius_mm=float(
                wall_from_fat_config.get("lumen_correction_close_radius_mm", 1.0)
            ),
            lumen_correction_lower_margin_hu=_optional_float(
                wall_from_fat_config.get("lumen_correction_lower_margin_hu", None)
            ),
            lumen_correction_min_hu=_optional_float(
                wall_from_fat_config.get("lumen_correction_min_hu", None)
            ),
            lumen_correction_max_above_reference_hu=_optional_float(
                wall_from_fat_config.get("lumen_correction_max_above_reference_hu", None)
            ),
            software_version=software_version,
        )
        wall_from_fat_frame = wall_from_fat_result.features
        shell_masks["aortic_wall_contrast_lumen_from_centerline_hu"] = (
            wall_from_fat_result.contrast_lumen_mask
        )
        shell_masks["aortic_wall_candidate_from_fat_lumen"] = wall_from_fat_result.wall_candidate_mask
        shell_masks["aortic_wall_hu_refined_aorta_trace"] = wall_from_fat_result.hu_refined_aorta_mask
        shell_masks["aortic_wall_outer_closed_from_fat_5mm"] = (
            wall_from_fat_result.closed_outer_envelope_mask
        )
        shell_masks["aortic_wall_fat_support_0_5mm"] = wall_from_fat_result.fat_support_mask
        if bool(wall_from_fat_config.get("save_masks", True)) and bool(config["outputs"]["save_masks"]):
            for stale_wall_path in masks_dir.glob(f"{case_id}_aortic_wall_*from_fat*.nii.gz"):
                stale_wall_path.unlink()
            write_mask_like(
                wall_from_fat_result.contrast_lumen_mask,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz",
            )
            write_mask_like(
                wall_from_fat_result.fat_support_mask,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_fat_support_0_5mm.nii.gz",
            )
            write_mask_like(
                wall_from_fat_result.closed_outer_envelope_mask,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_outer_closed_from_fat_5mm.nii.gz",
            )
            write_mask_like(
                wall_from_fat_result.wall_candidate_mask,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_candidate_from_fat_lumen.nii.gz",
            )
            write_mask_like(
                wall_from_fat_result.hu_refined_aorta_mask,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_hu_refined_aorta_trace.nii.gz",
            )
            write_label_like(
                wall_from_fat_result.labelmap,
                image.image,
                masks_dir / f"{case_id}_aortic_wall_from_fat_lumen_labels.nii.gz",
            )
        wall_lumen_protrusion_config = wall_from_fat_config.get("protrusions", {})
        if bool(wall_lumen_protrusion_config.get("enabled", False)):
            wall_lumen_protrusion_result = detect_lumen_protrusions(
                lumen_mask=wall_from_fat_result.contrast_lumen_mask,
                spacing_xyz=spacing_xyz,
                case_id=case_id,
                image_hu=image.array,
                segment_labels=segment_labels,
                segment_names=SEGMENT_LABELS,
                analysis_mask_override=(
                    wall_from_fat_result.contrast_lumen_mask | wall_from_fat_result.wall_candidate_mask
                ),
                centerline_interval_mm=float(wall_lumen_protrusion_config.get("centerline_interval_mm", 2.0)),
                centerline_smoothing_mm=float(wall_lumen_protrusion_config.get("centerline_smoothing_mm", 6.0)),
                plane_spacing_mm=float(wall_lumen_protrusion_config.get("plane_spacing_mm", 0.75)),
                radial_sample_step_mm=float(wall_lumen_protrusion_config.get("radial_sample_step_mm", 0.25)),
                max_radius_mm=float(wall_lumen_protrusion_config.get("max_radius_mm", 35.0)),
                angular_bins=int(wall_lumen_protrusion_config.get("angular_bins", 144)),
                angular_median_window_deg=float(
                    wall_lumen_protrusion_config.get("angular_median_window_deg", 50.0)
                ),
                inward_angular_median_window_deg=_optional_float(
                    wall_lumen_protrusion_config.get("inward_angular_median_window_deg", None)
                ),
                outward_angular_median_window_deg=_optional_float(
                    wall_lumen_protrusion_config.get("outward_angular_median_window_deg", None)
                ),
                longitudinal_smoothing_mm=float(
                    wall_lumen_protrusion_config.get("longitudinal_smoothing_mm", 12.0)
                ),
                inward_longitudinal_smoothing_mm=_optional_float(
                    wall_lumen_protrusion_config.get("inward_longitudinal_smoothing_mm", None)
                ),
                outward_longitudinal_smoothing_mm=_optional_float(
                    wall_lumen_protrusion_config.get("outward_longitudinal_smoothing_mm", None)
                ),
                min_depth_mm=float(wall_lumen_protrusion_config.get("min_depth_mm", 2.0)),
                outward_min_depth_mm=_optional_float(
                    wall_lumen_protrusion_config.get("outward_min_depth_mm", 1.5)
                ),
                high_risk_depth_mm=float(wall_lumen_protrusion_config.get("high_risk_depth_mm", 4.0)),
                min_angular_width_deg=float(wall_lumen_protrusion_config.get("min_angular_width_deg", 5.0)),
                max_angular_width_deg=float(wall_lumen_protrusion_config.get("max_angular_width_deg", 90.0)),
                outward_min_angular_width_deg=_optional_float(
                    wall_lumen_protrusion_config.get("outward_min_angular_width_deg", 2.5)
                ),
                outward_max_angular_width_deg=_optional_float(
                    wall_lumen_protrusion_config.get("outward_max_angular_width_deg", 110.0)
                ),
                min_length_mm=float(wall_lumen_protrusion_config.get("min_length_mm", 1.0)),
                max_length_mm=float(wall_lumen_protrusion_config.get("max_length_mm", 25.0)),
                outward_min_length_mm=_optional_float(
                    wall_lumen_protrusion_config.get("outward_min_length_mm", 1.0)
                ),
                outward_max_length_mm=_optional_float(
                    wall_lumen_protrusion_config.get("outward_max_length_mm", 30.0)
                ),
                min_peak_prominence_mm=_optional_float(
                    wall_lumen_protrusion_config.get("min_peak_prominence_mm", None)
                ),
                outward_min_peak_prominence_mm=_optional_float(
                    wall_lumen_protrusion_config.get("outward_min_peak_prominence_mm", None)
                ),
                max_median_depth_fraction=_optional_float(
                    wall_lumen_protrusion_config.get("max_median_depth_fraction", None)
                ),
                outward_max_median_depth_fraction=_optional_float(
                    wall_lumen_protrusion_config.get("outward_max_median_depth_fraction", None)
                ),
                min_focality_ratio=_optional_float(wall_lumen_protrusion_config.get("min_focality_ratio", None)),
                outward_min_focality_ratio=_optional_float(
                    wall_lumen_protrusion_config.get("outward_min_focality_ratio", None)
                ),
                end_margin_mm=float(wall_lumen_protrusion_config.get("end_margin_mm", 10.0)),
                patch_longitudinal_padding_mm=float(
                    wall_lumen_protrusion_config.get("patch_longitudinal_padding_mm", 2.0)
                ),
                patch_angular_padding_deg=float(
                    wall_lumen_protrusion_config.get("patch_angular_padding_deg", 10.0)
                ),
                surface_projection_depth_mm=float(
                    wall_lumen_protrusion_config.get("surface_projection_depth_mm", 1.0)
                ),
                surface_core_relative_threshold=float(
                    wall_lumen_protrusion_config.get("surface_core_relative_threshold", 0.75)
                ),
                surface_core_depth_window_mm=float(
                    wall_lumen_protrusion_config.get("surface_core_depth_window_mm", 1.0)
                ),
                surface_core_longitudinal_padding_mm=float(
                    wall_lumen_protrusion_config.get("surface_core_longitudinal_padding_mm", 0.0)
                ),
                surface_core_angular_padding_deg=float(
                    wall_lumen_protrusion_config.get("surface_core_angular_padding_deg", 2.5)
                ),
                detect_inward=bool(wall_lumen_protrusion_config.get("detect_inward", True)),
                detect_outward=bool(wall_lumen_protrusion_config.get("detect_outward", True)),
                intensity_gate_enabled=bool(wall_lumen_protrusion_config.get("intensity_gate_enabled", True)),
                centerline_core_radius_mm=float(
                    wall_lumen_protrusion_config.get("centerline_core_radius_mm", 2.0)
                ),
                contrast_lower_margin_hu=float(
                    wall_lumen_protrusion_config.get("contrast_lower_margin_hu", 120.0)
                ),
                min_contrast_hu=float(wall_lumen_protrusion_config.get("min_contrast_hu", 150.0)),
                max_contrast_hu_above_reference=_optional_float(
                    wall_lumen_protrusion_config.get("max_contrast_hu_above_reference", 300.0)
                ),
                contrast_reference_lower_fraction=_optional_float(
                    wall_lumen_protrusion_config.get("contrast_reference_lower_fraction", None)
                ),
                contrast_reference_upper_fraction=_optional_float(
                    wall_lumen_protrusion_config.get("contrast_reference_upper_fraction", None)
                ),
                max_external_contrast_component_volume_mm3=_optional_float(
                    wall_lumen_protrusion_config.get("max_external_contrast_component_volume_mm3", None)
                ),
                max_candidate_outside_aorta_fraction=_optional_float(
                    wall_lumen_protrusion_config.get("max_candidate_outside_aorta_fraction", None)
                ),
                clip_candidate_masks_to_analysis_mask=True,
                software_version=software_version,
            )
            wall_lumen_summary = wall_lumen_protrusion_result.summary_features.assign(
                analysis_source="wall_from_fat_lumen_wall"
            )
            wall_lumen_candidates = wall_lumen_protrusion_result.candidates.assign(
                analysis_source="wall_from_fat_lumen_wall"
            )
            wall_lumen_points = wall_lumen_protrusion_result.point_features.assign(
                analysis_source="wall_from_fat_lumen_wall"
            )
            lumen_protrusion_summary = pd.concat(
                [lumen_protrusion_summary, wall_lumen_summary], ignore_index=True
            )
            lumen_protrusion_candidates = pd.concat(
                [lumen_protrusion_candidates, wall_lumen_candidates], ignore_index=True
            )
            lumen_protrusion_point_features = pd.concat(
                [lumen_protrusion_point_features, wall_lumen_points], ignore_index=True
            )
            if bool(wall_lumen_protrusion_config.get("save_masks", True)) and bool(config["outputs"]["save_masks"]):
                _write_thresholded_lumen_protrusion_qc_masks(
                    candidates=wall_lumen_protrusion_result.candidates,
                    case_id=case_id,
                    reference_image=image.image,
                    masks_dir=masks_dir,
                    write_label_like=write_label_like,
                    inward_core_labelmap=wall_lumen_protrusion_result.inward_aorta_surface_core_labelmap,
                    outward_core_labelmap=wall_lumen_protrusion_result.outward_aorta_surface_core_labelmap,
                    inward_projection_labelmap=wall_lumen_protrusion_result.inward_aorta_surface_projection_labelmap,
                    outward_projection_labelmap=wall_lumen_protrusion_result.outward_aorta_surface_projection_labelmap,
                    inward_native_labelmap=wall_lumen_protrusion_result.inward_aorta_surface_native_labelmap,
                    outward_native_labelmap=wall_lumen_protrusion_result.outward_aorta_surface_native_labelmap,
                    inward_thresholds_mm=_parse_float_list(
                        wall_lumen_protrusion_config.get("inward_qc_depth_thresholds_mm", [2, 3, 4])
                    ),
                    outward_thresholds_mm=_parse_float_list(
                        wall_lumen_protrusion_config.get("outward_qc_depth_thresholds_mm", [1.5, 2, 3, 4])
                    ),
                    sources=list(
                        wall_lumen_protrusion_config.get(
                            "thresholded_qc_sources", ["aorta_surface_native", "aorta_surface_core"]
                        )
                    ),
                    output_prefix="wall_lumen_protrusion",
                )
    else:
        wall_from_fat_result = None
        wall_from_fat_frame = pd.DataFrame()

    radiomics_frame = _extract_configured_radiomics(
        config=config,
        project_root=project_root,
        image_path=Path(image_path),
        masks_dir=masks_dir,
        cleaned_mask_path=cleaned_mask_path,
        shell_masks=shell_masks,
        reference_image=image.image,
        case_id=case_id,
        software_version=software_version,
    )

    if bool(config["outputs"]["save_figures"]):
        _save_figures(image.array, cleaned_mask, calcium_seed, case_id, figures_dir)

    case_level_features = _qc_to_feature_rows(qc_metrics, software_version)
    all_long_features = pd.concat(
        [
            ensure_feature_columns(case_level_features),
            ensure_feature_columns(calcification_frame),
            ensure_feature_columns(calcium_omics_frame),
            ensure_feature_columns(fat_omics_frame),
            ensure_feature_columns(lumen_protrusion_summary),
            ensure_feature_columns(wall_from_fat_frame),
            ensure_feature_columns(radiomics_frame),
        ],
        ignore_index=True,
    )
    wide_features = long_to_wide_features(all_long_features)

    write_csv(qc_frame, qc_dir / "qc_summary.csv")
    write_csv(calcification_frame, features_dir / "calcification_features.csv")
    write_csv(calcium_omics_frame, features_dir / "calcium_omics_features.csv")
    write_csv(fat_omics_frame, features_dir / "fat_omics_features.csv")
    write_csv(lumen_protrusion_summary, features_dir / "lumen_protrusion_summary_features.csv")
    write_csv(lumen_protrusion_candidates, features_dir / "lumen_protrusion_candidates.csv")
    write_csv(lumen_protrusion_point_features, features_dir / "lumen_protrusion_point_features.csv")
    write_csv(wall_from_fat_frame, features_dir / "wall_from_fat_features.csv")
    write_csv(radiomics_frame, features_dir / "radiomics_features.csv")
    write_csv(case_level_features, features_dir / "case_level_features.csv")
    write_csv(centerline_frame, features_dir / "centerline_points.csv")
    write_csv(geometry_frame, features_dir / "centerline_point_features.csv")
    write_csv(segment_frame, features_dir / "segment_level_features.csv")
    write_csv(wide_features, features_dir / "modeling_wide_features.csv")

    return CaseResult(
        qc=qc_frame,
        calcification=calcification_frame,
        calcium_omics=calcium_omics_frame,
        fat_omics=fat_omics_frame,
        lumen_protrusion_summary=lumen_protrusion_summary,
        lumen_protrusion_candidates=lumen_protrusion_candidates,
        lumen_protrusion_point_features=lumen_protrusion_point_features,
        radiomics=radiomics_frame,
        case_level_features=case_level_features,
        centerline_points=centerline_frame,
        centerline_point_features=geometry_frame,
        segment_level_features=segment_frame,
        wall_from_fat_features=wall_from_fat_frame,
        wide_features=wide_features,
    )


def run_single(
    image: Path,
    aorta_mask: Path,
    case_id: str,
    outdir: Path = Path("outputs"),
    config: Path | None = None,
) -> None:
    """Run one case."""
    _configure_logging()
    result = run_pipeline_case(image, aorta_mask, case_id, outdir, config)
    print(f"Wrote outputs for {case_id} to {outdir}")
    print(f"QC rows: {len(result.qc)}; calcification rows: {len(result.calcification)}")


def run_batch(
    manifest: Path,
    outdir: Path = Path("outputs"),
    config: Path | None = None,
    metadata_filter: str = "none",
    metadata_include_keywords: list[str] | None = None,
    metadata_exclude_keywords: list[str] | None = None,
    allow_missing_metadata: bool = False,
) -> None:
    """Run all cases from a manifest CSV."""
    import pandas as pd

    from .metadata_filter import evaluate_neuro_cta_metadata

    _configure_logging()
    manifest_frame = pd.read_csv(manifest)
    required = {"case_id", "image_path", "aorta_mask_path"}
    missing = required - set(manifest_frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

    outdir = Path(outdir)
    metadata_rows: list[dict[str, object]] = []
    results: list[CaseResult] = []
    for row in manifest_frame.to_dict(orient="records"):
        if metadata_filter == "neuro-cta":
            eligibility = evaluate_neuro_cta_metadata(
                row,
                manifest_base=manifest.parent,
                include_keywords=metadata_include_keywords or [],
                exclude_keywords=metadata_exclude_keywords or [],
                allow_missing_metadata=allow_missing_metadata,
            )
            metadata_rows.append(eligibility.as_dict())
            if not eligibility.eligible:
                continue
        elif metadata_filter != "none":
            raise ValueError(f"Unsupported metadata filter: {metadata_filter}")

        case_id = str(row["case_id"])
        logger.info("Running case %s", case_id)
        results.append(
            run_pipeline_case(
                image_path=row["image_path"],
                aorta_mask_path=row["aorta_mask_path"],
                case_id=case_id,
                outdir=outdir,
                config_path=config,
            )
        )
    if metadata_rows:
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metadata_rows).to_csv(outdir / "metadata_eligibility.csv", index=False)
    if not results:
        raise ValueError("No cases were processed after manifest and metadata filtering.")

    features_dir = outdir / "features"
    qc_dir = outdir / "qc"
    _write_aggregated(results, qc_dir, features_dir)
    print(f"Wrote batch outputs for {len(results)} cases to {outdir}")


def _extract_configured_radiomics(
    config: dict,
    project_root: Path,
    image_path: Path,
    masks_dir: Path,
    cleaned_mask_path: Path,
    shell_masks: dict[str, object],
    reference_image: object,
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    import pandas as pd

    from .config import resolve_project_path
    from .features import feature_row
    from .io import write_mask_like
    from .radiomics import extract_radiomics_features

    if not bool(config["radiomics"]["enabled"]):
        return pd.DataFrame()

    radiomics_backend = str(config["radiomics"].get("backend", "pyradiomics"))
    radiomics_device = str(config["radiomics"].get("device", "cpu"))
    settings_path = resolve_project_path(str(config["radiomics"]["settings_path"]), project_root)
    if settings_path.exists():
        shutil.copy2(settings_path, masks_dir / f"{case_id}_{radiomics_backend}_settings.yaml")
    else:
        logger.warning("Radiomics settings file not found: %s. Using backend defaults.", settings_path)
        settings_path = None

    region_paths: dict[str, Path] = {"aorta_mask": cleaned_mask_path}
    for region in config["radiomics"]["regions"]:
        if region in region_paths:
            continue
        if region in shell_masks:
            path = masks_dir / f"{case_id}_{region}.nii.gz"
            write_mask_like(shell_masks[region], reference_image, path)
            region_paths[region] = path

    frames: list[pd.DataFrame] = []
    for region in config["radiomics"]["regions"]:
        mask_path = region_paths.get(region)
        if mask_path is None:
            logger.warning("Skipping configured radiomics region without mask: %s", region)
            continue
        try:
            frames.append(
                extract_radiomics_features(
                    image_path=image_path,
                    mask_path=mask_path,
                    case_id=case_id,
                    region=region,
                    settings_path=settings_path,
                    include_diagnostics=bool(config["radiomics"]["include_diagnostics"]),
                    software_version=software_version,
                    backend=radiomics_backend,
                    device=radiomics_device,
                )
            )
        except Exception as exc:
            logger.warning("Radiomics extraction failed for %s/%s: %s", case_id, region, exc)
            frames.append(
                pd.DataFrame(
                    [
                        feature_row(
                            case_id=case_id,
                            region=region,
                            feature_group="radiomics_status",
                            feature_name="extraction_error",
                            feature_value=str(exc),
                            mask_name=mask_path.name,
                            software_version=software_version,
                        )
                    ]
                )
            )
            if isinstance(exc, ImportError):
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _qc_to_feature_rows(qc_metrics: dict[str, object], software_version: str) -> pd.DataFrame:
    import pandas as pd

    from .features import feature_row

    case_id = str(qc_metrics["case_id"])
    rows = [
        feature_row(
            case_id=case_id,
            region="whole_aorta",
            feature_group="qc",
            feature_name=key,
            feature_value=value,
            units=_qc_units(key),
            mask_name="aorta_mask_cleaned",
            software_version=software_version,
        )
        for key, value in qc_metrics.items()
        if key != "case_id"
    ]
    return pd.DataFrame(rows)


def _qc_units(key: str) -> str:
    if key.endswith("_mm3"):
        return "mm3"
    if key.endswith("_mm"):
        return "mm"
    if key.endswith("_hu"):
        return "HU"
    if "voxel" in key:
        return "voxels"
    return ""


def _calcification_mask_filename(case_id: str, roi_name: str, threshold: int | float) -> str:
    threshold_text = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
    if roi_name == "aorta_mask":
        return f"{case_id}_calcification_thr{threshold_text}HU.nii.gz"
    safe_roi = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in roi_name)
    return f"{case_id}_calcification_{safe_roi}_thr{threshold_text}HU.nii.gz"


def _highest_nonempty_mask(masks: dict[int, object]) -> object:
    import numpy as np

    if not masks:
        raise ValueError("No masks were provided.")
    for _, mask in sorted(masks.items(), reverse=True):
        if np.asarray(mask).any():
            return mask
    first = next(iter(masks.values()))
    return np.zeros_like(first, dtype=bool)


def _highest_nonempty_threshold(masks: dict[int, object]) -> int:
    import numpy as np

    for threshold, mask in sorted(masks.items(), reverse=True):
        if np.asarray(mask).any():
            return int(threshold)
    return int(next(iter(masks.keys())))


def _parse_ranges(values: object) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    if not isinstance(values, list):
        return ranges
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            ranges.append((float(item[0]), float(item[1])))
    return ranges


def _parse_named_ranges(values: object) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    if not isinstance(values, dict):
        return ranges
    for name, item in values.items():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            ranges[str(name)] = (float(item[0]), float(item[1]))
    return ranges


def _parse_float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def _write_thresholded_lumen_protrusion_qc_masks(
    candidates: object,
    case_id: str,
    reference_image: object,
    masks_dir: Path,
    write_label_like: object,
    inward_core_labelmap: object,
    outward_core_labelmap: object,
    inward_projection_labelmap: object,
    outward_projection_labelmap: object,
    inward_native_labelmap: object,
    outward_native_labelmap: object,
    inward_thresholds_mm: list[float],
    outward_thresholds_mm: list[float],
    sources: list[str],
    output_prefix: str = "lumen_protrusion",
) -> None:
    """Write threshold-specific surface labelmaps for Slicer review."""
    import numpy as np
    import pandas as pd

    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return

    source_maps = {
        "aorta_surface_core": {
            "inward": inward_core_labelmap,
            "outward_ulcer_like": outward_core_labelmap,
        },
        "aorta_surface_projection": {
            "inward": inward_projection_labelmap,
            "outward_ulcer_like": outward_projection_labelmap,
        },
        "aorta_surface_native": {
            "inward": inward_native_labelmap,
            "outward_ulcer_like": outward_native_labelmap,
        },
    }
    direction_specs = {
        "inward": ("max_protrusion_depth_mm", inward_thresholds_mm),
        "outward_ulcer_like": ("max_outward_ulcer_like_depth_mm", outward_thresholds_mm),
    }
    for source in sources:
        if source not in source_maps:
            continue
        for direction, (depth_column, thresholds) in direction_specs.items():
            if not thresholds or depth_column not in candidates.columns:
                continue
            base_labels = np.asarray(source_maps[source][direction])
            rows = candidates[candidates["candidate_direction"] == direction]
            for threshold in thresholds:
                keep_ids = rows.loc[rows[depth_column] >= float(threshold), "candidate_id"].astype(int).to_numpy()
                if keep_ids.size == 0:
                    continue
                threshold_labels = np.where(np.isin(base_labels, keep_ids), base_labels, 0).astype(base_labels.dtype)
                suffix = _threshold_suffix(threshold)
                write_label_like(
                    threshold_labels,
                    reference_image,
                    masks_dir
                    / f"{case_id}_{output_prefix}_{direction}_{source}_depth_ge_{suffix}mm_labels_3d.nii.gz",
                )


def _threshold_suffix(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _save_figures(
    image: object,
    mask: object,
    calcium_overlay: object,
    case_id: str,
    figures_dir: Path,
) -> None:
    from .visualization import save_overlay_quicklook

    try:
        save_overlay_quicklook(
            image,
            mask,
            figures_dir / f"{case_id}_aorta_qc_overlay.png",
            title=f"{case_id} aorta mask QC",
            overlay=calcium_overlay,
        )
    except Exception as exc:
        logger.warning("Could not save quicklook figure for %s: %s", case_id, exc)


def _write_aggregated(results: list[CaseResult], qc_dir: Path, features_dir: Path) -> None:
    import pandas as pd

    from .features import long_to_wide_features, write_csv

    qc_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        qc_dir / "qc_summary.csv": [result.qc for result in results],
        features_dir / "calcification_features.csv": [result.calcification for result in results],
        features_dir / "calcium_omics_features.csv": [result.calcium_omics for result in results],
        features_dir / "fat_omics_features.csv": [result.fat_omics for result in results],
        features_dir / "lumen_protrusion_summary_features.csv": [
            result.lumen_protrusion_summary for result in results
        ],
        features_dir / "lumen_protrusion_candidates.csv": [
            result.lumen_protrusion_candidates for result in results
        ],
        features_dir / "lumen_protrusion_point_features.csv": [
            result.lumen_protrusion_point_features for result in results
        ],
        features_dir / "radiomics_features.csv": [result.radiomics for result in results],
        features_dir / "case_level_features.csv": [result.case_level_features for result in results],
        features_dir / "centerline_points.csv": [result.centerline_points for result in results],
        features_dir / "centerline_point_features.csv": [result.centerline_point_features for result in results],
        features_dir / "segment_level_features.csv": [result.segment_level_features for result in results],
    }
    for path, frames in tables.items():
        write_csv(pd.concat(frames, ignore_index=True), path)

    all_features = pd.concat(
        [
            *[result.case_level_features for result in results],
            *[result.calcification for result in results],
            *[result.calcium_omics for result in results],
            *[result.fat_omics for result in results],
            *[result.lumen_protrusion_summary for result in results],
            *[result.radiomics for result in results],
        ],
        ignore_index=True,
    )
    write_csv(long_to_wide_features(all_features), features_dir / "modeling_wide_features.csv")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aorta-focused CTA radiomics pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("run-single", help="Run one CTA/aorta-mask pair.")
    single.add_argument("--image", required=True, type=Path, help="CTA NIfTI image path.")
    single.add_argument("--aorta-mask", required=True, type=Path, help="Aorta mask NIfTI path.")
    single.add_argument("--case-id", required=True, help="Case identifier.")
    single.add_argument("--outdir", default=Path("outputs"), type=Path, help="Output directory.")
    single.add_argument("--config", default=None, type=Path, help="YAML config path.")

    batch = subparsers.add_parser("run-batch", help="Run cases from a manifest CSV.")
    batch.add_argument("--manifest", required=True, type=Path, help="CSV with case_id,image_path,aorta_mask_path.")
    batch.add_argument("--outdir", default=Path("outputs"), type=Path, help="Output directory.")
    batch.add_argument("--config", default=None, type=Path, help="YAML config path.")
    batch.add_argument(
        "--metadata-filter",
        choices=["none", "neuro-cta"],
        default="none",
        help="Optional manifest/JSON metadata eligibility filter before processing.",
    )
    batch.add_argument(
        "--metadata-include-keyword",
        action="append",
        default=[],
        help="Extra neuro/stroke inclusion keyword for --metadata-filter neuro-cta. Repeatable.",
    )
    batch.add_argument(
        "--metadata-exclude-keyword",
        action="append",
        default=[],
        help="Extra non-target exclusion keyword for --metadata-filter neuro-cta. Repeatable.",
    )
    batch.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="With --metadata-filter neuro-cta, process rows lacking metadata instead of skipping them.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-single":
        run_single(args.image, args.aorta_mask, args.case_id, args.outdir, args.config)
    elif args.command == "run-batch":
        run_batch(
            args.manifest,
            args.outdir,
            args.config,
            metadata_filter=args.metadata_filter,
            metadata_include_keywords=args.metadata_include_keyword,
            metadata_exclude_keywords=args.metadata_exclude_keyword,
            allow_missing_metadata=args.allow_missing_metadata,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
