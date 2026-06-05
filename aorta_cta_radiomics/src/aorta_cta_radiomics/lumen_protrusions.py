"""Centerline-normal lumen protrusion candidate detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .features import feature_row
from .shells import external_shell, internal_boundary_shell


@dataclass
class LumenProtrusionResult:
    candidates: pd.DataFrame
    point_features: pd.DataFrame
    summary_features: pd.DataFrame
    analysis_mask: np.ndarray
    contrast_like_mask: np.ndarray
    candidate_mask: np.ndarray
    candidate_labelmap: np.ndarray
    boundary_mask: np.ndarray
    inward_candidate_mask: np.ndarray
    inward_candidate_labelmap: np.ndarray
    inward_boundary_mask: np.ndarray
    outward_candidate_mask: np.ndarray
    outward_candidate_labelmap: np.ndarray
    outward_boundary_mask: np.ndarray
    patch_mask: np.ndarray
    patch_labelmap: np.ndarray
    inward_patch_mask: np.ndarray
    inward_patch_labelmap: np.ndarray
    outward_patch_mask: np.ndarray
    outward_patch_labelmap: np.ndarray
    surface_sheet_mask: np.ndarray
    surface_sheet_labelmap: np.ndarray
    inward_surface_sheet_mask: np.ndarray
    inward_surface_sheet_labelmap: np.ndarray
    outward_surface_sheet_mask: np.ndarray
    outward_surface_sheet_labelmap: np.ndarray
    aorta_surface_projection_mask: np.ndarray
    aorta_surface_projection_labelmap: np.ndarray
    inward_aorta_surface_projection_mask: np.ndarray
    inward_aorta_surface_projection_labelmap: np.ndarray
    outward_aorta_surface_projection_mask: np.ndarray
    outward_aorta_surface_projection_labelmap: np.ndarray
    aorta_surface_core_mask: np.ndarray
    aorta_surface_core_labelmap: np.ndarray
    inward_aorta_surface_core_mask: np.ndarray
    inward_aorta_surface_core_labelmap: np.ndarray
    outward_aorta_surface_core_mask: np.ndarray
    outward_aorta_surface_core_labelmap: np.ndarray
    aorta_surface_native_mask: np.ndarray
    aorta_surface_native_labelmap: np.ndarray
    inward_aorta_surface_native_mask: np.ndarray
    inward_aorta_surface_native_labelmap: np.ndarray
    outward_aorta_surface_native_mask: np.ndarray
    outward_aorta_surface_native_labelmap: np.ndarray


@dataclass
class _Centerline:
    points_xyz: np.ndarray
    points_zyx: np.ndarray
    tangent_xyz: np.ndarray
    normal_u_xyz: np.ndarray
    normal_v_xyz: np.ndarray
    s_mm: np.ndarray


def detect_lumen_protrusions(
    lumen_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    image_hu: np.ndarray | None = None,
    segment_labels: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
    centerline_interval_mm: float = 2.0,
    centerline_smoothing_mm: float = 6.0,
    plane_spacing_mm: float = 0.75,
    radial_sample_step_mm: float = 0.5,
    max_radius_mm: float = 35.0,
    angular_bins: int = 72,
    angular_median_window_deg: float = 50.0,
    inward_angular_median_window_deg: float | None = None,
    outward_angular_median_window_deg: float | None = None,
    longitudinal_smoothing_mm: float = 12.0,
    inward_longitudinal_smoothing_mm: float | None = None,
    outward_longitudinal_smoothing_mm: float | None = None,
    min_depth_mm: float = 2.0,
    outward_min_depth_mm: float | None = None,
    high_risk_depth_mm: float = 4.0,
    min_angular_width_deg: float = 5.0,
    max_angular_width_deg: float = 90.0,
    outward_min_angular_width_deg: float | None = None,
    outward_max_angular_width_deg: float | None = None,
    min_length_mm: float = 1.0,
    max_length_mm: float = 25.0,
    outward_min_length_mm: float | None = None,
    outward_max_length_mm: float | None = None,
    min_peak_prominence_mm: float | None = None,
    outward_min_peak_prominence_mm: float | None = None,
    max_median_depth_fraction: float | None = None,
    outward_max_median_depth_fraction: float | None = None,
    min_focality_ratio: float | None = None,
    outward_min_focality_ratio: float | None = None,
    end_margin_mm: float = 10.0,
    analysis_inner_layer_mm: float = 0.0,
    analysis_outer_layer_mm: float = 0.0,
    patch_longitudinal_padding_mm: float = 2.0,
    patch_angular_padding_deg: float = 10.0,
    surface_sheet_thickness_mm: float = 1.0,
    surface_projection_depth_mm: float = 1.0,
    surface_core_relative_threshold: float = 0.75,
    surface_core_depth_window_mm: float = 1.0,
    surface_core_longitudinal_padding_mm: float = 0.0,
    surface_core_angular_padding_deg: float = 2.5,
    detect_inward: bool = True,
    detect_outward: bool = False,
    intensity_gate_enabled: bool = True,
    centerline_core_radius_mm: float = 2.0,
    contrast_lower_margin_hu: float = 120.0,
    min_contrast_hu: float = 150.0,
    max_contrast_hu_above_reference: float | None = None,
    contrast_reference_lower_fraction: float | None = None,
    contrast_reference_upper_fraction: float | None = None,
    max_external_contrast_component_volume_mm3: float | None = None,
    max_candidate_outside_aorta_fraction: float | None = None,
    clip_candidate_masks_to_analysis_mask: bool = False,
    analysis_mask_override: np.ndarray | None = None,
    software_version: str = __version__,
) -> LumenProtrusionResult:
    """Detect focal local boundary deviations in centerline-normal coordinates.

    The output is a candidate map for review, not a histologic plaque
    segmentation or diagnostic classifier.
    """
    lumen = np.asarray(lumen_mask, dtype=bool)
    if analysis_mask_override is not None:
        analysis_mask = np.asarray(analysis_mask_override, dtype=bool)
        if analysis_mask.shape != lumen.shape:
            raise ValueError("analysis_mask_override must have the same shape as lumen_mask.")
        analysis_mask = analysis_mask | lumen
    else:
        analysis_mask = _analysis_mask(
            lumen,
            spacing_xyz,
            inner_layer_mm=analysis_inner_layer_mm,
            outer_layer_mm=analysis_outer_layer_mm,
        )
    image = np.asarray(image_hu, dtype=float) if image_hu is not None else None
    if image is not None and image.shape != lumen.shape:
        raise ValueError("image_hu must have the same shape as lumen_mask.")
    centerline = _extract_centerline(
        lumen,
        spacing_xyz=spacing_xyz,
        interval_mm=centerline_interval_mm,
        smooth_mm=centerline_smoothing_mm,
    )
    empty = np.zeros(lumen.shape, dtype=bool)
    empty_labels = np.zeros(lumen.shape, dtype=np.uint16)
    contrast_like_mask = analysis_mask.copy()
    if len(centerline.points_xyz) < 3:
        return LumenProtrusionResult(
            candidates=pd.DataFrame(),
            point_features=pd.DataFrame(),
            summary_features=_summary_features(pd.DataFrame(), case_id, software_version),
            analysis_mask=analysis_mask,
            contrast_like_mask=contrast_like_mask,
            candidate_mask=empty,
            candidate_labelmap=empty_labels,
            boundary_mask=empty,
            inward_candidate_mask=empty,
            inward_candidate_labelmap=empty_labels,
            inward_boundary_mask=empty,
            outward_candidate_mask=empty,
            outward_candidate_labelmap=empty_labels,
            outward_boundary_mask=empty,
            patch_mask=empty,
            patch_labelmap=empty_labels,
            inward_patch_mask=empty,
            inward_patch_labelmap=empty_labels,
            outward_patch_mask=empty,
            outward_patch_labelmap=empty_labels,
            surface_sheet_mask=empty,
            surface_sheet_labelmap=empty_labels,
            inward_surface_sheet_mask=empty,
            inward_surface_sheet_labelmap=empty_labels,
            outward_surface_sheet_mask=empty,
            outward_surface_sheet_labelmap=empty_labels,
            aorta_surface_projection_mask=empty,
            aorta_surface_projection_labelmap=empty_labels,
            inward_aorta_surface_projection_mask=empty,
            inward_aorta_surface_projection_labelmap=empty_labels,
            outward_aorta_surface_projection_mask=empty,
            outward_aorta_surface_projection_labelmap=empty_labels,
            aorta_surface_core_mask=empty,
            aorta_surface_core_labelmap=empty_labels,
            inward_aorta_surface_core_mask=empty,
            inward_aorta_surface_core_labelmap=empty_labels,
            outward_aorta_surface_core_mask=empty,
            outward_aorta_surface_core_labelmap=empty_labels,
            aorta_surface_native_mask=empty,
            aorta_surface_native_labelmap=empty_labels,
            inward_aorta_surface_native_mask=empty,
            inward_aorta_surface_native_labelmap=empty_labels,
            outward_aorta_surface_native_mask=empty,
            outward_aorta_surface_native_labelmap=empty_labels,
        )

    centerline_hu = _centerline_hu_reference(
        image=image,
        lumen=lumen,
        centerline=centerline,
        spacing_xyz=spacing_xyz,
        core_radius_mm=centerline_core_radius_mm,
    )
    lower_thresholds, upper_thresholds = _contrast_thresholds(
        centerline_hu=centerline_hu,
        lower_margin_hu=contrast_lower_margin_hu,
        min_contrast_hu=min_contrast_hu,
        max_above_reference_hu=max_contrast_hu_above_reference,
        reference_lower_fraction=contrast_reference_lower_fraction,
        reference_upper_fraction=contrast_reference_upper_fraction,
    )
    if image is not None and intensity_gate_enabled:
        contrast_like_mask = _global_contrast_like_mask(
            analysis_mask=analysis_mask,
            image=image,
            centerline_hu=centerline_hu,
            lower_margin_hu=contrast_lower_margin_hu,
            min_contrast_hu=min_contrast_hu,
            max_above_reference_hu=max_contrast_hu_above_reference,
            reference_lower_fraction=contrast_reference_lower_fraction,
            reference_upper_fraction=contrast_reference_upper_fraction,
        )

    actual_radii = _sample_boundary_radii(
        lumen=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        max_radius_mm=max_radius_mm,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        image_hu=image if intensity_gate_enabled else None,
        lower_hu_thresholds=lower_thresholds if intensity_gate_enabled else None,
        upper_hu_thresholds=upper_thresholds if intensity_gate_enabled else None,
    )
    inward_expected_radii = _expected_radii(
        actual_radii,
        angular_window_deg=(
            float(inward_angular_median_window_deg)
            if inward_angular_median_window_deg is not None
            else float(angular_median_window_deg)
        ),
        angular_bins=angular_bins,
        longitudinal_sigma_points=max(
            (
                float(inward_longitudinal_smoothing_mm)
                if inward_longitudinal_smoothing_mm is not None
                else float(longitudinal_smoothing_mm)
            )
            / centerline_interval_mm,
            0.0,
        ),
    )
    outward_expected_radii = _expected_radii(
        actual_radii,
        angular_window_deg=(
            float(outward_angular_median_window_deg)
            if outward_angular_median_window_deg is not None
            else float(angular_median_window_deg)
        ),
        angular_bins=angular_bins,
        longitudinal_sigma_points=max(
            (
                float(outward_longitudinal_smoothing_mm)
                if outward_longitudinal_smoothing_mm is not None
                else float(longitudinal_smoothing_mm)
            )
            / centerline_interval_mm,
            0.0,
        ),
    )
    inward_depth = inward_expected_radii - actual_radii
    outward_depth = actual_radii - outward_expected_radii
    valid_inward = np.isfinite(actual_radii) & np.isfinite(inward_expected_radii)
    valid_outward = np.isfinite(actual_radii) & np.isfinite(outward_expected_radii)
    inward_cells = (
        valid_inward & (inward_depth >= float(min_depth_mm)) if detect_inward else np.zeros_like(valid_inward)
    )
    outward_depth_threshold = float(outward_min_depth_mm) if outward_min_depth_mm is not None else float(min_depth_mm)
    outward_cells = (
        valid_outward & (outward_depth >= outward_depth_threshold)
        if detect_outward
        else np.zeros_like(valid_outward)
    )
    if end_margin_mm > 0 and centerline.s_mm.size:
        end_mask = (centerline.s_mm < float(end_margin_mm)) | (
            centerline.s_mm > float(centerline.s_mm[-1] - end_margin_mm)
        )
        inward_cells[end_mask, :] = False
        outward_cells[end_mask, :] = False

    inward_rows, inward_components = _component_candidates(
        candidate_cells=inward_cells,
        depth=inward_depth,
        actual_radii=actual_radii,
        expected_radii=inward_expected_radii,
        centerline=centerline,
        spacing_xyz=spacing_xyz,
        segment_labels=segment_labels,
        segment_names=segment_names or {},
        angular_bins=angular_bins,
        interval_mm=centerline_interval_mm,
        min_width_deg=min_angular_width_deg,
        max_width_deg=max_angular_width_deg,
        min_length_mm=min_length_mm,
        max_length_mm=max_length_mm,
        min_peak_prominence_mm=min_peak_prominence_mm,
        max_median_depth_fraction=max_median_depth_fraction,
        min_focality_ratio=min_focality_ratio,
        case_id=case_id,
        candidate_direction="inward",
        candidate_id_start=1,
    )
    outward_rows, outward_components = _component_candidates(
        candidate_cells=outward_cells,
        depth=outward_depth,
        actual_radii=actual_radii,
        expected_radii=outward_expected_radii,
        centerline=centerline,
        spacing_xyz=spacing_xyz,
        segment_labels=segment_labels,
        segment_names=segment_names or {},
        angular_bins=angular_bins,
        interval_mm=centerline_interval_mm,
        min_width_deg=float(outward_min_angular_width_deg)
        if outward_min_angular_width_deg is not None
        else min_angular_width_deg,
        max_width_deg=float(outward_max_angular_width_deg)
        if outward_max_angular_width_deg is not None
        else max_angular_width_deg,
        min_length_mm=float(outward_min_length_mm) if outward_min_length_mm is not None else min_length_mm,
        max_length_mm=float(outward_max_length_mm) if outward_max_length_mm is not None else max_length_mm,
        min_peak_prominence_mm=(
            float(outward_min_peak_prominence_mm)
            if outward_min_peak_prominence_mm is not None
            else min_peak_prominence_mm
        ),
        max_median_depth_fraction=(
            float(outward_max_median_depth_fraction)
            if outward_max_median_depth_fraction is not None
            else max_median_depth_fraction
        ),
        min_focality_ratio=(
            float(outward_min_focality_ratio) if outward_min_focality_ratio is not None else min_focality_ratio
        ),
        case_id=case_id,
        candidate_direction="outward_ulcer_like",
        candidate_id_start=len(inward_rows) + 1,
    )
    outward_rows, outward_components = _filter_outward_external_contrast_components(
        rows=outward_rows,
        components=outward_components,
        lumen=lumen,
        contrast_like_mask=contrast_like_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        actual_radii=actual_radii,
        expected_radii=outward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        max_external_component_volume_mm3=max_external_contrast_component_volume_mm3,
        max_candidate_outside_aorta_fraction=max_candidate_outside_aorta_fraction,
    )
    for new_id, row in enumerate(outward_rows, start=len(inward_rows) + 1):
        row["candidate_id"] = new_id
    inward_mask, inward_labelmap, inward_boundary = _candidate_labelmaps(
        lumen=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=inward_components,
        actual_radii=actual_radii,
        expected_radii=inward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_direction="inward",
        candidate_id_start=1,
    )
    outward_mask, outward_labelmap, outward_boundary = _candidate_labelmaps(
        lumen=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=outward_components,
        actual_radii=actual_radii,
        expected_radii=outward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_direction="outward_ulcer_like",
        candidate_id_start=len(inward_rows) + 1,
    )
    candidate_mask = inward_mask | outward_mask
    candidate_labelmap = np.maximum(inward_labelmap, outward_labelmap)
    boundary_mask = inward_boundary | outward_boundary
    inward_patch_mask, inward_patch_labelmap = _candidate_patch_labelmaps(
        analysis_mask=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=inward_components,
        expected_radii=inward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_id_start=1,
        interval_mm=centerline_interval_mm,
        inner_layer_mm=analysis_inner_layer_mm,
        outer_layer_mm=analysis_outer_layer_mm,
        longitudinal_padding_mm=patch_longitudinal_padding_mm,
        angular_padding_deg=patch_angular_padding_deg,
    )
    outward_patch_mask, outward_patch_labelmap = _candidate_patch_labelmaps(
        analysis_mask=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=outward_components,
        expected_radii=outward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_id_start=len(inward_rows) + 1,
        interval_mm=centerline_interval_mm,
        inner_layer_mm=analysis_inner_layer_mm,
        outer_layer_mm=analysis_outer_layer_mm,
        longitudinal_padding_mm=patch_longitudinal_padding_mm,
        angular_padding_deg=patch_angular_padding_deg,
    )
    patch_mask = inward_patch_mask | outward_patch_mask
    patch_labelmap = np.maximum(inward_patch_labelmap, outward_patch_labelmap)
    inward_surface_sheet_mask, inward_surface_sheet_labelmap = _candidate_surface_sheet_labelmaps(
        analysis_mask=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=inward_components,
        expected_radii=inward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_id_start=1,
        interval_mm=centerline_interval_mm,
        longitudinal_padding_mm=patch_longitudinal_padding_mm,
        angular_padding_deg=patch_angular_padding_deg,
        sheet_thickness_mm=surface_sheet_thickness_mm,
    )
    outward_surface_sheet_mask, outward_surface_sheet_labelmap = _candidate_surface_sheet_labelmaps(
        analysis_mask=analysis_mask,
        spacing_xyz=spacing_xyz,
        centerline=centerline,
        components=outward_components,
        expected_radii=outward_expected_radii,
        radial_step_mm=radial_sample_step_mm,
        angular_bins=angular_bins,
        candidate_id_start=len(inward_rows) + 1,
        interval_mm=centerline_interval_mm,
        longitudinal_padding_mm=patch_longitudinal_padding_mm,
        angular_padding_deg=patch_angular_padding_deg,
        sheet_thickness_mm=surface_sheet_thickness_mm,
    )
    surface_sheet_mask = inward_surface_sheet_mask | outward_surface_sheet_mask
    surface_sheet_labelmap = np.maximum(inward_surface_sheet_labelmap, outward_surface_sheet_labelmap)
    inward_aorta_surface_projection_mask, inward_aorta_surface_projection_labelmap = (
        _candidate_aorta_surface_projection_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=inward_components,
            angular_bins=angular_bins,
            candidate_id_start=1,
            interval_mm=centerline_interval_mm,
            longitudinal_padding_mm=patch_longitudinal_padding_mm,
            angular_padding_deg=patch_angular_padding_deg,
            surface_depth_mm=surface_projection_depth_mm,
        )
    )
    outward_aorta_surface_projection_mask, outward_aorta_surface_projection_labelmap = (
        _candidate_aorta_surface_projection_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=outward_components,
            angular_bins=angular_bins,
            candidate_id_start=len(inward_rows) + 1,
            interval_mm=centerline_interval_mm,
            longitudinal_padding_mm=patch_longitudinal_padding_mm,
            angular_padding_deg=patch_angular_padding_deg,
            surface_depth_mm=surface_projection_depth_mm,
        )
    )
    aorta_surface_projection_mask = inward_aorta_surface_projection_mask | outward_aorta_surface_projection_mask
    aorta_surface_projection_labelmap = np.maximum(
        inward_aorta_surface_projection_labelmap,
        outward_aorta_surface_projection_labelmap,
    )
    inward_aorta_surface_core_mask, inward_aorta_surface_core_labelmap = (
        _candidate_aorta_surface_core_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=inward_components,
            residual_depth=inward_depth,
            angular_bins=angular_bins,
            candidate_id_start=1,
            interval_mm=centerline_interval_mm,
            longitudinal_padding_mm=surface_core_longitudinal_padding_mm,
            angular_padding_deg=surface_core_angular_padding_deg,
            surface_depth_mm=surface_projection_depth_mm,
            relative_threshold=surface_core_relative_threshold,
            depth_window_mm=surface_core_depth_window_mm,
        )
    )
    outward_aorta_surface_core_mask, outward_aorta_surface_core_labelmap = (
        _candidate_aorta_surface_core_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=outward_components,
            residual_depth=outward_depth,
            angular_bins=angular_bins,
            candidate_id_start=len(inward_rows) + 1,
            interval_mm=centerline_interval_mm,
            longitudinal_padding_mm=surface_core_longitudinal_padding_mm,
            angular_padding_deg=surface_core_angular_padding_deg,
            surface_depth_mm=surface_projection_depth_mm,
            relative_threshold=surface_core_relative_threshold,
            depth_window_mm=surface_core_depth_window_mm,
        )
    )
    aorta_surface_core_mask = inward_aorta_surface_core_mask | outward_aorta_surface_core_mask
    aorta_surface_core_labelmap = np.maximum(
        inward_aorta_surface_core_labelmap,
        outward_aorta_surface_core_labelmap,
    )
    inward_aorta_surface_native_mask, inward_aorta_surface_native_labelmap = (
        _candidate_aorta_surface_native_core_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=inward_components,
            residual_depth=inward_depth,
            angular_bins=angular_bins,
            candidate_id_start=1,
            surface_depth_mm=surface_projection_depth_mm,
            relative_threshold=surface_core_relative_threshold,
            depth_window_mm=surface_core_depth_window_mm,
        )
    )
    outward_aorta_surface_native_mask, outward_aorta_surface_native_labelmap = (
        _candidate_aorta_surface_native_core_labelmaps(
            aorta_mask=lumen,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            components=outward_components,
            residual_depth=outward_depth,
            angular_bins=angular_bins,
            candidate_id_start=len(inward_rows) + 1,
            surface_depth_mm=surface_projection_depth_mm,
            relative_threshold=surface_core_relative_threshold,
            depth_window_mm=surface_core_depth_window_mm,
        )
    )
    aorta_surface_native_mask = inward_aorta_surface_native_mask | outward_aorta_surface_native_mask
    aorta_surface_native_labelmap = np.maximum(
        inward_aorta_surface_native_labelmap,
        outward_aorta_surface_native_labelmap,
    )
    if clip_candidate_masks_to_analysis_mask:
        inward_mask &= analysis_mask
        inward_labelmap[~analysis_mask] = 0
        inward_boundary &= analysis_mask
        outward_mask &= analysis_mask
        outward_labelmap[~analysis_mask] = 0
        outward_boundary &= analysis_mask
        candidate_mask = inward_mask | outward_mask
        candidate_labelmap = np.maximum(inward_labelmap, outward_labelmap)
        boundary_mask = inward_boundary | outward_boundary
        inward_patch_mask &= analysis_mask
        inward_patch_labelmap[~analysis_mask] = 0
        outward_patch_mask &= analysis_mask
        outward_patch_labelmap[~analysis_mask] = 0
        patch_mask = inward_patch_mask | outward_patch_mask
        patch_labelmap = np.maximum(inward_patch_labelmap, outward_patch_labelmap)
        inward_surface_sheet_mask &= analysis_mask
        inward_surface_sheet_labelmap[~analysis_mask] = 0
        outward_surface_sheet_mask &= analysis_mask
        outward_surface_sheet_labelmap[~analysis_mask] = 0
        surface_sheet_mask = inward_surface_sheet_mask | outward_surface_sheet_mask
        surface_sheet_labelmap = np.maximum(inward_surface_sheet_labelmap, outward_surface_sheet_labelmap)
        inward_aorta_surface_native_mask &= analysis_mask
        inward_aorta_surface_native_labelmap[~analysis_mask] = 0
        outward_aorta_surface_native_mask &= analysis_mask
        outward_aorta_surface_native_labelmap[~analysis_mask] = 0
        aorta_surface_native_mask = inward_aorta_surface_native_mask | outward_aorta_surface_native_mask
        aorta_surface_native_labelmap = np.maximum(
            inward_aorta_surface_native_labelmap,
            outward_aorta_surface_native_labelmap,
        )
    candidates = pd.DataFrame([*inward_rows, *outward_rows])
    point_expected_radii = np.where(np.isfinite(inward_expected_radii), inward_expected_radii, outward_expected_radii)
    point_features = _point_features(
        actual_radii=actual_radii,
        expected_radii=point_expected_radii,
        inward_depth=inward_depth,
        outward_depth=outward_depth,
        inward_cells=inward_cells,
        outward_cells=outward_cells,
        centerline=centerline,
        case_id=case_id,
        angular_bins=angular_bins,
        software_version=software_version,
    )
    return LumenProtrusionResult(
        candidates=candidates,
        point_features=point_features,
        summary_features=_summary_features(candidates, case_id, software_version, high_risk_depth_mm),
        analysis_mask=analysis_mask,
        contrast_like_mask=contrast_like_mask,
        candidate_mask=candidate_mask,
        candidate_labelmap=candidate_labelmap,
        boundary_mask=boundary_mask,
        inward_candidate_mask=inward_mask,
        inward_candidate_labelmap=inward_labelmap,
        inward_boundary_mask=inward_boundary,
        outward_candidate_mask=outward_mask,
        outward_candidate_labelmap=outward_labelmap,
        outward_boundary_mask=outward_boundary,
        patch_mask=patch_mask,
        patch_labelmap=patch_labelmap,
        inward_patch_mask=inward_patch_mask,
        inward_patch_labelmap=inward_patch_labelmap,
        outward_patch_mask=outward_patch_mask,
        outward_patch_labelmap=outward_patch_labelmap,
        surface_sheet_mask=surface_sheet_mask,
        surface_sheet_labelmap=surface_sheet_labelmap,
        inward_surface_sheet_mask=inward_surface_sheet_mask,
        inward_surface_sheet_labelmap=inward_surface_sheet_labelmap,
        outward_surface_sheet_mask=outward_surface_sheet_mask,
        outward_surface_sheet_labelmap=outward_surface_sheet_labelmap,
        aorta_surface_projection_mask=aorta_surface_projection_mask,
        aorta_surface_projection_labelmap=aorta_surface_projection_labelmap,
        inward_aorta_surface_projection_mask=inward_aorta_surface_projection_mask,
        inward_aorta_surface_projection_labelmap=inward_aorta_surface_projection_labelmap,
        outward_aorta_surface_projection_mask=outward_aorta_surface_projection_mask,
        outward_aorta_surface_projection_labelmap=outward_aorta_surface_projection_labelmap,
        aorta_surface_core_mask=aorta_surface_core_mask,
        aorta_surface_core_labelmap=aorta_surface_core_labelmap,
        inward_aorta_surface_core_mask=inward_aorta_surface_core_mask,
        inward_aorta_surface_core_labelmap=inward_aorta_surface_core_labelmap,
        outward_aorta_surface_core_mask=outward_aorta_surface_core_mask,
        outward_aorta_surface_core_labelmap=outward_aorta_surface_core_labelmap,
        aorta_surface_native_mask=aorta_surface_native_mask,
        aorta_surface_native_labelmap=aorta_surface_native_labelmap,
        inward_aorta_surface_native_mask=inward_aorta_surface_native_mask,
        inward_aorta_surface_native_labelmap=inward_aorta_surface_native_labelmap,
        outward_aorta_surface_native_mask=outward_aorta_surface_native_mask,
        outward_aorta_surface_native_labelmap=outward_aorta_surface_native_labelmap,
    )


def _analysis_mask(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    inner_layer_mm: float,
    outer_layer_mm: float,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if inner_layer_mm > 0:
        internal = internal_boundary_shell(binary, spacing_xyz, depth_mm=float(inner_layer_mm))
    else:
        internal = binary.copy()
    if outer_layer_mm > 0:
        external = external_shell(binary, spacing_xyz, inner_mm=0.0, outer_mm=float(outer_layer_mm))
    else:
        external = np.zeros_like(binary, dtype=bool)
    return internal | external


def _centerline_hu_reference(
    image: np.ndarray | None,
    lumen: np.ndarray,
    centerline: _Centerline,
    spacing_xyz: tuple[float, float, float],
    core_radius_mm: float,
) -> np.ndarray:
    if image is None:
        return np.full(len(centerline.points_zyx), np.nan, dtype=float)
    refs = np.full(len(centerline.points_zyx), np.nan, dtype=float)
    binary = np.asarray(lumen, dtype=bool)
    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    radii = np.maximum(1, np.ceil(float(core_radius_mm) / np.maximum(spacing_zyx, 1e-6)).astype(int))
    for idx, point in enumerate(centerline.points_zyx):
        center = np.asarray([int(round(float(value))) for value in point], dtype=int)
        starts = np.maximum(center - radii, 0)
        stops = np.minimum(center + radii + 1, binary.shape)
        slices = tuple(slice(int(starts[axis]), int(stops[axis])) for axis in range(3))
        local_lumen = binary[slices]
        local_image = image[slices]
        local_coords = np.argwhere(local_lumen)
        if local_coords.size:
            global_coords = local_coords + starts[None, :]
            distances = np.linalg.norm((global_coords - point[None, :]) * spacing_zyx[None, :], axis=1)
            values = local_image[local_lumen][distances <= float(core_radius_mm)]
            if values.size:
                refs[idx] = float(np.mean(values))
                continue
        rounded = _round_zyx(point, image.shape)
        if rounded is not None:
            refs[idx] = float(image[rounded])
    finite = np.isfinite(refs)
    if finite.sum() >= 2:
        refs = np.interp(np.arange(len(refs)), np.where(finite)[0], refs[finite])
        refs = _smooth_1d(refs, sigma=2.0)
    return refs


def _contrast_thresholds(
    centerline_hu: np.ndarray,
    lower_margin_hu: float,
    min_contrast_hu: float,
    max_above_reference_hu: float | None,
    reference_lower_fraction: float | None,
    reference_upper_fraction: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not np.isfinite(centerline_hu).any():
        return None, None
    if reference_lower_fraction is None:
        lower = centerline_hu - float(lower_margin_hu)
    else:
        lower = centerline_hu * (1.0 - float(reference_lower_fraction))
    lower = np.maximum(float(min_contrast_hu), lower)
    upper = None
    if reference_upper_fraction is not None:
        upper = centerline_hu * (1.0 + float(reference_upper_fraction))
    if max_above_reference_hu is not None:
        upper_margin = centerline_hu + float(max_above_reference_hu)
        upper = upper_margin if upper is None else np.minimum(upper, upper_margin)
    return lower, upper


def _global_contrast_like_mask(
    analysis_mask: np.ndarray,
    image: np.ndarray,
    centerline_hu: np.ndarray,
    lower_margin_hu: float,
    min_contrast_hu: float,
    max_above_reference_hu: float | None,
    reference_lower_fraction: float | None,
    reference_upper_fraction: float | None,
) -> np.ndarray:
    if np.isfinite(centerline_hu).any():
        reference = float(np.nanmedian(centerline_hu))
    else:
        reference = float(min_contrast_hu + lower_margin_hu)
    if reference_lower_fraction is None:
        lower = reference - float(lower_margin_hu)
    else:
        lower = reference * (1.0 - float(reference_lower_fraction))
    lower = max(float(min_contrast_hu), lower)
    contrast_like = np.asarray(analysis_mask, dtype=bool) & (image >= lower)
    upper = None
    if reference_upper_fraction is not None:
        upper = reference * (1.0 + float(reference_upper_fraction))
    if max_above_reference_hu is not None:
        upper_margin = reference + float(max_above_reference_hu)
        upper = upper_margin if upper is None else min(upper, upper_margin)
    if upper is not None:
        contrast_like &= image <= upper
    return contrast_like


def _extract_centerline(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    interval_mm: float,
    smooth_mm: float,
) -> _Centerline:
    skeleton_centerline = _extract_skeleton_centerline(mask, spacing_xyz, interval_mm, smooth_mm)
    if len(skeleton_centerline.points_xyz) >= 3:
        return skeleton_centerline
    return _extract_slice_centerline(mask, spacing_xyz, interval_mm, smooth_mm)


def _extract_slice_centerline(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    interval_mm: float,
    smooth_mm: float,
) -> _Centerline:
    binary = np.asarray(mask, dtype=bool)
    rows: list[tuple[float, float, float]] = []
    for z in np.where(binary.any(axis=(1, 2)))[0]:
        ys, xs = np.where(binary[z])
        if ys.size:
            rows.append((float(z), float(ys.mean()), float(xs.mean())))
    if len(rows) < 2:
        empty = np.zeros((0, 3), dtype=float)
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    zyx = np.asarray(rows, dtype=float)
    sigma_slices = smooth_mm / max(float(spacing_xyz[2]), 1e-6)
    if len(zyx) >= 3 and sigma_slices > 0:
        zyx[:, 1] = _smooth_1d(zyx[:, 1], sigma_slices)
        zyx[:, 2] = _smooth_1d(zyx[:, 2], sigma_slices)
    xyz = _zyx_to_xyz_mm(zyx, spacing_xyz)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(steps)])
    if float(s_raw[-1]) == 0:
        empty = np.zeros((0, 3), dtype=float)
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))
    sample_s = np.arange(0.0, float(s_raw[-1]) + interval_mm * 0.5, float(interval_mm))
    sample_xyz = np.column_stack([np.interp(sample_s, s_raw, xyz[:, axis]) for axis in range(3)])
    sample_zyx = _xyz_mm_to_zyx(sample_xyz, spacing_xyz)
    sample_zyx = _snap_points_to_mask(sample_zyx, binary, spacing_xyz, max_search_mm=6.0)
    sample_xyz = _zyx_to_xyz_mm(sample_zyx, spacing_xyz)
    tangent = _unit_vectors(np.gradient(sample_xyz, axis=0))
    normal_u, normal_v = _normal_bases(tangent)
    return _Centerline(sample_xyz, sample_zyx, tangent, normal_u, normal_v, sample_s)


def _extract_skeleton_centerline(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    interval_mm: float,
    smooth_mm: float,
) -> _Centerline:
    binary = np.asarray(mask, dtype=bool)
    empty = np.zeros((0, 3), dtype=float)
    if int(binary.sum()) < 3:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))
    coords = np.argwhere(binary)
    min_corner = np.maximum(coords.min(axis=0) - 2, 0)
    max_corner = np.minimum(coords.max(axis=0) + 3, binary.shape)
    slices = tuple(slice(int(min_corner[axis]), int(max_corner[axis])) for axis in range(3))
    cropped = binary[slices]

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components, dijkstra
        from skimage.morphology import skeletonize
    except Exception:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    try:
        skeleton = np.asarray(skeletonize(cropped), dtype=bool)
    except Exception:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    skeleton_coords = np.argwhere(skeleton)
    if len(skeleton_coords) < 3:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    index_by_coord = {tuple(coord.tolist()): idx for idx, coord in enumerate(skeleton_coords)}
    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    row_idx: list[int] = []
    col_idx: list[int] = []
    weights: list[float] = []
    offsets = [
        np.asarray([dz, dy, dx], dtype=int)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    for idx, coord in enumerate(skeleton_coords):
        for offset in offsets:
            neighbor = tuple((coord + offset).tolist())
            neighbor_idx = index_by_coord.get(neighbor)
            if neighbor_idx is None:
                continue
            row_idx.append(idx)
            col_idx.append(neighbor_idx)
            weights.append(float(np.linalg.norm(offset * spacing_zyx)))

    if not weights:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    graph = csr_matrix((weights, (row_idx, col_idx)), shape=(len(skeleton_coords), len(skeleton_coords)))
    n_components, labels = connected_components(graph, directed=False)
    if n_components > 1:
        component_counts = np.bincount(labels)
        keep_label = int(np.argmax(component_counts))
        keep = np.where(labels == keep_label)[0]
        old_to_new = {int(old): new for new, old in enumerate(keep)}
        keep_set = set(int(value) for value in keep)
        new_rows: list[int] = []
        new_cols: list[int] = []
        new_weights: list[float] = []
        for r, c, w in zip(row_idx, col_idx, weights):
            if r in keep_set and c in keep_set:
                new_rows.append(old_to_new[int(r)])
                new_cols.append(old_to_new[int(c)])
                new_weights.append(w)
        skeleton_coords = skeleton_coords[keep]
        graph = csr_matrix(
            (new_weights, (new_rows, new_cols)),
            shape=(len(skeleton_coords), len(skeleton_coords)),
        )
        if len(skeleton_coords) < 3 or not new_weights:
            return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    start = 0
    distances = dijkstra(graph, directed=False, indices=start, return_predecessors=False)
    if not np.isfinite(distances).any():
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))
    end_a = int(np.nanargmax(np.where(np.isfinite(distances), distances, -1.0)))
    distances, predecessors = dijkstra(graph, directed=False, indices=end_a, return_predecessors=True)
    end_b = int(np.nanargmax(np.where(np.isfinite(distances), distances, -1.0)))
    if not np.isfinite(distances[end_b]) or distances[end_b] <= 0:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    path_indices = [end_b]
    current = end_b
    while current != end_a:
        current = int(predecessors[current])
        if current < 0:
            return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))
        path_indices.append(current)
    path_indices.reverse()

    path_zyx = skeleton_coords[np.asarray(path_indices, dtype=int)].astype(float) + min_corner[None, :]
    path_xyz = _zyx_to_xyz_mm(path_zyx, spacing_xyz)
    steps = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
    keep_steps = np.concatenate([[True], steps > 0])
    path_zyx = path_zyx[keep_steps]
    path_xyz = path_xyz[keep_steps]
    if len(path_xyz) < 3:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))

    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_xyz, axis=0), axis=1))])
    if float(s_raw[-1]) == 0:
        return _Centerline(empty, empty, empty, empty, empty, np.zeros(0, dtype=float))
    sigma_points = smooth_mm / max(float(np.median(np.diff(s_raw))), 1e-6)
    if len(path_xyz) >= 5 and sigma_points > 0:
        for axis in range(3):
            path_xyz[:, axis] = _smooth_1d(path_xyz[:, axis], sigma_points)
    sample_s = np.arange(0.0, float(s_raw[-1]) + interval_mm * 0.5, float(interval_mm))
    sample_xyz = np.column_stack([np.interp(sample_s, s_raw, path_xyz[:, axis]) for axis in range(3)])
    sample_zyx = _xyz_mm_to_zyx(sample_xyz, spacing_xyz)
    sample_zyx = _snap_points_to_mask(sample_zyx, binary, spacing_xyz, max_search_mm=6.0)
    sample_xyz = _zyx_to_xyz_mm(sample_zyx, spacing_xyz)
    tangent = _unit_vectors(np.gradient(sample_xyz, axis=0))
    normal_u, normal_v = _normal_bases(tangent)
    return _Centerline(sample_xyz, sample_zyx, tangent, normal_u, normal_v, sample_s)


def _sample_boundary_radii(
    lumen: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    max_radius_mm: float,
    radial_step_mm: float,
    angular_bins: int,
    image_hu: np.ndarray | None = None,
    lower_hu_thresholds: np.ndarray | None = None,
    upper_hu_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
    except Exception as exc:
        raise ImportError("SciPy is required for centerline-normal lumen protrusion sampling.") from exc

    radii = np.arange(0.0, float(max_radius_mm) + float(radial_step_mm) * 0.5, float(radial_step_mm))
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_bins), endpoint=False)
    output = np.full((len(centerline.points_xyz), int(angular_bins)), np.nan, dtype=float)
    lumen_float = lumen.astype(np.float32)
    image_float = image_hu.astype(np.float32) if image_hu is not None else None
    for point_index, center_xyz in enumerate(centerline.points_xyz):
        u = centerline.normal_u_xyz[point_index]
        v = centerline.normal_v_xyz[point_index]
        lower_threshold = (
            float(lower_hu_thresholds[point_index])
            if lower_hu_thresholds is not None and np.isfinite(lower_hu_thresholds[point_index])
            else None
        )
        upper_threshold = (
            float(upper_hu_thresholds[point_index])
            if upper_hu_thresholds is not None and np.isfinite(upper_hu_thresholds[point_index])
            else None
        )
        for angle_index, angle in enumerate(angles):
            direction = np.cos(angle) * u + np.sin(angle) * v
            sample_xyz = center_xyz[None, :] + radii[:, None] * direction[None, :]
            sample_zyx = _xyz_mm_to_zyx(sample_xyz, spacing_xyz)
            roi_samples = ndi.map_coordinates(
                lumen_float,
                [sample_zyx[:, 0], sample_zyx[:, 1], sample_zyx[:, 2]],
                order=0,
                mode="constant",
                cval=0.0,
            ) > 0.5
            samples = roi_samples
            if image_float is not None and lower_threshold is not None:
                hu_samples = ndi.map_coordinates(
                    image_float,
                    [sample_zyx[:, 0], sample_zyx[:, 1], sample_zyx[:, 2]],
                    order=1,
                    mode="nearest",
                )
                samples = samples & (hu_samples >= lower_threshold)
                if upper_threshold is not None:
                    samples = samples & (hu_samples <= upper_threshold)
            inside = np.where(samples)[0]
            if inside.size == 0:
                continue
            first = int(inside[0])
            last = first
            while last + 1 < len(samples) and bool(samples[last + 1]):
                last += 1
            output[point_index, angle_index] = float(radii[last])
    return output


def _expected_radii(
    actual_radii: np.ndarray,
    angular_window_deg: float,
    angular_bins: int,
    longitudinal_sigma_points: float,
) -> np.ndarray:
    angular_window_bins = max(3, int(round(float(angular_window_deg) / 360.0 * int(angular_bins))))
    if angular_window_bins % 2 == 0:
        angular_window_bins += 1
    expected = np.full_like(actual_radii, np.nan, dtype=float)
    half = angular_window_bins // 2
    for point_index in range(actual_radii.shape[0]):
        row = actual_radii[point_index]
        for angle_index in range(actual_radii.shape[1]):
            idx = [(angle_index + offset) % actual_radii.shape[1] for offset in range(-half, half + 1)]
            values = row[idx]
            if np.isfinite(values).any():
                expected[point_index, angle_index] = float(np.nanmedian(values))
    if longitudinal_sigma_points > 0 and expected.shape[0] >= 3:
        try:
            from scipy.ndimage import gaussian_filter1d

            for angle_index in range(expected.shape[1]):
                column = expected[:, angle_index]
                finite = np.isfinite(column)
                if finite.sum() < 2:
                    continue
                filled = np.interp(np.arange(len(column)), np.where(finite)[0], column[finite])
                expected[:, angle_index] = gaussian_filter1d(filled, longitudinal_sigma_points, mode="nearest")
        except Exception:
            pass
    return expected


def _component_candidates(
    candidate_cells: np.ndarray,
    depth: np.ndarray,
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    centerline: _Centerline,
    spacing_xyz: tuple[float, float, float],
    segment_labels: np.ndarray | None,
    segment_names: dict[int, str],
    angular_bins: int,
    interval_mm: float,
    min_width_deg: float,
    max_width_deg: float,
    min_length_mm: float,
    max_length_mm: float,
    min_peak_prominence_mm: float | None,
    max_median_depth_fraction: float | None,
    min_focality_ratio: float | None,
    case_id: str,
    candidate_direction: str,
    candidate_id_start: int,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    components_all = _toroidal_components(candidate_cells)
    rows: list[dict[str, object]] = []
    kept_components: list[np.ndarray] = []
    dtheta = 2.0 * np.pi / int(angular_bins)
    for component in components_all:
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        width_deg = _circular_width_degrees(angle_indices, angular_bins)
        length_mm = float(centerline.s_mm[point_indices.max()] - centerline.s_mm[point_indices.min()] + interval_mm)
        if width_deg < min_width_deg or width_deg > max_width_deg:
            continue
        if length_mm < min_length_mm or length_mm > max_length_mm:
            continue
        component_depth = depth[point_indices, angle_indices]
        finite_component_depth = component_depth[np.isfinite(component_depth)]
        if finite_component_depth.size == 0:
            continue
        max_cell = int(np.nanargmax(component_depth))
        max_point = int(point_indices[max_cell])
        max_angle = int(angle_indices[max_cell])
        max_depth = float(component_depth[max_cell])
        median_depth = float(np.nanmedian(finite_component_depth))
        peak_prominence = float(max_depth - median_depth)
        median_depth_fraction = float(median_depth / max_depth) if max_depth > 0 else 1.0
        focality_ratio = float(max_depth / max(median_depth, 1e-6)) if max_depth > 0 else 0.0
        if min_peak_prominence_mm is not None and peak_prominence < float(min_peak_prominence_mm):
            continue
        if max_median_depth_fraction is not None and median_depth_fraction > float(max_median_depth_fraction):
            continue
        if min_focality_ratio is not None and focality_ratio < float(min_focality_ratio):
            continue
        area_by_point = _area_delta_by_point(component, actual_radii, expected_radii, dtheta, candidate_direction)
        max_area_delta = float(max(area_by_point.values())) if area_by_point else 0.0
        expected_area = _cross_section_area(expected_radii[max_point], dtheta)
        actual_area = _cross_section_area(actual_radii[max_point], dtheta)
        percent_delta = 100.0 * max_area_delta / expected_area if expected_area > 0 else 0.0
        max_slice_radii = actual_radii[max_point]
        finite = max_slice_radii[np.isfinite(max_slice_radii)]
        max_radius = float(np.max(finite)) if finite.size else 0.0
        min_radius = float(np.min(finite)) if finite.size else 0.0
        asymmetry = float((max_radius - min_radius) / max_radius) if max_radius > 0 else 0.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (min_radius / max_radius) ** 2))) if max_radius > 0 else 0.0
        center_xyz = centerline.points_xyz[max_point]
        center_zyx = centerline.points_zyx[max_point]
        segment_label, segment_name = _segment_at_point(center_zyx, segment_labels, segment_names)
        candidate_id = candidate_id_start + len(rows)
        is_inward = candidate_direction == "inward"
        rows.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "candidate_direction": candidate_direction,
                "centerline_index": max_point,
                "centerline_s_mm": float(centerline.s_mm[max_point]),
                "centerline_x_mm": float(center_xyz[0]),
                "centerline_y_mm": float(center_xyz[1]),
                "centerline_z_mm": float(center_xyz[2]),
                "angle_degrees": float(max_angle * 360.0 / angular_bins),
                "max_residual_depth_mm": max_depth,
                "max_protrusion_depth_mm": max_depth if is_inward else 0.0,
                "max_outward_ulcer_like_depth_mm": 0.0 if is_inward else max_depth,
                "median_residual_depth_mm": median_depth,
                "peak_prominence_mm": peak_prominence,
                "median_depth_fraction": median_depth_fraction,
                "focality_ratio": focality_ratio,
                "angular_width_degrees": width_deg,
                "longitudinal_length_mm": length_mm,
                "affected_cross_sectional_area_mm2": max_area_delta,
                "percent_lumen_compromise": float(percent_delta) if is_inward else 0.0,
                "percent_outer_area_excess": 0.0 if is_inward else float(percent_delta),
                "actual_cross_section_area_mm2": actual_area,
                "expected_cross_section_area_mm2": expected_area,
                "eccentricity": eccentricity,
                "asymmetry_index": asymmetry,
                "aortic_segment_label": segment_label,
                "aortic_segment_name": segment_name,
                "candidate_method": "centerline_normal_wall_band_expected_boundary_v1",
                "candidate_interpretation": (
                    "inward_lumen_encroachment_candidate_not_plaque_segmentation_or_diagnosis"
                    if is_inward
                    else "outward_ulcer_like_candidate_not_ulcer_diagnosis_or_plaque_segmentation"
                ),
            }
        )
        kept_components.append(component)
    return rows, kept_components


def _candidate_labelmaps(
    lumen: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    radial_step_mm: float,
    angular_bins: int,
    candidate_direction: str,
    candidate_id_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_mask = np.zeros(lumen.shape, dtype=bool)
    boundary_mask = np.zeros(lumen.shape, dtype=bool)
    labelmap = np.zeros(lumen.shape, dtype=np.uint16)
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_bins), endpoint=False)
    for candidate_id, component in enumerate(components, start=candidate_id_start):
        for point_index, angle_index in component:
            actual = float(actual_radii[point_index, angle_index])
            expected = float(expected_radii[point_index, angle_index])
            if not np.isfinite(actual) or not np.isfinite(expected):
                continue
            if candidate_direction == "inward":
                if expected <= actual:
                    continue
                radius_start, radius_stop = actual, expected
                boundary_radius = actual
            else:
                if actual <= expected:
                    continue
                radius_start, radius_stop = expected, actual
                boundary_radius = actual
            center_xyz = centerline.points_xyz[point_index]
            direction = (
                np.cos(angles[angle_index]) * centerline.normal_u_xyz[point_index]
                + np.sin(angles[angle_index]) * centerline.normal_v_xyz[point_index]
            )
            boundary_xyz = center_xyz + boundary_radius * direction
            boundary_index = _round_zyx(_xyz_mm_to_zyx(boundary_xyz[None, :], spacing_xyz)[0], lumen.shape)
            if boundary_index is not None:
                boundary_mask[boundary_index] = True
            for radius in np.arange(
                radius_start,
                radius_stop + radial_step_mm * 0.5,
                max(radial_step_mm * 0.5, 0.25),
            ):
                point_xyz = center_xyz + radius * direction
                index = _round_zyx(_xyz_mm_to_zyx(point_xyz[None, :], spacing_xyz)[0], lumen.shape)
                if index is None:
                    continue
                candidate_mask[index] = True
                labelmap[index] = max(labelmap[index], np.uint16(min(candidate_id, 65535)))
    return candidate_mask, labelmap, boundary_mask


def _candidate_patch_labelmaps(
    analysis_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    expected_radii: np.ndarray,
    radial_step_mm: float,
    angular_bins: int,
    candidate_id_start: int,
    interval_mm: float,
    inner_layer_mm: float,
    outer_layer_mm: float,
    longitudinal_padding_mm: float,
    angular_padding_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow sparse residual cells into local 3D wall-band review patches."""
    patch_mask = np.zeros(analysis_mask.shape, dtype=bool)
    labelmap = np.zeros(analysis_mask.shape, dtype=np.uint16)
    if not components:
        return patch_mask, labelmap
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_bins), endpoint=False)
    point_padding = max(0, int(np.ceil(float(longitudinal_padding_mm) / max(float(interval_mm), 1e-6))))
    angle_padding = max(0, int(np.ceil(float(angular_padding_deg) / 360.0 * int(angular_bins))))
    radius_step = max(float(radial_step_mm), 0.25)
    radial_inner = max(float(inner_layer_mm), 0.0)
    radial_outer = max(float(outer_layer_mm), 0.0)
    if radial_inner == 0.0 and radial_outer == 0.0:
        radial_inner = radial_outer = max(radius_step, 1.0)

    n_points = len(centerline.points_xyz)
    for candidate_id, component in enumerate(components, start=candidate_id_start):
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        point_start = max(0, int(point_indices.min()) - point_padding)
        point_stop = min(n_points - 1, int(point_indices.max()) + point_padding)
        angle_set: set[int] = set()
        for angle_index in angle_indices:
            for offset in range(-angle_padding, angle_padding + 1):
                angle_set.add(int((int(angle_index) + offset) % int(angular_bins)))
        for point_index in range(point_start, point_stop + 1):
            center_xyz = centerline.points_xyz[point_index]
            for angle_index in angle_set:
                expected = float(expected_radii[point_index, angle_index])
                if not np.isfinite(expected):
                    continue
                direction = (
                    np.cos(angles[angle_index]) * centerline.normal_u_xyz[point_index]
                    + np.sin(angles[angle_index]) * centerline.normal_v_xyz[point_index]
                )
                radius_start = max(0.0, expected - radial_inner)
                radius_stop = max(radius_start, expected + radial_outer)
                for radius in np.arange(radius_start, radius_stop + radius_step * 0.5, radius_step):
                    point_xyz = center_xyz + radius * direction
                    index = _round_zyx(_xyz_mm_to_zyx(point_xyz[None, :], spacing_xyz)[0], analysis_mask.shape)
                    if index is None or not bool(analysis_mask[index]):
                        continue
                    patch_mask[index] = True
                    labelmap[index] = max(labelmap[index], np.uint16(min(candidate_id, 65535)))
    return patch_mask, labelmap


def _candidate_surface_sheet_labelmaps(
    analysis_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    expected_radii: np.ndarray,
    radial_step_mm: float,
    angular_bins: int,
    candidate_id_start: int,
    interval_mm: float,
    longitudinal_padding_mm: float,
    angular_padding_deg: float,
    sheet_thickness_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Write thin curved sheets on the expected local wall surface."""
    sheet_mask = np.zeros(analysis_mask.shape, dtype=bool)
    labelmap = np.zeros(analysis_mask.shape, dtype=np.uint16)
    if not components:
        return sheet_mask, labelmap
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_bins), endpoint=False)
    point_padding = max(0, int(np.ceil(float(longitudinal_padding_mm) / max(float(interval_mm), 1e-6))))
    angle_padding = max(0, int(np.ceil(float(angular_padding_deg) / 360.0 * int(angular_bins))))
    half_thickness = max(float(sheet_thickness_mm) / 2.0, max(float(radial_step_mm), 0.25) / 2.0)
    radius_step = max(float(radial_step_mm), 0.25)
    n_points = len(centerline.points_xyz)
    for candidate_id, component in enumerate(components, start=candidate_id_start):
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        point_start = max(0, int(point_indices.min()) - point_padding)
        point_stop = min(n_points - 1, int(point_indices.max()) + point_padding)
        angle_set: set[int] = set()
        for angle_index in angle_indices:
            for offset in range(-angle_padding, angle_padding + 1):
                angle_set.add(int((int(angle_index) + offset) % int(angular_bins)))
        for point_index in range(point_start, point_stop + 1):
            center_xyz = centerline.points_xyz[point_index]
            for angle_index in angle_set:
                expected = float(expected_radii[point_index, angle_index])
                if not np.isfinite(expected):
                    continue
                direction = (
                    np.cos(angles[angle_index]) * centerline.normal_u_xyz[point_index]
                    + np.sin(angles[angle_index]) * centerline.normal_v_xyz[point_index]
                )
                radius_start = max(0.0, expected - half_thickness)
                radius_stop = expected + half_thickness
                for radius in np.arange(radius_start, radius_stop + radius_step * 0.5, radius_step):
                    point_xyz = center_xyz + radius * direction
                    index = _round_zyx(_xyz_mm_to_zyx(point_xyz[None, :], spacing_xyz)[0], analysis_mask.shape)
                    if index is None or not bool(analysis_mask[index]):
                        continue
                    sheet_mask[index] = True
                    labelmap[index] = max(labelmap[index], np.uint16(min(candidate_id, 65535)))
    return sheet_mask, labelmap


def _candidate_aorta_surface_projection_labelmaps(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    angular_bins: int,
    candidate_id_start: int,
    interval_mm: float,
    longitudinal_padding_mm: float,
    angular_padding_deg: float,
    surface_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project candidate sectors onto actual aorta boundary voxels.

    This is the main viewable QC layer: it paints the real aorta mask surface
    instead of writing synthetic samples in centerline-normal planes.
    """
    binary = np.asarray(aorta_mask, dtype=bool)
    projection_mask = np.zeros(binary.shape, dtype=bool)
    labelmap = np.zeros(binary.shape, dtype=np.uint16)
    if not components or not binary.any() or len(centerline.points_xyz) == 0:
        return projection_mask, labelmap
    surface = internal_boundary_shell(binary, spacing_xyz, depth_mm=max(float(surface_depth_mm), 0.1))
    surface_coords = np.argwhere(surface)
    if surface_coords.size == 0:
        return projection_mask, labelmap

    surface_xyz = _zyx_to_xyz_mm(surface_coords.astype(float), spacing_xyz)
    nearest_point = _nearest_centerline_indices(surface_xyz, centerline.points_xyz)
    vectors = surface_xyz - centerline.points_xyz[nearest_point]
    u = centerline.normal_u_xyz[nearest_point]
    v = centerline.normal_v_xyz[nearest_point]
    angle_values = np.mod(np.arctan2(np.sum(vectors * v, axis=1), np.sum(vectors * u, axis=1)), 2.0 * np.pi)
    surface_angle_bins = np.floor(angle_values / (2.0 * np.pi) * int(angular_bins)).astype(int) % int(angular_bins)

    point_padding = max(0, int(np.ceil(float(longitudinal_padding_mm) / max(float(interval_mm), 1e-6))))
    angle_padding = max(0, int(np.ceil(float(angular_padding_deg) / 360.0 * int(angular_bins))))
    n_points = len(centerline.points_xyz)
    for candidate_id, component in enumerate(components, start=candidate_id_start):
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        point_start = max(0, int(point_indices.min()) - point_padding)
        point_stop = min(n_points - 1, int(point_indices.max()) + point_padding)
        angle_set: set[int] = set()
        for angle_index in angle_indices:
            for offset in range(-angle_padding, angle_padding + 1):
                angle_set.add(int((int(angle_index) + offset) % int(angular_bins)))
        point_match = (nearest_point >= point_start) & (nearest_point <= point_stop)
        angle_match = np.isin(surface_angle_bins, np.fromiter(angle_set, dtype=int))
        selected = point_match & angle_match
        if not selected.any():
            continue
        coords = surface_coords[selected]
        projection_mask[tuple(coords.T)] = True
        current = labelmap[tuple(coords.T)]
        labelmap[tuple(coords.T)] = np.maximum(current, np.uint16(min(candidate_id, 65535)))
    return projection_mask, labelmap


def _candidate_aorta_surface_core_labelmaps(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    residual_depth: np.ndarray,
    angular_bins: int,
    candidate_id_start: int,
    interval_mm: float,
    longitudinal_padding_mm: float,
    angular_padding_deg: float,
    surface_depth_mm: float,
    relative_threshold: float,
    depth_window_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project only the peak residual part of each ROI onto the aorta surface."""
    binary = np.asarray(aorta_mask, dtype=bool)
    core_mask = np.zeros(binary.shape, dtype=bool)
    labelmap = np.zeros(binary.shape, dtype=np.uint16)
    if not components or not binary.any() or len(centerline.points_xyz) == 0:
        return core_mask, labelmap
    surface = internal_boundary_shell(binary, spacing_xyz, depth_mm=max(float(surface_depth_mm), 0.1))
    surface_coords = np.argwhere(surface)
    if surface_coords.size == 0:
        return core_mask, labelmap

    surface_xyz = _zyx_to_xyz_mm(surface_coords.astype(float), spacing_xyz)
    nearest_point = _nearest_centerline_indices(surface_xyz, centerline.points_xyz)
    vectors = surface_xyz - centerline.points_xyz[nearest_point]
    u = centerline.normal_u_xyz[nearest_point]
    v = centerline.normal_v_xyz[nearest_point]
    angle_values = np.mod(np.arctan2(np.sum(vectors * v, axis=1), np.sum(vectors * u, axis=1)), 2.0 * np.pi)
    surface_angle_bins = np.floor(angle_values / (2.0 * np.pi) * int(angular_bins)).astype(int) % int(angular_bins)

    point_padding = max(0, int(np.ceil(float(longitudinal_padding_mm) / max(float(interval_mm), 1e-6))))
    angle_padding = max(0, int(np.ceil(float(angular_padding_deg) / 360.0 * int(angular_bins))))
    n_points = len(centerline.points_xyz)
    for candidate_id, component in enumerate(components, start=candidate_id_start):
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        depths = residual_depth[point_indices, angle_indices]
        finite = np.isfinite(depths)
        if not finite.any():
            continue
        max_depth = float(np.nanmax(depths[finite]))
        threshold = max(
            max_depth * float(relative_threshold),
            max_depth - max(float(depth_window_mm), 0.0),
        )
        core_component = component[finite & (depths >= threshold)]
        if core_component.size == 0:
            peak_index = int(np.nanargmax(depths))
            core_component = component[[peak_index]]
        core_points = core_component[:, 0]
        core_angles = core_component[:, 1]
        point_start = max(0, int(core_points.min()) - point_padding)
        point_stop = min(n_points - 1, int(core_points.max()) + point_padding)
        angle_set: set[int] = set()
        for angle_index in core_angles:
            for offset in range(-angle_padding, angle_padding + 1):
                angle_set.add(int((int(angle_index) + offset) % int(angular_bins)))
        point_match = (nearest_point >= point_start) & (nearest_point <= point_stop)
        angle_match = np.isin(surface_angle_bins, np.fromiter(angle_set, dtype=int))
        selected = point_match & angle_match
        if not selected.any():
            continue
        coords = surface_coords[selected]
        core_mask[tuple(coords.T)] = True
        current = labelmap[tuple(coords.T)]
        labelmap[tuple(coords.T)] = np.maximum(current, np.uint16(min(candidate_id, 65535)))
    return core_mask, labelmap


def _candidate_aorta_surface_native_core_labelmaps(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    components: list[np.ndarray],
    residual_depth: np.ndarray,
    angular_bins: int,
    candidate_id_start: int,
    surface_depth_mm: float,
    relative_threshold: float,
    depth_window_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Localize peak candidate residuals as connected components on true surface voxels."""
    binary = np.asarray(aorta_mask, dtype=bool)
    native_mask = np.zeros(binary.shape, dtype=bool)
    labelmap = np.zeros(binary.shape, dtype=np.uint16)
    if not components or not binary.any() or len(centerline.points_xyz) == 0:
        return native_mask, labelmap
    surface = internal_boundary_shell(binary, spacing_xyz, depth_mm=max(float(surface_depth_mm), 0.1))
    surface_coords = np.argwhere(surface)
    if surface_coords.size == 0:
        return native_mask, labelmap

    surface_xyz = _zyx_to_xyz_mm(surface_coords.astype(float), spacing_xyz)
    nearest_point = _nearest_centerline_indices(surface_xyz, centerline.points_xyz)
    vectors = surface_xyz - centerline.points_xyz[nearest_point]
    u = centerline.normal_u_xyz[nearest_point]
    v = centerline.normal_v_xyz[nearest_point]
    angle_values = np.mod(np.arctan2(np.sum(vectors * v, axis=1), np.sum(vectors * u, axis=1)), 2.0 * np.pi)
    surface_angle_bins = np.floor(angle_values / (2.0 * np.pi) * int(angular_bins)).astype(int) % int(angular_bins)
    surface_codes = nearest_point.astype(np.int64) * int(angular_bins) + surface_angle_bins.astype(np.int64)

    try:
        from scipy.ndimage import generate_binary_structure, label
    except ImportError:
        label = None
        generate_binary_structure = None

    for candidate_id, component in enumerate(components, start=candidate_id_start):
        point_indices = component[:, 0]
        angle_indices = component[:, 1]
        depths = residual_depth[point_indices, angle_indices]
        finite = np.isfinite(depths)
        if not finite.any():
            continue
        max_depth = float(np.nanmax(depths[finite]))
        threshold = max(
            max_depth * float(relative_threshold),
            max_depth - max(float(depth_window_mm), 0.0),
        )
        core_component = component[finite & (depths >= threshold)]
        if core_component.size == 0:
            peak_index = int(np.nanargmax(depths))
            core_component = component[[peak_index]]
        core_codes = (
            core_component[:, 0].astype(np.int64) * int(angular_bins)
            + core_component[:, 1].astype(np.int64)
        )
        selected = np.isin(surface_codes, core_codes)
        if not selected.any():
            expanded_codes: set[int] = set()
            max_point = len(centerline.points_xyz) - 1
            for point_index, angle_index in core_component:
                for point_offset in (-1, 0, 1):
                    expanded_point = min(max(int(point_index) + point_offset, 0), max_point)
                    for angle_offset in (-1, 0, 1):
                        expanded_angle = (int(angle_index) + angle_offset) % int(angular_bins)
                        expanded_codes.add(expanded_point * int(angular_bins) + expanded_angle)
            selected = np.isin(surface_codes, np.fromiter(expanded_codes, dtype=np.int64))
        if not selected.any():
            continue
        selected_coords = surface_coords[selected]
        candidate_surface = np.zeros(binary.shape, dtype=bool)
        candidate_surface[tuple(selected_coords.T)] = True
        if label is None or generate_binary_structure is None:
            native_mask[candidate_surface] = True
            labelmap[candidate_surface] = np.maximum(
                labelmap[candidate_surface],
                np.uint16(min(candidate_id, 65535)),
            )
            continue
        component_labels, component_count = label(candidate_surface, structure=generate_binary_structure(3, 2))
        for component_label in range(1, int(component_count) + 1):
            surface_component = component_labels == component_label
            if not surface_component.any():
                continue
            native_mask[surface_component] = True
            labelmap[surface_component] = np.maximum(
                labelmap[surface_component],
                np.uint16(min(candidate_id, 65535)),
            )
    return native_mask, labelmap


def _nearest_centerline_indices(points_xyz: np.ndarray, centerline_xyz: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(centerline_xyz).query(points_xyz, k=1)
        return np.asarray(indices, dtype=int)
    except Exception:
        indices = np.zeros(len(points_xyz), dtype=int)
        chunk_size = 50_000
        for start in range(0, len(points_xyz), chunk_size):
            stop = min(start + chunk_size, len(points_xyz))
            distances = np.sum((points_xyz[start:stop, None, :] - centerline_xyz[None, :, :]) ** 2, axis=2)
            indices[start:stop] = np.argmin(distances, axis=1)
        return indices


def _filter_outward_external_contrast_components(
    rows: list[dict[str, object]],
    components: list[np.ndarray],
    lumen: np.ndarray,
    contrast_like_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    radial_step_mm: float,
    angular_bins: int,
    max_external_component_volume_mm3: float | None,
    max_candidate_outside_aorta_fraction: float | None,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    apply_external_component_filter = (
        max_external_component_volume_mm3 is not None and max_external_component_volume_mm3 > 0
    )
    apply_outside_fraction_filter = (
        max_candidate_outside_aorta_fraction is not None and max_candidate_outside_aorta_fraction >= 0
    )
    if not apply_external_component_filter and not apply_outside_fraction_filter:
        return rows, components
    if not rows or not components:
        return rows, components
    try:
        from scipy import ndimage as ndi
    except Exception:
        return rows, components

    lumen_bool = np.asarray(lumen, dtype=bool)
    external_contrast = np.asarray(contrast_like_mask, dtype=bool) & ~lumen_bool
    if external_contrast.any():
        crop, slices = _crop_true(external_contrast | lumen_bool)
        external_crop = external_contrast[slices]
        labels_crop, _ = ndi.label(external_crop)
        sizes = np.bincount(labels_crop.ravel())
        starts = np.asarray([sl.start or 0 for sl in slices], dtype=int)
    else:
        labels_crop = np.zeros((0, 0, 0), dtype=int)
        sizes = np.zeros(1, dtype=int)
        starts = np.zeros(3, dtype=int)
    voxel_volume = float(np.prod(spacing_xyz))
    max_external_voxels = (
        max(1, int(np.floor(float(max_external_component_volume_mm3) / max(voxel_volume, 1e-9))))
        if apply_external_component_filter
        else None
    )

    kept_rows: list[dict[str, object]] = []
    kept_components: list[np.ndarray] = []
    for row, component in zip(rows, components):
        voxels = _candidate_component_voxels(
            component=component,
            shape=lumen.shape,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            actual_radii=actual_radii,
            expected_radii=expected_radii,
            radial_step_mm=radial_step_mm,
            angular_bins=angular_bins,
            candidate_direction="outward_ulcer_like",
        )
        if voxels.size == 0:
            row["external_contrast_component_max_volume_mm3"] = 0.0
            kept_rows.append(row)
            kept_components.append(component)
            continue
        outside_voxels = voxels[~lumen_bool[tuple(voxels.T)]]
        outside_fraction = float(len(outside_voxels) / max(len(voxels), 1))
        row["candidate_outside_aorta_fraction"] = outside_fraction
        if apply_outside_fraction_filter and outside_fraction > float(max_candidate_outside_aorta_fraction):
            continue
        if outside_voxels.size == 0:
            row["external_contrast_component_max_volume_mm3"] = 0.0
            kept_rows.append(row)
            kept_components.append(component)
            continue
        local = outside_voxels - starts[None, :]
        in_crop = np.all((local >= 0) & (local < np.asarray(labels_crop.shape)[None, :]), axis=1)
        local = local[in_crop]
        component_labels = labels_crop[tuple(local.T)] if local.size else np.asarray([], dtype=int)
        component_labels = component_labels[component_labels > 0]
        max_size = int(sizes[np.unique(component_labels)].max()) if component_labels.size else 0
        row["external_contrast_component_max_volume_mm3"] = float(max_size * voxel_volume)
        if max_external_voxels is None or max_size <= max_external_voxels:
            kept_rows.append(row)
            kept_components.append(component)
    return kept_rows, kept_components


def _candidate_component_voxels(
    component: np.ndarray,
    shape: tuple[int, ...],
    spacing_xyz: tuple[float, float, float],
    centerline: _Centerline,
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    radial_step_mm: float,
    angular_bins: int,
    candidate_direction: str,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_bins), endpoint=False)
    indices: set[tuple[int, int, int]] = set()
    for point_index, angle_index in component:
        actual = float(actual_radii[point_index, angle_index])
        expected = float(expected_radii[point_index, angle_index])
        if not np.isfinite(actual) or not np.isfinite(expected):
            continue
        if candidate_direction == "inward":
            if expected <= actual:
                continue
            radius_start, radius_stop = actual, expected
        else:
            if actual <= expected:
                continue
            radius_start, radius_stop = expected, actual
        center_xyz = centerline.points_xyz[point_index]
        direction = (
            np.cos(angles[angle_index]) * centerline.normal_u_xyz[point_index]
            + np.sin(angles[angle_index]) * centerline.normal_v_xyz[point_index]
        )
        for radius in np.arange(
            radius_start,
            radius_stop + radial_step_mm * 0.5,
            max(radial_step_mm * 0.5, 0.25),
        ):
            point_xyz = center_xyz + radius * direction
            index = _round_zyx(_xyz_mm_to_zyx(point_xyz[None, :], spacing_xyz)[0], shape)
            if index is not None:
                indices.add(index)
    if not indices:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(sorted(indices), dtype=int)


def _point_features(
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    inward_depth: np.ndarray,
    outward_depth: np.ndarray,
    inward_cells: np.ndarray,
    outward_cells: np.ndarray,
    centerline: _Centerline,
    case_id: str,
    angular_bins: int,
    software_version: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dtheta = 2.0 * np.pi / int(angular_bins)
    for point_index, center_xyz in enumerate(centerline.points_xyz):
        point_inward_depth = inward_depth[point_index]
        point_outward_depth = outward_depth[point_index]
        inward_angles = inward_cells[point_index]
        outward_angles = outward_cells[point_index]
        expected_area = _cross_section_area(expected_radii[point_index], dtheta)
        actual_area = _cross_section_area(actual_radii[point_index], dtheta)
        max_inward_depth = float(np.nanmax(point_inward_depth)) if np.isfinite(point_inward_depth).any() else 0.0
        max_outward_depth = float(np.nanmax(point_outward_depth)) if np.isfinite(point_outward_depth).any() else 0.0
        area_deficit = max(0.0, expected_area - actual_area)
        area_excess = max(0.0, actual_area - expected_area)
        rows.append(
            {
                "case_id": case_id,
                "centerline_index": point_index,
                "centerline_s_mm": float(centerline.s_mm[point_index]),
                "x_mm": float(center_xyz[0]),
                "y_mm": float(center_xyz[1]),
                "z_mm": float(center_xyz[2]),
                "max_protrusion_depth_mm": max_inward_depth,
                "max_outward_ulcer_like_depth_mm": max_outward_depth,
                "candidate_angle_count": int(inward_angles.sum() + outward_angles.sum()),
                "inward_candidate_angle_count": int(inward_angles.sum()),
                "outward_candidate_angle_count": int(outward_angles.sum()),
                "candidate_angular_width_degrees": float((inward_angles | outward_angles).sum() * 360.0 / angular_bins),
                "inward_candidate_angular_width_degrees": float(inward_angles.sum() * 360.0 / angular_bins),
                "outward_candidate_angular_width_degrees": float(outward_angles.sum() * 360.0 / angular_bins),
                "actual_cross_section_area_mm2": actual_area,
                "expected_cross_section_area_mm2": expected_area,
                "area_deficit_mm2": area_deficit,
                "area_excess_mm2": area_excess,
                "percent_lumen_compromise": float(100.0 * area_deficit / expected_area) if expected_area > 0 else 0.0,
                "percent_outer_area_excess": float(100.0 * area_excess / expected_area) if expected_area > 0 else 0.0,
                "protrusion_method": "centerline_normal_wall_band_expected_boundary_v1",
                "protrusion_interpretation": "local_boundary_candidate_not_plaque_or_ulcer_classifier",
                "software_version": software_version,
            }
        )
    return pd.DataFrame(rows)


def _summary_features(
    candidates: pd.DataFrame,
    case_id: str,
    software_version: str,
    high_risk_depth_mm: float = 4.0,
) -> pd.DataFrame:
    if candidates.empty:
        values = {
            "candidate_count": 0,
            "inward_candidate_count": 0,
            "outward_ulcer_like_candidate_count": 0,
            "candidate_count_depth_ge_4mm": 0,
            "max_protrusion_depth_mm": 0.0,
            "max_outward_ulcer_like_depth_mm": 0.0,
            "max_percent_lumen_compromise": 0.0,
            "max_percent_outer_area_excess": 0.0,
            "max_affected_cross_sectional_area_mm2": 0.0,
        }
    else:
        inward = candidates[candidates["candidate_direction"] == "inward"]
        outward = candidates[candidates["candidate_direction"] == "outward_ulcer_like"]
        values = {
            "candidate_count": int(len(candidates)),
            "inward_candidate_count": int(len(inward)),
            "outward_ulcer_like_candidate_count": int(len(outward)),
            "candidate_count_depth_ge_4mm": int((candidates["max_residual_depth_mm"] >= high_risk_depth_mm).sum()),
            "max_protrusion_depth_mm": float(inward["max_protrusion_depth_mm"].max()) if not inward.empty else 0.0,
            "max_outward_ulcer_like_depth_mm": (
                float(outward["max_outward_ulcer_like_depth_mm"].max()) if not outward.empty else 0.0
            ),
            "max_percent_lumen_compromise": float(candidates["percent_lumen_compromise"].max()),
            "max_percent_outer_area_excess": float(candidates["percent_outer_area_excess"].max()),
            "max_affected_cross_sectional_area_mm2": float(candidates["affected_cross_sectional_area_mm2"].max()),
        }
    rows = [
        feature_row(
            case_id=case_id,
            region="aorta_lumen",
            feature_group="lumen_protrusions",
            feature_name=name,
            feature_value=value,
            units=_summary_units(name),
            mask_name="lumen_protrusion_candidates",
            software_version=software_version,
        )
        for name, value in values.items()
    ]
    return pd.DataFrame(rows)


def _toroidal_components(binary: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(binary.shape, dtype=bool)
    components: list[np.ndarray] = []
    n_points, n_angles = binary.shape
    for point_index, angle_index in np.argwhere(binary):
        if visited[point_index, angle_index]:
            continue
        stack = [(int(point_index), int(angle_index))]
        visited[point_index, angle_index] = True
        cells: list[tuple[int, int]] = []
        while stack:
            point, angle = stack.pop()
            cells.append((point, angle))
            for dp in (-1, 0, 1):
                for da in (-1, 0, 1):
                    if dp == 0 and da == 0:
                        continue
                    np_idx = point + dp
                    if np_idx < 0 or np_idx >= n_points:
                        continue
                    na_idx = (angle + da) % n_angles
                    if binary[np_idx, na_idx] and not visited[np_idx, na_idx]:
                        visited[np_idx, na_idx] = True
                        stack.append((np_idx, na_idx))
        components.append(np.asarray(cells, dtype=int))
    return components


def _area_delta_by_point(
    component: np.ndarray,
    actual_radii: np.ndarray,
    expected_radii: np.ndarray,
    dtheta: float,
    candidate_direction: str,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for point_index in np.unique(component[:, 0]):
        angles = component[component[:, 0] == point_index, 1]
        actual = actual_radii[point_index, angles]
        expected = expected_radii[point_index, angles]
        valid = np.isfinite(actual) & np.isfinite(expected)
        if candidate_direction == "inward":
            valid &= expected > actual
            area = float(0.5 * np.sum((expected[valid] ** 2 - actual[valid] ** 2) * dtheta))
        else:
            valid &= actual > expected
            area = float(0.5 * np.sum((actual[valid] ** 2 - expected[valid] ** 2) * dtheta))
        out[int(point_index)] = area
    return out


def _cross_section_area(radii: np.ndarray, dtheta: float) -> float:
    valid = np.isfinite(radii)
    if not valid.any():
        return 0.0
    return float(0.5 * np.sum((radii[valid] ** 2) * dtheta))


def _circular_width_degrees(indices: np.ndarray, angular_bins: int) -> float:
    unique = np.unique(indices.astype(int))
    if unique.size == 0:
        return 0.0
    if unique.size == angular_bins:
        return 360.0
    sorted_idx = np.sort(unique)
    gaps = np.diff(np.concatenate([sorted_idx, [sorted_idx[0] + angular_bins]]))
    largest_gap = int(np.max(gaps))
    occupied_span_bins = int(angular_bins - largest_gap + 1)
    return float(occupied_span_bins * 360.0 / angular_bins)


def _segment_at_point(
    point_zyx: np.ndarray,
    segment_labels: np.ndarray | None,
    segment_names: dict[int, str],
) -> tuple[int, str]:
    if segment_labels is None:
        return 1, "whole_aorta"
    index = _round_zyx(point_zyx, segment_labels.shape)
    if index is None:
        return 0, "unknown"
    label = int(segment_labels[index])
    return label, segment_names.get(label, f"label_{label}")


def _zyx_to_xyz_mm(zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    spacing = np.asarray(spacing_xyz, dtype=float)
    return zyx[:, [2, 1, 0]] * spacing[None, :]


def _xyz_mm_to_zyx(xyz: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    spacing = np.asarray(spacing_xyz, dtype=float)
    return (xyz / spacing[None, :])[:, [2, 1, 0]]


def _crop_true(mask: np.ndarray, pad: int = 2) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        slices = tuple(slice(0, dim) for dim in mask.shape)
        return mask, slices  # type: ignore[return-value]
    mins = np.maximum(coords.min(axis=0) - int(pad), 0)
    maxs = np.minimum(coords.max(axis=0) + int(pad) + 1, np.asarray(mask.shape))
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    return mask[slices], slices  # type: ignore[index, return-value]


def _snap_points_to_mask(
    points_zyx: np.ndarray,
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    max_search_mm: float,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    radii = np.maximum(1, np.ceil(float(max_search_mm) / np.maximum(spacing_zyx, 1e-6)).astype(int))
    snapped = np.asarray(points_zyx, dtype=float).copy()
    for idx, point in enumerate(snapped):
        rounded = _round_zyx(point, binary.shape)
        if rounded is not None and binary[rounded]:
            continue
        center = np.asarray([int(round(float(value))) for value in point], dtype=int)
        starts = np.maximum(center - radii, 0)
        stops = np.minimum(center + radii + 1, binary.shape)
        slices = tuple(slice(int(starts[axis]), int(stops[axis])) for axis in range(3))
        local = np.argwhere(binary[slices])
        if local.size == 0:
            continue
        candidates = local + starts[None, :]
        distances = np.linalg.norm((candidates - point[None, :]) * spacing_zyx[None, :], axis=1)
        snapped[idx] = candidates[int(np.argmin(distances))].astype(float)
    return snapped


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors, dtype=float)
    valid = norms > 0
    out[valid] = vectors[valid] / norms[valid, None]
    if (~valid).any() and valid.any():
        out[~valid] = out[np.where(valid)[0][0]]
    return out


def _normal_bases(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal_u = np.zeros_like(tangents, dtype=float)
    normal_v = np.zeros_like(tangents, dtype=float)
    for idx, tangent in enumerate(tangents):
        reference = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(tangent, reference))) > 0.85:
            reference = np.asarray([0.0, 1.0, 0.0])
        u = np.cross(tangent, reference)
        norm = float(np.linalg.norm(u))
        if norm == 0:
            u = np.asarray([1.0, 0.0, 0.0])
        else:
            u = u / norm
        v = np.cross(tangent, u)
        v_norm = float(np.linalg.norm(v))
        normal_u[idx] = u
        normal_v[idx] = v / v_norm if v_norm else np.asarray([0.0, 1.0, 0.0])
    return normal_u, normal_v


def _round_zyx(point_zyx: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    idx = tuple(int(round(float(value))) for value in point_zyx)
    if any(idx[axis] < 0 or idx[axis] >= shape[axis] for axis in range(3)):
        return None
    return idx  # type: ignore[return-value]


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(values, sigma, mode="nearest")
    except Exception:
        radius = max(1, int(round(sigma * 2)))
        padded = np.pad(values, radius, mode="edge")
        kernel = np.ones(radius * 2 + 1, dtype=float)
        kernel /= kernel.sum()
        return np.convolve(padded, kernel, mode="valid")


def _summary_units(feature_name: str) -> str:
    if feature_name.endswith("_mm"):
        return "mm"
    if feature_name.endswith("_mm2"):
        return "mm2"
    if "percent" in feature_name:
        return "percent"
    if "count" in feature_name:
        return "count"
    return ""
