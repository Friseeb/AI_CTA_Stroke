"""Configuration loading for the aorta CTA radiomics pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "image": {
        "assume_hu": True,
        "resample_mask_if_needed": True,
    },
    "compute": {
        "crop_margin_mm": 8.0,
    },
    "mask_cleaning": {
        "keep_largest_component": True,
        "fill_holes": True,
        "min_component_voxels": 50,
        "small_mask_volume_mm3": 1000.0,
        "large_mask_volume_mm3": 500000.0,
    },
    "shells": {
        "base": [
            {"name": "shell_0_2mm", "inner_mm": 0.0, "outer_mm": 2.0},
            {"name": "shell_2_5mm", "inner_mm": 2.0, "outer_mm": 5.0},
            {"name": "shell_5_10mm", "inner_mm": 5.0, "outer_mm": 10.0},
        ],
        "combined_internal_mm": 1.0,
        "aorta_wall_internal_mm": 2.0,
        "aorta_wall_external_mm": 2.0,
        "calcification_local_outer_mm": 5.0,
    },
    "calcification": {
        "enabled": True,
        "thresholds_hu": [130, 300, 500, 600],
        "roi": "aorta_wall_band",
        "save_masks": True,
        "agatston_like": True,
        "dynamic_wall": {
            "enabled": True,
            "seed_threshold_hu": 500,
            "lumen_margin_hu": 75,
            "min_candidate_hu": 300,
            "lumen_core_distance_mm": 5.0,
            "search_internal_mm": 5.0,
            "search_external_mm": 2.0,
            "smooth_lumen_profile_mm": 10.0,
            "min_core_voxels_per_slice": 20,
            "exclude_external_contrast_touching": True,
            "external_contrast_tolerance_hu": 75,
        },
    },
    "geometry": {
        "enabled": True,
        "min_slice_voxels": 20,
        "max_branch_link_distance_mm": 20.0,
        "max_components_per_slice": 4,
    },
    "lumen_protrusions": {
        "enabled": False,
        "centerline_interval_mm": 2.0,
        "centerline_smoothing_mm": 6.0,
        "plane_spacing_mm": 0.75,
        "radial_sample_step_mm": 0.5,
        "max_radius_mm": 35.0,
        "angular_bins": 72,
        "angular_median_window_deg": 50.0,
        "inward_angular_median_window_deg": None,
        "outward_angular_median_window_deg": None,
        "longitudinal_smoothing_mm": 12.0,
        "inward_longitudinal_smoothing_mm": None,
        "outward_longitudinal_smoothing_mm": None,
        "min_depth_mm": 2.0,
        "outward_min_depth_mm": None,
        "high_risk_depth_mm": 4.0,
        "min_angular_width_deg": 5.0,
        "max_angular_width_deg": 90.0,
        "outward_min_angular_width_deg": None,
        "outward_max_angular_width_deg": None,
        "min_length_mm": 1.0,
        "max_length_mm": 25.0,
        "outward_min_length_mm": None,
        "outward_max_length_mm": None,
        "min_peak_prominence_mm": None,
        "outward_min_peak_prominence_mm": None,
        "max_median_depth_fraction": None,
        "outward_max_median_depth_fraction": None,
        "min_focality_ratio": None,
        "outward_min_focality_ratio": None,
        "end_margin_mm": 10.0,
        "analysis_inner_layer_mm": 0.0,
        "analysis_outer_layer_mm": 0.0,
        "patch_longitudinal_padding_mm": 2.0,
        "patch_angular_padding_deg": 10.0,
        "surface_sheet_thickness_mm": 1.0,
        "surface_projection_depth_mm": 1.0,
        "surface_core_relative_threshold": 0.75,
        "surface_core_depth_window_mm": 1.0,
        "surface_core_longitudinal_padding_mm": 0.0,
        "surface_core_angular_padding_deg": 2.5,
        "inward_qc_depth_thresholds_mm": [],
        "outward_qc_depth_thresholds_mm": [],
        "thresholded_qc_sources": ["aorta_surface_native", "aorta_surface_core"],
        "detect_inward": True,
        "detect_outward": False,
        "intensity_gate_enabled": True,
        "centerline_core_radius_mm": 2.0,
        "contrast_lower_margin_hu": 120.0,
        "min_contrast_hu": 150.0,
        "max_contrast_hu_above_reference": 300.0,
        "contrast_reference_lower_fraction": None,
        "contrast_reference_upper_fraction": None,
        "max_external_contrast_component_volume_mm3": 50.0,
        "max_candidate_outside_aorta_fraction": 1.0,
        "clip_candidate_masks_to_analysis_mask": True,
        "save_masks": True,
    },
    "fat_omics": {
        "enabled": True,
        "external_radius_mm": 5.0,
        "adipose_hu_min": -190.0,
        "adipose_hu_max": -30.0,
        "angle_bins": 12,
        "texture_levels": 16,
        "radial_bins_mm": [[0.0, 2.0], [2.0, 5.0]],
        "high_hu_bins": {
            "m70_m30": [-70.0, -30.0],
            "m50_m30": [-50.0, -30.0],
        },
        "save_mask": True,
    },
    "wall_from_fat": {
        "enabled": False,
        "outer_limit_mm": 5.0,
        "close_radius_mm": 3.0,
        "lumen_core_distance_mm": 5.0,
        "centerline_core_radius_mm": 2.0,
        "contrast_lower_margin_hu": 120.0,
        "min_lumen_hu": 150.0,
        "max_lumen_hu_above_reference": 300.0,
        "lumen_reference_lower_fraction": None,
        "lumen_reference_upper_fraction": None,
        "lumen_reference_statistic": "median",
        "require_lumen_seed_connectivity": False,
        "use_input_aorta_as_lumen_floor": False,
        "smooth_lumen_profile_mm": 10.0,
        "min_core_voxels_per_slice": 20,
        "wall_hu_min": -30.0,
        "wall_hu_max": 1200.0,
        "exclude_fat_from_wall": True,
        "exclude_calcification_hu": None,
        "include_calcification_in_wall": True,
        "lumen_correction_enabled": False,
        "lumen_correction_outer_mm": 2.0,
        "lumen_correction_close_radius_mm": 1.0,
        "lumen_correction_lower_margin_hu": None,
        "lumen_correction_min_hu": None,
        "lumen_correction_max_above_reference_hu": None,
        "save_masks": True,
        "protrusions": {
            "enabled": False,
        },
    },
    "radiomics": {
        "enabled": True,
        "backend": "pyradiomics",
        "device": "cpu",
        "settings_path": "configs/radiomics.yaml",
        "regions": ["aorta_mask", "aorta_wall_band", "periaortic_fat", "shell_0_2mm", "shell_2_5mm", "shell_5_10mm"],
        "include_diagnostics": False,
    },
    "outputs": {
        "save_masks": True,
        "save_figures": True,
        "software_version": "0.1.0",
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update values into a copy of base."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge it over the built-in defaults."""
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve a config path relative to the project root if it is not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate
