import numpy as np

from aorta_cta_radiomics.lumen_protrusions import detect_lumen_protrusions


def _tube_mask(shape=(25, 72, 72), radius=12, curve_amplitude=0.0):
    z, y, x = np.indices(shape)
    center_y = shape[1] / 2
    center_x = shape[2] / 2 + curve_amplitude * np.sin((z - shape[0] / 2) / shape[0] * np.pi)
    return ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2


def test_lumen_protrusion_detector_keeps_smooth_curved_tube_negative():
    mask = _tube_mask(curve_amplitude=8.0)

    result = detect_lumen_protrusions(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=20.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=60,
        longitudinal_smoothing_mm=8,
        min_depth_mm=3.0,
    )

    assert result.candidates.empty
    assert result.candidate_mask.sum() == 0
    assert result.aorta_surface_projection_mask.sum() == 0
    assert result.aorta_surface_core_mask.sum() == 0
    assert result.summary_features.loc[
        result.summary_features["feature_name"] == "candidate_count",
        "feature_value",
    ].iloc[0] == 0


def test_lumen_protrusion_detector_flags_focal_inward_notch():
    mask = _tube_mask(radius=14)
    z, y, x = np.indices(mask.shape)
    center_y = mask.shape[1] / 2
    center_x = mask.shape[2] / 2
    angle = np.degrees(np.arctan2(y - center_y, x - center_x))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    notch = (z >= 11) & (z <= 13) & (np.abs(angle) <= 16) & (radius > 9)
    mask[notch] = False

    result = detect_lumen_protrusions(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        max_angular_width_deg=90,
        max_length_mm=12,
        end_margin_mm=0,
    )

    assert not result.candidates.empty
    assert set(result.candidates["candidate_direction"]) == {"inward"}
    assert result.candidates["max_protrusion_depth_mm"].max() >= 3.0
    assert result.candidates["angular_width_degrees"].max() <= 90
    assert result.candidate_mask.sum() > 0
    assert result.candidate_labelmap.max() > 0
    assert result.inward_candidate_mask.sum() > 0
    assert result.outward_candidate_mask.sum() == 0
    assert result.inward_aorta_surface_projection_mask.sum() > 0
    assert result.inward_aorta_surface_projection_labelmap.max() > 0
    assert np.all(result.inward_aorta_surface_projection_mask <= mask)
    assert result.inward_aorta_surface_core_mask.sum() > 0
    assert result.inward_aorta_surface_core_labelmap.max() > 0
    assert result.inward_aorta_surface_core_mask.sum() <= result.inward_aorta_surface_projection_mask.sum()
    assert np.all(result.inward_aorta_surface_core_mask <= mask)
    assert result.inward_aorta_surface_native_mask.sum() > 0
    assert result.inward_aorta_surface_native_labelmap.max() > 0
    assert result.inward_aorta_surface_native_mask.sum() <= result.inward_aorta_surface_core_mask.sum()
    assert np.all(result.inward_aorta_surface_native_mask <= mask)
    assert result.boundary_mask.sum() > 0
    assert result.summary_features["feature_name"].str.contains("candidate_count").any()


def test_lumen_protrusion_detector_flags_outward_ulcer_like_pocket():
    mask = _tube_mask(radius=12)
    image = np.zeros(mask.shape, dtype=float)
    image[mask] = 420.0
    z, y, x = np.indices(mask.shape)
    center_y = mask.shape[1] / 2
    center_x = mask.shape[2] / 2
    angle = np.degrees(np.arctan2(y - center_y, x - center_x))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    pocket = (z >= 11) & (z <= 13) & (np.abs(angle) <= 14) & (radius > 12) & (radius <= 16)
    image[pocket] = 390.0

    result = detect_lumen_protrusions(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        image_hu=image,
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        max_angular_width_deg=90,
        max_length_mm=12,
        end_margin_mm=0,
        analysis_outer_layer_mm=5.0,
        detect_outward=True,
        contrast_lower_margin_hu=100.0,
        min_contrast_hu=150.0,
    )

    assert not result.candidates.empty
    assert "outward_ulcer_like" in set(result.candidates["candidate_direction"])
    assert result.candidates["max_outward_ulcer_like_depth_mm"].max() >= 2.0
    assert result.outward_candidate_mask.sum() > 0
    assert result.outward_candidate_labelmap.max() > 0
    assert result.outward_aorta_surface_projection_mask.sum() > 0
    assert result.outward_aorta_surface_projection_labelmap.max() > 0
    assert np.all(result.outward_aorta_surface_projection_mask <= mask)
    assert result.outward_aorta_surface_core_mask.sum() > 0
    assert result.outward_aorta_surface_core_labelmap.max() > 0
    assert result.outward_aorta_surface_core_mask.sum() <= result.outward_aorta_surface_projection_mask.sum()
    assert np.all(result.outward_aorta_surface_core_mask <= mask)
    assert result.outward_aorta_surface_native_mask.sum() > 0
    assert result.outward_aorta_surface_native_labelmap.max() > 0
    assert result.outward_aorta_surface_native_mask.sum() <= result.outward_aorta_surface_core_mask.sum()
    assert np.all(result.outward_aorta_surface_native_mask <= mask)


def test_lumen_protrusion_focality_guard_rejects_broad_smooth_outward_sector():
    lumen = _tube_mask(radius=12)
    wall = _tube_mask(radius=17) & ~lumen
    image = np.zeros(lumen.shape, dtype=float)
    image[lumen] = 420.0
    z, y, x = np.indices(lumen.shape)
    center_y = lumen.shape[1] / 2
    center_x = lumen.shape[2] / 2
    angle = np.degrees(np.arctan2(y - center_y, x - center_x))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    smooth_sector = (z >= 8) & (z <= 16) & (np.abs(angle) <= 38) & (radius > 12) & (radius <= 15)
    image[smooth_sector] = 390.0

    result = detect_lumen_protrusions(
        lumen,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        image_hu=image,
        analysis_mask_override=lumen | wall,
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        outward_min_depth_mm=1.5,
        outward_max_angular_width_deg=110,
        outward_max_length_mm=20,
        end_margin_mm=0,
        detect_outward=True,
        contrast_lower_margin_hu=100.0,
        min_contrast_hu=150.0,
        outward_min_peak_prominence_mm=1.0,
        outward_max_median_depth_fraction=0.8,
        outward_min_focality_ratio=1.2,
    )

    assert result.candidates.empty
    assert result.outward_candidate_mask.sum() == 0


def test_lumen_protrusion_focality_guard_keeps_peaked_outward_sector():
    lumen = _tube_mask(radius=12)
    wall = _tube_mask(radius=18) & ~lumen
    image = np.zeros(lumen.shape, dtype=float)
    image[lumen] = 420.0
    z, y, x = np.indices(lumen.shape)
    center_y = lumen.shape[1] / 2
    center_x = lumen.shape[2] / 2
    angle = np.degrees(np.arctan2(y - center_y, x - center_x))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    shoulder = (z >= 11) & (z <= 13) & (np.abs(angle) <= 16) & (radius > 12) & (radius <= 14)
    peak = (z >= 11) & (z <= 13) & (np.abs(angle) <= 6) & (radius > 14) & (radius <= 17)
    image[shoulder | peak] = 390.0

    result = detect_lumen_protrusions(
        lumen,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        image_hu=image,
        analysis_mask_override=lumen | wall,
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        outward_min_depth_mm=1.5,
        outward_max_angular_width_deg=110,
        outward_max_length_mm=20,
        end_margin_mm=0,
        detect_outward=True,
        contrast_lower_margin_hu=100.0,
        min_contrast_hu=150.0,
        outward_min_peak_prominence_mm=0.5,
        outward_max_median_depth_fraction=0.9,
        outward_min_focality_ratio=1.1,
    )

    assert "outward_ulcer_like" in set(result.candidates["candidate_direction"])
    assert result.candidates["peak_prominence_mm"].max() >= 0.5
    assert result.outward_candidate_mask.sum() > 0
    assert result.outward_aorta_surface_native_mask.sum() > 0


def test_lumen_protrusion_detector_uses_wall_lumen_analysis_override_for_outward_pocket():
    lumen = _tube_mask(radius=12)
    wall = _tube_mask(radius=17) & ~lumen
    image = np.zeros(lumen.shape, dtype=float)
    image[lumen] = 420.0
    z, y, x = np.indices(lumen.shape)
    center_y = lumen.shape[1] / 2
    center_x = lumen.shape[2] / 2
    angle = np.degrees(np.arctan2(y - center_y, x - center_x))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    pocket = (z >= 11) & (z <= 13) & (np.abs(angle) <= 14) & (radius > 12) & (radius <= 16)
    image[pocket] = 390.0

    result = detect_lumen_protrusions(
        lumen,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        image_hu=image,
        analysis_mask_override=lumen | wall,
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        max_angular_width_deg=90,
        max_length_mm=12,
        end_margin_mm=0,
        detect_outward=True,
        contrast_lower_margin_hu=100.0,
        min_contrast_hu=150.0,
    )

    assert "outward_ulcer_like" in set(result.candidates["candidate_direction"])
    assert result.outward_candidate_mask.sum() > 0
    assert np.all(result.outward_candidate_mask <= (lumen | wall))
    assert result.contrast_like_mask[pocket].any()


def test_lumen_protrusion_hu_gate_ignores_low_hu_outer_shell():
    mask = _tube_mask(radius=12)
    image = np.zeros(mask.shape, dtype=float)
    image[mask] = 420.0

    result = detect_lumen_protrusions(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        image_hu=image,
        centerline_interval_mm=1.0,
        centerline_smoothing_mm=3.0,
        max_radius_mm=22.0,
        angular_bins=72,
        radial_sample_step_mm=0.5,
        angular_median_window_deg=70,
        longitudinal_smoothing_mm=8,
        min_depth_mm=2.0,
        max_angular_width_deg=90,
        max_length_mm=12,
        end_margin_mm=0,
        analysis_outer_layer_mm=5.0,
        detect_outward=True,
        contrast_lower_margin_hu=100.0,
        min_contrast_hu=150.0,
    )

    assert result.candidates.empty
    assert result.outward_candidate_mask.sum() == 0
    assert result.outward_aorta_surface_projection_mask.sum() == 0
    assert result.outward_aorta_surface_core_mask.sum() == 0
    assert result.contrast_like_mask.sum() == mask.sum()
