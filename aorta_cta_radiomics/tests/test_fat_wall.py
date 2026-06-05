import numpy as np

from aorta_cta_radiomics.calcification import _slice_lumen_reference_hu
from aorta_cta_radiomics.fat_wall import _reference_hu_bounds, extract_fat_closed_aortic_wall


def test_fat_closed_wall_excludes_contrast_lumen_and_fat_support():
    image = np.full((9, 13, 13), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 4:9, 4:9] = True
    image[aorta] = 70.0
    image[:, 5:8, 5:8] = 350.0

    fat = np.zeros_like(aorta)
    fat[:, 4:9, 10] = True
    image[fat] = -80.0
    image[:, 4:9, 9] = 60.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=fat,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=150.0,
        wall_hu_min=-30.0,
    )

    assert result.contrast_lumen_mask[:, 5:8, 5:8].all()
    assert not result.wall_candidate_mask[:, 6, 6].any()
    assert result.wall_candidate_mask[:, 4, 4].any()
    assert result.wall_candidate_mask[:, 6, 9].any()
    assert not result.wall_candidate_mask[fat].any()
    assert set(np.unique(result.labelmap)) >= {0, 1, 2, 3}
    assert not result.features.empty


def test_fat_closed_wall_handles_empty_fat_support():
    image = np.full((7, 9, 9), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 2:7, 2:7] = True
    image[aorta] = 60.0
    image[:, 3:6, 3:6] = 320.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        min_lumen_hu=150.0,
    )

    assert result.fat_support_mask.sum() == 0
    assert result.closed_outer_envelope_mask.any()
    assert result.wall_candidate_mask.any()
    assert not (result.wall_candidate_mask & result.contrast_lumen_mask).any()


def test_fat_closed_wall_can_hu_correct_lumen_outside_input_aorta():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 70.0
    image[:, 6:9, 6:9] = 360.0
    image[:, 6:9, 9:11] = 360.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=150.0,
        lumen_correction_enabled=True,
        lumen_correction_outer_mm=2.0,
        lumen_correction_close_radius_mm=0.0,
    )

    assert result.contrast_lumen_mask[:, 6:9, 10].all()
    assert result.hu_refined_aorta_mask[:, 6:9, 10].all()
    assert not result.wall_candidate_mask[:, 6:9, 10].any()
    added = result.features.loc[
        result.features["feature_name"] == "lumen_added_outside_input_aorta_volume_mm3",
        "feature_value",
    ].iloc[0]
    assert added > 0


def test_fat_closed_wall_hu_correction_can_use_stricter_floor():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 70.0
    image[:, 6:9, 6:9] = 360.0
    image[:, 6:9, 9] = 360.0
    image[:, 6:9, 10] = 220.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=150.0,
        lumen_correction_enabled=True,
        lumen_correction_outer_mm=2.0,
        lumen_correction_close_radius_mm=0.0,
        lumen_correction_lower_margin_hu=60.0,
        lumen_correction_min_hu=300.0,
    )

    assert result.contrast_lumen_mask[:, 6:9, 9].all()
    assert not result.contrast_lumen_mask[:, 6:9, 10].any()


def test_fat_closed_wall_strict_base_lumen_excludes_low_wall_hu():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 260.0
    image[:, 6:9, 6:9] = 410.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=60.0,
        min_lumen_hu=300.0,
    )

    assert result.contrast_lumen_mask[:, 6:9, 6:9].all()
    assert not result.contrast_lumen_mask[:, 5, 5].any()


def test_reference_percent_lumen_window_uses_local_median_band():
    lower, upper = _reference_hu_bounds(
        np.asarray([400.0, 420.0]),
        lower_margin_hu=200.0,
        min_hu=150.0,
        max_above_reference_hu=None,
        reference_lower_fraction=0.08,
        reference_upper_fraction=0.12,
    )

    np.testing.assert_allclose(lower, [368.0, 386.4])
    np.testing.assert_allclose(upper, [448.0, 470.4])


def test_lumen_reference_can_use_mean_for_strict_expansion_gate():
    image = np.zeros((1, 3, 3), dtype=float)
    core = np.ones_like(image, dtype=bool)
    core_values = np.asarray([300.0, 380.0, 400.0, 420.0, 420.0, 420.0, 430.0, 440.0, 700.0])
    image[core] = core_values

    reference = _slice_lumen_reference_hu(
        image,
        core,
        core,
        min_voxels_per_slice=1,
        statistic="mean",
    )
    lower, upper = _reference_hu_bounds(
        reference,
        lower_margin_hu=200.0,
        min_hu=375.0,
        max_above_reference_hu=None,
        reference_lower_fraction=0.0,
        reference_upper_fraction=0.20,
    )

    np.testing.assert_allclose(reference, [434.4444444444])
    np.testing.assert_allclose(lower, reference)
    np.testing.assert_allclose(upper, reference * 1.2)


def test_fat_closed_wall_can_require_lumen_seed_connectivity():
    image = np.full((7, 21, 21), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 4:17, 4:17] = True
    image[aorta] = 70.0
    image[:, 8:12, 8:12] = 410.0
    image[:, 4:7, 14:17] = 410.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=300.0,
        max_lumen_hu_above_reference=None,
        require_lumen_seed_connectivity=True,
    )

    assert result.contrast_lumen_mask[:, 8:12, 8:12].all()
    assert not result.contrast_lumen_mask[:, 4:7, 14:17].any()


def test_fat_closed_wall_can_use_input_aorta_as_lumen_floor_and_add_contrast():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 70.0
    image[:, 6:9, 6:9] = 410.0
    image[:, 6:9, 9] = 620.0
    image[:, 6:9, 10] = 410.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=300.0,
        max_lumen_hu_above_reference=None,
        exclude_calcification_hu=500.0,
        include_calcification_in_wall=True,
        use_input_aorta_as_lumen_floor=True,
        lumen_correction_enabled=True,
        lumen_correction_outer_mm=2.0,
        lumen_correction_close_radius_mm=0.0,
    )

    assert result.contrast_lumen_mask[:, 5, 5].all()
    assert not result.contrast_lumen_mask[:, 6:9, 9].any()
    assert result.wall_candidate_mask[:, 6:9, 9].all()
    assert result.contrast_lumen_mask[:, 6:9, 10].all()


def test_fat_closed_wall_excludes_calcium_range_hu_from_lumen_but_keeps_wall():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 70.0
    image[:, 6:9, 6:9] = 360.0
    image[:, 6:9, 9] = 620.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=150.0,
        exclude_calcification_hu=500.0,
        include_calcification_in_wall=True,
        lumen_correction_enabled=True,
        lumen_correction_outer_mm=2.0,
        lumen_correction_close_radius_mm=0.0,
        lumen_correction_lower_margin_hu=60.0,
        lumen_correction_min_hu=300.0,
    )

    assert result.contrast_lumen_mask[:, 6:9, 6:9].all()
    assert not result.contrast_lumen_mask[:, 6:9, 9].any()
    assert result.wall_candidate_mask[:, 6:9, 9].all()
    assert result.hu_refined_aorta_mask[:, 6:9, 9].all()


def test_fat_closed_wall_can_exclude_calcium_range_hu_from_wall_when_requested():
    image = np.full((7, 15, 15), -1000.0, dtype=float)
    aorta = np.zeros_like(image, dtype=bool)
    aorta[:, 5:10, 5:10] = True
    image[aorta] = 70.0
    image[:, 6:9, 6:9] = 360.0
    image[:, 6:9, 9] = 620.0

    result = extract_fat_closed_aortic_wall(
        image=image,
        aorta_mask=aorta,
        fat_mask=np.zeros_like(aorta),
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        outer_limit_mm=5.0,
        close_radius_mm=3.0,
        lumen_core_distance_mm=1.0,
        contrast_lower_margin_hu=100.0,
        min_lumen_hu=150.0,
        exclude_calcification_hu=500.0,
        include_calcification_in_wall=False,
        lumen_correction_enabled=True,
        lumen_correction_outer_mm=2.0,
        lumen_correction_close_radius_mm=0.0,
        lumen_correction_lower_margin_hu=60.0,
        lumen_correction_min_hu=300.0,
    )

    assert not result.contrast_lumen_mask[:, 6:9, 9].any()
    assert not result.wall_candidate_mask[:, 6:9, 9].any()
    assert not result.hu_refined_aorta_mask[:, 6:9, 9].any()
