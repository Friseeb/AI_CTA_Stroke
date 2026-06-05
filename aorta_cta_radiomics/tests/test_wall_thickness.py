import numpy as np

from aorta_cta_radiomics.wall_thickness import (
    measure_wall_thickness,
    thickness_bins,
    thickness_threshold_summary,
    wall_thickness_threshold_mask,
)


def test_wall_thickness_recovers_simple_slab_thickness():
    lumen = np.zeros((7, 5, 5), dtype=bool)
    wall = np.zeros_like(lumen)
    lumen[:2, :, :] = True
    wall[2:5, :, :] = True

    result = measure_wall_thickness(lumen, wall, (1.0, 1.0, 1.0), case_id="slab")

    assert result.inner_surface_mask[2, 2, 2]
    assert result.outer_surface_mask[4, 2, 2]
    assert np.isclose(result.thickness_map_mm[3, 2, 2], 3.0)
    assert np.isclose(np.median(result.thickness_map_mm[result.wall_mask]), 3.0)


def test_thickness_bins_are_reproducible():
    values = np.array([[[1.5, 2.5, 3.5, 4.5, 5.5, 0.0]]], dtype=float)
    wall = values > 0

    labels = thickness_bins(values, wall)

    assert labels.tolist() == [[[1, 2, 3, 4, 5, 0]]]


def test_wall_thickness_threshold_mask_uses_strict_greater_than_by_default():
    values = np.array([[[3.9, 4.0, 4.1, 5.0]]], dtype=float)
    wall = np.ones_like(values, dtype=bool)

    strict = wall_thickness_threshold_mask(values, wall, threshold_mm=4.0)
    inclusive = wall_thickness_threshold_mask(values, wall, threshold_mm=4.0, inclusive=True)

    assert strict.tolist() == [[[False, False, True, True]]]
    assert inclusive.tolist() == [[[False, True, True, True]]]


def test_thickness_threshold_summary_reports_fraction_and_volume():
    mask = np.array([[[True, False, True, False]]])
    wall = np.ones_like(mask, dtype=bool)

    summary = thickness_threshold_summary("case", mask, wall, (2.0, 1.0, 1.0), threshold_mm=4.0)

    values = dict(zip(summary["feature_name"], summary["feature_value"]))
    assert values["wall_thickness_gt4mm_voxel_count"] == 2
    assert values["wall_thickness_gt4mm_volume_mm3"] == 4.0
    assert values["wall_thickness_gt4mm_wall_fraction"] == 0.5


def test_empty_wall_returns_empty_maps_and_summary():
    lumen = np.zeros((5, 5, 5), dtype=bool)
    wall = np.zeros_like(lumen)
    lumen[2, 2, 2] = True

    result = measure_wall_thickness(lumen, wall, (1.0, 1.0, 1.0), case_id="empty")

    assert not result.thickness_map_mm.any()
    assert int(result.summary.loc[result.summary["feature_name"] == "wall_voxel_count", "feature_value"].iloc[0]) == 0
