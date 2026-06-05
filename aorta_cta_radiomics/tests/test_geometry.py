import numpy as np

from aorta_cta_radiomics.centerline import approximate_centerline_by_slices
from aorta_cta_radiomics.lumen_geometry import slice_geometry_features


def _cylinder_mask(shape=(6, 21, 21), radius=4):
    z, y, x = np.indices(shape)
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    return ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2


def test_approximate_centerline_returns_one_point_per_occupied_slice():
    mask = _cylinder_mask()
    frame = approximate_centerline_by_slices(mask, spacing_xyz=(1.0, 1.0, 2.0), case_id="CASE")

    assert len(frame) == mask.shape[0]
    assert set(["x", "y", "z", "tangent_x", "curvature"]).issubset(frame.columns)
    assert frame["centerline_method"].eq("slice_center_of_mass_approximate").all()


def test_slice_geometry_returns_descriptive_features_only():
    mask = _cylinder_mask()
    frame = slice_geometry_features(mask, spacing_xyz=(1.0, 1.0, 2.0), case_id="CASE")

    assert len(frame) == mask.shape[0]
    assert frame["geometry_method"].eq("axial_component_branch_fallback_not_orthogonal").all()
    assert "irregularity_score" not in frame.columns
    assert "high_irregularity_attention" not in frame.columns
    assert frame["geometry_interpretation"].eq(
        "descriptive_geometry_only_not_irregularity_or_plaque_classifier"
    ).all()


def test_slice_geometry_tracks_separate_aortic_components():
    mask = np.zeros((4, 32, 32), dtype=bool)
    z, y, x = np.indices(mask.shape)
    left = ((x - 10) ** 2 + (y - 16) ** 2) <= 4**2
    right = ((x - 22) ** 2 + (y - 16) ** 2) <= 4**2
    mask[left | right] = True

    frame = slice_geometry_features(mask, spacing_xyz=(1.0, 1.0, 1.0), case_id="CASE")

    assert len(frame) == 8
    assert frame["component_count_in_slice"].eq(2).all()
    assert frame["branch_id"].nunique() == 2
