import numpy as np

from aorta_cta_radiomics.wall_morphology import extract_wall_morphology


def _single_slice_disk(radius=18, shape=(1, 80, 80)):
    z, y, x = np.indices(shape)
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    return ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2


def test_wall_morphology_keeps_smooth_disk_below_candidate_threshold():
    mask = _single_slice_disk()

    result = extract_wall_morphology(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        min_slice_voxels=50,
        axial_step_mm=1.0,
        angular_bins=36,
        smoothing_bins=9,
        candidate_depth_mm=4.0,
    )

    assert not result.sector_features.empty
    assert result.sector_features["wall_morphology_candidate"].sum() == 0
    assert result.candidate_boundary_mask.sum() == 0
    assert result.inward_candidate_boundary_mask.sum() == 0
    assert result.outward_candidate_boundary_mask.sum() == 0
    assert result.boundary_direction_labelmap.max() == 0
    assert result.direction_labelmap.max() == 0
    assert result.focal_direction_labelmap.max() == 0
    assert result.parcel_features.empty
    assert result.parcel_labelmap.max() == 0
    assert {"component_malinowska", "component_circularity", "component_roughness_ratio"}.issubset(
        result.sector_features.columns
    )


def test_wall_morphology_flags_inward_protrusion_like_notch():
    mask = _single_slice_disk()
    _, y, x = np.indices(mask.shape)
    center_y = mask.shape[1] // 2
    center_x = mask.shape[2] // 2
    mask[(x > center_x + 10) & (np.abs(y - center_y) < 6)] = False

    result = extract_wall_morphology(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        min_slice_voxels=50,
        axial_step_mm=1.0,
        angular_bins=36,
        smoothing_bins=9,
        candidate_depth_mm=3.0,
    )

    assert result.sector_features["inward_protrusion_like_candidate"].any()
    assert result.candidate_neighborhood_mask.sum() >= result.candidate_boundary_mask.sum() > 0
    assert result.inward_candidate_boundary_mask.sum() > 0
    assert result.inward_candidate_neighborhood_mask.sum() >= result.inward_candidate_boundary_mask.sum()
    assert 0 < result.inward_focal_mask.sum() < result.inward_candidate_neighborhood_mask.sum()
    assert 1 in set(np.unique(result.boundary_direction_labelmap).tolist())
    assert 1 in set(np.unique(result.direction_labelmap).tolist())
    assert 1 in set(np.unique(result.focal_direction_labelmap).tolist())
    assert not result.parcel_features.empty
    assert result.inward_parcel_labelmap.max() > 0
    assert result.parcel_labelmap.max() > 0
    assert result.parcel_features["direction"].eq("inward").any()


def test_wall_morphology_flags_outward_crater_like_outpouching():
    mask = _single_slice_disk()
    _, y, x = np.indices(mask.shape)
    center_y = mask.shape[1] // 2
    center_x = mask.shape[2] // 2
    mask |= ((x - (center_x + 21)) ** 2 + (y - center_y) ** 2) <= 6**2

    result = extract_wall_morphology(
        mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        min_slice_voxels=50,
        axial_step_mm=1.0,
        angular_bins=36,
        smoothing_bins=9,
        candidate_depth_mm=3.0,
    )

    assert result.sector_features["outward_crater_like_candidate"].any()
    assert result.outward_candidate_boundary_mask.sum() > 0
    assert result.outward_candidate_neighborhood_mask.sum() >= result.outward_candidate_boundary_mask.sum()
    assert 0 < result.outward_focal_mask.sum() < result.outward_candidate_neighborhood_mask.sum()
    assert 2 in set(np.unique(result.boundary_direction_labelmap).tolist())
    assert 2 in set(np.unique(result.direction_labelmap).tolist())
    assert 2 in set(np.unique(result.focal_direction_labelmap).tolist())
    assert not result.parcel_features.empty
    assert result.outward_parcel_labelmap.max() > 0
    assert result.parcel_features["direction"].eq("outward").any()
    assert result.summary_features["feature_name"].str.contains("candidate_sector_count").any()
