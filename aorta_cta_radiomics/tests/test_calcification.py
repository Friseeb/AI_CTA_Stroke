import numpy as np

from aorta_cta_radiomics.calcification import (
    extract_calcification_masks,
    extract_dynamic_wall_calcification,
    summarize_calcification,
)
from aorta_cta_radiomics.shells import create_aorta_wall_band_masks


def test_calcification_volume_uses_voxel_volume():
    image = np.zeros((4, 4, 4), dtype=float)
    roi = np.zeros_like(image, dtype=bool)
    roi[1:3, 1:3, 1:3] = True
    image[1, 1, 1] = 350
    image[1, 1, 2] = 500
    image[2, 2, 2] = 301

    masks = extract_calcification_masks(image, roi, thresholds_hu=[300])
    frame = summarize_calcification(image, masks, spacing_xyz=(1.0, 2.0, 3.0), case_id="CASE", region="aorta", mask_name="aorta")

    volume = frame.loc[frame["feature_name"] == "calcium_volume", "feature_value"].iloc[0]
    voxel_count = frame.loc[frame["feature_name"] == "calcium_voxel_count", "feature_value"].iloc[0]
    assert voxel_count == 3
    assert volume == 18.0


def test_empty_calcification_mask_has_zero_volume_and_no_crash():
    image = np.zeros((3, 3, 3), dtype=float)
    roi = np.ones_like(image, dtype=bool)

    masks = extract_calcification_masks(image, roi, thresholds_hu=[300])
    frame = summarize_calcification(image, masks, spacing_xyz=(1.0, 1.0, 1.0), case_id="CASE", region="aorta", mask_name="aorta")

    volume = frame.loc[frame["feature_name"] == "calcium_volume", "feature_value"].iloc[0]
    max_hu = frame.loc[frame["feature_name"] == "calcium_max_hu", "feature_value"].iloc[0]
    assert volume == 0.0
    assert np.isnan(max_hu)


def test_wall_band_calcification_excludes_core_lumen_signal():
    image = np.zeros((11, 11, 11), dtype=float)
    aorta_mask = np.zeros_like(image, dtype=bool)
    aorta_mask[3:8, 3:8, 3:8] = True
    image[5, 5, 5] = 900  # central contrast/core signal should not count as wall calcium
    image[3, 5, 5] = 700  # inner boundary wall band
    image[2, 5, 5] = 800  # external peri-wall band

    wall_band = create_aorta_wall_band_masks(
        aorta_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        internal_mm=1.0,
        external_mm=1.0,
    )["aorta_wall_band"]
    masks = extract_calcification_masks(image, wall_band, thresholds_hu=[500])

    assert masks[500][3, 5, 5]
    assert masks[500][2, 5, 5]
    assert not masks[500][5, 5, 5]
    assert int(masks[500].sum()) == 2


def test_dynamic_wall_calcification_grows_connected_intimal_tail_only():
    image = np.zeros((11, 11, 11), dtype=float)
    aorta_mask = np.zeros_like(image, dtype=bool)
    aorta_mask[3:8, 3:8, 3:8] = True
    image[aorta_mask] = 250
    image[2, 5, 5] = 650  # high-confidence wall-adjacent calcium seed
    image[3, 5, 5] = 390  # connected intimal-side tail below the fixed 500 HU threshold
    image[3, 7, 7] = 390  # candidate intensity, but disconnected from a 500 HU seed

    result = extract_dynamic_wall_calcification(
        image=image,
        aorta_mask=aorta_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        seed_threshold_hu=500,
        lumen_margin_hu=75,
        min_candidate_hu=300,
        lumen_core_distance_mm=2,
        search_internal_mm=1,
        search_external_mm=1,
        smooth_lumen_profile_mm=0,
    )

    assert result.high_confidence_seed_mask[2, 5, 5]
    assert result.mask[2, 5, 5]
    assert result.mask[3, 5, 5]
    assert not result.mask[3, 7, 7]
    assert float(np.nanmedian(result.dynamic_threshold_hu_by_slice)) == 325.0


def test_dynamic_wall_calcification_rejects_external_artery_contrast_touching_component():
    image = np.zeros((11, 11, 11), dtype=float)
    aorta_mask = np.zeros_like(image, dtype=bool)
    aorta_mask[3:8, 3:8, 3:8] = True
    image[aorta_mask] = 250

    # Aortic calcium seed with an intimal tail: this intersects the aorta mask and is kept.
    image[2, 5, 5] = 650
    image[3, 5, 5] = 390

    # Neighboring non-aortic artery: high-HU focus touches outside contrast-like lumen.
    image[2, 2, 2] = 650
    image[2, 2, 3] = 250

    result = extract_dynamic_wall_calcification(
        image=image,
        aorta_mask=aorta_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        seed_threshold_hu=500,
        lumen_margin_hu=75,
        min_candidate_hu=300,
        lumen_core_distance_mm=2,
        search_internal_mm=1,
        search_external_mm=2,
        smooth_lumen_profile_mm=0,
        exclude_external_contrast_touching=True,
        external_contrast_tolerance_hu=25,
    )

    assert result.mask[2, 5, 5]
    assert result.mask[3, 5, 5]
    assert result.external_contrast_rejected_mask[2, 2, 2]
    assert not result.mask[2, 2, 2]
