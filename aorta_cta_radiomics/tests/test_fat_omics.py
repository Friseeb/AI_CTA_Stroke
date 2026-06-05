import numpy as np
import pandas as pd

from aorta_cta_radiomics.fat_omics import extract_periaortic_fat_omics


def _feature_value(frame: pd.DataFrame, feature_name: str, region: str = "periaortic_fat") -> float:
    value = frame.loc[
        (frame["region"] == region) & (frame["feature_name"] == feature_name),
        "feature_value",
    ].iloc[0]
    return float(value)


def test_periaortic_fat_omics_extracts_hu_bins_volume_and_texture_proxies():
    image = np.zeros((7, 7, 7), dtype=float)
    aorta_mask = np.zeros_like(image, dtype=bool)
    aorta_mask[3, 3, 3] = True

    fat_voxels = {
        (2, 3, 3): -100.0,
        (4, 3, 3): -60.0,
        (3, 2, 3): -40.0,
        (3, 2, 2): -45.0,
        (3, 4, 3): -50.0,
        (3, 3, 2): -20.0,
        (3, 3, 4): -200.0,
    }
    for index, value in fat_voxels.items():
        image[index] = value

    centerline = pd.DataFrame({"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 20.0]})
    segment_labels = np.zeros_like(image, dtype=np.uint8)
    segment_labels[aorta_mask] = 1

    result = extract_periaortic_fat_omics(
        image=image,
        aorta_mask=aorta_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        centerline_points=centerline,
        segment_labels=segment_labels,
        segment_names={1: "whole_aorta"},
        external_radius_mm=1.5,
        adipose_hu_min=-190,
        adipose_hu_max=-30,
        high_hu_bins={"m70_m30": (-70, -30), "m50_m30": (-50, -30)},
        radial_bins_mm=[(0, 1.5)],
        angle_bins=8,
        texture_levels=4,
    )

    frame = result.features
    assert int(result.fat_mask.sum()) == 5
    assert set(result.fat_layer_masks) == {"periaortic_fat_0_1p5mm"}
    assert int(result.fat_layer_masks["periaortic_fat_0_1p5mm"].sum()) == 5
    assert _feature_value(frame, "periaortic_fat_volume_mm3") == 5.0
    assert _feature_value(frame, "periaortic_fat_volume_per_cm") == 2.5
    assert _feature_value(frame, "periaortic_mean_HU") == -59.0
    assert _feature_value(frame, "periaortic_median_HU") == -50.0
    assert _feature_value(frame, "periaortic_high_HU_fraction_m70_m30") == 0.8
    assert _feature_value(frame, "periaortic_high_HU_fraction_m50_m30") == 0.6
    assert np.isfinite(_feature_value(frame, "periaortic_glcm_cluster_tendency"))
    assert np.isfinite(_feature_value(frame, "periaortic_glrlm_short_run_emphasis"))
    assert np.isfinite(_feature_value(frame, "periaortic_glszm_small_zone_emphasis"))
    assert _feature_value(frame, "periaortic_fat_volume_mm3", "aorta_segment:whole_aorta") == 5.0


def test_periaortic_fat_omics_creates_disjoint_three_layer_masks():
    image = np.zeros((9, 9, 9), dtype=float)
    aorta_mask = np.zeros_like(image, dtype=bool)
    aorta_mask[4, 4, 4] = True
    image[4, 4, 5] = -80.0  # 1 mm
    image[4, 4, 7] = -70.0  # 3 mm
    image[4, 4, 8] = -60.0  # 4 mm

    result = extract_periaortic_fat_omics(
        image=image,
        aorta_mask=aorta_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        case_id="CASE",
        external_radius_mm=4.0,
        radial_bins_mm=[(0, 2), (2, 4), (4, 6)],
        texture_levels=4,
    )

    assert int(result.fat_layer_masks["periaortic_fat_0_2mm"].sum()) == 1
    assert int(result.fat_layer_masks["periaortic_fat_2_4mm"].sum()) == 2
    assert int(result.fat_layer_masks["periaortic_fat_4_6mm"].sum()) == 0
    layered = np.zeros_like(result.fat_mask, dtype=int)
    for layer in result.fat_layer_masks.values():
        layered += layer.astype(int)
    assert int(layered.max()) == 1
    assert int(layered.sum()) == int(result.fat_mask.sum())
