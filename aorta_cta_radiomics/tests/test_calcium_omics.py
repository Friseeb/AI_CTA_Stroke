import numpy as np
import pandas as pd

from aorta_cta_radiomics.calcium_omics import summarize_calcium_omics


def _feature_value(frame: pd.DataFrame, feature_name: str, region: str = "aorta") -> float:
    value = frame.loc[
        (frame["region"] == region) & (frame["feature_name"] == feature_name),
        "feature_value",
    ].iloc[0]
    return float(value)


def test_calcium_omics_reports_mass_lesions_dense_fraction_and_segment_burden():
    image = np.zeros((6, 6, 6), dtype=float)
    aorta_mask = np.ones_like(image, dtype=bool)
    calcium_mask = np.zeros_like(image, dtype=bool)
    calcium_mask[1, 1, 1] = True
    calcium_mask[1, 1, 2] = True
    calcium_mask[4, 4, 4] = True
    image[1, 1, 1] = 600.0
    image[1, 1, 2] = 1100.0
    image[4, 4, 4] = 1200.0

    segment_labels = np.zeros_like(image, dtype=np.uint8)
    segment_labels[:3] = 1
    segment_labels[3:] = 2
    centerline = pd.DataFrame(
        {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 30.0],
        }
    )

    frame = summarize_calcium_omics(
        image=image,
        calcium_mask=calcium_mask,
        aorta_mask=aorta_mask,
        spacing_xyz=(1.0, 1.0, 2.0),
        case_id="CASE",
        mask_name="aorta_wall_dynamic_seed500HU",
        threshold_label="dynamic_lumen_referenced_seed500HU",
        centerline_points=centerline,
        segment_labels=segment_labels,
        segment_names={1: "ascending", 2: "descending"},
    )

    assert _feature_value(frame, "num_lesions") == 2.0
    assert _feature_value(frame, "log1p_num_lesions") == np.log1p(2)
    assert _feature_value(frame, "mass_total") == 5800.0
    assert _feature_value(frame, "aortic_mass_proxy") == 5800.0
    assert _feature_value(frame, "aortic_volume_mm3") == 6.0
    assert _feature_value(frame, "hu_gt_1000_volume") == 4.0
    assert _feature_value(frame, "hu_gt_1000_fraction") == 4.0 / 6.0
    assert _feature_value(frame, "top_bottom_distance_mm") == 8.0
    assert _feature_value(frame, "aortic_length_cm") == 3.0
    assert _feature_value(frame, "calcium_per_cm") == 2.0
    assert _feature_value(frame, "diffusivity") == 2.0 / 3.0
    assert _feature_value(frame, "num_territories_involved") == 2.0
    assert _feature_value(frame, "calcium_by_segment", "aorta_segment:ascending") == 4.0
    assert _feature_value(frame, "calcium_by_segment", "aorta_segment:descending") == 2.0
    assert _feature_value(frame, "mass_by_territory", "aorta_segment:ascending") == 3400.0
    assert _feature_value(frame, "mass_by_territory", "aorta_segment:descending") == 2400.0
