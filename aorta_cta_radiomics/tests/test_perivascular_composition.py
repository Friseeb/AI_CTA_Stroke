import numpy as np

from aorta_cta_radiomics.perivascular.composition import (
    TissueLabel,
    TissueThresholds,
    classify_tissue_composition,
)


def test_exact_50_25_25_mutually_exclusive_composition():
    shape = (4, 4, 4)
    roi = np.ones(shape, dtype=bool)
    image = np.full(shape, 40.0)
    image[:, :, :2] = -100.0  # 32/64 fat
    muscle = np.zeros(shape, dtype=bool)
    muscle[:, :2, 2:] = True  # 16/64 muscle

    result = classify_tissue_composition(image, roi, (1, 1, 1), muscle_mask=muscle)

    assert result.fractions["adipose"] == 0.5
    assert result.fractions["skeletal_muscle"] == 0.25
    assert result.fractions["other_soft_tissue"] == 0.25
    assert result.counts["valid_denominator"] == 64
    assert np.isclose(
        sum(value for key, value in result.fractions.items() if key != "valid_denominator"), 1.0
    )
    assert set(np.unique(result.labels)) == {
        int(TissueLabel.ADIPOSE),
        int(TissueLabel.SKELETAL_MUSCLE),
        int(TissueLabel.OTHER_SOFT_TISSUE),
    }


def test_overlap_priority_keeps_vein_and_bone_out_of_fat_and_residual():
    shape = (5, 5, 5)
    roi = np.ones(shape, dtype=bool)
    image = np.full(shape, -100.0)
    vein = np.zeros(shape, dtype=bool)
    bone = np.zeros(shape, dtype=bool)
    vein[:, :, 0] = True
    bone[:, :, 1] = True
    # Deliberately overlap one voxel: vein has the documented higher priority.
    bone[2, 2, 0] = True

    result = classify_tissue_composition(
        image,
        roi,
        (1, 1, 1),
        vein_mask=vein,
        bone_mask=bone,
    )

    assert result.labels[2, 2, 0] == int(TissueLabel.VEIN)
    assert result.input_overlap_voxels == 50  # union of anatomy voxels that also overlap HU fat
    assert not np.any(result.fat_mask & vein)
    assert not np.any(result.fat_mask & bone)
    assert result.counts["vein"] == 25
    assert result.counts["bone"] == 25
    assert result.counts["adipose"] == 75
    assert result.counts["other_soft_tissue"] == 0


def test_configurable_fat_windows_high_density_and_uncertain_denominator():
    image = np.array([[[-180.0, -120.0, -25.0, 500.0, np.nan]]])
    roi = np.ones_like(image, dtype=bool)
    thresholds = TissueThresholds(
        primary_fat_hu=(-190, -30),
        narrow_fat_hu=(-150, -50),
        wide_fat_hu=(-200, -20),
        high_density_min_hu=300,
    )
    result = classify_tissue_composition(image, roi, (1, 1, 1), thresholds=thresholds)

    assert result.labels.tolist() == [[[
        int(TissueLabel.ADIPOSE),
        int(TissueLabel.ADIPOSE),
        int(TissueLabel.OTHER_SOFT_TISSUE),
        int(TissueLabel.HIGH_DENSITY),
        int(TissueLabel.UNCERTAIN),
    ]]]
    assert int(result.fat_sensitivity_masks["narrow"].sum()) == 1
    assert int(result.fat_sensitivity_masks["wide"].sum()) == 3
    assert result.counts["valid_denominator"] == 4
    assert result.acquisition_metadata["metadata_missing"] is True
