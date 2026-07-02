import numpy as np

from aorta_cta_radiomics.crop import crop_region_for_mask


def test_crop_region_uses_physical_margin_and_pastes_back():
    mask = np.zeros((10, 12, 14), dtype=bool)
    mask[4:6, 5:7, 6:8] = True

    region = crop_region_for_mask(mask, spacing_xyz=(2.0, 1.0, 4.0), margin_mm=2.0)
    cropped = region.crop(mask)
    assert cropped.shape == (4, 6, 4)
    assert cropped.sum() == mask.sum()

    labels = np.ones(cropped.shape, dtype=np.uint16)
    pasted = region.paste(labels)
    assert pasted.shape == mask.shape
    assert pasted.dtype == np.uint16
    assert pasted[region.slices].all()
    assert pasted.sum() == labels.sum()


def test_empty_crop_region_returns_full_image():
    mask = np.zeros((3, 4, 5), dtype=bool)
    region = crop_region_for_mask(mask, spacing_xyz=(1.0, 1.0, 1.0), margin_mm=5.0)
    assert region.shape == mask.shape
    assert region.crop(mask).shape == mask.shape
