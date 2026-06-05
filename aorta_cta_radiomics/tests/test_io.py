import numpy as np
import pytest

from aorta_cta_radiomics.io import load_image_and_mask


sitk = pytest.importorskip("SimpleITK")


def test_load_image_and_mask_same_space(tmp_path):
    image_array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask_array = np.zeros_like(image_array, dtype=np.uint8)
    mask_array[:, 1, 1:3] = 1

    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    image.SetSpacing((1.0, 1.5, 2.0))
    mask.CopyInformation(image)

    image_path = tmp_path / "cta.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))

    loaded_image, loaded_mask, was_resampled = load_image_and_mask(image_path, mask_path)

    assert loaded_image.array.shape == (2, 3, 4)
    assert loaded_mask.array.dtype == bool
    assert loaded_mask.array.sum() == 4
    assert not was_resampled
