"""Tests for cta_common.geometry (the consolidated bbox + scalar helpers)."""

import numpy as np
import pytest

from cta_common.geometry import (
    BoundingBox,
    crop_array,
    mm_to_voxels,
    pad_to_multiple,
    slice_area_mm2,
    slice_thickness_mm,
    voxel_volume_mm3,
    z_index_to_mm,
)


def _mask(shape, slices):
    a = np.zeros(shape, dtype=bool)
    a[slices] = True
    return a


def test_from_mask_and_both_alias_spellings():
    bbox = BoundingBox.from_mask(_mask((10, 10, 10), (slice(2, 5), slice(3, 6), slice(1, 4))))
    # canonical
    assert list(bbox.min_idx) == [2, 3, 1]
    assert list(bbox.max_idx) == [4, 5, 3]
    # dental alias
    assert list(bbox.min_ijk) == [2, 3, 1]
    assert list(bbox.max_ijk) == [4, 5, 3]
    # stroke alias
    assert list(bbox.min_zyx) == [2, 3, 1]
    assert list(bbox.max_zyx) == [4, 5, 3]


def test_empty_mask_raises_with_empty_in_message():
    with pytest.raises(ValueError, match="empty"):
        BoundingBox.from_mask(np.zeros((4, 4, 4), dtype=bool))


def test_expand_voxels_clamps_min_to_zero():
    bbox = BoundingBox(np.array([1, 1, 1]), np.array([5, 5, 5])).expand_voxels([5, 5, 5])
    assert all(bbox.min_idx >= 0)
    assert list(bbox.max_idx) == [10, 10, 10]


def test_expand_mm_non_isotropic_uses_ceil():
    bbox = BoundingBox(np.array([20, 20, 20]), np.array([30, 30, 30])).expand_mm(10.0, [2.0, 1.0, 0.5])
    assert bbox.min_idx[0] == 15 and bbox.min_idx[1] == 10 and bbox.min_idx[2] == 0


def test_clip_to_shape():
    bbox = BoundingBox(np.array([-2, -3, 0]), np.array([100, 200, 50])).clip_to_shape([50, 50, 50])
    assert list(bbox.min_idx) == [0, 0, 0]
    assert list(bbox.max_idx) == [49, 49, 49]


def test_slices_aliases_match():
    bbox = BoundingBox(np.array([2, 3, 4]), np.array([5, 6, 7]))
    assert bbox.to_slices() == (slice(2, 6), slice(3, 7), slice(4, 8))
    assert bbox.slices() == bbox.to_slices()  # stroke alias


def test_shape_and_to_dict_and_physical():
    bbox = BoundingBox(np.array([0, 0, 0]), np.array([10, 10, 10]))
    assert list(bbox.shape()) == [11, 11, 11]
    d = bbox.to_dict()
    assert d["min_ijk"] == [0, 0, 0] and d["shape_ijk"] == [11, 11, 11]
    phys = bbox.to_physical(origin=[0.0, 0.0, 0.0], spacing=[0.5, 0.5, 0.5])
    assert phys["max_physical_mm"] == [5.0, 5.0, 5.0]
    assert phys["size_mm"] == [5.0, 5.0, 5.0]


def test_crop_array():
    arr = np.arange(1000).reshape(10, 10, 10)
    cropped = crop_array(arr, BoundingBox(np.array([2, 3, 4]), np.array([4, 5, 6])))
    assert cropped.shape == (3, 3, 3)
    assert cropped[0, 0, 0] == arr[2, 3, 4]


def test_pad_to_multiple():
    padded, pads = pad_to_multiple(np.ones((5, 7, 16)), multiple=16)
    assert padded.shape == (16, 16, 16)
    assert pads[2] == (0, 0)


def test_scalar_helpers():
    assert voxel_volume_mm3([1.0, 0.5, 2.0]) == pytest.approx(1.0)
    assert slice_area_mm2([0.4, 0.5, 3.0]) == pytest.approx(0.2)  # x*y
    assert slice_thickness_mm([0.4, 0.5, 3.0]) == pytest.approx(3.0)
    assert mm_to_voxels(2.0, 0.5) == 4
    assert mm_to_voxels(0.0, 0.5) == 1  # floored to at least 1
    assert z_index_to_mm(10, [0.0, 0.0, 100.0], [1.0, 1.0, 2.0]) == pytest.approx(120.0)


def test_compute_spacing_from_sitk_reverses():
    class _Img:
        def GetSpacing(self):
            return (0.4, 0.5, 3.0)  # ITK x, y, z

    from cta_common.geometry import compute_spacing_from_sitk

    assert compute_spacing_from_sitk(_Img()) == (3.0, 0.5, 0.4)  # numpy z, y, x
