"""Unit tests for geometry utilities."""

import numpy as np
import pytest

from cta_dental.geometry import BoundingBox, crop_array, voxel_volume_mm3


def _make_mask(shape, nonzero_slices):
    arr = np.zeros(shape, dtype=bool)
    arr[nonzero_slices] = True
    return arr


class TestBoundingBox:

    def test_from_mask_basic(self):
        mask = _make_mask((10, 10, 10), (slice(2, 5), slice(3, 6), slice(1, 4)))
        bbox = BoundingBox.from_mask(mask)
        assert list(bbox.min_ijk) == [2, 3, 1]
        assert list(bbox.max_ijk) == [4, 5, 3]

    def test_from_empty_mask_raises(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        with pytest.raises(ValueError, match="empty"):
            BoundingBox.from_mask(mask)

    def test_expand_voxels(self):
        bbox = BoundingBox(np.array([5, 5, 5]), np.array([8, 8, 8]))
        expanded = bbox.expand_voxels([2, 2, 2])
        assert list(expanded.min_ijk) == [3, 3, 3]
        assert list(expanded.max_ijk) == [10, 10, 10]

    def test_expand_voxels_clamps_to_zero(self):
        bbox = BoundingBox(np.array([1, 1, 1]), np.array([5, 5, 5]))
        expanded = bbox.expand_voxels([5, 5, 5])
        assert all(expanded.min_ijk >= 0)

    def test_expand_mm(self):
        bbox = BoundingBox(np.array([10, 10, 10]), np.array([20, 20, 20]))
        expanded = bbox.expand_mm(10.0, [1.0, 1.0, 1.0])
        assert list(expanded.min_ijk) == [0, 0, 0]
        assert list(expanded.max_ijk) == [30, 30, 30]

    def test_expand_mm_non_isotropic(self):
        bbox = BoundingBox(np.array([20, 20, 20]), np.array([30, 30, 30]))
        expanded = bbox.expand_mm(10.0, [2.0, 1.0, 0.5])
        # 10mm / 2mm = 5 vox in axis 0, 10 in axis 1, 20 in axis 2
        assert expanded.min_ijk[0] == 15
        assert expanded.min_ijk[1] == 10
        assert expanded.min_ijk[2] == 0

    def test_clip_to_shape(self):
        bbox = BoundingBox(np.array([-2, -3, 0]), np.array([100, 200, 50]))
        clipped = bbox.clip_to_shape([50, 50, 50])
        assert list(clipped.min_ijk) == [0, 0, 0]
        assert list(clipped.max_ijk) == [49, 49, 49]

    def test_to_slices(self):
        bbox = BoundingBox(np.array([2, 3, 4]), np.array([5, 6, 7]))
        s = bbox.to_slices()
        assert s == (slice(2, 6), slice(3, 7), slice(4, 8))

    def test_shape(self):
        bbox = BoundingBox(np.array([0, 0, 0]), np.array([9, 9, 9]))
        assert list(bbox.shape()) == [10, 10, 10]

    def test_to_dict(self):
        bbox = BoundingBox(np.array([1, 2, 3]), np.array([4, 5, 6]))
        d = bbox.to_dict()
        assert d["min_ijk"] == [1, 2, 3]
        assert d["max_ijk"] == [4, 5, 6]
        assert d["shape_ijk"] == [4, 4, 4]

    def test_to_physical(self):
        bbox = BoundingBox(np.array([0, 0, 0]), np.array([10, 10, 10]))
        phys = bbox.to_physical(origin=[0.0, 0.0, 0.0], spacing=[0.5, 0.5, 0.5])
        assert phys["min_physical_mm"] == [0.0, 0.0, 0.0]
        assert phys["max_physical_mm"] == [5.0, 5.0, 5.0]
        assert phys["size_mm"] == [5.0, 5.0, 5.0]


class TestCropArray:

    def test_crop_basic(self):
        arr = np.arange(1000).reshape(10, 10, 10)
        bbox = BoundingBox(np.array([2, 3, 4]), np.array([4, 5, 6]))
        cropped = crop_array(arr, bbox)
        assert cropped.shape == (3, 3, 3)
        assert cropped[0, 0, 0] == arr[2, 3, 4]

    def test_crop_full(self):
        arr = np.ones((5, 5, 5))
        bbox = BoundingBox(np.array([0, 0, 0]), np.array([4, 4, 4]))
        cropped = crop_array(arr, bbox)
        assert cropped.shape == (5, 5, 5)


class TestVoxelVolume:

    def test_isotropic(self):
        assert voxel_volume_mm3([0.5, 0.5, 0.5]) == pytest.approx(0.125)

    def test_anisotropic(self):
        assert voxel_volume_mm3([1.0, 0.5, 2.0]) == pytest.approx(1.0)
