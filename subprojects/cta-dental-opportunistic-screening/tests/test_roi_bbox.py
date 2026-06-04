"""Unit tests for ROI bbox expansion and crop/uncrop transforms."""

import numpy as np
import pytest
import SimpleITK as sitk

from cta_dental.geometry import BoundingBox, crop_array


def _make_sitk_image(shape_ijk, spacing=(1.0, 1.0, 1.0), value=0.0):
    """Create a SimpleITK image filled with *value*."""
    arr = np.full(shape_ijk, value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(list(reversed(spacing)))  # SimpleITK: (x, y, z)
    return img


class TestROIBBoxExpansion:

    def test_expand_20mm_on_1mm_isotropic(self):
        bbox = BoundingBox(np.array([30, 30, 30]), np.array([60, 60, 60]))
        expanded = bbox.expand_mm(20.0, [1.0, 1.0, 1.0])
        assert list(expanded.min_ijk) == [10, 10, 10]
        assert list(expanded.max_ijk) == [80, 80, 80]

    def test_expand_20mm_on_half_mm_isotropic(self):
        # 20mm / 0.5mm = 40 voxels
        bbox = BoundingBox(np.array([50, 50, 50]), np.array([100, 100, 100]))
        expanded = bbox.expand_mm(20.0, [0.5, 0.5, 0.5])
        assert expanded.min_ijk[0] == 10
        assert expanded.max_ijk[0] == 140

    def test_expand_clips_at_zero(self):
        bbox = BoundingBox(np.array([5, 5, 5]), np.array([10, 10, 10]))
        expanded = bbox.expand_mm(10.0, [1.0, 1.0, 1.0])
        assert all(expanded.min_ijk >= 0), "min_ijk must be >= 0 after expansion"

    def test_expand_clips_to_shape(self):
        shape = [50, 50, 50]
        bbox = BoundingBox(np.array([40, 40, 40]), np.array([48, 48, 48]))
        expanded = bbox.expand_mm(20.0, [1.0, 1.0, 1.0]).clip_to_shape(shape)
        assert all(expanded.max_ijk < np.array(shape))

    def test_bbox_from_mask_then_expand_and_crop(self):
        shape = (100, 100, 100)
        mask = np.zeros(shape, dtype=bool)
        mask[40:60, 40:60, 40:60] = True
        arr = np.random.rand(*shape).astype(np.float32)
        spacing = [0.5, 0.5, 0.5]

        bbox = BoundingBox.from_mask(mask)
        expanded = bbox.expand_mm(10.0, spacing).clip_to_shape(shape)
        cropped = crop_array(arr, expanded)

        # 20 vox core + 20 vox margin on each side (clipped to 100)
        expected_size = 20 + 20 + 20  # margin=10mm/0.5mm=20vox each side
        assert cropped.shape[0] == min(expected_size, 100)

    def test_physical_coords_match_spacing(self):
        bbox = BoundingBox(np.array([10, 20, 30]), np.array([40, 50, 60]))
        phys = bbox.to_physical(origin=[0.0, 0.0, 0.0], spacing=[0.5, 0.5, 0.5])
        assert phys["min_physical_mm"] == pytest.approx([5.0, 10.0, 15.0])
        assert phys["max_physical_mm"] == pytest.approx([20.0, 25.0, 30.0])


class TestCropUncrop:

    def test_crop_recovers_values(self):
        arr = np.arange(8000, dtype=np.float32).reshape(20, 20, 20)
        bbox = BoundingBox(np.array([5, 5, 5]), np.array([14, 14, 14]))
        cropped = crop_array(arr, bbox)
        assert cropped[0, 0, 0] == arr[5, 5, 5]
        assert cropped[-1, -1, -1] == arr[14, 14, 14]

    def test_uncrop_via_replace(self):
        arr = np.zeros((20, 20, 20), dtype=np.float32)
        bbox = BoundingBox(np.array([5, 5, 5]), np.array([9, 9, 9]))
        patch = np.ones((5, 5, 5), dtype=np.float32)
        slices = bbox.to_slices()
        arr[slices] = patch
        assert arr[5, 5, 5] == 1.0
        assert arr[0, 0, 0] == 0.0

    def test_crop_shape_matches_bbox_shape(self):
        arr = np.zeros((50, 50, 50))
        bbox = BoundingBox(np.array([10, 10, 10]), np.array([29, 29, 29]))
        cropped = crop_array(arr, bbox)
        assert list(cropped.shape) == [20, 20, 20]
