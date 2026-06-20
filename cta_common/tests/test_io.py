"""Tests for cta_common.io (volume/mask I/O + geometry equality)."""

import numpy as np
import pytest

from cta_common.io import same_physical_space


class _FakeImage:
    """Duck-typed stand-in for a SimpleITK image (only geometry getters)."""

    def __init__(self, size, spacing, origin, direction):
        self._size, self._spacing, self._origin, self._direction = size, spacing, origin, direction

    def GetSize(self):
        return self._size

    def GetSpacing(self):
        return self._spacing

    def GetOrigin(self):
        return self._origin

    def GetDirection(self):
        return self._direction


_IDENT = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def _img(size=(10, 10, 10), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    return _FakeImage(size, spacing, origin, _IDENT)


def test_same_physical_space_identical():
    assert same_physical_space(_img(), _img()) is True


def test_same_physical_space_differs_on_size():
    assert same_physical_space(_img(size=(10, 10, 11)), _img()) is False


def test_same_physical_space_differs_on_spacing_and_origin():
    assert same_physical_space(_img(spacing=(1.0, 1.0, 2.0)), _img()) is False
    assert same_physical_space(_img(origin=(0.0, 0.0, 5.0)), _img()) is False


def test_same_physical_space_within_tolerance():
    # tiny perturbations under the default tolerances should still match
    assert same_physical_space(_img(spacing=(1.0, 1.0, 1.0 + 1e-7)), _img()) is True


def test_read_write_roundtrip(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from cta_common.io import read_volume, read_mask, write_mask_like

    arr = np.zeros((6, 5, 4), dtype=np.int16)
    arr[1:3, 1:3, 1:3] = 100
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.8))
    path = tmp_path / "vol.nii.gz"
    sitk.WriteImage(img, str(path))

    vol = read_volume(path)
    assert vol.array.shape == (6, 5, 4)
    assert vol.spacing_xyz == pytest.approx((0.5, 0.5, 0.8))  # NIfTI stores float32

    mask_path = write_mask_like(vol.array > 0, vol.image, tmp_path / "m.nii.gz")
    m = read_mask(mask_path)
    assert m.array.dtype == bool
    assert int(m.array.sum()) == int((arr > 0).sum())
    assert same_physical_space(m.image, vol.image)
