"""Tests for --skip-existing reuse (BaseSegmenter.load_existing) and the shared
label-array cache."""

import json

import numpy as np
import pytest
import SimpleITK as sitk

from cta_dental.segmenters.base import BaseSegmenter, SegmentationResult


class _StubSegmenter(BaseSegmenter):
    """Minimal concrete segmenter; run() is never called in these tests."""

    @property
    def name(self) -> str:
        return "stub"

    def check_available(self) -> bool:
        return True

    def run(self, input_nifti, output_dir, config) -> SegmentationResult:  # pragma: no cover
        raise AssertionError("run() must not be called when reusing existing outputs")

    def labels(self) -> dict[str, int]:
        return {"tooth_a": 1, "tooth_b": 2}


def _write_label(path, shape=(4, 4, 4)):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[1, 1, 1] = 1
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def test_load_existing_none_when_manifest_absent(tmp_path):
    assert _StubSegmenter().load_existing(tmp_path) is None


def test_load_existing_reconstructs_from_manifest(tmp_path):
    a, b = tmp_path / "tooth_a.nii.gz", tmp_path / "tooth_b.nii.gz"
    _write_label(a)
    _write_label(b)
    (tmp_path / "labels.json").write_text(json.dumps({"labels": {"tooth_a": str(a), "tooth_b": str(b)}}))

    result = _StubSegmenter().load_existing(tmp_path)
    assert result is not None and result.success
    assert set(result.label_files) == {"tooth_a", "tooth_b"}
    assert result.meta.get("reused_existing") is True


def test_load_existing_none_when_a_label_file_missing(tmp_path):
    a = tmp_path / "tooth_a.nii.gz"
    _write_label(a)
    # tooth_b referenced but never written -> incomplete -> None
    (tmp_path / "labels.json").write_text(
        json.dumps({"labels": {"tooth_a": str(a), "tooth_b": str(tmp_path / "tooth_b.nii.gz")}})
    )
    assert _StubSegmenter().load_existing(tmp_path) is None


def test_load_existing_none_on_corrupt_manifest(tmp_path):
    (tmp_path / "labels.json").write_text("{not json")
    assert _StubSegmenter().load_existing(tmp_path) is None


def test_label_cache_reads_once(tmp_path, monkeypatch):
    from cta_dental import imaging_cache

    p = tmp_path / "lab.nii.gz"
    _write_label(p)
    imaging_cache.clear_cache()

    calls = {"n": 0}
    real_read = sitk.ReadImage

    def _counting_read(path_str):
        calls["n"] += 1
        return real_read(path_str)

    monkeypatch.setattr(imaging_cache.sitk, "ReadImage", _counting_read)

    a1 = imaging_cache.label_array(p)
    a2 = imaging_cache.label_array(p)
    assert calls["n"] == 1  # second call served from cache
    assert np.array_equal(a1, a2)

    imaging_cache.clear_cache()
    imaging_cache.label_array(p)
    assert calls["n"] == 2  # cleared -> re-read
