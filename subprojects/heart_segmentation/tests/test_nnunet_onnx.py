"""Unit tests of the nnU-Net ONNX runtime pieces, with SimpleITK as the oracle for reorientation and a
stub session standing in for the network so the whole predict() path runs without weights."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nnunet_onnx as no  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[3] / "external" / "holorama"


def _image(shape_zyx, spacing_zyx, code, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(-1000, 1500, size=shape_zyx, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    img.SetDirection(sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(code))
    img.SetOrigin((10.0, -20.0, 5.0))
    return img, arr


@pytest.mark.parametrize("source,target", [
    ("LAS", "LAS"), ("LAS", "RAS"), ("LAS", "LPS"), ("RAS", "LAS"), ("LPS", "LAS"), ("LAI", "LAS"),
    ("RPI", "LAS"), ("ASL", "LAS"), ("PIR", "LAS"), ("LAS", "ASL"), ("SLA", "LAS"),
])
def test_reorient_array_matches_simpleitk(source, target):
    img, arr = _image((5, 6, 7), (2.0, 1.0, 0.5), source)
    expected = sitk.GetArrayFromImage(sitk.DICOMOrient(img, target))
    got, perm = no.reorient_array(arr, source, target)
    np.testing.assert_array_equal(got, expected)
    spacing_zyx = (2.0, 1.0, 0.5)
    oriented = sitk.DICOMOrient(img, target)
    assert [spacing_zyx[p] for p in perm] == list(oriented.GetSpacing()[::-1])


def test_reorient_round_trip_and_errors():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    there, _ = no.reorient_array(arr, "RPI", "LAS")
    back, _ = no.reorient_array(there, "LAS", "RPI")
    np.testing.assert_array_equal(back, arr)
    with pytest.raises(ValueError):
        no.reorient_array(arr, "LAS", "LLS")
    with pytest.raises(ValueError):
        no.reorient_array(arr, "LA", "LAS")


def test_compute_steps_and_gaussian():
    assert no.compute_steps((128,), (128,), 0.75) == [[0]]
    steps = no.compute_steps((846, 665), (128, 256), 0.75)
    assert len(steps[0]) == 9 and steps[0][0] == 0 and steps[0][-1] == 846 - 128
    assert len(steps[1]) == 4 and steps[1][-1] == 665 - 256
    g = no.compute_gaussian((8, 16, 12))
    assert g.shape == (8, 16, 12) and g.dtype == np.float32
    assert g.max() == pytest.approx(10.0) and g.min() > 0
    assert np.unravel_index(g.argmax(), g.shape) == (4, 8, 6)


def test_nonzero_bbox_and_normalisation():
    v = np.zeros((6, 7, 8), dtype=np.int16)
    v[1:4, 2:5, 3:7] = 5
    assert no.nonzero_bbox(v) == (slice(1, 4), slice(2, 5), slice(3, 7))
    assert no.nonzero_bbox(np.zeros((2, 3, 4))) == (slice(0, 2), slice(0, 3), slice(0, 4))
    x = np.array([-2000.0, 0.0, 199.3, 5000.0], dtype=np.float32)
    y = no.ct_normalization(x, mean_intensity=199.3, std_intensity=220.9, percentile_00_5=-163.0, percentile_99_5=826.0)
    assert y[0] == pytest.approx((-163.0 - 199.3) / 220.9) and y[3] == pytest.approx((826.0 - 199.3) / 220.9)
    assert y[2] == pytest.approx(0.0, abs=1e-6)


def test_resampling_shapes_and_separate_z():
    data = np.random.default_rng(1).standard_normal((1, 10, 20, 24)).astype(np.float32)
    new_shape = no.compute_new_shape(data.shape[1:], (1.0, 0.5, 0.5), (0.5, 1.0, 1.0))
    assert list(new_shape) == [20, 10, 12]
    out = no.resample_data_to_shape(data, new_shape, (1.0, 0.5, 0.5), (0.5, 1.0, 1.0), order=3)
    assert out.shape == (1, 20, 10, 12) and out.dtype == np.float32
    # anisotropic (> 3x): in-plane cubic then nearest along z, still the right shape
    out_z = no.resample_data_to_shape(data, (5, 20, 24), (5.0, 1.0, 1.0), (10.0, 1.0, 1.0), order=3, order_z=0)
    assert out_z.shape == (1, 5, 20, 24)
    # identity when the shape does not change
    assert no.resample_data_to_shape(data, data.shape[1:], (1, 1, 1), (1, 1, 1)) is data


def test_resample_logits_to_labels_equals_argmax():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((4, 6, 8, 10)).astype(np.float32)
    new_shape = (9, 12, 15)
    labels = no.resample_logits_to_labels(logits, new_shape, (1.0, 1.0, 1.0), (1.5, 1.5, 1.5))
    full = np.stack([
        no.resample_data_to_shape(logits[c : c + 1], new_shape, (1.0, 1.0, 1.0), (1.5, 1.5, 1.5), order=1)[0]
        for c in range(4)
    ])
    np.testing.assert_array_equal(labels, full.argmax(0).astype(np.uint8))


def test_model_config_from_files(tmp_path):
    cfg = {
        "patch_size": [8, 16, 12], "spacing": [1.0, 0.5, 0.5], "mean": 100.0, "std": 50.0,
        "percentile_00_5": -100.0, "percentile_99_5": 500.0,
        "labels": {"background": 0, "a": 1, "b": 2}, "orientation": "las", "tile_step_size": 0.6, "batch_size": 2,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    c = no.ModelConfig.from_files(p, {"mirror_axes": "0,2", "source": "trainer x"})
    assert c.patch_size == (8, 16, 12) and c.num_classes == 3 and c.orientation == "LAS"
    assert c.mirror_axes == (0, 2) and c.source == "trainer x" and c.value_to_name[2] == "b"
    del cfg["mean"]
    p.write_text(json.dumps(cfg))
    with pytest.raises(no.SegmentationUnavailable, match="lacks"):
        no.ModelConfig.from_files(p)
    est, shape = no.estimate_inference_bytes((100, 200, 200), (1.0, 0.5, 0.5), c)
    assert tuple(shape) == (100, 200, 200) and est == (3 + 4) * 4 * 100 * 200 * 200


@pytest.mark.skipif(not (WEIGHTS / "config.json").is_file(), reason="HolOrama weights not fetched")
def test_real_config_matches_label_table():
    from holorama_labels import check_against_model_config

    c = no.ModelConfig.from_files(WEIGHTS / "config.json")
    check_against_model_config(c.labels)
    assert c.patch_size == (128, 256, 224) and c.orientation == "LAS" and c.num_classes == 15


class _StubSession:
    """Stands in for onnxruntime: class 1 where the normalised intensity is above 0, class 0 elsewhere."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.calls = 0

    def run(self, _outputs, feeds):
        x = feeds["image"]  # (B, 1, pz, py, px)
        self.calls += 1
        logits = np.zeros((x.shape[0], self.num_classes, *x.shape[2:]), dtype=np.float32)
        logits[:, 1] = x[:, 0] * 5.0
        logits[:, 0] = -x[:, 0] * 5.0
        return [logits]


def _stub_segmenter(tmp_path, patch=(8, 16, 12), spacing=(1.0, 1.0, 1.0), orientation="LAS", step=0.5):
    cfg = {
        "patch_size": list(patch), "spacing": list(spacing), "mean": 100.0, "std": 100.0,
        "percentile_00_5": -500.0, "percentile_99_5": 1000.0,
        "labels": {"background": 0, "bright": 1}, "orientation": orientation, "tile_step_size": step, "batch_size": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    model = tmp_path / "stub.onnx"
    model.write_bytes(b"not a real model")
    seg = no.OnnxNnUNet3D(model, device="cpu")
    seg._session = _StubSession()
    seg._config = no.ModelConfig.from_files(tmp_path / "config.json")
    seg.active_device, seg.active_precision = "stub", "fp32"
    return seg


def test_predict_recovers_a_bright_block_through_reorientation_and_resampling(tmp_path):
    seg = _stub_segmenter(tmp_path, spacing=(1.0, 1.0, 1.0))
    # volume in RPI with anisotropic spacing: the engine must reorient to LAS and resample to 1 mm
    vol = np.full((12, 20, 26), -50, dtype=np.int16)  # dark: normalised (-50-100)/100 < 0 -> class 0
    vol[3:9, 5:15, 8:20] = 400  # bright block -> class 1
    labels = seg.predict(vol, spacing=(2.0, 0.5, 0.75), orientation="RPI")
    assert labels.shape == vol.shape and labels.dtype == np.uint8
    truth = vol > 0
    pred = labels == 1
    dice = 2 * (truth & pred).sum() / (truth.sum() + pred.sum())
    assert dice > 0.9
    assert seg._session.calls > 1  # tiled: the resampled volume is larger than one patch


def test_predict_validates_inputs(tmp_path):
    seg = _stub_segmenter(tmp_path)
    with pytest.raises(ValueError, match="spacing"):
        seg.predict(np.zeros((4, 8, 8), dtype=np.int16), spacing=(1.0, 1.0))
    with pytest.raises(ValueError, match="volume"):
        seg.predict(np.zeros((8, 8), dtype=np.int16), spacing=(1.0, 1.0, 1.0))
    cancelled = seg.predict(np.ones((4, 8, 8), dtype=np.int16) * 500, spacing=(1.0, 1.0, 1.0), progress_cb=lambda d, t: False)
    assert cancelled is None


def test_load_reports_missing_pieces(tmp_path):
    seg = no.OnnxNnUNet3D(tmp_path / "none.onnx", device="cpu")
    with pytest.raises(no.SegmentationUnavailable, match="Model not found"):
        seg.load()
    (tmp_path / "none.onnx").write_bytes(b"x")
    with pytest.raises(no.SegmentationUnavailable, match="config.json"):
        seg.load()
    with pytest.raises(no.SegmentationUnavailable, match="device must be"):
        no.OnnxNnUNet3D(tmp_path / "none.onnx", device="tpu").load()
