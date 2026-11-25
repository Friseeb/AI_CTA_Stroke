"""End-to-end test of the case pipeline with a stub model: a synthetic CTA in RAS orientation with two
contrast-filled spheres, a tight crop mask, and a segmenter that labels by Hounsfield units. Checks that
every output lands on the CTA's own grid, that the crop/paste-back and the field-of-view flags are right."""
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from heart_segmentation import Params, segment_case  # noqa: E402
from holorama_labels import HOLORAMA_CONFIG_NAMES, LABELS  # noqa: E402
from nnunet_onnx import ModelConfig  # noqa: E402


class StubSegmenter:
    """HU > 300 in the lower (inferior) half -> LV (4), in the upper half -> LA (3); 60..120 HU -> epicardial fat (10)."""

    config = ModelConfig(
        patch_size=(8, 8, 8), spacing=(1.0, 1.0, 1.0), mean=0.0, std=1.0, percentile_00_5=-1000.0,
        percentile_99_5=1000.0, labels=HOLORAMA_CONFIG_NAMES, orientation="LAS",
    )
    active_device = "stub"

    def __init__(self):
        self.seen = []

    def predict(self, volume, spacing, orientation=None, progress_cb=None):
        assert orientation == "LAS"
        self.seen.append((volume.shape, tuple(spacing)))
        out = np.zeros(volume.shape, dtype=np.uint8)
        z = np.arange(volume.shape[0])[:, None, None]
        bright = volume > 300
        out[bright & (z < volume.shape[0] // 2)] = 4
        out[bright & (z >= volume.shape[0] // 2)] = 3
        out[(volume > 60) & (volume < 120)] = 10
        return out

    def describe(self):
        return {"model_path": "stub", "config": self.config.to_dict()}


def _sphere(shape, c, r):
    g = np.mgrid[: shape[0], : shape[1], : shape[2]]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r * r


def test_pipeline_outputs_on_source_grid(tmp_path):
    # nibabel array order is (x, y, z); a positive diagonal affine is RAS (SimpleITK code 'RAS')
    shape = (40, 48, 56)
    spacing = (0.8, 0.8, 1.0)
    hu = np.full(shape, 40, np.float32)
    lv = _sphere(shape, (20, 24, 12), 9)  # inferior sphere, touches z = 0? no (12 - 9 = 3)
    la = _sphere(shape, (20, 24, 40), 8)  # superior sphere
    fat = _sphere(shape, (20, 44, 52), 3)  # 90 HU blob, outside the crop box even after the margin
    hu[lv] = 350
    hu[la] = 350
    hu[fat] = 90
    aff = np.diag([*spacing, 1.0])
    aff[:3, 3] = (-15.0, -20.0, 100.0)
    cta = tmp_path / "case.nii.gz"
    nib.save(nib.Nifti1Image(hu.astype(np.int16), aff), cta)
    # crop mask: both spheres but not the fat blob; the LA sphere pokes out of the box on top
    crop = np.zeros(shape, np.uint8)
    crop[6:34, 10:38, 2:44] = 1
    mask = tmp_path / "heart.nii.gz"
    nib.save(nib.Nifti1Image(crop, aff), mask)

    seg = StubSegmenter()
    out = tmp_path / "out"
    manifest = segment_case(cta, out, "case", Params(crop_margin_mm=2.0, write_qc=True), crop_mask_path=mask, segmenter=seg, log=lambda m: None)

    # the segmenter saw a cropped LAS volume with (dz, dy, dx) spacing
    assert len(seg.seen) == 1
    sub_shape, sub_spacing = seg.seen[0]
    assert sub_spacing == pytest.approx((1.0, 0.8, 0.8))  # NIfTI stores float32 spacing
    assert all(a < b for a, b in zip(sub_shape, (56, 48, 40)))
    assert manifest["cta"]["source_orientation"] == "RAS" and manifest["model_orientation"] == "LAS"
    assert manifest["resampled_back_to_source_grid"] is False

    # labelmap on the CTA's own grid, label values at the sphere centres in the CTA's voxel indexing
    seg_img = nib.load(out / "case_holorama_heart_seg.nii.gz")
    assert seg_img.shape == shape and np.allclose(seg_img.affine, aff)
    lab = np.asarray(seg_img.dataobj)
    assert lab[20, 24, 12] == 4 and lab[20, 24, 40] == 3
    assert lab[20, 44, 52] == 0  # fat blob was outside the crop box -> background
    assert lab.dtype == np.uint8
    assert set(np.unique(lab).tolist()) == {0, 3, 4}

    # per-structure masks: all 14 files, LV mask equals label 4
    sdir = out / "structures"
    assert sorted(p.name for p in sdir.glob("*.nii.gz")) == sorted(f"{l.key}.nii.gz" for l in LABELS)
    lv_mask = np.asarray(nib.load(sdir / "left_ventricle.nii.gz").dataobj) > 0
    np.testing.assert_array_equal(lv_mask, lab == 4)

    # volumes and field-of-view flags
    rows = {r["structure"]: r for r in csv.DictReader((out / "case_holorama_heart_volumes.csv").open())}
    voxel_ml = np.prod(spacing) / 1000
    assert float(rows["left_ventricle"]["volume_ml"]) == round(lv.sum() * voxel_ml, 3)
    assert rows["left_ventricle"]["present"] == "True" and rows["epicardial_fat"]["present"] == "False"
    assert rows["left_atrium"]["touches_crop_box"] == "True"  # LA sphere reaches the crop box's top (44 + 2 mm margin = 46 < 48)
    assert rows["left_ventricle"]["touches_crop_box"] == "False"
    assert rows["left_atrium"]["touches_scan_superior"] == "False"
    assert manifest["structures_touching_crop_box"] == ["left_atrium"]
    assert manifest["structures_truncated_by_scan"] == []

    # sidecars
    labels_json = json.loads((out / "case_holorama_heart_labels.json").read_text())
    assert labels_json["labelmap"] == "case_holorama_heart_seg.nii.gz" and len(labels_json["labels"]) == 14
    assert (out / "case_holorama_heart_qc.png").is_file()
    assert (out / "case_holorama_heart_manifest.json").is_file()
    assert manifest["inputs"]["crop_mask_sha256"] and manifest["crop_box_zyx"][0][0] == 0  # z crop starts at 0 (2 - 2 mm margin)


def test_scan_edge_flags_without_crop(tmp_path):
    shape = (24, 24, 30)
    hu = np.full(shape, 40, np.float32)
    hu[_sphere(shape, (12, 12, 3), 5)] = 350  # LV sphere cut by the inferior scan edge (z = 0)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    cta = tmp_path / "case.nii.gz"
    nib.save(nib.Nifti1Image(hu.astype(np.int16), aff), cta)
    manifest = segment_case(cta, tmp_path / "out", "case", Params(write_qc=False, write_structures=False), segmenter=StubSegmenter(), log=lambda m: None)
    rows = {r["structure"]: r for r in csv.DictReader((tmp_path / "out" / "case_holorama_heart_volumes.csv").open())}
    assert rows["left_ventricle"]["touches_scan_inferior"] == "True"
    assert rows["left_ventricle"]["touches_scan_superior"] == "False"
    assert rows["left_ventricle"]["touches_crop_box"] == "False"  # no crop: every face is a scan face
    assert manifest["structures_truncated_by_scan"] == ["left_ventricle"]
    assert not (tmp_path / "out" / "structures").exists()
