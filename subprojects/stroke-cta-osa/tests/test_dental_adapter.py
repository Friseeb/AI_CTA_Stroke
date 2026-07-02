"""DentalAirwayAdapter reads JSON + NIfTI artefacts without importing cta_dental."""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from stroke_cta_osa.adapters import DentalAirwayAdapter
from stroke_cta_osa.cli import _discover_dental_artifacts


def test_dental_adapter_unavailable_when_paths_missing():
    a = DentalAirwayAdapter(None, None, None)
    assert a.is_available() is False


def test_dental_adapter_reads_landmarks_json(synth_cta, tmp_path):
    lm_path = tmp_path / "landmarks.json"
    lm_path.write_text(json.dumps({
        "hyoid": [40, 44, 40],
        "epiglottis_tip": [25, 44, 40],
        "posterior_nasal_spine": [60, 44, 40],
        "soft_palate_inferior": None,
    }))
    a = DentalAirwayAdapter(None, lm_path, None)
    assert a.is_available() is False  # without mask/features it's still false
    lm = a.get_landmarks(synth_cta)
    assert lm.hyoid == (40, 44, 40)
    assert lm.epiglottis_tip == (25, 44, 40)


def test_dental_adapter_reads_features_json(tmp_path):
    feats_path = tmp_path / "airway_features.json"
    feats_path.write_text(json.dumps({
        "airway_volume_ml": 12.34,
        "airway_min_csa_mm2": 56.7,
        "non_numeric_field": "should_be_skipped",
    }))
    a = DentalAirwayAdapter(None, None, feats_path)
    assert a.is_available() is True
    feats = a.get_existing_features()
    assert feats.values["airway_volume_ml"] == 12.34
    assert feats.values["airway_min_csa_mm2"] == 56.7
    assert "non_numeric_field" not in feats.values


def test_dental_adapter_reads_mask(synth_cta, synth_airway_mask_path):
    a = DentalAirwayAdapter(synth_airway_mask_path, None, None)
    assert a.is_available() is True
    info = a.get_airway_mask(synth_cta)
    assert info is not None
    assert info.method == "dental_adapter"
    assert info.mask_zyx.any()


def test_dental_adapter_payload_features_flow_into_orchestrator(
    synth_nifti_path, synth_airway_mask_path, tmp_path,
):
    """End-to-end: dental-provided features show up in the CTA feature row
    with the *_from_dental suffix, alongside CTA-recomputed values."""
    from stroke_cta_osa.config import PipelineConfig, apply_overrides
    from stroke_cta_osa.features import extract_case

    feats_path = tmp_path / "airway_features.json"
    feats_path.write_text(json.dumps({"airway_volume_ml": 99.9}))
    cfg = apply_overrides(PipelineConfig(), {
        "airway.use_existing_dental_airway_outputs": True,
        "airway.dental_airway_mask_path": str(synth_airway_mask_path),
        "airway.dental_features_path": str(feats_path),
    })
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="ext_dental")
    row = result.to_feature_row()
    assert row["airway_method"] == "dental_adapter"
    assert row["airway_volume_ml_from_dental"] == 99.9
    # The CTA-recomputed value should still be present (and different)
    assert row["airway_volume_ml"] != 99.9


def test_dental_artifact_discovery_finds_lower_jawbone(tmp_path):
    case_dir = tmp_path / "sub-001" / "roi" / "_tseg_teeth"
    case_dir.mkdir(parents=True)
    jaw = case_dir / "lower_jawbone.nii.gz"
    jaw.write_bytes(b"placeholder")

    found = _discover_dental_artifacts(tmp_path, case_id="sub-001")

    assert found["mandible"] == jaw


def test_dental_mandible_mask_flows_into_orchestrator(
    synth_nifti_path, synth_array, tmp_path,
):
    from stroke_cta_osa.config import PipelineConfig, apply_overrides
    from stroke_cta_osa.features import extract_case

    mask = np.zeros_like(synth_array, dtype=np.uint8)
    mask[60:68, 20:34, 20:60] = 1
    img = sitk.GetImageFromArray(mask)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    jaw_path = tmp_path / "lower_jawbone.nii.gz"
    sitk.WriteImage(img, str(jaw_path), useCompression=True)

    cfg = apply_overrides(PipelineConfig(), {
        "mandible.dental_mandible_mask_path": str(jaw_path),
    })
    result = extract_case(
        synth_nifti_path, tmp_path, cfg, patient_id="dental_jawbone",
    )

    assert result.mandible["mandible_mask_available"] is True
    assert result.mandible["mandible_mask_method"] == "dental_mandible_mask"
    assert result.mandible["mandible_volume_ml"] > 0
