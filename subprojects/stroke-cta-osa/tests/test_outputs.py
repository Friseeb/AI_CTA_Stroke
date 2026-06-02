"""Orchestrator end-to-end on synthetic data, output schema."""

import json
from pathlib import Path

import pandas as pd
import pytest

from stroke_cta_osa.config import PipelineConfig, apply_overrides
from stroke_cta_osa.features import extract_case
from stroke_cta_osa.output import write_outputs, append_processing_log


REQUIRED_FEATURE_COLUMNS = {
    "patient_id", "study_id", "scan_id", "pipeline", "pipeline_version",
    "config_hash", "processing_timestamp", "input_path_hash", "airway_source",
    "qc_pass", "qc_coverage_score", "qc_has_upper_airway",
    "airway_mask_available", "airway_method", "airway_volume_ml",
    "airway_min_csa_mm2", "airway_csa_p10_mm2", "airway_csa_median_mm2",
    "fat_hu_min_used", "fat_cervical_volume_ml", "fat_cervical_mean_hu",
    "fat_subcutaneous_cervical_volume_ml",
    "fat_deep_cervical_volume_ml", "fat_deep_to_subcutaneous_ratio",
    "fat_parapharyngeal_total_volume_ml",
    "fat_parapharyngeal_asymmetry_index",
    "fat_retropharyngeal_volume_ml",
    "cta_osa_anatomy_score_untrained",
    "cta_osa_fat_score_untrained",
}


def test_extract_case_end_to_end(synth_nifti_path, tmp_path):
    cfg = PipelineConfig()
    result = extract_case(input_path=synth_nifti_path, out_dir=tmp_path, cfg=cfg,
                          patient_id="synth_001")
    row = result.to_feature_row()
    missing = REQUIRED_FEATURE_COLUMNS - set(row.keys())
    assert not missing, f"missing required feature columns: {missing}"
    assert row["airway_mask_available"] is True
    assert row["airway_method"] == "threshold_connected_component"
    assert row["fat_cervical_volume_ml"] > 0


def test_write_outputs_creates_csvs(synth_nifti_path, tmp_path):
    cfg = PipelineConfig()
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="synth_002")
    paths = write_outputs([result], out_dir=tmp_path)
    feat = pd.read_csv(paths["features"])
    qc = pd.read_csv(paths["qc"])
    assert len(feat) == 1
    assert len(qc) == 1
    # Identifier columns come first
    assert feat.columns[0] == "pipeline"
    meta = json.loads(paths["feature_metadata"].read_text())
    assert meta["pipeline"] == "stroke_cta_osa"
    assert meta["n_rows"] == 1


def test_external_mask_path_via_overrides(synth_nifti_path, synth_airway_mask_path, tmp_path):
    cfg = apply_overrides(PipelineConfig(), {
        "airway.fallback_method": "external_mask_only",
        "airway.external_mask_path": str(synth_airway_mask_path),
    })
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="synth_ext")
    assert result.airway["airway_method"] == "external_mask"
    assert result.airway["airway_mask_available"] is True


def test_processing_log_jsonl(synth_nifti_path, tmp_path):
    cfg = PipelineConfig()
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="synth_003")
    log_path = tmp_path / "log.jsonl"
    append_processing_log(log_path, result, {"input_path": str(synth_nifti_path)})
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["patient_id"] == "synth_003"
    assert "qc_pass" in entry


def test_no_phi_in_feature_row(synth_nifti_path, tmp_path):
    """Identifier columns must never contain PHI keywords; even when the input
    has no DICOM headers, the orchestrator still produces opaque hashes."""
    cfg = PipelineConfig()
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="synth_004")
    row = result.to_feature_row()
    forbidden = {"PatientName", "PatientID", "PatientBirthDate",
                 "AccessionNumber", "InstitutionName", "MRN"}
    for k in row.keys():
        assert k not in forbidden
    assert row["study_id"].startswith("stu_")
    assert row["scan_id"].startswith("scn_")
