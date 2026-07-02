import json
from pathlib import Path

from aorta_cta_radiomics.metadata_filter import evaluate_neuro_cta_metadata, resolve_metadata_path


def test_neuro_cta_json_sidecar_is_eligible(tmp_path: Path):
    image = tmp_path / "sub-001_acq-CTA_ct.nii.gz"
    image.touch()
    image.with_name("sub-001_acq-CTA_ct.json").write_text(
        json.dumps(
            {
                "SeriesDescription": "CTA Head and Neck",
                "ProtocolName": "Hyperacute stroke angio",
                "BodyPartExamined": "HEADNECK",
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_neuro_cta_metadata(
        {"case_id": "sub-001", "image_path": image.name},
        manifest_base=tmp_path,
    )

    assert result.eligible
    assert result.reason == "eligible_neuro_cta"
    assert "cta" in result.matched_cta_terms
    assert "stroke" in result.matched_neuro_terms


def test_non_neuro_cta_json_sidecar_is_rejected(tmp_path: Path):
    image = tmp_path / "case_ct.nii.gz"
    image.touch()
    image.with_name("case_ct.json").write_text(
        json.dumps({"SeriesDescription": "CTA chest pulmonary embolism protocol"}),
        encoding="utf-8",
    )

    result = evaluate_neuro_cta_metadata(
        {"case_id": "case", "image_path": image.name},
        manifest_base=tmp_path,
    )

    assert not result.eligible
    assert result.reason == "excluded_non_neuro_protocol"
    assert "pulmonary embol" in result.matched_exclude_terms


def test_manifest_metadata_columns_can_make_case_eligible(tmp_path: Path):
    result = evaluate_neuro_cta_metadata(
        {
            "case_id": "sub-002",
            "image_path": "missing_file.nii.gz",
            "SeriesDescription": "CT angiography brain neck code stroke",
        },
        manifest_base=tmp_path,
    )

    assert result.eligible
    assert result.metadata_source == "manifest_columns"


def test_missing_metadata_skips_by_default_and_can_be_allowed(tmp_path: Path):
    row = {"case_id": "sub-003", "image_path": "sub-003_acq-CTA_ct.nii.gz"}

    skipped = evaluate_neuro_cta_metadata(row, manifest_base=tmp_path)
    allowed = evaluate_neuro_cta_metadata(row, manifest_base=tmp_path, allow_missing_metadata=True)

    assert not skipped.eligible
    assert skipped.reason == "no_metadata"
    assert allowed.eligible
    assert allowed.reason == "missing_metadata_allowed"


def test_explicit_metadata_path_column_is_resolved_relative_to_manifest(tmp_path: Path):
    sidecar = tmp_path / "metadata" / "case.json"
    sidecar.parent.mkdir()
    sidecar.write_text("{}", encoding="utf-8")

    resolved = resolve_metadata_path(
        {"image_path": "case.nii.gz", "metadata_path": "metadata/case.json"},
        manifest_base=tmp_path,
    )

    assert resolved == sidecar.resolve()
