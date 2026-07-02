from pathlib import Path
import sys

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "slicer_module"
sys.path.insert(0, str(MODULE_DIR))

from vertebral_review_core import (  # noqa: E402
    LEFT_LABEL_VALUE,
    RIGHT_LABEL_VALUE,
    append_queue_status,
    build_review_log,
    first_pending_index,
    infer_case_id,
    label_contract_metadata,
    latest_queue_status_by_case,
    normalize_reviewer_id,
    output_paths,
    queue_status_path,
    queue_status_row,
    read_manifest,
    review_csv_row,
    validate_label_values,
    validate_review_status,
)


def test_infer_case_id_prefers_bids_subject_id():
    assert infer_case_id("/tmp/sub-547_acq-CTA_ct.nii.gz") == "sub-547"
    assert infer_case_id("sub-ABC123_vert_clean") == "sub-ABC123"


def test_output_paths_preserve_existing_clean_label_name(tmp_path: Path):
    paths = output_paths(tmp_path, "sub-547")

    assert paths.clean_label == tmp_path / "sub-547_vert_clean.nii.gz"
    assert paths.log_json == tmp_path / "sub-547_vert_clean_log.json"
    assert paths.review_csv == tmp_path / "sub-547_vertebral_review.csv"
    assert paths.centerlines == tmp_path / "sub-547_vertebral_centerlines.mrk.json"


def test_review_status_and_reviewer_validation():
    assert validate_review_status("accepted") == "accepted"
    with pytest.raises(ValueError):
        validate_review_status("done")

    assert normalize_reviewer_id(" sf ") == "sf"
    assert normalize_reviewer_id("", required=False) == "unspecified"
    with pytest.raises(ValueError):
        normalize_reviewer_id("")


def test_label_contract_metadata_is_bilateral_one_two():
    metadata = label_contract_metadata()

    assert metadata["label_contract"] == "bilateral_vertebral_v1"
    assert metadata["labels"][str(LEFT_LABEL_VALUE)]["name"] == "Vert L"
    assert metadata["labels"][str(RIGHT_LABEL_VALUE)]["name"] == "Vert R"


def test_validate_label_values_reports_missing_and_extra():
    assert validate_label_values({0, 1, 2}) == []
    warnings = validate_label_values({0, 1, 3})

    assert any("Missing" in warning for warning in warnings)
    assert any("Unexpected" in warning for warning in warnings)


def test_review_log_and_csv_row_schema(tmp_path: Path):
    paths = output_paths(tmp_path, "sub-547")
    log = build_review_log(
        case_id="sub-547",
        reviewer_id="sf",
        review_status="accepted",
        cta_node="cta",
        label_node="vertebral_seg",
        cta_path="/data/sub-547.nii.gz",
        label_path=None,
        output_paths=paths,
        params={"notes": "good case"},
        negative_priors=[
            {
                "node": "foramen_prior",
                "path": "/data/sub-547_foramen_prior.nii.gz",
                "overlap_voxels": 12,
                "overlap_by_label": {"1": 5, "2": 7},
            }
        ],
        centerlines_saved=True,
        scene_saved=True,
    )
    row = review_csv_row(log)

    assert log["reviewer_id"] == "sf"
    assert log["review_status"] == "accepted"
    assert log["output_label"].endswith("sub-547_vert_clean.nii.gz")
    assert row["notes"] == "good case"
    assert row["centerlines_path"].endswith("sub-547_vertebral_centerlines.mrk.json")
    assert row["negative_prior_nodes"] == "foramen_prior"
    assert row["negative_prior_overlap_voxels"] == "12"


def test_manifest_queue_helpers(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,cta_path,label_path,foramen_prior_path,reviewer_id,review_status,notes\n"
        "sub-001,/data/sub-001_0000.nii.gz,,/data/sub-001_foramen.nii.gz,sf,in_progress,check\n"
        ",/data/sub-002_0000.nii.gz,,,,,\n",
        encoding="utf-8",
    )
    rows = read_manifest(manifest)

    assert len(rows) == 2
    assert rows[0]["case_id"] == "sub-001"
    assert rows[1]["case_id"] == "sub-002"
    assert rows[0]["foramen_prior_path"].endswith("sub-001_foramen.nii.gz")

    status_path = queue_status_path(manifest, tmp_path / "out")
    assert status_path == tmp_path / "out" / "manifest_queue_status.csv"

    append_queue_status(status_path, queue_status_row(rows[0], queue_status="completed", output_label="out.nii.gz"))
    latest = latest_queue_status_by_case(status_path)

    assert latest["sub-001"]["queue_status"] == "completed"
    assert first_pending_index(rows, latest) == 1
