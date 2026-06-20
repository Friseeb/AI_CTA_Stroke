"""Unit tests for the LAA annotation core (no Slicer required)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "slicer_module"
sys.path.insert(0, str(MODULE_DIR))

from laa_annotation_core import (  # noqa: E402
    CANDIDATE_SOURCE_FILES,
    LAA_LABEL_CONTRACT,
    SESSION_CSV_FIELDS,
    TYPE1_LABEL,
    WHOLE_LAA_LABEL,
    PilotMetrics,
    Prompt,
    PromptLog,
    append_session_csv,
    build_monai_inference_request,
    build_session_log,
    comparison_labelmap,
    comparison_metrics,
    dice,
    hd95,
    infer_case_id,
    interrater_report,
    label_contract_metadata,
    output_paths,
    read_repro_manifest,
    resolve_candidate_file,
    session_csv_row,
    surface_dice,
    validate_label_values,
    validate_prompt,
)


# --- core / source hygiene -------------------------------------------------

def test_core_has_no_slicer_dependencies():
    src = (MODULE_DIR / "laa_annotation_core.py").read_text()
    for forbidden in ("import slicer", "import vtk", "import qt"):
        assert forbidden not in src


# --- label contract --------------------------------------------------------

def test_label_contract_primary_targets():
    meta = label_contract_metadata()
    assert meta["primary_target"] == WHOLE_LAA_LABEL
    assert meta["labels"][str(WHOLE_LAA_LABEL)]["primary"] is True
    assert meta["labels"][str(TYPE1_LABEL)]["primary"] is True
    assert meta["labels"]["3"]["primary"] is False
    assert set(meta["labels"]) == {str(v) for v in LAA_LABEL_CONTRACT}


def test_validate_label_values():
    assert validate_label_values([0, 1, 2]) == []
    assert any("Whole-LAA" in w for w in validate_label_values([0, 3]))
    assert any("outside the contract" in w for w in validate_label_values([1, 9]))
    # Type 1 may be absent (empty) without warning
    assert validate_label_values([1]) == []


# --- prompts ---------------------------------------------------------------

def test_validate_prompt_rejects_mismatched_category():
    validate_prompt("positive", "distal_tip")
    validate_prompt("negative", "aorta")
    with pytest.raises(ValueError):
        validate_prompt("positive", "aorta")  # negative category on positive
    with pytest.raises(ValueError):
        validate_prompt("sideways", "distal_tip")


def test_prompt_sets_timestamp_and_coordinate_floats():
    p = Prompt("positive", "distal_tip", (1, 2, 3))
    assert p.coordinate == (1.0, 2.0, 3.0)
    assert p.timestamp


def test_prompt_log_counts_and_roundtrip(tmp_path: Path):
    log = PromptLog(case_id="sub-001", reader_id="A")
    log.add("positive", "distal_tip", (1, 2, 3), model_used="vista3d")
    log.add("positive", "distal_lobe", (4, 5, 6))
    log.add("negative", "aorta", (7, 8, 9))
    assert log.count == 3
    assert log.positive_count == 2
    assert log.negative_count == 1
    assert log.by_category()["distal_tip"] == 1

    path = log.save(tmp_path / "prompts.json")
    loaded = PromptLog.load(path)
    assert loaded.count == 3
    assert loaded.positive_count == 2
    assert loaded.prompts[0].model_used == "vista3d"


# --- pilot metrics ---------------------------------------------------------

def test_pilot_metrics_roundtrip(tmp_path: Path):
    pm = PilotMetrics(
        case_id="sub-001",
        reader_id="A",
        annotation_time_s=420.0,
        prompt_count=3,
        positive_prompt_count=2,
        negative_prompt_count=1,
        segmentation_confidence=0.8,
        type1_confidence=0.5,
        image_quality=4,
        type1_present=True,
    )
    path = pm.save(tmp_path / "pilot.json")
    loaded = PilotMetrics.load(path)
    assert loaded.annotation_time_s == 420.0
    assert loaded.image_quality == 4
    assert loaded.validate() == []


def test_pilot_metrics_validation_flags():
    pm = PilotMetrics(
        case_id="",
        segmentation_confidence=1.5,
        image_quality=9,
        prompt_count=5,
        positive_prompt_count=1,
        negative_prompt_count=1,
    )
    warnings = pm.validate()
    assert any("case_id" in w for w in warnings)
    assert any("segmentation_confidence" in w for w in warnings)
    assert any("image_quality" in w for w in warnings)
    assert any("prompt_count" in w for w in warnings)


# --- output paths ----------------------------------------------------------

def test_output_paths_tree(tmp_path: Path):
    paths = output_paths(tmp_path, "sub-001").mkdirs()
    assert paths.root == tmp_path / "laa_annotation"
    for sub in ("candidate_masks", "manual_masks", "type1_masks", "iterations",
                "logs", "screenshots", "metrics"):
        assert (paths.root / sub).is_dir()
    assert paths.whole_laa_mask().name == "sub-001_whole_laa.nii.gz"
    assert paths.type1_mask().parent == paths.type1_masks


def test_output_paths_reader_subfolder(tmp_path: Path):
    paths = output_paths(tmp_path, "sub-001", reader_id="readerB")
    assert paths.root == tmp_path / "laa_annotation" / "readerB"
    assert paths.session_csv().parent == paths.logs


# --- session log -----------------------------------------------------------

def test_build_session_log_and_csv(tmp_path: Path):
    log = PromptLog(case_id="sub-001", reader_id="A")
    log.add("positive", "distal_tip", (1, 2, 3))
    log.add("negative", "aorta", (4, 5, 6))
    pm = PilotMetrics(
        case_id="sub-001", reader_id="A", model_used="vista3d",
        annotation_time_s=300.0, edit_count=12, image_quality=4,
        prompt_count=2, positive_prompt_count=1, negative_prompt_count=1,
    )
    session = build_session_log(
        case_id="sub-001", reader_id="A", pilot=pm, prompt_log=log,
        output_dir=tmp_path, whole_laa_mask=tmp_path / "m.nii.gz",
    )
    assert session["prompt_count"] == 2
    assert session["positive_prompt_count"] == 1
    assert session["label_contract"] == "laa_completion_v1"
    assert len(session["prompts"]) == 2

    csv_path = tmp_path / "session.csv"
    append_session_csv(csv_path, session)
    append_session_csv(csv_path, session)  # second case -> no duplicate header
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert list(rows[0]) == list(SESSION_CSV_FIELDS)
    assert rows[0]["edit_count"] == "12"

    # row builder tolerates None values
    sparse = session_csv_row({"case_id": "x"})
    assert sparse["annotation_time_s"] == ""


# --- MONAILabel request ----------------------------------------------------

def test_build_monai_inference_request_splits_prompts():
    log = PromptLog()
    log.add("positive", "distal_tip", (1, 2, 3))
    log.add("negative", "aorta", (4, 5, 6))
    req = build_monai_inference_request(
        image="/data/sub-001_ct.nii.gz", model="vista3d_laa",
        prompt_log=log, current_label="/data/cand.nii.gz",
    )
    assert req["foreground"] == [[1.0, 2.0, 3.0]]
    assert req["background"] == [[4.0, 5.0, 6.0]]
    assert req["label"] == "/data/cand.nii.gz"
    assert req["model"] == "vista3d_laa"


# --- reproducibility metrics ----------------------------------------------

def _cube(shape, lo, hi):
    m = np.zeros(shape, dtype=np.uint8)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    m[sl] = 1
    return m


def test_dice_identity_and_disjoint():
    a = _cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert dice(a, a) == pytest.approx(1.0)
    b = _cube((20, 20, 20), (0, 0, 0), (3, 3, 3))
    assert dice(a, b) == 0.0
    assert dice(np.zeros((4, 4, 4)), np.zeros((4, 4, 4))) == 1.0


def test_hd95_and_surface_dice():
    a = _cube((30, 30, 30), (10, 10, 10), (20, 20, 20))
    assert hd95(a, a) == pytest.approx(0.0)
    assert surface_dice(a, a, tolerance_mm=1.0) == pytest.approx(1.0)
    # one empty -> inf / 0
    empty = np.zeros((30, 30, 30), dtype=np.uint8)
    assert hd95(a, empty) == float("inf")
    assert surface_dice(a, empty) == 0.0
    # slightly shifted cube -> small but finite HD95
    b = _cube((30, 30, 30), (11, 10, 10), (21, 20, 20))
    d = hd95(a, b, spacing=(1.0, 1.0, 1.0))
    assert 0.0 < d < 5.0


def test_interrater_report():
    a = _cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    b = _cube((20, 20, 20), (6, 5, 5), (16, 15, 15))
    report = interrater_report({"A": a, "B": b, "C": a})
    assert report["readers"] == ["A", "B", "C"]
    assert report["n_pairs"] == 3
    assert 0.0 < report["mean_dice"] <= 1.0
    # A vs C identical -> a pair with dice 1.0 present
    assert any(p["dice"] == pytest.approx(1.0) for p in report["pairs"])


# --- manifest helpers ------------------------------------------------------

def test_infer_case_id():
    assert infer_case_id("/tmp/sub-547_acq-CTA_ct.nii.gz") == "sub-547"
    assert infer_case_id("caseX_whole_laa.nii.gz") == "caseX"


def test_read_repro_manifest(tmp_path: Path):
    p = tmp_path / "manifest.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "reader_id", "mask_path"])
        w.writeheader()
        w.writerow({"case_id": "sub-001", "reader_id": "A", "mask_path": "/d/a.nii.gz"})
        w.writerow({"case_id": "", "reader_id": "B", "mask_path": "/d/sub-002_whole_laa.nii.gz"})
    rows = read_repro_manifest(p)
    assert rows[0]["case_id"] == "sub-001"
    assert rows[1]["case_id"] == "sub-002"  # inferred

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        read_repro_manifest(bad)


def test_comparison_labelmap_encodes_added_removed_unchanged():
    old = np.zeros((4, 4, 4), dtype=np.uint8)
    new = np.zeros((4, 4, 4), dtype=np.uint8)
    old[0, 0, 0] = 1  # removed by reader
    old[1, 1, 1] = 1  # kept
    new[1, 1, 1] = 1  # kept
    new[2, 2, 2] = 1  # added by reader
    lm = comparison_labelmap(old, new)
    assert lm[1, 1, 1] == 1  # unchanged
    assert lm[2, 2, 2] == 2  # added
    assert lm[0, 0, 0] == 3  # removed
    assert lm[3, 3, 3] == 0  # background


def test_comparison_labelmap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        comparison_labelmap(np.zeros((2, 2, 2)), np.zeros((3, 3, 3)))


def test_comparison_metrics_volumes_and_dice():
    old = np.zeros((10, 10, 10), dtype=np.uint8)
    new = np.zeros((10, 10, 10), dtype=np.uint8)
    old[:2] = 1  # 200 voxels
    new[:3] = 1  # 300 voxels, fully contains old
    m = comparison_metrics(old, new, spacing=(2.0, 1.0, 1.0))  # 2 mm^3/voxel
    assert m["old_voxels"] == 200
    assert m["new_voxels"] == 300
    assert m["added_voxels"] == 100
    assert m["removed_voxels"] == 0
    # 100 voxels * 2 mm^3 / 1000 = 0.2 mL
    assert m["added_volume_ml"] == pytest.approx(0.2)
    assert m["volume_change_ml"] == pytest.approx(0.2)
    assert m["volume_change_pct"] == pytest.approx(50.0)
    assert m["dice"] == pytest.approx(dice(old, new))


def test_comparison_metrics_empty_old_is_all_added():
    old = np.zeros((4, 4, 4), dtype=np.uint8)
    new = np.zeros((4, 4, 4), dtype=np.uint8)
    new[0, 0, 0] = 1
    m = comparison_metrics(old, new)
    assert m["added_voxels"] == 1
    assert m["volume_change_pct"] is None  # undefined when old is empty


def test_resolve_candidate_file_prefers_exact_then_glob(tmp_path: Path):
    d = tmp_path / "candidate_masks"
    d.mkdir()
    (d / "vista3d_laa.nii.gz").write_text("x")
    (d / "sub-255_nudf_laa.nii.gz").write_text("x")
    # exact stem match
    assert resolve_candidate_file([d], ("vista3d_laa",)).name == "vista3d_laa.nii.gz"
    # prefixed glob match (*_<stem>.nii.gz)
    assert resolve_candidate_file([d], ("nudf_laa",)).name == "sub-255_nudf_laa.nii.gz"
    # missing -> None; missing dirs skipped
    assert resolve_candidate_file([tmp_path / "nope", d], ("consensus_laa",)) is None


def test_totalsegmentator_source_maps_to_atrial_appendage():
    # TotalSegmentator's `total` task emits the LAA as atrial_appendage_left.
    assert "atrial_appendage_left" in CANDIDATE_SOURCE_FILES["TotalSegmentator"]
    d_stems = CANDIDATE_SOURCE_FILES["TotalSegmentator"]
    assert "totalseg_laa" in d_stems
