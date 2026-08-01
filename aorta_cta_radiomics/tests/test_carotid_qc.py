from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from aorta_cta_radiomics.carotid_qc import (
    MONTAGE_NAME,
    QC_JSON_NAME,
    CarotidQCError,
    main,
    run_carotid_qc,
)
from aorta_cta_radiomics.flowcat_adapter import DEFAULT_REQUIRED_OUTPUTS


def _write_nifti(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    description: bytes | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    if description is not None:
        image.header["descrip"] = description
    nib.save(image, path)
    return path


def _write_manifest(path: Path, *, complete: bool, extra: dict | None = None) -> Path:
    artifacts = []
    for name in DEFAULT_REQUIRED_OUTPUTS:
        present = complete or name == "segmentation"
        artifacts.append(
            {
                "name": name,
                "required": True,
                "present": present,
                "validation_status": "validated_file" if present else "missing",
                "path": f"/ignored/patient-name/{name}",
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "artifacts": artifacts,
        "missing_required_outputs": [] if complete else list(DEFAULT_REQUIRED_OUTPUTS[1:]),
    }
    payload.update(extra or {})
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_case(tmp_path: Path, *, complete_manifest: bool = False):
    shape = (8, 9, 10)
    affine = np.asarray(
        [[0.8, 0.0, 0.0, 10.0], [0.0, 1.1, 0.0, -20.0], [0.0, 0.0, 2.0, 30.0], [0, 0, 0, 1]]
    )
    grid = np.indices(shape)
    cta = (grid[0] * 20 + grid[1] * 5 + grid[2]).astype(np.float32)
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[2:6, 3:7, 2:8] = 1
    cta_path = _write_nifti(tmp_path / "cta.nii.gz", cta, affine)
    segmentation_path = _write_nifti(tmp_path / "segmentation.nii.gz", segmentation, affine)
    manifest_path = _write_manifest(tmp_path / "flowcat_manifest.json", complete=complete_manifest)
    return cta_path, segmentation_path, manifest_path, affine, segmentation


def test_binary_only_case_generates_honest_pending_qc(tmp_path):
    cta, segmentation, manifest, _, mask = _synthetic_case(tmp_path)
    outdir = tmp_path / "qc"

    report = run_carotid_qc(
        case_id="570",
        cta_path=cta,
        segmentation_path=segmentation,
        flowcat_manifest_path=manifest,
        outdir=outdir,
    )

    assert (outdir / MONTAGE_NAME).is_file()
    assert (outdir / MONTAGE_NAME).stat().st_size > 1_000
    assert (outdir / QC_JSON_NAME).is_file()
    saved = json.loads((outdir / QC_JSON_NAME).read_text())
    assert saved["case_id"] == "570"
    assert report["required_output_completeness"]["complete"] is False
    assert report["required_output_completeness"]["missing_required_outputs"] == sorted(
        DEFAULT_REQUIRED_OUTPUTS[1:]
    )
    assert report["segmentation_metrics"]["nonzero_voxels"] == int(mask.sum())
    assert report["named_artery_results"]["available"] is False
    assert report["named_artery_results"]["vessel_results"] == []
    assert report["named_artery_results"]["status"] == "not_generated_labelled_artery_overlay_unavailable"
    assert report["visual_review"] == {
        "status": "pending",
        "pending": True,
        "reviewed_by": None,
        "reviewed_at": None,
    }
    assert report["manual_review"]["required"] is True
    assert "required_flowcat_outputs_incomplete" in report["manual_review"]["reasons"]
    assert "NAMED_ARTERY_RESULTS_UNAVAILABLE" in {flag["code"] for flag in report["qc_flags"]}
    assert report["geometry"]["reference"] == "original_nifti_geometry"
    assert report["display"]["coordinate_convention"] == "canonical_RAS_plus"
    assert report["display"]["orientation"] == ["R", "A", "S"]
    selection = report["display"]["selection"]
    assert selection["method"] == "superior_arterial_context_occupied_slice_percentile"
    assert selection["superior_percentile"] == 0.85
    assert selection["percentile_basis"] == "unweighted_axial_slices_with_whole_tree_foreground"
    assert selection["anatomical_claim"] == "not_a_named_carotid_localization"
    assert selection["canonical_voxel_ijk"][2] == 6
    assert report["cta_display"]["selected_canonical_voxel_ijk"] == selection["canonical_voxel_ijk"]


def test_complete_manifest_and_all_optional_overlays_are_recorded_without_inferred_results(tmp_path):
    cta, segmentation, manifest, affine, mask = _synthetic_case(tmp_path, complete_manifest=True)
    labelled = np.zeros_like(mask, dtype=np.uint8)
    labelled[2:4, 3:7, 2:8] = 3
    labelled[4:6, 3:7, 2:8] = 9
    shell = (mask * 2).astype(np.uint8)
    tissue = (mask * 4).astype(np.uint8)
    calcium = np.zeros_like(mask, dtype=np.uint8)
    calcium[3, 4, 4] = 1
    paths = {
        "labelled_artery_path": _write_nifti(tmp_path / "labelled.nii.gz", labelled, affine),
        "shell_path": _write_nifti(tmp_path / "shell.nii.gz", shell, affine),
        "tissue_path": _write_nifti(tmp_path / "tissue.nii.gz", tissue, affine),
        "calcium_path": _write_nifti(tmp_path / "calcium.nii.gz", calcium, affine),
    }

    report = run_carotid_qc(
        case_id="synthetic",
        cta_path=cta,
        segmentation_path=segmentation,
        flowcat_manifest_path=manifest,
        outdir=tmp_path / "qc",
        focus_overlay="shell",
        **paths,
    )

    assert report["required_output_completeness"]["complete"] is True
    assert all(report["overlay_availability"].values())
    assert report["overlay_metrics"]["labelled_artery"]["nonzero_label_ids"] == [3, 9]
    assert report["overlay_metrics"]["labelled_artery"]["label_voxel_counts"] == {
        "3": 48,
        "9": 48,
    }
    assert report["overlay_metrics"]["calcium"]["nonzero_voxels"] == 1
    assert report["named_artery_results"]["available"] is False
    assert report["named_artery_results"]["status"] == (
        "label_map_available_but_named_measurements_not_generated"
    )
    assert report["named_artery_results"]["label_ids_available_for_visual_qc"] == [3, 9]
    selection = report["display"]["selection"]
    assert selection["method"] == "provided_overlay_foreground_voxel_nearest_physical_median"
    assert selection["focus_overlay"] == "shell"
    assert selection["anatomical_claim"] == "focus_from_provided_overlay_not_inferred_anatomy"
    assert shell[tuple(selection["canonical_voxel_ijk"])] != 0
    assert report["manual_review"]["required"] is True
    assert report["manual_review"]["reasons"] == ["visual_review_pending"]


def test_display_is_canonical_ras_for_permuted_and_flipped_nifti(tmp_path):
    shape = (10, 8, 9)
    # Voxel axes are S, L, P respectively; canonical display must reorder and flip.
    affine = np.asarray(
        [[0.0, -1.0, 0.0, 20.0], [0.0, 0.0, -1.5, 30.0], [2.0, 0.0, 0.0, -10.0], [0, 0, 0, 1]]
    )
    grid = np.indices(shape)
    cta_data = (grid[0] * 100 + grid[1] * 10 + grid[2]).astype(np.float32)
    segmentation_data = np.zeros(shape, dtype=np.uint8)
    segmentation_data[1:9, 2:4, 3:5] = 1
    cta = _write_nifti(tmp_path / "permuted_cta.nii.gz", cta_data, affine)
    segmentation = _write_nifti(tmp_path / "permuted_seg.nii.gz", segmentation_data, affine)
    manifest = _write_manifest(tmp_path / "flowcat_manifest.json", complete=False)

    report = run_carotid_qc(
        case_id="permuted",
        cta_path=cta,
        segmentation_path=segmentation,
        flowcat_manifest_path=manifest,
        outdir=tmp_path / "qc",
    )

    assert report["geometry"]["orientation"] == ["S", "L", "P"]
    assert report["geometry"]["shape_ijk"] == list(shape)
    assert report["display"]["orientation"] == ["R", "A", "S"]
    assert report["display"]["shape_ijk"] == [8, 9, 10]
    assert report["display"]["reorientation"] == "axis_permutation_and_flip_only_no_resampling"
    assert report["display"]["plane_direction_labels"] == {
        "axial": {"left": "L", "right": "R", "top": "A", "bottom": "P"},
        "coronal": {"left": "L", "right": "R", "top": "S", "bottom": "I"},
        "sagittal": {"left": "P", "right": "A", "top": "S", "bottom": "I"},
    }
    selection = report["display"]["selection"]
    assert selection["canonical_voxel_ijk"][2] == 7
    original_index = tuple(selection["original_voxel_ijk_nearest"])
    assert segmentation_data[original_index] == 1
    assert (tmp_path / "qc" / MONTAGE_NAME).is_file()


def test_superior_selection_uses_occupied_slice_extent_not_thoracic_voxel_mass(tmp_path):
    shape = (8, 9, 24)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    cta_data = np.indices(shape).sum(axis=0).astype(np.float32)
    segmentation_data = np.zeros(shape, dtype=np.uint8)
    # Large inferior/thoracic cross-sections dominate voxel count.
    segmentation_data[1:7, 1:8, 2:10] = 1
    # Small superior vessel context remains present through k=21.
    segmentation_data[2:4, 3:5, 10:22] = 1
    cta = _write_nifti(tmp_path / "cta.nii.gz", cta_data, affine)
    segmentation = _write_nifti(tmp_path / "segmentation.nii.gz", segmentation_data, affine)
    manifest = _write_manifest(tmp_path / "flowcat_manifest.json", complete=False)

    report = run_carotid_qc(
        case_id="head-to-heart",
        cta_path=cta,
        segmentation_path=segmentation,
        flowcat_manifest_path=manifest,
        outdir=tmp_path / "qc",
    )

    selection = report["display"]["selection"]
    assert selection["canonical_voxel_ijk"][2] == 18
    assert selection["selected_axial_foreground_voxels"] == 4
    assert selection["occupied_axial_slice_count"] == 20
    assert selection["anatomical_claim"] == "not_a_named_carotid_localization"


def test_shape_mismatch_is_rejected_before_outputs_are_created(tmp_path):
    cta, _, manifest, affine, _ = _synthetic_case(tmp_path)
    bad_segmentation = _write_nifti(
        tmp_path / "bad_shape.nii.gz", np.zeros((7, 9, 10), dtype=np.uint8), affine
    )
    outdir = tmp_path / "qc"

    with pytest.raises(CarotidQCError, match="shape"):
        run_carotid_qc(
            case_id="synthetic",
            cta_path=cta,
            segmentation_path=bad_segmentation,
            flowcat_manifest_path=manifest,
            outdir=outdir,
        )
    assert not outdir.exists()


def test_affine_mismatch_is_rejected(tmp_path):
    cta, _, manifest, affine, mask = _synthetic_case(tmp_path)
    shifted = affine.copy()
    shifted[0, 3] += 5.0
    bad_segmentation = _write_nifti(tmp_path / "bad_affine.nii.gz", mask, shifted)

    with pytest.raises(CarotidQCError, match="affine matrices differ"):
        run_carotid_qc(
            case_id="synthetic",
            cta_path=cta,
            segmentation_path=bad_segmentation,
            flowcat_manifest_path=manifest,
            outdir=tmp_path / "qc",
        )


def test_nonbinary_segmentation_is_rejected(tmp_path):
    cta, _, manifest, affine, mask = _synthetic_case(tmp_path)
    nonbinary = mask.copy()
    nonbinary[3, 4, 5] = 2
    bad_segmentation = _write_nifti(tmp_path / "nonbinary.nii.gz", nonbinary, affine)

    with pytest.raises(CarotidQCError, match="must be binary"):
        run_carotid_qc(
            case_id="synthetic",
            cta_path=cta,
            segmentation_path=bad_segmentation,
            flowcat_manifest_path=manifest,
            outdir=tmp_path / "qc",
        )


def test_optional_overlay_geometry_is_validated(tmp_path):
    cta, segmentation, manifest, affine, mask = _synthetic_case(tmp_path)
    shifted = affine.copy()
    shifted[1, 3] += 3.0
    shell = _write_nifti(tmp_path / "bad_shell.nii.gz", mask, shifted)

    with pytest.raises(CarotidQCError, match="shell overlay geometry does not match CTA"):
        run_carotid_qc(
            case_id="synthetic",
            cta_path=cta,
            segmentation_path=segmentation,
            flowcat_manifest_path=manifest,
            outdir=tmp_path / "qc",
            shell_path=shell,
        )


def test_descriptive_nifti_and_unrelated_manifest_metadata_are_not_reported(tmp_path):
    cta, segmentation, _, affine, _ = _synthetic_case(tmp_path)
    cta_data = np.asarray(nib.load(cta).dataobj)
    _write_nifti(cta, cta_data, affine, description=b"Patient Name MRN-123456")
    manifest = _write_manifest(
        tmp_path / "flowcat_manifest.json",
        complete=False,
        extra={"patient_name": "Must Not Escape", "medical_record_number": "MRN-123456"},
    )

    report = run_carotid_qc(
        case_id="570",
        cta_path=cta,
        segmentation_path=segmentation,
        flowcat_manifest_path=manifest,
        outdir=tmp_path / "qc",
    )
    serialized = json.dumps(report)

    assert "Patient Name" not in serialized
    assert "MRN-123456" not in serialized
    assert "Must Not Escape" not in serialized
    assert report["source_data_policy"]["descriptive_nifti_metadata_inspected"] is False
    assert report["source_data_policy"]["dicom_metadata_inspected"] is False


def test_headless_cli_accepts_required_arguments_and_emits_json(tmp_path, capsys):
    cta, segmentation, manifest, _, _ = _synthetic_case(tmp_path)
    outdir = tmp_path / "cli_qc"

    exit_code = main(
        [
            "--case-id",
            "570",
            "--cta",
            str(cta),
            "--segmentation",
            str(segmentation),
            "--flowcat-manifest",
            str(manifest),
            "--outdir",
            str(outdir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "qc_generated_pending_visual_review"
    assert payload["manual_review_required"] is True
    assert Path(payload["qc_json"]) == outdir / QC_JSON_NAME
    assert Path(payload["montage_png"]) == outdir / MONTAGE_NAME
    assert not captured.err


def test_cli_returns_two_for_geometry_failure(tmp_path, capsys):
    cta, _, manifest, affine, _ = _synthetic_case(tmp_path)
    bad = _write_nifti(tmp_path / "bad.nii.gz", np.zeros((3, 3, 3), dtype=np.uint8), affine)

    exit_code = main(
        [
            "--case-id",
            "570",
            "--cta",
            str(cta),
            "--segmentation",
            str(bad),
            "--flowcat-manifest",
            str(manifest),
            "--outdir",
            str(tmp_path / "qc"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error_type"] == "CarotidQCError"
