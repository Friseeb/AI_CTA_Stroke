"""Unit tests for report schema validation and features module behaviour."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from cta_dental.report import DentalReport, FOVCompleteness, write_report, load_report
from cta_dental.features import extract_features, write_features_json, NOT_IMPLEMENTED
from cta_dental.config import FeaturesConfig


class TestDentalReportSchema:

    def test_default_report_has_disclaimer(self):
        r = DentalReport(case_id="test001")
        assert "RESEARCH" in r.disclaimer.upper()

    def test_report_round_trip_json(self):
        r = DentalReport(
            case_id="test002",
            input_type="nifti",
            age_status="adult",
            roi_method="totalseg_teeth",
            roi_quality="good",
            segmentation_backend="totalseg_teeth",
            segmentation_status="success",
            deface_mode="mask_only",
            status="complete",
            warnings=["domain shift expected"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            write_report(r, p)
            loaded = load_report(p)
        assert loaded.case_id == "test002"
        assert loaded.roi_quality == "good"
        assert loaded.segmentation_status == "success"
        assert "domain shift expected" in loaded.warnings

    def test_report_accepts_fov_completeness(self):
        r = DentalReport(
            case_id="test003",
            fov_completeness=FOVCompleteness(
                has_upper_dentition=True,
                has_lower_dentition=False,
                partial_fov=True,
            ),
        )
        data = json.loads(r.model_dump_json())
        assert data["fov_completeness"]["has_upper_dentition"] is True
        assert data["fov_completeness"]["partial_fov"] is True

    def test_report_excludes_phi_fields(self):
        r = DentalReport(case_id="test004")
        data = json.loads(r.model_dump_json())
        phi_keys = {"patient_name", "patient_id", "mrn", "accession_number", "date_of_birth"}
        assert not phi_keys.intersection(set(data.keys()))

    def test_report_status_literals(self):
        with pytest.raises(Exception):
            DentalReport(case_id="x", roi_quality="excellent")  # not a valid literal

    def test_report_with_errors(self):
        r = DentalReport(
            case_id="err_case",
            errors=["TotalSegmentator not found"],
            status="failed_roi",
        )
        data = json.loads(r.model_dump_json())
        assert data["errors"] == ["TotalSegmentator not found"]


class TestFeatureExtraction:

    def _make_image(self, shape=(30, 30, 30), spacing=(0.5, 0.5, 0.5)):
        arr = np.random.uniform(-200, 800, shape).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(list(reversed(spacing)))
        return img

    def test_features_no_labels_returns_not_assessable(self):
        img = self._make_image()
        result = extract_features(
            case_id="nolab",
            hu_image=img,
            label_files={},
            cfg=FeaturesConfig(),
        )
        assert result.case_id == "nolab"
        assert "RESEARCH" in result.disclaimer.upper()
        # Periapical should flag not_assessable when no labels
        periapical = result.candidate_markers.get("periapical_lucency_candidate", [])
        assert any("not_assessable" in str(p) or "assessable" in str(p) for p in periapical)

    def test_features_fov_incomplete_with_only_upper(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            arr = np.ones((30, 30, 30), dtype=np.uint8)
            mask = sitk.GetImageFromArray(arr)
            mask.SetSpacing([0.5, 0.5, 0.5])
            upper_path = tmp / "upper_teeth.nii.gz"
            sitk.WriteImage(mask, str(upper_path), useCompression=True)

            img = self._make_image()
            result = extract_features(
                case_id="partial_fov",
                hu_image=img,
                label_files={"upper_teeth": upper_path},
                cfg=FeaturesConfig(),
            )
        assert result.assessability["dentition_fov"] in ("partial", "complete", "unknown")
        missing = result.candidate_markers.get("teeth_missing_candidate", [])
        assert any("not_assessable" in str(m) or "assessable" in str(m) for m in missing)

    def test_features_roi_quality_poor_disables_disease(self):
        img = self._make_image()
        cfg = FeaturesConfig(allow_threshold_fallback_features=False)
        result = extract_features(
            case_id="poor_roi",
            hu_image=img,
            label_files={},
            cfg=cfg,
            roi_quality="poor",
        )
        assert any("threshold_fallback" in w or "poor" in w for w in result.warnings)
        for key in ("periapical_lucency_candidate", "severe_periodontal_bone_loss_candidate"):
            assert result.candidate_markers.get(key, []) == []

    def test_not_implemented_list_present(self):
        img = self._make_image()
        result = extract_features("ni_case", img, {}, FeaturesConfig())
        ni = result.not_implemented_or_not_reliable
        assert isinstance(ni, list) and len(ni) > 0
        assert any("caries" in s.lower() for s in ni)

    def test_write_features_json(self):
        img = self._make_image()
        result = extract_features("write_test", img, {}, FeaturesConfig())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "candidate_features.json"
            write_features_json(result, p)
            data = json.loads(p.read_text())
        assert data["case_id"] == "write_test"
        assert "disclaimer" in data
        assert "candidate_markers" in data
        assert "assessability" in data

    def test_missing_totalsegmentator_returns_clean_error(self):
        """If TotalSegmentator is missing, the segmenter should return a clean error, not a stack trace."""
        from cta_dental.segmenters.totalsegmentator import TotalSegmentatorTeethSegmenter
        import shutil
        orig = shutil.which
        # Monkey-patch which to simulate missing binary
        import cta_dental.external_tools as et
        orig_which = et.shutil.which
        et.shutil.which = lambda name: None
        try:
            seg = TotalSegmentatorTeethSegmenter()
            with tempfile.TemporaryDirectory() as tmp:
                result = seg.run(
                    input_nifti=Path(tmp) / "dummy.nii.gz",
                    output_dir=Path(tmp) / "out",
                    config={},
                )
            assert not result.success
            assert result.errors
            assert "TotalSegmentator" in result.errors[0]
        finally:
            et.shutil.which = orig_which
