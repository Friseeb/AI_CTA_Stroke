"""QC behaviour: never crash, always emit a full row, respect coverage requirements."""

import math

from stroke_cta_osa.config import CoverageRequirements, QCConfig
from stroke_cta_osa.qc import qc_to_row, run_qc
from stroke_cta_osa.shared_schema import SharedAirwayLandmarks
from stroke_cta_osa.types import AirwayMaskInfo


def test_qc_pass_on_synthetic(synth_cta):
    mask = (synth_cta.array == -800).astype(bool)
    info = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                          confidence="medium", notes="")
    res = run_qc(synth_cta, CoverageRequirements(), QCConfig(),
                 info, SharedAirwayLandmarks())
    assert res.qc_pass is True
    assert res.has_upper_airway_region is True
    assert res.has_cervical_soft_tissue is True
    assert 0 <= res.qc_coverage_score <= 1


def test_qc_required_landmark_fails(synth_cta):
    mask = (synth_cta.array == -800).astype(bool)
    info = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                          confidence="medium", notes="")
    cov = CoverageRequirements(include_hyoid="required")
    res = run_qc(synth_cta, cov, QCConfig(), info, SharedAirwayLandmarks())
    assert res.qc_pass is False
    assert any("hyoid" in r.lower() for r in res.qc_failure_reasons)


def test_qc_row_has_stable_keys(synth_cta):
    res = run_qc(synth_cta, CoverageRequirements(), QCConfig(),
                 None, SharedAirwayLandmarks())
    row = qc_to_row(res)
    for k in ("qc_pass", "qc_coverage_score", "qc_has_upper_airway",
              "qc_has_cervical_soft_tissue", "qc_spacing_x_mm",
              "qc_z_extent_mm"):
        assert k in row
    # NaN-safe
    assert isinstance(row["qc_spacing_x_mm"], float)


def test_qc_truncation_flag_default_off(synth_cta):
    res = run_qc(synth_cta, CoverageRequirements(), QCConfig(),
                 None, SharedAirwayLandmarks())
    assert res.truncation_flag is False
