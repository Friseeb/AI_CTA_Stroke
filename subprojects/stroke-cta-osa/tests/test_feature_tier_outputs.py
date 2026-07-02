"""Tiered subset CSV outputs: core stays clean, exploratory has everything."""

from pathlib import Path

import pandas as pd
import pytest

from stroke_cta_osa import evidence_registry as er
from stroke_cta_osa import feature_sets as fs
from stroke_cta_osa.evidence_registry import EvidenceTier
from stroke_cta_osa.output import write_outputs
from stroke_cta_osa.types import CaseResult


def _one_result() -> CaseResult:
    return CaseResult(
        identifiers={"patient_id": "sub-1", "pipeline": "stroke_cta_osa"},
        qc={"qc_pass": True},
        airway={"airway_min_csa_mm2": 55.0},
        fat={"fat_cervical_total_volume_ml": 12.3,
             "fat_retropharyngeal_volume_ml": 2.1},
        composite={"cta_osa_fat_burden_index_untrained": 9.9},
    )


@pytest.fixture()
def out_dir(tmp_path) -> Path:
    write_outputs([_one_result()], tmp_path)
    return tmp_path


def test_all_subset_files_written(out_dir):
    for f in (
        "features.csv", "features_core_osa_backed.csv",
        "features_core_plus_anatomic_extensions.csv",
        "features_core_plus_cardiometabolic_ct.csv",
        "features_all_exploratory.csv",
        "feature_evidence_summary.csv", "feature_missingness_by_tier.csv",
        "feature_metadata.json",
    ):
        assert (out_dir / f).is_file(), f


def test_core_has_no_tier234_features(out_dir):
    core = set(pd.read_csv(out_dir / "features_core_osa_backed.csv").columns)
    t2 = {e.feature_name for e in er.by_tier(EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC)}
    t3 = {e.feature_name for e in er.by_tier(EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR)}
    t4 = {e.feature_name for e in er.by_tier(EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY)}
    assert not (core & t2)
    assert not (core & t3)
    assert not (core & t4)


def test_core_contains_tier1(out_dir):
    core = set(pd.read_csv(out_dir / "features_core_osa_backed.csv").columns)
    assert "fat_cervical_total_volume_ml" in core
    assert "airway_min_csa_mm2" in core


def test_exploratory_contains_all_implemented(out_dir):
    allc = set(pd.read_csv(out_dir / "features_all_exploratory.csv").columns)
    assert "fat_retropharyngeal_volume_ml" in allc          # Tier 2
    assert "cta_osa_fat_burden_index_untrained" in allc      # Tier 4
    assert "fat_cervical_total_volume_ml" in allc            # Tier 1


def test_missing_optional_features_are_na_not_absent(out_dir):
    """Planned Tier-2 features appear as NA columns in the anatomic subset."""
    anat = pd.read_csv(out_dir / "features_core_plus_anatomic_extensions.csv")
    assert "fat_periairway_to_airway_volume_ratio" in anat.columns
    assert anat["fat_periairway_to_airway_volume_ratio"].isna().all()


def test_missingness_by_tier_rows(out_dir):
    df = pd.read_csv(out_dir / "feature_missingness_by_tier.csv")
    assert set(df["evidence_tier"]) == {t.value for t in EvidenceTier}
    for col in ("n_features", "n_available", "n_missing", "percent_missing"):
        assert col in df.columns


def test_evidence_summary_columns(out_dir):
    df = pd.read_csv(out_dir / "feature_evidence_summary.csv")
    for col in ("feature_name", "evidence_tier", "evidence_class", "feature_set",
                "analysis_role", "prior_osa_link", "prior_ct_link",
                "true_anatomic_vs_proxy", "contrast_sensitive",
                "confidence_field_name", "reference_tags"):
        assert col in df.columns


def test_tier4_disabled_does_not_break_tier1(out_dir):
    """Tier-1 output remains intact regardless of Tier-4 presence."""
    core = pd.read_csv(out_dir / "features_core_osa_backed.csv")
    assert core.loc[0, "fat_cervical_total_volume_ml"] == 12.3
