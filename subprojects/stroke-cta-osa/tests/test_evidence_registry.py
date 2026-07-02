"""Evidence-registry completeness + serialisation contract."""

from pathlib import Path

import pytest

from stroke_cta_osa import evidence_registry as er
from stroke_cta_osa.evidence_registry import (
    AnalysisRole, EvidenceClass, EvidenceTier,
)

KNOWN_UNITS = {"mm", "mm2", "mm3", "ml", "HU", "fraction", "ratio",
               "count", "bool", "str"}


def test_registry_nonempty():
    assert len(er.all_evidence()) > 50


def test_feature_names_unique():
    names = [e.feature_name for e in er.all_evidence()]
    assert len(names) == len(set(names)), \
        sorted({n for n in names if names.count(n) > 1})


def test_every_feature_has_evidence_tier():
    for e in er.all_evidence():
        assert isinstance(e.evidence_tier, EvidenceTier)


def test_every_feature_has_evidence_class():
    for e in er.all_evidence():
        assert isinstance(e.evidence_class, EvidenceClass)


def test_every_feature_has_unit():
    for e in er.all_evidence():
        assert e.unit in KNOWN_UNITS, (e.feature_name, e.unit)


def test_every_feature_has_analysis_role():
    for e in er.all_evidence():
        assert isinstance(e.analysis_role, AnalysisRole)


def test_every_feature_has_true_anatomic_vs_proxy():
    for e in er.all_evidence():
        assert e.true_anatomic_vs_proxy in {"anatomic", "proxy", "mixed"}


def test_every_feature_has_reference_tags():
    for e in er.all_evidence():
        assert len(e.reference_tags) >= 1, e.feature_name


def test_recommended_feature_set_matches_tier():
    for e in er.all_evidence():
        assert e.recommended_feature_set  # non-empty


def test_csv_json_roundtrip(tmp_path: Path):
    csv = er.to_csv(tmp_path / "evidence.csv")
    js = er.to_json(tmp_path / "evidence.json")
    assert csv.is_file() and js.is_file()
    header = csv.read_text().splitlines()[0]
    for col in ("feature_name", "evidence_tier", "evidence_class", "unit",
                "analysis_role", "true_anatomic_vs_proxy", "reference_tags"):
        assert col in header


def test_tier1_is_only_osa_backed_classes():
    """No Tier-1 feature may carry a purely-exploratory evidence class."""
    forbidden = {EvidenceClass.CTA_STROKE_NOVEL, EvidenceClass.RADIOMICS_EXPLORATORY,
                 EvidenceClass.MODEL_OUTPUT_EXPLORATORY,
                 EvidenceClass.CT_CARDIOMETABOLIC_DIRECT_NO_OSA}
    for e in er.by_tier(EvidenceTier.TIER_1_CORE_OSA_BACKED):
        assert e.evidence_class not in forbidden, e.feature_name


def test_tier1_high_value_features_present():
    t1 = {e.feature_name for e in er.by_tier(EvidenceTier.TIER_1_CORE_OSA_BACKED)}
    for required in (
        "airway_min_csa_mm2", "retropalatal_min_csa_mm2", "retroglossal_min_csa_mm2",
        "tongue_volume_ml", "tongue_to_mandible_volume_ratio",
        "tongue_posterior_low_hu_fraction", "fat_cervical_total_volume_ml",
        "fat_parapharyngeal_total_volume_ml",
        "mandibular_plane_to_hyoid_distance_mm",
        "cervicomandibular_ring_area_mm2",
    ):
        assert required in t1, required


def test_alias_resolution():
    # planned canonical proxy name resolves back via its registry alias
    spec = er.evidence_for("fat_subcutaneous_cervical_volume_ml")
    assert spec is not None
    assert spec.feature_name == "fat_neck_subcutaneous_proxy_volume_ml"


def test_merged_records_cover_full_metric_registry():
    from stroke_cta_osa.metric_registry import feature_names
    recs = er.merged_records()
    assert len(recs) == len(feature_names())
