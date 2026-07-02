"""Feature-set membership contract: tier gating must be exact."""

import pytest

from stroke_cta_osa import evidence_registry as er
from stroke_cta_osa import feature_sets as fs
from stroke_cta_osa.evidence_registry import EvidenceTier


def _names_for(tier):
    return {e.feature_name for e in er.by_tier(tier)}


T1 = _names_for(EvidenceTier.TIER_1_CORE_OSA_BACKED)
T2 = _names_for(EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC)
T3 = _names_for(EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR)
T4 = _names_for(EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY)


def test_list_feature_sets():
    assert fs.list_feature_sets() == [
        "core_osa_backed", "core_plus_anatomic_extensions",
        "core_plus_cardiometabolic_ct", "all_features_exploratory",
    ]


def test_core_is_tier1_only():
    core = set(fs.evidence_features("core_osa_backed"))
    assert core == T1
    assert not (core & T2)
    assert not (core & T3)
    assert not (core & T4)


def test_core_plus_anatomic_is_tier1_and_tier2_only():
    s = set(fs.evidence_features("core_plus_anatomic_extensions"))
    assert s == (T1 | T2)
    assert not (s & T3)
    assert not (s & T4)


def test_core_plus_cardiometabolic_is_tier1_and_tier3_only():
    s = set(fs.evidence_features("core_plus_cardiometabolic_ct"))
    assert s == (T1 | T3)
    assert not (s & T2)
    assert not (s & T4)


def test_all_features_contains_all_tiers():
    s = set(fs.evidence_features("all_features_exploratory"))
    assert s == (T1 | T2 | T3 | T4)


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        fs.tiers_for("not_a_set")


def test_support_columns_are_not_evidence_features():
    support = set(fs.support_columns())
    evidence = {e.feature_name for e in er.all_evidence()}
    assert not (support & evidence)


def test_subset_columns_includes_planned_as_na():
    # A Tier-2 planned ratio should still be listed as a column for the
    # anatomic subset even if the available frame doesn't contain it.
    cols = fs.subset_columns("core_plus_anatomic_extensions",
                             available=["patient_id"])
    assert "fat_periairway_to_airway_volume_ratio" in cols


def test_membership_table_shape():
    rows = fs.membership_table()
    assert len(rows) == len(er.all_evidence())
    for r in rows:
        assert set(fs.ALLOWED_FEATURE_SETS).issubset(r.keys())
