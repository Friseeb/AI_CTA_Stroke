"""No feature without units; metadata/registry/docs stay consistent."""

import pytest

from stroke_cta_osa import evidence_registry as er
from stroke_cta_osa import feature_sets as fs
from stroke_cta_osa import metric_registry as mr

KNOWN_UNITS = {"mm", "mm2", "mm3", "ml", "HU", "fraction", "ratio",
               "count", "bool", "str"}


def test_no_metric_without_units():
    bad = [m.feature_name for m in mr.all_metrics() if m.unit not in KNOWN_UNITS]
    assert not bad, bad


def test_no_evidence_feature_without_units():
    bad = [e.feature_name for e in er.all_evidence() if e.unit not in KNOWN_UNITS]
    assert not bad, bad


def test_confidence_fields_exist_as_columns():
    """Every confidence_field_name referenced by an evidence spec is a real
    registry column (so it can actually be populated/serialised)."""
    cols = set(mr.feature_names())
    missing = sorted({
        e.confidence_field_name for e in er.all_evidence()
        if e.confidence_field_name and e.confidence_field_name not in cols
    })
    assert not missing, missing


def test_every_tier1_feature_has_confidence_field():
    for e in er.by_tier(er.EvidenceTier.TIER_1_CORE_OSA_BACKED):
        assert e.confidence_field_name, e.feature_name


def test_implemented_evidence_features_are_registry_columns():
    """Each evidence feature flagged implemented must resolve to a real column
    (by canonical name or a known alias)."""
    cols = set(mr.feature_names())
    for e in er.all_evidence():
        if not e.implemented:
            continue
        ok = e.feature_name in cols or any(a in cols for a in e.aliases)
        assert ok, e.feature_name


def test_metric_registry_did_not_shrink():
    assert len(mr.all_metrics()) > 100


def test_feature_set_membership_in_metadata_records():
    """feature_metadata.json content is built from these helpers; ensure each
    set maps to a non-empty, tier-consistent membership list."""
    for name in fs.ALLOWED_FEATURE_SETS:
        feats = fs.evidence_features(name)
        assert feats
        tiers = set(fs.tiers_for(name))
        for fn in feats:
            spec = er.evidence_for(fn)
            assert spec is not None and spec.evidence_tier in tiers
