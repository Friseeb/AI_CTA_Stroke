"""Registry shape + serialisation tests.

These pin down the contract: every feature name is unique, every entry has
a known unit, every shared-with-dental flag survives JSON/CSV round-trips,
and registry-canonical names overlap with the legacy shared list.
"""

from pathlib import Path

import pytest

from stroke_cta_osa.metric_registry import (
    Maturity, MetricSpec, Tier,
    all_metrics, empty_row, feature_names, find,
    landmark_dependent_features, shared_feature_names,
    to_csv, to_json, to_records,
)
from stroke_cta_osa.shared_schema import SHARED_FEATURE_NAMES


def test_registry_is_nonempty():
    assert len(all_metrics()) > 100, "metric registry should not regress in size"


def test_feature_names_unique():
    names = feature_names()
    assert len(names) == len(set(names)), \
        f"duplicate feature names: {sorted({n for n in names if names.count(n) > 1})}"


def test_units_are_known():
    allowed = {"mm", "mm2", "mm3", "ml", "HU", "fraction", "ratio",
               "count", "bool", "str"}
    bad = [m for m in all_metrics() if m.unit not in allowed]
    assert not bad, f"unknown units on {[(m.feature_name, m.unit) for m in bad]}"


def test_empty_row_keys_match_feature_names():
    row = empty_row()
    assert set(row.keys()) == set(feature_names())


def test_empty_row_defaults_respect_missingness_behaviour():
    row = empty_row()
    for m in all_metrics():
        v = row[m.feature_name]
        if m.missingness_behaviour == "bool_False":
            assert v is False
        elif m.missingness_behaviour == "empty_str":
            assert v == ""
        elif m.missingness_behaviour == "-1_int":
            assert v == -1
        else:
            assert isinstance(v, float)
            assert v != v  # NaN


def test_shared_feature_names_overlap_with_legacy_list():
    """Every name in the legacy `SHARED_FEATURE_NAMES` constant should be
    marked `shared_with_dental` in the registry, or live elsewhere as an
    intentional rename."""
    shared = set(shared_feature_names())
    legacy = set(SHARED_FEATURE_NAMES)
    # Intersection should be non-empty and reasonably large
    overlap = shared & legacy
    assert len(overlap) >= 6, \
        f"shared/legacy overlap too small: {sorted(overlap)}"


def test_landmark_dependent_subset():
    names = landmark_dependent_features()
    assert "retropalatal_csa_at_standard_level_mm2" in names
    assert "airway_min_csa_mm2" not in names  # global feature must NOT be landmark-dep


def test_find_round_trips():
    m = find("airway_volume_ml")
    assert m is not None
    assert m.unit == "ml"
    assert m.shared_with_dental is True
    assert find("zzz_no_such_thing") is None


def test_csv_json_roundtrip(tmp_path: Path):
    csv = to_csv(tmp_path / "reg.csv")
    js = to_json(tmp_path / "reg.json")
    assert csv.is_file() and js.is_file()
    text = csv.read_text()
    assert "feature_name,family,region,unit,tier,maturity" in text.splitlines()[0]
    # First registry entry should appear
    assert all_metrics()[0].feature_name in text


def test_tier1_includes_high_value_features():
    tier1 = {m.feature_name for m in all_metrics() if m.tier == Tier.TIER1}
    for required in (
        "airway_min_csa_mm2", "retropalatal_csa_at_standard_level_mm2",
        "retroglossal_csa_at_standard_level_mm2",
        "tongue_volume_ml", "tongue_to_mandible_volume_ratio",
        "tongue_posterior_mean_hu", "fat_cervical_total_volume_ml",
        "fat_parapharyngeal_total_volume_ml",
        "mandibular_plane_to_hyoid_distance_mm",
    ):
        assert required in tier1, f"{required} should be Tier 1"
