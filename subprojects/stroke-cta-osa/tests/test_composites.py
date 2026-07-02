"""Composite indices — exploratory, untrained.

These tests pin down:
  * disabled config → every composite is NaN and method=='disabled';
  * require_batch_standardization + no cohort_stats → skip everything with
    the explicit method='skipped_require_batch_standardization_and_no_cohort_stats';
  * Any NaN component blocks the whole composite (no partial credit);
  * Cohort z-scoring applies per-component direction signs;
  * Combined index is the mean of the four sub-composites only when all four
    are present;
  * Disclaimer text always appears regardless of inputs (it's a permanent flag).

We don't pin specific numeric values for composites — the choice of feature
list / weights is exploratory by design.
"""

from __future__ import annotations

import math

import pytest

from stroke_cta_osa.composites import (
    CohortStats, CompositeConfig, COMPONENT_DIRECTIONS, COMPONENT_GROUPS,
    compute_composites, _compute_one,
)


_NAN = float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_with_all_components_finite() -> dict[str, object]:
    """Build a feature_row covering every component listed in every
    COMPONENT_GROUPS entry, with deterministic synthetic finite values."""
    seen: set[str] = set()
    row: dict[str, object] = {}
    for components in COMPONENT_GROUPS.values():
        for name in components:
            if name in seen:
                continue
            seen.add(name)
            row[name] = 1.0
    return row


def _cohort_stats_for(row: dict[str, object],
                      mean: float = 0.5, std: float = 0.5) -> CohortStats:
    return CohortStats(
        means={k: mean for k in row},
        stds={k: std for k in row},
    )


# ---------------------------------------------------------------------------
# Disabled / disclaimer
# ---------------------------------------------------------------------------

def test_disabled_returns_nan_and_method_disabled():
    out = compute_composites({}, CompositeConfig(enabled=False))
    assert out["composite_score_method"] == "disabled"
    for name in COMPONENT_GROUPS:
        assert math.isnan(out[name])
    assert math.isnan(out["cta_osa_combined_anatomy_index_untrained"])
    assert "EXPLORATORY" in out["composite_score_disclaimer"]


def test_disclaimer_present_even_when_enabled():
    row = _row_with_all_components_finite()
    cohort = _cohort_stats_for(row)
    out = compute_composites(row, CompositeConfig(enabled=True), cohort)
    assert "EXPLORATORY" in out["composite_score_disclaimer"]
    assert "_untrained is" in out["composite_score_disclaimer"]


# ---------------------------------------------------------------------------
# require_batch_standardization gate
# ---------------------------------------------------------------------------

def test_require_batch_standardization_without_cohort_skips_everything():
    row = _row_with_all_components_finite()
    out = compute_composites(
        row, CompositeConfig(enabled=True, require_batch_standardization=True),
    )
    assert out["composite_score_method"] == \
        "skipped_require_batch_standardization_and_no_cohort_stats"
    for name in COMPONENT_GROUPS:
        assert math.isnan(out[name])


def test_require_batch_standardization_with_cohort_computes_composites():
    row = _row_with_all_components_finite()
    cohort = _cohort_stats_for(row)
    out = compute_composites(
        row, CompositeConfig(enabled=True, require_batch_standardization=True),
        cohort_stats=cohort,
    )
    assert out["composite_score_method"] == "cohort_zscore_then_sum"
    # All four sub-composites should be finite
    for name in COMPONENT_GROUPS:
        assert isinstance(out[name], float) and not math.isnan(out[name])


def test_no_batch_standardization_required_allows_raw_sum():
    row = _row_with_all_components_finite()
    out = compute_composites(
        row,
        CompositeConfig(enabled=True, require_batch_standardization=False),
    )
    assert out["composite_score_method"] == "raw_linear_unstandardized_v2"
    for name in COMPONENT_GROUPS:
        assert isinstance(out[name], float) and not math.isnan(out[name])


# ---------------------------------------------------------------------------
# Missing-component handling
# ---------------------------------------------------------------------------

def test_any_nan_component_blocks_composite():
    """If a single component is NaN, the whole composite stays NaN."""
    row = _row_with_all_components_finite()
    # Knock out one airway-narrowing component
    row["airway_min_csa_mm2"] = _NAN
    cohort = _cohort_stats_for(row)
    out = compute_composites(row, CompositeConfig(enabled=True), cohort)
    assert math.isnan(out["cta_osa_airway_narrowing_index_untrained"])
    # Others should still work
    assert not math.isnan(out["cta_osa_tongue_crowding_index_untrained"])


def test_any_missing_component_blocks_composite():
    """If a component key is absent from the row, the composite stays NaN."""
    row = _row_with_all_components_finite()
    row.pop("tongue_volume_ml", None)
    cohort = _cohort_stats_for(row)
    out = compute_composites(row, CompositeConfig(enabled=True), cohort)
    assert math.isnan(out["cta_osa_tongue_crowding_index_untrained"])


# ---------------------------------------------------------------------------
# Combined index
# ---------------------------------------------------------------------------

def test_combined_index_only_when_all_four_present():
    row = _row_with_all_components_finite()
    cohort = _cohort_stats_for(row)
    out = compute_composites(row, CompositeConfig(enabled=True), cohort)
    sub_scores = [out[name] for name in COMPONENT_GROUPS
                  if isinstance(out[name], float) and not math.isnan(out[name])]
    assert len(sub_scores) == 4
    assert out["cta_osa_combined_anatomy_index_untrained"] == pytest.approx(
        sum(sub_scores) / 4.0, abs=1e-3)


def test_combined_index_nan_when_any_subcomposite_blocked():
    row = _row_with_all_components_finite()
    row["airway_min_csa_mm2"] = _NAN  # blocks airway sub-composite
    cohort = _cohort_stats_for(row)
    out = compute_composites(row, CompositeConfig(enabled=True), cohort)
    assert math.isnan(out["cta_osa_combined_anatomy_index_untrained"])


# ---------------------------------------------------------------------------
# Direction signs
# ---------------------------------------------------------------------------

def test_component_directions_cover_every_component():
    """Every component listed in any COMPONENT_GROUP must have an explicit
    direction sign — otherwise the default +1 silently kicks in and that's
    a bug we'd rather catch at the contract level."""
    for components in COMPONENT_GROUPS.values():
        for name in components:
            assert name in COMPONENT_DIRECTIONS, \
                f"COMPONENT_DIRECTIONS missing {name}"
            assert COMPONENT_DIRECTIONS[name] in (-1, +1)


def test_direction_sign_inverts_contribution():
    """If we flip airway_min_csa_mm2 (direction=-1) above the cohort mean,
    its contribution to airway narrowing should DECREASE the sub-composite."""
    row = _row_with_all_components_finite()
    # row has all values at 1.0, cohort mean at 0.5, std 0.5 → each z = +1
    cohort = _cohort_stats_for(row, mean=0.5, std=0.5)
    out_base = compute_composites(row, CompositeConfig(enabled=True), cohort)
    base = out_base["cta_osa_airway_narrowing_index_untrained"]

    # Now move the min_csa even higher above mean → direction -1 → bigger
    # negative contribution → composite goes DOWN
    row_high = dict(row)
    row_high["airway_min_csa_mm2"] = 10.0  # z = (10 - 0.5) / 0.5 = 19
    out_high = compute_composites(row_high, CompositeConfig(enabled=True), cohort)
    high = out_high["cta_osa_airway_narrowing_index_untrained"]
    assert high < base, f"expected composite to decrease, got {high} vs {base}"


# ---------------------------------------------------------------------------
# _compute_one internals
# ---------------------------------------------------------------------------

def test_compute_one_returns_nan_on_first_missing():
    out = _compute_one(
        components=("a", "b", "c"),
        feature_row={"a": 1.0, "b": float("nan"), "c": 3.0},
        use_zscore=False, cohort=None,
    )
    assert math.isnan(out)


def test_compute_one_zero_std_is_skipped_not_failure():
    """A component with std=0 in cohort_stats cannot be z-scored — it should
    be skipped, not crash."""
    cohort = CohortStats(
        means={"a": 1.0, "b": 1.0},
        stds={"a": 0.0, "b": 1.0},
    )
    out = _compute_one(
        components=("a", "b"),
        feature_row={"a": 5.0, "b": 5.0},
        use_zscore=True, cohort=cohort,
    )
    # 'a' is skipped, only 'b' contributes
    assert not math.isnan(out)


def test_compute_one_returns_nan_when_no_components_present():
    cohort = CohortStats(means={"a": 1.0}, stds={"a": 0.0})
    out = _compute_one(
        components=("a",),
        feature_row={"a": 5.0},
        use_zscore=True, cohort=cohort,
    )
    # std=0 → skipped → no parts → NaN
    assert math.isnan(out)


# ---------------------------------------------------------------------------
# Contract: every composite name ends in _untrained
# ---------------------------------------------------------------------------

def test_every_composite_name_carries_untrained_suffix():
    for name in COMPONENT_GROUPS:
        assert name.endswith("_untrained"), \
            f"composite {name!r} must end in '_untrained'"
