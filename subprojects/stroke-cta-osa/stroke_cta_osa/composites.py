"""Exploratory composite indices.

These are **unvalidated** linear combinations of single-feature inputs.
They exist so analysts have a quick sanity-check signal per case; they are
NOT predictive models. Names end in ``_untrained`` to make the status
visible at every step of analysis.

Rules:
  * A composite is only computed when ALL its component features are
    available (i.e. not NaN). Otherwise the composite stays NaN.
  * When ``cfg.require_batch_standardization`` is True the composite is
    only emitted if cohort z-score statistics (means + stds) are supplied
    via the ``cohort_stats`` argument; otherwise it stays NaN. This forces
    the analyst to make an explicit choice about whether case-level
    standardisation has been done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import math


_NAN = float("nan")


@dataclass
class CompositeConfig:
    enabled: bool = False
    require_batch_standardization: bool = True
    suffix: str = "untrained"


COMPONENT_GROUPS: dict[str, tuple[str, ...]] = {
    "cta_osa_airway_narrowing_index_untrained": (
        "airway_min_csa_mm2",
        "retropalatal_min_csa_mm2",
        "retroglossal_min_csa_mm2",
        "airway_lateral_narrowing_index",
        "airway_length_mm",
    ),
    "cta_osa_tongue_crowding_index_untrained": (
        "tongue_volume_ml",
        "tongue_to_mandible_volume_ratio",
        "tongue_to_oral_cavity_volume_ratio",
        "tongue_base_to_retroglossal_airway_ratio",
        "tongue_posterior_low_hu_fraction",
    ),
    "cta_osa_fat_burden_index_untrained": (
        "fat_cervical_total_volume_ml",
        "fat_deep_cervical_volume_ml",
        "fat_parapharyngeal_total_volume_ml",
        "fat_retropharyngeal_volume_ml",
        "fat_parapharyngeal_to_airway_ratio_min_csa",
    ),
    "cta_osa_skeletal_restriction_index_untrained": (
        "mandibular_plane_to_hyoid_distance_mm",
        "cervicomandibular_ring_area_mm2",
        "hyoid_to_c3_distance_mm",
        "tongue_to_skeletal_enclosure_ratio",
    ),
}


# Direction: +1 means "higher value increases the OSA-anatomy signal",
# -1 means "higher value DECREASES the signal" (inverted before summing).
COMPONENT_DIRECTIONS: dict[str, int] = {
    "airway_min_csa_mm2": -1,
    "retropalatal_min_csa_mm2": -1,
    "retroglossal_min_csa_mm2": -1,
    "airway_lateral_narrowing_index": +1,
    "airway_length_mm": +1,
    "tongue_volume_ml": +1,
    "tongue_to_mandible_volume_ratio": +1,
    "tongue_to_oral_cavity_volume_ratio": +1,
    "tongue_base_to_retroglossal_airway_ratio": +1,
    "tongue_posterior_low_hu_fraction": +1,
    "fat_cervical_total_volume_ml": +1,
    "fat_deep_cervical_volume_ml": +1,
    "fat_parapharyngeal_total_volume_ml": +1,
    "fat_retropharyngeal_volume_ml": +1,
    "fat_parapharyngeal_to_airway_ratio_min_csa": +1,
    "mandibular_plane_to_hyoid_distance_mm": +1,
    "cervicomandibular_ring_area_mm2": -1,
    "hyoid_to_c3_distance_mm": +1,
    "tongue_to_skeletal_enclosure_ratio": +1,
}


@dataclass
class CohortStats:
    """Per-feature cohort mean + std for z-scoring composites.

    The structure matches what an upstream cohort-builder would persist
    (one row per feature_name, two columns: mean, std). When `std == 0`
    the contribution from that feature is skipped (it can't be z-scored).
    """
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)


def compute_composites(
    feature_row: dict[str, object],
    cfg: CompositeConfig,
    cohort_stats: Optional[CohortStats] = None,
) -> dict[str, object]:
    """Compute the five composite indices when conditions allow.

    Always returns a dict with the composite + provenance keys (NaN /
    explanation strings if disabled), so the caller can blindly merge.
    """
    out: dict[str, object] = {
        n: _NAN for n in COMPONENT_GROUPS
    }
    out["cta_osa_combined_anatomy_index_untrained"] = _NAN
    out["composite_score_method"] = "raw_linear_unstandardized_v2"
    out["composite_score_disclaimer"] = (
        "EXPLORATORY — these are not predictive models. Suffix _untrained is "
        "intentional. Do not use for clinical decisions."
    )

    if not cfg.enabled:
        out["composite_score_method"] = "disabled"
        return out

    use_zscore = cohort_stats is not None
    if cfg.require_batch_standardization and not use_zscore:
        out["composite_score_method"] = (
            "skipped_require_batch_standardization_and_no_cohort_stats"
        )
        return out

    if use_zscore:
        out["composite_score_method"] = "cohort_zscore_then_sum"

    component_scores: dict[str, float] = {}
    for composite, components in COMPONENT_GROUPS.items():
        score = _compute_one(
            components=components, feature_row=feature_row,
            use_zscore=use_zscore, cohort=cohort_stats,
        )
        out[composite] = score
        if isinstance(score, float) and score == score:
            component_scores[composite] = score

    # Combined index — average of the four if all are present
    if len(component_scores) == 4:
        out["cta_osa_combined_anatomy_index_untrained"] = round(
            sum(component_scores.values()) / len(component_scores), 4)
    return out


def _compute_one(
    *, components: Iterable[str],
    feature_row: dict[str, object],
    use_zscore: bool,
    cohort: Optional[CohortStats],
) -> float:
    parts: list[float] = []
    for name in components:
        v = feature_row.get(name)
        if not isinstance(v, (int, float)):
            return _NAN
        if isinstance(v, float) and v != v:
            return _NAN
        direction = COMPONENT_DIRECTIONS.get(name, +1)
        x = float(v)
        if use_zscore and cohort is not None:
            mu = cohort.means.get(name)
            sd = cohort.stds.get(name)
            if mu is None or sd is None or sd == 0:
                continue
            x = (x - mu) / sd
        parts.append(direction * x)
    if not parts:
        return _NAN
    return round(float(sum(parts) / len(parts)), 4)
