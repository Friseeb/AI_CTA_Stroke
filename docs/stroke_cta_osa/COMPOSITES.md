# Composite indices (exploratory, untrained)

The composites module
([stroke_cta_osa/composites.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/composites.py))
emits four sub-composite indices and a combined index that average together
features across modalities. **They are not predictive models.** Every name
ends in `_untrained` and the disclaimer string accompanies every row.

## Names and components

| Composite | Components |
|---|---|
| `cta_osa_airway_narrowing_index_untrained` | `airway_min_csa_mm2`, `retropalatal_min_csa_mm2`, `retroglossal_min_csa_mm2`, `airway_lateral_narrowing_index`, `airway_length_mm` |
| `cta_osa_tongue_crowding_index_untrained` | `tongue_volume_ml`, `tongue_to_mandible_volume_ratio`, `tongue_to_oral_cavity_volume_ratio`, `tongue_base_to_retroglossal_airway_ratio`, `tongue_posterior_low_hu_fraction` |
| `cta_osa_fat_burden_index_untrained` | `fat_cervical_total_volume_ml`, `fat_deep_cervical_volume_ml`, `fat_parapharyngeal_total_volume_ml`, `fat_retropharyngeal_volume_ml`, `fat_parapharyngeal_to_airway_ratio_min_csa` |
| `cta_osa_skeletal_restriction_index_untrained` | `mandibular_plane_to_hyoid_distance_mm`, `cervicomandibular_ring_area_mm2`, `hyoid_to_c3_distance_mm`, `tongue_to_skeletal_enclosure_ratio` |
| `cta_osa_combined_anatomy_index_untrained` | unweighted mean of the four sub-composites, only when all four are present |

## Component direction signs

Every component carries an explicit `+1` or `-1` in `COMPONENT_DIRECTIONS`
controlling how it enters the linear sum:

* `+1` — higher value increases the OSA-anatomy signal (e.g. larger tongue
  volume, more fat).
* `-1` — higher value DECREASES it (e.g. larger airway CSA, larger
  cervicomandibular ring area).

The test suite enforces that every listed component has an explicit sign so
default `+1` can never silently override the contract.

## Required gating

By default `enabled=False` — composites do **not** appear in output rows
unless the analyst opts in via `composites.enabled = true`.

When enabled:

* **`require_batch_standardization=True`** (default): the composite is only
  emitted if `cohort_stats` (per-feature mean + std across the cohort) is
  supplied. Otherwise `composite_score_method` returns
  `skipped_require_batch_standardization_and_no_cohort_stats`.
* **`require_batch_standardization=False`**: raw linear sum without z-scoring.
  The method string is `raw_linear_unstandardized_v2`. Use only for ablation
  / debugging — raw composites are not unit-consistent across modules.

A composite is NaN whenever **any** of its components is NaN or missing. This
is intentional: partial composites mix apples and oranges and would silently
confuse downstream analysis.

The combined index requires all four sub-composites to be present.

## Cohort statistics

`CohortStats(means, stds)` is the only input that connects a single case to a
cohort. The orchestrator picks it up from `--cohort-stats-json` when
configured. Stds of zero are treated as missing (the component is skipped, not
NaN'd) so that a hand-built test cohort with a singleton column still
processes the rest cleanly.

## Disclaimer column

`composite_score_disclaimer` is filled with the string

> EXPLORATORY — these are not predictive models. Suffix _untrained is
> intentional. Do not use for clinical decisions.

regardless of input — so downstream consumers always see it in the row.
