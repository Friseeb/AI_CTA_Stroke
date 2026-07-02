# Feature sets

Four canonical, evidence-gated feature sets select columns purely by
[evidence tier](EVIDENCE_TIERS.md). They are defined in
`stroke_cta_osa/feature_sets.py` and drive the `--feature-set` CLI flag and the
tiered subset CSVs.

| Feature set | Tiers included | Purpose |
|-------------|----------------|---------|
| `core_osa_backed` | Tier 1 | **Primary** analysis set — only features with prior OSA imaging support. |
| `core_plus_anatomic_extensions` | Tier 1 + Tier 2 | Secondary, anatomy-grounded mechanism discovery. |
| `core_plus_cardiometabolic_ct` | Tier 1 + Tier 3 | Stroke / MACE / AF / AFDAS / metabolic-risk modelling. |
| `all_features_exploratory` | Tier 1-4 | Hypothesis generation, ML, radiomics, future manuscripts. |

## Guarantees

- `core_osa_backed` contains **no** Tier 2/3/4 feature.
- `core_plus_anatomic_extensions` contains **only** Tier 1 + Tier 2.
- `core_plus_cardiometabolic_ct` contains **only** Tier 1 + Tier 3.
- `all_features_exploratory` contains every implemented feature.
- Identifier and QC columns are *support* columns appended to every subset; they
  are never evidence features, so they never violate the "core is clean" rule.
- Missing optional/planned features appear as **NA** columns, not absent
  columns — the schema is stable across cohorts.

## Selecting a feature set

```bash
# write a Tier-1-only feature dictionary
stroke-cta-osa list-features --feature-set core_osa_backed --out core_dict.csv

# record the default modelling set for a run (all subset CSVs are still written)
stroke-cta-osa extract case.nii.gz --out out/ --feature-set core_osa_backed

# per-tier completeness on an existing features.csv
stroke-cta-osa summarize out/features.csv --by-evidence-tier
```

Feature-set selection affects the **subset outputs and the recorded default
modelling set**, not the canonical registry: the pipeline always computes every
implemented feature, and `features.csv` always carries all stable columns.

## Output files

| File | Feature set |
|------|-------------|
| `features.csv` | everything (canonical) |
| `features_core_osa_backed.csv` | `core_osa_backed` |
| `features_core_plus_anatomic_extensions.csv` | `core_plus_anatomic_extensions` |
| `features_core_plus_cardiometabolic_ct.csv` | `core_plus_cardiometabolic_ct` |
| `features_all_exploratory.csv` | `all_features_exploratory` |
| `feature_metadata.json` | full registry + evidence metadata + set membership |
| `feature_evidence_summary.csv` | per-feature evidence provenance |
| `feature_missingness_by_tier.csv` | per-tier availability/missingness |

## Configuration

`configs/default.yaml` exposes:

```yaml
feature_selection:
  output_all_features: true
  output_feature_sets: true
  default_modeling_feature_set: core_osa_backed
  allowed_feature_sets: [core_osa_backed, core_plus_anatomic_extensions,
                         core_plus_cardiometabolic_ct, all_features_exploratory]

evidence_tiers:
  include_tier_1_core_osa_backed: true   # always available
  include_tier_2_osa_plausible_ct_anatomic: true
  include_tier_3_ct_cardiometabolic_or_vascular: true
  include_tier_4_stroke_cta_novel_exploratory: true
```

Disabling Tier 4 does not affect Tier 1 extraction. The full feature-set
specification (including fat HU thresholds and compartment toggles) is mirrored
in `configs/stroke_cta_osa_feature_sets.yaml`.
