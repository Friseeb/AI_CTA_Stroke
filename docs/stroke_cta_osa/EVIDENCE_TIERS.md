# Evidence tiers

`stroke_cta_osa` classifies every feature on **two orthogonal axes**:

1. **Analysis tier** (`metric_registry.Tier`: `tier1` / `tier2` / `exploratory`)
   — an *engineering* statement about extraction robustness.
2. **Evidence tier** (`evidence_registry.EvidenceTier`) — a *scientific*
   statement about how strongly prior adult OSA imaging literature backs the
   feature. This page is about the second axis.

The evidence registry (`stroke_cta_osa/evidence_registry.py`) is the source of
truth. Export it with:

```bash
stroke-cta-osa list-features --format csv --out feature_dictionary.csv
stroke-cta-osa list-features --evidence-tier TIER_1_CORE_OSA_BACKED
```

## The four evidence tiers

| Tier | Conceptual summary |
|------|--------------------|
| **Tier 1 — `TIER_1_CORE_OSA_BACKED`** | Previously OSA-linked CT/CBCT/MRI metrics. Use for the **primary** CTA-OSA phenotype. |
| **Tier 2 — `TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC`** | CT/anatomy-supported but not established OSA CT biomarkers. Use for **secondary** mechanism discovery. |
| **Tier 3 — `TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR`** | CT cardiometabolic/vascular adiposity metrics. Use for **stroke / MACE / AF / AFDAS** risk models. |
| **Tier 4 — `TIER_4_STROKE_CTA_NOVEL_EXPLORATORY`** | Novel engineered CTA/stroke features, radiomics, untrained composites. Use for **exploratory** analysis only. |

## Evidence classes

`evidence_class` records *what kind* of prior evidence (or lack thereof) backs a
feature:

`OSA_CT_DIRECT`, `OSA_CBCT_DIRECT`, `OSA_MRI_DIRECT`, `OSA_IMAGING_INDIRECT`,
`CT_ANATOMY_DIRECT_NO_OSA`, `CT_CARDIOMETABOLIC_DIRECT_NO_OSA`,
`CTA_STROKE_NOVEL`, `ENGINEERED_PROXY`, `RADIOMICS_EXPLORATORY`,
`MODEL_OUTPUT_EXPLORATORY`.

## Analysis roles

`analysis_role` is the *permission* a feature carries into modelling:

`primary_candidate`, `secondary_candidate`, `mechanistic_secondary`,
`cardiometabolic_secondary`, `exploratory`, `do_not_model_without_validation`.

## Per-feature attributes

Each `EvidenceSpec` also records:

- **`true_anatomic_vs_proxy`** — `anatomic` when real masks/landmarks define the
  feature, `proxy` when it is a geometric approximation. Proxy features carry
  `_proxy` in the name or this flag; e.g. supra-/subplatysmal fat is a proxy
  unless a platysma mask exists.
- **`contrast_sensitive`** — HU-based values (tongue HU, fat HU, vessels) shift
  with CTA contrast phase.
- **`artifact_sensitive`** — dental/streak artifact degrades the measurement.
- **`confidence_field_name`** — which `*_confidence` column governs the feature
  (`high` / `moderate` / `low` / `missing`).
- **`missingness_behavior`** — how a missing value is represented (`NA`,
  `bool_False`, `empty_str`, `-1_int`).
- **`reference_tags`** — the supporting literature (see the reference table in
  the [README](README.md)).

## Overlap with the dental/CBCT pipeline

Airway, tongue, mandible, and MP-H features can be **shared with the dental
pipeline** (`metric_registry.MetricSpec.shared_with_dental`). Cervical,
parapharyngeal, retropharyngeal, submandibular, periairway, and C5 fat
compartments are **stroke-CTA-specific** because they sit below the typical
CBCT field of view. The evidence axis does not change this: a Tier-1 airway
feature may be shared, while a Tier-1 cervical-fat feature is CTA-specific.
