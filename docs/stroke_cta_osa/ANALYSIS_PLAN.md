# Analysis plan

This plan operationalises the [evidence tiers](EVIDENCE_TIERS.md) and
[feature sets](FEATURE_SETS.md) into a staged statistical strategy. It is a
*research* plan: the pipeline produces features; this document describes how a
downstream analyst should use them without contaminating the primary phenotype.

## Stage 1 — Primary

Model the **`core_osa_backed`** (Tier 1) feature set against sleep-study OSA
labels, after a clinical-variables-only baseline.

- Baseline: clinical variables only (age, sex, BMI, neck circumference if
  available, hypertension, AF).
- Primary imaging model: clinical + `core_osa_backed`.
- Endpoints (where sleep data exist):
  - AHI/REI ≥ 15 (moderate-to-severe)
  - AHI/REI ≥ 30 (severe)
  - ODI
  - hypoxic burden
  - minimum SpO₂
  - T90

Pre-register the Tier-1 feature list. Do **not** add Tier 2/3/4 features here.

## Stage 2 — Secondary

Use Tier 1 (and pre-specified Tier 2 mechanistic features) against
stroke-relevant endpoints:

- wake-up stroke vs non-wake-up stroke
- AF / AF-detected-after-stroke (AFDAS)
- ESUS / cardioembolic phenotype
- PFO / right-to-left-shunt interaction
- small-vessel-disease burden
- recurrent stroke / MACE / mortality

The **`core_plus_cardiometabolic_ct`** set (Tier 1 + Tier 3 C5 NAT / pericarotid
fat) is appropriate for AF/MACE/stroke-risk models, framed as cardiometabolic
risk rather than OSA anatomy.

## Stage 3 — Exploratory

Use **`all_features_exploratory`** (Tier 2-4) for hypothesis generation only:

- Tier 2 anatomic extensions (retropharyngeal/submandibular/periairway fat,
  soft palate, lateral wall)
- Tier 4 engineered ratios and **untrained** composite scores
  (`*_untrained` — never clinical)
- radiomics
- pericarotid / cardiometabolic CT features as mechanism probes

All Stage-3 findings require independent validation before any modelling claim.

## Handling missingness and confidence

- Filter on the per-feature `*_confidence` field (`high`/`moderate`/`low`/
  `missing`) and on `feature_missingness_by_tier.csv` before modelling.
- Account for contrast phase: `contrast_sensitive` HU features should be
  adjusted for, or stratified by, contrast status.
- Proxy features (`true_anatomic_vs_proxy='proxy'`) should not be interpreted as
  true anatomy without the corresponding mask.

## What NOT to do

- Do not move a Tier 2/3/4 feature into the primary model post hoc.
- Do not interpret any `*_untrained` composite as a diagnostic score.
- Do not claim OSA diagnosis from CTA; confirm with PSG/HSAT.
