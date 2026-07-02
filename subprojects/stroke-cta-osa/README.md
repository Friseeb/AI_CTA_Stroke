# stroke_cta_osa

Research pipeline that extracts CTA-derived **upper-airway and cervical/parapharyngeal/retropharyngeal adiposity features** from routine acute-stroke head/neck CT angiograms, for downstream phenotyping work on:

- obstructive sleep apnoea (OSA) and nocturnal vascular-risk surrogates
- wake-up stroke
- AF / AF-detected-after-stroke (AFDAS)
- PFO / right-to-left shunt interactions
- small-vessel disease, recurrence, MACE

> **This pipeline does not diagnose OSA.** It produces reproducible image-derived
> features that downstream statistical work can correlate with sleep-study labels
> (AHI / ODI / T90 / min-SpO₂) and stroke outcomes. Every output is flagged as
> research / experimental.

## Evidence-tiered features

Every feature is separated by **strength of prior OSA imaging evidence** so the
primary phenotype is never contaminated by exploratory features:

| Tier | Feature set | Use |
|---|---|---|
| Tier 1 — core OSA-backed | `core_osa_backed` | primary CTA-OSA phenotype |
| Tier 2 — OSA-plausible CT anatomy | `core_plus_anatomic_extensions` | secondary / mechanistic |
| Tier 3 — CT cardiometabolic/vascular | `core_plus_cardiometabolic_ct` | stroke / MACE / AF / AFDAS risk |
| Tier 4 — novel stroke-CTA | `all_features_exploratory` | exploratory only |

```bash
stroke-cta-osa list-features --feature-set core_osa_backed --out core_dict.csv
stroke-cta-osa extract case.nii.gz --out out/ --feature-set core_osa_backed
stroke-cta-osa summarize out/features.csv --by-evidence-tier
```

See the full evidence-based guide and reference table in
**[docs/stroke_cta_osa/README.md](../../docs/stroke_cta_osa/README.md)** and
[EVIDENCE_TIERS.md](../../docs/stroke_cta_osa/EVIDENCE_TIERS.md) ·
[FEATURE_SETS.md](../../docs/stroke_cta_osa/FEATURE_SETS.md) ·
[FAT_COMPARTMENTS.md](../../docs/stroke_cta_osa/FAT_COMPARTMENTS.md) ·
[ANALYSIS_PLAN.md](../../docs/stroke_cta_osa/ANALYSIS_PLAN.md).

## Quick start

```bash
# from the subproject directory
pip install -e .

# one CTA → features.csv + qc.csv
stroke-cta-osa extract /path/to/cta.nii.gz --out outputs/run01 --patient-id sub-001

# reuse an airway mask already produced by the dental / CBCT pipeline
stroke-cta-osa extract /path/to/cta.nii.gz --out outputs/run01 \
    --dental-mask /dental/runs/sub-001/airway.nii.gz \
    --dental-landmarks /dental/runs/sub-001/landmarks.json

# reuse the dental pipeline jawbone as the OSA mandible mask
stroke-cta-osa extract /path/to/cta.nii.gz --out outputs/run01 \
    --dental-mandible-mask /dental/runs/sub-001/roi/_tseg_teeth/lower_jawbone.nii.gz

# reuse TotalSegmentator/VISTA/manual anatomy masks as fat/tongue priors
stroke-cta-osa extract /path/to/cta.nii.gz --out outputs/run01 \
    --external-tongue-mask /totalseg/head_muscles/tongue.nii.gz \
    --dental-mandible-mask /dental/runs/sub-001/roi/_tseg_teeth/lower_jawbone.nii.gz

# or point at a dental pipeline output directory and let the CLI discover it
stroke-cta-osa extract /path/to/cta.nii.gz --out outputs/run01 \
    --dental-artifacts-dir /dental/runs/sub-001

# batch a directory of NIfTI inputs
stroke-cta-osa batch /data/cohort_niftis --out outputs/cohort --glob "*.nii.gz"

# compare against dental airway features
stroke-cta-osa compare-dental cta/features.csv dental/features.csv --out outputs/compare

# merge with a clinical/outcome CSV
stroke-cta-osa merge-clinical outputs/cohort/features.csv clinical/outcomes.csv \
    --out outputs/cohort
```

## What it produces

| File | Contents |
|---|---|
| `features.csv` | one row per CTA, full feature schema (see [docs/stroke_cta_osa/FEATURES.md](../../docs/stroke_cta_osa/FEATURES.md)) |
| `qc.csv` | per-case QC flags + coverage score |
| `feature_metadata.json` | column list + pipeline version |
| `case_processing_log.jsonl` | one JSON-line per case, warnings + errors |
| `<case_hash>/mask_*.nii.gz` | optional masks (airway, fat compartments) |
| `<case_hash>/qc_*.png` | optional axial QC overlays |

## Pipeline overview

```
ingest ─► QC ─► airway provider chain ─► airway features
                  │                          │
                  └─► fat ROIs ──────────────┴─► fat features
                                              │
                                              ▼
                                       composite + radiomics ─► CSV
```

**Airway provider chain (priority order):**

1. `DentalAirwayAdapter` — reads a NIfTI mask + landmarks/features JSON produced
   by the sibling dental/CBCT pipeline. Zero dental-package dependency: contract
   is JSON + NIfTI on disk.
2. `ExternalMaskAdapter` — user-supplied airway mask NIfTI.
3. `CTAFallbackAirwayAdapter` — HU-threshold + connected-component pharyngeal
   column selector. Marked `confidence='low'` in every output row.
4. `NullAirwayAdapter` — emits NaN for every airway feature.

Mandible is handled separately from the airway provider chain. A real dental
pipeline mandible/jawbone mask is preferred when supplied with
`--dental-mandible-mask` or discovered under `--dental-artifacts-dir`
(`lower_jawbone.nii.gz`, `mandible.nii.gz`, or the TotalSegmentator-teeth
subfolders). CTA-only mandible HU fallback is disabled by default because it
was visually unstable.

Tongue, mandible, oral-cavity, soft-palate, uvula, and tonsil masks can be
provided from TotalSegmentator, VISTA-style outputs, manual Slicer edits, or the
dental pipeline. When present, these masks are used as conservative anatomy
priors for local parapharyngeal / retroglossal / subglosso-supraglottic /
retropharyngeal fat ROIs. Fat itself is still defined by the configured HU
window; the anatomy masks only constrain ROI geometry and are recorded in
`fat_anatomy_prior_masks_used`.

## Design principles

- **Stable column names.** Adding features never renames existing ones.
- **NaN, not crash.** Missing landmarks, missing masks, failed loads — every
  row keeps the same schema; the affected columns are NaN with a `*_method`
  string explaining why.
- **No PHI in logs.** DICOM tags are filtered through `dicom_utils.scrub_dicom_metadata`
  before anything reaches stderr or disk. Identifiers in CSV are SHA-1 hashes.
- **Provenance.** Every row carries `pipeline_version`, `config_hash`,
  `airway_source`, and `airway_provider_notes` so analyses can be filtered or
  re-keyed when the config changes.
- **Optional everything.** PyRadiomics, matplotlib, the dental pipeline,
  carotid masks, epicardial masks: missing them never breaks the pipeline.

## Non-goals (v1)

- No deep-learning training — the pipeline does not assume AirwayNet-MM-H weights are available.
- No carotid segmentation from scratch — perivascular features only fire when a mask is supplied.
- No epicardial-fat segmentation from scratch — same.
- No diagnostic claim. Composite scores end in `_untrained` and are explicitly *not* standardized.

See [docs/stroke_cta_osa/OPEN_QUESTIONS.md](../../docs/stroke_cta_osa/OPEN_QUESTIONS.md) for the open work list.
