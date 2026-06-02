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
