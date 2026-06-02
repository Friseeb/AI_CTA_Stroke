# stroke_cta_osa — overview

`stroke_cta_osa` is a research feature-extraction pipeline that turns a routine
head/neck **CT angiogram** acquired during acute stroke care into a row of
**airway-geometry and cervical-adiposity features**. The goal is downstream
phenotyping work for OSA / nocturnal vascular-risk hypotheses — never clinical
OSA diagnosis from CTA alone.

## Scope

- **Input.** CTA in DICOM (directory or zip) or NIfTI (single file ±sidecar).
- **Output.** Per-case row in `features.csv` + `qc.csv`; optional NIfTI masks
  and QC overlays; appended JSONL processing log.
- **Reuses where it can.** Upper-airway mask / landmarks / pre-computed
  features from a sibling dental-/CBCT- pipeline can be plugged in via the
  `DentalAirwayAdapter` (file-based contract — no Python dependency on the
  dental package).
- **Falls back where it must.** Without a precomputed mask, an HU-threshold +
  connected-component fallback produces a pharyngeal-column mask flagged
  `airway_method='threshold_connected_component'` with `confidence='low'`.

## Document index

| Doc | Purpose |
|---|---|
| [FEATURES.md](FEATURES.md) | Every output column: type, unit, method, missing-value behaviour |
| [QC.md](QC.md) | Quality-control checks, coverage score, artefact heuristics |
| [DENTAL_PIPELINE_INTEGRATION.md](DENTAL_PIPELINE_INTEGRATION.md) | What is shared with the dental subproject, what is CTA-specific |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Known unknowns, calibration TODOs, validation gates |

## Research-only disclaimer

All feature values, composite scores, and adapter outputs are **research
prototypes**. None of them — including the composite `*_score_untrained`
columns — should be used for patient-care decisions. The pipeline is designed
for retrospective cohort analysis where the imaging features are entered into
statistical models alongside formal sleep-study and clinical-outcome data.
