# radselect

`radselect` is a Python package and command-line tool for leakage-safe feature
selection and dimensionality reduction on already-extracted tabular features.
It is designed for radiomic features, clinical variables, and combined
radiomic-clinical datasets.

This package does not perform image segmentation or radiomic extraction.

## Installation

From this repository checkout:

```bash
python -m pip install -e subprojects/radselect
```

Install optional report plotting and survival support when needed:

```bash
python -m pip install -e 'subprojects/radselect[reports,survival]'
```

The CLI is available as `radselect`. In environments where console scripts are
not on `PATH`, the same commands can be run as `python -m radselect ...`.

## Current scope

- Binary classification, multiclass classification, and regression pipelines.
- Survival and competing-risk selection with optional `lifelines` Cox models.
  If `lifelines` is unavailable or Cox fitting fails, `radselect` falls back to
  a training-derived signed risk score and records that status in the
  performance table.
  Competing-risk mode is a cause-specific analysis for the configured event
  code of interest; competing events are treated as censored for selection and
  C-index evaluation. It is not a Fine-Gray subdistribution hazard model.
- Robustness filtering from an external CSV.
- Optional feature metadata/provenance auditing, including enforcement of
  already-extracted IBSI-compliant radiomic features when a metadata CSV is
  supplied.
- Sample-level outcome/time/event auditing before any model fit or validation.
- Missingness and near-zero-variance filtering inside training folds.
- Correlation redundancy filtering inside training folds.
- Conventional feature screening by univariate relevance or mutual information.
- Inner-loop elastic-net hyperparameter tuning inside each outer validation
  fold.
- Elastic-net model-based selection with the tuned parameters.
- Stability selection across repeated training resamples.
- Optional PCA or PLS projection outputs and foldwise projection validation.
- Internal nested-style outer validation, center/group-held-out validation when
  a group column is supplied, and external validation when a separate CSV is
  supplied.
- Named center/site/scanner holdout validation from a single table via
  `--holdout-group`.
- CSV/JSON/HTML reporting.
- Reproducibility artifacts: `manifest.json`, `provenance.json`,
  `output_manifest.json`, `effective_config.json`, `tuning_summary.csv`, and
  `selected_feature_frequency.csv`.

The validation design keeps all missingness filtering, variance filtering,
univariate relevance screening, correlation filtering, hyperparameter tuning,
and model-based selection inside each training fold. The held-out fold, held-out
center/group, or external CSV is only used for evaluation.

For survival and competing-risk Cox models, feature imputation and scaling are
also fit on the training partition only and then applied to held-out or external
rows. If optional Cox modeling is unavailable or fails, the fallback signed risk
score uses the same train-only preprocessing principle.

## Selection order

The intended selection sequence is:

1. Use an already-extracted, IBSI-compatible radiomics feature table.
2. Audit or enforce IBSI/provenance metadata when `--feature-metadata-csv` and
   `--require-ibsi-compliant` are supplied.
3. Apply test-retest, segmentation, or acquisition robustness screening from
   `--robustness-csv`.
4. Remove missing and near-zero-variance features inside each training fold.
5. Apply conventional screening and correlation redundancy filtering inside
   each training fold.
6. Fit/tune elastic-net selection inside each training fold.
7. Estimate stability across repeated resamples of the retained development
   data.
8. Evaluate only on outer folds, a center/site/scanner holdout, or a separate
   external validation CSV.

## Outputs

Each CLI run writes:

- `selected_features.csv`
- `column_audit.csv`
- `modality_audit.csv`
- `correlation_audit.csv`
- `schema_audit.csv`
- `robustness_audit.csv`
- `feature_metadata_audit.csv`
- `sample_audit.csv`
- `quality_checks.csv`
- `dependency_audit.csv`
- `dropped_features.csv`
- `selected_feature_frequency.csv`
- `validation_splits.csv`
- `stability_selection.csv`
- `stability_resamples.csv`
- `tuning_summary.csv`
- `performance.csv`
- `predictions.csv`
- `composite_scores.csv`
- `final_signature.csv`
- `final_signature_parameters.csv`
- `final_composite_scores.csv`
- `manifest.json`
- `provenance.json`
- `output_manifest.json`
- `run_invocation.json`
- `effective_config.json`
- `radselect_report.html`

When projection is enabled, `projection_scores.csv` and
`projection_loadings.csv` are also written for transparent exploration.
`projection_performance.csv` and `projection_predictions.csv` are written from
leakage-safe projection validation where PCA/PLS is fit only on each training
fold, then transformed into the held-out fold or external validation cohort.
`final_projection_scores.csv` and `final_projection_parameters.csv` are also
written; these are fit on retained development rows and store the imputation
medians, scaling parameters, and component weights needed to reproduce the
final PCA/PLS transform after validation.
When plotting dependencies are available, report figures are written under
`report_assets/`.

To apply a saved final projection to another already-extracted feature table:

```bash
radselect project \
  --input new_cohort_features.csv \
  --parameters radselect_out/final_projection_parameters.csv \
  --output new_cohort_projection_scores.csv \
  --id-column case_id
```

The projection command is strict in the same way as `radselect score`: every
feature required by the saved projection signature must be present in the new
input table. It writes a sidecar manifest such as
`new_cohort_projection_scores_manifest.json`.

`manifest.json` records the input CSV path, SHA-256 hash, byte size, row count,
column count, and column names. If an external validation CSV or JSON run config
is provided, those files are fingerprinted as well.

`run_invocation.json` records the working directory, captured CLI arguments,
full effective settings, the absolute `effective_config.json` path, and a
recommended rerun command that changes back to the original working directory
before running `radselect run --config <absolute-effective-config-path>`.

`output_manifest.json` records every generated output artifact except itself,
including relative path, byte size, SHA-256 hash, and CSV row/column counts
where applicable. The CLI refreshes this file after the HTML report and report
assets are written.

Audit, selection, prediction, performance, scoring, and projection CSVs keep
stable headers even when a particular run records zero rows for that artifact,
so downstream code can read them without special empty-file handling.

`validation_splits.csv` records the exact ID membership for every outer
train/test split and any external or center-held-out validation rows. Use it to
audit that held-out rows were not used during fold-local screening, redundancy
filtering, elastic-net tuning, or model-based selection.

`quality_checks.csv` is a compact machine-readable checklist for run mechanics,
including split recording, external schema validation, metadata-column
protection, modality/domain definition recording, final score/projection
reproducibility artifacts, and runtime dependency posture. It is an audit aid,
not evidence of clinical validity.

`dependency_audit.csv` records the package's runtime-required dependencies,
separates optional/dev extras, and flags blocked LLM/OpenAI package names. This
makes the no-LLM runtime guarantee machine-readable in addition to the
provenance note.

`modality_audit.csv` records which features were listed and retained in each
modality or user-defined domain, including generated `combined` signatures.
Use it to answer how many radiomics, clinical, combined, or domain-specific
variables entered the fold-local filtering and elastic-net stages.

`correlation_audit.csv` records above-threshold redundancy decisions made by
the fold-local correlation filter. Each row names the retained feature, the
dropped feature, the absolute correlation, the configured threshold/method, and
the two features' screening relevance values.

`stability_selection.csv` reports aggregate selection probabilities across
repeated resamples. `stability_resamples.csv` records each stability resample's
training-row membership, selected features, candidate-feature count, and model
status by modality so the stability analysis can be audited. When a
`group_column` is configured, stability resampling samples whole groups rather
than individual rows and records `sampling_unit`, `n_train_groups`, and
`train_groups`.

`tuning_summary.csv` records inner-loop elastic-net candidates and selected
hyperparameters for each outer fold when tuning is enabled. `manifest.json`
also includes a `nested_tuning` summary, and `quality_checks.csv` records
whether nested elastic-net tuning was disabled, recorded, or unexpectedly
missing selected candidates.

`composite_scores.csv` contains foldwise selected-feature composite scores. For
each held-out fold or external/center-held-out cohort, the selected features are
median-imputed and standardized using the corresponding training data only, then
combined as a signed weighted sum using the selector's feature weights. Positive
scores are oriented toward higher predicted outcome/risk based on the fitted
training-fold selector.

`final_signature.csv`, `final_signature_parameters.csv`, and
`final_composite_scores.csv` are final-refit artifacts for downstream use after
validation. They are trained on the retained development rows, include
stability-selection probabilities when available, and should not be interpreted
as held-out performance estimates. The parameters file records each selected
feature's imputation median, standardization mean/std, and composite-score
weight so the final score can be reproduced outside `radselect`.

To apply a saved final signature to another already-extracted feature table
without rerunning selection:

```bash
radselect score \
  --input new_cohort_features.csv \
  --parameters radselect_out/final_signature_parameters.csv \
  --output new_cohort_scores.csv \
  --id-column case_id
```

The scoring command is strict: all features required by each saved modality
signature must be present in the new input table. It writes the score CSV plus a
sidecar manifest named after the output file, for example
`new_cohort_scores_manifest.json`.

Rows with missing classification outcomes, non-numeric regression outcomes, or
invalid survival time/event fields are dropped before splitting or model fitting.
Every retained or dropped row is recorded in `sample_audit.csv`.

When an external CSV or single-table holdout group is supplied, `schema_audit.csv`
records whether required metadata and candidate feature columns are present in
development and external data. Missing external feature columns stop the run
before validation so selected-feature and projection evaluations cannot silently
use different schemas.

## CLI example

```bash
radselect init-config --output radselect_config.json

radselect run \
  --input features.csv \
  --target mace \
  --id-column case_id \
  --task binary \
  --feature-regex '^rad_|^clinical_' \
  --radiomics-regex '^rad_' \
  --clinical-regex '^clinical_' \
  --feature-metadata-csv feature_metadata.csv \
  --require-ibsi-compliant \
  --screening-method mutual_info \
  --outdir radselect_out
```

The same run can be stored in JSON and executed reproducibly:

```bash
radselect run --config radselect_config.json
```

`radselect init-config` writes a comprehensive template with the material
selection, robustness, tuning, stability, projection, and validation controls.
Explicit command-line flags override values loaded from `--config`; omitted
flags do not reset JSON values. Use `--tune-elastic-net` or
`--no-tune-elastic-net` to override tuning explicitly.
Numeric controls are validated before any split or model fit. Invalid values
such as negative variance thresholds, out-of-range correlation/stability
thresholds, or zero projection components stop the run before outputs are
written.

`column_audit.csv` marks protected metadata columns such as IDs, outcomes,
time/event fields, and group labels. These are excluded from feature use even if
they are accidentally listed as candidate features. Candidate feature names that
look outcome-like, such as names containing `target`, `event`, `death`,
`mortality`, or `mace`, are flagged for review.

`feature_metadata_audit.csv` records feature-level provenance metadata when
`--feature-metadata-csv` is supplied. If `--require-ibsi-compliant` is set,
features listed as non-compliant or unknown in that metadata are rejected before
robustness and outcome-driven selection. Add `--ibsi-require-listed` when every
candidate feature must be listed in the metadata CSV.

The feature metadata file should contain a feature-name column named one of
`feature`, `feature_name`, `variable`, `column`, or `name`. To enforce IBSI
compliance it also needs a boolean-like column named one of `ibsi_compliant`,
`ibsi`, `ibsi_compliance`, `compliant`, or `radiomics_compliant`.

To use an external validation cohort:

```bash
radselect run \
  --input development.csv \
  --external-input external.csv \
  --target mace \
  --task binary \
  --feature-regex '^rad_|^clinical_' \
  --outdir radselect_external_out
```

To reserve one or more centers/sites/scanners from the same input table:

```bash
radselect run \
  --input features.csv \
  --target mace \
  --task binary \
  --group-column center \
  --holdout-group site_c \
  --feature-regex '^rad_|^clinical_' \
  --outdir radselect_center_holdout_out
```

Rows with the requested `--holdout-group` values are removed before internal
cross-validation and are evaluated afterward as the `external` fold.

For survival:

```bash
radselect run \
  --input features.csv \
  --task survival \
  --time-column followup_days \
  --event-column event \
  --outdir radselect_survival_out
```

For competing-risk analysis:

```bash
radselect run \
  --input features.csv \
  --task competing_risk \
  --time-column followup_days \
  --event-column event_code \
  --competing-event-code 1 \
  --outdir radselect_competing_risk_out
```

`manifest.json` records the competing-risk method as cause-specific, the event
code of interest, event-code counts, and how competing events were handled.

## Robustness CSV

The robustness file should contain a feature-name column named one of
`feature`, `feature_name`, `variable`, `column`, or `name`. It may contain:

- a boolean keep/pass column: `robust`, `keep`, `pass`, `passes`, or `include`;
- or a numeric robustness column such as `icc`, `test_retest_icc`,
  `segmentation_icc`, `acquisition_icc`, `robustness`, or `ccc`.

Features failing the configured threshold are excluded before outcome-driven
selection.

`robustness_audit.csv` records the retained/rejected decision for each listed
feature, the weakest recorded robustness score when numeric metrics are used,
the configured threshold, and the robustness metric columns or pass/fail column
used for the decision.
