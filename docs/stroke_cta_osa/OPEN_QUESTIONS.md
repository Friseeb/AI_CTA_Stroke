# Open questions / planned work

The list is grouped by likely impact on downstream associations. None of
these block v1 use on a cohort, but each should be addressed before any
publication-grade analysis.

## Airway

- **Centerline-orthogonal CSA.** v1 reports axial CSA only
  (`airway_csa_orientation='axial_approximation'`). For tilted necks or
  retropalatal narrowing, axial CSA over-estimates true min CSA. A centerline
  + orthogonal cross-sections pass would change `airway_min_csa_mm2` and
  the percentile features. The toggle exists (`airway.centerline_orthogonal_csa`)
  but is not yet wired.
- **Landmark detection.** Without dental-provided landmarks, all
  retropalatal / retroglossal / retrolingual outputs are NaN. A light CTA-only
  landmark detector (PNS, hyoid, mandibular plane) would unblock the
  region-specific columns even when the dental pipeline isn't available.
  AirwayNet-MM-H weights are not assumed to be on disk; a heuristic
  (cervical-vertebrae anchor) is a reasonable starting point.
- **Length vs vertical extent.** `airway_length_mm` is currently
  `n_nonzero_z_slices × dz`. For a curved airway this under-estimates true
  length. Should switch to centerline arc length when (1) is done.
- **Fallback robustness.** The HU-threshold component selector picks the
  pharyngeal column by upper-axial centroid. On scans with a partial mouth
  open or with a nasogastric tube, the wrong component can win. Add a
  trachea-rejection step using inferior-axial extent.

## Fat

- **Body envelope on contrast-CTA.** `fat.body_air_threshold_hu` is currently
  −250 HU. Contrast in the carotids does not affect this, but external
  artefact (table edges) can. A simple morphological closing-then-largest-CC
  is already in place but could be tightened.
- **Subcutaneous-vs-deep boundary.** v1 uses fixed erosion distance
  (`fat.subcutaneous_erosion_mm=6`). Should ideally be patient-size-adjusted
  (BMI- or neck-circumference-adjusted) when those are available in the
  clinical CSV.
- **Vessel exclusion.** Default `fat.exclude_vessels_hu_min=120` is correct
  for fat HU but we do NOT explicitly subtract a vessel mask from the
  parapharyngeal / retropharyngeal ROIs. On peak-arterial CTA this is fine
  (vessels are too bright to be fat), but consider explicit subtraction for
  multi-phase studies.
- **Parapharyngeal anchor.** Currently the anchor is the min-CSA slice. For
  patients without a clear narrowing this picks the slice with the smallest
  measurement noise (i.e. mid-pharynx). Should additionally anchor at the
  PNS-to-hyoid midpoint when landmarks exist.
- **Tongue surrogate.** v1 ROI is a coarse anterior-to-airway slab. It
  contaminates with floor-of-mouth musculature. Either skip the column for
  v1 publications or wait for a tongue segmenter.

## Composite scores

- **Cohort standardization.** `cta_osa_*_score_untrained` are deliberately
  raw values. For comparable scores across cohorts we need at minimum
  age/sex/BMI z-scoring against a reference distribution. The
  `merge-clinical` step now produces the needed data; a downstream notebook
  should be added to compute the standardized score.
- **Score weighting.** The combined score is a flat sum. A weighted
  combination (e.g. logistic-regression coefficients from a validation
  cohort) would be more useful but is left out of v1 to avoid claiming
  predictive validity.

## Calibration & validation

- **Inter-pipeline agreement.** `compare-dental` produces the merged CSV
  but we have no agreement target yet. Once both pipelines emit overlapping
  features on the same cohort, lock down a target Pearson r and Bland-Altman
  bias band per feature.
- **AHI association.** No analysis script ships in v1. The intended
  downstream test is `airway_min_csa_mm2` and
  `fat_parapharyngeal_to_airway_ratio` vs AHI / ODI on a sleep-study
  sub-cohort. This is the first validation gate.
- **Outcome associations.** wake-up stroke / AFDAS / PFO / MACE need
  larger cohorts; the merged CSV is the entry point for those analyses,
  not this package.

## Engineering

- **Resampling.** Default is `ingestion.resample_spacing_mm: null` (native
  spacing). For multi-site cohorts a fixed isotropic spacing makes
  voxel-volume features directly comparable. Consider 0.5 mm or 0.625 mm
  isotropic for primary analysis once memory is profiled.
- **Speed.** Single case on a synthetic 80³ volume runs in milliseconds.
  On a real 512×512×400 CTA the fat module (especially body silhouette +
  thickness loops) is the long pole. Profile + vectorize before scaling
  to thousands of cases.
- **GPU.** Not used anywhere in v1. AirwayNet-MM-H or any DL segmenter
  would change that; expose a `--device cuda|cpu` once any DL is added.
- **Persistent provenance.** `case_processing_log.jsonl` records per-case
  warnings/errors but not the full config that ran. Either persist the
  rendered YAML per case directory or include the config JSON in the log.
