# Open questions / planned work

The list is grouped by likely impact on downstream associations. None of
these block v1 use on a cohort, but each should be addressed before any
publication-grade analysis.

## v2 upgrade summary (this release)

The pipeline now ships a metric registry, an explicit landmark schema with
provider chain, dedicated tongue / mandible / soft-tissue / skeletal modules,
regional airway + fat features, exploratory composites, and an expanded
radiomics ROI list. See [CT_OSA_METRICS.md](CT_OSA_METRICS.md) for the
family overview and the per-family docs for the geometry contracts.

Open items below are unchanged unless tagged **(updated)**.

## Airway

- **Centerline-orthogonal CSA.** v1 reports axial CSA only
  (`airway_csa_orientation='axial_approximation'`). For tilted necks or
  retropalatal narrowing, axial CSA over-estimates true min CSA. A centerline
  + orthogonal cross-sections pass would change `airway_min_csa_mm2` and
  the percentile features. The toggle exists (`airway.centerline_orthogonal_csa`)
  but is not yet wired.
- **Landmark detection.** **(updated)** v2 introduces a four-provider chain
  (`build_landmark_bundle`): explicit JSON → dental adapter → conservative
  heuristic from the airway → empty bundle. The heuristic only fires when the
  airway is tall enough and only populates `retroglossal_level`,
  `tongue_base_level`, and `hard_palate_plane`. A real PNS / hyoid /
  mandibular-plane detector would still unblock the rest; AirwayNet-MM-H or
  cervical-vertebrae anchors remain candidate sources.
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
- **Tongue surrogate.** **(updated)** v2 separates the global tongue mask
  path from the landmark-only fallback. The fallback is opt-in
  (`allow_posterior_roi_fallback`) and is always flagged
  `tongue_roi_confidence='low'`. Floor-of-mouth contamination remains a
  concern for the fallback box but the column is clearly distinguished from
  the mask-driven posterior tongue ROI. A proper tongue segmenter (e.g. a
  TotalSegmentator extension) is still the right long-term answer.

## Composite scores

- **Cohort standardization.** **(updated)** v2 composites are off by default
  and, when enabled, require `cohort_stats` (per-feature mean + std) to emit
  z-scored values. Raw `composite_score_method='raw_linear_unstandardized_v2'`
  remains available for ablation but should not be used for cross-cohort
  comparison. Component direction signs are explicit in `COMPONENT_DIRECTIONS`
  (test-enforced) so flipped contributions cannot silently default to +1.
- **Score weighting.** The combined score is still an unweighted average of
  the four sub-composites. A weighted combination (e.g. logistic-regression
  coefficients from a validation cohort) would be more useful but is left
  out of this release to avoid claiming predictive validity. The composite
  disclaimer column makes the un-validated status visible in every row.

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
