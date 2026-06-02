# Feature dictionary

Every column in `features.csv`. Stable column names: adding new features will
never rename an existing column. Missing values are NaN unless the column is
boolean (in which case the missingness is encoded in a `*_method` /
`*_available` neighbour).

Shared between dental / CBCT pipeline and stroke CTA pipeline → marked **(shared)**.
The dental subproject does not currently emit these — when it does, the
columns will line up directly for `compare-dental` joins.

---

## Identifiers and provenance

| Column | Type | Notes |
|---|---|---|
| `pipeline` | str | always `stroke_cta_osa` |
| `pipeline_version` | str | `__version__` of the running package |
| `config_hash` | str (12-char SHA-1) | reproducibility key |
| `processing_timestamp` | ISO-8601 (UTC) | |
| `patient_id` | str | user-provided OR opaque `study_id` hash if omitted |
| `study_id` | str | `stu_<sha1[:12]>` of StudyInstanceUID or input parent path |
| `scan_id` | str | `scn_<sha1[:12]>` of SeriesInstanceUID or input path |
| `input_path_hash` | str | sha1 of resolved input path — no PHI |
| `input_kind` | str | `dicom_dir` / `dicom_zip` / `nifti` |
| `airway_source` | str | which adapter produced the airway mask |
| `airway_provider_notes` | str | one-line provenance string from adapter |

---

## QC

| Column | Type | Unit | Method | Missingness |
|---|---|---|---|---|
| `qc_pass` | bool | — | All `coverage.include_*` 'required' checks satisfied AND z-extent ≥ `qc.min_z_extent_mm` | always present |
| `qc_warning_count` | int | — | non-fatal flags raised by QC | 0 if none |
| `qc_failure_reasons` | str | semicolon-joined | reasons `qc_pass` is False | empty string if pass |
| `qc_coverage_score` | float | 0–1 | weighted sum: airway 0.4 + soft 0.3 + hyoid 0.1 + epi 0.1 + no-truncation 0.1 | 0 on load failure |
| `qc_dental_artifact_score` | float | voxel fraction | fraction of voxels above `qc.dental_artifact_hu_threshold` | NaN on empty array |
| `qc_has_upper_airway` | bool | — | airway mask available and non-empty | always |
| `qc_has_cervical_soft_tissue` | bool | — | ≥5 % soft-tissue HU voxels in lower 2/3 of image | always |
| `qc_has_hyoid_region` | bool | — | hyoid landmark from adapter is not None | always |
| `qc_has_epiglottis_region` | bool | — | epiglottis landmark not None | always |
| `qc_truncation_flag` | bool | — | high-HU voxels touch L/R image border on >20 % of slices | always |
| `qc_spacing_{x,y,z}_mm` | float | mm | voxel spacing from sitk image | NaN on load failure |
| `qc_contrast_enhanced` | bool | — | DICOM-inferred (`ContrastBolusAgent` / `ImageType`) | False if unknown |
| `qc_z_extent_mm` | float | mm | shape_z × spacing_z | NaN on load failure |

---

## Airway geometry  (shared with dental pipeline)

All CSAs in v1 are **axial approximations** — `airway_csa_orientation`
records this; centerline-orthogonal CSAs are a future feature toggled by
`airway.centerline_orthogonal_csa`.

| Column | Type | Unit | Method | Missingness |
|---|---|---|---|---|
| `airway_mask_available` | bool | — | provider chain produced a non-empty mask | always |
| `airway_method` | str | — | `dental_adapter` / `external_mask` / `threshold_connected_component` / `null` | always |
| `airway_confidence` | str | — | `low` / `medium` / `high` | always |
| `airway_csa_orientation` | str | — | `axial_approximation` (v1) | always |
| `airway_volume_mm3` (shared) | float | mm³ | mask voxel count × voxel volume | NaN |
| `airway_volume_ml` (shared) | float | mL | ÷ 1000 | NaN |
| `airway_length_mm` (shared) | float | mm | # nonzero z slices × dz; **vertical extent**, not curve length | NaN |
| `airway_min_csa_mm2` (shared) | float | mm² | smallest per-slice voxel count × in-plane area | NaN |
| `airway_min_csa_slice_index` | int | — | z-index of min CSA | −1 |
| `airway_min_csa_z_mm` (shared) | float | mm | physical z of that slice (origin-relative) | NaN |
| `airway_csa_p05_mm2` | float | mm² | 5th percentile | NaN |
| `airway_csa_p10_mm2` | float | mm² | 10th percentile | NaN |
| `airway_csa_p25_mm2` | float | mm² | 25th percentile | NaN |
| `airway_csa_median_mm2` | float | mm² | median | NaN |
| `airway_lateral_diameter_min_mm` | float | mm | axis-aligned L-R bbox at min-CSA slice | NaN |
| `airway_ap_diameter_min_mm` | float | mm | axis-aligned A-P bbox at min-CSA slice | NaN |
| `airway_eccentricity_at_min_csa` | float | — | √(1 − (min/max)²) of those two diameters | NaN |
| `airway_region_method` | str | — | `landmarked` if any landmark provided, else `unavailable` | always |
| `retropalatal_csa_mm2` (shared) | float | mm² | mean CSA in ±window/2 of PNS / soft palate | NaN |
| `retroglossal_csa_mm2` (shared) | float | mm² | mean CSA in ±window/2 of epiglottis / hyoid | NaN |
| `retrolingual_csa_mm2` | float | mm² | mean CSA in ±window/2 of hyoid | NaN |
| `retropalatal_volume_ml` (shared) | float | mL | volume in window | NaN |
| `retroglossal_volume_ml` (shared) | float | mL | volume in window | NaN |

When the dental adapter supplies pre-computed shared values, those are
recorded **with a `_from_dental` suffix** (e.g. `airway_volume_ml_from_dental`)
so the CTA-recomputed column and the dental-reported column can be diffed.

---

## Fat compartments

### A. Total cervical fat
ROI = body silhouette ∩ z range [z_lo, z_hi] (landmark-or-airway-defined).

| `fat_cervical_volume_ml`, `fat_cervical_mean_hu`, `fat_cervical_median_hu`,
`fat_cervical_p10_hu`, `fat_cervical_p90_hu`, `fat_cervical_std_hu` |

`fat_cervical_z_lo_index`, `fat_cervical_z_hi_index` document the ROI extent
in voxel indices; `fat_roi_method` records which heuristic chose them.

### B. Subcutaneous cervical fat
Body voxels within `fat.subcutaneous_erosion_mm` of the body surface.
`fat_subcutaneous_fraction_of_neck_area` = subcutaneous voxels / body voxels.

### C. Deep cervical fat
Body voxels deeper than the subcutaneous band. `fat_deep_to_subcutaneous_ratio`
is a simple ratio (NaN if subcutaneous = 0).

### D. Parapharyngeal fat (needs airway mask)
Two slabs ±`fat.parapharyngeal_lateral_band_mm` lateral to the airway
centroid, restricted to ±`fat.parapharyngeal_axial_window_mm` around the
min-CSA z. Fat HU + body envelope intersect.

| Column | Unit | Notes |
|---|---|---|
| `fat_parapharyngeal_{left,right,total}_volume_ml` | mL | |
| `fat_parapharyngeal_{left,right,total}_mean_hu` | HU | |
| `fat_parapharyngeal_asymmetry_index` | — | (right − left) / total |
| `fat_parapharyngeal_to_airway_ratio` | — | total mL ÷ airway mL |
| `fat_parapharyngeal_area_at_min_airway_csa_mm2` | mm² | at min-CSA slice |
| `fat_parapharyngeal_area_retropalatal_mm2` | mm² | at PNS / soft-palate z (NaN if missing) |
| `fat_parapharyngeal_area_retroglossal_mm2` | mm² | at epiglottis / hyoid z (NaN if missing) |
| `fat_parapharyngeal_roi_method` | str | `anchored_z=<n>` or `airway_centroid_z=<n>` |

### E. Retropharyngeal fat (needs airway mask)
Posterior to the airway, anterior to the cervical spine region.

| Column | Unit |
|---|---|
| `fat_retropharyngeal_volume_ml` | mL |
| `fat_retropharyngeal_mean_hu` | HU |
| `fat_retropharyngeal_p10_hu` / `_p90_hu` / `_median_hu` / `_std_hu` | HU |
| `fat_retropharyngeal_max_thickness_mm` | mm | longest contiguous AP run × dy |
| `fat_retropharyngeal_mean_thickness_mm` | mm | mean over columns |
| `fat_retropharyngeal_roi_method` | str | |

### F. Posterior tongue surrogate (heuristic — optional)
Anterior to the airway, ±axial-window of min CSA. Marked
`heuristic` because there is no tongue segmentation in v1.

| `tongue_posterior_mean_hu`, `tongue_posterior_low_hu_fraction`,
`tongue_fat_surrogate_available`, `tongue_roi_method` |

---

## Optional modules

### Perivascular  (carotid mask required)
| `perivascular_available`, `pericarotid_shell_mm_used`,
`pericarotid_fat_{left,right,}_volume_ml`, `pericarotid_fat_{left,right,}_mean_hu`,
`pericarotid_fat_asymmetry`, `carotid_calcification_present`,
`carotid_plaque_volume_ml` |

### Thoracic / cardiac  (epicardial / mediastinal masks required)
| `thoracic_available`, `mediastinal_fat_volume_ml`, `mediastinal_fat_mean_hu`,
`epicardial_adipose_tissue_volume_ml`, `epicardial_adipose_tissue_mean_hu`,
`pericardial_fat_volume_ml`, `thoracic_fat_mean_hu` |

### Radiomics  (pyradiomics required, `radiomics.enabled=True`)
Columns: `rad_<roi>_<feature>` for ROIs in `radiomics.rois`. Always also
emits `radiomics_available` and `rad_<roi>_available` so missingness is
explicit.

---

## Composite exploratory scores  (UNVALIDATED)

These columns end in **`_untrained`** because they are simple, deterministic
combinations of the raw features and are **not standardized** against any
cohort. They exist so analysts have a quick sanity-check signal, never as
clinical predictions.

| Column | Method |
|---|---|
| `cta_osa_anatomy_score_untrained` | 100 ÷ `airway_min_csa_mm2` |
| `cta_osa_fat_score_untrained` | sum of `fat_parapharyngeal_total_volume_ml` + `fat_retropharyngeal_volume_ml` |
| `cta_osa_combined_score_untrained` | sum of the two above |
| `composite_score_method` | `raw_linear_unstandardized_v1` |
| `composite_score_disclaimer` | "EXPLORATORY — not standardized…" |

For proper analysis, fit z-scores in your analysis script using cohort-level
means and SDs adjusted for age/sex/BMI.
