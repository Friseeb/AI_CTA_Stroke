# Quality control

The QC module's job is to flag — never to silently drop — cases that cannot
support OSA-style feature extraction. Every case is processed end-to-end;
QC writes `qc_pass`, `qc_failure_reasons`, and per-feature missingness so the
analyst can decide what to keep.

## Hard requirements (configurable)

`coverage.include_*` defaults can be overridden per analysis:

| Setting | Default | What 'required' means |
|---|---|---|
| `include_hard_palate` | `optional` | `landmarks.posterior_nasal_spine` must be present |
| `include_hyoid` | `optional` | `landmarks.hyoid` must be present |
| `include_epiglottis` | `optional` | `landmarks.epiglottis_tip` must be present |
| `include_cervical_soft_tissues` | `required` | ≥5% of lower-2/3 voxels in `[-250, 200]` HU |

Anything 'required' that's missing populates `qc_failure_reasons` and sets
`qc_pass=False`. Anything 'optional' that's missing → a warning, not a failure.

## Geometry sanity

| Check | Default | Effect on failure |
|---|---|---|
| z-extent ≥ `qc.min_z_extent_mm` | 60 mm | failure (case can't physically include both hyoid and PNS) |
| slice thickness ≤ `qc.max_slice_thickness_mm` | 3 mm | warning |

## Artefact heuristics

- **Lateral truncation:** axial slices where a high-HU column reaches the
  L/R image border — if >20 % of slices show this, `qc_truncation_flag=True`.
- **Dental artefact:** voxel fraction above `qc.dental_artifact_hu_threshold`
  (default 2500 HU). High fractions usually indicate metallic streak in the
  oral cavity — the airway and parapharyngeal columns may still be valid,
  but radiomic statistics may not.

## Coverage score

A simple 0..1 weighted sum, intended only as a sortable severity field — not
a normative threshold:

| Component | Weight |
|---|---|
| airway mask present | 0.40 |
| cervical soft tissue present | 0.30 |
| hyoid landmark present | 0.10 |
| epiglottis landmark present | 0.10 |
| no lateral truncation | 0.10 |

## Behaviour under load failure

If ingestion fails (corrupt DICOM, age < `ingestion.age_floor_years`, etc.)
the orchestrator still emits a row:

- `qc_pass=False`
- `qc_failure_reasons='load_failed: <short message>'`
- spacing columns NaN
- airway/fat columns all NaN

This keeps the schema stable across batches and lets `summarize` report
failure rates without dropping rows.

## Per-feature missingness flags

Beyond the global `qc_*` columns, individual features carry their own
provenance fields:

- `airway_method`, `airway_confidence`, `airway_csa_orientation`
- `airway_region_method`  (`landmarked` vs `unavailable`)
- `fat_roi_method`, `fat_parapharyngeal_roi_method`, `fat_retropharyngeal_roi_method`, `tongue_roi_method`
- `perivascular_available` / `_reason`
- `thoracic_available` / `_reason`
- `radiomics_available` / `_reason` / `rad_<roi>_available`

When in doubt, downstream analyses should filter on `airway_confidence in ('medium', 'high')` for primary results and use 'low' (i.e. fallback) for sensitivity analysis only.
