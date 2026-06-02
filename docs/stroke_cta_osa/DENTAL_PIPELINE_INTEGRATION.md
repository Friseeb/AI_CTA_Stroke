# Dental ↔ stroke-CTA integration

The dental subproject (`cta_dental` at `subprojects/cta-dental-opportunistic-screening`)
and `stroke_cta_osa` are **separate Python packages**: neither imports the
other. They share data via files. This keeps either pipeline runnable when the
other isn't installed, and lets us version their schemas independently.

## What is shared

These airway-geometry columns have stable names in both pipelines (defined in
[`stroke_cta_osa.shared_schema.SHARED_FEATURE_NAMES`](../../subprojects/stroke-cta-osa/stroke_cta_osa/shared_schema.py)):

- `airway_volume_ml`
- `airway_min_csa_mm2`
- `airway_min_csa_z_mm`
- `airway_csa_p05_mm2`, `..._p10_mm2`, `..._p25_mm2`, `..._median_mm2`
- `airway_length_mm`
- `airway_lateral_diameter_min_mm`, `airway_ap_diameter_min_mm`, `airway_eccentricity_at_min_csa`
- `retropalatal_csa_mm2`, `retroglossal_csa_mm2`, `retrolingual_csa_mm2`
- `retropalatal_volume_ml`, `retroglossal_volume_ml`

When both pipelines run on the same patient (e.g. dental on the same CTA, or
on an adjacent CBCT), `compare-dental` joins their `features.csv` files on
`patient_id` + optional `scan_id` and emits:

- `dental_cta_feature_comparison.csv`
- `bland_altman_table.csv` — long-form per-feature per-case `mean`, `diff`
- `correlation_summary.csv` — n, Pearson r, bias, limits of agreement
- `missingness_summary.csv`

## File contract

When the dental pipeline (or any other provider) wants to feed airway outputs
to `stroke_cta_osa`, it must write one or more of:

```
<some_dir>/<case_id>/
    airway.nii.gz         # binary airway mask in the CTA's geometry
    landmarks.json        # see schema below
    airway_features.json  # flat dict {feature_name: float}
```

### `landmarks.json` schema

Voxel-index tuples `[z, y, x]` in the CTA's frame. Unknown landmarks may be
omitted or set to `null`.

```json
{
  "posterior_nasal_spine": [60, 44, 40],
  "soft_palate_inferior":  [55, 46, 40],
  "hyoid":                 [20, 44, 40],
  "epiglottis_tip":        [25, 44, 40],
  "mandibular_plane_z":    35
}
```

### `airway_features.json` schema

Flat dictionary, keys drawn from `SHARED_FEATURE_NAMES`. Non-numeric values
are ignored.

```json
{
  "airway_volume_ml": 12.34,
  "airway_min_csa_mm2": 56.7,
  "retropalatal_csa_mm2": 82.1
}
```

The stroke pipeline records these values with a **`_from_dental` suffix**
(e.g. `airway_volume_ml_from_dental`) alongside its own re-computed values —
that way the original number is preserved and you can audit reuse without
losing the CTA-side measurement.

## How to enable reuse

CLI:

```bash
stroke-cta-osa extract /cta/sub-001.nii.gz --out outputs/run01 \
    --dental-mask /dental/sub-001/airway.nii.gz \
    --dental-landmarks /dental/sub-001/landmarks.json \
    --dental-features /dental/sub-001/airway_features.json
```

Batch mode auto-discovers per-case dental artefacts from a single root:

```bash
stroke-cta-osa batch /data/niftis --out outputs/cohort \
    --dental-artifacts-dir /dental/runs
# expects /dental/runs/<case_dir_basename>/{airway.nii.gz,landmarks.json,airway_features.json}
```

Programmatic:

```python
from stroke_cta_osa.config import PipelineConfig, apply_overrides
cfg = apply_overrides(PipelineConfig(), {
    "airway.use_existing_dental_airway_outputs": True,
    "airway.dental_airway_mask_path": "/dental/sub-001/airway.nii.gz",
    "airway.dental_landmarks_path":   "/dental/sub-001/landmarks.json",
})
```

## CTA-specific (not shared)

These features are stroke-CTA-only because either the dental pipeline doesn't
acquire the anatomy (caudal cervical fat) or doesn't use IV contrast:

- contrast-CTA QC flag (`qc_contrast_enhanced`)
- cervical / subcutaneous / deep cervical fat
- parapharyngeal fat (all asymmetry and ratios)
- retropharyngeal fat (volume + thickness)
- perivascular hooks (carotid + plaque)
- thoracic hooks (epicardial / mediastinal)
- clinical / outcome merge (`merge-clinical`)
- composite exploratory scores (`cta_osa_*_score_untrained`)

These columns will be NaN in any dental-derived `features.csv` and so
`compare-dental` will simply skip them. That is the intended behaviour.

## Why not import from `cta_dental`?

1. **Independence.** Either subproject must work when the other isn't installed.
2. **Versioning.** The dental schema is evolving (today it does not even emit
   airway features). A file contract is cheaper to evolve than a Python API.
3. **PHI surface area.** The two pipelines have separate scrubbing logic;
   coupling them would force a shared identifier convention before we're sure
   what that should be.

If/when stable, shared schemas justify a Python contract, the dental package
can add `cta_dental.airway.shared_payload_writer(...)` that produces exactly
the JSON described above; `stroke_cta_osa` will continue to consume the files
unchanged.
