# Tongue metrics

The tongue module ([stroke_cta_osa/tongue.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/tongue.py))
extracts geometric and HU-based features from a tongue mask plus an optional
landmark-only posterior-tongue ROI fallback. It deliberately **does not**
segment the tongue itself — that's left to upstream nnU-Net / TotalSegmentator
weights or, for cleaner cohorts, a manual mask.

## Why this module exists

Tongue features dominate OSA-anatomy literature but were missing or
under-specified in earlier versions of the pipeline. The set below covers:

* **Tongue body**: volume in mL + HU statistics (mean / median / std / p10 /
  p90 / low-HU fraction). The low-HU fraction at a calibrated threshold is the
  most-cited *tongue-fat surrogate* — though we never use the word
  "tongue fat" alone in feature names: low-HU is an attenuation surrogate,
  not a tissue-class diagnosis.
* **Posterior tongue ROI**: the posterior 1/3 of the mask along the y-axis,
  re-stat'd the same way as the full body. When no mask is available and the
  user opts in via `allow_posterior_roi_fallback=True`, a coarse landmark-or
  airway-anchored box stands in for the posterior tongue and is flagged
  `tongue_roi_confidence='low'`.
* **Tongue base**: a z-band around the `tongue_base_level` landmark (or the
  inferior 1/3 of the mask z extent when the landmark is missing), area at
  the retroglossal level, and posterior + inferior displacement against the
  airway anterior wall.
* **Ratios**: `tongue_to_mandible_volume_ratio` and
  `tongue_to_oral_cavity_volume_ratio` against externally-provided volumes.

## Inputs

| Argument | Used for | Source |
|---|---|---|
| `tongue_mask` | global volume + HU stats | external NIfTI, dental adapter, or none |
| `landmarks` | `tongue_base_level`, `retroglossal_level`, `hyoid_centroid` | bundle (preferred) or heuristic |
| `airway` | tongue/airway adjacency, fallback ROI anchoring | airway adapter |
| `mandible_volume_ml` | tongue/mandible ratio | mandible module |
| `oral_cavity_volume_ml` | tongue/oral-cavity ratio | oral-cavity mask |

## Fallback behaviour

| Mask | Landmarks | Airway | Output |
|---|---|---|---|
| Present | Present | any | full mask-driven feature set |
| Present | Missing | any | mask-driven + inferior-1/3 base band |
| Absent | Any | Absent | every feature NaN, `tongue_qc_pass=False` |
| Absent | Present + `allow_posterior_roi_fallback=True` | Present | landmark posterior box; HU stats; `roi_confidence='low'` |
| Absent | Missing + `allow_posterior_roi_fallback=True` | Present | airway-min-CSA posterior box; HU stats; `roi_confidence='low'` |

`allow_posterior_roi_fallback` defaults to **False**: the analyst must opt in
to landmark-only HU surrogates.

## Output columns

All names are exported by the registry and round-trip to CSV/JSON. NaN means
missing.

| Column | Unit | Notes |
|---|---|---|
| `tongue_mask_available` | bool | True only when a real mask was supplied |
| `tongue_mask_method` | str | `external_or_dental`, `absent`, `disabled` |
| `tongue_volume_mm3` / `tongue_volume_ml` | mm³ / mL | from mask |
| `tongue_mean_hu`, `tongue_median_hu`, `tongue_std_hu`, `tongue_p10_hu`, `tongue_p90_hu` | HU | over the mask |
| `tongue_low_hu_fraction` | fraction | voxels below `low_hu_threshold` |
| `tongue_low_hu_threshold_used` | HU | for reproducibility |
| `tongue_contrast_sensitive` | bool | True for contrast-enhanced CTAs |
| `tongue_posterior_roi_available` | bool | True for mask or fallback ROI |
| `tongue_posterior_roi_method` | str | how the ROI was derived |
| `tongue_posterior_volume_ml`, `tongue_posterior_mean_hu`, … | as above | on the posterior ROI |
| `tongue_base_volume_ml` | mL | tongue-base z band |
| `tongue_base_area_at_retroglossal_level_mm2` | mm² | base area at RG slice |
| `tongue_base_to_retroglossal_airway_ratio` | ratio | base area / airway area at RG |
| `tongue_base_posterior_displacement_mm` | mm | posterior crowding indicator |
| `tongue_base_inferior_displacement_mm` | mm | inferior displacement vs airway top |
| `tongue_to_mandible_volume_ratio` | ratio | tongue mL / mandible mL |
| `tongue_to_oral_cavity_volume_ratio` | ratio | tongue mL / oral-cavity mL |
| `tongue_qc_pass` / `tongue_qc_failure_reasons` | bool / str | gate signal |

## What we deliberately don't compute

* **Genioglossus volume / orientation.** No reliable adult-CTA prior.
* **"Tongue fat" as a tissue class.** Low-HU fraction is a surrogate, not a
  tissue-typed measurement.
* **Mid-sagittal tongue thickness.** Needs reliable midline registration.
