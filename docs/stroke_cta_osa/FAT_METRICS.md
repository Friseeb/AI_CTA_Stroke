# Fat metrics

There are two fat modules:

| Module | Source file | Responsibility |
|---|---|---|
| Global fat | [stroke_cta_osa/fat.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/fat.py) | total cervical / subcutaneous / deep / parapharyngeal / retropharyngeal volumes |
| Regional fat | [stroke_cta_osa/fat_regions.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/fat_regions.py) | level-anchored areas at hyoid/RP/RG; per-side parapharyngeal; facial/buccal |

Both share the same HU window — voxels in `[fat_hu_min, fat_hu_max]` (default
`[-190, -30]`) — and the same body-mask convention so global and regional
numbers are directly comparable.

## Global fat (carried over)

The headline volumes are emitted by the existing `fat.py`:

| Column | Notes |
|---|---|
| `fat_cervical_total_volume_ml` | every fat voxel inside the body mask |
| `fat_subcutaneous_volume_ml` | fat in the subcutaneous shell (eroded body) |
| `fat_deep_cervical_volume_ml` | total − subcutaneous |
| `fat_parapharyngeal_total_volume_ml` | airway-relative band, both sides combined |
| `fat_retropharyngeal_volume_ml` | posterior-to-airway band |

These remain unchanged so anyone with downstream code reading them keeps
working.

## Level-anchored areas

`fat_regions.py` adds **single-slice** areas at three landmark levels. They
populate when the corresponding z-level is available.

| Column | Anchor | Definition |
|---|---|---|
| `fat_cervical_area_at_hyoid_level_mm2` | `hyoid_centroid.z` | every fat voxel inside body at that slice |
| `fat_cervical_area_at_retropalatal_level_mm2` | `retropalatal_level` | same |
| `fat_cervical_area_at_retroglossal_level_mm2` | `retroglossal_level` | same |

## Per-side parapharyngeal areas

At each of three z-anchors (retropalatal, retroglossal, subglosso-supraglottic
which is RG + 15 mm), we compute the left and right parapharyngeal fat areas
inside a lateral band around the airway, gated by the deep-fat mask. The
sub-glosso anchor only populates when the RG level is known.

| Column | Notes |
|---|---|
| `fat_parapharyngeal_area_<level>_left_mm2` | left side area at the anchor slice |
| `fat_parapharyngeal_area_<level>_right_mm2` | right side |
| `fat_parapharyngeal_area_<level>_total_mm2` | sum |
| `fat_parapharyngeal_to_airway_ratio_<level>` | total / airway area at the anchor |

The L/R split uses the airway slice's x-centre; voxels with `x < cx` are left,
`x > cx` are right.

The method used to define the per-side ROI is recorded in
`fat_regional_parapharyngeal_roi_method`:

* `airway_relative_box_deep_fat_gated` — coarse axis-aligned box, no anatomy
  priors.
* `<sector_method>_anatomy_prior_sector_deep_fat_gated` — when anatomy priors
  are enabled and at least one mask is available.

## Retropharyngeal level areas

| Column | Anchor | Notes |
|---|---|---|
| `fat_retropharyngeal_area_at_retropalatal_level_mm2` | RP | fat in the posterior-to-airway band at that slice |
| `fat_retropharyngeal_area_at_retroglossal_level_mm2` | RG | same |

When a prevertebral anatomy mask is supplied, the band is constrained by it
(method string in `fat_regional_anatomy_prior_masks_used`).

## Facial / buccal fat (opt-in)

`enable_facial_fat=False` by default — the upper-1/3-of-body proxy is too
coarse for routine use. When enabled, three columns populate:

| Column | Notes |
|---|---|
| `fat_facial_total_volume_ml` | every fat voxel in the upper 1/3 of body extent |
| `fat_buccal_left_volume_ml` | left half by axial midline |
| `fat_buccal_right_volume_ml` | right half |
| `fat_facial_to_parapharyngeal_ratio` | facial / parapharyngeal total volume (when both exist) |

## Anatomy priors

When `use_anatomy_priors=True` and the orchestrator passes prevertebral /
muscle / etc. masks, those regions are excluded from the regional ROIs so
streak artefact and adjacent muscle don't get counted as fat. Which priors
were applied is recorded in `fat_regional_anatomy_prior_masks_used`.
