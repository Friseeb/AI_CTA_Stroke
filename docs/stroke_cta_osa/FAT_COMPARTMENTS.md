# Fat compartments by evidence tier

Fat compartments are grouped by [evidence tier](EVIDENCE_TIERS.md) in
`stroke_cta_osa/fat_ontology.py`, the fat-specific companion to the evidence
registry. Every compartment records whether it is a **true anatomic** region or
a geometric **proxy**, the masks/landmarks required to upgrade a proxy to
anatomic, contrast/artifact sensitivity, and which `*_confidence` column governs
it.

| feature_family | example features | evidence_tier | prior evidence | intended use |
|---|---|---|---|---|
| cervical total fat | `fat_cervical_total_volume_ml`, `fat_cervical_mean_hu` | Tier 1 | Ernst 2023; Shelton 1993 | primary phenotype |
| parapharyngeal fat pad | `fat_parapharyngeal_total_volume_ml`, `fat_parapharyngeal_area_retroglossal_total_mm2` | Tier 1 | Chen 2019; Shelton 1993 | primary phenotype |
| pharyngeal airway-adjacent fat | `fat_deep_peripharyngeal_volume_ml` | Tier 1 | Shelton 1993 | primary phenotype |
| retropharyngeal fat | `fat_retropharyngeal_volume_ml`, `fat_retropharyngeal_mean_thickness_mm` | Tier 2 | anatomic / Shelton 1993 | mechanistic secondary |
| submandibular / submental fat | `fat_submandibular_space_total_volume_ml`, `fat_submental_total_volume_ml` | Tier 2 | anatomic | mechanistic secondary |
| surface-shell fat | `fat_surface_shell_0_5mm_volume_ml` … | Tier 2 | proxy | mechanistic secondary |
| supra-/subplatysmal proxy | `fat_supraplatysmal_proxy_volume_ml` | Tier 2 | proxy (no platysma mask) | mechanistic secondary |
| periairway distance-shell fat | `fat_periairway_shell_0_5mm_volume_ml` … | Tier 2 | proxy | mechanistic secondary |
| C5 NAT compartments | `fat_c5_nat_subcutaneous_area_mm2`, `fat_c5_nat_perivertebral_area_mm2` | Tier 3 | Torriani 2014 | cardiometabolic risk |
| pericarotid fat | `fat_pericarotid_left_volume_ml` | Tier 3 | vascular CT | cardiometabolic / vascular risk |
| engineered fat ratios | `fat_periairway_to_min_csa_ratio`, `fat_parapharyngeal_to_tongue_base_ratio` | Tier 4 | none (novel) | exploratory only |

## True anatomic vs proxy

A compartment is labelled **anatomic** only when real segmentation masks (or
validated landmarks) define it; otherwise it is a **proxy** and its feature
names carry `_proxy` or `true_anatomic_vs_proxy='proxy'`. Enforced rules:

- Surface-shell fat is **not** labelled supraplatysmal/subplatysmal unless a
  platysma mask is supplied.
- Submandibular fat is **not** labelled gland-excluded unless a submandibular
  gland mask is supplied (`fat_submandibular_gland_excluded_flag`).
- Vessels and glands are excluded from fat ROIs when masks are available
  (contrast-enhanced vessel exclusion via `fat.exclude_vessels_hu_min`).

## Sensitivity / contrast / artifact

- **Contrast sensitivity** — all `*_mean_hu` fat metrics shift with CTA contrast
  phase; flagged `contrast_sensitive` and surfaced through QC.
- **Dental artifact** — degrades parapharyngeal, submandibular, retropharyngeal,
  and tongue-base fat; surfaced via `qc_dental_artifact_*`.
- **HU thresholds** — the primary fat window is HU ∈ [−190, −30]. Sensitivity
  profiles (`hu200_50`, `strict`) are opt-in for robustness analyses
  (`configs/stroke_cta_osa_feature_sets.yaml`).

## Missingness behaviour

Optional compartments emit **NA** (not absent columns) when their masks or
landmarks are unavailable. Each compartment's confidence collapses to:

- `high` — all required masks present and the compartment is anatomic,
- `moderate` — partial masks or a robust landmark-based method,
- `low` — geometric proxy only,
- `missing` — a required mask is absent.

## Overlap with the dental/CBCT pipeline vs stroke-CTA-specific

Airway-adjacent / parapharyngeal fat geometry can be partly reconstructed from
shared airway masks, but **cervical, retropharyngeal, submandibular, periairway,
C5 NAT, and pericarotid compartments are stroke-CTA-specific** because they fall
below the typical CBCT field of view and depend on contrast-CTA tissue contrast.
