# AI_CTA_Stroke Subprojects

This repository is now organized as a multi-subproject workspace.

Primary workflow documentation lives in `README.md`.

## Current Split

- `segmentation`: `subprojects/segmentation/`
- `analysis`: `subprojects/analysis/`
- `studies`: `subprojects/studies/`
- `la_laa`: `subprojects/la_laa/`
- `vertebral_manual`: `subprojects/vertebral_manual/`
- `aorta_cta_radiomics`: `aorta_cta_radiomics/`
- `cta-dental-opportunistic-screening`: `subprojects/cta-dental-opportunistic-screening/` — opportunistic dental analysis from head/neck CTA
- `stroke-cta-osa`: `subprojects/stroke-cta-osa/` — CTA-derived airway / cervical-adiposity features for OSA & nocturnal vascular-risk phenotyping in stroke cohorts (see `docs/stroke_cta_osa/`)

## Design Rules

1. Keep the backbone stage-first (`P10/P20/P30/P60/P80`).
2. Keep ROI-specific behavior in `configs/roi/*.yaml`.
3. Keep study-specific combinations in `configs/profiles/*.yaml`.
4. Keep core logic in shared scripts/modules, not copied per study.

## Cleanup Notes

- `leadership_elastic_app` was removed from this repository (maintained in its own separate repo)
- centerline stack was removed for now (pipeline/modules/docs/envs/scripts)
- Nested virtual environments are ignored via `.gitignore` (`.venv*/`, `**/.venv*/`)
