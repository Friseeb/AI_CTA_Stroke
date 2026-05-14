# AI_CTA_Stroke Subprojects

This repository is now organized as a multi-subproject workspace.

Primary workflow documentation lives in `README.md`.

## Current Split

- `segmentation`: `subprojects/segmentation/`
- `analysis`: `subprojects/analysis/`
- `studies`: `subprojects/studies/`
- `la_laa`: `subprojects/la_laa/`
- `la_laa_slaao`: `subprojects/la_laa_slaao/`
- `vertebral_manual`: `subprojects/vertebral_manual/`

### la_laa_slaao

Extension of `la_laa` for thrombus-inclusive LAA segmentation and SLAAO filling-state
representation learning. Adds CT-native foundation encoders, visual tokenization,
positive/negative anatomical priors, filling-defect mapping, and uncertainty-aware
segmentation. See `docs/protocols/laa_slaao_framework.md` for the full specification.

## Design Rules

1. Keep the backbone stage-first (`P10/P20/P30/P60/P80`).
2. Keep ROI-specific behavior in `configs/roi/*.yaml`.
3. Keep study-specific combinations in `configs/profiles/*.yaml`.
4. Keep core logic in shared scripts/modules, not copied per study.

## Cleanup Notes

- `leadership_elastic_app` was removed from this repository (maintained in its own separate repo)
- centerline stack was removed for now (pipeline/modules/docs/envs/scripts)
- Nested virtual environments are ignored via `.gitignore` (`.venv*/`, `**/.venv*/`)
