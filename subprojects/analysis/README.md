# Analysis Subproject

This subproject is for post-segmentation analysis modules.

## Modules

- `radiomics`: feature extraction from masks/labelmaps
- `shape`: mesh and geometric analysis
- `calcium_score`: calcification quantification
- `atheroburden`: plaque burden and distribution metrics
- `tortuosity`: centerline-free or centerline-lite vessel tortuosity metrics

## Design Rule

Analysis modules should be reusable and ROI-agnostic.

- Module code: grouped by analysis type
- ROI behavior: driven by `configs/roi/*.yaml`
- Study presets: driven by `configs/profiles/*.yaml`

Avoid duplicating scripts by ROI unless algorithmically required.
