# Organization Blueprint

This is the target project organization.

## Top-Level Strategy

1. Backbone by stage:
   - ingest/deface
   - segmentation
   - analysis
2. Within stage, separate by ROI where needed.
3. Keep analysis code grouped by method type, not by ROI.
4. Use profile configs for study-specific combinations.

## Canonical Flow

- P10 Ingest: DICOM -> NIfTI
- P20 Privacy: Deface
- P30 Segmentation: heart/cervical/intracranial auto-seg
- P60 Analysis: radiomics/shape/calcium/atheroburden/tortuosity
- P80 Clustering and reports

## Configuration Layers

- `configs/roi/*.yaml`: ROI labels, mask names, merge policies
- `configs/profiles/*.yaml`: run selections and module combinations
- `configs/manifests/*.csv`: case lists and input pointers

## Why This Works

- avoids `ROI x analysis` script explosion
- keeps modules testable and reusable
- supports adding new studies by config only
