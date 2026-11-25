# Segmentation Subproject

This subproject is for automatic deep-learning segmentations.

## ROI Tracks

- `heart`: LA/LAA, aorta, cardiac structures (TotalSegmentator `heartchambers_highres`, NUDF / VISTA3D LAA,
  and the HolOrama 14-structure whole-heart model in `subprojects/heart_segmentation/`)
- `cervical`: carotids, vertebral arteries, neck vessels
- `intracranial`: Circle of Willis and intracranial arteries

## Backbone Role

Segmentation is the canonical intermediate stage after:

1. DICOM -> NIfTI
2. NIfTI defacing
3. ROI segmentation (this subproject)

All downstream analysis modules should consume segmentation outputs from this stage,
not raw CTA directly.

## Current Runtime Entry Points

- Single-case backbone: `scripts/run_full_segmentation_pipeline.py`
- Batch backbone: `scripts/run_full_segmentation_batch.py`
- TopCoW-only: `scripts/run_topcow_claim.py`
- NUDF LAA-only: `scripts/run_cardiac_ct_explorer_nudf_only.py`
- NV LAA-only: `scripts/run_nv_segment_ct_laa.py`
- Whole heart, 14 labels incl. LAA + epicardial / pericardial fat (HolOrama, onnxruntime, CPU ok):
  `subprojects/heart_segmentation/run_heart_segmentation_case.py` / `..._batch.py`

## Configs

- ROI label definitions: `configs/roi/*.yaml`
- Profile selection: `configs/profiles/*.yaml`
