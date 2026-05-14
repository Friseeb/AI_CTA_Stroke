# LA/LAA Substudy

This subproject defines the Left Atrium / Left Atrial Appendage substudy scope
within `AI_CTA_Stroke`.

## Scope

- LAA-focused segmentation outputs (NUDF / NV-Segment-CT)
- LA + LAA mesh generation
- LA/LAA relational shape metrics
- Report generation for LA/LAA shape/radiomics interpretation

## Extension: Thrombus-Inclusive SLAAO Framework

See `subprojects/la_laa_slaao/` for the extension that adds:

- Thrombus-inclusive anatomical LAA segmentation
- SLAAO multi-label filling-state classification
- CT-native foundation encoder integration
- Visual tokenization of filling states
- Filling-defect and uncertainty mapping
- Positive and negative anatomical prior fusion

Full specification: `docs/protocols/laa_slaao_framework.md`

## Canonical Data Flow

1. Input: defaced CTA NIfTI (`<BIDS_ROOT>/derivatives/defaced/...`)
2. Segment: LA/LAA masks in `derivatives/nudf_la/` and/or `derivatives/nv_segment_ct_laa/`
3. Mesh export: LA/LAA meshes
4. Batch metrics: LA-LAA geometric/relational metrics CSV
5. Report: HTML summary with optional integrated clustering

## Scripts Used

- `scripts/run_cardiac_ct_explorer_nudf_only.py`
- `scripts/run_cardiac_ct_explorer_laa.py`
- `scripts/run_nv_segment_ct_laa.py`
- `scripts/run_laa_shape_descriptors.py`
- `scripts/run_la_laa_metrics_batch.py`
- `scripts/generate_la_laa_shape_report.py`
- `scripts/build_radiomics_manifest_nudf_la.py`

## Protocol References

- `docs/protocols/laa_highres_dataset_setup.md`
- `docs/protocols/laa_slaao_framework.md`
- `docs/NEXT_STEPS_RADIOMICS_MESH_DEFACE.md`

## Minimal Run Skeleton

```bash
# 1) LAA segmentation (NUDF)
python <PROJECT_ROOT>/scripts/run_cardiac_ct_explorer_nudf_only.py \
  --input <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/cardiac_ct_explorer \
  --laa-output <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \
  --run-totalseg

# 2) Mesh + descriptors
python <PROJECT_ROOT>/scripts/run_laa_shape_descriptors.py \
  --input <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/shape_meshes/<CASE_ID>

# 3) Batch metrics
python <PROJECT_ROOT>/scripts/run_la_laa_metrics_batch.py \
  --mesh-root <BIDS_ROOT>/derivatives/shape_meshes \
  --out-csv <BIDS_ROOT>/derivatives/shape_meshes/la_laa_metrics_batch.csv

# 4) Report
python <PROJECT_ROOT>/scripts/generate_la_laa_shape_report.py \
  --metrics-csv <BIDS_ROOT>/derivatives/shape_meshes/la_laa_metrics_batch.csv \
  --mesh-root <BIDS_ROOT>/derivatives/shape_meshes \
  --output-html <BIDS_ROOT>/derivatives/shape_meshes/la_laa_shape_report.html
```
