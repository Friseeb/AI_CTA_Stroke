# Script Setup (Golden Path)

This is the recommended, minimal script set for routine use.

For a full script map + naming rubric, see:

- `scripts/README.md`

Architecture references:

- `docs/architecture/ORGANIZATION_BLUEPRINT.md`
- `docs/architecture/MIGRATION_MAP.md`

## Organization Decision (Recommended)

Use stage-first backbone scripts, then apply ROI/study specificity by config:

- Backbone stages: ingest -> deface -> segmentation -> analysis -> reports
- ROI configs: `configs/roi/*.yaml`
- Study profiles: `configs/profiles/*.yaml`

Do not duplicate scripts per anatomy unless technically necessary.

## 1) Environments (one-time)

Use only the env setup scripts below:

```bash
bash <PROJECT_ROOT>/scripts/setup_totalseg_env_mac.sh
bash <PROJECT_ROOT>/scripts/setup_topcow_env.sh
bash <PROJECT_ROOT>/scripts/setup_nv_segment_ct_env.sh
bash <PROJECT_ROOT>/scripts/setup_cardiac_ct_explorer_env.sh
# Optional (LA/LAA mesh/shape analysis):
bash <PROJECT_ROOT>/scripts/setup_laa_shape_env.sh
```

Notes:
- On Linux/CUDA, use `scripts/setup_totalseg_env_cuda.sh` instead of the mac script.
- Default env names expected by orchestrators:
  - `totalseg-mac`
  - `topcow_claim`
  - `nv-segment-ct`
  - `cardiac-ct-explorer`

## 2) Single-Case End-to-End

Primary orchestrator:

- `scripts/run_full_segmentation_pipeline.py`

Example:

```bash
python <PROJECT_ROOT>/scripts/run_full_segmentation_pipeline.py \
  --input-nifti <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/full_pipeline/<CASE_ID> \
  --case-id <CASE_ID> \
  --run-totalseg \
  --run-topcow \
  --run-nv \
  --run-nudf \
  --merge-labels \
  --topcow-yolo-model <TOPCOW_YOLO_MODEL_PT> \
  --topcow-nnunet-model-dir <TOPCOW_MODEL_DIR>
```

## 3) Batch End-to-End

Batch wrapper:

- `scripts/run_full_segmentation_batch.py`

Manifest template:

- `configs/manifests/cta_inputs.template.csv`

Example:

```bash
python <PROJECT_ROOT>/scripts/run_full_segmentation_batch.py \
  --manifest <PROJECT_ROOT>/configs/manifests/cta_inputs.template.csv \
  --output-root <BIDS_ROOT>/derivatives/full_seg_batch \
  --run-totalseg \
  --run-topcow \
  --run-nv \
  --run-nudf \
  --merge-labels \
  --topcow-yolo-model <TOPCOW_YOLO_MODEL_PT> \
  --topcow-nnunet-model-dir <TOPCOW_MODEL_DIR>
```

## 4) LA/LAA Substudy

Use this sequence:

1. `scripts/run_cardiac_ct_explorer_nudf_only.py`
2. `scripts/run_laa_shape_descriptors.py`
3. `scripts/run_la_laa_metrics_batch.py`
4. `scripts/generate_la_laa_shape_report.py`

See also:

- `subprojects/la_laa/README.md`
- `configs/profiles/p60_analysis_la_laa.yaml`

## 5) Vertebral Manual Substudy

Use only:

- `subprojects/vertebral_manual/` (single canonical location)

## 6) Circle of Willis Substudy

Primary script:

- `scripts/run_topcow_claim.py`

Profile:

- `configs/profiles/p60_analysis_intracranial.yaml`

## 7) Carotid Substudy

Primary segmentation source:

- `scripts/run_full_segmentation_pipeline.py` or `scripts/run_full_segmentation_batch.py`

Profile:

- `configs/profiles/p60_analysis_carotid.yaml`

## 8) DICOM/NIfTI + Defacing

Use:

- DICOM -> NIfTI: `scripts/convert_daylightdicom_to_bids.py`
- Batch deface: `scripts/run_cta_deface_dl_batch.py`
- Single deface: `scripts/deface_cta_simple.py`

## 9) Scripts To Avoid In Routine Runs

These are experimental/specialized and not part of the default workflow:

- `create_vessel_mask_stepwise.py`
- `create_vessel_mask_v2.py`
- `create_vessel_mask_seeded.py`
- `batch_deface.py` (legacy helper; prefer `run_cta_deface_dl_batch.py`)
- ad-hoc clustering/report utilities unless needed for a specific analysis
