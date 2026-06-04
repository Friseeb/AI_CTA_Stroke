# AI_CTA_Stroke Master Workflow

This repository is organized around a single canonical flow:

1. DICOM -> NIfTI  
2. NIfTI defacing  
3. Branch into focused substudies:
   - LA/LAA substudy
   - CTA vertebral optimization substudy
   - CTA carotids substudy
   - Circle of Willis substudy

## Recommended Organization Model

Use a hybrid model:

1. Backbone by pipeline stage (ingest -> privacy -> segmentation -> analysis -> reports)
2. ROI behavior from configs (`configs/roi/*.yaml`)
3. Study combinations from profiles (`configs/profiles/*.yaml`)

This avoids duplicated scripts while keeping study-specific outputs clear.

Architecture docs:

- `docs/architecture/ORGANIZATION_BLUEPRINT.md`
- `docs/architecture/MIGRATION_MAP.md`

## Path Placeholders

Use these placeholders in commands (do not hardcode local machine paths):

- `<PROJECT_ROOT>`: local clone of this repository
- `<DATA_ROOT>`: dataset root folder
- `<DICOM_ROOT>`: raw DICOM input root
- `<BIDS_ROOT>`: NIfTI dataset root (for example, DAYLIGHTBIDS-style structure)
- `<CASE_ID>`: case identifier (for example `sub-001_acq-CTA_ct`)
- `<CASE_NIFTI>`: single-case CTA NIfTI path
- `<CASE_DICOM_DIR>`: single-case DICOM folder
- `<TOPCOW_MODEL_DIR>`: directory containing `topcow-claim-models`
- `<TOPCOW_YOLO_MODEL_PT>`: path to `yolo-cow-detection.pt`

## Stage 1: DICOM -> NIfTI

### Cohort conversion

```bash
python <PROJECT_ROOT>/scripts/convert_daylightdicom_to_bids.py \
  --src-root <DICOM_ROOT> \
  --out-root <BIDS_ROOT>
```

### Single case conversion (dcm2niix)

```bash
dcm2niix -z y -f <CASE_ID> -o <BIDS_ROOT> <CASE_DICOM_DIR>
```

## Stage 2: NIfTI Defacing

### Single case

```bash
python <PROJECT_ROOT>/scripts/deface_cta_simple.py \
  --input <CASE_NIFTI> \
  --output <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz
```

### Batch

```bash
python <PROJECT_ROOT>/scripts/run_cta_deface_dl_batch.py \
  --input-dir <BIDS_ROOT> \
  --output-dir <BIDS_ROOT>/derivatives/defaced \
  --mask-dir <BIDS_ROOT>/derivatives/deface_masks
```

## Substudy A: LA/LAA

Subproject home:

- `subprojects/la_laa/README.md`

### Segmentation (NUDF / CardiacCTExplorer)

```bash
python <PROJECT_ROOT>/scripts/run_cardiac_ct_explorer_nudf_only.py \
  --input <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/cardiac_ct_explorer \
  --laa-output <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \
  --run-totalseg
```

### Optional alternate LAA segmentation (NV-Segment-CT)

```bash
python <PROJECT_ROOT>/scripts/run_nv_segment_ct_laa.py \
  --input <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output <BIDS_ROOT>/derivatives/nv_segment_ct_laa/<CASE_ID>_laa108.nii.gz
```

### Shape pipeline

1. Mesh extraction: `scripts/run_laa_shape_descriptors.py`
2. LA/LAA metrics batch: `scripts/run_la_laa_metrics_batch.py`
3. HTML report: `scripts/generate_la_laa_shape_report.py`

## Substudy B: CTA Vertebral Optimization

Manual workflow is isolated in:

- `subprojects/vertebral_manual/`

Primary references:

- `subprojects/vertebral_manual/README.md`
- `subprojects/vertebral_manual/slicer_module/README.md`

## Substudy C: CTA Carotids

Recommended approach: run TotalSegmentator + merged multi-label map, then use carotid labels.

```bash
python <PROJECT_ROOT>/scripts/run_full_segmentation_pipeline.py \
  --input-nifti <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/full_pipeline/<CASE_ID> \
  --case-id <CASE_ID> \
  --run-totalseg \
  --merge-labels
```

Carotid labels in merged output:

- `4`: common_carotid_artery_left
- `5`: common_carotid_artery_right
- `6`: internal_carotid_artery_left
- `7`: internal_carotid_artery_right

## Substudy D: Circle of Willis (TopCoW)

```bash
python <PROJECT_ROOT>/scripts/run_topcow_claim.py \
  --input <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_defaced.nii.gz \
  --output <BIDS_ROOT>/derivatives/topcow/<CASE_ID> \
  --nnunet-model-dir <TOPCOW_MODEL_DIR> \
  --yolo-model <TOPCOW_YOLO_MODEL_PT>
```

Optional integration into merged labelmap is supported by:

- `scripts/run_full_segmentation_pipeline.py` (`--run-topcow --merge-labels`)
- `scripts/build_all_segmentations_labelmap.py`

## End-to-End Orchestrator

If you want one wrapper from DICOM/NIfTI through segmentation outputs:

```bash
python <PROJECT_ROOT>/scripts/run_full_segmentation_pipeline.py --help
```

This wrapper supports DICOM->NIfTI, defacing, TotalSegmentator, TopCoW, NV, NUDF, and merged labels.

## Repository Notes

- Subproject boundaries: `SUBPROJECTS.md`
- Script setup guide: `docs/protocols/script_setup.md`
- Script map + naming rubric: `scripts/README.md`
- Segmentation subproject: `subprojects/segmentation/README.md`
- Analysis subproject: `subprojects/analysis/README.md`
- Study presets: `subprojects/studies/README.md`
- Centerline stack is intentionally removed for now
- Avoid committing machine-specific absolute paths in scripts/docs
