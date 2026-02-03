# TopCoW 2024 (CLAIM) Inference Setup

This repo includes a wrapper script for the TopCoW 2024 winning solution from the Charité CLAIM group.

## Model output labels

The output segmentation labels are:

```
0  background
1  BA
2  R-PCA
3  L-PCA
4  R-ICA
5  R-MCA
6  L-ICA
7  L-MCA
8  R-Pcom
9  L-Pcom
10 Acom
11 R-ACA
12 L-ACA
15 3rd-A2
```

Note: CLAIM's inference script converts label 13 -> 15 for 3rd-A2 to match the TopCoW label set.

## Dependencies (per CLAIM README)

Suggested environment:

```
conda create -n topcow_claim python=3.11
conda activate topcow_claim
pip install ultralytics
cd /path/to/topcow-2024-nnunet
pip install -e .
```

You also need:
- torch (with CUDA if available)
- opencv-python
- SimpleITK
- batchgenerators

### One-shot env setup (included)

```
bash AI_CTA_Stroke/scripts/setup_topcow_env.sh --cuda
```

Use `--cpu` for CPU-only installs or `--env-name` to override the env name.

## Download weights

Download weights from Zenodo (TopCoW 2024 CLAIM release):
- https://zenodo.org/records/14191592

Place files as:
- `models/yolo-cow-detection.pt`
- `models/topcow-claim-models/` (nnUNet model folder)

### Auto-download helper (included)

```
python AI_CTA_Stroke/scripts/download_topcow_claim_weights.py \
  --output /path/to/topcow_weights
```

This creates `/path/to/topcow_weights/models/yolo-cow-detection.pt` and
`/path/to/topcow_weights/models/topcow-claim-models/` when possible.

## Run inference (wrapper)

```
python AI_CTA_Stroke/scripts/run_topcow_claim.py \
  --input /path/to/cta_or_mra.nii.gz \
  --output /path/to/topcow_out \
  --yolo-model /path/to/models/yolo-cow-detection.pt \
  --nnunet-model-dir /path/to/models/topcow-claim-models \
  --labels-json /path/to/topcow_out/topcow_labels.json
```

For folders, pass `--input /path/to/folder` and all `.nii/.nii.gz` files will be processed.

## Integrated into CTA pipeline

You can run TopCoW as part of `scripts/run_cta_pipeline.py`:

```
python -u AI_CTA_Stroke/scripts/run_cta_pipeline.py \
  --input /path/to/cta.nii.gz \
  --output /path/to/output \
  --run-topcow \
  --topcow-yolo-model /path/to/topcow_weights/models/yolo-cow-detection.pt \
  --topcow-nnunet-model-dir /path/to/topcow_weights/models/topcow-claim-models
```

## Multi-label merge (extra + intracranial)

To build a single multi-label NIfTI that includes aorta/subclavians/carotids/LA/LAA
plus TopCoW intracranial labels:

```

By default, TopCoW labels are offset by +100 in the merged label map to avoid
collisions with extracranial labels. Set `--multilabel-topcow-offset 0` if you
want the native TopCoW IDs (note that this will collide with extracranial IDs).
python -u AI_CTA_Stroke/scripts/run_cta_pipeline.py \
  --input /path/to/cta.nii.gz \
  --output /path/to/output \
  --run-topcow \
  --topcow-yolo-model /path/to/topcow_weights/models/yolo-cow-detection.pt \
  --topcow-nnunet-model-dir /path/to/topcow_weights/models/topcow-claim-models \
  --build-multilabel
```

## Notes

- GPU is strongly recommended. Use `--device cuda` or let the script auto-detect.
- Output files are named `*_topcow_seg.nii.gz`.
- Label mapping is also available at `AI_CTA_Stroke/configs/topcow_labels.json`.
