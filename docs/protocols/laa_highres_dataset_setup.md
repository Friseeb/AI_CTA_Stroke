# High-Resolution LAA Dataset Setup (STACOM 2025)

This protocol prepares a high-resolution LAA dataset using:
- **ImageCAS CCTA volumes** (Kaggle)
- **STACOM 2025 ImageCAS labels** (segmentation zip)

The labels include a dedicated **LAA class (label 8)**, which you can extract into
binary masks for LAA-focused training.

## 1) Download data

- ImageCAS CCTA volumes (Kaggle):  
  https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas

- Segmentation labels (zip):  
  https://people.compute.dtu.dk/rapa/STACOM2025/ImageCAS-STACOM2025-02-10-2025.zip

## 2) Extract the segmentation zip

```
python AI_CTA_Stroke/scripts/setup_laa_highres_dataset.py extract \
  --seg-zip /path/to/ImageCAS-STACOM2025-02-10-2025.zip \
  --out-dir /path/to/LAADATA
```

## 3) Create LAA-only masks (label 8)

```
python AI_CTA_Stroke/scripts/setup_laa_highres_dataset.py laa-only \
  --dataset-dir /path/to/LAADATA \
  --out-dir /path/to/LAADATA/labels_laa \
  --full-only
```

`--full-only` uses the list of cases with complete LAAs (no cutoff at scan boundary).

## 4) Build nnUNetv2 dataset structure (optional)

```
python AI_CTA_Stroke/scripts/setup_laa_highres_dataset.py nnunet \
  --images-dir /path/to/ImageCAS/images \
  --labels-dir /path/to/LAADATA/labels_laa \
  --out-dir /path/to/nnUNet_raw/Dataset901_LAA \
  --laa-only
```

This creates `imagesTr/`, `labelsTr/`, and `dataset.json`.

## Notes

- If you want multi-label training (LA/LV/RV/etc.), skip `laa-only` and point
  `--labels-dir` to the original `segmentations/` folder.
- For LAA-only training, labels are reduced to:
  - 0 = background
  - 1 = LAA
