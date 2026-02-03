# nnUNetv2 CTA Artery Segmentation (MPS)

This pipeline targets Mac MPS for training/inference and follows the Canals-style configuration:
- Optimizer: SGD with Nesterov momentum (mu=0.99)
- Loss: Dice + CE/BCE (nnUNetv2 default for binary/region targets)
- LR schedule: ReduceLROnPlateau (initial LR 0.01)

## 1) Environment (MPS)

```
conda activate cta_centerline_mps_julia
```

nnUNetv2 is installed in this env and supports `-device mps`.

## 2) Dataset layout

Create a standard nnUNetv2 dataset:

```
nnUNet_raw/
  DatasetXXX_CTAArteries/
    imagesTr/
      case_0000.nii.gz
    labelsTr/
      case.nii.gz
    imagesTs/
      case_0000.nii.gz
    dataset.json
```

## 3) Plan + preprocess

```
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

nnUNetv2_plan_and_preprocess -d XXX -c 3d_fullres
```

## 4) Install custom trainer

This repo includes a custom trainer class:
`AI_CTA_Stroke/nnunet/trainers/nnUNetTrainer_CTA_ReduceLROnPlateau.py`

Install it into nnUNetv2:

```
/opt/anaconda3/envs/cta_centerline_mps_julia/bin/python \
  AI_CTA_Stroke/scripts/install_nnunet_trainer.py \
  --trainer AI_CTA_Stroke/nnunet/trainers/nnUNetTrainer_CTA_ReduceLROnPlateau.py
```

## 5) Train on MPS

```
nnUNetv2_train -d XXX -c 3d_fullres -f 0 \
  -tr nnUNetTrainer_CTA_ReduceLROnPlateau \
  -device mps
```

Notes:
- MPS supports single-GPU training only (no DDP).
- Training is slower than CUDA but works on Apple Silicon.

## 6) Inference on MPS

```
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/predictions \
  -d XXX -c 3d_fullres -f 0 \
  -tr nnUNetTrainer_CTA_ReduceLROnPlateau \
  -device mps
```

After prediction, feed the artery mask into the centerline pipeline.
