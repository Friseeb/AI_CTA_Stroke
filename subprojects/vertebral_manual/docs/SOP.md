# SOP: Manual Vertebral Artery Segmentation (3D Slicer)

## Purpose
Create high-quality manual vertebral artery labels for CTA scans when no
public model weights are available.

## Inputs
- CTA volume (`.nii.gz`)
- Manual labelmap (`.nii.gz`) created in Slicer

## Standard Steps
1. Load CTA and labelmap in Slicer.
2. Verify voxel spacing and orientation are correct.
3. Inspect whether either node is under a transform; apply/harden if needed.
4. Run the fix script to export a cleaned labelmap aligned to the CTA grid.

## Fix Script
Script path:
`AI_CTA_Stroke/subprojects/vertebral_manual/scripts/fix_slicer_save.py`

What it does:
- Finds CTA + labelmap by substring
- Hardens any transforms
- Imports the labelmap into a segmentation
- Exports to a new labelmap aligned to CTA geometry
- Saves CTA + cleaned labelmap

## Output
`<OUTPUT_DIR>/sub-XXX-acq-CTA_ct_clean.nii.gz`  
`<OUTPUT_DIR>/sub-XXX-acq-CTA_ct_vert_clean.nii.gz`

## Suggested Naming
- CTA: `sub-XXX_acq-CTA_ct.nii.gz`
- Label: `sub-XXX_acq-CTA_ct_vert.nii.gz`

## QC Checklist
- Overlay label on CTA in axial/sagittal/coronal
- Verify arteries are continuous
- Check for off-by-one slice shifts
- Confirm no transform warnings remain
