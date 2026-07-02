# SOP: Manual Vertebral Artery Segmentation (3D Slicer)

## Purpose
Create high-quality manual vertebral artery labels for CTA scans when no
public model weights are available.

## Inputs
- CTA volume (`.nii.gz`)
- Existing pipeline vertebral labelmap (`.nii.gz`) or manual segmentation in Slicer
- Optional vertebral foramen mask(s) (`.nii.gz`) used as negative-prior QC context

## Standard Steps
1. Load CTA in Slicer.
2. Open `CTA Vertebral Wizard`.
3. Select the CTA, output folder, reviewer ID, and review status.
4. Create or select `vertebral_centerline_L` and `vertebral_centerline_R`.
5. Load the existing vertebral `.nii.gz` labelmap if one is already available.
6. Load vertebral foramen mask(s) as negative priors if available.
7. Create or select the editable vertebral segmentation if manual correction is needed.
8. Ensure the final segmentation/labelmap follows the required bilateral contract:
   - `Vert L` -> output label `1`
   - `Vert R` -> output label `2`
9. Edit the two segments in Segment Editor when needed.
10. Run Finalize SOP to export a cleaned labelmap aligned to the CTA grid.

## Queue Review
Use queue mode for multi-subject review. Do not preload all subjects.

1. Select the manifest CSV in the `Queue` section.
2. Click `Load Manifest`.
3. Review the loaded case.
4. Click `Finalize + Next Queue Case`.
5. The wizard saves the outputs, appends `*_queue_status.csv`, clears current
   case nodes from Slicer, and loads the next case.

Generate a manifest from available subject CTAs:
```
python subprojects/vertebral_manual/scripts/discover_subject_niigz.py \
  --cta-root external/CTA-DEFACE/batch_input \
  --out subprojects/vertebral_manual/manifests/discovered_subjects_manifest.csv
```

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
`<OUTPUT_DIR>/sub-XXX_vert_clean.nii.gz`
`<OUTPUT_DIR>/sub-XXX_vert_clean_log.json`
`<OUTPUT_DIR>/sub-XXX_vertebral_review.csv`
`<OUTPUT_DIR>/sub-XXX_vertebral_centerlines.mrk.json` if centerline curves contain points
`<OUTPUT_DIR>/sub-XXX_vertebral_review.mrml` when saved from the wizard

## Suggested Naming
- CTA: `sub-XXX_acq-CTA_ct.nii.gz`
- Label: `sub-XXX_acq-CTA_ct_vert.nii.gz`

## QC Checklist
- Overlay label on CTA in axial/sagittal/coronal
- Confirm label `1` is left vertebral artery and label `2` is right vertebral artery
- Confirm any vertebral foramen negative-prior overlap warnings are reviewed
- Verify arteries are continuous
- Check for off-by-one slice shifts
- Confirm no transform warnings remain
