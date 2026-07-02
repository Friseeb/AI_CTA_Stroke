# Vertebral Artery Manual Segmentation (3D Slicer)

This subproject documents the manual vertebral artery labeling workflow and
provides a Slicer script to fix common saving issues (misaligned centers /
transforms not hardened).

## Folder Layout
```
subprojects/vertebral_manual/
  README.md
  docs/
    SOP.md
  scripts/
    fix_slicer_save.py
  templates/
    manifest_template.csv
```

## Quick Start (Slicer)
Use the guided Slicer module:
`subprojects/vertebral_manual/slicer_module/CTAVertebralWizard.py`

See: `subprojects/vertebral_manual/slicer_module/README.md`

## Guided Vertebral Review

The Slicer modules now support a guided single-case bilateral vertebral review.
The wizard uses explicit CTA and vertebral segmentation/label selectors. It can
load existing pipeline `.nii.gz` subjects, create left/right vertebral
centerline curves, load optional vertebral foramen masks as negative-prior
context, open an editable bilateral segmentation when needed, and export a
CTA-aligned bilateral labelmap with JSON/CSV review logs.

Output label contract:
- label `1`: `Vert L`
- label `2`: `Vert R`

Foramen negative priors are not exported as labels. They are used for QC
context and logged as overlap checks against `Vert L/R`.

## Queue Mode

For batch manual review, use a manifest instead of preloading every subject in
Slicer. The wizard loads one case, finalizes it, appends a queue-status row,
clears the current case nodes, and then loads the next case.

Current discovered manifest:
`subprojects/vertebral_manual/manifests/discovered_subjects_manifest.csv`

Regenerate it from existing subject NIfTIs:
```
python subprojects/vertebral_manual/scripts/discover_subject_niigz.py \
  --cta-root external/CTA-DEFACE/batch_input \
  --out subprojects/vertebral_manual/manifests/discovered_subjects_manifest.csv
```

## Notes
- This script hardens all transforms before exporting a new labelmap aligned to
  the CTA grid, which resolves most "center" / misalignment issues.
- The output labelmap is saved as `<case_id>_vert_clean.nii.gz`.
