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
Use the one-click Slicer module:
`subprojects/vertebral_manual/slicer_module/CTAFinalizeSOP.py`

See: `subprojects/vertebral_manual/slicer_module/README.md`

## Notes
- This script hardens all transforms before exporting a new labelmap aligned to
  the CTA grid, which resolves most "center" / misalignment issues.
- The output labelmap is saved as `<label>_clean.nii.gz`.
