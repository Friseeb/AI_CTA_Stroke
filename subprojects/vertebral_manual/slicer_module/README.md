## CTA Finalize / SOP Slicer Module

Drop-in scripted module for one-click CTA + vertebral label finalization.

### Install in Slicer (macOS)
1. Open Slicer.
2. `Slicer → Settings → Modules`
3. Add this folder to “Additional module paths”:
   `AI_CTA_Stroke/subprojects/vertebral_manual/slicer_module`
4. Restart Slicer.

### Use
1. Load the CTA and vertebral labelmap (name contains `vert`).
2. Open the **CTA Finalize / SOP** module.
3. Click **Finalize Current Case**.

### Output
Saved next to the label file (or CTA if label has no path):
```
sub-XXX_vert_clean.nii.gz
sub-XXX_vert_clean_log.json
```

### Behavior (SOP)
- Hardens all transforms
- Fixes cropped labelmaps
- Enforces CTA-aligned geometry
- Saves ITK-SNAP + ML-safe NIfTIs
- No transforms saved

---

## CTA Vertebral Wizard (Step-by-Step)

File:
`subprojects/vertebral_manual/slicer_module/CTAVertebralWizard.py`

Provides a minimal SOP wizard:
1. Select CTA
2. Draw curve
3. Segment (Segment Editor)
4. Finalize and save clean labelmap

Wizard also logs user parameters (tube diameter, tolerance, notes) into the
`*_vert_clean_log.json` alongside the output label.
