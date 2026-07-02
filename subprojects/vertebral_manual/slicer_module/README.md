## CTA Finalize / SOP Slicer Module

Drop-in scripted module for one-click CTA + vertebral label finalization.

### Install in Slicer (macOS)
1. Open Slicer.
2. `Slicer → Settings → Modules`
3. Add this folder to “Additional module paths”:
   `AI_CTA_Stroke/subprojects/vertebral_manual/slicer_module`
4. Restart Slicer.

### Use
1. Load or select the CTA volume.
2. Open the **CTA Vertebral Wizard** module.
3. Enter reviewer ID and review status.
4. Load an existing vertebral `.nii.gz` labelmap if the pipeline already has one.
5. Load optional vertebral foramen `.nii.gz` masks as negative-prior context.
6. Create/select the bilateral centerline curves.
7. Create/edit the vertebral segmentation only when manual correction is needed.
8. Edit or confirm the two required segments:
   - `Vert L` -> output label `1`
   - `Vert R` -> output label `2`
9. Click **Run Finalize SOP**.

### Queue Mode
1. Open **CTA Vertebral Wizard**.
2. Select a manifest CSV.
3. Click **Load Manifest**.
4. Review/edit the loaded case.
5. Click **Finalize + Next Queue Case**.

The wizard writes an append-only `*_queue_status.csv`, clears the current case
nodes from Slicer, and loads the next case. This avoids preloading all subjects
and prevents stale nodes from being saved into the wrong case.

To generate a manifest from existing subject CTAs:
```
python subprojects/vertebral_manual/scripts/discover_subject_niigz.py \
  --cta-root external/CTA-DEFACE/batch_input \
  --out subprojects/vertebral_manual/manifests/discovered_subjects_manifest.csv
```

### Output
Saved next to the label file (or CTA if label has no path):
```
sub-XXX_vert_clean.nii.gz
sub-XXX_vert_clean_log.json
sub-XXX_vertebral_review.csv
sub-XXX_vertebral_centerlines.mrk.json  # if curves contain points
sub-XXX_vertebral_review.mrml           # scene snapshot when saved from wizard
```

### Behavior (SOP)
- Hardens all transforms
- Fixes cropped labelmaps
- Enforces CTA-aligned geometry
- Logs vertebral foramen negative-prior overlap with labels `1/2` when supplied
- Saves ITK-SNAP + ML-safe NIfTIs
- No transforms saved

---

## CTA Vertebral Wizard (Step-by-Step)

File:
`subprojects/vertebral_manual/slicer_module/CTAVertebralWizard.py`

Provides a minimal SOP wizard:
1. Select CTA
2. Load/select existing vertebral NIfTI label or segmentation
3. Load/select foramen negative-prior context if available
4. Draw bilateral curves
5. Edit in Segment Editor if needed
6. Finalize and save clean labelmap

Wizard also logs user parameters (tube diameter, tolerance, notes) into the
`*_vert_clean_log.json` alongside the output label.

## Review Log

The wizard writes structured review metadata into JSON and appends a one-row
CSV log for each finalized case. The log includes reviewer ID, status, CTA path,
input segmentation/label path, output label path, scene path, centerline export
path, negative-prior overlap summary, and free-text notes.
