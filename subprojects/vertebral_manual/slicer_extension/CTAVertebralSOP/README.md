# CTAVertebralSOP Extension (3D Slicer)

This is a scripted extension that bundles two modules:
- **CTAVertebralWizard** (guided bilateral vertebral review workflow)
- **CTAFinalizeSOP** (finalize/save clean bilateral labelmap)

No console usage is required once installed.

## Install via Extension Manager (local)
1. Open **3D Slicer**.
2. Enable **Developer Tools** (if not already):
   - `Slicer -> Preferences -> Developer -> Enable Developer Mode`
   - Restart Slicer.
3. Open **Extension Wizard**: `View -> Extension Wizard`.
4. Click **Add Extension** and select this folder:
   - `.../AI_CTA_Stroke/subprojects/vertebral_manual/slicer_extension/CTAVertebralSOP`
5. Click **Install** (or **Load** for a dev install).
6. Restart Slicer.

You will now find the modules in the Modules list:
- `CTAVertebralWizard`
- `CTAFinalizeSOP`

## Notes
- Scripted-only extension (no compilation needed).
- If you edit the Python files, restart Slicer to reload.
- The wizard can load existing pipeline CTA and vertebral label `.nii.gz`
  subjects before manual correction.
- Queue mode loads one manifest case at a time, writes queue status, clears
  current case nodes, and advances to the next case.
- Optional vertebral foramen masks can be loaded as negative-prior QC context;
  overlap with labels `1/2` is logged but not automatically erased.
- Output label contract: `1 = Vert L`, `2 = Vert R`.
- Finalization writes the clean NIfTI, JSON review log, append-only CSV row,
  optional centerline markup JSON, and optional MRML scene snapshot.
