# CTAVertebralSOP Extension (3D Slicer)

This is a scripted extension that bundles two modules:
- **CTAVertebralWizard** (guided SOP workflow)
- **CTAFinalizeSOP** (finalize/save clean labelmap)

No console usage is required once installed.

## Install via Extension Manager (local)
1. Open **3D Slicer**.
2. Enable **Developer Tools** (if not already):
   - `Slicer -> Preferences -> Developer -> Enable Developer Mode`
   - Restart Slicer.
3. Open **Extension Wizard**: `View -> Extension Wizard`.
4. Click **Add Extension** and select this folder:
   - `.../AI_CTA_Stroke/vertebral_manual/slicer_extension/CTAVertebralSOP`
5. Click **Install** (or **Load** for a dev install).
6. Restart Slicer.

You will now find the modules in the Modules list:
- `CTAVertebralWizard`
- `CTAFinalizeSOP`

## Notes
- Scripted-only extension (no compilation needed).
- If you edit the Python files, restart Slicer to reload.
