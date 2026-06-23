# cta-dental-opportunistic-screening

> **RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS.**
> All outputs are experimental candidate markers only.
> This tool is not validated for clinical use and must not inform patient care decisions.

A research-grade Python prototype for opportunistic adult dental and oral-health analysis from routine head/neck CT angiography (CTA). The pipeline detects a dentition/dentoalveolar ROI, runs open-source dental segmentation models within that ROI, and produces conservative candidate markers and QC images.

---

## Limitations — read before using

### CTA vs CBCT domain shift
- Dental CBCT: high resolution (0.1–0.4 mm), no contrast, focused FOV, excellent bone/root detail.
- Head/neck CTA: lower resolution (0.5–1.0 mm), contrast-enhanced vascular kernel, wide FOV, significant beam-hardening artefacts from metal.
- All CBCT-trained models (DentalSegmentator, OralSeg, RAIL) are **cbct_to_cta_unvalidated**. Domain shift is expected.

### What CTA can support (with caveats)
- Gross tooth presence/absence.
- Large implants, crowns, bridges (if the segmentation model labels them).
- Large periapical lucencies (experimental, low confidence).
- Gross mandible/maxilla anatomy.

### What CTA cannot reliably support
- Subtle caries (insufficient resolution and soft-tissue contrast) — **not implemented**.
- Gingivitis, plaque (not visible on CT) — **not implemented**.
- Mucosal disease (requires clinical or MRI assessment) — **not implemented**.
- Mild/moderate periodontal staging — **not implemented in v1**.
- Root fractures — **not implemented**.
- Pain, halitosis — clinical findings; cannot be assessed from imaging.

---

## Installation

### Core dependencies

```bash
pip install cta-dental-opportunistic-screening
# or from source:
pip install -e ".[dicom,dev]"
```

This package depends on the repo's shared `cta_common` utilities (geometry/IO),
which are not published to PyPI. When installing from source, also install it:

```bash
pip install -e ../../cta_common   # from this subproject dir
```

Requires Python ≥ 3.10.

Core Python dependencies (installed automatically):
- `typer`, `rich` — CLI
- `SimpleITK` — primary image I/O and resampling
- `nibabel` — secondary NIfTI support
- `numpy`, `scipy`, `scikit-image` — array/image processing
- `pydantic` ≥ 2.6 — config and report schemas
- `matplotlib` — QC PNG generation
- `pyyaml` — config file parsing
- `pydicom` (optional) — DICOM metadata extraction

### External CLI dependencies

#### TotalSegmentator (required for default ROI and segmentation)
```bash
pip install TotalSegmentator
```
TotalSegmentator will auto-download model weights on first use (~500 MB).
For air-gapped environments, download weights manually and set `--weights-dir`.

Relevant tasks used by this pipeline:
- `teeth` — primary dental segmentation task (preferred)
- `craniofacial_structures` — fallback for ROI detection

#### nnU-Net v2 (required for DentalSegmentator backend)
```bash
pip install nnunetv2
```

#### OralSeg (optional experimental backend)
```bash
pip install torch torchvision huggingface_hub
# Clone OralSeg repository:
git clone https://github.com/OttoYouZhou/oralseg
# Integrate inference script — see segmenters/oralseg.py
```

#### RAIL (optional experimental backend)
```bash
pip install torch torchvision huggingface_hub
# Clone RAIL repository:
git clone https://github.com/Tournesol-Saturday/RAIL
# Integrate inference script — see segmenters/rail.py
```

---

## Model weights

### DentalSegmentator (Dataset112)
Source: [Zenodo record 10829675](https://zenodo.org/records/10829675)

```bash
wget https://zenodo.org/records/10829675/files/Dataset112_DentalSegmentator_v100.zip
```

Pass the downloaded path to the CLI:
```bash
cta-dental segment --segmenter dentalsegmentator \
  --dentalseg-weights /data/weights/Dataset112_DentalSegmentator_v100.zip \
  ...
```

Or unpack manually and pass the results folder:
```bash
unzip Dataset112_DentalSegmentator_v100.zip -d /data/nnunet_results/
# Then configure nnunet_results_dir in configs/default.yaml
```

### TotalSegmentator
Weights are downloaded automatically on first use. To pre-download:
```bash
TotalSegmentator --task teeth --download-weights-only
```

---

## Example commands

### Convert DICOM to NIfTI
```bash
cta-dental convert /data/dicoms/patient001/ --out /data/nifti/patient001.nii.gz
```

### Detect dentition ROI
```bash
cta-dental roi /data/nifti/patient001.nii.gz \
  --out /data/outputs/patient001/roi/ \
  --roi-method totalseg_teeth \
  --target-spacing 0.5
```

### Run segmentation
```bash
cta-dental segment /data/outputs/patient001/roi/dentition_roi.nii.gz \
  --out /data/outputs/patient001/seg/ \
  --segmenter totalseg_teeth
```

### Extract candidate features
```bash
cta-dental features /data/nifti/patient001.nii.gz \
  --labels-dir /data/outputs/patient001/seg/ \
  --out /data/outputs/patient001/candidate_features.json \
  --case-id patient001
```

### Full pipeline (recommended)
```bash
cta-dental run /data/dicoms/patient001/ \
  --out /data/outputs/patient001/ \
  --case-id patient001 \
  --roi-method totalseg_teeth \
  --segmenter totalseg_teeth \
  --target-spacing 0.5 \
  --deface-mode mask_only \
  --skip-existing            # reuse a completed segmentation if present (skips re-running the model)
```

`--skip-existing` reuses a prior segmentation in the output dir (validated via its
`labels.json` manifest) instead of re-running the model — useful when re-running a
case to iterate on ROI/features/QC. It is off by default.

### Full pipeline with DentalSegmentator backend
```bash
cta-dental run /data/dicoms/patient001/ \
  --out /data/outputs/patient001/ \
  --case-id patient001 \
  --roi-method dentalsegmentator_coarse \
  --segmenter dentalsegmentator \
  --dentalseg-weights /data/weights/Dataset112_DentalSegmentator_v100.zip \
  --target-spacing 0.5 \
  --deface-mode mask_only
```

### Threshold fallback (no external models required — degraded mode)
```bash
cta-dental run /data/nifti/patient001.nii.gz \
  --out /data/outputs/patient001/ \
  --case-id patient001 \
  --roi-method threshold_fallback \
  --segmenter none \
  --deface-mode none
```
> Warning: threshold_fallback produces poor-quality ROI and disables disease feature extraction.

---

## Defacing / privacy guidance

This pipeline handles PHI carefully:

- **Metadata de-identification**: automatically strips patient name, MRN, accession number, DOB, and raw study dates from all JSON sidecars.
- **Pixel defacing**: four modes are available via `--deface-mode`:

| Mode | Description |
|------|-------------|
| `none` | No defacing. Analysis input unmodified. Default for research use with stored data. |
| `mask_only` | Computes face/privacy mask but does not alter the analysis image. Default for new data. |
| `posthoc` | Creates a defaced export copy only. Analysis uses the original unmodified image. |
| `pre` | **Defacing applied before analysis.** Loud warning: segmentation performance may degrade. Undefaced working image preserved in protected intermediate folder. |

**Do not use `--deface-mode pre` as the default**. Facial soft-tissue context may be used by models for jaw/tooth localisation. If defacing for export/publication, use `posthoc`.

**Never upload images or NIfTI files to cloud services** through this tool. All model inference is local.

---

## Output structure

```
<outdir>/
  report.json                    # DentalReport — pipeline provenance
  candidate_features.json        # Candidate markers (experimental)
  preprocessed.nii.gz            # Resampled, reoriented CTA
  preprocessing_meta.json        # Metadata sidecar (de-identified)
  roi/
    dentition_roi.nii.gz         # Cropped CTA in dentition ROI
    roi_mask.nii.gz              # Binary ROI mask (original space)
    roi_bbox.json                # Bounding box (voxel + physical coords)
  segmentations/<segmenter>/
    *.nii.gz                     # Per-label binary masks
    labels.json                  # Label manifest
  deid/
    face_mask.nii.gz             # (mask_only mode) face region mask
    defaced_export.nii.gz        # (posthoc mode) defaced copy
  qc/
    roi_axial.png
    roi_coronal.png
    roi_sagittal.png
    roi_mip.png
    seg_axial.png
    seg_coronal.png
    seg_sagittal.png
    qc_summary.json
```

---

## Configuration

Edit `configs/default.yaml` or pass `--config /path/to/custom.yaml`.

Key options:
```yaml
preprocessing:
  target_spacing_mm: 0.5      # Isotropic resampling target
  orientation: "RAS"

roi:
  method: "totalseg_teeth"
  margin_mm: 20.0             # ROI expansion around label bbox

deface:
  mode: "mask_only"

segmentation:
  backend: "totalseg_teeth"

features:
  allow_threshold_fallback_features: false  # must be explicitly enabled
```

---

## Development and testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests use synthetic 3D arrays only — no real patient data required.

---

## Related open-source tools

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator)
- [OralSeg](https://github.com/OttoYouZhou/oralseg)
- [RAIL](https://github.com/Tournesol-Saturday/RAIL)
- [CTA-DEFACE](https://github.com/CCI-Bonn/CTA-DEFACE)
- [CT-Defacer](https://github.com/MaximilianLindholz/CT-Defacer)
- [ToothFairy2](https://github.com/AImageLab-zip/ToothFairy2-Benchmark)

---

## Disclaimer

This software is a **research prototype**. It is:
- Not CE marked or FDA cleared.
- Not validated for clinical diagnostic use.
- Not suitable for patient screening, diagnosis, or treatment planning.
- Provided without warranty of any kind.

All candidate markers produced by this pipeline are **experimental** and must be interpreted by qualified dental/medical professionals using appropriate clinical context and validated imaging.

CTA is not the primary modality for dental disease detection. All CBCT-trained model outputs are **cbct_to_cta_unvalidated**.
