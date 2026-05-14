# Thrombus-Inclusive LAA Segmentation and SLAAO Foundation Framework

## Project Summary

This document specifies the extension of the existing eCTA pipeline (`AI_CTA_Stroke`) to support:

1. Thrombus-inclusive anatomical LAA segmentation
2. SLAAO filling-state representation learning
3. Visual-token / CT foundation-model integration
4. Filling-defect mapping
5. Uncertainty-aware segmentation
6. Radiomics and topology analysis of filling states
7. Positive and negative anatomical priors
8. Residual pathological adaptation on top of healthy anatomical priors

Existing components preserved and extended:

- NUDF implicit anatomical segmentation
- VISTA3D segmentation priors
- TotalSegmentator anatomical priors
- Existing radiomics extraction
- Existing shape analysis
- Bend-angle analysis
- Existing eCTA processing pipelines

Framework requirements:

- Open-source
- Local/offline
- Reproducible
- Python-based
- MONAI/3D Slicer compatible

---

## Core Scientific Concept

Current segmentation priors trained on healthy anatomy segment contrast-opacified lumen
rather than the true anatomical LAA cavity.

This causes failures in cases with:

- thrombus
- slow flow
- contrast stagnation
- rim enhancement
- mixed filling states

The new framework learns:

- pathological intraluminal filling states
- thrombus-inclusive anatomy
- latent visual representations of LAA filling behavior
- anatomical constraints
- topology-aware cavity completion

using:

- CT-native foundation encoders
- visual tokenization
- positive anatomical priors
- negative anatomical priors
- semi-automatic expert correction
- uncertainty-aware learning

---

## Existing Priors

Already available in this repository:

### Anatomical Priors

- NUDF
- VISTA3D
- TotalSegmentator

These are preserved and integrated.

The new model learns:

- residual pathological adaptation
- thrombus/filling-state completion
- correction learning
- latent filling-state representation

---

## Positive Anatomical Priors

Structures likely belonging to:

- left atrium
- left atrial appendage
- ostium region
- appendage continuity
- atrial cavity

These support:

- cavity completion
- topology preservation
- thrombus-inclusive segmentation
- latent anatomical consistency

---

## Negative Anatomical Priors

Structures known NOT to belong to the LAA:

- coronary arteries
- lungs
- pulmonary veins
- aorta
- pulmonary artery
- myocardium
- mediastinum
- pericardial fat
- bone/calcification/artifact

Purpose:

- reduce false-positive extension
- constrain latent anatomical space
- improve topology preservation
- improve thrombus localization
- improve uncertainty estimation

Methods:

- exclusion masks
- distance transforms
- attention masking
- latent-space penalties
- token-level negative embeddings

---

## Token-Based Anatomical Representation

The model learns latent token states including:

- normal opacified LAA lumen
- dark thrombus-like defect
- contrast stagnation
- rim enhancement
- distal pooling
- mixed-density filling state
- trabeculated low-flow
- coronary artery token
- pulmonary vein token
- lung token
- myocardium token

---

## SLAAO Representation

SLAAO is modeled as independent multi-label features (not mutually exclusive categories):

```python
dark_thrombus_component = True/False
contrast_stagnation     = True/False
rim_pattern             = True/False
whole_LAA_involvement   = True/False
regional_pooling        = True/False
distal_tip_involvement  = True/False
mixed_pattern           = True/False
uncertain_artifact      = True/False
```

Rationale: filling states overlap, multiple patterns coexist, and latent token
representations are naturally multi-label.

---

## Annotation Framework

### Platform

- 3D Slicer with MONAI / MONAILabel integration
- Segment Editor effects: Grow from Seeds, Local Threshold, Islands, Smoothing,
  Logical Operators, Level Tracing, Scissors, Margin, Hollow, Wrap Solidify

### Workflow

```text
CT volume
→ NUDF/VISTA3D/TotalSegmentator priors
→ prior fusion
→ MONAI/MONAILabel inference
→ initial thrombus-inclusive suggestion
→ expert correction in 3D Slicer
→ filling-defect annotation
→ SLAAO labels
→ uncertainty review
→ final approved ground truth
```

### Multi-Rater Validation

- Minimum 2 raters (stroke neurologists, neuroradiologists, trained imaging researchers)
- Expert adjudication for disagreements
- Metrics: Dice, Hausdorff, Cohen kappa, multi-label agreement, boundary agreement,
  uncertainty overlap

---

## Proposed Architecture

### Inputs

```text
CT volume (eCTA / cardiac CT)
+ NUDF mask
+ VISTA3D mask
+ TotalSegmentator mask
+ Optional coronary segmentation
+ Optional lung segmentation
+ Optional vessel segmentation
+ Optional distance maps
+ Optional disagreement maps between priors
```

### Prior Fusion

Generates:

- consensus anatomical prior
- union/intersection masks
- disagreement maps
- positive anatomical priors
- negative anatomical priors
- exclusion maps
- distance-transform maps

### Segmentation Layer

Candidate architectures:

- MONAI UNETR
- SwinUNETR
- Residual correction decoder
- Prior-guided transformer
- Token-guided segmentation decoder

### Foundation Encoders

| Encoder   | Type              |
|-----------|-------------------|
| DINOv2    | General visual    |
| CT-FM     | CT-native         |
| VoxelFM   | CT-native         |
| Merlin    | CT-native         |

### Tokenization Layer

- VQ-VAE
- VQGAN
- Residual VQ
- `vector-quantize-pytorch`

---

## Outputs

| Output | Description |
|--------|-------------|
| `corrected_LAA_mask.nii.gz` | Thrombus-inclusive anatomical LAA mask |
| `filling_defect_map.nii.gz` | Voxel-level thrombus/stagnation/rim map |
| `uncertainty_map.nii.gz` | Segmentation + filling-state uncertainty |
| `SLAAO_labels.json` | Multi-label SLAAO structured output |
| `correction_map.nii.gz` | Residual correction relative to anatomical priors |

---

## Radiomics Layer

Extracted from:

- filling-defect map
- uncertainty map
- thrombus-inclusive cavity
- topology-aware cavity representation

### Feature Groups

**Intensity:** mean HU, median HU, min HU, max HU, entropy, variance

**Texture:** GLCM, GLRLM, GLSZM, NGTDM

**Shape:** sphericity, elongation, compactness, surface area, volume, curvature, topology

**Anatomical:** ostium distance, distal/proximal localization, rim thickness, wall contact,
regional pooling, thrombus burden

Existing radiomics, shape analysis, and bend-angle outputs are preserved.

---

## Full Pipeline

```text
CT volume
→ NUDF/VISTA3D/TotalSegmentator priors
→ positive anatomical priors
→ negative anatomical priors
→ prior fusion
→ MONAI/MONAILabel inference
→ 3D Slicer correction
→ thrombus-inclusive ground truth
→ CT-native encoder
→ visual tokenization
→ transformer latent representation
→ thrombus-inclusive segmentation
→ filling-defect map
→ SLAAO outputs
→ uncertainty map
→ radiomics + topology analysis
→ clinical association models
```

---

## Software Stack

| Category | Libraries |
|----------|-----------|
| Core | Python, PyTorch, MONAI, 3D Slicer, MONAILabel |
| Imaging | nibabel, SimpleITK, pydicom, TorchIO |
| Segmentation | nnU-Net, MONAI UNETR, SwinUNETR |
| Tokenization | vector-quantize-pytorch |
| Analysis | PyRadiomics, scikit-image, scipy, numpy, pandas, networkx, trimesh |

---

## Dataset Strategy

- 200–300 thrombus/filling-defect patients
- Larger non-thrombus control cohort

Priority annotation cases:

- thrombus cases
- difficult filling states
- ambiguous stagnation
- rim cases
- mixed cases
- distal tip failures

Controls: representative subset with quality-control verification.

---

## Scientific Questions

1. Can pretrained healthy anatomical priors be adapted to thrombus-inclusive segmentation?
2. Do CT-native foundation encoders outperform general visual encoders?
3. Can latent visual tokens represent SLAAO filling states?
4. Can uncertainty-aware filling-defect maps improve characterization of thrombus vs stagnation?
5. Can positive and negative anatomical priors improve topology-aware segmentation?
6. Can radiomics/topology derived from filling-state maps improve stroke/AF phenotyping?

---

## Development Phases

### Phase 1 — Annotation Infrastructure

- Integrate prior fusion
- Add positive and negative anatomical priors
- Build MONAI/3D Slicer annotation workflow
- Add MONAILabel integration
- Store correction maps and SLAAO multi-label metadata

### Phase 2 — Dataset Curation

- Generate curated thrombus-inclusive dataset
- Create uncertainty/disagreement maps
- Implement multi-rater review workflow

### Phase 3 — Encoder Comparison

- Compare DINOv2, CT-FM, VoxelFM, Merlin encoder

### Phase 4 — Tokenization

- Add visual tokenization layer

### Phase 5 — Radiomics + Topology

- Add radiomics + topology analysis on filling-state maps

### Phase 6 — Clinical Modeling

- Clinical association models for stroke/AF phenotyping

---

## Related Files in This Repository

- `subprojects/la_laa_slaao/README.md` — subproject home
- `configs/profiles/p60_analysis_la_laa_slaao.yaml` — pipeline profile
- `configs/roi/laa_slaao.yaml` — ROI configuration
- `subprojects/la_laa/README.md` — base LAA subproject (extended by this framework)
- `docs/protocols/laa_highres_dataset_setup.md` — dataset setup protocol
