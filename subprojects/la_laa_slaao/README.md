# LA/LAA SLAAO Subproject

This subproject extends `subprojects/la_laa/` with thrombus-inclusive segmentation
and SLAAO (Spontaneous Left Atrial Appendage Occlusion) filling-state representation learning.

## Scope

- Thrombus-inclusive anatomical LAA segmentation
- SLAAO multi-label filling-state classification
- CT-native foundation encoder integration
- Visual tokenization of LAA filling states
- Filling-defect and uncertainty mapping
- Positive and negative anatomical prior fusion
- Radiomics and topology analysis of filling states

## Scientific Motivation

Existing anatomical priors (NUDF, VISTA3D, TotalSegmentator) are trained on healthy
contrast-opacified lumen and fail in the presence of thrombus, slow flow, rim enhancement,
and mixed filling states. This subproject learns residual pathological adaptation on top
of those priors.

## Data Flow

```text
CT volume
→ NUDF/VISTA3D/TotalSegmentator priors        (existing)
→ positive + negative anatomical prior fusion  (new)
→ MONAI/MONAILabel inference
→ 3D Slicer expert correction
→ thrombus-inclusive ground truth
→ CT-native encoder + visual tokenization
→ transformer latent representation
→ thrombus-inclusive segmentation mask
→ filling-defect map
→ SLAAO multi-label outputs
→ uncertainty map
→ radiomics + topology analysis
→ clinical association models
```

## SLAAO Labels (Multi-Label)

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

Labels are independent; multiple patterns can coexist in a single case.

## Outputs Per Case

| File | Description |
|------|-------------|
| `corrected_LAA_mask.nii.gz` | Thrombus-inclusive anatomical LAA mask |
| `filling_defect_map.nii.gz` | Voxel-level thrombus/stagnation/rim map |
| `uncertainty_map.nii.gz` | Segmentation + filling-state uncertainty |
| `correction_map.nii.gz` | Residual correction relative to anatomical priors |
| `SLAAO_labels.json` | Multi-label SLAAO structured output |

## Anatomical Priors

### Positive

- left atrium, LAA, ostium region, appendage continuity, atrial cavity

### Negative

- coronary arteries, lungs, pulmonary veins, aorta, pulmonary artery,
  myocardium, mediastinum, pericardial fat, bone/calcification/artifact

Negative priors are enforced via exclusion masks, distance transforms,
attention masking, and latent-space penalties.

## Foundation Encoders (Phase 3)

| Encoder | Type |
|---------|------|
| DINOv2 | General visual |
| CT-FM | CT-native |
| VoxelFM | CT-native |
| Merlin | CT-native |

## Annotation Workflow

Annotation uses 3D Slicer + MONAI/MONAILabel with at least 2 raters and expert adjudication.

Segment Editor effects used:
Grow from Seeds, Local Threshold, Islands, Smoothing, Logical Operators,
Level Tracing, Scissors, Margin, Hollow, Wrap Solidify.

## Dataset Target

- 200–300 thrombus/filling-defect cases
- Larger non-thrombus control cohort

Priority annotation: thrombus, ambiguous stagnation, rim, mixed, distal tip failure cases.

## Development Phases

| Phase | Goal |
|-------|------|
| 1 | Annotation infrastructure (prior fusion, MONAILabel, 3D Slicer workflow) |
| 2 | Curated thrombus-inclusive dataset + multi-rater validation |
| 3 | Foundation encoder comparison |
| 4 | Visual tokenization layer |
| 5 | Radiomics + topology analysis |
| 6 | Clinical modeling |

## References

- Full framework spec: `docs/protocols/laa_slaao_framework.md`
- Base LAA subproject: `subprojects/la_laa/README.md`
- Dataset setup: `docs/protocols/laa_highres_dataset_setup.md`
- Pipeline profile: `configs/profiles/p60_analysis_la_laa_slaao.yaml`
