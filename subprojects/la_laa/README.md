# LA/LAA Substudy

This subproject defines the Left Atrium / Left Atrial Appendage substudy scope
within `AI_CTA_Stroke`, including the SLAAO (Suspicious LAA Occlusion) foundation
framework for thrombus-inclusive segmentation.

## Scope

- LAA-focused segmentation outputs (NUDF / NV-Segment-CT / VISTA3D)
- Prior fusion: consensus, union, intersection, disagreement maps
- Positive and negative anatomical priors
- Thrombus-inclusive LAA ground truth annotation (3D Slicer / MONAILabel)
- SLAAO multi-label filling-state metadata
- Correction map storage
- LA + LAA mesh generation and shape metrics
- Radiomics and topology analysis

---

## Phase 1 (Implemented)

### Prior fusion

Combines NUDF + VISTA3D + TotalSegmentator LAA masks.

Outputs per case:

- `<case>_nudf_laa.nii.gz` — binary NUDF mask
- `<case>_vista3d_laa.nii.gz` — binary VISTA3D mask
- `<case>_totalseg_laa.nii.gz` — binary TotalSegmentator mask
- `<case>_union_laa.nii.gz` — union of all available priors
- `<case>_intersection_laa.nii.gz` — intersection
- `<case>_consensus_laa.nii.gz` — majority vote (≥50% priors agree)
- `<case>_disagreement_map.nii.gz` — per-voxel fraction of disagreement
- `<case>_positive_prior.nii.gz` — LA + LAA cavity region support
- `<case>_negative_prior.nii.gz` — aorta, lung, myocardium exclusion mask
- `<case>_negative_distance_mm.nii.gz` — distance transform from negative prior
- `<case>_prior_fusion_summary.json`

### SLAAO multi-label schema

Independent yes/no features stored in `SLAAO_labels.json`:

```python
dark_thrombus_component: True/False/None
contrast_stagnation:      True/False/None
rim_pattern:              True/False/None
whole_laa_involvement:    True/False/None
regional_pooling:         True/False/None
distal_tip_involvement:   True/False/None
mixed_pattern:            True/False/None
uncertain_artifact:       True/False/None
```

### Annotation store layout

```text
<store_root>/<case_id>/
  annotation/
    <case_id>_ct.nii.gz
    consensus_laa.nii.gz
    positive_prior.nii.gz
    negative_prior.nii.gz
    corrected_LAA_mask.nii.gz     <- expert-corrected output
    filling_defect_map.nii.gz
    uncertainty_map.nii.gz
    correction_map.nii.gz         <- delta: corrected - consensus
    SLAAO_labels.json             <- adjudicated labels
    SLAAO_labels.<rater_id>.json  <- per-rater copies
    session.json
```

---

## Canonical Data Flow

```text
CT volume
→ NUDF/VISTA3D/TotalSegmentator priors
→ run_prior_fusion.py          <- Phase 1
→ run_slaao_annotation_prep.py <- Phase 1
→ 3D Slicer expert correction
→ run_laa_shape_descriptors.py
→ run_la_laa_metrics_batch.py
→ generate_la_laa_shape_report.py
```

---

## Phase 1 Scripts

### Prior fusion (single case)

```bash
python scripts/run_prior_fusion.py \
  --case-id sub-001_acq-CTA_ct \
  --nudf-laa derivatives/nudf_la/sub-001/sub-001_laa_nudf.nii.gz \
  --vista3d-combined derivatives/nudf_la/sub-001/cardiac_ct_explorer/all_segmentations/sub-001_cardiac_segmentations.nii.gz \
  --totalseg-dir derivatives/totalseg/sub-001 \
  --out-dir derivatives/prior_fusion/sub-001
```

### Annotation package preparation (single case)

```bash
python scripts/run_slaao_annotation_prep.py \
  --case-id sub-001_acq-CTA_ct \
  --ct-path derivatives/defaced/sub-001_defaced.nii.gz \
  --prior-fusion-dir derivatives/prior_fusion/sub-001 \
  --store-root derivatives/annotation_store
```

### Annotation package preparation (batch)

```bash
python scripts/run_slaao_annotation_prep.py \
  --batch-csv cases.csv \
  --ct-root derivatives/defaced \
  --prior-fusion-root derivatives/prior_fusion \
  --store-root derivatives/annotation_store \
  --monailabel-dataset
```

---

## Existing Shape Pipeline Scripts

- `scripts/run_cardiac_ct_explorer_nudf_only.py`
- `scripts/run_cardiac_ct_explorer_laa.py`
- `scripts/run_nv_segment_ct_laa.py`
- `scripts/run_laa_shape_descriptors.py`
- `scripts/run_la_laa_metrics_batch.py`
- `scripts/generate_la_laa_shape_report.py`
- `scripts/build_radiomics_manifest_nudf_la.py`

## Python Package

- `python/laa_slaao/prior_fusion.py` — prior fusion + anatomical priors
- `python/laa_slaao/slaao_schema.py` — SLAAO multi-label dataclass + JSON
- `python/laa_slaao/annotation_store.py` — correction map + annotation I/O
- `python/laa_slaao/peri_laa_fat.py` — peri-LAA fat shell features (multi-shell
  radial bands, fat HU window, partial-volume buffer for lung/contrast edges)

## Peri-LAA Fat Shells

Adipose-density features in radial bands around any LAA mask
(consensus / expert / single prior). Analogous to peri-coronary FAI.

Per shell, the pipeline emits:

- `volume_ml`, `voxel_count` — fat voxels (HU ∈ window) within the shell
- `geom_volume_ml` — the geometric shell volume *before* HU filtering, so
  downstream code can compute the fat fraction `volume_ml / geom_volume_ml`
- `mean_hu`, `median_hu`, `p10_hu`, `p90_hu`, `std_hu` — HU statistics

Plus aggregate `peri_laa_fat_total_*` columns over all shells.

### Lung / aorta partial-volume buffer

By default the shells are also subtracted by a buffer around voxels with
HU < `--peri-laa-fat-air-hu` (-300, lung air) or
HU > `--peri-laa-fat-vessel-hu` (+100, contrast lumen), within
`pv_buffer_mm` (1 mm). This handles the partial-volume bleed that otherwise
puts lung-edge and aortic-wall voxels into the fat window.

### CLI — within annotation prep

```bash
python scripts/run_slaao_annotation_prep.py \
  --case-id sub-547_acq-CTA_ct \
  --ct-path data/sub-547_acq-CTA_ct.nii.gz \
  --prior-fusion-dir outputs/test/prior_fusion_547 \
  --store-root outputs/annotation_store \
  --peri-laa-fat \
  --peri-laa-shells-mm 0-2,2-5,5-10 \
  --peri-laa-fat-laa-source consensus_laa
```

Adds `peri_laa_fat_labels.nii.gz` + `peri_laa_fat_metrics.json` to
`<store>/<case_id>/annotation/` and flags `peri_laa_fat_staged: true`
in `session.json`.

### CLI — standalone

```bash
python scripts/run_peri_laa_fat.py \
  --case-id sub-547 \
  --ct-path data/sub-547_acq-CTA_ct.nii.gz \
  --laa-mask outputs/test/prior_fusion_547/sub-547_acq-CTA_ct_consensus_laa.nii.gz \
  --negative-prior outputs/test/prior_fusion_547/sub-547_acq-CTA_ct_negative_prior.nii.gz \
  --out-dir outputs/peri_laa_fat/sub-547 \
  --shells 0-2,2-5,5-10 \
  --write-per-shell-masks
```

Batch mode reads `case_id,ct_path,laa_mask[,negative_prior]` from CSV
and writes to `<out-root>/<case_id>/`.

## Protocol References

- `docs/protocols/laa_highres_dataset_setup.md`
- `docs/NEXT_STEPS_RADIOMICS_MESH_DEFACE.md`

---

## Roadmap

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 1 | Prior fusion + annotation workflow + SLAAO schema | Done |
| 2 | Curated thrombus-inclusive dataset + multi-rater review | Pending |
| 3 | Encoder comparison (DINOv2 / CT-FM / VoxelFM / Merlin) | Pending |
| 4 | Visual tokenization layer (VQ-VAE / VQGAN) | Pending |
| 5 | Radiomics + topology analysis on filling-state maps | Pending |
| 6 | Clinical association models | Pending |
