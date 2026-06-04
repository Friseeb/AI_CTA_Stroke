# Scripts Usage Guide

This file is the practical navigation map for `scripts/`.

If you only want the default path, start with:

- `docs/protocols/script_setup.md`

---

## Pipeline-Ordered Script Map

Legend:
- `Tier`: `backbone` (default workflow) or `optional` (substudy/analysis-specific)
- `Scope`: `single` (one case) or `batch` (many cases)

| Stage | Tier | Scope | Script | Purpose |
|---|---|---|---|---|
| P00 | backbone | n/a | `setup_totalseg_env_mac.sh` / `setup_totalseg_env_cuda.sh` | TotalSegmentator env setup |
| P00 | backbone | n/a | `setup_topcow_env.sh` | TopCoW env setup |
| P00 | backbone | n/a | `setup_nv_segment_ct_env.sh` | NV-Segment-CT env setup |
| P00 | backbone | n/a | `setup_cardiac_ct_explorer_env.sh` | CardiacCTExplorer env setup |
| P10 | backbone | batch | `convert_daylightdicom_to_bids.py` | DICOM -> NIfTI cohort conversion |
| P20 | backbone | single | `deface_cta_simple.py` | Single-case defacing |
| P20 | backbone | batch | `run_cta_deface_dl_batch.py` | Batch defacing |
| P30 | backbone | single | `run_full_segmentation_pipeline.py` | Main single-case orchestrator |
| P30 | backbone | batch | `run_full_segmentation_batch.py` | Main batch orchestrator |
| P40 | optional | single | `run_topcow_claim.py` | Circle of Willis segmentation |
| P40 | optional | single | `run_nv_segment_ct_laa.py` | NV LAA (or other label-id) segmentation |
| P40 | optional | single | `run_cardiac_ct_explorer_nudf_only.py` | NUDF LAA workflow |
| P40 | optional | single | `run_cardiac_ct_explorer_laa.py` | CardiacCTExplorer LAA export wrapper |
| P50 | optional | single/batch | `build_all_segmentations_labelmap.py` / `build_multilabel_vascular_map.py` | Labelmap merge products |
| P60 | optional | batch | `run_pyradiomics_ibsi_batch.py` | Radiomics extraction |
| P60 | optional | batch | `run_radiomics_batch.py` | Alternate radiomics batch pipeline |
| P70 | optional | single/batch | `run_laa_shape_descriptors.py` | Mesh + shape descriptors |
| P70 | optional | batch | `run_la_laa_metrics_batch.py` | LA/LAA relational metrics |
| P70 | optional | single/batch | `generate_la_laa_shape_report.py` | LA/LAA HTML report |
| P80 | optional | batch | `run_radiomics_clustering.py` / `cluster_radiomics_profiles.py` | Radiomics clustering |
| P80 | optional | batch | `cluster_la_laa_shape_metrics.py` | Shape clustering |

---

## Naming Rubric

Use this rubric for all new scripts (and gradual renames).

### 1) Verb Prefix by Intent

- `setup_...` -> environment/bootstrap
- `convert_...` -> format transformation
- `run_...` -> execution/orchestration
- `build_...` -> deterministic artifact assembly
- `generate_...` -> reports/outputs for presentation
- `cluster_...` -> unsupervised grouping/analytics
- `monitor_...` -> progress/health tracking

### 2) Scope Token (Required)

Use one explicit scope token in filename:
- `_single` for one case
- `_batch` for many cases

If a script supports both, prefer `_single` and `_batch` separate wrappers, or
document clearly in the docstring first lines.

### 3) Domain Token (Required)

Include domain in filename:
- `cta`, `topcow`, `nudf`, `la_laa`, `radiomics`, `vertebral`, etc.

### 4) Order Tag (Recommended for New Backbone Scripts)

For new backbone orchestrators, add pipeline order tag:
- `p10_...`, `p20_...`, `p30_...`

Example:
- `p30_run_cta_full_segmentation_batch.py`

Do not mass-rename existing files unless migration wrappers are added.

### 5) Tier Metadata (Backbone vs Optional)

Keep tier in script docstring header:

```text
Tier: backbone
Stage: P30
Scope: batch
```

This avoids brittle filename inflation while preserving machine/human readability.

---

## Recommended Defaults

For day-to-day production runs:

1. `run_full_segmentation_pipeline.py` (single)
2. `run_full_segmentation_batch.py` (batch)
3. Add substudies only when needed (`run_topcow_claim.py`, LA/LAA scripts, radiomics scripts)

---

## Experimental / Legacy (Use With Caution)

- `create_vessel_mask_stepwise.py`
- `create_vessel_mask_v2.py`
- `create_vessel_mask_seeded.py`
- `batch_deface.py`

These are not part of the default backbone path.
