# Migration Map (Current Scripts -> Target Modules)

This map defines incremental migration without breaking current workflows.

## Backbone

| Current Script | Target Location | Action | Phase |
|---|---|---|---|
| `scripts/run_full_segmentation_pipeline.py` | `subprojects/segmentation` | keep as backbone entrypoint | now |
| `scripts/run_full_segmentation_batch.py` | `subprojects/segmentation` | keep as backbone entrypoint | now |
| `scripts/convert_daylightdicom_to_bids.py` | ingest layer | keep | now |
| `scripts/run_cta_deface_dl_batch.py` | privacy layer | keep | now |
| `scripts/deface_cta_simple.py` | privacy layer | keep | now |

## Segmentation Modules

| Current Script | Target Module | Action | Phase |
|---|---|---|---|
| `scripts/run_topcow_claim.py` | intracranial segmentation | keep | now |
| `scripts/run_nv_segment_ct_laa.py` | heart segmentation | keep | now |
| `scripts/run_cardiac_ct_explorer_nudf_only.py` | heart segmentation | keep | now |
| `scripts/run_cardiac_ct_explorer_laa.py` | heart segmentation | keep | now |
| `scripts/segment_external_models.py` | segmentation adapters | keep | now |

## Analysis Modules

| Current Script | Target Module | Action | Phase |
|---|---|---|---|
| `scripts/run_pyradiomics_ibsi_batch.py` | analysis/radiomics | keep | now |
| `scripts/run_laa_shape_descriptors.py` | analysis/shape | keep | now |
| `scripts/run_la_laa_metrics_batch.py` | analysis/shape | keep | now |
| `scripts/generate_la_laa_shape_report.py` | analysis reporting | keep | now |
| `scripts/run_radiomics_clustering.py` | analysis clustering | keep | now |
| `scripts/generate_radiomics_clustering_report.py` | analysis reporting | keep | now |
| `scripts/cluster_la_laa_shape_metrics.py` | analysis clustering | keep | now |

## Experimental / Legacy

| Current Script | Action |
|---|---|
| `scripts/create_vessel_mask_seeded.py` | mark legacy/experimental |
| `scripts/create_vessel_mask_stepwise.py` | mark legacy/experimental |
| `scripts/create_vessel_mask_v2.py` | mark legacy/experimental |
| `scripts/batch_deface.py` | keep legacy, do not use in backbone |

## Rename Policy (Gradual)

Do not hard-rename existing scripts immediately.

1. Add wrapper aliases first.
2. Update docs/profiles.
3. Deprecate old names after one release cycle.
