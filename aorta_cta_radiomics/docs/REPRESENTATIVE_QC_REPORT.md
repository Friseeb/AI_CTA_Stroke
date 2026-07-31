# Representative carotid-pipeline QC: `sub-570_acq-CTA_ct`

## Scope and status

This report records a de-identified development example reviewed on 2026-07-31. It is an integrity and triage review, not anatomical ground truth or clinical validation.

Overall status: `blocked_missing_native_flowcat_outputs`.

The broad FLOWCAT binary arterial mask is technically valid and can support whole-tree development checks. It cannot support named right/left CCA, carotid-bulb, or proximal-ICA phenotyping because the native FLOWCAT branch model, centreline array, predicted anatomical graph, and local graph are absent. The existing skeleton/EDT centreline is a technical fallback without anatomical labels and was not substituted.

## Inputs and integrity

- Case ID: `sub-570_acq-CTA_ct` (de-identified).
- Cohort/arm: Daylight Extended.
- CTA: `/Volumes/DICOM5/daylightbids/sub-570_acq-CTA_ct.nii.gz`.
- CTA SHA-256: `ad0ab110ea688a2554ec6b5b72ad48627ff251fe247b8f15ce97744ec5244d8a`.
- CTA shape: `512 x 512 x 1783` voxels.
- CTA spacing: `0.539 x 0.539 x 0.250 mm`.
- CTA orientation: LPS axis codes in the stored NIfTI affine; nibabel physical world coordinates remain RAS+.
- FLOWCAT binary segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.
- Segmentation SHA-256: `46151cf209318c28b98c87f5fee8d3cfcc2a5512da870a11bff428740169d549`.
- FLOWCAT source: branch `models/2026-03`, commit `1991ca5b238d7c868c9c3a0a390057bc89aee559`.

The segmentation is finite, binary, shape-matched, and affine-matched to the CTA. It contains 3,443,029 foreground voxels (250.068 mL), 84 connected components, and a 99.94% largest-component fraction. Its automated integrity state is `automated_pass_visual_pending`; `many_connected_components` remains a warning. CPU inference took 747.3 seconds in the recorded run.

## Required-output completeness

The generated adapter manifest is:

`outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/flowcat_output_manifest.json`

Present and hashed:

- `extracranial_vessels/segmentation.nii.gz`.
- Eight local FLOWCAT model payloads: three segmentation/mandible checkpoints
  and five extracranial vessel-labelling folds, with paths and SHA-256 retained
  in provenance.

Required but missing:

- `extracranial_vessels/branch_model.vtk` or `.vtp`.
- `extracranial_vessels/centerline_segments_array.npy`.
- `extracranial_vessels/segments_graph_pred.pickle`.
- `extracranial_vessels/local_graph.pickle`.

Consequently, no anatomical artery volume, label-confidence map, carotid vessel-coordinate system, named-vessel shell, sector, contact, or three-level carotid feature output was generated for this case.

## Visual review performed

The following existing montages were opened and inspected directly:

- `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct/extracranial_vessels/segmentation_qc_montage.png`.
- `outputs/anterior_circulation_supraaortic/adaptive_vessel_burden_v2/development_whole_tree/cases/sub-570_acq-CTA_ct/qc/adaptive_burden_montage.png`.
- `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/whole_tree_burden/cases/sub-570_acq-CTA_ct/qc/whole_tree_burden_qc_montage.png`.

Findings:

- The broad mask follows substantial portions of the aorta and head-to-neck arterial tree on sampled axial, coronal, and sagittal views.
- The montage is sufficient for broad segmentation triage only. It does not establish branch continuity, laterality, or named carotid accuracy.
- The adaptive-v2 montage demonstrates the narrower capped fat envelope and bone-aware retained/ambiguous high-density partition, but it is explicitly a technical whole-tree fallback.
- Bone and dental proximity remain important ambiguity sources, particularly around small vessels near the skull and vertebral column.
- Named-artery continuity, bifurcation assignment, centreline-inside-lumen coverage, and left/right labels require 3D Slicer review after the native FLOWCAT artifacts are produced.

Visual disposition: `manual_review_required`; no anatomical visual pass was recorded.

The new headless QC command was also executed successfully on the full native
CTA grid. It produced:

- `outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/qc/carotid_qc.json`.
- `outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/qc/carotid_qc_montage.png`.

The JSON independently recovered the same shape, spacing, 84 connected
components, 3,443,029 foreground voxels, and 250.068-mL binary-mask volume. It
records all four native products as missing, all named-vessel results as
unavailable, and `manual_review.required=true`. No labelled-artery, shell,
tissue, or calcium overlay was supplied because the named carotid run stopped at
preflight as designed.

For display only, the final QC reorients by axis permutation/flips to canonical
RAS without interpolation. Its axial panel uses the 85th percentile of occupied
whole-tree slices (`k=1335`) to avoid the aortic-mass median; the JSON and figure
explicitly state `not_a_named_carotid_localization`. Coronal and sagittal panels
retain whole head-to-heart context and anatomical direction labels. This slice
selection improves superior-tree triage but does not replace a labelled carotid
overlay or reader-selected CCA/bulb/ICA interval.

## Development-only outputs retained as comparators

The adaptive whole-tree outputs under
`outputs/anterior_circulation_supraaortic/adaptive_vessel_burden_v2/development_whole_tree/cases/sub-570_acq-CTA_ct/`
include adaptive periarterial-fat masks, raw/retained/ambiguous/definite-bone high-density masks, features, provenance, and QC. Their automated integrity checks pass, including geometry checks for the independent bone union. They remain non-anatomical development comparators and must not be reported as CCA/bulb/ICA results.

## Decision

Do not run or report named carotid perivascular phenotyping for this case yet. The next gate is one complete native FLOWCAT execution, with raw logs preserved, followed by coordinate reconstruction checks and Slicer review of the anatomical artery rasterization. Optional muscle, jugular, thyroid, and airway context should only be integrated after that gate; missing masks must remain missing or uncertain rather than being inferred.
