# aorta_cta_radiomics

Open-source Python research pipeline for stroke CTA / extended CTA analysis focused only on the aorta.

The pipeline starts from:

1. a CTA NIfTI image, with intensities assumed to be in HU by default
2. an existing aorta mask in NIfTI format

It then produces reproducible aorta mask QC metrics, physical-distance peri-aortic shells, threshold-based calcification summaries, descriptive lumen geometry, PyRadiomics features, and CSV outputs for statistical modeling.

## Scientific Scope

Target anatomy:

- ascending aorta
- aortic arch
- descending thoracic aorta
- optional abdominal aorta when present in the scan

Preferred mask sources:

- TotalSegmentator aorta masks
- future MONAI VISTA3D/VISTA-style masks
- manual or corrected masks exported from 3D Slicer

Version 1 assumes the segmentation already exists. It does not train or run a segmentation model.

## What This Pipeline Does Not Do

This project does not segment plaque and does not classify tissue as plaque versus non-plaque.

Plaque boundaries on routine contrast CTA are unreliable and subjective. The current stable strategy is:

- quantify calcification with transparent HU thresholds
- create reproducible peri-luminal/peri-aortic shells
- extract radiomics from objective masks and shells

All previous exploratory irregularity/adaptive ROI versions have been removed from the active pipeline. Geometry outputs are descriptive only.

## Install on macOS

The core pipeline is CPU-compatible and suitable for Apple Silicon macOS. CUDA is not required.

```bash
cd aorta_cta_radiomics
conda env create -f environment.yml
conda activate aorta-cta-radiomics
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

Optional extras:

```bash
conda install -c conda-forge pyvista trimesh
pip install monai torch
```

MONAI/PyTorch are future optional extensions and are not required for the version-1 pipeline.

## Run One Case

```bash
aorta-cta run-single \
  --image path/to/cta.nii.gz \
  --aorta-mask path/to/aorta_mask.nii.gz \
  --case-id CASE001 \
  --outdir outputs/ \
  --config configs/default.yaml
```

Equivalent script:

```bash
python scripts/run_single_case.py \
  --image path/to/cta.nii.gz \
  --aorta-mask path/to/aorta_mask.nii.gz \
  --case-id CASE001 \
  --outdir outputs/ \
  --config configs/default.yaml
```

## Run a Batch

Manifest columns:

- `case_id`
- `image_path`
- `aorta_mask_path`
- `optional_notes`

Example:

```bash
aorta-cta run-batch \
  --manifest examples/manifest.csv \
  --outdir outputs/ \
  --config configs/default.yaml
```

## Outputs

Per-case masks are written under `outputs/masks/<case_id>/`:

- cleaned aorta mask
- shell masks: `shell_0_2mm`, `shell_2_5mm`, `shell_5_10mm`
- aortic wall-band masks: `aorta_wall_internal`, `aorta_wall_external`, `aorta_wall_band`
- calcification-local shell
- calcification masks for each HU threshold
- centerline-normal lumen protrusion candidate masks and labelmap, when enabled
- wall morphology candidate boundary mask
- wall morphology candidate 2 mm neighborhood mask
- version-1 segment label map

CSV outputs:

- `outputs/qc/qc_summary.csv`
- `outputs/features/case_level_features.csv`
- `outputs/features/segment_level_features.csv`
- `outputs/features/centerline_points.csv`
- `outputs/features/centerline_point_features.csv`
- `outputs/features/calcification_features.csv`
- `outputs/features/calcium_omics_features.csv`
- `outputs/features/fat_omics_features.csv`
- `outputs/features/lumen_protrusion_summary_features.csv`
- `outputs/features/lumen_protrusion_candidates.csv`
- `outputs/features/lumen_protrusion_point_features.csv`
- `outputs/features/wall_morphology_features.csv`
- `outputs/features/wall_morphology_sector_features.csv`
- `outputs/features/radiomics_features.csv`
- `outputs/features/modeling_wide_features.csv`

Figures:

- `outputs/figures/<case_id>/<case_id>_aorta_qc_overlay.png`

## 3D Slicer QC Review Sets

The QC helper builds a review queue from a manifest, pipeline outputs, clinical variables, and feature outlier rules. It writes audit files and one Slicer loader script per selected case.

The default aorta QC scene is intentionally narrow so it stays readable in Slicer. It loads the preferred VISTA full-aorta trace for context, the dynamic calcium candidate mask, periaortic fat `0-2 mm` and `2-5 mm`, and depth-thresholded inward protrusion-like and outward ulcer-like surface labelmaps. Slicer display labels are short: `Aorta`, `Bone`, `Fat 0-2`, `Fat 2-5`, `P2 core`, `P2 proj`, `U1.5 core`, and `U1.5 proj`; child segments inside multi-label maps are named only `001`, `002`, etc. The review colors are bone/off-white for calcification, yellow for fat, carmine for protrusion-like layers, and violet for ulcer-like layers. Helper masks such as broad wall bands, ROI patches, contrast-reference masks, and boundary debug masks remain saved on disk but are not selected for routine QC.

The active experimental QC scene also includes `Lumen`, `Wall`, and `Aorta HU`. `Lumen` is the contrast-filled aortic lumen estimated from slice-local centerline-core HU with a configurable floor. When `lumen_correction_enabled` is true, high-HU connected contrast can expand the lumen just outside the VISTA trace within a local correction shell, so obvious contrast lumen is not left inside the wall candidate. `Wall` is a review-only candidate region built by closing discontinuous periaortic fat support within 5 mm of the aorta, then taking non-fat, non-lumen tissue between that closed outer envelope and the contrast lumen. `Aorta HU` is the HU-refined wall+lumen trace for comparison with the original VISTA `Aorta`. This is not a validated wall or plaque segmentation; it is an inspectable ROI for method development.

For a lean wall-from-fat run that skips calcium, radiomics, encoders, and figures, use `configs/wall_from_fat_only.yaml`. This config derives `Lumen`, `Wall`, and `Aorta HU`, then optionally assesses focal inward protrusion-like and outward ulcer-like candidates only inside the combined `Lumen | Wall` domain. The matching QC command should use only `--task wall_from_fat --task adipose_tissue`; the resulting Slicer scene contains `Aorta`, `Aorta HU`, `Wall`, `Lumen`, `Fat 0-2`, `Fat 2-5`, and any thresholded wall/lumen candidate maps.

Example: open aorta calcification masks for SLAO cases with high dynamic calcium volume:

```bash
aorta-cta-qc \
  --manifest examples/manifest.csv \
  --clinical-table path/to/clinical.csv \
  --outputs-root outputs/patient_test_sub547_calcium_dynamic_500hu_vista_aorta \
  --anatomy aorta \
  --task calcification \
  --filter SLAO=1 \
  --outlier-feature aorta_wall_dynamic:calcium_volume:dynamic_lumen_referenced_seed500HU \
  --outlier-method quantile \
  --outlier-direction high \
  --outlier-quantile 0.95 \
  --reviewer reviewer_id \
  --task-comment "QC dynamic wall calcium and intimal tails" \
  --outdir outputs/qc_slicer_dynamic_calcium
```

Add `--open-slicer` to launch the first selected case. If Slicer is not found automatically, pass `--slicer-executable /Applications/Slicer.app/Contents/MacOS/Slicer` or set `SLICER_EXECUTABLE`.

QC outputs:

- `qc_slicer_selection.csv`: one row per selected mask
- `qc_review_tasks.csv`: one row per selected case, with reviewer, task, status, script path, and task comments
- `qc_review_comments_template.csv`: structured per-case comment template for pass/fail, finding category, severity, location, and action
- `qc_review_selection.json`: structured machine-readable selection payload
- `qc_slicer_run_log.csv`: append-style log of who generated each review set
- `slicer_scripts/<case_id>_load_qc_in_slicer.py`: Slicer loader script

Color categories are assigned from filename/task cues: artery, calcium/bone, tissue/wall, fat/adipose, flow, and shape. The task/anatomy vocabulary is intentionally open-ended, so future analyses can be selected by filename tokens before a formal module exists.

## Version-1 Pipeline Stages

1. Load CTA image and aorta mask with SimpleITK.
2. Check physical metadata and resample the mask to image space if needed.
3. Clean the mask by keeping the largest component and filling small holes.
4. Write QC metrics: volume, voxel count, bounding box, spacing, mean HU, and warning flags.
5. Generate physical-distance shells and aortic wall-band masks with distance transforms using image spacing.
6. Threshold calcification inside the configured ROI at `[130, 300, 500, 600]` HU by default.
7. Write calcification burden and an explicitly labelled `agatston_like_not_ecg_gated` sensitivity feature.
8. Write calcium-omics features for regression-ready aortic calcium burden, density, lesion components, span, and wall-location summaries.
9. Optionally detect centerline-normal focal lumen protrusion candidates for review.
10. Create a periaortic fat mask and write fat-omics features.
11. Compute an approximate slice-centerline and axial slice geometry fallback.
12. Compute local wall-sector morphology features for candidate review.
13. Extract PyRadiomics features from the aorta mask and configured shells.
14. Write long- and wide-format CSVs.

## Wall-Band Calcium Definition

By default, calcium is no longer thresholded across the full aorta mask. The configured ROI is `aorta_wall_band`, a physical-distance band around the aorta mask boundary:

- `aorta_wall_internal`: voxels inside the aorta mask within 2 mm of the boundary
- `aorta_wall_external`: voxels outside the aorta mask within 2 mm of the boundary
- `aorta_wall_band`: union of the internal and external wall bands

This definition excludes the central contrast-filled lumen/core of the aorta mask. It is intended to reduce CTA lumen contamination while still retaining boundary-adjacent calcium and blooming signal. The band thicknesses are configurable with `shells.aorta_wall_internal_mm` and `shells.aorta_wall_external_mm`.

Saved wall-focused calcium maps include the ROI in the filename, for example:

- `<case_id>_calcification_aorta_wall_band_thr500HU.nii.gz`

The pipeline also writes an optional dynamic wall-calcium map:

- `<case_id>_aorta_lumen_core_for_dynamic_threshold.nii.gz`
- `<case_id>_aorta_wall_calcium_search_band.nii.gz`
- `<case_id>_aorta_external_contrast_like_for_dynamic_threshold.nii.gz`
- `<case_id>_calcification_aorta_wall_dynamic_seed500HU_high_confidence_seed.nii.gz`
- `<case_id>_calcification_aorta_wall_dynamic_seed500HU_candidate.nii.gz`
- `<case_id>_calcification_aorta_wall_dynamic_seed500HU_rejected_external_contrast_touching.nii.gz`
- `<case_id>_calcification_aorta_wall_dynamic_seed500HU.nii.gz`

This dynamic map estimates the local contrast-lumen reference from the central aorta core, uses `max(500 HU, local_lumen_HU + 75 HU)` as the high-confidence seed threshold, then keeps connected wall-adjacent voxels above `local_lumen_HU + 75 HU` with a 300 HU floor. The search band is wider than the conservative wall band by default, so it can recover intimal-side partial-volume tails while still requiring connection to a high-confidence calcium seed.

External candidates are also checked against outside-aorta contrast. By default, a candidate component outside the aorta mask is rejected if it touches voxels outside the aorta whose HU is close to the local central-aorta contrast reference (`calcification.dynamic_wall.external_contrast_tolerance_hu`, default 75 HU). This is intended to reduce capture of calcifications from adjacent contrast-filled arteries while preserving components that extend into the aorta mask.

The recent coronary CCTA deblooming literature supports the general principle of using a focused calcium semantic region rather than a broad vessel mask, but this repository does not implement a deblooming GAN or train a calcium/plaque segmentation model.

## Calcium Omics

`calcium_omics_features.csv` adds aorta-only calcium features intended for downstream statistical modeling:

- HU-volume burden: `mass_total`, `aortic_mass_proxy`, `aortic_volume_mm3`, `calcium_per_cm`, and `calcium_mass_proxy_per_cm`
- 3D lesion components: `num_lesions`, `log1p_num_lesions`, and lesion diffusivity by aortic length or calcium span
- density burden: `hu_gt_1000_volume`, `hu_gt_1000_fraction`, and `aortic_agatston_modified`
- extent and wall-location summaries: `top_bottom_distance_mm`, `circumferential_arc_mean`, `circumferential_arc_max`, and anterior/posterior image-plane proxy fractions
- segment rows: `calcium_by_segment`, `mass_by_territory`, and `num_territories_involved`

Mass is currently an HU-volume proxy (`sum(HU) * voxel volume`) unless a future scanner-specific calibration phantom is added. The Agatston-style feature is modified for the aortic ROI and is not an ECG-gated coronary Agatston score. The circumferential and anterior/posterior features are image-plane wall-location summaries for research/QC, not diagnostic plaque labels.

Coronary territory features such as LM/LAD/LCx/RCA mass are not emitted by the aorta pipeline unless validated coronary masks or territory labels are supplied in a future module. Current territory rows refer only to the configured aortic segment label map.

## Periaortic Fat Omics

`fat_omics_features.csv` adds aorta-only periaortic adipose tissue features. The active mask is:

- `<case_id>_periaortic_fat_roi.nii.gz`: external periaortic search region, 0-5 mm by default
- `<case_id>_periaortic_fat.nii.gz`: voxels in that region with adipose HU, default `-190` to `-30` HU
- `<case_id>_periaortic_fat_0_2mm.nii.gz`: immediate boundary-adjacent adipose layer, useful for QC/partial-volume sensitivity
- `<case_id>_periaortic_fat_2_5mm.nii.gz`: primary local periaortic/PVAT layer

The rationale is related to the EAT/PCAT/PVAT literature, but the implementation is deliberately periaortic. EAT is the broader epicardial fat depot inside the pericardium, PCAT is coronary-adjacent fat, and PVAT is the general term for vessel-adjacent adipose tissue. Coronary papers often define PVAT by a radial distance from the vessel wall, commonly linked to vessel diameter. This pipeline applies the same reproducible idea to the aorta using a configurable physical-distance shell around the aorta mask.

The default layers are intentionally capped at 5 mm. Aortic wall thickness is commonly on the order of a few millimeters, and the boundary of automated CTA aorta masks can vary between lumen-adjacent and outer-wall-adjacent definitions. For this reason, `0-2 mm` is treated as a near-wall/QC layer, while `2-5 mm` is the primary local periaortic fat layer. Broader outer layers such as `5-10 mm` can be enabled in config for sensitivity analysis, but they are not part of the default model-ready output because they may capture unrelated mediastinal or background fat.

Hand-crafted fat-omics features include:

- `periaortic_fat_volume_mm3` and `periaortic_fat_volume_per_cm`
- `periaortic_mean_HU`, `periaortic_median_HU`, `periaortic_std_HU`, `periaortic_skewness_HU`, and `periaortic_kurtosis_HU`
- `periaortic_high_HU_fraction_m70_m30` and `periaortic_high_HU_fraction_m50_m30`
- `periaortic_radial_gradient`
- `periaortic_circumferential_sector_max_HU` and `periaortic_circumferential_sector_std_HU`
- local texture proxies: `periaortic_glcm_cluster_prominence`, `periaortic_glcm_cluster_tendency`, GLRLM short/long run emphasis, and GLSZM small/large zone emphasis
- segment-level rows for `aortic_segment_label`, periaortic fat volume, and periaortic mean HU

These are hand-crafted periaortic fat descriptors, not a claim of EAT/PCAT segmentation and not a standard radiomics-only feature set. Aneurysm-specific variables such as `aneurysm_neck_PVAT_HU`, `aneurysm_sac_PVAT_HU`, and `aneurysm_sac_minus_neck_HU` require validated aneurysm neck/sac masks or landmarks and are not emitted by the default stroke/aorta workflow.

## Centerline-Normal Wall/Lumen Boundary Candidates

`lumen_protrusions` is a review-oriented module for focal local deviations of an aortic boundary mask. It is disabled in the default config and enabled in `configs/calcium_dynamic_500hu.yaml` for the current sub-547 QC workflow.

By default in the dynamic calcium sub-547 workflow, the detector does not use the lumen mask alone. It builds a surface-band analysis envelope around the aorta mask boundary, currently `4 mm` inside the mask plus `4 mm` outside the mask, so wall-adjacent morphology can be reviewed on both sides of the segmentation surface. In `configs/wall_from_fat_only.yaml`, the same detector instead uses the derived contrast `Lumen` for the centerline and constrains sampling to the derived `Lumen | Wall` masks. That lean mode writes only depth-thresholded wall/lumen QC labelmaps with the `wall_lumen_protrusion` prefix.

The actual sampled boundary is HU-gated, not the entire surface band. For each centerline point, the module estimates a local central-lumen HU reference from a small centerline core, then samples voxels above `centerline_reference_HU - contrast_lower_margin_hu` with a configurable floor. In the active config this is `local centerline HU - 120 HU`, floored at `150 HU`. An optional upper bound relative to the centerline reference helps avoid treating very dense calcium as contrast. Outward candidates are rejected when they overlap a larger external contrast-filled component. The written candidate masks are clipped to the 4 mm internal / 4 mm external surface band, allowing candidates on either side of the segmentation surface without drifting into the central lumen or distant adjacent vessels.

The sparse candidate masks are intended as boundary seeds. For visual review, the pipeline can write grown 3D wall-band patch ROIs around each seed, thin curved surface-sheet labelmaps on the expected local wall plane, a broad aorta-surface projection layer, a peak-localized surface core, and a surface-native connected core. The broad projection paints actual aorta boundary voxels but still uses the centerline-point by angle-bin footprint, so it can look rectangular or band-like. The surface-native layer is now preferred for routine QC: it takes the orthogonal detection result, keeps only peak residual cells, maps them to true surface voxels, and runs connected components on the surface shell. These outputs are still review ROIs, not plaque or ulcer segmentations.

The active sub-547 config also writes separate depth-thresholded QC labelmaps for inward protrusion-like candidates and outward ulcer-like candidates. These use different cutoffs because the review question is different: inward protrusion layers default to `>=2`, `>=3`, and `>=4 mm`, while outward/ulcer-like layers default to `>=1.5`, `>=2`, `>=3`, and `>=4 mm`. Routine thresholded QC layers are generated from `aorta_surface_native` and `aorta_surface_core`; the older broad projection layer remains available as a debug source but is no longer the default QC view.

Ulcer-like and protrusion-like candidates can now use separate smooth reference contours. The inward/protrusion branch can use `inward_angular_median_window_deg` and `inward_longitudinal_smoothing_mm`; the outward/ulcer branch can use `outward_angular_median_window_deg` and `outward_longitudinal_smoothing_mm`. This matters in the arch and in a mildly dilated ascending aorta: broad smooth radius changes should become geometry/diameter features, not focal ulcer or protrusion candidates.

The current wall/lumen review config also applies a focality guard before writing QC masks:

- `min_peak_prominence_mm`: requires the peak residual to stand above the candidate median residual
- `max_median_depth_fraction`: suppresses candidates where most of the region is almost as deep as the peak, which is typical of broad smooth dilation
- `min_focality_ratio`: requires peak residual / median residual to be large enough
- each setting has an `outward_*` variant so ulcer-like outpouchings can be tuned separately from inward protrusions

Practical review settings:

- sensitive screening: lower prominence and focality, then inspect `P2 surf` and `U1.5 surf`
- balanced review: current `configs/wall_from_fat_only.yaml`; start with `P3 surf`, `P4 surf`, `U2 surf`, and `U3 surf`
- high-specificity review: raise peak prominence/focality and mainly inspect `P4 surf`, `U3 surf`, and `U4 surf`

The method uses local vessel coordinates rather than the original axial CT plane:

- estimate a smoothed skeleton/graph centerline from the lumen/aorta mask, with a slice-centerline fallback
- sample radial rays in planes orthogonal to the local centerline tangent
- estimate an expected smooth boundary with angular rolling-median radius and longitudinal smoothing
- flag focal inward deviations where the expected radius exceeds the actual boundary by a configured depth
- flag focal outward deviations where the actual boundary exceeds the expected radius, intended for ulcer-like/outpouching review
- suppress broad or long findings that are more consistent with tapering, centerline curvature, or normal anatomy
- suppress configurable centerline end margins so scan/mask caps are not reported as protrusions

Outputs:

- `lumen_protrusion_candidates.csv`: one row per connected protrusion candidate
- `lumen_protrusion_point_features.csv`: one row per sampled centerline point
- `lumen_protrusion_summary_features.csv`: case-level summary rows
- `<case_id>_lumen_protrusion_analysis_surface_band.nii.gz`
- `<case_id>_lumen_protrusion_contrast_like_from_centerline_hu.nii.gz`
- `<case_id>_lumen_protrusion_candidate_mask.nii.gz`
- `<case_id>_lumen_protrusion_candidate_labels.nii.gz`
- `<case_id>_lumen_protrusion_candidate_boundary.nii.gz`
- `<case_id>_lumen_protrusion_inward_candidate_mask.nii.gz`
- `<case_id>_lumen_protrusion_inward_candidate_labels.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_candidate_mask.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_candidate_labels.nii.gz`
- `<case_id>_lumen_protrusion_patch_roi_4mm_in_4mm_out.nii.gz`
- `<case_id>_lumen_protrusion_patch_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_inward_patch_roi_4mm_in_4mm_out.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_patch_roi_4mm_in_4mm_out.nii.gz`
- `<case_id>_lumen_protrusion_surface_sheet_1mm.nii.gz`
- `<case_id>_lumen_protrusion_surface_sheet_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_inward_surface_sheet_1mm.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_surface_sheet_1mm.nii.gz`
- `<case_id>_lumen_protrusion_aorta_surface_projection_1mm.nii.gz`
- `<case_id>_lumen_protrusion_aorta_surface_projection_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_inward_aorta_surface_projection_1mm.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_aorta_surface_projection_1mm.nii.gz`
- `<case_id>_lumen_protrusion_aorta_surface_core_1mm.nii.gz`
- `<case_id>_lumen_protrusion_aorta_surface_core_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_inward_aorta_surface_core_1mm.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_aorta_surface_core_1mm.nii.gz`
- `<case_id>_lumen_protrusion_inward_aorta_surface_core_depth_ge_<threshold>mm_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_aorta_surface_core_depth_ge_<threshold>mm_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_inward_aorta_surface_projection_depth_ge_<threshold>mm_labels_3d.nii.gz`
- `<case_id>_lumen_protrusion_outward_ulcer_like_aorta_surface_projection_depth_ge_<threshold>mm_labels_3d.nii.gz`

The boundary masks and broader helper masks are still saved for debugging, but the QC Slicer selector excludes them by default. Routine QC loads only the VISTA aorta trace, dynamic calcium candidate, `0-2 mm` and `2-5 mm` fat layers, and the depth-thresholded protrusion/ulcer-like surface labelmaps.

Reported metrics include candidate direction, maximal residual depth in mm, angular width, longitudinal length, affected cross-sectional area, percent lumen compromise for inward candidates, percent outer-area excess for outward/ulcer-like candidates, asymmetry/eccentricity, centerline coordinate, and aortic segment label when a segment map is available.

Interpretation is limited to review candidates. Inward findings are not plaque segmentations, and outward findings are not ulcer diagnoses. The current centerline uses a 3D skeleton graph when possible and falls back to a smoothed slice-centerline if skeletonization fails. It is still a version-1 implementation and needs formal validation before clinical or endpoint modelling use.

## Wall Morphology Candidate Review

The wall morphology module is a fresh local-sector feature extractor. It does not reuse the deleted adaptive irregularity code and does not segment plaque. It samples axial wall sectors and compares each local radial profile with a smoothed expected contour.

Main features include:

- Malinowska coefficient: `P / (2 * sqrt(pi * A)) - 1`
- circularity: `4 * pi * A / P^2`
- compactness, solidity, and contour roughness ratio
- sector radius coefficient of variation
- inward residuals, intended as protrusion-like review signals
- outward residuals, intended as crater/outpouching-like review signals
- configurable candidate flags, set to 4 mm by default to match the common TEE complex-aortic-atheroma scale

Outputs:

- `wall_morphology_sector_features.csv`: one row per sampled local wall sector
- `wall_morphology_parcel_features.csv`: one row per small labelled wall-surface parcel
- `wall_morphology_features.csv`: case-level summary rows for modeling
- `<case_id>_wall_morphology_candidate_boundary.nii.gz`
- `<case_id>_wall_morphology_candidate_2mm.nii.gz`
- `<case_id>_wall_morphology_inward_candidate_boundary.nii.gz`
- `<case_id>_wall_morphology_inward_candidate_2mm.nii.gz`
- `<case_id>_wall_morphology_outward_candidate_boundary.nii.gz`
- `<case_id>_wall_morphology_outward_candidate_2mm.nii.gz`
- `<case_id>_wall_morphology_direction_labels_wall.nii.gz`: wall-surface-only labelmap where `1 = inward`, `2 = outward`, and `3 = overlap`
- `<case_id>_wall_morphology_parcels_wall.nii.gz`: unique parcel IDs on the candidate wall surface
- `<case_id>_wall_morphology_inward_parcels_wall.nii.gz`
- `<case_id>_wall_morphology_outward_parcels_wall.nii.gz`
- `<case_id>_wall_morphology_direction_labels_2mm.nii.gz`: viewable labelmap where `1 = inward`, `2 = outward`, and `3 = overlap`

Interpretation is deliberately limited to `review_candidate_not_plaque_segmentation_or_diagnosis`.

## CT Foundation Encoder Extension

The encoder extension is kept as optional infrastructure, but it is no longer tied to any irregularity/adaptive ROI implementation. By default it can sample patches from high-HU calcification regions and extract CT foundation-model embeddings from those patches. This produces a patch manifest and optional embedding features:

- `outputs/features/encoder_patch_manifest.csv`
- `outputs/features/encoder_features.csv`

The implemented encoder interface supports multiple optional backends over the same deterministic patch manifest:

- `tap_ct_hf`: Hugging Face TAP-CT 3D CT transformer embeddings.
- `ct_fm_lighter_zoo`: CT-FM feature extractor embeddings through `lighter_zoo`.
- `voxelfm_hf`: generic Hugging Face 3D AutoModel hook for VoxelFM once the exact released model identifier/API is pinned.

TAP-CT is a task-agnostic 3D CT ViT/DINO-style foundation model. CT-FM is a 3D CT feature extractor trained on large-scale CT data and exposed as `project-lighter/ct_fm_feature_extractor`. VoxelFM is conceptually aligned with the project because it is described as a DINO-style 3D CT visual feature model, but this repository keeps it disabled until a stable checkpoint path is configured.

Run a small encoder job:

```bash
pip install -e ".[encoders]"

aorta-cta run-single \
  --image path/to/cta.nii.gz \
  --aorta-mask path/to/aorta_mask.nii.gz \
  --case-id CASE001 \
  --outdir outputs_encoder/ \
  --config configs/encoders_tap_ct.yaml
```

Run TAP-CT and CT-FM over the same local aortic patch manifest:

```bash
pip install -e ".[all-encoders]"

aorta-cta run-single \
  --image path/to/cta.nii.gz \
  --aorta-mask path/to/aorta_mask.nii.gz \
  --case-id CASE001 \
  --outdir outputs_encoder_ensemble/ \
  --config configs/encoders_ct_foundation_ensemble.yaml
```

By default, encoder patches are sampled from:

- `calcification_500HU`: local high-HU CTA calcification candidate regions in the configured calcium ROI, `aorta_wall_band` by default
- `wall_morphology_inward_parcels_wall`: small inward/protrusion-like wall parcels
- `wall_morphology_outward_parcels_wall`: small outward/crater-like wall parcels
- `wall_surface_grid`: optional dense local wall-sector sampling when enabled in the encoder config

These embeddings should complement, not replace, transparent HU and geometry features. They are patch descriptors for retrieval, clustering, or later lightweight supervised probes, not plaque segmentations.

## Radiomics Reproducibility

PyRadiomics settings are stored in `configs/radiomics.yaml` and copied next to the generated masks for each case. Defaults include:

- fixed bin width: 25
- resampled spacing: 1 mm isotropic
- interpolation: BSpline
- normalization: disabled by default
- feature classes: first order, shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM

Wavelet features are intentionally disabled by default. Enable them only after runtime and stability checks.

## CTA Calcium Caveats

The 130 HU threshold is standard for non-contrast CT but can be confounded by contrast-enhanced CTA. The default config also runs 300, 500, and 600 HU thresholds as sensitivity analyses. Any Agatston-like output is not an ECG-gated Agatston score.

For CTA studies, report threshold sensitivity and consider restricting calcium extraction to a peri-luminal or wall-adjacent ROI when a suitable mask definition is available.

## TotalSegmentator, VISTA, and 3D Slicer Masks

TotalSegmentator:

- run TotalSegmentator externally
- pass the exported `aorta.nii.gz` mask as `--aorta-mask`
- verify full ascending aorta, arch, and descending thoracic coverage before using it for wall-band calcium

VISTA/VISTA3D:

- VISTA/NV-Segment-CT aorta masks can be used if exported as NIfTI in the CTA image space
- when VISTA provides the full aorta and TotalSegmentator high-resolution task truncates coverage, prefer VISTA as the canonical aorta mask
- if spacing/origin/direction differ, the CLI can resample the mask with nearest-neighbor interpolation

3D Slicer:

- export the corrected segmentation as a labelmap NIfTI
- ensure the aorta label is non-zero and other labels are removed before passing it as `--aorta-mask`

## Current Limits

The anatomical segment map is conservative in version 1 and labels the whole aorta as one region. The baseline geometry table remains axial slice-based. The lumen-protrusion module samples centerline-normal radial planes using a skeleton/graph centerline when possible, but the implementation should still be treated as a review aid until validated against curated annotations. These interfaces are present so validated aortic zones and more rigorous orthogonal cross-section extraction can replace the fallback implementation later.

All prior irregularity/adaptive ROI versions are removed from the active code path. The current wall morphology module is a fresh candidate-review feature extractor, not an adaptive plaque segmentation model.

## Future Work

- validated ascending/arch/descending/abdominal aorta zone labels
- graph or skeleton centerline extraction
- true orthogonal cross-sections and radial boundary sampling
- validated adaptive ROI design from a fresh method
- supervised or weakly supervised review models on labelled local wall patches

This future encoder work belongs in `src/aorta_cta_radiomics/encoders.py` and should remain aorta-specific.
