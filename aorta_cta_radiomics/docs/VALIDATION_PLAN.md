# Validation plan: CTA-derived carotid perivascular phenotyping

## Scope and claims

This plan validates a research measurement pipeline, not a diagnostic device. The first locked target is bilateral common carotid artery, operational carotid-bulb windows, and proximal cervical internal carotid artery. Vertebral and subclavian specialist models are out of scope until the carotid milestone passes the technical gates below.

Validation is staged so a downstream result cannot compensate for a failed upstream coordinate system. A case may be used for algorithm development while remaining excluded from biological association analyses. Automated integrity success is recorded as `automated_pass_visual_pending`; only a documented reader decision can produce a visual pass.

## Datasets and splits

Use only de-identified identifiers in analysis artifacts. Freeze patient-level development, tuning, internal test, and external test partitions before reader-reference scoring. Keep all scans from one patient in one partition. Prespecify scanner, reconstruction, acquisition, contrast phase, slice thickness, field-of-view, and artifact strata. The current five Daylight cases are development/QC cases and must not be reported as a held-out validation cohort.

Minimum reference subsets should deliberately cover bilateral CCA/bulb/proximal ICA; low and high fat; calcification; jugular and muscle contact; low contrast; dental and shoulder artifact; motion; incomplete field of view; obesity and low body fat; tortuosity; and anatomical variants. A second institution or cohort should be reserved for external transportability testing.

## Reference standard and annotation

Two trained readers independently correct arterial lumen and named-branch labels and annotate selected vessel-centred slabs for fat, named/combined muscle, internal jugular vein, bone, thyroid, other soft tissue, uncertain tissue, and non-assessable tissue. Readers also mark bifurcation and artifact intervals. A third reader adjudicates disagreements without seeing model measurements.

Store case ID, reader, date, software version, label-protocol version, confidence, adjudication status, vessel, laterality, anatomical segment, centreline start/end, artifact type, exclusion reason, original spacing, and resampled spacing. Evaluation code sees the adjudicated reference only after configuration freeze.

## 1. Arterial segmentation validation

Evaluate the original FLOWCAT binary mask and the anatomically labelled derived volume separately. Report by vessel, laterality, patient, and acquisition stratum:

- Dice coefficient and surface Dice at prespecified physical tolerances.
- 95th-percentile and maximum Hausdorff distance.
- Average symmetric surface distance.
- Absolute and relative volume error.
- Per-class precision, recall, and confusion matrix.
- Connected-component count, missing-branch rate, false bridge rate, and truncation rate.
- Contact-interface accuracy in the juxtaluminal band, because high global Dice can hide a poor vessel boundary.

Technical gate: all required CCA and proximal ICA segments must be present, no left/right swap may occur, and the boundary metrics must meet thresholds chosen before the held-out evaluation. Results from the broad binary fast-lowres mask alone do not satisfy this gate.

## 2. Centreline validation

Compare FLOWCAT/VMTK centrelines with reader-corrected or independently generated reference paths:

- Mean and 95th-percentile symmetric centreline distance error in millimetres.
- Endpoint localization error and coverage fraction.
- Branch-detection sensitivity and false-branch count.
- Bifurcation localization error in millimetres.
- Fraction of centreline samples inside the reference lumen.
- Local-radius agreement, intraclass correlation coefficient (ICC), and Bland-Altman bias/limits.
- Frame stability: adjacent-normal rotation distribution and number of sign flips.
- Physical-to-vessel-to-physical round-trip error.

Technical gate: native FLOWCAT/VMTK outputs and their provenance must exist. Skeleton/EDT fallback paths are a sensitivity analysis only and cannot satisfy the authoritative centreline gate.

## 3. Vessel-label validation

Score FLOWCAT graph labels before and after voxel propagation:

- Edge-level and length-weighted vessel-label accuracy.
- Per-class precision and recall for AA, BT, RCCA, LCCA, RICA, LICA, RECA, and LECA; contextual labels are reported separately.
- Left/right accuracy and swap count.
- Branch-detection accuracy and topology-edit distance.
- Bifurcation and touching-branch confusion.
- Calibration of label confidence by reliability plot and Brier score.
- Voxel agreement between nearest-centreline and branch-aware propagation; disagreements remain uncertain.

Compare strategy A (nearest compatible labelled centreline) with strategy B (branch-aware connected propagation). Prespecify the final uncertainty policy. Never score label `14` as an invented vessel class; report its fraction and location.

## 4. Shell-geometry validation

First use analytic and synthetic geometry, then reader-reviewed real masks:

- Physical shell thickness error for isotropic and anisotropic grids.
- Shell volume error around straight cylinders of known radius and length.
- Correct exclusion of lumen, other arteries, bifurcation zones, and partial-volume bands.
- Exact disjointness and union conservation across radial bands.
- Scan-boundary truncation detection and correct valid-volume denominator.
- Adaptive-law recovery for `t(s) = clip(alpha * D_ref(s), t_min, t_max)` including the resolution floor and maximum.
- Reproducibility under array orientation changes with invariant physical geometry.

Technical gate: median absolute thickness error no greater than one half voxel diagonal on synthetic tests; no normalization by the theoretical complete shell when a boundary truncates it.

## 5. Tissue-classification validation

Use mutually exclusive reader masks within fixed vessel-centred evaluation envelopes. Report:

- Fat-volume, muscle-volume, vein-volume, bone-volume, thyroid-volume, residual-soft-tissue, and uncertain-volume agreement.
- Per-class precision, recall, Dice, surface Dice where a surface is meaningful, and volume error.
- Vein-exclusion accuracy and fraction of venous voxels incorrectly counted as PVAT.
- Bone-exclusion accuracy and fraction of bone counted as calcium or residual soft tissue.
- Fat attenuation mean/median/percentile agreement.
- Radial-profile and circumferential-sector agreement.
- ICC and Bland-Altman analysis for continuous composition features.
- Test-retest reproducibility where repeat CTA is available.

The residual class is `other non-fat soft tissue`; it is not labelled arterial wall. High-density/blooming and uncertain/non-assessable voxels are reported separately. Radiomics is evaluated only for regions meeting the locked voxel-count, stability, resampling, and quantization criteria.

## 6. Surface-contact validation

Readers annotate interface arcs on selected orthogonal slabs. Compare contact with fat, sternocleidomastoid, longus colli, scalene/strap/other muscle, vein, bone, thyroid, residual soft tissue, and uncertain tissue at 0.5, 1.0, and 2.0 mm:

- Contact-class precision and recall.
- Absolute contact-area error in square millimetres.
- Contact-fraction error and ICC.
- Contiguous contact-length error.
- Number and location of separate contact-region errors.
- Longitudinal and circumferential profile agreement.

Analyze contact-distance and voxel-resolution sensitivity explicitly. A volume fraction is never substituted for surface-contact fraction.

## 7. Longitudinal-feature validation

For each named vessel preserve sample-point rows and compare curves before summarizing:

- Assessable and non-assessable length error.
- Pointwise and functional agreement of fat fraction, attenuation, muscle contact, vein contact, calcium burden, and uncertainty.
- Radial attenuation/fat-fraction gradient agreement.
- Proximal-to-distal gradient and left-right asymmetry agreement.
- Transition-point localization error for fat-rich to muscle-rich/residual-rich environments.
- Agreement in counts and lengths of prespecified fat-poor and muscle-dominant intervals.
- Circumferential and radial heterogeneity reproducibility.

Use functional ICC/correlation only after checking systematic path-registration error. Do not validate a vessel solely through one global mean.

## 8. Reproducibility validation

Lock code commit, configuration hash, model/checkpoint hashes, environment, input hashes, and random seeds. Repeat a stratified sample under:

- Same input and environment (determinism).
- Independent installation from the documented environment.
- macOS ARM64 CPU/MPS where supported and Linux x86-64 CUDA.
- Rerun by a second operator.
- Repeated CTA acquisition where ethically and clinically available.

For deterministic outputs require bitwise identity where libraries permit; otherwise prespecify numerical tolerances. Report feature ICC, coefficient of variation, Bland-Altman bias/limits, mask Dice, and maximum coordinate error. Architecture-specific algorithm substitutions are prohibited; an unavailable dependency must fail clearly or use a separately reported fallback.

## 9. Clinical-association analysis

Clinical association begins only after technical lock and remains exploratory until independently replicated. Prespecify the outcome, causal estimand, covariates, missing-data handling, multiplicity control, model form, and validation cohort. Preserve the separation between CTA-derived microenvironment features and conventional stenosis/diameter measures. Evaluate incremental information beyond clinical covariates and standard radiological measurements without implying causality or clinical utility.

Use patient-level resampling or clustered methods for bilateral vessels. Prevent leakage across vessels from one patient. Report discrimination/calibration only when clinically relevant; emphasize effect estimates with uncertainty. No threshold becomes a clinical cut point from the five-case development panel.

## Sensitivity analyses

Run the following as prespecified, labelled analyses rather than silent parameter changes:

- Primary, narrow, and wide fat HU windows.
- Fixed radial bands and diameter-adaptive envelopes, including the adaptive maximum.
- Contact distances of 0.5, 1.0, and 2.0 mm.
- Original voxel spacing, resampling to supported target spacing, and no unsupported upsampling claim.
- Reconstruction kernel, slice thickness, manufacturer, tube voltage, contrast phase, and missing metadata.
- Linear versus nearest-neighbour/BSpline interpolation as appropriate to data type.
- Centreline sample spacing and smoothing window.
- 8, 16, and 32 circumferential sectors.
- Alternative deterministic overlap hierarchies.
- Calcium thresholds, bone masks, blooming uncertainty, and ambiguous-component treatment.
- Strategy A versus strategy B artery-volume mapping.
- Native FLOWCAT/VMTK radius versus skeleton/EDT fallback as a non-authoritative sensitivity analysis.

## Statistical reporting

Report case counts and exclusions at every stage. Confidence intervals use patient-level bootstrap or an appropriate hierarchical model. Missing anatomical masks are not negative labels. Report failures, manual intervention, uncertainty fractions, and visual-review state. Publish the locked data dictionary and configuration hash with every analysis.

## Current readiness and release gates

As of 2026-07-31, deterministic unit tests and five-case binary segmentation/adaptive development runs exist. The local environment does not contain native FLOWCAT `branch_model.vtk`, `centerline_segments_array.npy`, `segments_graph_pred.pickle`, or `local_graph.pickle`; therefore centreline, vessel-label, named carotid shell, sector, contact, and clinical validation gates remain unmet. The next validation action is one full native FLOWCAT case on a compatible Linux VMTK/PyG environment, followed by blinded CCA/bulb/proximal-ICA Slicer QC.
