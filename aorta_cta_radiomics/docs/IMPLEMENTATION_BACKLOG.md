# Prioritized implementation backlog

This backlog starts from verified repository state on 2026-07-31. A priority is not evidence that a component is clinically valid.

## P0 — correctness and reproducibility

### P0.1 Produce one complete native FLOWCAT case

Priority: P0

Task: Run segmentation, VMTK centreline/branch extraction, GNN anatomical labelling, and local-graph creation for one de-identified case in a single compatible environment.

Reason: No local case has `branch_model.vtk`, `centerline_segments_array.npy`, `segments_graph_pred.pickle`, or `local_graph.pickle`; named carotid analysis cannot be demonstrated without them.

Dependencies: Pinned FLOWCAT commit and hashes; Linux x86-64; VTK/VMTK; PyTorch/PyG/torch-cluster; CUDA recommended; input SHA-256.

Expected output: Complete FLOWCAT case tree, stdout/stderr logs per stage, error JSON, runtime/hardware record, output hashes, and Slicer QC package.

Risk: The broad-tree surface may reproduce the surface-normal/degeneracy failure seen in case 570.

Recommended next action: Use one case only, preserve the first raw failure, and test documented bottom-cut/surface-cleaning parameters without changing FLOWCAT source.

### P0.2 Validate FLOWCAT coordinate reconstruction

Priority: P0

Task: Compare adapter-reconstructed centreline RAS points against the native VTK model and CTA lumen for RAS, LAS, and LPS inputs.

Reason: Upstream arrays subtract an LPI-corner translation and explicitly branch on only three orientations.

Dependencies: P0.1 outputs plus a small orientation fixture set.

Expected output: Round-trip tests, centreline-inside-lumen fraction, transform metadata, and a locked coordinate convention.

Risk: VTK and NIfTI coordinate conventions may differ by axis sign or translation.

Recommended next action: Fail closed when no transform candidate meets the prespecified inside-lumen tolerance.

### P0.3 Make provenance immutable

Priority: P0

Task: Add input/model/config/code hashes, dependency/hardware capture, immutable manifest snapshot, and per-stage attempt IDs to the maintained staged runner.

Reason: Current stage status can be overwritten across invocations and does not fully reproduce a run.

Dependencies: Existing staged runner and adaptive-v2 provenance schema.

Expected output: Append-only attempt records and an output-completeness manifest.

Risk: Compatibility with existing aggregation scripts.

Recommended next action: Add a new schema-versioned manifest without changing existing CSV columns first.

### P0.4 Complete clean-install dependency distribution

Priority: P0

Task: Publish/vendor `cta_common` for external installs and validate the optional Parquet/VTK dependency groups used by new outputs.

Reason: The milestone now declares `cta_common>=0.1.0`, documents the local editable install, and declares optional Parquet/VTK groups. The unresolved issue is supplying `cta_common` outside this workspace and validating native dependency combinations.

Dependencies: Packaging decision for the monorepo.

Expected output: Fresh-environment installation test.

Risk: Direct path dependency is unsuitable for external distribution.

Recommended next action: Retain the verified workspace install now; decide whether external distribution will publish, vendor, or package `cta_common` before release.

## P1 — functional carotid perivascular milestone

### P1.1 Reader-QC anatomical artery rasterization

Priority: P1

Task: Run nearest and branch-aware strategies on the complete FLOWCAT case; compare final RCCA/LCCA/RICA/LICA masks and label-14 zones in Slicer.

Reason: Synthetic correctness does not establish behavior at touching branches or noisy bifurcations.

Dependencies: P0.1–P0.2 and typed adapter.

Expected output: NIfTI strategy maps, confidence, uncertainty, bifurcation exclusion, summary JSON, and visual decision.

Risk: Merged branches or GNN mislabelling may make both strategies wrong in the same region.

Recommended next action: Correct the upstream label/branch reference rather than hand-patching the derived volume.

### P1.2 Lock CCA/bulb/proximal-ICA interval definitions

Priority: P1

Task: Prespecify path-distance windows and endpoint/bifurcation exclusions for bilateral CCA, bulb, and proximal ICA.

Reason: FLOWCAT has no separate bulb class; reproducible derived anatomy is required.

Dependencies: Valid branch topology and reader protocol.

Expected output: Versioned interval configuration and Slicer annotation guide.

Risk: Anatomical variants and short branches.

Recommended next action: Pilot the definitions on 10 varied cases before locking.

### P1.3 Integrate coarse carotid context masks

Priority: P1

Task: Resolve and geometry-check internal jugular, combined/named muscle, bone, thyroid, and airway masks in carotid crops.

Reason: Fat/residual-only analysis cannot quantify the requested artery-muscle/vein/bone interfaces.

Dependencies: Model registry, available head-and-neck tasks, native CTA crop transforms.

Expected output: Source-labelled masks, availability/confidence manifest, and interface QC.

Risk: Coarse whole-neck models are inaccurate at the arterial boundary; some named muscles may be unavailable.

Recommended next action: Use coarse masks as exclusions/priors and mark absent classes uncertain; do not generate pseudo-ground truth.

### P1.4 Execute the full deterministic carotid analysis

Priority: P1

Task: Run coordinates, fixed/adaptive shells, composition, sectors, contact, longitudinal profiles, calcium integration, three-level outputs, and QC for one passed case.

Reason: This is the first end-to-end milestone.

Dependencies: P1.1–P1.3 and configuration lock.

Expected output: Complete case/vessel/sample rows, volumes, plots, provenance, and visual review.

Risk: Shell truncation, high uncertain fraction, inadequate fat voxels, or missing anatomy may make intervals non-assessable.

Recommended next action: Preserve and report non-assessable intervals; do not relax QC thresholds to force completion.

### P1.5 Expand to a stratified technical cohort

Priority: P1

Task: Process development cases spanning low/high fat, calcium, jugular/muscle contact, dental/shoulder artifact, low contrast, incomplete FOV, and anatomical variants.

Reason: One case cannot reveal systematic failure modes.

Dependencies: Stable P1.4 runner and reader checklist.

Expected output: Failure taxonomy, timing/memory, feature missingness, and locked technical thresholds.

Risk: The current five-case selection was based on aortic whole-tree burden and does not cover all carotid phenotypes.

Recommended next action: Add cases for missing strata without using outcomes for selection.

## P2 — useful scientific extensions

### P2.1 Calcium–microenvironment colocalization

Priority: P2

Task: Adapt the existing bone-aware calcium partition to named vessel coordinates and quantify calcium by path, radial band, sector, and adjacent tissue.

Reason: This directly connects visible disease with microenvironment features while retaining bone/blooming ambiguity.

Dependencies: P1 milestone and complete bone context.

Expected output: Retained, ambiguous, and definite-bone profiles; calcium–fat/muscle/vein contact features.

Risk: Contrast-filled lumen and blooming limit specificity.

Recommended next action: Validate against reader-reviewed calcified plaques and bone-adjacent negatives.

### P2.2 Selected-region annotation tooling

Priority: P2

Task: Extend the existing Slicer assistant/QC pattern for vessel-orthogonal crops, interval selection, tissue correction, sectors, and contact review.

Reason: Difficult interfaces need efficient reference annotation rather than full-neck labelling.

Dependencies: P1 interval definitions and label protocol.

Expected output: NIfTI labels plus structured annotation metadata and adjudication state.

Risk: Slicer coordinate/export drift and reader burden.

Recommended next action: Build one narrow carotid module and test round-trip geometry before adding more tools.

### P2.3 Reproducibility and scanner sensitivity study

Priority: P2

Task: Apply the validation plan across spacing, kernel, manufacturer, voltage, phase, resampling, shell, fat-window, contact-distance, and sector settings.

Reason: CTA attenuation and small physical envelopes are acquisition-sensitive.

Dependencies: Locked technical pipeline and sufficient metadata.

Expected output: ICC/Bland-Altman/sensitivity report and robust feature subset.

Risk: Metadata missingness and confounding by cohort.

Recommended next action: Start with deterministic phantom/synthetic and repeat-acquisition cases.

## P3 — exploratory work

### P3.1 Vertebral/transverse-foramen specialist model

Priority: P3

Task: Develop a high-resolution vertebral artery/vein/foramen/deep-muscle model only after carotid validation.

Reason: Bone adjacency makes vertebral calcium and contact substantially different from carotid analysis.

Dependencies: Completed carotid validation, annotated vertebral crops, licence-reviewed training plan.

Expected output: Specialist model and separate validation report.

Risk: Very small vessel size, venous plexus ambiguity, and severe bone partial volume.

Recommended next action: Defer.

### P3.2 Subclavian/scalene specialist model

Priority: P3

Task: Develop a vessel-centred origin/pre-/inter-/post-scalene model.

Reason: Shoulder artifact and complex interfaces require a separate target.

Dependencies: Completed carotid validation and selected annotations.

Expected output: Specialist model and validation.

Risk: Shoulder artifact, FOV truncation, and poor small-structure reference.

Recommended next action: Defer.

### P3.3 Clinical association models

Priority: P3

Task: Test prespecified associations only after technical validation and feature lock.

Reason: Development outputs are not clinical evidence.

Dependencies: Validation gates, patient-level partitions, causal/statistical protocol.

Expected output: Effect estimates with uncertainty and independent replication plan.

Risk: Leakage, multiplicity, confounding, overfitting, and clinical overclaim.

Recommended next action: Do not begin from the five development cases.
