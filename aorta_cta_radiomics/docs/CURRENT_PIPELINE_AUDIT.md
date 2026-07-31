# Current pipeline audit

Audit date: 2026-07-31

Scope: repository snapshot at the current working tree, including the maintained
`aorta_cta_radiomics` package, shared `cta_common` code, the staged cohort runner,
the local FLOWCAT checkout, and the development-only anterior-circulation
supra-aortic substudy. This is a code-and-artifact audit, not a claim of clinical
validity. A module import was not treated as evidence that a capability works.

Status language used below is deliberate:

- **Verified** means the code path and a corresponding artifact or automated test
  were inspected.
- **Pending visual QC** means automated outputs exist but formal case review in 3D
  Slicer is incomplete.
- **Missing** means no implementation was found in the audited path.
- **Unable to verify** means a documented or partial path exists, but successful
  representative execution could not be established.

No protected health information is reproduced. Only de-identified `sub-<ID>`
identifiers and paths already organized under those identifiers are shown.

Operational artifacts and the 128-test baseline below were frozen before the
additive milestone implementation. The current working tree now also contains a
typed `aorta_cta_radiomics.flowcat_adapter`, anatomical artery-volume propagation,
deterministic `carotid_analysis`, the fail-closed `carotid_cli`, three-level output
schemas, and headless `carotid_qc`. These additions are synthetic/contract-tested.
They do not change which native FLOWCAT artifacts were generated in the original
five-case run. A real `sub-570_acq-CTA_ct` carotid run correctly stopped at
preflight because four required native products are absent, and headless QC
generated an honest binary-only montage/JSON with visual review pending. Current
interfaces and execution evidence are documented in `FLOWCAT_INTEGRATION.md` and
`BASELINE_EXECUTION_REPORT.md`. The finalized current package suite passes 133
tests; the focused additive core suite passes 67. These counts are kept separate
from the historical 128-test baseline.

## 1A. Repository overview

### Top-level structure and packages

- `aorta_cta_radiomics/` is the maintained aortic CTA analysis package. Its Python
  package is `aorta_cta_radiomics` under `src/`.
- Additive carotid modules are `flowcat_adapter/`, `artery_volume.py`,
  `carotid_analysis.py`, `perivascular/`, `output_schema.py`, `carotid_cli.py`,
  and `carotid_qc.py`. They consume complete, trusted native FLOWCAT outputs and
  fail closed rather than substituting the technical skeleton fallback.
- `cta_common/` provides shared image I/O, geometry checks, mask resampling, and
  physical-distance shell utilities. The additive milestone now declares
  `cta_common>=0.1.0` in package metadata and documents installing the local
  workspace package first. It is not published as part of this repository, so an
  external clean installation still needs an explicit distribution strategy.
- `scripts/` contains repository-wide DICOM conversion, multi-model segmentation,
  and legacy/experimental vessel-mask utilities.
- `subprojects/studies/anterior_circulation_supraaortic/` is a development-only
  study area for FLOWCAT segmentation and adaptive calcium/PVAT experiments.
- `external/arterial_flowcat/` is a local, Git-ignored FLOWCAT checkout. It is not
  a submodule, vendored source tree, or installed distribution.
- `aorta_cta_radiomics/outputs/` and study `outputs/` contain CSV, JSON, NIfTI,
  PNG, logs, and generated Slicer scripts. They are working artifacts, not an
  immutable results store.

### Main entry points and CLI commands

- `aorta_cta_radiomics.cli:app` exposes `aorta-cta run-single` and
  `aorta-cta run-batch`. `run_pipeline_case` is a large in-process, case-level
  implementation; `run_batch` is sequential.
- `aorta_cta_radiomics/scripts/run_manifest_staged.py` is the maintained cohort
  orchestrator. The console alias is `aorta-cta-staged`. It runs stages as
  subprocesses and records per-stage status and logs.
- Other console aliases are `aorta-cta-qc`, `aorta-cta-watchdog`, and
  `aorta-cta-progress`.
- `python -m aorta_cta_radiomics.carotid_cli manifest` creates a non-deserializing,
  hashed FLOWCAT inventory and returns status 2 when required outputs are missing.
  `python -m aorta_cta_radiomics.carotid_cli run` is the trusted-artifact,
  fail-closed single-case carotid executor.
- `python -m aorta_cta_radiomics.carotid_qc` performs headless geometry/completeness
  QC and can create JSON plus a three-plane montage even for a binary-only case.
- The supra-aortic study calls
  `scripts/run_flowcat_segmentation.py --profile fast-lowres --all` for the five
  pilot segmentations and `scripts/run_adaptive_vessel_burden_v2.py --all
  --fail-fast` for the adaptive burden outputs. The original interactive shell
  invocations were not persisted; these are the documented reproduction commands.

### Inputs, outputs, and processing sequence

The core aortic package expects a CTA NIfTI image and a geometrically compatible
binary aorta mask. DICOM ingestion occurs in separate repository scripts. The
principal outputs are NIfTI masks, long- and wide-form CSV files, JSON status/QC,
PNG QC images, logs, and Slicer-loading scripts. No maintained aortic Parquet or
database output was found.

The staged sequence is: optional VISTA segmentation; repository neurovascular
segmentation; base aortic analysis; calcium; fat/wall; protrusions; monolithic
analysis; radiomics; wall-thickness; and QC. Stage selection varies by manifest.
The main analysis computes basic segmentation QC, an approximate axial
centreline, geometry summaries, threshold-based calcium, a fixed 5-mm
perivascular fat shell, experimental fat-supported closure, and optional
PyRadiomics. Aorta subsegmentation is currently a placeholder assigning the whole
mask to one label.

The development supra-aortic sequence is separate: FLOWCAT low-resolution binary
arterial segmentation; local skeleton/EDT centreline fallback; diameter-adaptive
fat and calcium neighborhoods; bone-aware calcium partitioning; CSV/JSON/PNG
summaries. All five pilot cases are automated-pass but pending formal visual QC.

The additive carotid sequence is: non-executing manifest and SHA-256 inventory;
explicit trusted-artifact opt-in; complete native-output validation using the
original CTA path even when it is outside the FLOWCAT tree; graph-label mapping;
dual-strategy centreline-to-lumen anatomical rasterization with explicit
uncertainty; named RCCA/LCCA/RICA/LICA vessel-centred coordinates; fixed and
capped adaptive shells; exclusions, tissue composition, sectors, contacts and
profiles; typed case/vessel/sample outputs; then headless and manual QC. It only
reaches analysis when all required native inputs validate. On real `sub-570`, it
stopped at preflight as designed and wrote `failure.json`; no named measurement
was fabricated.

### Dependencies, models, hardware, and operating system

The package metadata requires Python 3.11 and NumPy, SciPy, pandas,
scikit-image, scikit-learn, SimpleITK, nibabel, PyRadiomics 3.0.1, YAML, Typer,
tqdm, Matplotlib, and seaborn. Optional paths use MONAI/PyTorch and surface tools.
`cta_common` is a declared but locally supplied workspace dependency. The audited FLOWCAT
segmentation uses locally reconstructed model files from its pinned checkout; it
does not download weights automatically.

The aortic path is CPU-first and has been used on macOS/Apple Silicon. The five
FLOWCAT segmentations recorded CPU execution; adaptive burden analysis also ran on
CPU. GPU memory is therefore not applicable to that baseline. Native FLOWCAT
post-processing requires compatible VTK/VMTK and PyTorch-Geometric components,
which were not available in the audited macOS environments. Linux/Ubuntu x86-64
with a validated VMTK stack is the recommended native-geometry environment; an
NVIDIA CUDA GPU is recommended for the full hybrid workload, but should not be
reported as required by CPU-only analysis stages.

### External repositories and licence

FLOWCAT is checked out from `https://github.com/FLOWCAT-CV/arterial`, branch
`models/2026-03`, commit
`1991ca5b238d7c868c9c3a0a390057bc89aee559`. The checkout was clean when audited
and the study runner enforces the commit. The repository `LICENSE`, rather than a
README badge, controls: Creative Commons Attribution-NonCommercial 4.0. This is a
material commercial-use and redistribution constraint requiring attribution and
licence review.

### Storage, naming, logging, provenance, and QC

Development CTA sources use `/Volumes/DICOM5` and BIDS-style filenames such as
`sub-570_acq-CTA_ct.nii.gz`. Outputs are organized by de-identified case ID. The
repository conversion script can place birth date and sex in JSON sidecars; those
sidecars must be treated as potentially identifying and are not reproduced here.

The staged runner stores command, log path, start/end time, return code, and a
sentinel-oriented status row. It overwrites `stage_status.csv` on a new invocation
and often tests only one sentinel, so it is not an immutable run ledger or a full
output-completeness check. Main outputs record software-version rows, but do not
consistently capture input/model hashes, Git state, configuration, hardware,
environment lock, or acquisition metadata. The adaptive pilot has stronger
config/input/output hashes and runtime versions, although it was produced from a
dirty development tree.

Automated QC covers mask volume, components, bounding box, spacing, simple shape
and mean-HU measures. `scripts/qc_slicer.py` supports case selection, task
categorization, Slicer scripts, and CSV/JSON review logs. It does not establish
anatomic completeness, named-artery correctness, field-of-view adequacy, or
clinical plausibility automatically. The additive headless `carotid_qc` validates
CTA/segmentation/optional overlay geometry, native-output completeness, and mask
metrics without reading descriptive NIfTI/DICOM metadata; it always leaves visual
review pending. A real binary-only `sub-570` QC JSON/montage exists, while formal
named-artery Slicer QC remains impossible until native labels are produced.

## 1B. Existing capability inventory

### CTA loading

Capability: CTA loading.
Current status: Verified for NIfTI CTA; DICOM CTA loading is separate.
Implementation location: `aorta_cta_radiomics.cli`, `cta_common/io.py`.
Entry point: `run_pipeline_case`; shared SimpleITK readers.
Input: CTA NIfTI path.
Output: SimpleITK image and NumPy array in `z,y,x` order.
Dependencies: SimpleITK, NumPy, `cta_common`.
Evidence that it is used: Cohort artifacts and five adaptive pilot CTAs were read successfully.
Tests: Covered indirectly by the 58 package tests and 38 `cta_common` tests; representative data executed.
Known limitations: An external install still needs a source for `cta_common`; NIfTI path is assumed.
Recommendation: Publish/vendor `cta_common` or retain the documented workspace install; validate geometry immediately after load.

### DICOM loading

Capability: DICOM loading.
Current status: Verified in repository conversion scripts, not integrated into the aortic package.
Implementation location: `scripts/convert_daylightdicom_to_bids.py` and full segmentation scripts.
Entry point: Conversion-script CLI.
Input: DICOM directory/series.
Output: Selected series in memory and converted NIfTI plus metadata.
Dependencies: pydicom, SimpleITK; dcm2niix in a separate full-pipeline path.
Evidence that it is used: Daylight BIDS-style CTA files exist on the development volume.
Tests: No audited end-to-end DICOM fixture test.
Known limitations: Series-selection rules are study-specific; source metadata can contain PHI.
Recommendation: Keep ingestion isolated, add de-identified fixtures and explicit series-selection audit.

### DICOM-to-NIfTI conversion

Capability: DICOM-to-NIfTI conversion.
Current status: Verified as a separate repository capability.
Implementation location: `scripts/convert_daylightdicom_to_bids.py`; `scripts/run_full_segmentation_pipeline.py`.
Entry point: Script CLI; the latter invokes dcm2niix.
Input: Selected DICOM CTA series.
Output: Compressed NIfTI and JSON sidecar.
Dependencies: SimpleITK/pydicom or dcm2niix.
Evidence that it is used: The five pilot inputs are converted BIDS-style CTA NIfTIs.
Tests: No deterministic conversion integration test was found.
Known limitations: Two implementations; affine/orientation equivalence is not demonstrated.
Recommendation: Select one authoritative converter and add voxel/physical-coordinate regression tests.

### DICOM-to-BIDS conversion

Capability: DICOM-to-BIDS conversion.
Current status: Partially verified; output is BIDS-style rather than demonstrated full BIDS compliance.
Implementation location: `scripts/convert_daylightdicom_to_bids.py`.
Entry point: Script CLI.
Input: Daylight DICOM hierarchy.
Output: `sub-*_acq-CTA_ct.nii.gz`, JSON, and CSV inventory.
Dependencies: pydicom, SimpleITK, study-specific naming logic.
Evidence that it is used: `/Volumes/DICOM5/daylightbids` supplies audited CTA cases.
Tests: No BIDS-validator result or automated compliance test found.
Known limitations: Sidecars may contain birth date and sex; this is a PHI/privacy risk.
Recommendation: Minimize metadata, run a BIDS validator, and document that existing data are BIDS-style.

### NIfTI loading

Capability: NIfTI loading.
Current status: Verified.
Implementation location: `cta_common/io.py`, aortic CLI, FLOWCAT study runner.
Entry point: Shared SimpleITK loaders and FLOWCAT inference wrapper.
Input: `.nii` or `.nii.gz` image/mask.
Output: Image objects, arrays, and geometry metadata.
Dependencies: SimpleITK; nibabel in some paths.
Evidence that it is used: Main cohort and all five supra-aortic pilot cases generated outputs.
Tests: Shared I/O unit tests plus representative execution.
Known limitations: FLOWCAT and local code do not use one unified affine abstraction.
Recommendation: Route all downstream adapters through one validated image-geometry schema.

### Orientation normalization

Capability: Orientation normalization.
Current status: Missing as a consistently enforced repository-wide step.
Implementation location: No authoritative implementation found; individual tools infer orientation.
Entry point: None.
Input: Arbitrarily oriented CTA/mask.
Output: No standardized canonical image is guaranteed.
Dependencies: Would require SimpleITK or nibabel orientation transforms.
Evidence that it is used: Unable to verify; pilot segmentations include both LAS and LPS orientations.
Tests: No orientation permutation/flip integration test found.
Known limitations: FLOWCAT inference derives spacing from affine diagonal and does not canonicalize orientation.
Recommendation: Add explicit canonicalization or preserve-and-validate transforms at every boundary.

### RAS/LPS conversion

Capability: RAS/LPS conversion.
Current status: Partially verified in development helpers; not unified across the maintained pipeline.
Implementation location: `cta_common/io.py` physical-coordinate helpers and supra-aortic adaptive code.
Entry point: Coordinate conversion helpers.
Input: Physical points and image geometry.
Output: Converted RAS/LPS points.
Dependencies: SimpleITK, NumPy.
Evidence that it is used: Adaptive centreline CSV handling records RAS/LPS choices.
Tests: Some study unit coverage; no end-to-end FLOWCAT-to-NIfTI landmark registration test.
Known limitations: FLOWCAT post-processing uses an LPI-corner-relative millimetre frame, not simply absolute RAS.
Recommendation: Make coordinate frame an explicit typed field and require round-trip landmark tests.

### Spacing validation

Capability: Spacing validation.
Current status: Verified for basic image/mask compatibility.
Implementation location: `cta_common/io.py`, segmentation QC.
Entry point: Geometry comparison/resampling and QC functions.
Input: CTA and mask geometry.
Output: Validation result, metadata, or resampled mask.
Dependencies: SimpleITK, NumPy.
Evidence that it is used: Pilot QC records original spacing; main outputs include spacing checks.
Tests: `cta_common` tests include geometry behavior.
Known limitations: Passing spacing alone does not prove origin, direction, or affine equivalence.
Recommendation: Validate size, spacing, origin, direction, affine, and physical corner agreement together.

### CTA resampling

Capability: CTA resampling.
Current status: Verified in analysis/radiomics paths.
Implementation location: `cta_common/io.py`, `aorta_cta_radiomics/radiomics.py`.
Entry point: Image resampling helpers; PyRadiomics preprocessing.
Input: CTA plus target geometry/spacing.
Output: Resampled CTA; masks use nearest-neighbour interpolation.
Dependencies: SimpleITK, PyRadiomics.
Evidence that it is used: Radiomics configuration specifies 1-mm resampling; mask alignment helpers are exercised.
Tests: Unit coverage for shared resampling; real PyRadiomics extraction was not baseline-tested.
Known limitations: Upsampling does not create true spatial resolution; interpolator choices differ by path.
Recommendation: Store source/target geometry and interpolation policy in provenance.

### Arterial segmentation

Capability: Arterial segmentation.
Current status: Verified for a broad binary FLOWCAT extracranial tree in five cases; pending formal visual QC.
Implementation location: Study `run_flowcat_segmentation.py`; separate TotalSegmentator/merge scripts.
Entry point: FLOWCAT `perform_single_inference_nnunet` wrapper.
Input: CTA NIfTI and local FLOWCAT model files.
Output: Binary `segmentation.nii.gz` per case.
Dependencies: Pinned FLOWCAT checkout, nnU-Net/PyTorch stack, model weights.
Evidence that it is used: Five `fast-lowres` segmentation artifacts feed adaptive burden analysis.
Tests: Five executions succeeded; no named-artery or clinical accuracy test.
Known limitations: Low-resolution binary tree, includes aorta, no anatomic labels; all cases visual-QC pending.
Recommendation: Treat as development input until native FLOWCAT labelling and formal Slicer QC pass.

### Aortic segmentation

Capability: Aortic segmentation.
Current status: Verified for existing masks and repository segmentation stages; source-model accuracy not revalidated here.
Implementation location: Staged runner, repository neurovascular/TotalSegmentator paths, aortic CLI.
Entry point: Staged `vista`/repository segmentation stages and `run_pipeline_case`.
Input: CTA.
Output: Binary aorta mask.
Dependencies: External/local segmentation models; SimpleITK.
Evidence that it is used: Large Daylight output cohorts and detailed `sub-570` aorta artifacts exist.
Tests: Package unit tests and artifact checks; formal segmentation ground truth absent.
Known limitations: Source varies by run; model and mask provenance are incomplete.
Recommendation: Record model/checkpoint/hash and validate against expert aortic masks.

### Aortic branch segmentation

Capability: Aortic branch segmentation.
Current status: Unable to verify as an anatomically labelled maintained output.
Implementation location: FLOWCAT may provide native graph labels; local aorta segmenter is a placeholder.
Entry point: No verified local entry point producing labelled arch branches.
Input: CTA/arterial tree.
Output: Expected labelled supra-aortic branches.
Dependencies: Native FLOWCAT geometry/labelling or validated alternative.
Evidence that it is used: No labelled branch-volume artifact found in the five-case run.
Tests: None found for arch branch identity.
Known limitations: Current five-case output is binary and includes the aorta.
Recommendation: Complete and validate native FLOWCAT output mapping before branch-specific analysis.

### Common carotid segmentation

Capability: Common carotid segmentation.
Current status: Coarse masks may be available from TotalSegmentator; not verified in the active FLOWCAT pilot.
Implementation location: Repository TotalSegmentator and merged-label scripts.
Entry point: `scripts/total_segmentator.py` and labelmap-building scripts.
Input: CTA.
Output: Left/right coarse carotid masks where supported.
Dependencies: TotalSegmentator and its model stack.
Evidence that it is used: Coarse labelmap construction exists; no named CCA artifact feeds the adaptive pilot.
Tests: No artery-specific segmentation accuracy test found.
Known limitations: Coarse anatomic prior, not a validated lumen/branch label for this pipeline.
Recommendation: Use only as a prior and validate FLOWCAT graph-to-volume CCA labels.

### Internal carotid segmentation

Capability: Internal carotid segmentation.
Current status: Unable to verify in the active pipeline.
Implementation location: Potential FLOWCAT vessel-type graph output; separate coarse segmentation scripts.
Entry point: Native FLOWCAT labelling chain is not currently generated.
Input: Arterial segmentation and native centreline/graph outputs.
Output: Expected left/right ICA mask/branch labels.
Dependencies: VMTK, FLOWCAT graph labelling, coordinate-safe volume mapping.
Evidence that it is used: Named arteries are reported unavailable for all five pilot cases.
Tests: No ICA ground-truth or mapping test.
Known limitations: Binary segmentation cannot distinguish ICA from ECA/CCA.
Recommendation: Make ICA labelling a P1 validation target before local perivascular claims.

### External carotid segmentation

Capability: External carotid segmentation.
Current status: Missing from verified active outputs.
Implementation location: Potential FLOWCAT graph labels only; not generated in the pilot.
Entry point: None verified end to end.
Input: Native FLOWCAT centreline/graph chain.
Output: Expected left/right ECA branch labels.
Dependencies: VMTK, FLOWCAT labeller, graph-to-volume adapter.
Evidence that it is used: No ECA-labelled output found.
Tests: None found.
Known limitations: Broad binary tree does not encode ECA identity or branch extent.
Recommendation: Validate class taxonomy and ECA mapping before use.

### Subclavian segmentation

Capability: Subclavian segmentation.
Current status: Coarse model masks exist in separate paths; not verified as active named FLOWCAT output.
Implementation location: TotalSegmentator/merged-label scripts; potential FLOWCAT graph labels.
Entry point: Separate segmentation scripts.
Input: CTA.
Output: Coarse left/right subclavian artery masks where available.
Dependencies: TotalSegmentator or native FLOWCAT chain.
Evidence that it is used: Labelmap merge code includes subclavian structures; adaptive pilot reports no named arteries.
Tests: No artery-specific reference-mask comparison.
Known limitations: Separate outputs may mix conventions and are not authoritative lumen masks.
Recommendation: Use coarse masks as priors only and validate named FLOWCAT branches.

### Vertebral segmentation (arterial)

Capability: Vertebral segmentation (arterial).
Current status: Coarse/legacy paths exist; active named-artery output is unavailable.
Implementation location: TotalSegmentator merge code, legacy vertebral workflow, potential FLOWCAT labels.
Entry point: Separate scripts/modules, not the active adaptive runner.
Input: CTA.
Output: Expected left/right vertebral artery masks.
Dependencies: External segmentation model or FLOWCAT native labelling.
Evidence that it is used: Broad binary FLOWCAT tree includes arterial voxels, but no verified vertebral label.
Tests: Legacy workflow availability only; no active segmentation accuracy test.
Known limitations: Bone proximity makes calcium attribution particularly error-prone.
Recommendation: Validate vertebral labels and bone-exclusion behavior separately.

### Intracranial artery segmentation

Capability: Intracranial artery segmentation.
Current status: Separate neurovascular segmentation paths exist; not integrated with the audited FLOWCAT extracranial output.
Implementation location: Repository TopCoW/neurovascular scripts; FLOWCAT supports a separate territory if configured.
Entry point: Full segmentation scripts, not the five-case extracranial runner.
Input: Head/neck CTA.
Output: Intracranial arterial masks/labels.
Dependencies: Corresponding segmentation models.
Evidence that it is used: Repository code exists; no intracranial FLOWCAT artifact in the five-case output tree.
Tests: No integrated extracranial-to-intracranial continuity test.
Known limitations: Different models, labels, resolutions, and coordinate conventions.
Recommendation: Define a separate validated contract before combining territories.

### Centreline extraction

Capability: Centreline extraction.
Current status: Verified only for approximate local methods; native FLOWCAT/VMTK failed on representative `sub-570`.
Implementation location: `aorta_cta_radiomics/centerline.py`; study skeleton/EDT fallback; FLOWCAT `CenterlineExtractor`.
Entry point: `approximate_centerline_by_slices`; `extract_skeleton_edt_centerlines.py`; native FLOWCAT preprocessing.
Input: Binary vessel mask or surface.
Output: Axial COM points or technical skeleton paths; native VMTK expected centreline VTK.
Dependencies: NumPy/SciPy/scikit-image; VTK/VMTK for native path.
Evidence that it is used: Local fallback feeds all five adaptive cases; VMTK smoke attempt on 570 was recorded.
Tests: Synthetic/local unit coverage; no native successful representative case.
Known limitations: Fallback is not VMTK-equivalent and has no anatomic branch identity.
Recommendation: Repair native Linux VMTK execution and retain fallback only with explicit non-equivalence.

### Centreline resampling

Capability: Centreline resampling.
Current status: Legacy resampling and additive 1-mm physical arc-length resampling are implemented; additive real native execution is blocked.
Implementation location: Aortic utilities, FLOWCAT post-processing, `perivascular/coordinates.py`.
Entry point: Local utilities or `analyze_carotid_case` after trusted validation.
Input: Ordered approximate or native physical-RAS branch coordinates.
Output: Resampled points, path distance, tangent/parallel-transport frame and derivatives.
Dependencies: NumPy/SciPy; complete native branch for additive real use.
Evidence that it is used: Legacy geometry uses ordered points; additive tests exercise the 1-mm contract, but native artifacts are absent.
Tests: Synthetic coordinate/core tests plus legacy unit coverage.
Known limitations: Direction/endpoints and sampling effects lack real native/landmark validation.
Recommendation: Validate arc-length and frame invariance on analytic and complete native branches.

### Branch extraction

Capability: Branch extraction.
Current status: Verified only as technical skeleton paths; anatomical branch extraction is unavailable.
Implementation location: Study skeleton fallback; FLOWCAT native branch models expected.
Entry point: `extract_skeleton_edt_centerlines.py`; FLOWCAT branch processing.
Input: Skeleton graph or VMTK centreline network.
Output: Technical paths or expected branch VTK/model files.
Dependencies: scikit-image/SciPy; VTK/VMTK for native.
Evidence that it is used: Five pilots contain 81–208 technical paths; no native branch models generated.
Tests: Study unit tests; no expert branch reference.
Known limitations: Technical paths are not anatomical arteries and may split at noise/spurs.
Recommendation: Use FLOWCAT branch graph as authority after successful native execution.

### Bifurcation detection

Capability: Bifurcation detection.
Current status: Technical skeleton junctions and additive anatomy-aware bifurcation exclusion are implemented; real named-carotid validation is blocked.
Implementation location: Study skeleton graph code; `artery_volume.py`; `carotid_analysis.py`.
Entry point: Skeleton extraction or `analyze_carotid_case` after native FLOWCAT validation.
Input: Technical skeleton, or trusted labelled centreline branches and lumen mask.
Output: Technical junctions; additive lumen/shell bifurcation-exclusion masks and distance fields.
Dependencies: NumPy/SciPy/scikit-image and complete native FLOWCAT branches for the additive path.
Evidence that it is used: Pilot endpoint counts exist; no real named-carotid exclusion output exists because `sub-570` failed preflight.
Tests: Synthetic topology and artery-volume/carotid-core tests; no expert landmark ground truth.
Known limitations: Endpoint proximity is not anatomical truth and variants/branch fragmentation remain unvalidated.
Recommendation: Compare exclusion zones and branch junctions with expert carotid landmarks on a complete native case.

### Anatomical artery labelling

Capability: Anatomical artery labelling.
Current status: Graph-to-branch and branch-to-lumen implementations are synthetic-tested, but named labels remain absent from all real pilot outputs.
Implementation location: FLOWCAT `VesselLabeller`; `flowcat_adapter/labels.py`; `artery_volume.py`; `carotid_analysis.py`.
Entry point: `validate_flowcat_case` followed by `analyze_carotid_case` or `python -m aorta_cta_radiomics.carotid_cli run`.
Input: Trusted predicted graph, centreline segments/radii, binary lumen, CTA geometry, and required branch codes.
Output: Typed artery identities plus final/nearest/connected label volumes, confidence, uncertainty, bifurcation, and strategy-disagreement masks.
Dependencies: Complete native FLOWCAT/VMTK/PyG output chain and additive adapter/core.
Evidence that it is used: Synthetic tests execute both propagation strategies; real `sub-570` correctly produced no labelled volume because four native inputs are missing.
Tests: Contract, mapping, analytic-volume, and carotid-core tests; no complete representative native case or expert truth.
Known limitations: The core refuses missing or multiply mapped RCCA/LCCA/RICA/LICA; label 14 is uncertainty; it never infers a carotid bulb without a supplied evidence-bearing interval.
Recommendation: Preserve fail-closed behavior and validate graph/volume labels, confidence, variants, and uncertainty in Slicer.

### Vessel graph generation

Capability: Vessel graph generation.
Current status: Technical fallback graph is verified; native FLOWCAT graphs are not generated.
Implementation location: Study skeleton extractor; FLOWCAT expected `segments_graph*.pickle` and `local_graph.pickle`.
Entry point: Skeleton graph builder; native FLOWCAT graph pipeline.
Input: Skeleton/centreline branches.
Output: Local technical graph or expected NetworkX pickles.
Dependencies: scikit-image/NetworkX; FLOWCAT/PyG for native labels.
Evidence that it is used: Technical paths drive adaptive distances; no native pickle exists in five cases.
Tests: Study graph tests; trusted-pickle native loading not baseline-executed.
Known limitations: Fallback graph lacks anatomical labels and native features.
Recommendation: Treat native graph as future authority and load pickles only from trusted local runs.

### Vessel surface generation

Capability: Vessel surface generation.
Current status: Unable to verify successfully for FLOWCAT pilot.
Implementation location: FLOWCAT `CenterlineExtractor.perform_preprocessing`; optional local surface tooling.
Entry point: Native FLOWCAT preprocessing.
Input: Binary arterial segmentation.
Output: VTK/STL surfaces.
Dependencies: VTK, VMTK and mesh-cleaning stack.
Evidence that it is used: A 570 smoke attempt reached surface processing but native centreline failed.
Tests: No successful representative surface-to-centreline chain.
Known limitations: Valid normals could not be computed after prescribed mesh-bottom retries.
Recommendation: Persist native stderr, validate watertight/normals QC, and rerun in supported Linux environment.

### Vessel geometry extraction

Capability: Vessel geometry extraction.
Current status: Approximate real outputs exist; additive native-FLOWCAT geometry consumption and vessel-centred derivation are synthetic-tested but not real-case executed.
Implementation location: Aortic `centerline.py`, `lumen_geometry.py`, study adaptive core, `carotid_analysis.py`, `perivascular/coordinates.py`.
Entry point: Legacy case analysis or fail-closed carotid analysis.
Input: Approximate mask/path, or trusted native labelled centreline with radii and CTA geometry.
Output: Length, radius/diameter, curvature, torsion, tortuosity, frame, and local/sample summaries.
Dependencies: NumPy/SciPy; complete native FLOWCAT geometry for authoritative carotid execution.
Evidence that it is used: Legacy/pilot CSVs exist; additive outputs are fixture-tested but `sub-570` stopped before geometry loading.
Tests: Legacy unit/five-case tests plus synthetic vessel-coordinate/carotid-core tests.
Known limitations: Legacy proxies are not native; additive geometry accuracy remains unverified on real FLOWCAT artifacts.
Recommendation: Keep method fields distinct and validate native geometry on analytic shapes and expert-reviewed cases.

### Diameter extraction

Capability: Diameter extraction.
Current status: Local proxy is real-case verified; additive carotid diameter is implemented as twice trusted FLOWCAT radius and synthetic-tested.
Implementation location: `lumen_geometry.py`, adaptive burden core, `carotid_analysis.py`.
Entry point: Legacy/adaptive analysis or `analyze_carotid_case`.
Input: Axial/technical reference, or native centreline `MaximumInscribedSphereRadius`.
Output: Diameter in millimetres with an explicit method field and adaptive shell law.
Dependencies: NumPy/SciPy; native FLOWCAT radius for additive carotid analysis.
Evidence that it is used: Real adaptive `sub-570` median proxy is 4.166 mm; additive real run never reached diameter extraction.
Tests: Five-case proxy execution and synthetic additive core/shell tests.
Known limitations: `2 * radius` is a lumen reference, never an outer-wall diameter or wall-thickness estimate.
Recommendation: Preserve the method label and validate against manual orthogonal lumen diameters while retaining floor/cap sensitivity.

### Radius extraction

Capability: Radius extraction.
Current status: EDT fallback is real-case verified; native radius loading/validation/propagation is implemented and synthetic-tested but no real native array exists.
Implementation location: Study skeleton/EDT extractor; `flowcat_adapter/centerlines.py`; `carotid_analysis.py`.
Entry point: Fallback extractor or trusted `validate_flowcat_case`/carotid run.
Input: Binary mask for fallback, or trusted `centerline_segments_array.npy` for native path.
Output: Technical EDT radius or typed native local radius in millimetres.
Dependencies: SciPy EDT; explicit trust and complete native FLOWCAT array for additive path.
Evidence that it is used: Fallback radii size all five adaptive shells; no real native radius was available to the additive run.
Tests: Five-case fallback and synthetic trusted-array/carotid-core tests.
Known limitations: EDT radius is not `MaximumInscribedSphereRadius`; native values still need real-case mesh/centreline validation.
Recommendation: Never substitute methods silently; compare both on complete native cases.

### Curvature extraction

Capability: Curvature extraction.
Current status: Approximate real output exists; additive physical-RAS resampling/curvature is implemented and synthetic-tested, not real native executed.
Implementation location: `centerline.py`; `perivascular/coordinates.py`; `carotid_analysis.py`.
Entry point: Legacy geometry or `analyze_carotid_case`.
Input: Ordered approximate points or trusted native labelled branch.
Output: Local curvature per millimetre and vessel mean with method/context.
Dependencies: NumPy/SciPy and valid native centreline for additive use.
Evidence that it is used: Legacy CSVs exist; additive real 570 stopped before geometry.
Tests: Coordinate/core synthetic tests; analytic/noise matrix remains incomplete.
Known limitations: Resampling, smoothing, direction and native centreline quality affect derivatives.
Recommendation: Validate circles/helices and real native branches across sampling choices.

### Torsion extraction

Capability: Torsion extraction.
Current status: Additive torsion per millimetre is implemented/synthetic-tested; no real native output was generated.
Implementation location: `perivascular/coordinates.py`; `carotid_analysis.py`; upstream local-graph features remain separately expected.
Entry point: `analyze_carotid_case`.
Input: Resampled physical-RAS native branch.
Output: Sample torsion per millimetre, nullable where unstable.
Dependencies: NumPy/SciPy and complete native centreline.
Evidence that it is used: Synthetic coordinate/core outputs only; real 570 stopped at preflight.
Tests: Straight/curved synthetic behavior is covered; full noisy-helix validation is incomplete.
Known limitations: High-order derivative is noise/sampling sensitive and may not match upstream FLOWCAT feature definition.
Recommendation: Compare with analytic helices and upstream native feature on identical points.

### Tortuosity extraction

Capability: Tortuosity extraction.
Current status: Legacy proxy and additive branch path/chord tortuosity are implemented; additive real execution is blocked.
Implementation location: Aortic geometry code; `carotid_analysis.py`.
Entry point: Legacy analysis or fail-closed carotid core.
Input: Ordered branch with defined endpoints.
Output: Dimensionless arc-length/chord ratio with branch identity.
Dependencies: NumPy and complete native labelled branch.
Evidence that it is used: Legacy outputs exist; additive fixture outputs only.
Tests: Unit/synthetic core coverage; no expert/native comparison.
Known limitations: Endpoint/branch definitions and method differ from other tortuosity metrics.
Recommendation: Preserve formula/numerator/denominator and validate identical endpoints.

### Calcium segmentation

Capability: Calcium segmentation.
Current status: Verified for threshold/seed methods and adaptive bone-aware candidates; pending formal visual QC.
Implementation location: `calcification.py`, `calcium_omics.py`, adaptive `bone_aware_calcium.py`.
Entry point: Main calcium stage; adaptive vessel-burden runner.
Input: CTA, lumen/vessel mask, and optional bone priors.
Output: Candidate/retained calcium masks and threshold summaries.
Dependencies: SimpleITK, NumPy, SciPy; TotalSegmentator for bone priors.
Evidence that it is used: Large cohort calcium tables and five adaptive masks/summaries exist.
Tests: Package/study unit tests and five executions.
Known limitations: CTA candidate calcium is not confirmed plaque; skull/vertebra contact can contaminate; no Agatston validity.
Recommendation: Preserve candidate terminology, anatomy-aware bone classes, and expert lesion validation.

### Calcium volume measurement

Capability: Calcium volume measurement.
Current status: Verified for candidate masks.
Implementation location: `calcium_omics.py` and adaptive burden runner.
Entry point: Calcium analysis stages.
Input: Binary calcium candidate masks plus voxel volume.
Output: Volumes in millilitres and per-component summaries.
Dependencies: NumPy, SimpleITK geometry.
Evidence that it is used: Five pilot retained volumes range from 0.120 to 4.218 mL; cohort tables exist.
Tests: Unit tests and representative execution.
Known limitations: Volume validity inherits segmentation and physical-geometry errors.
Recommendation: Store voxel count, voxel volume, mask hash, territory, and uncertainty partition.

### Calcium classification

Capability: Calcium classification.
Current status: Development-only partition verified; clinical plaque classification is missing.
Implementation location: Adaptive `bone_aware_calcium.py`; `calcium_omics.py` component summaries.
Entry point: Adaptive burden runner.
Input: Raw HU candidates, bone masks, contact/halo reconstruction.
Output: Retained, definite-bone, and ambiguous candidate masks.
Dependencies: NumPy/SciPy/SimpleITK; craniofacial bone model outputs.
Evidence that it is used: Exact three-way partition exists for all five cases.
Tests: Study unit tests and five executions.
Known limitations: Component-level propagation can overclassify; `sub-766` bone map is invalid because required teeth masks are empty.
Recommendation: Validate each class against expert voxel/component labels and retain ambiguity.

### Calcium proximity analysis

Capability: Calcium proximity analysis.
Current status: Partially verified for bone/dental proximity; full tissue interaction model is missing.
Implementation location: Adaptive bone-aware analysis and separate `sub-570` dental demonstration.
Entry point: Adaptive runner; dental analysis scripts.
Input: Calcium candidates plus bone/teeth/arterial geometry.
Output: Contact/proximity partitions and distances.
Dependencies: SimpleITK/SciPy, TotalSegmentator priors.
Evidence that it is used: Five-case physical-distance audit found no arterial candidate within 3 mm of teeth/mandible.
Tests: Development execution; no ground-truth proximity benchmark.
Known limitations: Skull/vertebra confounding remains; dental demonstration is experimental and separate.
Recommendation: Analyse artery, bone class, distance, and component uncertainty jointly.

### Perivascular shell generation

Capability: Perivascular shell generation.
Current status: Verified for fixed 5-mm and adaptive diameter-capped physical shells.
Implementation location: `cta_common/shells.py`, `fat_omics.py`, adaptive burden core.
Entry point: Main fat analysis and adaptive runner.
Input: Binary vessel mask, spacing, and thickness rule.
Output: Inner/outer physical-distance shell masks.
Dependencies: SciPy anisotropic EDT, SimpleITK geometry.
Evidence that it is used: Cohort PVAT tables and five adaptive inner/outer masks exist.
Tests: Shared/study unit tests and five-case execution.
Known limitations: Broad binary tree and field-of-view truncation; nearest-centreline assignment is a development approximation.
Recommendation: Use `clip(alpha * diameter, floor, maximum)` per named artery and flag truncation.

### Fat thresholding

Capability: Fat thresholding.
Current status: Verified.
Implementation location: `fat_omics.py` and adaptive burden configuration/core.
Entry point: Main fat stage and adaptive runner.
Input: CTA HU within perivascular shell.
Output: Voxels in the configured fat window, currently -190 to -30 HU.
Dependencies: NumPy/SimpleITK.
Evidence that it is used: Five pilot fat masks and main cohort fat results exist.
Tests: Unit tests and representative execution.
Known limitations: Scanner/reconstruction sensitivity and contrast-enhanced partial volume are unvalidated.
Recommendation: Record acquisition/reconstruction metadata and perform threshold sensitivity analysis.

### PVAT volume

Capability: PVAT volume.
Current status: Verified for candidate fat voxels; pending anatomical/named-vessel validation.
Implementation location: `fat_omics.py` and adaptive runner.
Entry point: Fat analysis stages.
Input: Perivascular shell and HU-thresholded fat mask.
Output: Total/layer fat volume in millilitres.
Dependencies: SimpleITK geometry, NumPy.
Evidence that it is used: Five adaptive totals range from 15.639 to 62.563 mL.
Tests: Unit tests and five-case execution.
Known limitations: Current broad tree includes aorta; intracranial “PVAT” is not physiologic PVAT.
Recommendation: Restrict to validated named extracranial arteries and preserve longitudinal profiles.

### PVAT attenuation

Capability: PVAT attenuation.
Current status: Verified in the maintained fat analysis path.
Implementation location: `fat_omics.py`.
Entry point: Main fat analysis.
Input: HU values in PVAT mask/layers.
Output: Mean and distributional attenuation summaries.
Dependencies: NumPy/pandas.
Evidence that it is used: Main cohort fat feature tables contain attenuation-derived output.
Tests: Package tests; five adaptive summary audit focused on volume rather than full attenuation plausibility.
Known limitations: Contrast phase, kernel, partial volume, and shell definition are confounders.
Recommendation: Stratify by acquisition and report robust local distributions, not only whole-vessel mean.

### PVAT radiomics

Capability: PVAT radiomics.
Current status: Implemented but unable to verify on representative real extraction in the audited baseline.
Implementation location: `aorta_cta_radiomics/radiomics.py` and `fat_omics.py` custom textures.
Entry point: Optional radiomics stage.
Input: CTA and sufficiently large stable PVAT ROI.
Output: PyRadiomics/custom intensity and texture features.
Dependencies: PyRadiomics 3.0.1, SimpleITK, NumPy.
Evidence that it is used: Radiomics code/config exists; the audited dynamic calcium profile disables radiomics/geometry.
Tests: Imports/unit paths only; no verified real-case PyRadiomics baseline.
Known limitations: Small regions, resampling, discretization, and scanner sensitivity threaten stability.
Recommendation: Gate by ROI size/reliability and add IBSI-oriented phantom/repeatability tests.

### Perivascular soft-tissue analysis

Capability: Perivascular soft-tissue analysis.
Current status: Experimental and incomplete.
Implementation location: `fat_wall.py`, wall-thickness code, adaptive residual concepts.
Entry point: Fat/wall and wall-thickness stages.
Input: CTA, vessel mask, and perivascular region.
Output: Fat-supported closure/residual soft-tissue masks and summaries.
Dependencies: NumPy/SciPy/SimpleITK.
Evidence that it is used: Cohort wall-from-fat and wall-thickness tables exist.
Tests: Unit paths and artifacts, but no histologic/reader validation.
Known limitations: Residual non-fat tissue is not arterial wall; stage sentinels can misstate completeness.
Recommendation: Rename as residual perivascular soft tissue and model composition/contact explicitly.

### Muscle segmentation

Capability: Muscle segmentation.
Current status: Coarse broad models exist elsewhere; targeted perivascular muscle integration is missing.
Implementation location: TotalSegmentator outputs and repository segmentation scripts.
Entry point: Broad anatomical segmentation scripts.
Input: CTA.
Output: Coarse muscle masks where model labels exist.
Dependencies: TotalSegmentator/model environment.
Evidence that it is used: Model path exists; no muscle mask was consumed by five-case adaptive analysis.
Tests: No perivascular muscle accuracy/contact test.
Known limitations: Label granularity and neck coverage vary.
Recommendation: Use coarse masks as priors, preserve residual class, validate at artery-contact surfaces.

### Sternocleidomastoid segmentation

Capability: Sternocleidomastoid segmentation.
Current status: Missing from the verified perivascular pipeline.
Implementation location: No targeted integrated implementation found.
Entry point: None.
Input: CTA neck volume.
Output: Expected left/right SCM masks.
Dependencies: A suitable anatomical model or deterministic/manual labels.
Evidence that it is used: No SCM output or downstream feature was found.
Tests: None.
Known limitations: Coarse general models may not provide reliable distinct SCM labels.
Recommendation: Evaluate available model labels before introducing new training.

### Longus colli segmentation

Capability: Longus colli segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA neck volume.
Output: Expected longus colli mask.
Dependencies: Validated anatomical segmentation source.
Evidence that it is used: No artifact or feature found.
Tests: None.
Known limitations: Small paired muscles and low CTA soft-tissue contrast.
Recommendation: Start with coarse/manual validation and an explicit unknown compartment.

### Scalene segmentation

Capability: Scalene segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA lower neck.
Output: Expected scalene muscle masks.
Dependencies: Validated anatomical segmentation source.
Evidence that it is used: No artifact or downstream feature found.
Tests: None.
Known limitations: FOV/shoulder artifact and intermuscular boundaries.
Recommendation: Defer until carotid core pipeline is functional; validate contact rather than whole-volume accuracy first.

### Strap-muscle segmentation

Capability: Strap-muscle segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA anterior neck.
Output: Expected strap-muscle masks.
Dependencies: Validated anatomical model.
Evidence that it is used: No output found.
Tests: None.
Known limitations: Closely apposed small muscles and variable contrast.
Recommendation: Treat as P2 and preserve grouped-vs-individual label uncertainty.

### Venous segmentation

Capability: Venous segmentation.
Current status: Coarse venous masks may exist in broad model outputs; not integrated into perivascular analysis.
Implementation location: TotalSegmentator/merged-label scripts.
Entry point: Broad anatomical segmentation scripts.
Input: CTA.
Output: Coarse vein masks.
Dependencies: TotalSegmentator/model environment.
Evidence that it is used: Merge code includes some venous labels; adaptive pipeline does not consume them.
Tests: No vein-contact validation.
Known limitations: Arterial-phase timing can make venous borders/contrast unreliable.
Recommendation: Add coarse vein priors and explicit confidence/contact QC before feature extraction.

### Internal jugular segmentation

Capability: Internal jugular segmentation.
Current status: Potential coarse labels available, but active use is unable to verify.
Implementation location: Broad anatomical segmentation/labelmap scripts.
Entry point: Separate segmentation scripts.
Input: CTA neck volume.
Output: Expected left/right internal jugular masks.
Dependencies: TotalSegmentator or another validated model.
Evidence that it is used: No jugular-contact output exists in the five adaptive cases.
Tests: None against expert masks.
Known limitations: Variable filling, collapse, and arterial-phase contrast.
Recommendation: Validate surface-contact measurements at CCA/bulb/ICA levels.

### Subclavian vein segmentation

Capability: Subclavian vein segmentation.
Current status: Coarse label exists in broad segmentation code; not integrated.
Implementation location: `scripts/total_segmentator.py` and merged labelmap logic.
Entry point: Broad anatomical segmentation scripts.
Input: CTA lower neck/chest.
Output: Coarse subclavian vein mask.
Dependencies: TotalSegmentator.
Evidence that it is used: Label is referenced in segmentation-union code; no perivascular feature consumes it.
Tests: No reference-mask or contact test.
Known limitations: Shoulder artifact and contrast timing; risk of artery/vein label confusion.
Recommendation: Keep as non-authoritative prior with automated overlap/side checks.

### Vertebral venous plexus segmentation

Capability: Vertebral venous plexus segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA around vertebral column.
Output: Expected plexus mask.
Dependencies: A specialized validated model or manual reference.
Evidence that it is used: No output found.
Tests: None.
Known limitations: Small variable venous structures and weak arterial-phase conspicuity.
Recommendation: Defer to P2/P3 pending scientific value and feasible ground truth.

### Bone segmentation

Capability: Bone segmentation.
Current status: Verified for coarse TotalSegmentator craniofacial/vertebral priors in the adaptive pilot.
Implementation location: Adaptive bone-aware runner and TotalSegmentator scripts.
Entry point: Bone-prior generation followed by `run_adaptive_vessel_burden_v2.py`.
Input: CTA.
Output: Bone union and named source masks where available.
Dependencies: TotalSegmentator, SimpleITK.
Evidence that it is used: Bone-aware partitions exist in all five cases; validity is true in four and false in `sub-766`.
Tests: Study unit tests and five-case execution.
Known limitations: Proximal target-bone scope is unresolved; model absence/emptiness can invalidate the map.
Recommendation: Define required bone classes by arterial territory and fail explicitly on missing prerequisites.

### Vertebral segmentation (bone)

Capability: Vertebral segmentation (bone).
Current status: Coarse model output is available; territory-specific use is incomplete.
Implementation location: TotalSegmentator bone outputs and adaptive bone union.
Entry point: Craniofacial/bone segmentation stage.
Input: CTA.
Output: Vertebral bone masks included in bone priors where present.
Dependencies: TotalSegmentator.
Evidence that it is used: Vertebral/skull confounding motivated bone-aware calcium partitioning.
Tests: No vertebra-specific accuracy or vertebral-artery calcium validation.
Known limitations: Close apposition can remove true arterial calcium or retain bone fragments.
Recommendation: Use distance/contact plus component uncertainty, validated specifically around vertebral arteries.

### Transverse-foramen segmentation

Capability: Transverse-foramen segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA/cervical bone volume.
Output: Expected transverse-foramen masks or centre coordinates.
Dependencies: Specialized bone anatomy model or deterministic CT analysis.
Evidence that it is used: No output found.
Tests: None.
Known limitations: Without this structure, vertebral artery/bone context is coarse.
Recommendation: Consider a deterministic vertebral-level prior after the core carotid milestone.

### Thyroid segmentation

Capability: Thyroid segmentation.
Current status: Coarse TotalSegmentator output may be available; not integrated into perivascular analysis.
Implementation location: Broad anatomical segmentation paths.
Entry point: TotalSegmentator scripts.
Input: CTA neck.
Output: Thyroid mask.
Dependencies: TotalSegmentator.
Evidence that it is used: No thyroid-contact or exclusion feature found in active outputs.
Tests: No representative accuracy/contact test.
Known limitations: Contrast and pathology may alter gland appearance.
Recommendation: Add only as a coarse contextual compartment with overlap QC.

### Airway segmentation

Capability: Airway segmentation.
Current status: Implemented in a separate OSA subproject, not integrated into the audited arterial pipeline.
Implementation location: `subprojects/stroke-cta-osa` and broad anatomy scripts.
Entry point: Subproject-specific analysis.
Input: CTA/head-neck imaging.
Output: Airway mask and subproject metrics.
Dependencies: Subproject/model-specific stack.
Evidence that it is used: Separate airway analysis exists; no arterial perivascular consumer found.
Tests: Not assessed as part of the arterial baseline.
Known limitations: Different study scope, contracts, and provenance.
Recommendation: Integrate through a documented mask contract only if justified by a prespecified analysis.

### Lymph-node segmentation

Capability: Lymph-node segmentation.
Current status: Missing.
Implementation location: No integrated implementation found.
Entry point: None.
Input: CTA neck.
Output: Expected lymph-node candidate masks.
Dependencies: Specialized detection/segmentation model and clinical validation.
Evidence that it is used: No output found.
Tests: None.
Known limitations: Very difficult on CTA without dedicated validation; false positives could dominate residual tissue.
Recommendation: Keep residual/unknown class; defer model development.

### Surface-contact analysis

Capability: Surface-contact analysis.
Current status: Deterministic distance-based contact analysis is implemented and synthetic-tested; no real named-carotid contact output exists.
Implementation location: `perivascular/contact.py`; `carotid_analysis.py`.
Entry point: `analyze_carotid_case` after native-output validation.
Input: Target lumen/shell, vessel coordinates, and optional geometry-matched tissue masks.
Output: Contact fractions at configured 0.5, 1, and 2 mm plus longitudinal/sample profiles.
Dependencies: SciPy physical-distance operations, named lumen, and supplied coarse tissue masks.
Evidence that it is used: Synthetic tests exercise contact logic; real `sub-570` failed before named analysis and supplied no contact overlays.
Tests: Analytic/synthetic sector-contact and carotid-core tests.
Known limitations: Missing tissue masks produce unavailable/`None`, not zero; anatomical and surface-distance accuracy remain unvalidated.
Recommendation: Validate against expert contact annotations and retain mask availability in every output.

### Radial tissue analysis

Capability: Radial tissue analysis.
Current status: Whole-tree shells are real-case verified; named-vessel radial composition is implemented/synthetic-tested but blocked on real native labels.
Implementation location: `cta_common/shells.py`, adaptive core, `perivascular/shells.py`, `composition.py`, `radial_profiles.py`, `carotid_analysis.py`.
Entry point: Legacy/adaptive stages or fail-closed carotid run.
Input: Named lumen/diameter, CTA HU, exclusion masks, and optional tissue masks.
Output: Raw/filtered fixed bands, capped adaptive envelope, tissue labels, radial gradients and profiles.
Dependencies: Physical EDT, trusted labelled branches, vessel-centred coordinates, optional coarse anatomy.
Evidence that it is used: Five broad-tree adaptive outputs exist; additive real 570 analysis generated none after preflight failure.
Tests: Shared/study execution plus synthetic shell/composition/carotid-core tests.
Known limitations: Additive results lack a complete native real-case validation; missing optional anatomy stays explicit.
Recommendation: Execute only after native label validation and compare raw/filtered bands plus uncertainty in Slicer.

### Circumferential sector analysis

Capability: Circumferential sector analysis.
Current status: Legacy axial sectors and additive parallel-transport sectors exist; additive path is synthetic-tested but not real-case executed.
Implementation location: `fat_omics.py`; `perivascular/coordinates.py`; `perivascular/sectors.py`; `carotid_analysis.py`.
Entry point: Legacy fat analysis or fail-closed carotid analysis.
Input: Vessel-centred frame, shell voxels, path/radial coordinates, and tissue classes.
Output: Primary 16-sector assignments and local/vessel summaries; configuration also declares 8/32 sensitivity counts.
Dependencies: Validated named centreline/frame and NumPy/SciPy.
Evidence that it is used: Legacy cohort artifacts exist; no real named-carotid sector mask was generated.
Tests: Synthetic coordinate/sector/contact tests include frame behavior.
Known limitations: Absolute angle initialization and frame transport require real-case/variant validation, especially at bifurcations.
Recommendation: Overlay sector orientation and test rotational/reversal invariance before biological inference.

### Longitudinal vessel analysis

Capability: Longitudinal vessel analysis.
Current status: Legacy axial summaries exist; named branch arc-length profiles and sample outputs are implemented/synthetic-tested but not real-case executed.
Implementation location: Legacy feature writers; `perivascular/coordinates.py`, `radial_profiles.py`, `carotid_analysis.py`, `output_schema.py`.
Entry point: Main legacy analysis or fail-closed carotid run.
Input: Labelled native branch, 1-mm resampled vessel-centred frame, local shells/tissues/contacts.
Output: Sample-point, longitudinal-bin, radial, sector, contact, and configured-interval summaries.
Dependencies: Stable native branch ordering, geometry, pandas, optional Parquet engine.
Evidence that it is used: Legacy aggregate rows exist; additive outputs are fixture-tested only because real 570 stopped at preflight.
Tests: Synthetic coordinate/profile/output-schema/carotid-core tests.
Known limitations: No bulb is inferred; an interval must be explicitly supplied with evidence. Direction/origin repeatability lacks real validation.
Recommendation: Validate branch direction and landmarks, preserve all local profiles, and avoid whole-vessel-only averages.

### Quality control

Capability: Quality control.
Current status: Basic/Slicer QC and a headless carotid integrity/completeness QC are verified; named-artery visual validation remains blocked.
Implementation location: `segmentation_qc.py`, `scripts/qc_slicer.py`, adaptive QC, `carotid_qc.py`.
Entry point: Existing QC commands or `python -m aorta_cta_radiomics.carotid_qc`.
Input: CTA, binary segmentation, FLOWCAT manifest, and optional labelled artery/shell/tissue/calcium overlays.
Output: Geometry/completeness/overlay metrics, flags, JSON, headless three-plane montage, and manual-review state.
Dependencies: nibabel, NumPy, SciPy, Matplotlib Agg; 3D Slicer for definitive interactive review.
Evidence that it is used: Real `sub-570` binary-only QC JSON/montage exists and honestly reports four missing native outputs/no named results.
Tests: Synthetic geometry, completeness, overlay, PHI-minimization, and CLI tests plus real artifact generation.
Known limitations: Automated montage/QC is not an anatomical pass; named continuity/laterality and clinical plausibility require a complete native case and reviewer.
Recommendation: Regenerate QC with final code when it changes, then perform structured Slicer review after native labelled overlays exist.

### Batch processing

Capability: Batch processing.
Current status: Verified for sequential/staged manifests.
Implementation location: Aortic CLI `run-batch`, `run_manifest_staged.py`, adaptive `--all` runner.
Entry point: `aorta-cta run-batch`, `aorta-cta-staged`, study script.
Input: Manifest of de-identified CTA/mask paths and stage options.
Output: Per-case directories, logs, status rows, aggregate CSVs.
Dependencies: Python environments and stage-specific tools.
Evidence that it is used: 20-case, 51-case, 77-case, and five-case artifacts exist; 407 eligible cases were dry-run preflighted.
Tests: Representative dry run and artifact audit; no transactional resume integration test.
Known limitations: Main batch is sequential/in-memory; stage status overwrites and sentinels are incomplete.
Recommendation: Use immutable run IDs, complete output contracts, atomic writes, and resumable per-stage manifests.

### Patient-level aggregation

Capability: Patient-level aggregation.
Current status: Legacy wide output is real-case verified; additive typed case summary/manifest is implemented and synthetic-tested.
Implementation location: Main aggregation plus `output_schema.py`, `carotid_analysis.py`, `carotid_cli.py`.
Entry point: Legacy aggregation or successful fail-closed carotid output writing.
Input: Per-case features/provenance and named-vessel/sample records.
Output: Legacy wide row or additive `case_summary.json` plus hashed output manifest.
Dependencies: pandas and successful upstream analysis.
Evidence that it is used: Legacy wide output has 683 rows; real 570 generated inventory/failure only and no additive case result.
Tests: Output-schema/core tests and legacy artifacts.
Known limitations: No cohort-level additive runner or successful real carotid case; legacy snapshots remain inconsistent.
Recommendation: Add immutable run IDs and batch only after one complete native case passes.

### Vessel-level aggregation

Capability: Vessel-level aggregation.
Current status: Typed named-vessel schema/writer is implemented and synthetic-tested; no real carotid rows were generated.
Implementation location: `output_schema.py`; `carotid_analysis.py`; `carotid_cli.py`.
Entry point: `write_output_bundle` through fail-closed carotid run.
Input: Validated per-branch geometry, shells, tissue/contact profiles and QC.
Output: `vessel_level_features.csv`, one record per required named branch.
Dependencies: Complete native labels and successful carotid analysis.
Evidence that it is used: Fixture tests write/validate vessel rows; real 570 stopped before output writing.
Tests: Output-schema and carotid-core/CLI tests.
Known limitations: No real named-vessel distribution/validity; calcium burden remains null in the current carotid core.
Recommendation: Retain stable case/vessel/branch keys and validate real row completeness after native execution.

### Sample-point-level aggregation

Capability: Sample-point-level aggregation.
Current status: Legacy technical rows exist; typed anatomically labelled sample schema/writer is implemented and synthetic-tested, with no real carotid table yet.
Implementation location: Legacy feature writers; `carotid_analysis.py`; `output_schema.py`.
Entry point: `write_output_bundle` after successful carotid analysis.
Input: Vessel-centred samples with RAS frame, branch label/confidence, radius/geometry, local composition/contact/QC.
Output: Parquet when available, otherwise explicitly recorded CSV fallback.
Dependencies: pandas, optional Parquet engine, complete native carotid inputs.
Evidence that it is used: Legacy rows and additive fixture outputs exist; real 570 produced no sample rows by design.
Tests: Output-schema and synthetic carotid-core tests.
Known limitations: No real native sample distribution; radiomics/calcium sample fields are not populated by the current core.
Recommendation: Validate sample counts/frames/units on a complete native case and retain explicit storage format.

### Failure handling

Capability: Failure handling.
Current status: Legacy handling remains partial; additive carotid manifest/run is fail-closed and verified on real `sub-570`.
Implementation location: Staged runner, adaptive runner, `carotid_cli.py`, adapter validation, headless QC.
Entry point: Stage runners or `python -m aorta_cta_radiomics.carotid_cli manifest|run`.
Input: Stage command/sentinels, or explicit CTA, case directory, fixed FLOWCAT repo, config and trust decision.
Output: Return status, hashed inventory, structured `failure.json`, or complete atomic output bundle.
Dependencies: Filesystem/subprocess; trusted native artifacts for analysis.
Evidence that it is used: Real 570 run returned status 2 and recorded preflight `FileNotFoundError` naming all four missing native artifacts; no fallback was substituted.
Tests: Missing/invalid/unsafe artifact tests plus real fail-closed execution; no cohort-scale fault-injection/resume test.
Known limitations: Legacy status still overwrites/uses incomplete sentinels, and the older native VMTK traceback remains unavailable.
Recommendation: Reuse additive required-output/failure contracts in batch orchestration and retain exact stderr/command provenance.

### Logging

Capability: Logging.
Current status: Verified but fragmented.
Implementation location: Staged runner logs, status CSV, per-script stdout/stderr, study JSON.
Entry point: Each runner/stage.
Input: Commands, warnings, subprocess output.
Output: Text logs, CSV/JSON status.
Dependencies: Python logging/subprocess/filesystem.
Evidence that it is used: Cohort log paths and dry-run `DRY RUN` records exist.
Tests: Artifact inspection; no logging-schema test.
Known limitations: No unified structured event schema/run ID and some exact commands/errors are not retained.
Recommendation: Emit append-only JSONL events with case/stage/run IDs and redacted commands.

### Provenance

Capability: Provenance.
Current status: Partial overall, with stronger additive FLOWCAT/output provenance implemented.
Implementation location: Legacy status/version rows, adaptive JSON, `flowcat_adapter/provenance.py`, `output_schema.py`, `carotid_cli.py`.
Entry point: Manifest/carotid run and output writers.
Input: Repository, controlled model files, parameters, config, inputs, dependencies and processing state.
Output: FLOWCAT URL/branch/commit, model paths/SHA-256, dependency versions, artifact hashes, config hash, runtime and case/vessel/sample manifests.
Dependencies: Git/package metadata and SHA-256 utilities.
Evidence that it is used: Real 570 manifest records fixed FLOWCAT identity, segmentation SHA-256 and parameters; finalized code also inventories controlled model payloads.
Tests: Provenance/manifest/output-schema contract tests; no complete real successful carotid output bundle.
Known limitations: Preimplementation and earlier 570 artifacts may lack later provenance fields; peak RAM/GPU and exact historical shell commands remain unavailable.
Recommendation: Regenerate representative artifacts with final code and require immutable manifests for every future run.

### Unit tests

Capability: Unit tests.
Current status: Verified.
Implementation location: `aorta_cta_radiomics/tests`, `cta_common/tests`, study `tests`.
Entry point: `python -m pytest`.
Input: Unit and small synthetic fixtures.
Output: Pass/fail test report.
Dependencies: pytest and package environments.
Evidence that it is used: Preimplementation snapshot: 58 + 38 + 32 = 128 passed; the additive suite has separate current focused/full-suite verification recorded in the baseline report.
Tests: Unit, analytic geometry, schema, CLI, fail-closed, QC, and synthetic integration fixtures.
Known limitations: Passing synthetic tests is not native FLOWCAT generation, expert anatomy validation, or a successful real carotid analysis.
Recommendation: Keep historical/current counts separate and add complete-native and Slicer-adjudicated regression cases.

### Integration tests

Capability: Integration tests.
Current status: Additive synthetic/contract integration and a real fail-closed preflight/QC are verified; no successful full native real-case integration exists.
Implementation location: Adapter/carotid/output/QC tests, `audit_carotid_pipeline.py`, and real 570 milestone artifacts.
Entry point: pytest, read-only audit, carotid manifest/run, and headless QC.
Input: Synthetic complete artifacts or de-identified real CTA/binary segmentation/incomplete native directory.
Output: Validated schemas/outputs for fixtures, or explicit real failure and pending-review QC.
Dependencies: Complete models/native outputs for a successful real path.
Evidence that it is used: Real 570 inventory/failure/QC proves fail-closed integration; synthetic tests reach artery volume and three-level output writing.
Tests: No successful real native FLOWCAT-to-named-carotid run, full DICOM path, PyRadiomics case, or cohort resume test.
Known limitations: Fixtures cannot reproduce native serialization/model/topology failure modes.
Recommendation: Add one controlled complete native 570 case as the next integration gate, without weakening missing-output checks.

### Synthetic geometry tests

Capability: Synthetic geometry tests.
Current status: Partial in local study/unit suites; insufficient for full geometry contract.
Implementation location: Package/shared/study tests.
Entry point: pytest.
Input: Programmatic masks/paths.
Output: Expected shell/geometry/topology measures.
Dependencies: NumPy, SciPy, SimpleITK, pytest.
Evidence that it is used: Some local geometry/shell tests contribute to the 128-test baseline.
Tests: No complete cylinder, taper, bifurcation, helix, RAS/LPS, contact, and FOV matrix.
Known limitations: Native VMTK/FLOWCAT feature schemas are not covered.
Recommendation: Add analytic cylinders/tapers/bifurcations/helices and coordinate round trips with tolerances.

## 1C. FLOWCAT integration audit

### Source and current integration mechanism

The intended and audited source is `https://github.com/FLOWCAT-CV/arterial`, branch
`models/2026-03`, fixed commit
`1991ca5b238d7c868c9c3a0a390057bc89aee559`. The local clean checkout is
`external/arterial_flowcat`. Its authoritative licence file is CC BY-NC 4.0.

- Clones FLOWCAT: **No automatically.** The repository assumes a manually cloned
  local checkout.
- Vendors FLOWCAT: **No.** The checkout is Git-ignored and not committed or a
  submodule.
- Installs FLOWCAT as a Python package: **No.** Study code prepends the checkout to
  `sys.path`.
- Imports FLOWCAT modules: **Yes.** The segmentation runner imports
  `arterial.segmentation.inference.perform_single_inference_nnunet` directly.
- Calls FLOWCAT through subprocesses: **Not for inference.** The local runner may be
  launched as a process, but inference is in-process. Model preparation invokes an
  upstream rebuild helper as a subprocess.
- Calls FLOWCAT through shell scripts: **No FLOWCAT shell integration found.**
- Modifies FLOWCAT source code: **No.** The audited runner requires a clean checkout.
- Consumes already-generated FLOWCAT outputs: **Yes, but only the binary
  segmentation** in the five-case adaptive pilot.
- Duplicates FLOWCAT functions independently: **Yes.** Local approximate/skeleton
  centrelines, branch paths, radii, and geometry proxies overlap native functions.
- Uses a fixed FLOWCAT commit: **Yes.** The runner checks the commit.
- Uses a floating branch: **No at execution.** The named branch documents intent,
  while the commit is authoritative.
- Stores FLOWCAT model weights locally: **Yes.** Git-chunked files are reconstructed
  locally and verified by size/SHA.
- Downloads FLOWCAT weights automatically: **No.**

The native post-segmentation chain is not currently functional on the audited Mac.
The `totalseg-mac` environment lacks VMTK, VTK, `torch_geometric`, and
`torch_cluster`. A representative `sub-570` native attempt found 19 endpoints but
failed for both original and cleaned surfaces because valid surface normals could
not be computed after the prescribed mesh-bottom retries. The persisted reason is
`flowcat_vmtk_automatic_centerline_failed_sub570_surface_normals`; the original
traceback was not persisted, so a more specific exception cannot be verified.

The newly added typed adapter supports non-executing discovery/manifests, binary
NIfTI validation, explicit-trust object-NPY/NetworkX loaders, 14-class graph-edge
label mapping to typed centreline branches, an external `cta_path`, coordinate
conversions, optional VTK, and provenance/model-payload hashing. The additive
`artery_volume` and `carotid_analysis` modules map already labelled branches back
into an uncertainty-preserving NIfTI volume and derive deterministic named-vessel
features. `carotid_cli` exposes manifest/run commands and `carotid_qc` exposes
headless QC. These modules do not run FLOWCAT or create a missing native branch
model/graph. Real `sub-570` therefore failed closed at preflight, while synthetic
fixtures exercise the downstream path.

The output paths below are relative to `<case_dir>`. “Expected” follows upstream
code/tests where possible. FLOWCAT README examples sometimes place
`extracranial_vessels_segmentation.nii.gz` at case root, while the audited study
runner writes `extracranial_vessels/segmentation.nii.gz`; this naming difference
must not be silently normalized.

### Extracranial arterial segmentation

Output name: Extracranial arterial segmentation.
Expected file path: `extracranial_vessels/segmentation.nii.gz` in the integrated runner; upstream README also documents a root-level named file.
Actual file path: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-<ID>_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz` for five cases.
Format: Binary NIfTI (`.nii.gz`).
Coordinate convention: Stored NIfTI affine; source pilot orientations include LAS and LPS and are not canonicalized.
Units: Millimetres for geometry; binary voxel labels 0/1.
Important fields or arrays: Image shape, affine/origin/direction/spacing, foreground mask.
Currently generated: Yes, five `fast-lowres` cases.
Currently used downstream: Yes, by adaptive burden and skeleton/EDT fallback.
Successfully tested: Operational generation succeeded; formal visual/anatomic validation is pending.
Known problems: Broad low-resolution tree includes aorta, lacks named branches, and may be FOV-truncated.
Recommendation: Retain raw mask/affine/hash and do not treat it as labelled or clinically validated.

### Intracranial arterial segmentation

Output name: Intracranial arterial segmentation.
Expected file path: Upstream mode-specific segmentation under `<case>/intracranial_vessels/` or upstream root-level naming, depending on entry point.
Actual file path: None in the five audited extracranial cases.
Format: Expected binary NIfTI.
Coordinate convention: Expected source NIfTI affine; upstream may slice the head and does not provide repository-wide canonicalization.
Units: Millimetres and binary voxel labels.
Important fields or arrays: Affine, crop/slice transform, probability/segmentation array.
Currently generated: No in the audited study.
Currently used downstream: No.
Successfully tested: No representative integrated run.
Known problems: Territory, crop transform, and continuity with extracranial results are unverified.
Recommendation: Define a separate output contract and registration/continuity test before use.

### extracranial_vessels_segmentation.nii.gz

Output name: `extracranial_vessels_segmentation.nii.gz`.
Expected file path: `<case>/extracranial_vessels_segmentation.nii.gz` according to the upstream README example.
Actual file path: Not generated under this exact name; integrated output is `extracranial_vessels/segmentation.nii.gz`.
Format: Expected binary NIfTI.
Coordinate convention: NIfTI affine.
Units: Millimetres and binary labels.
Important fields or arrays: Image geometry and mask voxels.
Currently generated: No under the exact documented name.
Currently used downstream: No under that name.
Successfully tested: No; only the nested naming contract was exercised.
Known problems: Documentation/code naming mismatch can cause false missing-output or duplicate-file logic.
Recommendation: Adapter discovery should recognize both explicitly and record the resolved source path.

### Surface representations

Output name: Surface representations.
Expected file path: `extracranial_vessels/segmentation.vtk`, `.stl`, and per-segment surface directories.
Actual file path: None in the five generated case directories.
Format: VTK PolyData and STL.
Coordinate convention: Expected VTK physical coordinates; exact relation to NIfTI RAS must be validated casewise.
Units: Millimetres.
Important fields or arrays: Points, polygons, normals, connected components.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No complete representative chain.
Known problems: `sub-570` surface-normal preparation prevented native centreline extraction.
Recommendation: Add mesh integrity/normals checks and physical landmark registration.

### VTK vessel surfaces

Output name: VTK vessel surfaces.
Expected file path: `extracranial_vessels/segmentation.vtk` and `segmentations/segmentation_<idx>.vtk`.
Actual file path: None in audited pilot outputs.
Format: Legacy/XML-compatible VTK PolyData as produced by upstream code.
Coordinate convention: VTK physical coordinates, expected RAS-like but not yet verified against NIfTI landmarks.
Units: Millimetres.
Important fields or arrays: Point coordinates, cells, surface normals.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: Optional VTK dependency absent in current analysis environment; normals failure on 570.
Recommendation: Generate on supported Linux and validate bounds/corners before consuming.

### STL vessel surfaces

Output name: STL vessel surfaces.
Expected file path: `extracranial_vessels/segmentation.stl`.
Actual file path: None in audited pilot outputs.
Format: STL triangular surface.
Coordinate convention: Physical surface coordinates without embedded medical-image frame metadata.
Units: Assumed millimetres; STL itself does not encode units.
Important fields or arrays: Vertices and triangle faces.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: No affine/frame metadata and no current artifact.
Recommendation: Use only with a sidecar containing source affine, units, transform, and hash.

### VMTK centrelines

Output name: VMTK centrelines.
Expected file path: `extracranial_vessels/centerlines/centerlines_<idx>.vtk`.
Actual file path: None; only local skeleton/EDT CSV fallback outputs exist.
Format: VTK PolyData centreline models.
Coordinate convention: VTK physical points; downstream FLOWCAT conversion creates an LPI-corner-relative frame.
Units: Millimetres; radius expected in millimetres.
Important fields or arrays: Centreline points/cells and `MaximumInscribedSphereRadius`.
Currently generated: No.
Currently used downstream: No; non-equivalent fallback is used.
Successfully tested: No; representative 570 native attempt failed.
Known problems: VMTK unavailable on Mac environment and surface normals failed.
Recommendation: Reproduce on Linux with pinned VTK/VMTK and compare to analytic/visual references.

### Centreline branch segmentation

Output name: Centreline branch segmentation.
Expected file path: `extracranial_vessels/branch_models/branch_model_<idx>.vtk` plus merged model.
Actual file path: None in audited pilot outputs.
Format: VTK PolyData.
Coordinate convention: Physical VTK coordinates; case-frame validation required.
Units: Millimetres.
Important fields or arrays: Branch cells, point arrays, radius, cell IDs.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: Depends on failed native centreline chain.
Recommendation: Validate branch count/topology and stable `cell_id` mapping before graph use.

### branch_model.vtk

Output name: `branch_model.vtk`.
Expected file path: `extracranial_vessels/branch_model.vtk`.
Actual file path: None in five generated case directories.
Format: VTK PolyData merged branch model.
Coordinate convention: VTK physical coordinates; presumed RAS-like but unverified.
Units: Millimetres.
Important fields or arrays: Cells keyed to IDs referenced by graph edges.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No representative artifact.
Known problems: Required by feature/graph mapping but blocked by VMTK preprocessing.
Recommendation: Make it a required adapter artifact and validate all graph `cell_id` references.

### centerline_segments_array.npy

Output name: `centerline_segments_array.npy`.
Expected file path: `extracranial_vessels/centerline_segments_array.npy`.
Actual file path: None in the five audited outputs.
Format: NumPy object array of shape `(n_segments, 2)`.
Coordinate convention: Segment coordinates are FLOWCAT LPI-corner-relative millimetres, not simply absolute NIfTI RAS.
Units: Coordinates and radii in millimetres.
Important fields or arrays: Column 0 coordinate arrays; column 1 `MaximumInscribedSphereRadius` arrays.
Currently generated: No.
Currently used downstream: No.
Successfully tested: Upstream schema inspected; no current case artifact.
Known problems: Object arrays require pickle-enabled NumPy loading and therefore explicit trust.
Recommendation: Load only with `trusted_source=True`, type-check every segment, and convert with recorded image geometry.

### Anatomical landmarks

Output name: Anatomical landmarks.
Expected file path: `extracranial_vessels/landmarks/landmarks.json` and `landmarks_slicer.json`.
Actual file path: None in audited pilot outputs.
Format: JSON.
Coordinate convention: Upstream/Slicer variants may use different physical conventions; must be read from explicit schema.
Units: Millimetres for physical points.
Important fields or arrays: Named landmark coordinates and source/reference frame.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No representative output.
Known problems: Native centreline/labelling prerequisites absent and frame contract unverified.
Recommendation: Require coordinate-frame metadata and compare landmarks in CTA and Slicer.

### segments_graph_pred.pickle

Output name: `segments_graph_pred.pickle`.
Expected file path: `extracranial_vessels/segments_graph_pred.pickle`.
Actual file path: None in audited pilot outputs.
Format: Python pickle containing a NetworkX graph.
Coordinate convention: Geometry references branch/centreline coordinates; explicit frame validation is required.
Units: Geometry/features are generally millimetre-based but each feature must be schema-checked.
Important fields or arrays: Edge `cell_id`, `vessel_type`, `vessel_type_name`, and feature dictionaries.
Currently generated: No.
Currently used downstream: No.
Successfully tested: Upstream schema/code inspected; no current case artifact.
Known problems: Pickle permits code execution; schema/version coupling and graph-to-cell mismatch are possible.
Recommendation: Load only trusted locally generated pinned outputs and immediately validate/convert to typed data.

### local_graph.pickle

Output name: `local_graph.pickle`.
Expected file path: `extracranial_vessels/local_graph.pickle`.
Actual file path: None in audited pilot outputs.
Format: Python pickle containing a dense NetworkX graph.
Coordinate convention: Node positions follow FLOWCAT centreline frame; conversion must be explicit.
Units: Positions/radius/length/diameter in millimetres; curvature/torsion units require feature-level confirmation.
Important fields or arrays: Node/edge positions, radius, vessel type, and local/segment/global feature dictionaries.
Currently generated: No.
Currently used downstream: No.
Successfully tested: Upstream implementation inspected; no representative artifact.
Known problems: Unsafe unpickling, dependency/schema version coupling, and absent native prerequisites.
Recommendation: Require trust opt-in, schema validation, and JSON/Parquet typed export for long-term use.

### Vessel-type predictions

Output name: Vessel-type predictions.
Expected file path: `extracranial_vessels/segments_graph_pred.pickle` and plot.
Actual file path: None in audited pilot outputs.
Format: NetworkX graph pickle with edge attributes; PNG visualization may accompany it.
Coordinate convention: Identity attaches to branch `cell_id`; geometry follows FLOWCAT centreline frame.
Units: Categorical class IDs/names; confidence scale must be verified from model output.
Important fields or arrays: `cell_id`, `vessel_type`, `vessel_type_name`, features, confidence if present.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No representative output; the new adapter defines a 14-class mapping but does not create predictions.
Known problems: Model/PyG chain unavailable and volume label propagation is not baseline-functional.
Recommendation: Validate class taxonomy, confidence, side, graph-cell linkage, and unknown class on expert cases.

### Local radius

Output name: Local radius.
Expected file path: `centerline_segments_array.npy`, centreline VTK radius array, and/or `local_graph.pickle`.
Actual file path: No native FLOWCAT file; technical EDT radius exists in local study CSVs.
Format: Native NumPy object array/VTK/graph; fallback CSV.
Coordinate convention: Native FLOWCAT centreline frame; fallback NIfTI-derived physical geometry.
Units: Millimetres.
Important fields or arrays: `MaximumInscribedSphereRadius` or node radius.
Currently generated: Only non-equivalent fallback.
Currently used downstream: Fallback radius is used to size adaptive shells.
Successfully tested: Fallback executed on five cases; native radius did not.
Known problems: EDT skeleton radius is not the native maximum-inscribed-sphere result.
Recommendation: Preserve method/source and validate paired values when native outputs become available.

### Diameter estimates

Output name: Diameter estimates.
Expected file path: Feature fields in `local_graph.pickle` and derived centreline/segment outputs.
Actual file path: No native graph; local adaptive summaries/CSV contain proxy diameter.
Format: Native graph pickle; fallback CSV/JSON.
Coordinate convention: Attached to local centreline points/branches.
Units: Millimetres.
Important fields or arrays: Local diameter, source radius, branch/sample ID.
Currently generated: Proxy only.
Currently used downstream: Yes, proxy controls capped shell thickness.
Successfully tested: Five fallback cases; no native comparison.
Known problems: EDT/axial diameter is not an orthogonal lumen diameter and is not a wall-thickness estimator.
Recommendation: Compare orthogonal/natively derived diameter and retain adaptive floor/cap.

### Curvature

Output name: Curvature.
Expected file path: Feature entries in `local_graph.pickle` or upstream feature exports.
Actual file path: No native FLOWCAT feature file; local approximate aortic curvature exists separately.
Format: Native graph pickle; local CSV.
Coordinate convention: Feature attached to ordered centreline samples.
Units: Expected inverse millimetres; verify upstream field semantics before use.
Important fields or arrays: Local curvature, point/branch ID, sampling distance.
Currently generated: Not by current FLOWCAT run.
Currently used downstream: No native value.
Successfully tested: Local proxy only; native output untested.
Known problems: Sampling and smoothing sensitivity; units/schema not validated on artifact.
Recommendation: Verify schema and test circle radii across sampling intervals.

### Torsion

Output name: Torsion.
Expected file path: Feature entries in `local_graph.pickle` or upstream feature exports.
Actual file path: None in current FLOWCAT/fallback pilot outputs.
Format: Expected graph pickle/derived feature table.
Coordinate convention: Attached to ordered 3D centreline samples.
Units: Expected inverse millimetres; unable to verify from a generated artifact.
Important fields or arrays: Local torsion, point/branch ID, smoothing/sampling parameters.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: Native pipeline blocked; highly noise-sensitive derivative.
Recommendation: Require explicit units and helix/straight-line synthetic tests.

### Vessel length

Output name: Vessel length.
Expected file path: Segment/global features in `local_graph.pickle` and centreline-derived outputs.
Actual file path: No native graph; local technical path lengths/proxies exist.
Format: Native graph pickle; local CSV/JSON.
Coordinate convention: Arc length in the corresponding centreline physical frame.
Units: Millimetres.
Important fields or arrays: Path/branch ID, endpoints, arc length, sampling method.
Currently generated: Proxy only.
Currently used downstream: Local paths support technical pilot summaries, not named-vessel results.
Successfully tested: Local execution; no native/landmark validation.
Known problems: Spur splitting and missing anatomical endpoints change length.
Recommendation: Define endpoints/territory and compare against native branch model.

### Tortuosity

Output name: Tortuosity.
Expected file path: Segment/global feature fields in `local_graph.pickle`.
Actual file path: No native graph; local aortic relative-length proxies exist elsewhere.
Format: Native graph pickle; local CSV.
Coordinate convention: Centreline path plus endpoint chord in one physical frame.
Units: Dimensionless for path/chord ratio; alternate upstream definitions must be schema-checked.
Important fields or arrays: Path length, chord length, formula, endpoints, branch label.
Currently generated: Not by current FLOWCAT run.
Currently used downstream: No native value.
Successfully tested: Local proxy only.
Known problems: Definitions and branch endpoints are not interchangeable.
Recommendation: Store numerator/denominator/formula and validate on analytic curves.

### Proximal geometry

Output name: Proximal geometry.
Expected file path: Proximal branch/sample features in graph/centreline outputs and landmarks.
Actual file path: None as an anatomically defined FLOWCAT output.
Format: Expected graph/JSON/typed derived table.
Coordinate convention: Vessel-specific arc length anchored to proximal landmark.
Units: Millimetres and derived geometry units.
Important fields or arrays: Vessel identity, proximal landmark, sample range, radius/curvature/length.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: No named arteries or validated landmarks; broad FOV is truncated in all pilot cases.
Recommendation: Define anatomy-specific proximal boundaries and explicit FOV completeness flags.

### Distal geometry

Output name: Distal geometry.
Expected file path: Distal branch/sample features in graph/centreline outputs and landmarks.
Actual file path: None as an anatomically defined FLOWCAT output.
Format: Expected graph/JSON/typed derived table.
Coordinate convention: Vessel-specific arc length anchored to distal landmark.
Units: Millimetres and derived geometry units.
Important fields or arrays: Vessel identity, distal endpoint/landmark, sample range and features.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: Branch labels/endpoints and intracranial transition are unverified.
Recommendation: Preserve distal truncation/uncertainty and validate landmark reproducibility.

### Aortic arch classification

Output name: Aortic arch classification.
Expected file path: No single verified path; expected in native anatomical/feature results if invoked.
Actual file path: None in audited pilot outputs.
Format: Expected categorical JSON/graph/feature output.
Coordinate convention: Based on labelled arch/branch geometry.
Units: Categorical class with supporting geometric measurements.
Important fields or arrays: Arch class, confidence, landmarks, branch take-off geometry.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No.
Known problems: No native branch labels or stable arch landmarks.
Recommendation: Do not infer from broad binary mask; specify method and expert reference first.

### Branch-anatomy classification

Output name: Branch-anatomy classification.
Expected file path: Predicted vessel graph and related anatomical classification output.
Actual file path: None in current cases.
Format: NetworkX graph pickle and/or typed derived table.
Coordinate convention: Identity mapped by branch `cell_id` to centreline/surface.
Units: Categorical labels/confidence.
Important fields or arrays: `cell_id`, vessel type ID/name, side, confidence, topology.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No representative case.
Known problems: Native labelling dependencies absent; fallback paths have no anatomy.
Recommendation: Validate against expert-labelled branch graphs and preserve unknown/variant anatomy.

### Access-prediction outputs

Output name: Access-prediction outputs.
Expected file path: `extracranial_vessels/access_prediction/{access}_{side}/access_prediction.json` with optional attention maps and preprocessing artifacts.
Actual file path: None in audited pilot outputs.
Format: JSON plus optional pickle, PNG, VTK, and point JSON.
Coordinate convention: Graph/VTK outputs follow native FLOWCAT centreline geometry.
Units: Prediction scores/categories; attention weights; physical coordinates in millimetres where present.
Important fields or arrays: Prediction, access route/side, attention weights, graph/point mapping.
Currently generated: No.
Currently used downstream: No.
Successfully tested: No in this repository baseline.
Known problems: Depends on native labelled/featurized graphs and adds unsafe pickle/model/version concerns.
Recommendation: Exclude from the first perivascular milestone and validate independently if later needed.

## 1D. Gap analysis

### Scientific gaps

- Missing biological measurements: no validated wall thickness, plaque identity,
  inflammation marker, or histologic/clinical ground truth. Calcium and PVAT are
  CTA-derived candidates, not diagnoses.
- Missing vessel territories: named CCA, bulb, ICA, ECA, subclavian, vertebral,
  and intracranial territories are not available in the active five-case output.
- Missing tissue compartments: targeted neck muscles, jugular/plexus veins,
  transverse foramina, lymph nodes, fascia, nerves, and reliable residual classes.
- Missing longitudinal validation: the additive core preserves branch/sample and
  binned profiles, but lacks real landmark/direction repeatability and longitudinal
  repeat imaging.
- Missing circumferential validation: parallel-transport sectors are implemented
  synthetically, but orientation and biological repeatability are unverified on a
  complete native case.
- Missing validated interaction features: the core implements optional
  artery-muscle, artery-vein, artery-bone/thyroid contact and radial composition,
  but no real named-carotid output or expert reference exists.
- Missing validation targets: no expert labelled branch graph, plaque/bone voxel
  truth, PVAT reference, tissue-contact annotation, or clinical repeatability set.

### Engineering gaps

- Broken code paths: native FLOWCAT/VMTK stops at surface normals on `sub-570`;
  exact traceback was not retained. Main calcium/wall stage filenames and
  sentinels can be hard-coded to one threshold/output.
- Missing real adapter validation: typed discovery, safe loading, coordinate
  conversion, graph-to-branch labels, and dual-strategy labelled-volume mapping
  are implemented; the missing native files prevent a complete real integration.
- Configuration/executor: `config/carotid_perivascular.yaml` and fail-closed
  `carotid_cli` now unify the first-milestone rules, but scientific parameter and
  acquisition-sensitivity validation remains missing.
- Partial dependency management: `cta_common` is now declared and the workspace
  install order is documented, but it is not independently distributed;
  VTK/VMTK/PyG versions and architecture-compatible locks are absent.
- Missing batch processing: batch exists, but immutable run manifests,
  transactional resume, parallel resource control, and complete-stage contracts
  are missing.
- Failure handling remains inconsistent in legacy staged runs; the additive
  carotid path now inventories every required file, refuses untrusted artifacts,
  writes structured failures, and never invokes the skeleton fallback.
- Output schemas now exist for additive case/vessel/sample records and hashed
  manifests, but upstream graph/VTK schemas and real-data schema stability remain
  unvalidated.

### Validation gaps

- Missing ground truth for segmentation, labels, calcium classes, perivascular
  tissues, and local geometry.
- Synthetic tests now cover core coordinates, shells, sectors, contact,
  composition, propagation and output contracts, but the full taper/helix,
  orientation/affine, resolution and FOV sensitivity matrix is incomplete.
- Missing inter-reader comparison and adjudication for Slicer QC.
- Missing reproducibility analysis across reruns, environments, CPU/GPU, and
  deterministic/non-deterministic model settings.
- Missing scanner/reconstruction-kernel/contrast sensitivity analysis.
- Missing resolution/partial-volume/shell-floor/cap sensitivity analysis.

### Documentation gaps

- Several experimental/legacy modules and stage relationships are undocumented.
- Some script arguments and the exact historical commands for existing runs were
  not persisted.
- Model checkpoint names/hashes and selection logic are incomplete outside the
  adaptive pilot.
- Native FLOWCAT output schemas, coordinates, units, and required/optional files
  are not documented locally in one authoritative contract.
- Assumptions about anatomy, candidate calcium, residual tissue, shell truncation,
  and use of coarse segmentation priors require explicit result-level wording.

### Quality-control gaps

- No completed formal visual QC for the five adaptive pilot cases.
- Headless carotid QC now covers geometry, required-output completeness, optional
  overlays, canonical-RAS display and explicit pending review; it still cannot
  validate named anatomy, branch continuity/confidence or mesh integrity without
  native products.
- Failure flags do not always distinguish missing, invalid, fallback, or partial
  output states.
- Uncertainty is partly preserved for calcium but not consistently for vessel
  labels, tissue classes, or feature stability.
- The additive carotid manifest/QC checks its five required FLOWCAT artifacts;
  legacy aortic stages still lack a universal output-completeness contract.

### Provenance gaps

- Final additive provenance discovers and hashes controlled FLOWCAT model payloads;
  legacy and older artifacts do not consistently contain them.
- Git commit and dirty state are not consistently captured by the main pipeline.
- Parameters/configuration and reconstructed command line are incomplete for older
  runs.
- Environment lock, dependency versions, CPU/GPU/RAM, and determinism settings are
  not consistently recorded.
- Acquisition metadata needed for scanner/contrast sensitivity are not joined to
  results, while source sidecars may contain identifying fields and need governed
  minimization.

### Licence and dependency risks

- FLOWCAT's `LICENSE` is CC BY-NC 4.0, creating non-commercial-use,
  redistribution, and attribution obligations; a conflicting badge must not be
  used to infer MIT terms.
- Compatibility of FLOWCAT's licence with distribution/deployment has not been
  legally reviewed.
- External model/data licences and required citations are not consolidated.
- The package declares local workspace dependency `cta_common`, but external
  distribution remains unresolved; compiled native dependencies are unpinned.
- Python, VTK, VMTK, PyTorch, PyG, and nnU-Net compatibility is environment- and
  architecture-sensitive.

### Hardware and architecture risks

- ARM64 macOS lacks the validated VTK/VMTK/PyG native chain; successful imports on
  another architecture would not prove equivalent execution.
- CUDA, PyTorch, and PyG wheel combinations may be incompatible; no pinned CUDA
  matrix is documented.
- GPU memory requirements for full native/model workloads were not measured in the
  five CPU segmentation runs.
- Very large 512 x 512 x 1,481–1,783 CTAs create material CPU RAM and I/O pressure;
  only the host's 128 GiB capacity, not per-process peak, is known.
- Existing containers are repository-wide and do not constitute a validated
  aortic/FLOWCAT image; an x86-64-only image could exclude ARM64/DGX Spark use.
- VTK/VMTK and PyG compiled extensions remain the highest-risk dependencies.

### Data-format inconsistencies

- RAS, LPS, LAS, and FLOWCAT LPI-corner-relative millimetres coexist.
- Pilot spacing varies by case and is anisotropic (0.25-mm slice spacing); native
  and analysis resampling policies differ.
- Additive NIfTI/FLOWCAT paths now use a shared typed geometry/coordinate contract,
  but legacy and upstream tools still differ in origins, directions and affine handling.
- SimpleITK `z,y,x` arrays and nibabel/NumPy conventions can be confused.
- Labels range from binary 0/1 to coarse TotalSegmentator IDs, technical path IDs,
  and expected FLOWCAT vessel classes.
- Root-level `extracranial_vessels_segmentation.nii.gz` and nested
  `extracranial_vessels/segmentation.nii.gz` naming conflict.
- Millimetres, millilitres, voxel counts, dimensionless ratios, and derivative
  units are not consistently encoded in field names/schemas.

## 1E. Duplication analysis

No duplicate is deleted in this audit. “Appears more reliable” refers to method
design and current operational evidence separately; without ground truth, it is
not an accuracy ranking.

### Arterial segmentation duplication

Duplicated capability: Arterial segmentation.
Existing local implementation: TotalSegmentator/merged-mask and legacy vessel-mask scripts.
FLOWCAT implementation: nnU-Net extracranial/intracranial segmentation.
Differences: Local broad/coarse label unions differ from FLOWCAT's task-specific binary vessel inference.
Which is currently used: FLOWCAT binary output in the five adaptive pilots; other masks in separate workflows.
Which appears more reliable: FLOWCAT is more task-specific and operationally generated, but accuracy is pending formal QC/ground truth.
Recommended authoritative implementation: Pinned FLOWCAT segmentation after validation.
Recommended fallback: Validated existing mask with explicit source/model provenance.
Migration risk: Label extent, resolution, FOV, and naming differences.
Validation required: Expert vessel masks, branch completeness, and cross-model comparison.

### Centreline extraction duplication

Duplicated capability: Centreline extraction.
Existing local implementation: Axial centre-of-mass and 3D Lee-skeleton/EDT technical paths.
FLOWCAT implementation: Surface preprocessing and VMTK centrelines.
Differences: Local methods are operational proxies; FLOWCAT targets surface-based physical centrelines/radius.
Which is currently used: Local fallback.
Which appears more reliable: Native FLOWCAT is methodologically preferable, but currently nonfunctional on representative data; local fallback is only operationally verified.
Recommended authoritative implementation: Validated native FLOWCAT/VMTK.
Recommended fallback: Local skeleton with explicit `fallback_non_equivalent` status.
Migration risk: Topology, points, radii, endpoints, and derived-feature discontinuity.
Validation required: Analytic geometry, expert landmarks, paired native/fallback case comparison.

### Centreline resampling duplication

Duplicated capability: Centreline resampling.
Existing local implementation: Approximate point/path interpolation and smoothing utilities.
FLOWCAT implementation: Native centreline post-processing/sampling.
Differences: Sampling frames, branch handling, and step definitions differ.
Which is currently used: Local approximate path.
Which appears more reliable: Unable to rank empirically; FLOWCAT is preferred once native output succeeds.
Recommended authoritative implementation: FLOWCAT coordinates followed by one typed arc-length resampler.
Recommended fallback: Local millimetre-based resampling with source flag.
Migration risk: Changed point counts and all point-level feature rows.
Validation required: Straight/curved analytic paths and spacing invariance.

### Branch splitting duplication

Duplicated capability: Branch splitting.
Existing local implementation: Skeleton degree-node technical path splitting.
FLOWCAT implementation: VMTK centreline branches and branch models.
Differences: Local topology follows voxel spurs; FLOWCAT binds surface branches to graph cell IDs.
Which is currently used: Local technical paths.
Which appears more reliable: FLOWCAT design is anatomy-oriented; current execution remains unverified.
Recommended authoritative implementation: Native branch models/graph.
Recommended fallback: Cleaned technical graph, never labelled anatomical.
Migration risk: Branch counts/IDs are entirely incompatible.
Validation required: Synthetic bifurcations and expert branch graph.

### Vessel labelling duplication

Duplicated capability: Vessel labelling.
Existing local implementation: Coarse TotalSegmentator labels and proposed nearest-centreline assignment.
FLOWCAT implementation: `VesselLabeller` with predicted graph vessel types.
Differences: Coarse volumetric priors and technical Voronoi assignment do not reproduce learned branch identity.
Which is currently used: No named labels in the five-case pilot.
Which appears more reliable: Unable to establish until FLOWCAT predictions are generated and expert-validated.
Recommended authoritative implementation: FLOWCAT graph label plus confidence, mapped through a validated adapter.
Recommended fallback: Coarse priors marked non-authoritative/unknown.
Migration risk: Wrong side/branch could contaminate every artery-specific feature.
Validation required: Expert labelled graph and voxel mapping overlap/continuity.

### Landmark detection duplication

Duplicated capability: Landmark detection.
Existing local implementation: Skeleton endpoints/junctions and axial anatomy proxies.
FLOWCAT implementation: Native landmark detection JSON/Slicer outputs.
Differences: Local points are topological, not necessarily anatomical landmarks.
Which is currently used: Technical endpoints/junctions only.
Which appears more reliable: Native FLOWCAT is intended for anatomy, but no current artifact proves reliability.
Recommended authoritative implementation: Validated FLOWCAT landmarks.
Recommended fallback: Explicit technical endpoints with no anatomical name.
Migration risk: Longitudinal origin and segment boundaries change.
Validation required: Inter-reader landmark distance and coordinate-frame checks.

### Local radius duplication

Duplicated capability: Local radius.
Existing local implementation: Anisotropic EDT sampled on a voxel skeleton.
FLOWCAT implementation: VMTK `MaximumInscribedSphereRadius` in centreline arrays/VTK.
Differences: Skeleton placement and surface construction yield different radii.
Which is currently used: Local EDT radius.
Which appears more reliable: Native VMTK is preferable for a validated surface; current native path failed.
Recommended authoritative implementation: FLOWCAT/VMTK radius after mesh QC.
Recommended fallback: EDT radius with method tag and uncertainty.
Migration risk: Adaptive shells and burden volumes will change.
Validation required: Synthetic cylinders/tapers and paired real vessels.

### Diameter duplication

Duplicated capability: Diameter.
Existing local implementation: Twice EDT radius and axial component geometry proxies.
FLOWCAT implementation: Local/segment diameter features derived from native geometry.
Differences: Neither should be assumed to estimate vessel wall; cross-sectional definitions differ.
Which is currently used: Local proxy with floor/maximum shell caps.
Which appears more reliable: Unable to rank without orthogonal lumen reference; native geometry is preferred for future use.
Recommended authoritative implementation: Validated orthogonal/native lumen diameter.
Recommended fallback: EDT diameter with hard cap and method field.
Migration risk: Systematic shell-width and burden changes.
Validation required: Manual orthogonal diameters across vessel sizes/territories.

### Curvature duplication

Duplicated capability: Curvature.
Existing local implementation: Finite-difference curvature on approximate ordered points.
FLOWCAT implementation: Native centreline feature extraction/local graph.
Differences: Smoothing, sample interval, and centreline geometry differ.
Which is currently used: Local aortic proxy.
Which appears more reliable: Native method is preferable after successful validation; neither is clinically validated.
Recommended authoritative implementation: FLOWCAT-derived centreline with documented estimator.
Recommended fallback: Local proxy named by method.
Migration risk: Non-comparable feature distributions.
Validation required: Circles/helices and resolution/smoothing sensitivity.

### Torsion duplication

Duplicated capability: Torsion.
Existing local implementation: No verified maintained output; related geometry utilities may overlap conceptually.
FLOWCAT implementation: Expected native local graph feature.
Differences: No operational local/native comparison exists.
Which is currently used: Neither in the five-case pilot.
Which appears more reliable: Unable to verify.
Recommended authoritative implementation: FLOWCAT after schema/unit/analytic validation.
Recommended fallback: None until a helix-tested implementation is available.
Migration risk: Introducing an unstable high-derivative feature.
Validation required: Straight line, planar circle, and 3D helix with noise sensitivity.

### Tortuosity duplication

Duplicated capability: Tortuosity.
Existing local implementation: Relative-length/path-to-chord proxies.
FLOWCAT implementation: Native segment/global graph features.
Differences: Endpoint selection, branch identity, and formulas may differ.
Which is currently used: Local aortic proxy; no native pilot value.
Which appears more reliable: Unable to rank without identical endpoints/formula.
Recommended authoritative implementation: FLOWCAT geometry with stored numerator/denominator/formula.
Recommended fallback: Local relative length explicitly named.
Migration risk: Apparent cohort shifts from definition rather than biology.
Validation required: Analytic paths and endpoint/branch sensitivity.

### Vessel length duplication

Duplicated capability: Vessel length.
Existing local implementation: Approximate axial/skeleton path length.
FLOWCAT implementation: Native branch/local graph length.
Differences: Local spurs and missing anatomical endpoints change path extent.
Which is currently used: Local technical path/aortic summaries.
Which appears more reliable: FLOWCAT design is preferable for labelled branches, but unverified operationally.
Recommended authoritative implementation: Native labelled branch arc length.
Recommended fallback: Technical length with endpoints/path ID.
Migration risk: Different branch definitions and truncation.
Validation required: Analytic curves and expert-defined proximal/distal endpoints.

### Graph generation duplication

Duplicated capability: Graph generation.
Existing local implementation: 26-neighbour voxel-skeleton technical graph.
FLOWCAT implementation: Segment graph, predicted graph, and dense local graph.
Differences: Local graph represents voxel connectivity; FLOWCAT graph binds branch cells, labels, and features.
Which is currently used: Local graph.
Which appears more reliable: FLOWCAT is more semantically complete; no current native output was generated.
Recommended authoritative implementation: Validated FLOWCAT graphs converted to typed safe structures.
Recommended fallback: Local graph marked technical/unlabelled.
Migration risk: Node/edge identity and all downstream joins change.
Validation required: Topology, cell references, labels, serialization trust, and versioned schema.

### Surface generation duplication

Duplicated capability: Surface generation.
Existing local implementation: Optional/local marching-cubes or surface utilities in separate paths.
FLOWCAT implementation: Marching-cubes preprocessing to VTK/STL and cleaned component surfaces.
Differences: Cleaning, normal orientation, coordinate frame, and file contracts differ.
Which is currently used: No successful surface in the five-case native chain.
Which appears more reliable: Unable to verify; representative FLOWCAT normals failed.
Recommended authoritative implementation: FLOWCAT surface only after mesh and registration QC.
Recommended fallback: A documented marching-cubes surface with identical QC, not silent substitution.
Migration risk: Centreline failure and coordinate misregistration.
Validation required: Watertightness, normals, component count, bounds, volume, and NIfTI landmark agreement.

## 1F. Immediate priorities

### Pin and disclose external inputs

Priority: P0.
Task: Pin FLOWCAT source, weights, environment, licence, and input/config hashes in every run manifest.
Reason: Correctness and reproducibility cannot be reconstructed from branch name or status CSV alone.
Dependencies: Git/model hashing, environment export, licence review.
Expected output: Immutable per-run and per-case provenance JSON.
Risk: Historical outputs remain only partially reconstructable.
Recommended next action: Make the fixed commit and required weight hashes hard failures, then capture dirty state and dependency versions.

### Repair native FLOWCAT geometry

Priority: P0.
Task: Establish a supported Linux VTK/VMTK/PyG environment and reproduce the native chain on `sub-570`.
Reason: Native centrelines, branches, labels, and geometry are prerequisites for artery-specific science.
Dependencies: Linux/Ubuntu, compatible VTK/VMTK/PyTorch/PyG, FLOWCAT models.
Expected output: Surface, centreline, branch model, arrays, predicted graph, local graph, and persisted logs.
Risk: Surface normals or topology may still fail and require upstream-compatible preprocessing.
Recommended next action: Containerize the pinned environment and retain full stdout/stderr plus mesh QC.

### Validate coordinate contracts

Priority: P0.
Task: Validate NIfTI RAS/LPS/LAS, VTK, and FLOWCAT LPI-corner-relative conversions.
Reason: A frame error silently corrupts label, radius, proximity, and volume mapping.
Dependencies: Typed adapter, image geometry, landmark fixtures.
Expected output: Round-trip tests and case-level registration QC.
Risk: Existing fallback and future native results may not be directly comparable.
Recommended next action: Add known-point/corner transforms and overlay landmarks/surfaces in Slicer.

### Make completion and failures explicit

Priority: P0.
Task: Replace single sentinels/overwritten status with immutable output contracts and structured failures.
Reason: Current status can label partial stages complete and loses exact errors/commands.
Dependencies: Versioned schemas, atomic writes, run IDs.
Expected output: Required/optional artifact manifest and append-only stage events.
Risk: Migration of existing cohort resumability logic.
Recommended next action: Validate every required artifact, hash it, and persist exception/stderr before declaring success.

### Protect de-identification

Priority: P0.
Task: Minimize DICOM-derived sidecar metadata and audit outputs/logs for PHI.
Reason: Existing BIDS-style conversion can preserve birth date and sex.
Dependencies: Governance-approved metadata policy and de-identified fixtures.
Expected output: Redacted/minimal sidecar schema and PHI-safe reporting checks.
Risk: Removing fields needed for acquisition analysis without a governed linkage strategy.
Recommended next action: Define allowed metadata fields and keep sensitive joins in a controlled source outside analysis exports.

### Produce named carotid branches

Priority: P1.
Task: Generate and validate CCA, bulb, ICA, and ECA graph labels and volume mappings.
Reason: Broad-tree sums cannot support artery-specific carotid perivascular inference.
Dependencies: Successful native FLOWCAT chain, adapter, expert labelled cases.
Expected output: Side-specific labelled graph, centreline, and artery volume with confidence/unknown class.
Risk: Anatomic variants and wrong graph-cell mappings.
Recommended next action: Start with `sub-570`, then review five pilot overlays against expert landmarks.

### Build vessel-centred coordinates

Priority: P1.
Task: Validate the implemented arc-length, tangent, and parallel-transport frames on complete native named centrelines.
Reason: The deterministic frame exists, but stable longitudinal/radial/circumferential interpretation requires real geometry validation.
Dependencies: Named branches, resampled centreline, coordinate validation.
Expected output: Real typed sample points with verified frame, radius, branch ID, label, confidence, and source.
Risk: Frame flips near low curvature/bifurcations.
Recommended next action: Extend synthetic invariance tests, then overlay frame orientation on the first complete native case.

### Validate adaptive perivascular equations

Priority: P1.
Task: Use capped diameter-adaptive fat/calcium thickness by named artery and quantify sensitivity.
Reason: Diameter influences appropriate shell width but must not imply proportional wall thickness; a maximum is necessary.
Dependencies: Reliable local diameter and named territories.
Expected output: Configured `clip(alpha * diameter, floor, maximum)` masks and sensitivity report.
Risk: Method changes current whole-tree burden values.
Recommended next action: Prespecify artery-specific floor/alpha/cap and compare against expert visual extent.

### Complete formal Slicer QC

Priority: P1.
Task: Complete structured Slicer review of existing whole-tree overlays and, when available, native named artery/centreline/label-volume outputs.
Reason: Headless binary-only QC and automated-pass status do not establish named anatomical plausibility.
Dependencies: Consistent Slicer loader and structured review form.
Expected output: Adjudicable whole-tree and named-territory QC flags, reviewer state, and failure reasons.
Risk: Review identifies model/preprocessing defects requiring reruns.
Recommended next action: Use the regenerated superior-context 570 montage for triage, but defer named-carotid disposition until native outputs exist.

### Add adjacent tissue compartments

Priority: P2.
Task: Supply and validate optional coarse muscle, jugular/vein, bone, thyroid, and airway masks through the implemented carotid interface.
Reason: Tissue composition/contact is scientifically richer than binary fat versus “wall.”
Dependencies: Named vessel surfaces/frames, validated coarse models, residual class.
Expected output: Real radial composition/contact features with explicit availability and uncertainty.
Risk: Coarse anatomy errors contaminate small local regions.
Recommended next action: Begin with internal jugular and bone contact on expert-reviewed carotid levels.

### Add local profiles and repeatability

Priority: P2.
Task: Execute the implemented longitudinal/circumferential profiles and study scanner/resolution/repeatability sensitivity.
Reason: Whole-vessel averages hide focal disease and may be scanner-dependent.
Dependencies: Stable coordinates, acquisition metadata, repeat/phantom cases.
Expected output: Local profile schema and reproducibility/sensitivity report.
Risk: Multiple-comparison burden and unstable small ROIs.
Recommended next action: Define minimum ROI/sample support and one prespecified repeatability experiment.

### Gate exploratory models and features

Priority: P3.
Task: Evaluate PVAT radiomics, dental analysis, intracranial extension, arch classification, and access prediction only after the core milestone.
Reason: These add dependency and validation burden without fixing current artery identity/geometry gaps.
Dependencies: P0/P1 outputs and scientific hypotheses.
Expected output: Separate feasibility reports with go/no-go criteria.
Risk: Scope expansion and false precision.
Recommended next action: Select at most one extension after the named-carotid pipeline passes QC.

## Audit conclusion

The repository can load and batch-process CTA/masks, produce broad binary
arterial/aortic masks, calculate candidate calcium and perivascular fat measures,
and generate automated/manual-QC artifacts. It now also contains a fail-closed,
typed deterministic carotid path from complete native FLOWCAT artifacts through
anatomical artery-volume propagation, vessel-centred shells/composition/sectors/
contacts, three-level outputs, and headless QC. The downstream implementation is
synthetic-tested; it has not produced a real named-carotid result. On real
`sub-570`, the run correctly stopped because `branch_model`, centreline array,
predicted graph, and local graph are missing, then produced binary-only QC with
manual review pending. The narrow next task is to reproduce the pinned native
FLOWCAT geometry/label chain for `sub-570` on supported Linux, preserve all logs,
then run the existing additive pipeline and validate its coordinate/label-volume
overlay in Slicer before cohort analysis.
