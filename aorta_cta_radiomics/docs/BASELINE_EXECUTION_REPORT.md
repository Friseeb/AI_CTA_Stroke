# Baseline execution report

Report date: 2026-07-31

This report records a preimplementation repository snapshot. It describes what
was actually generated for five de-identified Daylight Extended pilot cases and
does not establish clinical validity. Formal visual review in 3D Slicer is still
pending. No patient name, date of birth, accession, UID, or other direct
identifier is included.

## Baseline definition

The five cases were selected before the adaptive runs to span different existing
calcium/wall/fat burden profiles. They are not a random or prevalence-representative
sample. The active supra-aortic path produced a broad binary FLOWCAT
`fast-lowres` arterial tree, then used a non-equivalent 3D skeleton/anisotropic-EDT
fallback for local radius and capped diameter-adaptive shells:

- Fat thickness: `clip(1.0 * local_diameter, resolution_floor, 5 mm)`.
- Calcium neighborhood thickness: `clip(0.25 * local_diameter,
  resolution_floor, 2 mm)`.
- Candidate fat range: -190 to -30 HU.
- Calcium output is a CTA candidate partition, not confirmed plaque or an
  Agatston score.

The original interactive shell command lines were not persisted. The following
are the documented reproduction commands, not claimed verbatim historical
commands:

```bash
/opt/anaconda3/envs/totalseg-mac/bin/python \
  subprojects/studies/anterior_circulation_supraaortic/scripts/run_flowcat_segmentation.py \
  --profile fast-lowres --all

/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  subprojects/studies/anterior_circulation_supraaortic/scripts/run_adaptive_vessel_burden_v2.py \
  --all --fail-fast
```

The source manifest is
`subprojects/studies/anterior_circulation_supraaortic/manifests/daylight_extended_pilot5_adaptive.csv`.
CTA paths use de-identified filenames under `/Volumes/DICOM5/daylightbids`.
Binary segmentations are under
`outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/<case>_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

### Common generated outputs

Each adaptive case directory contains a cleaned binary arterial tree; total,
inner, and outer adaptive fat masks; raw, retained, definite-bone, and ambiguous
calcium masks; a bone-union mask; technical centreline-reference CSV; sum-level
CSV; QC/provenance JSON; and a rapid montage PNG. These are development artifacts.

### Common warnings and limitations

- Development-only broad `fast-lowres` tree; not a named-artery segmentation.
- Technical skeleton paths are not anatomical branches and do not reproduce VMTK.
- EDT radii are not `MaximumInscribedSphereRadius`.
- Nearest-centreline/Voronoi assignment is a development approximation, not
  anatomical boundary propagation.
- Intracranial fat should not be called physiologic PVAT.
- Calcium candidates are not confirmed plaque or clinical calcium scores.
- Local-lumen references are technical and do not prove orthogonal lumen geometry.
- Proximal bone scope remains incomplete.
- Every case touches an image boundary: `sub-766` at `y_low`; the other four at
  `z_low`.
- Rapid montages were reviewed, but structured Slicer QC is pending for all cases.

## Case-level baseline

### sub-766

Case ID: `sub-766`.

Reason selected: Low across the prior calcium/wall/fat burden measures; low-burden comparator.

Input CTA: `/Volumes/DICOM5/daylightbids/sub-766_acq-CTA_ct.nii.gz`.

Input segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-766_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

Original spacing: `[0.4880000055, 0.4880000055, 0.25]` mm in `x,y,z`; orientation recorded as LAS.

Image dimensions: `[512, 512, 1613]` voxels in `x,y,z`.

Available vessel labels: Binary background/foreground only (`0`, `1`); no named arteries.

Command used: Historical command unavailable; use the two documented reproduction commands above with the five-case manifest.

Environment: Segmentation Python 3.10.19; adaptive analysis Python 3.11.15 on macOS 26.5.2 arm64, SimpleITK 2.5.5, NumPy 2.4.4, SciPy 1.17.1; package and `cta_common` version 0.1.0. The adaptive provenance records Git commit `e5d55...` and a dirty tree.

GPU: FLOWCAT generation recorded device `cpu`; adaptive burden ran on CPU.

CPU: Run provenance records arm64 CPU execution but not CPU model; current host is Apple M4 Max with 16 logical CPUs and should not be assumed to be immutable historical provenance.

RAM: Per-process peak not recorded; current host has 128 GiB.

GPU memory: Not applicable to the recorded CPU stages; no peak was recorded.

Execution time: FLOWCAT segmentation 843.298 s; adaptive analysis 96.707 s; summed recorded stages 940.005 s.

Generated outputs: Common output set; fat 24.867 mL; raw calcium 0.2595 mL; retained calcium 0.1197 mL; definite bone 0.0151 mL; ambiguous calcium 0.1247 mL; unresolved fraction 0.09255; 90 pre-clean components; largest-component fraction 0.998316; 49 degree-one skeleton endpoints; 167 technical paths.

Warnings: All common warnings; bone map marked invalid.

Failures: Bone-aware prerequisite failure recorded exactly as `empty required craniofacial masks: teeth_lower, teeth_upper`; `bone_aware_valid=false`. The runner still produced a development result and did not establish valid bone-aware classification.

Manual intervention: None recorded during automated generation; formal manual Slicer review remains required.

Missing labels: CCA, carotid bulb, ICA, ECA, subclavian, vertebral, intracranial territories, side, and confidence.

Visual QC findings: Rapid montage only; shell is truncated at `y_low`. No formal anatomic plausibility determination.

Overall status: `automated_pass_visual_pending_development`, with invalid bone-aware prerequisites.

### sub-648

Case ID: `sub-648`.

Reason selected: Very low prior calcium burden with high prior wall/fat burden; contrast between burden types.

Input CTA: `/Volumes/DICOM5/daylightbids/sub-648_acq-CTA_ct.nii.gz`.

Input segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-648_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

Original spacing: `[0.5490000248, 0.5490000248, 0.25]` mm in `x,y,z`; orientation recorded as LPS.

Image dimensions: `[512, 512, 1743]` voxels in `x,y,z`.

Available vessel labels: Binary background/foreground only (`0`, `1`); no named arteries.

Command used: Historical command unavailable; use the documented reproduction commands above.

Environment: Segmentation Python 3.10.19; adaptive Python 3.11.15, macOS 26.5.2 arm64, SimpleITK 2.5.5, NumPy 2.4.4, SciPy 1.17.1, package/`cta_common` 0.1.0; dirty development tree recorded.

GPU: Recorded CPU execution; adaptive burden CPU.

CPU: arm64 recorded; CPU model absent from run provenance. Current host is Apple M4 Max, 16 logical CPUs.

RAM: Peak not recorded; current host capacity 128 GiB.

GPU memory: Not applicable/no peak recorded.

Execution time: FLOWCAT segmentation 787.057 s; adaptive analysis 121.255 s; summed recorded stages 908.312 s.

Generated outputs: Common output set; fat 59.153 mL; raw calcium 0.7353 mL; retained 0.4727 mL; definite bone 0.1287 mL; ambiguous 0.1339 mL; unresolved fraction 0.01852; 126 components; largest-component fraction 0.998762; 44 degree-one endpoints; 208 technical paths.

Warnings: All common warnings; shell truncated at `z_low`.

Failures: No automated stage failure; this does not establish anatomical validity.

Manual intervention: None recorded; formal Slicer review required.

Missing labels: All named artery, side, branch, bifurcation, and confidence labels.

Visual QC findings: Rapid montage generated; no structured review. Bone-aware prerequisite validity flag is true, but proximal bone coverage remains incomplete.

Overall status: `automated_pass_visual_pending_development`.

### sub-189

Case ID: `sub-189`.

Reason selected: Very high prior calcium/wall burden with very low prior fat; high-calcium/low-fat extreme.

Input CTA: `/Volumes/DICOM5/daylightbids/sub-189_acq-CTA_ct.nii.gz`.

Input segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-189_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

Original spacing: `[0.4880000055, 0.4880000055, 0.25]` mm in `x,y,z`; orientation recorded as LAS.

Image dimensions: `[512, 512, 1481]` voxels in `x,y,z`.

Available vessel labels: Binary `0/1`; no named arteries.

Command used: Historical command unavailable; use documented reproduction commands.

Environment: Segmentation Python 3.10.19; adaptive Python 3.11.15 on macOS 26.5.2 arm64 with SimpleITK 2.5.5, NumPy 2.4.4, SciPy 1.17.1, package/`cta_common` 0.1.0; dirty Git state recorded.

GPU: CPU recorded for FLOWCAT generation and adaptive analysis.

CPU: arm64 recorded; model not recorded. Current host is Apple M4 Max with 16 logical CPUs.

RAM: Peak not recorded; current host capacity 128 GiB.

GPU memory: Not applicable/no peak recorded.

Execution time: FLOWCAT segmentation 531.084 s; adaptive analysis 76.969 s; summed recorded stages 608.053 s.

Generated outputs: Common output set; fat 15.639 mL; raw calcium 3.3157 mL; retained 3.1993 mL; definite bone 0.0705 mL; ambiguous 0.0459 mL; unresolved fraction 0.08758; 63 components; largest-component fraction 0.994163; 38 degree-one endpoints; 161 technical paths.

Warnings: All common warnings; shell truncated at `z_low`.

Failures: No automated stage failure; anatomical truth unavailable.

Manual intervention: None recorded; formal Slicer QC required.

Missing labels: All named artery/side/branch and carotid bifurcation labels.

Visual QC findings: Rapid montage only. High retained calcium burden is a candidate measurement, not confirmed plaque.

Overall status: `automated_pass_visual_pending_development`.

### sub-570

Case ID: `sub-570`.

Reason selected: High across prior calcium/wall/fat burden and chosen as the detailed native-FLOWCAT, adaptive-shell, aortic, and dental demonstration case.

Input CTA: `/Volumes/DICOM5/daylightbids/sub-570_acq-CTA_ct.nii.gz`.

Input segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

Original spacing: `[0.5389999747, 0.5389999747, 0.25]` mm in `x,y,z`; orientation recorded as LPS.

Image dimensions: `[512, 512, 1783]` voxels in `x,y,z`.

Available vessel labels: Binary `0/1`; no named arteries.

Command used: Historical command unavailable; use documented reproduction commands. The exact native VMTK smoke-test command was not persisted.

Environment: Segmentation Python 3.10.19; adaptive Python 3.11.15 on macOS 26.5.2 arm64, SimpleITK 2.5.5, NumPy 2.4.4, SciPy 1.17.1, package/`cta_common` 0.1.0; dirty development tree recorded. The `totalseg-mac` environment does not expose VMTK, VTK, `torch_geometric`, or `torch_cluster` for the native chain.

GPU: FLOWCAT generation recorded CPU; adaptive burden CPU.

CPU: arm64 recorded, model absent; current host is Apple M4 Max, 16 logical CPUs.

RAM: Peak not recorded; current host capacity 128 GiB.

GPU memory: Not applicable/no peak recorded.

Execution time: FLOWCAT segmentation 747.253 s; adaptive analysis 112.928 s; summed recorded stages 860.181 s. Native VMTK and older aortic execution times are unavailable.

Generated outputs: Common adaptive output set; tree volume 249.922 mL; median diameter proxy 4.166 mm; median fat thickness 4.166 mm with 5-mm cap; median calcium thickness 1.078 mm with 2-mm cap; fat 62.5633 mL; raw calcium 6.1988 mL; retained calcium 4.2181 mL; definite bone 0.1617 mL; ambiguous calcium 1.8189 mL; unresolved fraction 0.015758; 84 components; largest-component fraction 0.9994168; 37 degree-one endpoints; 151 technical paths.

Warnings: All common warnings; large ambiguous calcium volume; whole binary tree includes aorta; shell truncated at `z_low`; component-level bone-contact propagation can label an entire component.

Failures: Native FLOWCAT found 19 automatic endpoints but failed on original and cleaned surfaces because it could not compute valid surface normals after all prescribed mesh-bottom retries. Persisted reason token: `flowcat_vmtk_automatic_centerline_failed_sub570_surface_normals`. Original stderr/traceback was not saved, so the exact thrown exception is unable to verify.

Manual intervention: Rapid montage inspection occurred; formal structured Slicer review is pending. Native retries used original and cleaned surfaces, but their exact interactive commands are unavailable.

Missing labels: CCA, bulb, ICA, ECA, subclavian, vertebral, intracranial, side, branch graph, anatomical landmarks, and confidence.

Visual QC findings: Rapid montage suggests the 5-mm maximum visibly narrows the adaptive fat extent relative to the prior fixed shell. High retained and ambiguous calcium warrant focused skull/vertebral and lumen-boundary review. Whole-tree extent includes the aorta. These are preliminary observations, not completed formal QC.

Overall status: Broad-tree adaptive output is `automated_pass_visual_pending_development`; the additive named-carotid run is `failed` at preflight because native geometry/graphs are absent; final headless binary-only QC is `qc_generated_pending_visual_review`.

### sub-433

Case ID: `sub-433`.

Reason selected: Mid-range burden reference between the selected extremes.

Input CTA: `/Volumes/DICOM5/daylightbids/sub-433_acq-CTA_ct.nii.gz`.

Input segmentation: `outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-433_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz`.

Original spacing: `[0.5440000296, 0.5440000296, 0.25]` mm in `x,y,z`; orientation recorded as LAS.

Image dimensions: `[512, 512, 1669]` voxels in `x,y,z`.

Available vessel labels: Binary `0/1`; no named arteries.

Command used: Historical command unavailable; use documented reproduction commands.

Environment: Segmentation Python 3.10.19; adaptive Python 3.11.15, macOS 26.5.2 arm64, SimpleITK 2.5.5, NumPy 2.4.4, SciPy 1.17.1, package/`cta_common` 0.1.0; dirty development tree recorded.

GPU: Recorded CPU execution for both stages.

CPU: arm64 recorded, CPU model absent; current host Apple M4 Max, 16 logical CPUs.

RAM: Peak not recorded; current host capacity 128 GiB.

GPU memory: Not applicable/no peak recorded.

Execution time: FLOWCAT segmentation 736.387 s; adaptive analysis 111.747 s; summed recorded stages 848.134 s.

Generated outputs: Common output set; fat 53.6414 mL; raw calcium 2.3747 mL; retained 1.7417 mL; definite bone 0.0758 mL; ambiguous 0.5572 mL; unresolved fraction 0.014463; 49 components; largest-component fraction 0.9949048; 25 degree-one endpoints; 81 technical paths.

Warnings: All common warnings; substantial ambiguous fraction; shell truncated at `z_low`.

Failures: No automated stage failure; anatomical/visual validation incomplete.

Manual intervention: None recorded during generation; formal Slicer review required.

Missing labels: All named artery/side/branch/landmark/confidence labels.

Visual QC findings: Rapid montage only; ambiguity and bone-contact propagation require focused review.

Overall status: `automated_pass_visual_pending_development`.

## Detailed `sub-570` evidence from other paths

### Maintained aortic output

An older maintained aortic case directory exists at
`aorta_cta_radiomics/outputs/daylight_full_run/cases/sub-570_acq-CTA_ct`.
The cleaned aorta mask contains 3,136,634 voxels, volume 227,814.49 mm³, one
connected component, source spacing `[0.5389999747, 0.5389999747, 0.25]` mm, and
array shape `[1783, 512, 512]` in `z,y,x`. It has multiple NIfTI masks, CSVs, and
QC PNGs. Its original command, environment, and stage timing cannot be recovered
from the current `stage_status.csv`, because later staged invocations overwrote
that file.

A separate earlier whole-tree/fixed-shell output reported 9.093 mL calcium and
68.617 mL fat in a fixed 5-mm shell. Those numbers are not interchangeable with
the adaptive result: segmentation extent, shell definition, bone partitioning,
and method differ. They are retained only as method-change evidence.

### Native FLOWCAT blocker

The native FLOWCAT output chain expected surface VTK/STL, VMTK centrelines,
`branch_model.vtk`, `centerline_segments_array.npy`, predicted/local graphs, and
anatomical labels. None was produced for the audited five-case run. On `sub-570`,
automatic endpoint detection reported 19 endpoints, followed by surface-normal
failure on both original and cleaned surface attempts. The exact persisted token
is `flowcat_vmtk_automatic_centerline_failed_sub570_surface_normals`. Because raw
stderr/traceback was not retained, no specific Python exception is reported as
the observed error.

The local fallback is a 3D Lee skeleton, anisotropic EDT radius, and 26-neighbour
technical graph. It is useful for development, but it is not VMTK-equivalent, has
no anatomical branch labels, and cannot establish native FLOWCAT correctness.

### Final additive carotid milestone execution

The finalized additive implementation provides a non-deserializing manifest
command, external-CTA-aware FLOWCAT validation, anatomical branch-to-lumen
propagation, deterministic named-carotid perivascular analysis, typed
case/vessel/sample outputs, and headless QC. It does not generate missing native
FLOWCAT artifacts and never substitutes the skeleton fallback.

A real fail-closed run was performed for `sub-570_acq-CTA_ct`. Its exact original
shell line was not persisted; this documented command reconstructs the invoked
interface and paths:

```bash
cd aorta_cta_radiomics
python -m aorta_cta_radiomics.carotid_cli run \
  --case-dir ../outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct \
  --case-id sub-570_acq-CTA_ct \
  --cta /Volumes/DICOM5/daylightbids/sub-570_acq-CTA_ct.nii.gz \
  --flowcat-repo ../external/arterial_flowcat \
  --config config/carotid_perivascular.yaml \
  --outdir ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct \
  --trusted-flowcat-artifacts
```

The run returned status 2 and wrote `failure.json`. The exact persisted failure is:

```text
stage=preflight
error_type=FileNotFoundError
error_message=Missing required FLOWCAT outputs: branch_model, centerline_segments_array, local_graph, segments_graph_pred
```

The milestone directory contains only:

- `failure.json`;
- `flowcat_output_manifest.json`;
- `qc/carotid_qc.json`; and
- `qc/carotid_qc_montage.png`.

It contains no anatomical artery label/confidence/uncertainty volume, named shell,
tissue/sector/contact mask, analysis manifest, or case/vessel/sample table. This is
the expected honest result of missing prerequisites, not a partial named-carotid
success.

The final regenerated manifest has SHA-256
`8d6f3fb9c8dc67d71362392606b534f8121d86c83750c0e06d460847e99bd2a8`.
It records the clean FLOWCAT branch/commit, segmentation SHA-256
`46151cf209318c28b98c87f5fee8d3cfcc2a5512da870a11bff428740169d549`,
all four missing required products, and SHA-256 for eight controlled local model
payloads (three segmentation checkpoints and five vessel-labelling folds). Its
`case.cta_path` is null because this is the separate non-executing inventory; a
successful `validate_flowcat_case(..., cta_path=...)` records the external CTA
without copying or linking it.

### Final headless `sub-570` QC

The final QC artifact was regenerated with the finalized canonical-display code.
The documented invocation is:

```bash
cd aorta_cta_radiomics
python -m aorta_cta_radiomics.carotid_qc \
  --case-id sub-570_acq-CTA_ct \
  --cta /Volumes/DICOM5/daylightbids/sub-570_acq-CTA_ct.nii.gz \
  --segmentation ../outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz \
  --flowcat-manifest ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/flowcat_output_manifest.json \
  --outdir ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/qc
```

`carotid_qc.json` has SHA-256
`1d28b08f39671469399203f656d44fcfbf296151c4c6b87e776b4bbe870184a8`;
the 2,250 x 825 montage has SHA-256
`a370afacfd909063f278efc1dd30df1d89019006fab3c92758b43cf4e55820e0`.
The report preserves original LPS geometry separately, then reorients display by
axis permutation/flip only—no resampling—to canonical RAS. It selects canonical
voxel `[292, 314, 1335]`, the 85th percentile of occupied superior axial slices,
for broad whole-tree arterial context and explicitly records
`not_a_named_carotid_localization`.

QC confirms CTA/segmentation shape and affine agreement, 3,443,029 binary
foreground voxels (250.0680 mL), 84 26-connected components, and bounding box
`[68,118,0]` to `[439,458,1595]`. All labelled-artery, shell, tissue, and calcium
overlays are unavailable. The four native artifacts remain missing; named results
are unavailable; manual-review reasons are missing named overlay, incomplete
native outputs, and pending visual review. A technical inspection confirmed that
the montage is oriented and labelled as superior whole-tree context, but it is not
a completed anatomical review or a named-carotid localization.

### Separate dental demonstration

`outputs/dental_example_sub-570/DENTAL_EXAMPLE_ANALYSIS.md` documents a separate
experimental analysis run on 2026-07-30. It is not part of the arterial baseline
pipeline and must not be treated as diagnosis. The analysis used 0.5-mm isotropic
working spacing, created 77 files, loaded 35 non-empty segments in Slicer, and
created 13 candidate fiducials. It described 19 FDI-labelled teeth (6 upper, 13
lower), 13 missing-position candidates (9 non-third-molar and 4 third-molar), no
implant candidates, no crown/bridge candidates, two over-sensitive periapical
candidates adjacent to FDI 37 (44.75 and 24.38 mm³), and 11 periodontal
candidates. These are experimental candidates, not clinical findings.

The dental report/model records `has_mandible=false` despite a non-empty lower
jawbone segment; this is a known heuristic/report inconsistency. A five-case
physical-distance audit found no arterial calcium candidate within 3 mm of upper
teeth, lower teeth, or mandible. The visible calcium-confounding concern in these
cases is therefore skull/vertebral proximity rather than demonstrated dental
contact. This does not exclude dental artifact elsewhere.

## Automated test baseline

The following tests were run before concurrent new adapter/perivascular work and
are the reproducibility baseline. All three commands disabled bytecode/cache
artifacts. Counts must not be combined with later untracked tests as if they were
one committed snapshot.

```bash
cd aorta_cta_radiomics
PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q -p no:cacheprovider
# 58 passed in 1.33s

cd ..
PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q -p no:cacheprovider cta_common/tests
# 38 passed in 0.37s

PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q -p no:cacheprovider \
  subprojects/studies/anterior_circulation_supraaortic/tests
# 32 passed in 0.30s
```

Baseline result: **128 passed, 0 failed**. This does not cover a complete DICOM
ingestion-to-analysis integration path, real-case PyRadiomics extraction, native
FLOWCAT geometry/labels, all orientation permutations, named artery volume
mapping, stable vessel-centred frames, tissue-contact analysis, FOV completeness,
or visual-anatomic validity.

### Current additive implementation tests

The historical 128-test baseline above remains unchanged. After the additive
implementation and final QC display update, the current package suite was rerun:

```bash
cd aorta_cta_radiomics
PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q -p no:cacheprovider
# 133 passed in 3.33s
```

A focused adapter, artery-volume, carotid analysis/CLI/QC, output-schema, and
perivascular suite passed **67 tests in 1.85 s**. Including the eight read-only
carotid audit tests gives 75 focused/audit tests. These are synthetic and contract
tests. They do not replace a successful complete-native real case, manual Slicer
review, biological ground truth, or scanner/reconstruction validation.

## Cohort and dry-run evidence

Existing artifacts demonstrate operational scale but not a single immutable
cohort run:

- `daylight_topo_pilot/stage_status.csv`: 20 cases x 4 stages = 80 `ok` rows,
  with sentinels present.
- `daylight_full_run/stage_status.csv`: 51 cases x 4 stages = 204 `ok` rows,
  while its case/aggregate directories contain broader historical material,
  including 683 wide case rows.
- `daylight_aorta_run`: 77 base-stage `ok` rows.
- `daylight_standard_arm_run`: 3 VISTA-only rows.

The SLAO dry-run manifest contains 566 rows. The metadata preflight classified
407 as eligible and 159 as ineligible:

- 407 `eligible_neuro_cta`.
- 83 `excluded_non_neuro_protocol`.
- 47 missing a CTA term.
- 29 missing a brain/neck/stroke term.

`aorta_batch_run_dryrun_preflight/stage_status.csv` contains 407 base-stage `ok`
rows, but logs explicitly say `DRY RUN` and no masks were produced. Therefore this
is eligibility/command-routing evidence only, not 407 successfully processed
cases. The exact original top-level command was not persisted; the artifact-derived
reproduction is:

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  aorta_cta_radiomics/scripts/run_manifest_staged.py \
  --manifest aorta_cta_radiomics/outputs/manifests/slaobids_aorta_manifest.csv \
  --outdir aorta_cta_radiomics/outputs/aorta_batch_run_dryrun_preflight \
  --stages base --metadata-filter neuro-cta --dry-run
```

## Baseline blockers and required next verification

1. Reproduce the native FLOWCAT chain on Linux/Ubuntu with pinned compatible
   VTK/VMTK/PyTorch/PyG; keep complete stdout/stderr, mesh QC, and every output.
2. Validate FLOWCAT LPI-corner-relative millimetres, NIfTI RAS/LPS, and VTK points
   with round-trip tests and a Slicer overlay.
3. Generate named CCA/ICA/ECA branches and validate graph `cell_id` to centreline
   and volume mapping before artery-specific sums.
4. Complete structured Slicer QC for CTA, arterial, fat, retained/ambiguous/bone
   calcium, and field-of-view on all five cases, beginning with `sub-570`.
5. Repeat `sub-766` only after its required craniofacial bone inputs are valid.
6. Treat concurrent untracked adapter/perivascular code as new implementation,
   not as evidence that this baseline already had those capabilities.

Overall baseline conclusion: five automated diameter-adaptive whole-tree outputs
exist; the historical and current additive suites pass; and batch/dry-run routing,
real fail-closed carotid preflight, model/artifact hashing, and final headless
binary-only 570 QC are demonstrated. The named-carotid downstream implementation
exists but produced no real named result because native FLOWCAT geometry/graphs
remain absent. Formal named-artery Slicer QC and clinical/anatomical validation
remain incomplete.
