# FLOWCAT integration

Document date: 2026-07-31

This document defines how this repository uses and consumes FLOWCAT. It separates
five things that must not be conflated:

1. upstream FLOWCAT model inference;
2. upstream native surface/centreline/graph/label post-processing;
3. the local typed adapter for already-generated outputs; and
4. the deterministic anatomical-volume/carotid analysis that consumes a complete
   validated native case; and
5. the non-equivalent local skeleton/EDT development fallback.

The adapter does not create a second arterial pipeline and does not make an
unvalidated output clinically valid.

## Pinned upstream identity and licence

- External repository: `https://github.com/FLOWCAT-CV/arterial`.
- Intended branch: `models/2026-03`.
- Authoritative commit:
  `1991ca5b238d7c868c9c3a0a390057bc89aee559`.
- Local checkout: `external/arterial_flowcat`.
- Authoritative licence: Creative Commons Attribution-NonCommercial 4.0, from
  the checkout's `LICENSE` file.
- Current use: research/development segmentation plus a deterministic, fail-closed
  consumer for complete native geometry/labelling outputs; the latter is tested
  synthetically but blocked on the real pilot artifacts.
- Redistribution implications: do not redistribute source, reconstructed model
  files, or a container containing them until licence/attribution and any
  model-specific terms have been reviewed. The README badge is not a substitute
  for the licence file.
- Commercial-use implications: CC BY-NC 4.0 restricts commercial use. Obtain
  appropriate permission/legal review before commercial deployment.
- Recommended mitigation: retain repository URL, branch, commit, licence,
  citations, model-file hashes, and use context in every run; keep the external
  checkout outside this repository.

No other licence should be inferred merely because a local package is MIT. The
licence of a downstream artifact or redistribution bundle may require separate
analysis.

## How FLOWCAT is integrated locally

The checkout is manually cloned and Git-ignored. It is not vendored, a Git
submodule, or installed by `pip`. The supra-aortic study runner adds the checkout
to the Python import path and imports
`arterial.segmentation.inference.perform_single_inference_nnunet` in-process. The
runner checks the fixed Git commit and rejects tracked source modifications.

`prepare_flowcat_models.py` calls the pinned upstream chunk-rebuild helper, then
verifies configured file size and SHA-256. It never downloads model files from the
network. The configured chunks and reconstructed model files remain local.

The audited five-case run generated only nested binary segmentations:

```text
outputs/anterior_circulation_supraaortic/
  flowcat_2026_03/fast-lowres/
    sub-<ID>_acq-CTA_ct/
      extracranial_vessels/
        segmentation.nii.gz
        segmentation_qc.json
        segmentation_qc_montage.png
```

The cases are `sub-189`, `sub-433`, `sub-570`, `sub-648`, and `sub-766`.
These segmentations feed local adaptive fat/calcium development code. No current
case contains the native FLOWCAT branch model, centreline array, predicted graph,
local graph, or named-artery volume.

The upstream README also documents a root-level
`extracranial_vessels_segmentation.nii.gz`. Local discovery supports both that
documented layout and the integrated nested layout; the resolved path is retained
in the manifest rather than silently renamed.

## Typed adapter

The local package is `aorta_cta_radiomics.flowcat_adapter`. It consumes existing
artifacts; it does not run inference or native FLOWCAT processing. Its low-level
API is Python-based:

- Discovery and manifests: `discover_flowcat_files`, `build_output_manifest`,
  `write_output_manifest`, `create_and_write_output_manifest`,
  `validate_required_outputs`.
- Safe NIfTI handling: `geometry_from_nifti`, `load_binary_nifti`.
- Centreline handling: `load_numpy_array`, `load_centerline_segments`,
  `branches_to_image_physical`.
- Graph handling: `load_networkx_graph`, `validate_graph_schema`,
  `build_typed_vessel_graph`.
- Labels: `FLOWCAT_ARTERY_CLASSES`, `map_graph_labels_to_branches`.
- Coordinates: RAS/LPS, FLOWCAT-relative/NIfTI-RAS, geometry compatibility,
  orientation, image-bound checks.
- Landmarks: `load_landmarks_ras_mm`.
- Optional surfaces: `load_polydata`, `load_vtk_polydata`.
- Provenance: `collect_processing_provenance` and dependency collection.
- Full contract: `validate_flowcat_case`.

The supported headless interface is
`python -m aorta_cta_radiomics.carotid_cli manifest|run`. The package metadata also
declares `aorta-cta-carotid` and `aorta-cta-carotid-qc`. Both entry points were
verified in the named Conda environment after editable installation; activate
that environment (or use its absolute `bin/` path) before invoking them. The
equivalent `python -m` commands remain available for module-based execution.

The adapter defines typed records for case metadata, image geometry, artery
identity, centreline point/branch, bifurcation, vessel graph, local geometric
feature, output manifest, processing provenance, and QC flags. The focused adapter
suite passes 20 tests; the complete additive core suite passes 67. These validate
fixtures/contracts, not upstream model accuracy or native case generation.

### Manifest behavior

`build_output_manifest` performs non-executing discovery. By default it computes
SHA-256, checks that resolved artifacts remain within the case directory, records
format/size/coordinate reference/units, and reports missing required outputs. It
does not open a pickle or object NumPy array. `write_output_manifest` writes schema
version `1.0.0` JSON atomically.

The default required set is:

- `segmentation`;
- `branch_model`;
- `centerline_segments_array`;
- `segments_graph_pred`; and
- `local_graph`.

`validate_required_outputs` raises one clear error listing all missing required
artifacts. Full `validate_flowcat_case` additionally loads/validates the binary
NIfTI, trusted centreline array and graphs, maps graph edge labels to centreline
`cell_id`, and optionally loads VTK. It intentionally defaults to
`trusted_source=False` and `load_vtk=False`.

### Current adapter limitations

- The five pilot directories have only `segmentation`, so full validation is
  expected to fail with missing native outputs.
- `validate_flowcat_case(..., cta_path=...)` now accepts the original CTA outside
  the FLOWCAT tree, resolves but does not copy/link it, validates segmentation
  geometry, records it in the returned manifest, and uses it to restore
  FLOWCAT-relative centreline points to NIfTI RAS physical millimetres. When
  omitted, discovery still falls back to case-local `cta.nii.gz`/`head_cta.nii.gz`.
- The separate `artery_volume.py` and `carotid_analysis.py` modules now map trusted
  labelled branches into an uncertainty-preserving NIfTI label volume and named
  perivascular outputs. This is downstream deterministic analysis, not part of
  FLOWCAT inference, and has no complete real case yet.
- Access-prediction directories and every optional upstream revision are not yet
  represented in the default discovery contract.
- VTK is optional. File presence is not mesh validity, coordinate validity, or
  anatomical validity.
- `carotid_cli` discovers and hashes eight controlled local FLOWCAT model payloads
  in the current checkout, but cannot reconstruct missing historical commands or
  prove model/anatomical accuracy.

## Trusted-artifact policy

Python pickle is executable serialization. NumPy object arrays also use pickle
internally. They must never be opened from email, cloud shares, downloads,
unreviewed archives, or another uncontrolled origin.

The policy is:

1. The source case must be generated locally from the clean pinned FLOWCAT commit
   and locally verified model payload.
2. The case directory and files must be access-controlled and case-local.
3. Record file SHA-256 before loading.
4. Inspect the manifest and required-output list without deserialization first.
5. Set `trusted_source=True` only after the preceding checks.
6. Convert accepted graph/object data immediately into typed structures and, for
   durable exchange, a non-executable schema such as JSON/Parquet plus VTK/NIfTI.
7. A checksum proves identity, not safety; the generating source must also be
   trusted.

The adapter enforces explicit trust for:

- `segments_graph.pickle`;
- `segments_graph_pred.pickle`;
- `local_graph.pickle`;
- access-prediction or attention-map pickles if later supported; and
- `centerline_segments_array.npy`, because its `(n_segments, 2)` array has object
  dtype.

NIfTI and JSON do not require pickle trust, but still require schema, size,
geometry, bounds, and content validation. VTK parsing is separately opt-in because
it adds a compiled optional dependency and a distinct attack/robustness surface.

## Output contracts

### CTA

Expected input: explicit external `cta_path`, or case-local `<case>/cta.nii.gz` /
`<case>/head_cta.nii.gz` fallback.

Contract: NIfTI affine is treated as RAS physical millimetres by the adapter.
Shape, affine, voxel sizes, and orientation are retained in `ImageGeometry`.
An explicit external CTA is not added to case-directory discovery, but its resolved
path is attached to the validated manifest and its geometry drives segmentation
compatibility and centreline coordinate reconstruction. No copy/link is created.
Oblique affines require special caution because the upstream inference adapter
uses affine-diagonal spacing unless explicitly allowed by the local runner.

### Arterial segmentation

Supported paths, in precedence order:

1. `<case>/<mode>/segmentation.nii.gz`;
2. `<case>/<mode>_segmentation.nii.gz`; and
3. `<case>/<mode>/<mode>_segmentation.nii.gz`.

Contract: finite, binary NIfTI with values 0/1 and geometry compatible with CTA
when a reference geometry is supplied. A binary mask carries no named artery
identity.

### Surface and branch model

Expected primary paths include `<case>/<mode>/segmentation.vtk`,
`segmentation.stl`, `branch_model.vtk`/`.vtp`, and component surfaces under
`segmentations/`, `centerlines/`, and `branch_models/`.

Contract: physical vertices/cells in millimetres, but surface formats do not by
themselves prove the NIfTI frame. Required QC includes non-empty points/cells,
finite coordinates, bounds, normal consistency, components, and landmark overlay.

### Centreline segments

Expected path: `<case>/<mode>/centerline_segments_array.npy`.

Contract: NumPy object array with shape `(n_segments, 2)`. For each row, column 0
is an `N x 3` coordinate array and column 1 is an `N` radius array corresponding
to upstream `MaximumInscribedSphereRadius`. Coordinates and radii are millimetres.
Segment index is the `cell_id` used for graph linkage. All values must be finite;
point/radius lengths must match; radii must be non-negative.

### Predicted segment graph

Expected path: `<case>/<mode>/segments_graph_pred.pickle`.

Contract: trusted NetworkX graph. Each labelled edge must provide `cell_id`,
`vessel_type`, and `vessel_type_name`. The current adapter validates FLOWCAT's
14-class convention (`0..13`: other, aorta/arch, brachiocephalic, bilateral CCA,
subclavian, vertebral, ICA, ECA, and basilar) and rejects conflicting/orphan
`cell_id` mappings. Label confidence is retained when available; absence of
confidence must not be replaced with certainty.

### Local graph

Expected path: `<case>/<mode>/local_graph.pickle` (legacy `graph.pickle` is also
discoverable).

Contract: trusted dense NetworkX graph. Upstream code may attach positions,
radius, vessel identity, and local/segment/global feature dictionaries including
length, diameter, tortuosity, curvature, and torsion. Each feature's definition
and units must be validated from the producing version before scientific use; a
field's presence is not validation.

### Landmarks

Expected path: `<case>/<mode>/landmarks/landmarks.json` or legacy
`<case>/<mode>/landmarks.json`.

Contract: finite named points with explicit physical frame. Slicer-specific and
upstream variants must not be assumed identical. Landmark coordinates must fall
within the CTA physical bounds after the stated transform.

### Access prediction

Upstream may write
`<case>/<mode>/access_prediction/{access}_{side}/access_prediction.json`, plus
attention-map pickle/PNG/VTK/point JSON and preprocessing graph artifacts. These
are optional, not generated or consumed by the current study, and are outside the
first carotid perivascular milestone.

## Anatomical artery volume and carotid core

`artery_volume.py` does not infer anatomy. It rasterizes labels already attached
to trusted native FLOWCAT branches. Arrays remain nibabel IJK and coordinates
remain NIfTI RAS physical millimetres. It computes two independent strategies:

- nearest compatible labelled centreline, constrained by component, local radius,
  label confidence and tie tolerance; and
- marker-controlled watershed constrained to connected lumen.

The final volume uses connected propagation but assigns derived label 14 wherever
strategies disagree, in the bifurcation-exclusion zone, or where support/confidence
is inadequate. It writes final labels, confidence, uncertainty, bifurcation,
nearest, connected and disagreement volumes. Disconnected unseeded lumen cannot
inherit an anatomical label. Derived labels 1–12 retain the corresponding FLOWCAT
classes; FLOWCAT “other” and basilar map to derived 13 while their original source
codes remain on typed branches. Label 0 is background.

`carotid_analysis.py` defaults to exactly one RCCA, LCCA, RICA and LICA. It fails
when any is absent or maps to multiple branches rather than guessing how to merge
them. It requires positive native `MaximumInscribedSphereRadius`, rejects sheared
affines, explicitly transposes nibabel IJK to perivascular KJI/ZYX, and produces:

- 1-mm vessel-centred path sampling and parallel-transport frames;
- fixed 0–1, 1–2, 2–3, 3–5 and 5–10 mm radial bands;
- a fat envelope using `clip(1.0 * diameter, 2 * maximum spacing, 5 mm)`;
- raw/filtered masks with other-artery/bifurcation/anatomical exclusions;
- tissue composition with explicit unavailable/uncertain classes;
- 16-sector circumferential assignment, longitudinal/radial profiles and
  contact summaries at 0.5, 1 and 2 mm;
- case-, vessel-, and sample-point-level records; and
- label/confidence/uncertainty, shell, tissue and sector NIfTIs on success.

Optional binary vein, bone, airway, thyroid, muscle, uncertain and high-density
masks must match CTA geometry. Missing optional anatomy is recorded and its
contact measurement is null, never silently converted to zero. A carotid bulb is
not inferred: it appears only when the caller supplies an evidence-bearing CCA/ICA
path interval. The current milestone does not execute its configured calcium
envelope and leaves `calcium_burden_mm3=null`; calcium remains a separate
development candidate/optional QC overlay. Every vessel remains manual-review
required.

On successful analysis, `carotid_cli run` writes case JSON, vessel CSV,
sample-point Parquet (or explicitly recorded CSV fallback), hashed output manifest,
anatomical/QC NIfTIs, `artery_label_summary.json`, and `analysis_manifest.json`.
No such named output exists for the current real 570 case because preflight failed.

## Headless carotid QC

`python -m aorta_cta_radiomics.carotid_qc` accepts CTA, binary segmentation,
FLOWCAT manifest, output directory, and optional labelled-artery/shell/tissue/
calcium overlays. It validates binary/overlay geometry, inventories required
native outputs, computes mask metrics, and writes `carotid_qc.json` plus a
three-plane montage using Matplotlib's headless Agg backend.

Original image geometry is preserved in the report. For display only, the final
code applies axis permutation/flips—no interpolation/resampling—to canonical RAS.
It selects the 85th percentile of axial slices containing whole-tree foreground
for superior context and records `not_a_named_carotid_localization`. Optional
overlays are visual QC inputs only; this command does not infer vessel names or
generate burdens. It does not inspect descriptive NIfTI/DICOM metadata and always
leaves visual review pending/manual review required.

## Coordinate conventions

This is the highest-risk integration boundary.

- NIfTI affines are handled as RAS physical millimetres by nibabel/adapter
  geometry.
- SimpleITK commonly exposes LPS physical coordinates; RAS/LPS conversion flips
  the first two axes.
- Pilot NIfTIs include LAS and LPS orientation codes; orientation was not
  canonicalized before all upstream operations.
- FLOWCAT centreline post-processing subtracts the physical corner it calls the
  LPI corner, producing corner-relative coordinates in millimetres. These are not
  absolute NIfTI RAS and must not be used directly for voxel lookup.
- The adapter computes the LPI corner from image shape/orientation/affine, restores
  translation for FLOWCAT-relative to image-RAS conversion, and provides the
  inverse operation.
- VTK centreline/surface points appear intended as physical, RAS-like coordinates,
  but this must be verified with case landmarks and bounds. Do not infer the frame
  from file extension.

Required acceptance tests are: RAS-to-LPS round trip; FLOWCAT-relative-to-RAS
round trip; all eight image corners; LAS/LPS/RAS permutations; known landmark
overlay; point-in-image bounds; and VTK/NIfTI surface alignment in Slicer. The
current adapter unit fixtures exercise coordinate arithmetic, but the missing
native `sub-570` outputs prevent a representative end-to-end overlay.

## Missing native output chain

The local macOS environment does not provide the complete VMTK, VTK,
`torch_geometric`, and `torch_cluster` combination needed by native FLOWCAT
post-processing. A representative native attempt on `sub-570` detected 19
automatic endpoints, then failed on both original and cleaned surfaces because it
could not compute valid surface normals after all prescribed mesh-bottom retries.
The only exact persisted failure token is:

```text
flowcat_vmtk_automatic_centerline_failed_sub570_surface_normals
```

The original stderr/traceback was not retained. Therefore a more specific Python
exception is unable to verify and must not be reported as observed.

Because the chain stopped, the following are missing for all five current cases:

- validated VTK/STL surface;
- VMTK centreline VTK files;
- branch component and merged `branch_model.vtk`;
- `centerline_segments_array.npy`;
- native landmarks;
- `segments_graph_pred.pickle`;
- `local_graph.pickle`;
- vessel-type predictions and confidence;
- native radius/diameter/curvature/torsion/length/tortuosity; and
- access prediction.

File discovery or a Python import does not make these capabilities functional.

## Real `sub-570` fail-closed result

The finalized interface was exercised on `sub-570_acq-CTA_ct`. The exact original
run shell line was not retained, so commands below are reproducible interface
commands rather than claimed verbatim history. The named analysis stopped at
preflight with status 2 and this exact error:

```text
FileNotFoundError: Missing required FLOWCAT outputs: branch_model, centerline_segments_array, local_graph, segments_graph_pred
```

The milestone directory contains only `failure.json`, the regenerated
`flowcat_output_manifest.json`, and regenerated headless QC JSON/PNG. It contains
no labelled artery volume, shells, tissues, feature tables or analysis manifest.

The final manifest SHA-256 is
`8d6f3fb9c8dc67d71362392606b534f8121d86c83750c0e06d460847e99bd2a8`.
It hashes the valid binary segmentation and eight controlled local model payloads,
records the pinned clean FLOWCAT source, and flags all four absent native files.
Its null `case.cta_path` is expected for the separate non-executing inventory;
external CTA attachment occurs during successful full validation.

Final headless QC nevertheless succeeds honestly on the CTA plus binary mask.
`carotid_qc.json` SHA-256 is
`1d28b08f39671469399203f656d44fcfbf296151c4c6b87e776b4bbe870184a8`;
montage SHA-256 is
`a370afacfd909063f278efc1dd30df1d89019006fab3c92758b43cf4e55820e0`.
It reports canonical-RAS display voxel `[292,314,1335]`, 3,443,029 foreground
voxels (250.068 mL), 84 components, incomplete required outputs, no optional
overlays/no named results, and `qc_generated_pending_visual_review`. The montage
is explicitly superior whole-tree context, not named-carotid localization or a
manual anatomical pass.

## Fallback non-equivalence

`extract_skeleton_edt_centerlines.py` produces a 3D Lee voxel skeleton, samples an
anisotropic Euclidean distance transform for a technical radius, and splits
26-neighbour connectivity into technical paths. It enabled five diameter-adaptive
development runs, but differs from native FLOWCAT/VMTK in surface construction,
centreline location, branch topology, endpoints, radius definition, sampling,
anatomical labels, graph features, and coordinate products.

Fallback outputs must carry a source such as `skeleton_edt_fallback`, the native
failure reason, and `non_equivalent=true`. They must not be written with native
FLOWCAT filenames, passed off as `MaximumInscribedSphereRadius`, or silently used
when a native stage fails. Nearest-centreline Voronoi assignment is also a
technical approximation, not anatomical label propagation.

## Recommended execution environment

Use a pinned Linux/Ubuntu x86-64 environment or container first, because compatible
VMTK/VTK/PyTorch/PyG packages are more readily supportable there than in the
audited Apple Silicon environment. Record base-image digest, OS/architecture,
Python, VTK, VMTK, PyTorch, CUDA, PyG, nnU-Net, FLOWCAT commit, model hashes,
command, CPU/GPU/RAM, and deterministic settings.

An NVIDIA CUDA GPU is recommended for hybrid/high-resolution inference; VMTK is
primarily CPU/compiled geometry work. Measure GPU and CPU memory instead of
assuming requirements from device capacity. For ARM64 targets such as DGX Spark,
verify every compiled wheel/library and run the identical validation suite; if a
component is unavailable, use a documented x86-64 execution service/container.
Do not silently substitute an algorithm because of architecture.

The first Linux acceptance case should be `sub-570`, followed by the other four
pilot cases. Persist native stdout/stderr and inspect the resulting overlay in 3D
Slicer before cohort execution.

## Standard commands

Run commands from the repository root unless a `cd` is shown. Replace placeholder
paths only with controlled de-identified case paths.

### Create and install the local analysis environment

```bash
cd aorta_cta_radiomics
conda env create -f environment.yml
conda activate aorta-cta-radiomics
python -m pip install -e ../cta_common -e '.[dev,table]'
```

Native FLOWCAT requires a separate pinned Linux environment/container; the command
above does not install VMTK or make the native chain functional.

### Verify the checkout and model payload

```bash
git -C external/arterial_flowcat rev-parse --abbrev-ref HEAD
git -C external/arterial_flowcat rev-parse HEAD
git -C external/arterial_flowcat status --short

/opt/anaconda3/envs/totalseg-mac/bin/python \
  subprojects/studies/anterior_circulation_supraaortic/scripts/prepare_flowcat_models.py \
  --profile hybrid --include-vessel-labelling --verify-only
```

Expected commit is
`1991ca5b238d7c868c9c3a0a390057bc89aee559`, and tracked status must be clean.
Remove `--verify-only` only when local chunk reconstruction is intended.

### Run the current segmentation wrapper

```bash
/opt/anaconda3/envs/totalseg-mac/bin/python \
  subprojects/studies/anterior_circulation_supraaortic/scripts/run_flowcat_segmentation.py \
  --profile fast-lowres --case-id sub-570 --fail-fast
```

This command generates the current binary segmentation only. It does not generate
native surfaces, centrelines, graphs, or labels.

### Run upstream native analysis in the supported Linux environment

```bash
python external/arterial_flowcat/perform_analysis.py \
  --case_dir /controlled/path/sub-570_acq-CTA_ct \
  --cta_nifti_path /controlled/path/sub-570_acq-CTA_ct/cta.nii.gz \
  --mode extracranial_vessels \
  --fast_segmentation \
  --skip_access_prediction
```

This is the upstream entry point, not a verified successful command on the current
Mac. Do not use `--skip_centerline_extraction`, `--skip_branching`,
`--skip_vessel_labelling`, or `--skip_feature_extraction` for the native adapter
milestone.

### Create a non-executing output manifest

```bash
cd aorta_cta_radiomics
python -m aorta_cta_radiomics.carotid_cli manifest \
  --case-dir ../outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct \
  --case-id sub-570_acq-CTA_ct \
  --flowcat-repo ../external/arterial_flowcat \
  --output ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/flowcat_output_manifest.json
```

This command is safe for an untrusted case because it does not deserialize
pickle/object-NPY content. It always writes the inventory, includes controlled
model hashes, and returns 2 when required outputs are missing. The current 570
manifest therefore records four missing native products rather than success.

### Validate a complete trusted local native case

Only after verifying that the case was generated locally from the clean pinned
checkout and recording file hashes:

```bash
cd aorta_cta_radiomics
python - <<'PY'
from pathlib import Path
from aorta_cta_radiomics.flowcat_adapter import validate_flowcat_case

case_dir = Path("/controlled/path/sub-570_acq-CTA_ct")
case = validate_flowcat_case(
    case_dir,
    case_id="sub-570",
    cta_path="/controlled/input/sub-570_acq-CTA_ct.nii.gz",
    trusted_source=True,
    load_vtk=True,
)
print(case.manifest.missing_required_outputs)
print(len(case.branches))
PY
```

Do not run that command on downloaded, emailed, shared, or otherwise untrusted
pickle/NPY files.

### Run the fail-closed carotid analysis

Only after the manifest is complete and all unsafe artifacts are confirmed as a
controlled local output:

```bash
cd aorta_cta_radiomics
python -m aorta_cta_radiomics.carotid_cli run \
  --case-dir /controlled/path/sub-570_acq-CTA_ct \
  --case-id sub-570_acq-CTA_ct \
  --cta /controlled/input/sub-570_acq-CTA_ct.nii.gz \
  --flowcat-repo ../external/arterial_flowcat \
  --config config/carotid_perivascular.yaml \
  --outdir /controlled/output/sub-570_acq-CTA_ct \
  --trusted-flowcat-artifacts
```

Optional binary masks are supplied with `--vein-mask`, `--bone-mask`,
`--airway-mask`, `--thyroid-mask`, `--muscle-mask`, `--uncertain-mask`, and
`--high-density-mask`; each must match CTA geometry. Missing native products or
unsafe trust state writes `failure.json` and returns 2. The current real 570 case
fails at preflight and creates no named measurement.

### Run the development adaptive analysis

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  subprojects/studies/anterior_circulation_supraaortic/scripts/run_adaptive_vessel_burden_v2.py \
  --case-id sub-570 --fail-fast
```

This remains broad-tree development output until named native branches and formal
visual QC are available.

### Run tests

```bash
cd aorta_cta_radiomics
PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q -p no:cacheprovider
```

Current full snapshot: `133 passed` (verified in 3.33 s on this host). The focused
adapter/artery-volume/carotid/output/perivascular suite is 67 passed. Neither is a
substitute for a successful native real case.

### Generate headless QC for incomplete or complete output

```bash
cd aorta_cta_radiomics
python -m aorta_cta_radiomics.carotid_qc \
  --case-id sub-570_acq-CTA_ct \
  --cta /Volumes/DICOM5/daylightbids/sub-570_acq-CTA_ct.nii.gz \
  --segmentation ../outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres/sub-570_acq-CTA_ct/extracranial_vessels/segmentation.nii.gz \
  --flowcat-manifest ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/flowcat_output_manifest.json \
  --outdir ../outputs/anterior_circulation_supraaortic/carotid_perivascular_milestone1/sub-570_acq-CTA_ct/qc
```

When successful named volumes exist, optional `--labelled-artery`, `--shell`,
`--tissue`, and `--calcium` arguments add display overlays. They do not ask the QC
command to infer missing results. Review remains pending until recorded manually.

### Generate a Slicer QC selection for `sub-570`

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  aorta_cta_radiomics/scripts/qc_slicer.py \
  --manifest subprojects/studies/anterior_circulation_supraaortic/manifests/daylight_extended_pilot5_adaptive.csv \
  --outputs-root outputs/anterior_circulation_supraaortic/flowcat_2026_03/fast-lowres \
  --outdir outputs/anterior_circulation_supraaortic/qc/sub-570 \
  --case-id sub-570 \
  --anatomy carotid --task arterial --task calcium --task fat \
  --reviewer reviewer-id
```

Inspect the generated case script/selection before adding `--open-slicer`. A
generated script is not a completed review; record reviewer, flags, overlays, and
adjudication in the QC log.

## Narrow next task

Reproduce the pinned native FLOWCAT chain for `sub-570` in a supported Linux
environment, generate all five default required artifacts, validate them against
the external original CTA, then run the existing deterministic carotid core and
review CTA/surface/centreline/label-volume/one CCA-ICA path in Slicer. Do not
proceed to cohort-level artery-specific PVAT or calcium sums until that case passes
the output contract and visual overlay; carotid-core calcium is not yet implemented.
