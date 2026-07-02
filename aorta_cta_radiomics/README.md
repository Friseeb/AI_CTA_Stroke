# aorta_cta_radiomics

A CPU-first Python research pipeline for aorta-focused stroke CTA / extended CTA analysis. The pipeline starts from a CTA NIfTI image and an existing aorta segmentation, then produces reproducible aorta QC, calcium, periaortic fat, wall/lumen, wall thickness, protrusion/ulcer-candidate, and radiomics outputs.

This is not a plaque segmentation model. It does not claim histologic plaque boundaries. The intended method is to use validated/open anatomic segmentations, HU thresholds, local wall/lumen geometry, and reproducible ROIs for downstream research modeling.

## Scope

- CTA or extended CTA in NIfTI format.
- Existing aorta mask from VISTA, TotalSegmentator, or manual/corrected 3D Slicer export.
- Aorta-only analysis: ascending aorta, arch, descending thoracic aorta, and abdominal aorta when present.
- macOS / Apple Silicon friendly. Core pipeline is CPU-compatible and avoids CUDA-only dependencies.

## Maintained Outputs

- Mask QC metrics.
- Cleaned aorta mask.
- Approximate centerline and slice-based geometry.
- Calcium candidate masks and calcium omics.
- Periaortic fat masks and fat omics.
- Wall/lumen masks derived from aorta plus periaortic fat support.
- Wall thickness maps, including a `>4 mm` TEE-analogue research flag.
- Protrusion-like and ulcer-like candidate maps from wall/lumen geometry.
- PyRadiomics features.
- Long and wide CSVs for statistical modeling.
- 3D Slicer QC loading scripts.

## Install

```bash
export AORTA_REPO=/path/to/aorta_cta_radiomics
conda env create -f "$AORTA_REPO/environment.yml"
conda activate aorta-cta-radiomics
pip install -e "$AORTA_REPO"
```

Optional mesh or MONAI dependencies are intentionally separate:

```bash
conda install -c conda-forge pyvista trimesh
pip install monai torch
```

## Inputs

Single-case runs require:

- `image_path`: CTA NIfTI image.
- `aorta_mask_path`: aorta segmentation NIfTI in the same space or resamplable to image space.
- `case_id`: stable case identifier.

Batch manifests require:

```csv
case_id,image_path,aorta_mask_path,metadata_path,optional_notes
CASE001,/path/to/cta.nii.gz,/path/to/aorta_mask.nii.gz,/path/to/cta.json,
```

## Metadata Eligibility

For large batches, enable the neuro CTA filter so the pipeline only processes rows whose metadata looks like brain/neck, stroke, or hyperacute CTA:

```bash
--metadata-filter neuro-cta
```

The filter reads:

- an explicit manifest column such as `metadata_path`, `json_path`, or `sidecar_path`;
- a BIDS-style JSON sidecar next to the image, for example `sub-001_acq-CTA_ct.json`;
- manifest metadata columns such as `SeriesDescription`, `ProtocolName`, `StudyDescription`, `BodyPartExamined`, or similar description/protocol/acquisition fields.

A case is kept when metadata contains CTA/angiography language and a neuro/stroke/head-neck term. Chest, coronary, pulmonary embolism, abdomen/pelvis, runoff, and similar non-target protocols are skipped unless neuro terms are also present. Skipped and kept cases are written to `metadata_eligibility.csv` in the run output directory.

## Run One Case

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m aorta_cta_radiomics.cli run-single \
  --image /path/to/cta.nii.gz \
  --aorta-mask /path/to/aorta_mask.nii.gz \
  --case-id CASE001 \
  --outdir "$AORTA_REPO/outputs/case_test" \
  --config "$AORTA_REPO/configs/calcium_dynamic_500hu.yaml"
```

## Run Batch

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m aorta_cta_radiomics.cli run-batch \
  --manifest /path/to/manifest.csv \
  --outdir "$AORTA_REPO/outputs/batch_run" \
  --config "$AORTA_REPO/configs/calcium_dynamic_500hu.yaml"
```

## Staged Batch

For larger cohorts, use the staged runner so expensive steps can be separated and parallelized by stage:

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  "$AORTA_REPO/scripts/run_manifest_staged.py" \
  --manifest /path/to/manifest.csv \
  --outdir "$AORTA_REPO/outputs/aorta_batch_run" \
  --stages base,calcium,fat-wall,protrusions,wall-thickness,radiomics \
  --config "$AORTA_REPO/configs/calcium_dynamic_500hu.yaml" \
  --metadata-filter neuro-cta \
  --base-workers 2 \
  --calcium-workers 2 \
  --fat-wall-workers 1 \
  --protrusion-workers 1 \
  --wall-thickness-workers 2 \
  --radiomics-workers 1 \
  --skip-existing
```

Recommended worker counts on a local Apple Silicon laptop are conservative: VISTA or other segmentation inference should usually be `1` worker; base/calcium/wall-thickness can often use `2`; fat-wall and protrusion stages are memory-sensitive and should start at `1`; PyRadiomics is often the bottleneck and should be split or run separately when needed.

## Watched Batch With ntfy

For long runs, use the maintained watchdog runner. It launches the staged runner as a subprocess, forces the neuro CTA metadata filter by default, writes runner/watchdog logs, sends ntfy start/heartbeat/stall/final notifications, and keeps `metadata_eligibility.csv` plus `stage_status.csv` as the audit trail.

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python \
  "$AORTA_REPO/scripts/run_batch_with_watchdog.py" \
  --manifest /path/to/manifest.csv \
  --outdir "$AORTA_REPO/outputs/aorta_batch_run" \
  --config "$AORTA_REPO/configs/calcium_dynamic_500hu.yaml" \
  --stages vista,base,calcium,fat-wall,protrusions,wall-thickness,radiomics \
  --metadata-filter neuro-cta \
  --run-label slaobids-aorta-full \
  --ntfy-topic auto \
  --nv-device auto \
  --vista-workers 1 \
  --base-workers 2 \
  --calcium-workers 2 \
  --fat-wall-workers 1 \
  --protrusion-workers 1 \
  --wall-thickness-workers 2 \
  --radiomics-workers 1 \
  --radiomics-split-by-region \
  --radiomics-region-workers 4 \
  --notify-every-minutes 30 \
  --stall-minutes 90
```

With `--ntfy-topic auto`, the topic is derived from the run label, for example `aorta-cta-slaobids-aorta-full`. The runner prints the subscribe URL and writes it to `ntfy_topic.txt` inside the output directory. If `--ntfy-topic` is omitted, the runner still works and only writes local logs under `outputs/aorta_batch_run/logs/batch_watchdog/`.

Check progress from another terminal while the batch is running:

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m aorta_cta_radiomics.batch_progress \
  --outdir "$AORTA_REPO/outputs/aorta_batch_run" \
  --watch \
  --interval-seconds 30
```

The progress command is read-only. It summarizes `metadata_eligibility.csv`, `stage_status.csv`, latest file activity, failed cases, and the tail of `logs/batch_watchdog/staged_runner.log`.

For a richer local browser monitor with stage ETAs, artifact counts, process CPU/RSS memory, process elapsed time, failures, latest activity, and log tail:

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m aorta_cta_radiomics.batch_progress \
  --outdir "$AORTA_REPO/outputs/aorta_batch_run" \
  --serve \
  --port 8765 \
  --interval-seconds 30
```

Open `http://127.0.0.1:8765/`. The same monitor also exposes machine-readable status at `http://127.0.0.1:8765/data.json`.

The ntfy watchdog heartbeat/stall/final messages use a compact version of this monitor summary: current stage, ETA, process count, CPU/RSS memory, artifact counts, latest activity, stage progress, and recent failures.

## 3D Slicer QC

Build a Slicer QC selection from an existing run:

```bash
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m aorta_cta_radiomics.qc_slicer \
  --manifest /path/to/manifest.csv \
  --outputs-root "$AORTA_REPO/outputs/aorta_batch_run" \
  --outdir "$AORTA_REPO/outputs/qc_slicer" \
  --anatomy aorta \
  --task segmentation \
  --task calcification \
  --task adipose_tissue \
  --task wall_from_fat \
  --task wall_thickness \
  --task lumen_protrusion
```

The QC script writes one Slicer Python loader per selected case plus structured CSV/JSON review logs. Use `--open-slicer` only when you want it to launch 3D Slicer directly.

## Output Layout

```text
outputs/
  features/
    case_level_features.csv
    calcification_features.csv
    calcium_omics_features.csv
    fat_omics_features.csv
    lumen_protrusion_summary_features.csv
    wall_from_fat_features.csv
    wall_thickness_summary.csv
    wall_thickness_summary_with_thresholds.csv
    radiomics_features.csv
    modeling_wide_features.csv
  qc/
    qc_summary.csv
  masks/
    <case_id>/
  figures/
    <case_id>/
```

Generated outputs are ignored by git. Keep reusable code, configs, and documentation in the repository; regenerate patient-specific outputs as needed.

## Method Notes

### Calcium

The default research config uses a 500 HU seed for dynamic calcium candidates. CTA contrast can confound calcium, so calcium outputs should be interpreted as candidate burden and reviewed in QC. The dynamic candidate method is designed to keep calcium tied to the aortic wall region and reduce obvious non-aortic contrast contamination.

### Fat and Wall

Periaortic fat is defined from HU-thresholded adipose voxels near the aorta, with maintained local layers focused on 0-2 mm and 2-5 mm. The wall-from-fat stage estimates a closed outer wall support from the non-continuous periaortic fat layer and combines it with the aorta/lumen estimate to create inspectable wall and lumen masks.

### Protrusion and Ulcer Candidates

The maintained protrusion/ulcer stage uses the wall/lumen domain, with calcium incorporated into wall support so calcified wall is not mislabeled as an ulcer-like defect. Candidate maps are geometric research markers for review, not diagnoses.

### Wall Thickness

Wall thickness maps are research estimates from the derived lumen/wall masks. The `>4 mm` label is a TEE-inspired risk analogue for QC and modeling; it should not be presented as a validated CTA equivalent without further validation.

## Tests

```bash
cd "$AORTA_REPO"
/opt/anaconda3/envs/aorta-cta-radiomics/bin/python -m pytest -q
```

## Repository Rules

- No hard-coded patient paths in committed configs or docs.
- No generated outputs committed.
- No one-off patient scripts in the maintained tree.
- Prefer config-driven reusable stages over ad hoc scripts.
- Keep experimental model work outside the maintained aorta pipeline until it has a stable interface, tests, and documentation.
