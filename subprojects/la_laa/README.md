# LA/LAA Substudy

This subproject defines the Left Atrium / Left Atrial Appendage substudy scope
within `AI_CTA_Stroke`.

## Scope

- LAA-focused segmentation outputs (NUDF / NV-Segment-CT)
- LA + LAA mesh generation
- LA/LAA relational shape metrics
- Report generation for LA/LAA shape/radiomics interpretation

## Canonical Data Flow

1. Input: defaced CTA NIfTI (`<BIDS_ROOT>/derivatives/defaced/...`)
2. Segment: LA/LAA masks in `derivatives/nudf_la/` and/or `derivatives/nv_segment_ct_laa/`
3. Mesh export: LA/LAA meshes into `derivatives/shape_meshes/`
4. Batch metrics: LA-LAA geometric/relational metrics CSV
5. Report: HTML summary with optional integrated clustering

## Scripts Used

| Script | Purpose |
|--------|---------|
| `scripts/run_cardiac_ct_explorer_nudf_only.py` | Run TotalSegmentator + CardiacCTExplorer NUDF; extract LA and LAA masks |
| `scripts/run_cardiac_ct_explorer_laa.py` | Alternative CardiacCTExplorer LAA-only runner |
| `scripts/run_nv_segment_ct_laa.py` | Alternative LAA segmentation via NV-Segment-CT |
| `scripts/run_laa_shape_descriptors.py` | Generate surface meshes from binary masks (marching cubes) |
| `scripts/run_la_laa_metrics_batch.py` | Compute LA/LAA relational metrics from mesh pairs |
| `scripts/generate_la_laa_shape_report.py` | Generate HTML report with figures and clustering |
| `scripts/build_radiomics_manifest_nudf_la.py` | Build PyRadiomics manifest for LA/aorta radiomics |
| `run_batch_la_metrics.py` | Cohort-level orchestration script (extract → mesh → metrics) |

## Protocol References

- `docs/protocols/laa_highres_dataset_setup.md`
- `docs/NEXT_STEPS_RADIOMICS_MESH_DEFACE.md`

---

## Directory Structure

The pipeline imposes a specific folder layout under `BIDS_ROOT`. Understanding
this layout is essential for troubleshooting missing-file errors.

```
BIDS_ROOT/
├── derivatives/
│   │
│   ├── defaced/                             # Step 0 — defaced CTA inputs
│   │   ├── sub-001_acq-CTA_ct_defaced.nii.gz
│   │   └── ...
│   │
│   ├── nudf_la/                             # Step 1 — segmentation outputs
│   │   ├── sub-001/
│   │   │   │
│   │   │   ├── cardiac_ct_explorer/         # Created by run_cardiac_ct_explorer_nudf_only.py
│   │   │   │   │                            # (folder name is fixed by the script)
│   │   │   │   ├── TotalSegmentator/        # Created by TotalSegmentator (library-imposed)
│   │   │   │   │   └── sub-001_acq-CTA_ct_defaced/   # Named after the input scan
│   │   │   │   │       └── heartchambers_highres.nii.gz  # Multi-label heart volume ← key file
│   │   │   │   └── [other CardiacCTExplorer outputs]
│   │   │   │
│   │   │   ├── sub-001_laa_nudf.nii.gz      # LAA binary mask  (from CardiacCTExplorer)
│   │   │   ├── sub-001_left_atrium_highres.nii.gz   # LA binary mask  (label 2 extracted from heartchambers_highres)
│   │   │   ├── sub-001_aorta_highres_ts.nii.gz      # Aorta from TotalSegmentator (optional)
│   │   │   ├── sub-001_aorta_highres_monai.nii.gz   # Aorta from MONAI/VISTA3D    (optional)
│   │   │   └── sub-001_aorta_highres.nii.gz         # Canonical aorta (copied from chosen source)
│   │   │
│   │   ├── sub-002/
│   │   │   └── [same structure]
│   │   │
│   │   ├── qc_summary.csv                   # QC table written after batch run
│   │   ├── qc_summary_live.csv              # Incremental per-case QC (written during run)
│   │   └── _logs/
│   │       ├── sub-001.log                  # Subprocess stdout/stderr per case
│   │       └── ...
│   │
│   ├── shape_meshes/                        # Step 2 — surface meshes
│   │   ├── sub-001/
│   │   │   ├── surfaces/                    # Primary mesh output from run_laa_shape_descriptors.py
│   │   │   │   ├── sub-001_left_atrium_highres_laa_surface.vtk
│   │   │   │   └── sub-001_laa_nudf_laa_surface.vtk
│   │   │   │
│   │   │   │   # run_la_laa_metrics_batch.py looks for meshes at the CASE ROOT, not in surfaces/.
│   │   │   │   # run_batch_la_metrics.py copies them up automatically (copy_to_root step).
│   │   │   ├── sub-001_left_atrium_highres_laa_surface.vtk   # copied from surfaces/
│   │   │   └── sub-001_laa_nudf_laa_surface.vtk              # copied from surfaces/
│   │   │
│   │   ├── sub-002/
│   │   │   └── [same structure]
│   │   │
│   │   └── la_laa_metrics_batch.csv         # Step 3 output — relational metrics for all cases
│   │
│   └── nv_segment_ct_laa/                   # Optional alternative LAA source
│       ├── sub-001_laa108.nii.gz
│       └── ...
```

### Key naming conventions

| Token | Meaning | Example |
|-------|---------|---------|
| `sub-XXX` | Subject/case identifier | `sub-101` |
| `<CASE_ID>_acq-CTA_ct_defaced` | Defaced CTA file stem | `sub-101_acq-CTA_ct_defaced` |
| `<CASE_ID>_left_atrium_highres` | LA binary mask stem | `sub-101_left_atrium_highres` |
| `<CASE_ID>_laa_nudf` | LAA binary mask stem (NUDF source) | `sub-101_laa_nudf` |
| `<CASE_ID>_<mask>_laa_surface` | Surface mesh stem | `sub-101_left_atrium_highres_laa_surface` |

### heartchambers_highres label map

`heartchambers_highres.nii.gz` is a multi-label volume produced by TotalSegmentator
(task `heartchambers_highres`, model ID 301). Label assignments:

| Label | Structure | Extracted as |
|-------|-----------|--------------|
| 1 | `heart_myocardium` | — (not extracted by default) |
| **2** | `heart_atrium_left` | `<CASE_ID>_left_atrium_highres.nii.gz` |
| 3 | `heart_ventricle_left` | — |
| 4 | `heart_atrium_right` | — |
| 5 | `heart_ventricle_right` | — |
| **6** | `aorta` | `<CASE_ID>_aorta_highres.nii.gz` |
| 7 | `pulmonary_artery` | — |

> **FOV exclusion:** eCTA scans focused on the head/neck may not cover the heart.
> TotalSegmentator still produces `heartchambers_highres.nii.gz` in these cases,
> but label 2 will contain zero or near-zero voxels. The pipeline detects this
> automatically: any case where label 2 yields fewer than **1 000 voxels** is
> flagged as `skip_la_fov` in `qc_summary.csv` and excluded from downstream
> mesh generation and metrics. Adjust `MIN_LA_VOXELS` in
> `run_daylightbids_nudf_la_batch.py` or `run_batch_la_metrics.py` if needed.

### Why meshes are copied from `surfaces/` to the case root

`run_laa_shape_descriptors.py` writes meshes into a `surfaces/` subfolder.
`run_la_laa_metrics_batch.py` searches for meshes directly at the **case root**
(e.g. `shape_meshes/sub-001/sub-001_left_atrium_highres_laa_surface.vtk`).

`run_batch_la_metrics.py` bridges this gap with an explicit copy step
(`copy_to_root`). If you call `run_la_laa_metrics_batch.py` directly without
going through `run_batch_la_metrics.py`, you must copy or symlink the mesh
files from `surfaces/` to the case root first, or use the
`--la-suffix` / `--laa-suffix` arguments to point at the full path.

---

## Minimal Run Skeleton

```bash
# 1) LAA segmentation (NUDF) — also runs TotalSegmentator heartchambers_highres
python <PROJECT_ROOT>/scripts/run_cardiac_ct_explorer_nudf_only.py \
  --input <BIDS_ROOT>/derivatives/defaced/<CASE_ID>_acq-CTA_ct_defaced.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/cardiac_ct_explorer \
  --laa-output <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \
  --run-totalseg

# 1b) Extract LA mask (label 2) from heartchambers_highres
#     (run_batch_la_metrics.py does this automatically; manual alternative below)
python - <<'EOF'
import nibabel as nib, numpy as np
from pathlib import Path
hc = Path("<BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/cardiac_ct_explorer/TotalSegmentator/<CASE_ID>_acq-CTA_ct_defaced/heartchambers_highres.nii.gz")
img = nib.load(hc); la = (img.get_fdata() == 2).astype("uint8")
out = hc.parents[3] / "<CASE_ID>_left_atrium_highres.nii.gz"
nib.save(nib.Nifti1Image(la, img.affine, img.header), out)
EOF

# 2) Mesh + descriptors (LA and LAA)
python <PROJECT_ROOT>/scripts/run_laa_shape_descriptors.py \
  --input <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_left_atrium_highres.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/shape_meshes/<CASE_ID>

python <PROJECT_ROOT>/scripts/run_laa_shape_descriptors.py \
  --input <BIDS_ROOT>/derivatives/nudf_la/<CASE_ID>/<CASE_ID>_laa_nudf.nii.gz \
  --output-dir <BIDS_ROOT>/derivatives/shape_meshes/<CASE_ID>

# 2b) Copy meshes from surfaces/ to case root (required by metrics script)
cp <BIDS_ROOT>/derivatives/shape_meshes/<CASE_ID>/surfaces/*.vtk \
   <BIDS_ROOT>/derivatives/shape_meshes/<CASE_ID>/

# 3) Batch metrics
python <PROJECT_ROOT>/scripts/run_la_laa_metrics_batch.py \
  --mesh-root <BIDS_ROOT>/derivatives/shape_meshes \
  --out-csv <BIDS_ROOT>/derivatives/shape_meshes/la_laa_metrics_batch.csv

# 4) Report
python <PROJECT_ROOT>/scripts/generate_la_laa_shape_report.py \
  --metrics-csv <BIDS_ROOT>/derivatives/shape_meshes/la_laa_metrics_batch.csv \
  --mesh-root <BIDS_ROOT>/derivatives/shape_meshes \
  --output-html <BIDS_ROOT>/derivatives/shape_meshes/la_laa_shape_report.html
```

### Batch shortcut (cohort-level)

For processing a full cohort, `run_batch_la_metrics.py` combines steps 1b–3
into a single script. Edit the `CONFIGURATION` block at the top of the file
(paths, subject list, label IDs) before running:

```bash
python <PROJECT_ROOT>/run_batch_la_metrics.py
```

QC status values written to `qc_summary.csv`:

| Status | Meaning |
|--------|---------|
| `ok` | Case processed successfully |
| `skipped` | All outputs already present (use `--force` to reprocess) |
| `skip_la_fov` | LA label 2 below `MIN_LA_VOXELS`; heart not in CT field of view |
| `skip_missing_mesh_pair` | LA or LAA mesh not found at case root |
| `failed` | Unexpected error — see `message` column and `_logs/<case_id>.log` |
