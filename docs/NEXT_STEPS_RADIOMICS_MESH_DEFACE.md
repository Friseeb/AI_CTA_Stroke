# Next Steps: Radiomics, Meshes, PyDeface

## 1) PyRadiomics (IBSI-oriented batch)

Use the new script:

`scripts/run_pyradiomics_ibsi_batch.py`

Install prerequisite if missing:

```bash
<PYTHON_BIN_DIR>/pip install pyradiomics
```

What it enforces:
- mask/image geometry alignment (nearest-neighbor mask resampling)
- binary mask cleanup (+ optional largest component)
- isotropic resampling (`--isotropic-mm`, default `1.0`)
- CT HU clipping (default `-1024,3071`)
- fixed gray-level discretization (`--bin-width`, default `25`)

Example for all three regions:

```bash
<PYTHON_BIN_DIR>/python scripts/run_pyradiomics_ibsi_batch.py \
  --mask-suffix laa_nudf \
  --mask-suffix left_atrium_highres \
  --mask-suffix aorta_highres \
  --isotropic-mm 1.0 \
  --bin-width 25 \
  --save-preprocessed \
  --output-csv <BIDS_ROOT>/derivatives/radiomics/pyradiomics_ibsi_batch.csv
```

If you want a subset:

```bash
<PYTHON_BIN_DIR>/python scripts/run_pyradiomics_ibsi_batch.py \
  --subject 642 --subject 646 \
  --mask-suffix laa_nudf --mask-suffix left_atrium_highres --mask-suffix aorta_highres
```

## 2) Segmentation to Mesh conversion

Use the existing script (now with batch mesh mode):

`scripts/run_laa_shape_descriptors.py`

Capabilities:
- marching cubes on binary segmentations
- optional largest-component filtering
- optional smoothing/decimation
- outputs `stl`, `ply`, or `vtk`

Example:

```bash
<PYTHON_BIN_DIR>/python scripts/run_laa_shape_descriptors.py \
  --batch-mask-root <BIDS_ROOT>/derivatives/nudf_la \
  --output-dir <BIDS_ROOT>/derivatives/meshes \
  --mask-suffix laa_nudf \
  --mask-suffix left_atrium_highres \
  --mask-suffix aorta_highres \
  --mesh-format stl
```

## 3) PyDeface batch

Use the existing batch deface script with backend switch:

`scripts/run_cta_deface_dl_batch.py --backend pydeface`

Install prerequisite if missing:

```bash
python3 -m pip install pydeface
```

Example:

```bash
python3 scripts/run_cta_deface_dl_batch.py \
  --backend pydeface \
  --input-dir <BIDS_ROOT> \
  --glob 'sub-*_acq-CTA_ct.nii.gz' \
  --output-dir <BIDS_ROOT>/derivatives/defaced_pydeface \
  --pydeface-bin pydeface
```

Environment checks:

```bash
<PYTHON_BIN_DIR>/python scripts/run_pyradiomics_ibsi_batch.py --check-env
python3 -c "import shutil,sys; p=shutil.which('pydeface'); print(p); sys.exit(0 if p else 1)"
```

## 4) Suggested execution order

1. Finish/verify DICOM->NIfTI conversion consistency.
2. Re-run LA/LAA/Aorta segmentation where needed.
3. Run `run_cta_deface_dl_batch.py --backend pydeface` (if pydeface is installed).
4. Run `run_pyradiomics_ibsi_batch.py`.
5. Run `run_laa_shape_descriptors.py` in batch mesh mode.
