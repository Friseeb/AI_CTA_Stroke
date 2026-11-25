# Whole-heart segmentation (HolOrama CT model)

One labelmap with **14 cardiac structures** on a contrast CTA: the four chambers, LV myocardium, LAA,
aorta, pulmonary arteries and veins, SVC, IVC, coronary arteries, and **epicardial and pericardial fat**.
It runs the CT segmentation model shipped with [HolOrama](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama)
(AI in Cardiovascular Medicine, Inselspital Bern; MIT) as a headless, batchable step of this repository.

| id | structure (file name) | HolOrama name | id | structure | HolOrama name |
|---:|---|---|---:|---|---|
| 1 | `coronary_arteries` | Coronary Artery | 8 | `pulmonary_arteries` | Pulmonary Arteries |
| 2 | `lv_myocardium` | LVM | 9 | `pericardial_fat` | Precardial / Pericardial Fat (outside the pericardium) |
| 3 | `left_atrium` | LA (appendage excluded) | 10 | `epicardial_fat` | Epicardial Fat |
| 4 | `left_ventricle` | LV | 11 | `pulmonary_veins` | Pulmonary Vein |
| 5 | `right_atrium` | RA | 12 | `superior_vena_cava` | Superior Vena Cava |
| 6 | `right_ventricle` | RV | 13 | `inferior_vena_cava` | Inferior Vena Cava |
| 7 | `aorta` | Aorta | 14 | `left_atrial_appendage` | LA Appendage |

Why it is here: it is the only open model in this repository that gives the **LAA as its own label**
(a third LAA source next to NUDF and VISTA3D / NV-Segment-CT) and **epicardial vs pericardial fat** in one
pass, and it is light (14 MB ONNX, no torch, CPU-only is fine). It complements TotalSegmentator
`heartchambers_highres` (used by `subprojects/valve_calcium/`); label 3 here excludes the appendage,
TotalSegmentator's `heart_atrium_left` includes it.

## What it does

1. Reads the CTA (any orientation) and reorients it to the model's `LAS`.
2. **Crops to the heart** from a binary mask (default: TotalSegmentator `total --fast -rs heart`, ~1 min
   on CPU) plus a margin (15 mm). Stroke CTAs run from the aortic arch to the vertex; the model was trained on
   cardiac fields of view, and a full head-neck-chest volume costs ~150 tiles / 30 GB for nothing.
3. Runs nnU-Net 3d_fullres inference on onnxruntime (`nnunet_onnx.py`: crop-to-nonzero, CT normalisation,
   resampling to 0.5 x 0.35 x 0.35 mm, Gaussian sliding window with step 0.75, argmax; no test-time mirroring,
   the model was trained without it).
4. Writes everything back **on the CTA's own grid** (exact reorientation, no resampling of the labels).

Outputs per case in `<out-root>/<case>/`:

| file | content |
|---|---|
| `<case>_holorama_heart_seg.nii.gz` | uint8 labelmap, values as in the table |
| `structures/<structure>.nii.gz` | one binary mask per structure (TotalSegmentator-style; `--no-structures` to skip) |
| `<case>_holorama_heart_volumes.csv` | long format: `volume_ml` per structure + `touches_scan_inferior/superior/inplane` (cut by the scan) and `touches_crop_box` (crop margin too small) |
| `<case>_holorama_heart_labels.json` | ids, names, colours (HolOrama "Colorful" preset) for viewers |
| `<case>_holorama_heart_qc.png` | axial / coronal / sagittal through the segmentation centroid |
| `<case>_holorama_heart_manifest.json` | provenance: input sha256, model sha256 + plans, crop box, device, timings, package versions |
| `total_fast/heart.nii.gz`, `ts_total_fast.log` | the TotalSegmentator crop, when the driver made it |
| `case_status.json` | step timings / `ok` / `failed` with the error |

## Setup

```bash
# model (21 MB, MIT; pinned commit + sha256 in fetch_holorama_weights.py) -> external/holorama/
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
$PY subprojects/heart_segmentation/fetch_holorama_weights.py
# runtime: onnxruntime (CPU; on macOS CoreML cannot run this model: 5-D InstanceNormalization)
$PY -m pip install onnxruntime            # or onnxruntime-gpu for CUDA 12 on Linux / Windows
```

Python needs numpy, scipy, scikit-image, SimpleITK, onnxruntime (matplotlib for the QC PNG). The
`aorta-cta-radiomics` env on this Mac has them all; the `laa-pipeline` env from `envs/` has everything but
onnxruntime.

## Run

```bash
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
# one case; heart crop from TotalSegmentator (run on the fly, or reused from <out>/<case>/total_fast/)
$PY subprojects/heart_segmentation/run_heart_segmentation_case.py \
    --cta outputs/daylight_evt_extended/<case_id>.nii.gz --out-root outputs/heart_segmentation
# reuse the heart crops the valve-calcium pipeline already made for the 57 staged EVT cases
$PY subprojects/heart_segmentation/run_heart_segmentation_batch.py \
    --input-dir outputs/daylight_evt_extended --out-root outputs/heart_segmentation \
    --crop-mask 'outputs/valve_calcium/{case_id}/total_fast/heart.nii.gz'
# any other crop mask / no crop at all
$PY subprojects/heart_segmentation/run_heart_segmentation_case.py --cta <cta> --out-root <out> --crop-mask <mask.nii.gz>
$PY subprojects/heart_segmentation/run_heart_segmentation_case.py --cta <cta> --out-root <out> --no-crop --allow-large
# look at a case in 3D Slicer (macOS: must be `open -na`)
HEART_CASE=<case_id> open -na /Applications/Slicer.app --args \
    --python-script subprojects/heart_segmentation/slicer/load_heart_segmentation_case.py
# tests (no weights needed; one test is skipped until they are fetched)
$PY -m pytest subprojects/heart_segmentation/tests -q
```

Each step is skipped when its outputs exist (`--force` to redo the segmentation, batch `--redo-ok` to revisit
cases marked ok). Run from locally staged copies only (external-drive I/O rule); never from `/Volumes`.

Measured on this Mac (M-series, 16 cores, CPU, sub-112, 0.45 x 0.45 x 0.25 mm, heart crop + 15 mm =
96 x 131 x 168 mm): 12 tiles at ~14 s each, **201 s per case** end to end (6 s preprocessing, 166 s network, 15 s
resampling back, 10 s writing). **Peak resident memory 26 GB**: ~2.4 GB of arrays plus ~23 GB of onnxruntime
activations for one 128 x 256 x 224 tile, a fixed cost a smaller crop does not reduce. Plan for a 32 GB machine at
minimum on CPU. A CUDA GPU (`--device cuda`, optionally `--precision fp16`) is 10 to 20x faster.

### Cross-check against TotalSegmentator (sub-112, `heartchambers_highres`)

| structure | HolOrama mL | TotalSegmentator mL | Dice |
|---|---:|---:|---:|
| left atrium (+ LAA) | 56.8 | 54.6 | 0.88 |
| left ventricle | 32.2 | 33.3 | 0.85 |
| LV myocardium | 47.9 | 45.6 | 0.88 |
| right ventricle | 24.6 | 27.7 | 0.87 |
| pulmonary arteries | 68.1 | 65.0 | 0.93 |
| right atrium | 42.7 | 16.0 | 0.52 |
| aorta | 87.4 | 162.7 | 0.68 |

Both models were run on the same non-gated stroke CTA whose field of view ends through the heart. The chamber
agreement is what two different models give on a scan outside their training domain. The **aorta** is low by
design: HolOrama only sees the crop (heart mask + margin), so the arch and descending aorta are missing; use
TotalSegmentator / `aorta_cta_radiomics` for the aorta and treat `touches_crop_box` as the warning it is. The
**right atrium** disagreement (42.7 vs 16.0 mL) is a TotalSegmentator under-segmentation of an RA cut by the
inferior scan edge on the QC image, not a HolOrama failure, but it shows that neither model is a reference here.

## Python API

```python
from heart_segmentation import Params, segment_case          # subprojects/heart_segmentation on sys.path
manifest = segment_case(cta_path, out_dir, case_id, Params(), crop_mask_path=heart_mask)
```

`nnunet_onnx.OnnxNnUNet3D` can run **any** nnU-Net 3d_fullres model exported with HolOrama's
`scripts/model_export/export_onnx.py` plus a `config.json` (spacing, CT normalisation statistics, patch,
labels, orientation); `holorama_labels.check_against_model_config` guards against a silently retrained model.

## Known limitations

- **Field of view.** The DAYLIGHT extended-arm CTAs end through the heart (see `subprojects/valve_calcium/`):
  the inferior chambers and the IVC are often cut. Filter on `touches_scan_inferior` before any volume analysis;
  the per-structure volumes of a truncated structure are partial.
- **Non-gated, non-cardiac protocol.** The model was trained on coronary CTA (gated, cardiac FOV). Motion blur at
  the coronaries and valves, and the arterial-phase timing of a stroke CTA, differ from its training data. The
  trainer name (`...RandomMetalArtifactStepMotion...`) suggests motion / metal augmentation, but no validation on
  stroke CTA exists yet; review the QC PNGs or load cases in Slicer before trusting a label.
- **Fat labels** depend on the pericardium being resolved; on a 0.25 mm reconstruction with noise they can be
  fragmented. Compare with the shell-based peri-structure fat in `cta_common.pcat` / `scripts/run_peri_laa_fat.py`.
- **Memory.** Logits are kept for all 15 classes at model spacing (~76 bytes per resampled voxel). The driver
  refuses above 50 % of physical RAM (`--max-memory-fraction`, `--allow-large`).
- CoreML (Apple GPU / ANE) does not support the model's 5-D InstanceNormalization; the Mac runs on CPU.

## Provenance and licence

- Code adapted from HolOrama `src/segmentation/onnx_inference.py` (MIT, Sebastian Balzer) with nnU-Net pieces
  (Apache 2.0, DKFZ); details in the header of `nnunet_onnx.py`.
- Weights: HolOrama commit `e6ce046` (2026-11-20, v0.21.0), `src/segmentation/weights/ct_seg/`,
  trainer `nnUNetTrainerRandomMetalArtifactStepMotionNoMirrorU2NetS__nnUNetResEncUNetLPlans__3d_fullres`, fold 0,
  `checkpoint_best.pth`. `external/holorama/SOURCE.json` records what was fetched.
- Cite: Mohammadi Kazaj P, et al. *A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping.*
  arXiv:2607.11287 (2026); and Stark AW, et al. *Comput Methods Programs Biomed* 2025;109065 (the HolOrama software).

Patient data, segmentations and per-case outputs live under `outputs/` (git-ignored) and are never committed.
