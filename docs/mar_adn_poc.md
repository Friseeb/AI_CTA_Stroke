# Metal-Artifact Reduction (MAR) — ADN POC

Research POC evaluating learned, image-domain metal-artifact reduction on our
contrast CTA. **The ADN clone and its pretrained weights are NOT tracked in git**
(`external/adn/` is gitignored — the weights are ~626 MB). This file records how
to reproduce the setup and download the weights.

## Outcome (TL;DR)

ADN runs on this Mac (CPU/MPS, torch 2.x) and reduces streaks on its native
data, but **both pretrained models globally remap HU on contrast CTA** (domain
shift from non-contrast spine/abdomen CT) — unusable for quantitative endpoints
and inadequate for QC. **Use the exclusion+burden path instead**
(`cta_common.artifacts` + `scan_artifact_burden.py`), which fabricates no HU.
Making learned MAR work here would require **retraining ADN on contrast CTA**.

## Repo

- ADN: <https://github.com/liaohaofu/adn> (Liao et al., "Artifact Disentanglement
  Network for Unsupervised Metal Artifact Reduction", TMI 2020).

```bash
git clone https://github.com/liaohaofu/adn.git external/adn
```

## Pretrained weights (download — do not commit)

The repo's `demo.py` auto-downloads from Google Drive, or fetch manually with
`gdown` by file ID into the paths ADN expects:

```bash
pip install gdown
# spineweb (HU window [-1000, 2000]) — best fit for HU CTA
gdown 1eF-6YTJYlVa7fVMk8n9yQssAqzrhLO1T -O external/adn/runs/spineweb/spineweb_39.pt
# deep_lesion (attenuation-coefficient preprocessing)
gdown 1NqZtEDGMNemy5mWyzTU-6vIAVIk_Ht-N -O external/adn/runs/deep_lesion/deep_lesion_49.pt
```

## Environment

Run in `totalseg-mac` (torch 2.x + the usual imaging deps). Extra deps:

```bash
python -m pip install gdown "googledrivedownloader==0.4"
```

## Patches needed for torch 2.x / modern matplotlib

The 2019 code (tested on torch 1.0 / CUDA 9) needs two edits to run on torch 2.x
CPU/MPS — re-apply after cloning:

- `adn/models/base.py`, `resume()`:
  `torch.load(checkpoint_file)` → `torch.load(checkpoint_file, map_location="cpu", weights_only=False)`
- `demo.py`, `plot_image()`:
  `fig.savefig(output_file, frameon=False, bbox_inches='tight')` → drop `frameon=False`

## Run

```bash
cd external/adn
python demo.py spineweb --no_gpu   # bundled samples
```

For the contrast-CTA domain-shift test on a case slice, the POC script lived at
`/tmp/adn_cta_test.py` (preprocess to the model's HU/attenuation range, run
`ADNTest.forward(img_low)` → `pred_lh`, convert back to HU, compare). Results for
sub-529 are under `outputs/mar_poc/sub-529/` (`*_adn_spineweb_compare.png`,
`*_adn_deep_lesion_compare.png`) showing the global HU shift.

## See also

- `cta_common/src/cta_common/artifacts.py` — the recommended exclude+flag approach.
- `aorta_cta_radiomics/scripts/scan_artifact_burden.py` — cohort burden scan.
