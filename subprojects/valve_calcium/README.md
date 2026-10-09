# Valve calcium (AVC + MAC) on contrast CTA

Opportunistic aortic valve calcium (AVC) and mitral annular calcification (MAC) markers on the
non-gated stroke CTAs of the DAYLIGHT **extended** arm (whole heart in the field of view).

## What it does

1. **TotalSegmentator** (v2.12, licensed tasks already downloaded on this Mac)
   - `total --fast` with `heart aorta inferior_vena_cava` only -> crop masks shared by the next tasks
   - `heartchambers_highres` -> LA, LV, myocardium, aorta
   - `aortic_sinuses` -> LVOT + right / left / non-coronary cusp  (the aortic valve complex)
   - `coronary_arteries` -> excluded from both ROIs (ostial and circumflex calcium confound)
   - **STOPSTORM model 603** (nnU-Net v2, Zenodo 13752275, CC BY 4.0; download per `external/stopstorm/README.md`)
     on the heart crop -> learned `Valve_Mitral` label (`stopstorm_postprocess.py`); ~95 s/case on CPU
2. **ROIs** (`valve_calcium.py:build_rois`)
   - aortic valve complex = TotalSegmentator cusps + adjacent LVOT, dilated 2 mm (this segmentation is good
     enough on its own; no second aortic definition is produced)
   - mitral annulus (geometric) = ring where dilated LA and dilated LV meet, minus the aortic valve complex
   - mitral valve (STOPSTORM) = learned `Valve_Mitral` label dilated 2 mm, minus aortic complex and coronaries;
     both mitral definitions are exported (`mitral_annulus`, `mitral_valve_stopstorm`) because no open CT
     model delineates the mitral annulus and the two cross-validate each other
3. **Calcium** with a contrast-adaptive threshold: blood reference = median HU of the eroded blood pool;
   candidate >= max(300, ref + 150); seed >= max(500, candidate); calcium = candidates connected to a seed;
   plus fixed 300 / 500 / 600 / 800 HU volumes for sensitivity.
4. Outputs per case: ROI + calcium masks, long-format features CSV, QC montage PNG, manifest JSON.

Features: `calcium_volume_mm3`, `calcium_mass_score_mg_eq` (HU x volume), `agatston_like_3mm_equiv`
(slice-wise Agatston rule rescaled to 3 mm; **not** a gated Agatston score), `calcium_max_hu`,
`calcium_components`, blood reference and thresholds, ROI volume.

## Run

```bash
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
# one case
$PY subprojects/valve_calcium/run_valve_calcium_case.py \
    --cta outputs/daylight_evt_extended/<case_id>.nii.gz --out-root outputs/valve_calcium
# look at a processed case in 3D Slicer (macOS: must be `open -na`)
VALVE_CASE=<case_id> open -na /Applications/Slicer.app --args \
    --python-script subprojects/valve_calcium/slicer/load_valve_calcium_case.py
# serial, resumable batch over the locally staged extended-arm CTAs
$PY subprojects/valve_calcium/run_valve_calcium_batch.py \
    --input-dir outputs/daylight_evt_extended --out-root outputs/valve_calcium
# tests
$PY -m pytest subprojects/valve_calcium/tests -q
```

Each TotalSegmentator / STOPSTORM step is skipped when its outputs exist, so re-running only does the analysis
(`--force-analysis` to redo it, batch `--redo-ok` to revisit cases already marked ok).
Run from locally staged copies only (see the external-drive I/O rule); never from `/Volumes`.

## Cohort run (2026-10-08/09, 57 staged EVT extended-arm cases)

52 analysed, 5 failed because TotalSegmentator found no LV / cusps (heart outside the scan).
**Field of view is the limiting factor**: LV touches the inferior scan edge in 48/52, the mitral band is flagged
truncated in 45/52, and the aortic root is within 5 mm of the edge in 28/52. Usable aortic valve coverage in
~24 cases, usable mitral coverage in ~7. Filter on the `field_of_view` features before any analysis.
Outputs: `outputs/valve_calcium/<case>/`, `outputs/valve_calcium/valve_calcium_features_all.csv`, `batch_status.csv`.

## Known limitations

- Non-gated CTA: motion blur at the valve, blooming from contrast; scores are relative markers.
- `aortic_sinuses` and `heartchambers_highres` are TotalSegmentator research-licence tasks.
- The mitral ROI is a geometric band, not a leaflet segmentation; posterior annulus calcium next to the
  circumflex is handled by subtracting the coronary mask, which is itself imperfect.
- Validate against a reader on the QC montages before using the numbers.

Literature and open-pipeline reviews: `docs/literature/valve_calcium_ct_literature_review.md` and
`docs/literature/mitral_valve_segmentation_ct_landscape.md`.

Patient data: raw CTAs, segmentations and per-case outputs live under `outputs/` (git-ignored) and are never
committed; only code, tests and literature notes are in the repository.
