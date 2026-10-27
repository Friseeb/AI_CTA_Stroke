# Calcified cerebral emboli (CCE) detection — v0 scaffold

Goal: automatically flag calcified emboli in the intracranial (and cervical) arteries on stroke CTA, hand
them to a reader for confirmation in 3D Slicer, and link confirmed emboli to a calcium source
(aortic valve / mitral annulus from `subprojects/valve_calcium`, carotid bifurcation and aortic arch from
`aorta_cta_radiomics`, FLOWCAT carotid tree).

## What the literature says (see `docs/literature/calcified_cerebral_emboli_landscape.md`)

- CCE are in 2.7 % of stroke CTs and 1.3-2.6 % of EVT cases; MCA in ~84 %; median diameter 1.6 mm and median
  density ~150 HU on NCCT (occlusive EVT-grade emboli: ~330 HU, 4-20 mm). Sources: calcific aortic valve ~36 %,
  carotid plaque ~30 %, MAC ~11 %. Recurrence ~40 %; EVT reperfusion is markedly worse.
- **NCCT is the detector.** On CTA a calcified embolus is isodense to iodine ("false patency sign") unless it is
  very dense. Criteria in use: > 70 HU (sensitive) / >= 130 HU max (specific) on thin NCCT, round or ovoid,
  intraluminal, not tubular / crescentic (wall calcium), away from skull base, dura, choroid, basal ganglia.
- No published model detects CCE. Reusable neighbours: TopCoW CLAIM CoW segmentation (we have it),
  CoW centreline extraction (fmusio), nnunet-angio-infer CTA vessel model (Apache-2.0), Jin's intracranial
  calcification subtyping (sphericity features), Kockelkoren intimal/medial score. ISLES'24 is the only public
  NCCT+CTA set where CCE could be labelled retrospectively.

## v0b: whole-intracranial calcium search with anatomical classification (`search_intracranial_calcium.py`)

Sebastian's framing (2026-10-26): search *all* intracranial calcium and subtract what is explained by anatomy,
so the embolus candidates are the leftovers. Every bright focus inside the TotalSegmentator brain mask is
classified, in order:

| class | rule | masks |
| --- | --- | --- |
| skull_dural | within 2 mm of skull | `total` skull |
| dural_sinus | inside dilated venous sinuses (falx / tentorium calcium) | `brain_structures` venous_sinuses |
| arterial_wall | touches the CoW labelmap, eccentric / elongated | TopCoW CLAIM |
| proximal_intraluminal | touches the CoW, compact, >= 30 % inside the lumen | TopCoW CLAIM |
| choroid_plexus | inside ventricles dilated 2 mm | `brain_structures` ventricle |
| pineal_habenula | posterior cap, 12 mm around the 3rd ventricle | `ventricle_parts` third_ventricle |
| basal_ganglia | inside lentiform / caudate / thalamus | `brain_structures` |
| *_adjacent | small compact focus within 2.5 mm of a larger wall / skull / sinus calcification (partial-volume tail) | derived |
| symmetric_bilateral | mirror-image counterpart within 6 mm | midsagittal plane of the brain mask |
| subthreshold_noise | tier-2 (low-HU) focus smaller than 1.4 mm: single-voxel noise on thin / interpolated slices | derived |
| **distal_candidate** | the rest, compact, 1-10 mm (tier 2 needs >= 1.4 mm) | embolus candidates for review |

First real case (sub-570, CTA): 22 foci >= 500 HU inside the brain mask -> 15 skull/dural, 2 dural sinus, 1 BA wall
calcium, 3 non-compact parenchymal foci, and one 1 mm "intraluminal" speck at the cavernous ICA that is the
partial-volume tail of the adjacent wall calcium (hence the `_adjacent` rule). Zero distal candidates. Choroid and
pineal foci are below 500 HU on CTA and will only matter in NCCT mode.

Tiers: CTA >= 700 HU (tier 1) / >= 500 HU (tier 2); `--modality ncct` switches to >= 130 / >= 90 HU. Missing
masks just disable their rule and are recorded in the manifest. Outputs: class labelmap, per-focus CSV, Slicer
fiducials for the candidates, QC montage. Test: `tests/test_search_synthetic.py` (one phantom focus per class).

```bash
TS=/opt/anaconda3/envs/totalseg-mac/bin/TotalSegmentator; C=outputs/calcified_emboli/<case>
$TS -i <cta> -o $C/total_bone -ta total -d cpu -rs skull vertebrae_C1 vertebrae_C2 vertebrae_C3 brain
$TS -i <cta> -o $C/brain_structures -ta brain_structures -d cpu -cp $C/total_bone
$TS -i <cta> -o $C/ventricle_parts -ta ventricle_parts -d cpu -cp $C/total_bone
$PY subprojects/calcified_emboli/search_intracranial_calcium.py --image <cta> --total-dir $C/total_bone \
    --brain-structures-dir $C/brain_structures --ventricle-parts-dir $C/ventricle_parts --cow <topcow_seg> --out-dir $C/intracranial_search
```

## v0a design (CoW-corridor candidate generator, `detect_calcified_emboli.py`)

v0 runs on CTA alone and therefore only catches dense emboli (>= 700 HU); the NCCT-gated version is the next
step and the main item on the roadmap. On sub-570 the CTA-only pass with skull exclusion returns no candidates.


`detect_calcified_emboli.py`
1. Arterial mask: TopCoW CLAIM circle-of-Willis labelmap (`scripts/run_topcow_claim.py`), or any vessel mask
   (FLOWCAT for the neck). Resampled to the CTA grid if needed.
2. Lumen reference = median HU of the eroded vessel mask; threshold = max(700 HU, reference + 300 HU), so the
   opacified lumen (300-450 HU) never qualifies.
3. Candidates = bright connected components within 2 mm of the vessel mask, 1-8 mm equivalent diameter,
   compact (volume / bounding-sphere >= 0.25), with >= 30 % of voxels inside the lumen, and not within 2 mm of
   bone (TotalSegmentator skull + C1-C3). Eccentric crescents (wall atheroma) and extraluminal calcifications
   (dura, choroid, pineal, basal ganglia) fail the inside-lumen / compactness rules or sit outside the search ROI.
4. Each candidate gets a 0-1 score, the touching TopCoW vessel label, size, HU, and a Slicer fiducial
   (`<case>_cce_candidates.mrk.json`) plus a QC montage.

Tests: `tests/test_detect_synthetic.py` (embolus vs wall crescent vs far blob vs bone slab).

## Run

```bash
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
$PY subprojects/calcified_emboli/detect_calcified_emboli.py \
    --cta <cta.nii.gz> --vessels <topcow_seg.nii.gz> --bone <skull_union.nii.gz> --out-dir outputs/calcified_emboli/<case>
```
Bone union: `TotalSegmentator -ta total -rs skull vertebrae_C1 vertebrae_C2 vertebrae_C3 brain`, then OR the masks.

## Confirmed-CCE cohort (SanDisk drive, 10 patients, staged 2026-10-26)

Result (2026-10-26, all 10 cases, NCCT mode, no CoW mask, after the reader review below): every case has tier-1
candidates on the reported side, 1-9 candidates per case out of 70-420 calcium foci per head.
Reader-verified lesions: 1617 lateral left occipital (1.7 mm, 199 HU) and 986 left sylvian region (3.6 mm, 1516 HU).
CTA passes found candidates only where the embolus was very dense (2262, 2686, 361): emboli are isodense to
contrast ("false patency"), so **NCCT is the detector and the CTA pass is confirmatory only**.
Per-case table: `outputs/calcified_emboli_cohort/cohort_summary.csv` (de-identified reference sites in
`reference_sites.json`, extracted from the export's REPORT.PDF; the PDFs carry names and stay on the drive).

| case | reported side | foci | candidates | on reported side | low-conf demoted | strongest candidate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cce1057 | L | 72 | 3 | 2 | 0 | L 2.5 mm 471 HU |
| cce1569 | R | 419 | 6 | 6 | 1 | R 2.0 mm 677 HU |
| cce1617 | L | 75 | 2 | 2 | 1 | L 1.7 mm 199 HU (occipital, verified) |
| cce1901 | L | 185 | 4 | 4 | 2 | L 1.4 mm 442 HU |
| cce2262 | R | 89 | 3 | 3 | 0 | R 2.8 mm 757 HU |
| cce2357 | L | 71 | 8 | 6 | 1 | L 2.7 mm 532 HU |
| cce2686 | L | 103 | 6 | 5 | 2 | L 3.2 mm 1542 HU |
| cce2736 | R | 96 | 1 | 1 | 0 | R 1.9 mm 192 HU |
| cce361 | R | 128 | 9 | 9 | 0 | R 3.0 mm 559 HU |
| cce986 | L | 162 | 2 | 1 | 2 | L 3.6 mm 1516 HU (verified by reader) |

Reader review found one misclassification: 986's embolus had been labelled choroid_plexus because the
`brain_structures` ventricle mask on that Siemens scan is 161 mL (over-segmented) and its 2 mm dilation reached a
lateral focus. Choroid now requires >= 50 % of the focus inside the undilated lumen, masks > 80 mL are untrusted,
and `ventricle_mask_ml` is written to the manifest. Tiny + faint foci (< 1.5 mm and < 170 HU) are demoted to
`distal_candidate_low` (kept in the CSV, not reviewed). All 10 cases are positives; there is no negative control.

Lessons: tier-2 (90-130 HU) foci need the 1.4 mm size floor or 1-voxel noise floods the list (1569: 54 specks);
smoothing before thresholding loses faint 1.7 mm emboli and was rejected; EDT-based dilation makes a search ~30 s.
Laterality: RAS +x is the patient's right and these dcm2niix volumes run voxel i toward the left; the i-based
side label is correct, verified against DICOM columns. Specificity is untested: only positives here (sub-570 CTA is
the single negative, 0 candidates); a non-contrast negative set is the next requirement.

## v0a design (CoW-corridor candidate generator, `detect_calcified_emboli.py`)

v0 runs on CTA alone and therefore only catches dense emboli (>= 700 HU); the NCCT-gated version is the next
step and the main item on the roadmap. On sub-570 the CTA-only pass with skull exclusion returns no candidates.


`detect_calcified_emboli.py`
1. Arterial mask: TopCoW CLAIM circle-of-Willis labelmap (`scripts/run_topcow_claim.py`), or any vessel mask
   (FLOWCAT for the neck). Resampled to the CTA grid if needed.
2. Lumen reference = median HU of the eroded vessel mask; threshold = max(700 HU, reference + 300 HU), so the
   opacified lumen (300-450 HU) never qualifies.
3. Candidates = bright connected components within 2 mm of the vessel mask, 1-8 mm equivalent diameter,
   compact (volume / bounding-sphere >= 0.25), with >= 30 % of voxels inside the lumen, and not within 2 mm of
   bone (TotalSegmentator skull + C1-C3). Eccentric crescents (wall atheroma) and extraluminal calcifications
   (dura, choroid, pineal, basal ganglia) fail the inside-lumen / compactness rules or sit outside the search ROI.
4. Each candidate gets a 0-1 score, the touching TopCoW vessel label, size, HU, and a Slicer fiducial
   (`<case>_cce_candidates.mrk.json`) plus a QC montage.

Tests: `tests/test_detect_synthetic.py` (embolus vs wall crescent vs far blob vs bone slab).

## Run

```bash
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
$PY subprojects/calcified_emboli/detect_calcified_emboli.py \
    --cta <cta.nii.gz> --vessels <topcow_seg.nii.gz> --bone <skull_union.nii.gz> --out-dir outputs/calcified_emboli/<case>
```
Bone union: `TotalSegmentator -ta total -rs skull vertebrae_C1 vertebrae_C2 vertebrae_C3 brain`, then OR the masks.

## Confirmed-CCE cohort (SanDisk drive, 10 patients, staged 2026-10-26)

First two cases (NCCT): sub-cce1057 -> 3 candidates out of 72 foci, the strongest left one (2.5 mm, 471 HU) matching
the reported left sylvian MCA-branch calcification; sub-cce1569 -> 7 tier-1 candidates, all right-sided (538-677 HU),
matching the reported right M2 calcification, after 54 one-voxel noise specks were demoted. The same patient's CTA
holds only 661 voxels >= 500 HU in the whole brain and yields no candidate: the emboli are isodense to contrast
("false patency"), so **NCCT is the detector and the CTA pass is confirmatory only**. Smoothing before thresholding
was tried and rejected (0.6 mm lost faint 1.7 mm emboli); the tier-2 size floor alone removes the noise.

`convert_cce_cohort.py` reads the DICOMDIR-style exports straight from `/Volumes/SANDISK USB/Calcified emboli/<PIN>/`
(one sequential pass per series, nothing else copied) and writes `outputs/calcified_emboli_cohort/sub-cce<PIN>/raw/`:
thin NCCT (0.6 mm, all 10), CTA (6 of 10: "CTA AORTIC ARCH-ABOVE VERTEX" / "CTA NECK/HEAD"), and the 5 mm NC head.
GE head scans carry a 16 deg gantry tilt with uneven slice spacing; dcm2niix's tilt-corrected / slice-equalised
variant (`*_Tilt_Eq_*`) is kept (y padded, z interpolated to ~0.31 mm), the raw sheared volume is discarded.
`run_cce_cohort.py` then runs, per case and modality, TotalSegmentator (brain/skull, brain_structures,
ventricle_parts) and the intracranial search in NCCT mode (>= 130 / >= 90 HU) and CTA mode, serially and resumably,
writing `cohort_status.csv`. The confirmed emboli give the first sensitivity read-out of the classifier.

## Roadmap
- [x] Skull-base exclusion: TS skull + C1-C3 union removes the petrous / cavernous ICA wall-calcium hit on sub-570.
- [ ] **NCCT pairing (priority)**: stage the non-contrast head CT per EVT case (not staged locally today), rigid-register
      NCCT -> CTA, detect on NCCT (>= 90 / >= 130 HU tiers) inside the CTA-derived arterial corridor, record the CTA HU at
      the same voxels (isodense to lumen = false patency; distal non-opacification = occlusion).
- [ ] Extend the corridor beyond the CoW to M2/M3 (CTA threshold inside TS brain, skull subtracted) and along the neck (FLOWCAT).
- [ ] Shape features vs wall calcium: sphericity, elongation, concentricity to the lumen centreline, contiguity with > 8 mm wall calcium.
- [ ] Reader loop: Slicer loader with fiducials, accept / reject per candidate, log to CSV.
- [ ] Source linking: distance along the arterial tree to the nearest calcium pool; valve / arch / carotid attribution.
- [ ] Learned classifier (candidate patches -> embolus vs mimic) once ~100 reviewed candidates exist.
- [ ] Cohort run on the 57 staged EVT cases (TopCoW must be run first; only sub-570 has it today).

Literature and open-model landscape: `docs/literature/calcified_cerebral_emboli_landscape.md` (pending).
Research use only.

### Negative controls (NCCT without emboli)

Specificity is untested. To add negatives, export the non-contrast head CT the same way (DICOMDIR export per PIN) into a
folder on the drive, e.g. `/Volumes/SANDISK USB/Negatives/<PIN>/Export_*/`, then:

```bash
PY=/opt/anaconda3/envs/totalseg-mac/bin/python
$PY subprojects/calcified_emboli/convert_cce_cohort.py --source "/Volumes/SANDISK USB/Negatives" \
    --out-root outputs/calcified_emboli_cohort --prefix sub-neg
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
$PY subprojects/calcified_emboli/run_cce_cohort.py --root outputs/calcified_emboli_cohort --prefix sub-neg --no-cta
```
sub-570 (the CTA negative used in development) has no NCCT staged locally; its non-contrast head CT would be the first negative.
