# Opportunistic cancer / incidental-finding screen on stroke CTA — v0 scaffold

Goal: flag incidental findings with oncologic relevance on the head-neck-chest CTA of stroke patients and
route them to reader review: lung nodules / masses, thyroid nodules, salivary-gland and pharyngeal masses,
and cervical / supraclavicular / mediastinal lymphadenopathy.

## What the landscape review says (see `docs/literature/cancer_screen_cta_landscape.md`)

- Stroke CTA series report ~1.7 incidental findings per patient; thyroid nodules ~23 %, lung nodules ~15 %, enlarged
  nodes ~6 % of findings; definite/probable lung cancer ~1 %; 39-49 % of significant findings missing from the
  primary report. ESUS: 2-10 % get a cancer diagnosis within a year.
- Rules: Fleischner 2017 (lung), ACR 2015 thyroid (>= 1 cm < 35 y, >= 1.5 cm >= 35 y), cervical nodes short axis
  >= 10 mm (level-specific 8-15 mm), mediastinal >= 10 mm (escalate >= 15 mm / clustered).
- Open models worth adding: MONAI `lung_nodule_ct_detection` RetinaNet (Apache, HF weights), Bouget mediastinal
  lymph-node AGUNet (BSD, CPU-capable) and the UzL LNQ 2023 nnU-Net (CC BY-NC) for mediastinal nodes, putzfn
  CC0 head-neck nodal-level model for level assignment, LesionLocator / nnInteractive for interactive confirmation.
  No open CT thyroid-nodule or oral-cavity / tongue tumour detector exists; those need asymmetry screening + labels.

## v0 design

`cancer_screen.py` consumes TotalSegmentator outputs (all tasks are licensed and downloaded on this Mac):

| Target | Model used | v0 output |
| --- | --- | --- |
| Lung nodules | `lung_nodules` (TS, 1.5 mm) + `total` lobes | per-nodule max axial / equivalent diameter, volume, HU, lobe, Fleischner 2017 bin, solid vs sub-solid |
| Thyroid | `total` thyroid_gland | volume, lobe asymmetry, HU heterogeneity, focal hypo-/hyperdense candidates >= 8 mm (ACR incidental thyroid nodule thresholds: >= 1 cm < 35 y, >= 1.5 cm >= 35 y applied downstream) |
| Salivary glands / pharynx | `head_glands_cavities` | parotid / submandibular left-right volume asymmetry, naso/oro/hypopharynx airway volumes (mass surrogates) |
| Lymph nodes | none installed | placeholder: no model segments nodes; see landscape review for candidates (LNQ 2023 models, nnDetection) |
| Tongue / oral cavity | none installed | placeholder; head-neck tumour models are PET/CT or MRI |
| Chest second reader | NV-Reason-CT (NVIDIA, 3D CT VLM, CUDA only) | optional on the DGX Spark: structured chest report on the thoracic crop; chest/abdomen only, 2 mm resampling |

Outputs per case: long-format features CSV, lesion table, Slicer fiducials for lesions, manifest.

## Run

```bash
TS=/opt/anaconda3/envs/totalseg-mac/bin/TotalSegmentator; IN=<cta.nii.gz>; OUT=outputs/cancer_screen/<case>
$TS -i $IN -o $OUT/total -ta total -d cpu -rs thyroid_gland lung_upper_lobe_left lung_lower_lobe_left \
    lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right esophagus trachea skull brain \
    clavicula_left clavicula_right vertebrae_C1 vertebrae_C5 vertebrae_T1 vertebrae_T4
$TS -i $IN -o $OUT/lung_nodules -ta lung_nodules -d cpu -cp $OUT/total
$TS -i $IN -o $OUT/head_glands -ta head_glands_cavities -d cpu -cp $OUT/total
PY=/opt/anaconda3/envs/aorta-cta-radiomics/bin/python
$PY subprojects/cancer_screen/cancer_screen.py --cta $IN --ts-root $OUT --out-dir $OUT/analysis
```

## First real case (sub-112, extended-arm CTA)

- `lung_nodules` labelled one 75 mm, 45 mL, 78 HU component outside the lungs (mediastinum / heart) and nothing
  inside them: on contrast CTA the model needs the inside-lung rule (>= 50 % of the component within the 2 mm
  dilated lobes), which is now in place. The lungs also run to the inferior scan edge.
- Thyroid: the first pass flagged three 9-10 mm "candidates" that were an enhancing vessel, ossified cartilage and
  gland-edge partial volume. v0 now erodes the gland 2.5 mm, excludes TS thyroid / cricoid cartilage, uses the ICA
  blood pool as the enhancing-vessel reference (only when >= 150 HU: the left ICA of this EVT case reads 60 HU,
  i.e. unopacified) and reports >= 450 HU foci as calcifications. Result: 2 calcifications, no nodule candidates.
- Salivary glands and pharynx: symmetric, no flags. Lymph nodes: no model.

## Roadmap
- [ ] Validate `lung_nodules` on contrast CTA (trained on mixed CT; 1.5 mm resampling misses < 4 mm nodules).
- [ ] Lymph nodes: run Bouget AGUNet (BSD) and UzL LNQ 2023 (NC) on the mediastinum of extended-arm CTAs; cervical
      nodes via LoG blobs in the fat between TS vessels / muscles / glands + nnInteractive confirmation + putzfn levels.
- [ ] Lung: add MONAI RetinaNet as a second detector; suppress CTA-enhanced vessels with the TS `lung_vessels` task.
- [ ] Thyroid: replace HU-outlier rule with a learned nodule detector if labels become available.
- [ ] NV-Reason-CT chest report on the DGX Spark; reconcile with measurements.
- [ ] Reader loop in Slicer (fiducials -> accept / reject -> follow-up recommendation per guideline).
- [ ] Cohort run on the 57 staged extended-arm cases.

Landscape review: `docs/literature/cancer_screen_cta_landscape.md` (pending). Research use only.
