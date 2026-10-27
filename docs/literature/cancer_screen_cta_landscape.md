---
title: Opportunistic cancer / incidental-finding screening on stroke CTA — open-source landscape
date: 2026-10-26
status: compiled by Claude (web search); URLs fetched where stated, [unverified] otherwise
related: subprojects/cancer_screen/README.md
---

# Opportunistic cancer / incidental-finding screening on stroke head-neck-chest CTA — open-source landscape (26 Oct 2026)

Verification note: every URL below was fetched or appeared in a fetched/search result unless marked **[unverified]**. PubMed/PMC pages were cookie-blocked, so abstracts were pulled via the Europe PMC API.

## 1. Clinical evidence and guideline thresholds

**Prevalence on stroke CTA (head/neck, contrast, non-gated).**
- Rositzka et al., *Clin Neurol Neurosurg* 2020 (n=1,175 stroke CTAs): 1,988 incidental findings (1.7/patient); thyroid incidentalomas 23.2% of findings; 9.1% of findings of major relevance; **38.8% not mentioned in the primary report**. https://pubmed.ncbi.nlm.nih.gov/32203706/
- Chen et al., *Medicine* 2020 (n=1,698 neck CTAs): 376 potentially significant findings — thyroid nodules 23.4%, pulmonary nodules 14.9%, enlarged lymph nodes 6.4%, PE 5.1%; **48.9% missed in the initial report**. https://pmc.ncbi.nlm.nih.gov/articles/PMC7581090/
- Bentsen et al., *J Stroke Cerebrovasc Dis* 2015 (n=757 thrombolysis candidates): suspicious lung opacities 2.6%; definite/probable lung cancer **1.1%**. https://pubmed.ncbi.nlm.nih.gov/25817618/
- Lin et al., *JDMS* 2021 (n=13,563): incidental thyroid nodules 1.4%; malignancy 8% of nodules meeting biopsy criteria. https://doi.org/10.1177/87564793211018459
- Saito et al., *Cureus* 2024 (n=290 whole-body CTA for suspected stroke): 7 malignancies (2.4%). https://pmc.ncbi.nlm.nih.gov/articles/PMC11779692/
- Extended CTA (≥6 cm below carina, DAYLIGHT NCT05522244) adds mediastinum/hilum/lung to the FOV; ~6.7% malignancy in one extended-CTA series. https://clinicaltrials.gov/study/NCT05522244 ; https://radiologybusiness.com/topics/medical-imaging/computed-tomography-ct/extended-cta-exams-more-triple-stroke-detection
- ESUS/occult cancer: 2–10% of ESUS patients are diagnosed with cancer within a year; 5–10% have active cancer (Stroke 2021 review; AHA 2024 scientific statement; INVISIBLE-1 NCT06100718). https://www.ahajournals.org/doi/pdf/10.1161/STROKEAHA.120.032002 ; https://www.ahajournals.org/doi/10.1161/STR.0000000000000517 ; https://clinicaltrials.gov/study/NCT06100718

**Rules to convert masks to flags.**
- Lung: Fleischner 2017 — solid <6 mm no routine f/u (low risk); 6–8 mm CT at 6–12 mo; >8 mm CT 3 mo/PET/biopsy; volume thresholds 100 and 250 mm³; part-solid ≥6 mm f/u 3–6 mo. https://pubs.rsna.org/doi/full/10.1148/radiol.2017161659
- Thyroid: ACR white paper (Hoang 2015) — ultrasound if ≥1.0 cm (<35 y) or ≥1.5 cm (≥35 y), or any size with suspicious features/local invasion/nodes. https://www.jacr.org/article/S1546-1440(14)00627-9/fulltext **[URL unverified; paper verified]** ; PDF: https://geiselmed.dartmouth.edu/radiology/wp-content/uploads/sites/47/2019/04/ACR_thyroid2015.pdf
- Cervical nodes: short axis ≥10 mm general, 11 mm jugulodigastric (Hoang AJR 2013, Som AJR 1992), 15 mm level I/II, retropharyngeal ≥8 mm; plus necrosis, rounded shape, clustering ≥3. https://www.ajronline.org/doi/10.2214/AJR.12.8960 ; https://www.ajronline.org/doi/10.2214/ajr.158.5.1566697
- Mediastinal/hilar nodes: ACR white paper Munden 2018 (short axis >10 mm = enlarged; ≥15 mm or clustered → work-up). https://www.jacr.org/article/S1546-1440(18)30530-1/fulltext
- Adrenal/liver/renal ACR white papers (2017–2018): https://www.jacr.org/article/S1546-1440(17)30551-3/fulltext ; https://www.jacr.org/article/S1546-1440(17)30889-X/fulltext ; https://www.jacr.org/article/S1546-1440(17)30497-0/fulltext

## 2. Open models, code and weights

**Lymph nodes**
| Tool | What / modality | Code | Weights | Licence | Performance / CTA fit |
|---|---|---|---|---|---|
| LNQ 2023 winner "Skeleton Suns" (nnU-Net ResEnc + TotalSegmentator negative labels) | mediastinal LN ≥1 cm, **contrast CT** | not located **[unverified]** | no | — | median DSC 71.2%, ASSD 2.95 mm. Paper: https://arxiv.org/html/2408.10069 |
| LNQ 3rd: MICAI-IMI-UzL/LNQ2023 (nnU-Net + probabilistic LN atlas) | contrast CT | https://github.com/MICAI-IMI-UzL/LNQ2023 | **yes** (IMI Cloud) | CC BY-NC-ND 4.0 | good fit for extended-CTA mediastinum |
| LNQ 2nd: Kuailei/Lymph-Node-- (nnU-Net, airway-guided) | contrast CT | https://github.com/Kuailei/Lymph-Node-- | no | unspecified | training recipe only |
| WltyBY/LNQ2023_training_code (weak-sup VNetv2) | contrast CT | https://github.com/WltyBY/LNQ2023_training_code | yes (GDrive) | unspecified | val DSC 54.5 — weaker |
| Bouget ct_mediastinal_structures_segmentation (AGUNet, UNet-SW32) | mediastinal LN, contrast CT | https://github.com/dbouget/ct_mediastinal_structures_segmentation | **yes** (`setup.py`) | BSD-2 | ONNX/CPU-runnable; LyNoS data https://github.com/raidionics/LyNoS |
| HN lymph-node **levels** (Weissmann 2023, 20 levels, nnU-Net) | planning CT | https://github.com/putzfn/HNLNL_autosegmentation_trained_models | **yes** (releases) | CC0 | gives level labels, not nodes; use to assign level to detected nodes |
| Neck LN segmentation (Al Hasan 2024, Dice 0.81, 221 contrast neck CTs) | contrast CT | not released | no | — | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11612088/ |
| Cervical LN nnU-Net transfer (Sci Rep 2024, 626 pts, 11,013 nodes) | CT | https://github.com/Luoxd1996/ **[unverified repo path]** | unclear | — | https://www.nature.com/articles/s41598-024-84804-3 |
| HECKTOR 2022/2025 GTVn (nodal) | PET/CT | e.g. https://github.com/wangkaiwan/HECKTOR-2022-AIRT , https://github.com/HC94/swin_unetr_hecktor_2022 | partial | varies | PET-dependent; CT-only transfer is weak |
TotalSegmentator does **not** output lymph nodes (verified README).

**Lung nodules**
- TotalSegmentator `lung_nodules` (lung + lung_nodules; 1,353 subjects incl. LIDC; BLUEMIND): Apache-2.0, no published metrics. https://github.com/wasserth/TotalSegmentator
- MONAI `lung_nodule_ct_detection` (RetinaNet, LUNA16 fold 0; mAP 0.852, AR 0.998; resample to 0.70×0.70×1.25 mm): Apache-2.0, weights on HF. https://huggingface.co/MONAI/lung_nodule_ct_detection ; https://github.com/Project-MONAI/model-zoo/tree/dev/models/lung_nodule_ct_detection
- nnDetection (LUNA16 SOTA-level; ≥11 GB VRAM; "pretrained models coming soon" — train yourself): https://github.com/MIC-DKFZ/nnDetection
- Sybil (NLST LDCT risk; MIT; weights on releases) — trained on LDCT only; expect calibration failure on contrast CTA. https://github.com/reginabarzilaygroup/Sybil
- VISTA-3D class 23 "lung tumor" (not nodule): https://github.com/Project-MONAI/VISTA/tree/main/vista3d (code Apache-2.0; weights NCLS v1).
- Domain shift: all of the above trained on non-contrast/LDCT; LIDC-trained CAD does transfer to contrast CT but with sensitivity loss, and vessels enhanced by CTA raise FPs — use TS lung mask + vessel mask (`lung_vessels` task) to suppress. https://pmc.ncbi.nlm.nih.gov/articles/PMC10611755/

**Thyroid**
- No open CT thyroid-nodule detector with weights found. TotalSegmentator `total` → `thyroid_gland` (Apache); VISTA-3D class 126 thyroid gland. Literature only: Cascade/fusion CNN on contrast CT (2019, PMC6855097); CA-HRNet, *Gland Surg* 2026, 500 contrast CTs, Dice 78.6%, no weights. https://pmc.ncbi.nlm.nih.gov/articles/PMC13358517/

**Oral cavity / tongue / pharynx / larynx**
- SegRap 2023 (NPC GTVnx + GTVnd + 45 OARs; 200 pts; non-contrast + contrast pairs): https://segrap2023.grand-challenge.org/ ; 1st-place GTV code with Docker weights (nnU-Net + TotalSegmentator): https://github.com/Astarakee/segrap2023
- HECKTOR 2025 (>1,100 pts, 10 centres, PET/CT; Dice 0.75): https://hecktor25.grand-challenge.org/ ; https://arxiv.org/abs/2606.20143
- LaryngealCT (1,029 TCIA-derived CTs, staging benchmark, reconstruction scripts only): https://www.nature.com/articles/s41598-026-58171-0
- TotalSegmentator head tasks (all Apache): `head_glands_cavities` (parotid, submandibular, naso/oro/hypopharynx, soft/hard palate…), `headneck_bones_vessels` (larynx_air, thyroid/cricoid cartilage, hyoid, ICA, IJV), `headneck_muscles` (pharyngeal constrictors, SCM…). No oral-cavity/tongue tumour class anywhere open.

**Universal lesion / interactive**
- ULS23 baseline (nnU-Net ResEnc, 38,693 lesions; weights on Zenodo): https://zenodo.org/records/10777532 ; https://github.com/DIAGNijmegen/ULS23
- LesionLocator (zero-shot point/box lesion seg, CVPR 2025; GPL-3.0 code, trained incl. NIH Lymph Nodes & LIDC; CUDA/CPU/**MPS**): https://github.com/MIC-DKFZ/LesionLocator
- nnInteractive (points/scribbles/box/lasso; Apache code, **CC BY-NC-SA weights**; ≥6–10 GB VRAM or remote server): https://github.com/MIC-DKFZ/nnInteractive ; Slicer client https://github.com/coendevente/SlicerNNInteractive
- MedSAM2 (Apache-2.0; box prompts; Slicer plugin): https://github.com/bowang-lab/MedSAM2
- MULAN (DeepLesion, 2D Mask R-CNN, weights on Box): https://github.com/rsummers11/CADLab/tree/master/MULAN_universal_lesion_analysis
- Foundation models: CT-FM (MIT, 148k CTs, HF weights) https://github.com/project-lighter/CT-FM ; Merlin (MIT, abdominal CT VLM) https://huggingface.co/stanfordmimi/Merlin ; CT-CLIP/CT-CHAT (non-contrast chest) https://github.com/ibrahimethemhamamci/CT-CHAT

## 3. Pipelines / commercial
No published open-source "incidental-findings-on-CTA" pipeline exists; closest are TotalSegmentator-based opportunistic body-composition work and LCS incidental-findings AI (Radiology 2025 https://doi.org/10.1148/radiol.242118). Commercial (closed): Aidoc incidental pulmonary nodule (NLP + CT), Qure.ai qCT LN Quant (FDA, non-contrast), RevealAI-Lung (FDA), Viz.ai (stroke only). https://www.qure.ai/us/blog/Future-of-AI-powered-Incidental-Pulmonary-Nodule-detection

## 4. Datasets
| Dataset | Content | Contrast | Licence |
|---|---|---|---|
| DeepLesion (NIH) | 32k lesions, mixed body | mixed | open, Box https://nihcc.app.box.com/v/DeepLesion |
| LIDC-IDRI / LUNA16 | 1,018 / 888 chest CT | mostly non-contrast | CC BY 3.0 / CC BY 4.0 |
| NLST | LDCT | no | CDAS application |
| TCIA CT Lymph Nodes (Roth) | 90 mediastinal + 86 abdominal | contrast | CC BY 3.0 https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546 |
| LNQ 2023 | 513 chest CT | contrast | CC BY 4.0 https://lnq2023.grand-challenge.org/ |
| RADCURE / HN1 / HNSCC | 3,346 / 137 planning CT | 61.5% contrast (RADCURE) | TCIA restricted / No-commercial |
| SegRap 2023 | 200 NPC, paired nc+ce | both | challenge terms https://zenodo.org/records/7839896 |
| HECKTOR 2025 | >1,100 PET/CT | CT low-dose | challenge terms |
| CT-RATE | 50,188 vols, 18 labels (incl. lymphadenopathy, lung nodule) | **non-contrast** | CC BY-NC-SA https://huggingface.co/datasets/ibrahimhamamci/CT-RATE |
| RadGenome-ChestCT | grounded reports on CT-RATE | non-contrast | CC BY-NC-SA, gated |
| AbdomenAtlas | 8,448 CT, multi-phase | 37% arterial | CC BY-NC-SA |
| ULS23 | 38,693 lesions | mixed | challenge terms |
| MedShapeNet | shapes only, no images | — | **[unverified]** |

## 5. Recommended architecture (ranked by feasibility)

**Tier 0 — zero training, run now (Mac CPU/MPS for TS; Spark for the rest)**
1. Anatomy backbone: TS `total` (thyroid_gland, esophagus, trachea, lungs, liver/kidneys/adrenals when in FOV) + `lung_nodules` + `lung_vessels` + three head tasks; body-part regression to confirm FOV.
2. **Lung nodules**: TS `lung_nodules` ∪ MONAI RetinaNet (resample to 0.7/0.7/1.25 mm); keep candidates inside TS lung mask and outside dilated vessel mask; compute long/short axis + volume; emit Fleischner tier (<6 / 6–8 / >8 mm; 100/250 mm³).
3. **Mediastinal/hilar nodes** (extended CTA only): Bouget AGUNet (BSD, CPU-ok) and UzL LNQ model (NC licence — research use); connected components → short axis; flag ≥10 mm, escalate ≥15 mm/clustered (Munden 2018).
4. **Thyroid**: TS thyroid_gland → intra-gland hypo/hyper-enhancing blobs (HU-deviation + LoG) → axial diameter; flag ≥15 mm (≥35 y) / ≥10 mm (<35 y), any size with calcification/invasion. Expect FPs; this is reader-triage, not detection.
5. **Cervical nodes**: no open detector; stop-gap = LesionLocator/nnInteractive seeded by LoG blobs in the fat between TS vessels/muscles/glands, with level assignment from the putzfn CC0 level model; flag by the level-specific table above.

**Tier 1 — adapt (fine-tune on Spark, ~50–150 reader-corrected cases)**
- nnU-Net v2 (ResEnc) for cervical+supraclavicular nodes, initialised from the UzL/Bouget mediastinal weights plus TCIA CT Lymph Nodes, using your own CTAs annotated via SlicerNNInteractive; the 2024 neck study shows Dice ≈0.81 is reachable on contrast neck CT.
- Thyroid nodule nnU-Net from TS thyroid crops (CA-HRNet paper is the template, 500 cases).

**Tier 2 — needs labels / weak evidence**
- Oral cavity/tongue, oropharynx, larynx primaries: no open CT detector; start with asymmetry screening (left/right volume+HU of TS oropharynx, hypopharynx, larynx_air, constrictors, parotid/submandibular) → nnInteractive confirm; SegRap/HECKTOR weights are NPC/PET-bound and should only seed pretraining. Also esophageal wall thickness from TS esophagus (>5 mm distended) and bone lesions via VISTA-3D class 128 (NC weights).

**Reader loop**: SlicerTotalSegmentator + SlicerNNInteractive; each flag written as a Slicer markup (centroid, axes, rule fired) and a per-case JSON; reader accepts/rejects → feeds Tier 1 training. Report flags, not diagnoses; audit against the 39–49% radiologist miss rate above.

## References (verified unless marked)
Rositzka 2020 https://pubmed.ncbi.nlm.nih.gov/32203706/ · Chen 2020 https://pmc.ncbi.nlm.nih.gov/articles/PMC7581090/ · Bentsen 2015 https://pubmed.ncbi.nlm.nih.gov/25817618/ · Lin 2021 https://doi.org/10.1177/87564793211018459 · Saito 2024 https://pmc.ncbi.nlm.nih.gov/articles/PMC11779692/ · Fleischner 2017 https://pubs.rsna.org/doi/full/10.1148/radiol.2017161659 · ACR thyroid 2015 PDF (above) · Hoang AJR 2013 https://www.ajronline.org/doi/10.2214/AJR.12.8960 · Munden 2018 https://www.jacr.org/article/S1546-1440(18)30530-1/fulltext · LNQ 2023 https://arxiv.org/html/2408.10069 · TotalSegmentator https://github.com/wasserth/TotalSegmentator · MONAI nodule bundle https://huggingface.co/MONAI/lung_nodule_ct_detection · Sybil https://github.com/reginabarzilaygroup/Sybil · VISTA-3D https://github.com/Project-MONAI/VISTA/tree/main/vista3d · LesionLocator https://github.com/MIC-DKFZ/LesionLocator · nnInteractive https://github.com/MIC-DKFZ/nnInteractive · MedSAM2 https://github.com/bowang-lab/MedSAM2 · ULS23 https://zenodo.org/records/10777532 · SegRap https://github.com/Astarakee/segrap2023 · HECKTOR 2025 https://hecktor25.grand-challenge.org/ · CT-FM https://github.com/project-lighter/CT-FM · Merlin https://huggingface.co/stanfordmimi/Merlin · CT-RATE https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
