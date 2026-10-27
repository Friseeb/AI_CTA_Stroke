---
title: Calcified cerebral emboli on CT — literature, tooling, build plan
date: 2026-10-26
status: compiled by Claude (web search); citations checked by search, [unverified] where not
related: subprojects/calcified_emboli/README.md
---

# Calcified cerebral emboli (CCE): literature, tooling, and a build plan

## 1. Clinical literature

**Prevalence.** Walker/Shah/Osborn (AJNR 2014) found CCE on 2.7% of stroke CTs; 27% were initially misdiagnosed and 9% missed outright; MCA in 83%; presumed sources calcific aortic stenosis 36%, carotid plaque 30%, MAC 11%; spontaneous in 86%; recurrent stroke in 41% [1]. In EVT cohorts the rate is lower because only occlusive CCE are counted: 1.3% of 2,969 (7 European centres, Maurer/Dobrocky et al. AJNR 2020) [2]; 1.8% of 3,077 in the MR CLEAN Registry (Bruggeman 2021) [3]; 2.6% of 1,004 EVTs in Lübeck (Schacht 2025) [4]. Menounos & Matar's 2024 review quotes 2.7–5.9% of all AIS and a 43% recurrence rate [5]. Fasen et al. (AJNR 2020) showed why this matters for a CTA-based group: 20% of LVOs were initially missed on CTA, and calcified emboli were present in 24% of missed cases versus 0% of detected ones [6].

**Imaging phenotype.** The largest morphological series (Serghine et al. 2025; 242 CCE in 79 patients) gives median diameter 1.55 mm and median density 146 HU, MCA 84% and mostly distal, multiple CCE in 34%, multiterritorial in 15%, and diameter–density correlation r = 0.69 (bigger, prouder emboli are proximal) [7]. Walker reported mean 162 HU (79–435), 82% round / 18% ovoid, "not tubular or linear like wall calcification" [1]. Occlusive EVT-grade CCE are larger and denser: Maurer mean 327 HU (150–1,200), mean length 9.2 mm (4–20) [2]. HU cut-offs in use: >70 HU on NCCT (Menounos criteria) [5]; mean ≥90 HU (Bruggeman) [3]; max ≥130 HU (Schacht, borrowed from Agatston, explicitly to avoid confusing subacute blood >80 HU) [4]. Ordinary hyperdense-artery-sign thrombus runs ~50–90 HU [4]. The "salted pretzel sign" describes multiple 2–3 mm calcified foci along the MCA tree [8]. Menounos' proposed criteria: compulsory = small round/oval lesion >70 HU on NCCT localising to an arterial tract (plus clinically relevant territory if symptomatic, and no recent cardiac/vascular procedure); supportive = identifiable calcified source, bicuspid AV, prior CCE, MCA location, CKD [5]. Mimics listed there and in Walker: tubular/linear atherosclerotic wall calcification, intracranial stenosis, non-calcified HAS, small haemorrhage/contrast staining, calcified granuloma, cavernoma, cysticercosis, choroid plexus and other physiological calcification [1,5]. Important for CTA-first groups: the "false patency sign" — a CCE is isodense to iodine (~300–450 HU) and simulates a patent lumen on CTA, so NCCT (or CTP) is the detecting modality [5]. Dual-energy virtual non-calcium imaging separates thrombus from calcification and found CCE in 1/34 patients (Shinohara 2024) [9].

**Outcomes.** Every EVT series shows worse reperfusion: mTICI ≥2b 57.5% with 90-day mortality 55.9% [2]; eTICI ≥2b significantly lower, functional independence 29% [3]; MASC multicentre (Grand 2022) reperfusion 57%, good outcome 28%, meta-analysis of 136 patients 50%/29% [10]; propensity-matched Lübeck series mTICI ≥2b 62% vs 92%, first-pass 12% vs 46%, infarct growth 81% vs 42% [4]. A 2026 systematic review (14 studies, 160 patients) reports mRS ≤2 in 11–28% and mortality 0–62.5% [11]. Raghib 2018 (21 pooled cases) found EVT superior to IV rtPA [12]. Source attribution in the series: vascular 54%, mixed vascular/cardiac 32%, undetermined 14% [7]; carotid 3/9 and cardiac 3/9 in Mosqueira's series [13]. For TAVR: Van Mieghem (Circulation 2013) found calcified debris in 17% of filter captures, 0.55–1.8 mm, from valve and aortic wall [14]; Kroon 2021 found calcified material in 12–33% depending on valve platform [15]. I could not find CCE papers by Kakkos or Doehner [unverified — probably mis-remembered authors]; Kavanagh et al. AJNR 2006 is a 4-case series [16]; Bugnicourt's contribution is a 2008 ACA case [17].

## 2. Automated detection tools (with code status)

No published model detects calcified emboli specifically. Relevant neighbours:

| Tool | Task / modality | Code / weights | Licence |
|---|---|---|---|
| Bortsova et al. 2021 (Rotterdam) [18] | ICAC segmentation + volume, NCCT; sens 83.8%, PPV 88%, ICC 0.98 | not public (Rotterdam Study data by DUA) | — |
| Jin et al. 2025 MAE-ViT IAC segmentation (IST-3) [19] | intracranial arterial calcification, NCCT | github.com/bjin96/mae-medical-segmentation, plus bjin96/ct-processing (ICA ROI pipeline) | not stated on repo (paper CC BY) |
| Jin et al. 2026 intimal/medial IAC subtyping [20] | sphericity + MedShapeNet shape embeddings; automates Kockelkoren score [21] | github.com/bjin96/iac-subtyping (no weights) | not stated |
| Hou et al. 2025 DST [22] | segment-specific ICA calcification mapping, NCCT | "code will be released" — not found | — |
| Kim et al. 2025 JCN HAS segmentation [23] | 2D U-Net, thick NCCT, 1,038 scans/7 centres; AUROC 0.85; MCA calcification was a named false-positive source | not public | — |
| Tsuji et al. 2026 HAS detection [24] | 3-step DL pipeline, 690 NCCT; sens 74–76%, spec 83–87% | not public | — |
| Shinohara 2020 HMCAS Xception [25] | slice classification, NCCT | not public | — |
| Mojtahedi et al. 2022 [26] | dual-encoder U-Net thrombus seg, NCCT+CTA (MR CLEAN), Dice 0.62, surface Dice 0.78; NCCT clipped to 0–100 HU (CCE would saturate) | "available on reasonable request" | — |
| Patel et al. 2026 medRxiv [27] | dual-channel nnU-Net NCCT+CTA thrombus, Dice 0.64, uncertainty ensemble | not public | — |
| TopCoW 2024 winner (CLAIM Berlin) [28] | CoW multiclass seg, CTA + MRA, nnU-Net ResEncM + YOLO ROI | github.com/claim-berlin/TopCoW_2024_MRA_winning_solution; weights zenodo 14191592 | not stated |
| CoW_Centerline_Extraction (Musio) [29] | centreline graph + morphometrics from TopCoW masks | github.com/fmusio/CoW_Centerline_Extraction (Voreen + nnU-Net skeletoniser) | not stated |
| nnunet-angio-infer [30] | nnU-Net vessel seg for DSA/MRA/CTA, CTA model trained on ImageCAS + TopCoW 2024 | github.com/lankabelgezogen/nnunet-angio-infer | Apache 2.0 |
| CROWN challenge [31] | CoW classification/quantification, TOF-MRA only | challenge code, MRA | — |
| iCafe / ArteryX [32] | intracranial artery feature extraction, TOF-MRA | github.com/clatfd/GNN-ART-LABEL | — |

## 3. Public datasets

None carry CCE or intracranial-calcification labels. Useful anyway: **TopCoW** (125 paired CTA+MRA with CoW masks; training data CC BY-NC / OpenData Swiss "open use, commercial needs permission"; external test adds 26 ISLES'24 CTA + 20 LargeIA CTA, zenodo 15692630) [28,33]; **ISLES'24** (149 training cases: NCCT + CTA + CTP + follow-up DWI, LVO point mask on CTA, CoW labels; CC BY-NC-SA 4.0, zenodo 17652035) [34]; **CoW centreline graph dataset** (zenodo 17358162) [29]; head-CTA aneurysm DICOM dataset (zenodo 15697196, CC BY 4.0) [35]. IST-3 (used by Jin) and Rotterdam Study (Bortsova) are application-only. Because ISLES'24 pairs NCCT and CTA for LVO patients, it is the only public set where one could retrospectively label CCE (expected yield ~2–4 cases).

## 4. Detection design used in the literature

- **Modality**: NCCT is the detector; CTA is where CCE hide (false patency) [5]. Thin reconstructions matter: at 1.55 mm median diameter [7], 5 mm NCCT partial-volumes most emboli to <100 HU, which is why thick-slice HAS models call calcification a confounder [23].
- **HU**: >70 HU (sensitive), ≥90 mean, ≥130 max (specific) on NCCT [3–5]. On 0.5 mm CTA, enhanced lumen is 300–450 HU, so a calcium-only threshold on CTA fails; your contrast-adaptive detector's logic (local lumen HU + margin) is the right primitive, or use NCCT→CTA registration to find foci that are hyperdense on NCCT within a CTA-defined vessel.
- **Shape**: round/ovoid, discrete, intraluminal and concentric versus tubular/linear/crescentic eccentric wall calcification [1,5]. Kockelkoren's intimal/medial CT score (circularity, thickness ≥1.5 mm, patchy vs continuous) [21] and Jin's sphericity/shape-embedding subtyping [20] are directly reusable as "wall vs embolus" shape features.
- **Location**: M1/M2 dominant; distance from carotid siphon/skull base excludes the dense ICAC pool and dura/clinoid calcification; multiplicity along one tree supports embolism [1,7,8].
- **Pairing**: lesion present on NCCT at the CTA occlusion point or at a CTA "patent" segment with distal non-filling; CTP/DECT VNCa as tie-breakers [5,9].

## 5. Recommended approach (ranked)

1. **Rule-based candidate detector on NCCT, gated by CTA vessel masks (weeks 1–4).** Register NCCT→CTA (rigid, SimpleITK/ANTs). Build an intracranial arterial corridor by dilating (2–3 mm) the TopCoW CoW mask plus the nnunet-angio-infer/TotalSegmentator vessel mask; extend the corridor to M2/M3 by thresholding CTA within the brain mask (TotalSegmentator brain, skull subtracted). Candidates = connected components with max NCCT HU ≥130 (also log a ≥90 tier), volume 1–100 mm³, equivalent diameter 1–8 mm, inside corridor, >5 mm from skull/dura and outside the cavernous/petrous ICA. Compute sphericity, elongation, eccentricity relative to the CTA lumen centreline (from CoW_Centerline_Extraction / skimage skeleton), concentricity, HU max/mean, and the CTA HU at the same voxels (≈ lumen → false patency; distal non-opacification → occlusion). Discard crescentic/eccentric components and those contiguous with long wall calcification (>8 mm along the centreline).
2. **Reader review loop (weeks 3–8).** Push candidates into the napari Stroke CTA Studio as a ranked list with NCCT/CTA toggles; readers label embolus / wall calcification / dural-choroid / artefact. Expect 20–50 true CCE per 1,000 EVT cases, so grade all historical EVT NCCT+CTA pairs. Record Menounos criteria fields.
3. **Learned classifier (month 3+).** Train a small 3D patch classifier (nnU-Net or radiomics + gradient boosting) on the reviewed candidates; keep the rule stage as the proposal generator. Use Jin's MAE-ViT weights as an initialiser if their licence is confirmed. Report sensitivity at a fixed false-positive-per-scan rate.
4. **Source linking.** For each confirmed embolus, quantify candidate pools with the existing detectors: aortic-valve and MAC calcium (valve pipeline), arch and carotid-bifurcation calcium (FLOWCAT + contrast-adaptive detector), plus ICAC. Compute per-pool features: Agatston-like mass, largest-fragment HU histogram, surface roughness, and hemispheric laterality (a left-MCA CCE with only right carotid calcium points to heart/arch). Assign a source likelihood with a simple ordered rule set (ipsilateral carotid calcified plaque with irregular surface > arch ulcerated calcium > AV/MAC) then a logistic model once ≥30 cases are reviewed; cross-check against echo. Multiple/bilateral CCE strongly favour cardiac or arch origin [7].
5. **Cervical extension.** Run the same detector along the FLOWCAT carotid tree distal to the bifurcation; cervical calcified emboli are rare but the code is identical.

## References

1. Walker BS, Shah LM, Osborn AG. AJNR 2014;35:1515–9. doi:10.3174/ajnr.A3892. PMID 24651819.
2. Maurer CJ, Dobrocky T, et al. AJNR 2020;41:464–8. doi:10.3174/ajnr.A6412. PMID 32029470.
3. Bruggeman AAE, et al. (MR CLEAN Registry). J Neurosurg 2021;135:1402. doi:10.3171/2020.9.JNS201798. PMID 33799302.
4. Schacht H, et al. Stroke Res Treat 2025. doi:10.1155/srat/5538938. PMID 40438706.
5. Menounos S, Matar W. Front Neurol 2024;15:1401820. doi:10.3389/fneur.2024.1401820. PMID 39087019.
6. Fasen BACM, et al. AJNR 2020;41:607–11. doi:10.3174/ajnr.A6469. PMID 32165362.
7. Serghine Y, et al. Acta Neurol Belg 2025. doi:10.1007/s13760-025-02719-w. PMID 39813005.
8. Rodrigues BA, et al. Pract Neurol 2024. doi:10.1136/pn-2023-003947. PMID 38050103.
9. Shinohara Y, et al. J Comput Assist Tomogr 2024. doi:10.1097/RCT.0000000000001623. PMID 38657159.
10. Grand T, et al. (MASC). J Neuroradiol 2022;49:317–23. doi:10.1016/j.neurad.2022.02.006. PMID 35183595.
11. Pavic NV, et al. Clin Neuroradiol 2026. doi:10.1007/s00062-025-01611-7. PMID 41603946.
12. Raghib MF, et al. J Clin Neurosci 2018. doi:10.1016/j.jocn.2017.10.042. PMID 29102234.
13. Mosqueira AJ, et al. Neurologia (Engl Ed) 2022. doi:10.1016/j.nrleng.2019.04.002. PMID 34785159.
14. Van Mieghem NM, et al. Circulation 2013. doi:10.1161/CIRCULATIONAHA.112.001091. PMID 23652860.
15. Kroon H, et al. EuroIntervention 2021. doi:10.4244/EIJ-D-20-00744. PMID 32894232.
16. Kavanagh EC, et al. AJNR 2006;27:1996–9. PMID 17032882 (DOI not retrieved).
17. Bugnicourt JM, et al. J Mal Vasc 2008. doi:10.1016/j.jmv.2008.02.004. PMID 18455337.
18. Bortsova G, Bos D, et al. Radiol Artif Intell 2021. doi:10.1148/ryai.2021200226. PMID 34617024.
19. Jin B, et al. arXiv:2507.01744 (2025); Springer chapter doi:10.1007/978-3-032-08009-7_19 [venue unverified].
20. Jin B, et al. arXiv:2609.16035 (2026).
21. Kockelkoren R, et al. PLoS One 2017. doi:10.1371/journal.pone.0168360. PMID 28060941.
22. Hou X, et al. arXiv:2507.08214 (IEEE BIBM 2025).
23. Kim PE, et al. J Clin Neurol 2025. doi:10.3988/jcn.2024.0560. PMID 40635535.
24. Tsuji H, et al. Front Neurol 2026. doi:10.3389/fneur.2026.1771379. PMID 42088922.
25. Shinohara Y, et al. Jpn J Radiol 2020. doi:10.1007/s11604-019-00894-4. PMID 31673998; companion reader study PMID 32399602 [DOI unverified].
26. Mojtahedi M, et al. Diagnostics 2022;12:698. doi:10.3390/diagnostics12030698. PMID 35328251.
27. Patel TR, et al. medRxiv 2026. doi:10.64898/2026.01.23.26344730.
28. Yang K, Musio F, et al. (TopCoW). NEJM AI 2026. doi:10.1056/AIdbp2500994. PMID 42631290; arXiv:2312.17670.
29. Musio F, et al. Neuroimage: Reports 2026. doi:10.1016/j.neuri.2026.100265 [unverified; cited from repo].
30. github.com/lankabelgezogen/nnunet-angio-infer (Apache 2.0).
31. Vos IN, et al. (CROWN). Med Image Anal 2025. doi:10.1016/j.media.2025.103650. PMID 40505209.
32. Chen L, et al. iCafe reproducibility. Magn Reson Imaging 2019. PMID 30580079.
33. TopCoW data: zenodo.org/records/15692630.
34. ISLES'24: zenodo.org/records/17652035; arXiv:2408.11142.
35. Head CTA aneurysm DICOM dataset. Data 2026;11:74. doi:10.3390/data11040074; zenodo 15697196 [unverified beyond search listing].
