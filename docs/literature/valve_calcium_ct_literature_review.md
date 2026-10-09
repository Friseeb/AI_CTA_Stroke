---
title: Valve calcium (AVC, MAC) on CT — literature and open-pipeline review
date: 2026-10-08
status: draft, compiled by Claude (web search); citations checked by search, [unverified] where not
related: subprojects/valve_calcium/README.md, mitral_valve_segmentation_ct_landscape.md (follow-up search)
---

# Valve calcium (AVC, MAC) on CT and open-source segmentation — landscape review

Date: 2026-10-08. Scope: literature + open code relevant to building an opportunistic aortic-valve-calcium / mitral-annular-calcium marker on contrast-enhanced, mostly non-gated head-neck-chest CTA (0.5 mm) in stroke patients, with TotalSegmentator v2.12 (licensed tasks) and VISTA-3D available. Every citation below was checked by search; a few details that I could not confirm from the abstract are flagged [unverified]. Several citations in the brief had the wrong journal; corrections are noted inline.

## 1. Aortic valve calcium (AVC) scoring on non-contrast CT

- **Standard method.** Agatston scoring on ECG-gated non-contrast CT, 3 mm slices, 130 HU threshold, valve leaflets + annulus included, coronary/aortic-wall calcium excluded. The standardised protocol is laid out in Pawade et al., JACC Cardiovasc Imaging 2019 ("Why and how to measure aortic valve calcification") [DOI 10.1016/j.jcmg.2019.01.045].
- **Sex-specific severity thresholds.** Clavel et al., JACC 2013 (n = 794): best cut-offs for severe AS were ≥1,275 AU (women) and ≥2,065 AU (men); indexed ≥637 / ≥1,067 AU/m²; density ≥292 / ≥476 AU/cm² [DOI 10.1016/j.jacc.2013.08.1621]. Pawade et al., **Circ Cardiovasc Imaging** 2018 (not JACC CI as in the brief; 918 pts, 8 centres): optimal thresholds 1,377 AU (women) and 2,062 AU (men); C-statistic 0.92/0.89; thresholds independently predicted AVR/death (HR 3.90) [DOI 10.1161/CIRCIMAGING.117.007146; PMID 29555836].
- **Guidelines.** 2021 ESC/EACTS VHD guidelines (Vahanian et al., Eur Heart J 2022;43:561) encode: severe AS *highly likely* men >3,000 / women >1,600 AU; *likely* >2,000 / >1,200; *unlikely* <1,600 / <800. The 2025 ESC/EACTS update keeps >2,000 (men) / >1,200 (women) as the adjudication cut-offs. Also ACC/AHA 2020 (Otto et al., Circulation 2021).
- **Prognosis.** Beyond AS severity, AVC predicts AVR and mortality (Pawade 2018); in MESA, AVC ≥300 AU was associated with ~2× incident dementia over 16 years but *not* with stroke (Marrero et al., Atherosclerosis 2024; PMID 38890039). The Rotterdam Study found no AVC–stroke association (Bos et al., Stroke 2016;47:2859; DOI 10.1161/STROKEAHA.116.015200) and no AVC–dementia association (Wolters et al., J Alzheimers Dis 2017; DOI 10.3233/JAD-160871).
- **Reproducibility.** MESA (Budoff et al., Acad Radiol 2006;13:166): AVC interscan variability 9.7 ± 11.4 % (Agatston) / 8 ± 10.3 % (volume); MAC inter-reader 8–9 % but interscan 28–33 %. van Hamersvelt et al., Eur Radiol 2014 (DOI 10.1007/s00330-014-3191-0) specifically studied valve calcium reliability on low-dose *ungated* chest CT. Inter-reader ICC for AVC on gated CT is ~0.97 (Pawade 2019 review).
- **Non-gated non-contrast chest CT.** Ouchi et al., J Cardiovasc Comput Tomogr 2024;18:352 (5,385 pts; PMID 38556394) and Ouchi et al., Coron Artery Dis 2026 (PMID 41705777) showed volume score on non-gated CT tracks gated CT far better than Agatston (r 0.95 vs 0.69; Agatston bias −921 AU because of motion/slice differences); volume AUC ~0.975 for severe AS. Gunning et al., Eur Radiol 2025;35:6291 (DOI 10.1007/s00330-025-11553-w): a simple visual ordinal AVC grade on routine non-gated chest CT predicts mortality and changes management. The BSCI/BSCCT/BSTI consensus (Williams MC et al., Br J Radiol 2021;94:20200894; DOI 10.1259/bjr.20200894) recommends reporting incidental aortic-valve and MAC calcification on non-gated thoracic CT using a visual none/mild/moderate/severe scale.

## 2. AVC and MAC on contrast-enhanced CT (CTA)

Fixed 130 HU is meaningless in enhanced blood (aorta ~200–400 HU). Three families of approaches:

- **Patient-specific blood-pool thresholds (mean + k·SD).** Dong T et al., *Circ Cardiovasc Imaging* 2025;18:e017373 (PMID 40376780): "modified Agatston" with threshold = mean ascending-aorta HU + **4 SD** (resulting threshold ≈250 ± 98 HU); mean AVC 2,152 modified AU; optimal severe-AS thresholds **≥1,840 (men) / ≥1,430 (women)** modified AU, lower than non-contrast cut-offs because contrast raises the floor. Laohachewin et al., J Clin Med 2024;13:2386 (DOI 10.3390/jcm13082386): ROI mean + 4 SD, and a conversion formula to non-contrast Agatston, AVC_nc ≈ 691 + 1.83 × AVC_CE. Eberhard et al., EuroIntervention 2017;13:921 (TAVI cohort): compared fixed and dynamic (aortic-root mean + 2–4 SD) thresholds; mean + 3 SD represented calcium best [details partly from secondary sources]. Bettinger et al., J Cardiovasc Comput Tomogr 2017;11(4) ("Practical determination of aortic valve calcium volume score on contrast-enhanced CT prior to TAVR"; 104 CoreValve pts): compared 650 HU, 850 HU, LA×1.25, LA×1.5, LA+50, LA+100 (LA = luminal attenuation); thresholds *relative* to patient blood-pool attenuation predicted paravalvular regurgitation better than fixed cut-offs. (The brief listed this as JACC Imaging; it is JCCT.)
- **Fibrocalcific tissue volumes calibrated to blood pool.** Cartlidge TRG, Bing R, Kwiecinski J, et al., **Heart** 2021;107:1905 (not EHJ-CVI; DOI 10.1136/heartjnl-2020-318556; n = 164): calcific and non-calcific (fibrotic) valve volumes using HU thresholds calibrated against blood-pool radiodensity, indexed to annulus area; calcium volume correlated with Agatston AVC and peak velocity, and the fibrocalcific approach captured obstruction in fibrosis-dominant valves. The same Edinburgh group's "Quantitative CTA for valvular fibrocalcific volume" (JACC Cardiovasc Imaging 2024; DOI 10.1016/j.jcmg.2024.06.007) and Williams MC et al., Eur Heart J Cardiovasc Imaging 2021;22:262 (DOI 10.1093/ehjci/jeaa263; SCOT-HEART CCTA, prevalence/prognosis of AVC and MAC on contrast CCTA) extend this.
- **Fixed high thresholds / deep learning.** Jiang D et al., BMC Cardiovasc Disord 2026 (DOI 10.1186/s12872-026-06108-w; PMID 42323538): validated 450 HU, 850 HU and an nnU-Net against ex vivo *gravimetric* calcium mass of excised leaflets (n = 50 surgical validation): nnU-Net r = 0.967, bias +6 mg; 850 HU unbiased but less precise; 450 HU over-estimates with proportional bias. Park D et al., Sci Rep 2025 (DOI 10.1038/s41598-025-93744-5; Seoul NUH): DeepLabv3+ aorta segmentation + XGBoost valve-region refinement on CCTA, r = 0.93 vs non-contrast Agatston. Inomata et al., J Cardiovasc Dev Dis 2025;12:3 (DOI 10.3390/jcdd12010003): aortic valve extraction on contrast cardiac CT.
- **Non-gated contrast CTA specifically.** I found *no* study validating AVC or MAC quantification on non-gated contrast CTA (neck/chest CTA) against gated non-contrast Agatston. The closest are the non-gated non-contrast work (Ouchi 2024/2026; Gunning 2025) and gated contrast work (Dong 2025; Cartlidge 2021). This is a genuine gap your dataset could fill.
- **MAC on contrast CT.** No dedicated contrast-CT MAC threshold paper found; groups use the same blood-pool-relative thresholds (Williams 2021 CCTA cohort scored MAC presence/extent on CCTA). Caseous MAC (hypodense core with 880–1,000 HU rim) is a recognised CT pitfall and embolic source (multiple 2024–2026 case reports).

## 3. Mitral annular calcification

- **Quantification on CT.** MESA scores MAC by phantom-adjusted Agatston and volume on gated non-contrast CT (Kanjanauthai et al., Atherosclerosis 2010; reproducibility Budoff 2006 above). Hou J et al., Clin Cardiol 2023;46:1310 (DOI 10.1002/clc.24110) used non-contrast cardiac-CT MAC scores to predict surgical conversion/arrhythmia. "Mitral annulus calcium score" (Circ Cardiovasc Imaging 2018; DOI 10.1161/CIRCIMAGING.117.007508) [unverified details].
- **Severity classification.** Guerrero M et al., JACC Cardiovasc Imaging 2020;13:1945 (DOI 10.1016/j.jcmg.2020.03.013): CT MAC score = average annular calcium thickness (<5 mm = 1, 5–9.9 = 2, ≥10 = 3) + circumferential extent (<180° = 1, 180–270° = 2, >270° = 3) + trigone involvement (1 pt each) + leaflet involvement (1 pt each); mild ≤3, moderate 4–6, severe ≥7; score ≤6 → 60 % valve embolisation in ViMAC vs 9.7 %. The Heart Valve Collaboratory position statement (Guerrero M et al., JACC Cardiovasc Interv 2023;16(18); DOI 10.1016/j.jcin.2023.06.044) formalises a CT grading/staging scheme. These are geometric scores (thickness, arc) rather than Agatston, which suits a CTA where HU-based mass is uncertain.
- **Epidemiology / stroke / AF.** Benjamin et al., NEJM 1992;327:374 (Framingham; MAC RR for stroke 2.10 after adjustment). Fox et al., Circulation 2003;107:1492 (MAC predicts CVD events and mortality; DOI 10.1161/01.CIR.0000058168.26163.BC). Kohsaka et al., **JACC Cardiovasc Imaging** 2008;1:617 (NOMAS; not *Stroke* as in the brief; MAC >4 mm predicted MI and vascular death, borderline for ischaemic stroke; DOI 10.1016/j.jcmg.2008.07.006). Kizer et al., Stroke 2005;36:2533 (Strong Heart: MAC but not aortic sclerosis independently predicts incident stroke). Rodriguez CJ et al., JACC 2011;57:2172 (CHS: annular/valve calcification associated with covert MRI infarcts, RR 1.24). O'Neal et al., Europace 2015;17:358 (MESA: MAC → incident AF, HR 1.9) and O'Neal et al., Eur Heart J Cardiovasc Imaging 2018;19:279 (MAC progression → AF, HR 1.50). Fashanu et al., Atherosclerosis 2020;307:32 (MESA: MAC progression, not AVC, → ischaemic stroke HR 1.64; attenuated after excluding interim AF/CHD). Calicchio et al., J Cardiovasc Med 2023 (DOI 10.2459/JCM.0000000000001524; MESA, CT-Agatston MAC, 15 y): MAC → all stroke HR 1.68, ischaemic stroke HR 2.03 even after adjusting for AF and LA size. Meta-analysis: MAC associated with AF and MACE in AF patients (Medicine 2019; PMC6946188).
- **ESUS.** The ESC ESUS consensus (Ntaios et al., Eur Heart J 2024;45:1701; DOI 10.1093/eurheartj/ehae150) lists calcific aortic/mitral valve disease and MAC among low/uncertain-risk potential sources. I could not find a published CT-based MAC/AVC quantification in an ESUS or acute-stroke cohort; NAVIGATE-ESUS (Ntaios et al., Stroke 2020) tabulated potential embolic sources but MAC prevalence there is [unverified]. Evidence in 2020–2026 is dominated by caseous-MAC case reports — another gap your cohort can address.

## 4. Deep learning / automated valve calcium on routine CT

- Lessmann N et al., IEEE Trans Med Imaging 2018;37:615 (DOI 10.1109/TMI.2017.2769839): two-stage CNN on 1,744 NLST low-dose non-gated chest CTs; F1 0.89 coronary, 0.89 aorta, **0.67 aortic valve, 0.55 mitral valve**. Code: github.com/qurAI-amsterdam/calcium-scoring (GPL-3, training code only, **no weights**).
- van Velzen SGM et al., Radiology 2020;295:66 (DOI 10.1148/radiol.2020191621): same lineage across 6 cardiac/chest CT protocols; quantified coronary and thoracic-aorta calcium (valves not reported in the primary outcome).
- Zeleznik R et al., Nat Commun 2021;12:715 (DOI 10.1038/s41467-021-20966-2): CAC on gated and non-gated CT, outcome-validated. Eng D et al., npj Digit Med 2021;4:88 (DOI 10.1038/s41746-021-00460-1): gated + non-gated CAC, κ 0.89. Hagopian R et al., NEJM AI 2025;2(6) (DOI 10.1056/AIoa2400937): "AI-CAC" on non-gated non-contrast chest CT in the VA; **weights and code public** (github.com/Raffi-Hagopian/AI-CAC). None of these score valves.
- **Automated AVC on non-gated non-contrast CT.** Zheng J et al., JACC Adv 2026 (DOI 10.1016/j.jacadv.2026.103277): CNN, 33 sites, train 1,446 / test 239, r = 0.99 vs expert AVC, sens 0.92 / spec 0.98 for moderate–severe AS; code availability not stated [unverified]. Subramaniam A, Wu H, … Wilson DL, Sci Rep 2025 (DOI 10.1038/s41598-025-34818-2): multi-task network that localises a *cylindrical aortic-valve ROI* then applies Agatston criteria inside it; r ≈ 1.00 vs cardiologists; 20–22 % of a screening cohort had AVC. This "segment a valve region, then threshold" design is the one to copy.
- **Automated AVC on contrast CT.** Park 2025 (Sci Rep), Jiang 2026 (nnU-Net vs gravimetry), Inomata 2025 — see §2. Cartlidge/Dweck group's fibrocalcific CTA work also uses semi-automatic blood-pool-calibrated thresholds.
- **TotalSegmentator-based opportunistic calcium / valve planes.** Nalepa et al., arXiv 2501.11428 (2025) use multi-organ segmentation to anatomically constrain CAC scoring on non-contrast cardiac CT. Several CAC papers use TotalSegmentator only for aorta ROI / noise estimation. I found **no** published pipeline that derives valve planes from TotalSegmentator heart-chamber outputs for calcium scoring; PlatiPy (§5) does exactly that with its own chamber atlas, so the idea is published but not on TotalSegmentator.

## 5. Open-source models that segment valves/annuli on CT

| Tool | Segments | Input | Valves/annuli? | License | URL |
|---|---|---|---|---|---|
| **TotalSegmentator v2** `heartchambers_highres` | myocardium, LA, LV, RA, RV, aorta, pulmonary artery (sub-mm) | contrast or non-contrast, any gating | No | free academic / commercial licence | github.com/wasserth/TotalSegmentator |
| TotalSegmentator `aortic_sinuses` | LVOT, right/left/non-coronary cusps (sinus volumes) | CT | **Partial**: the three cusp-sinus regions bound the aortic valve; the annulus = plane between LVOT and the cusps | licensed | same |
| TotalSegmentator `coronary_arteries` | coronary tree | contrast or non-contrast | No | licensed | same |
| **PlatiPy** cardiac (Finnegan et al., *Phys Eng Sci Med* 2023, DOI 10.1007/s13246-023-01231-w — not Phys Med Biol) | whole heart, 4 chambers, great vessels, coronaries, SA/AV nodes, **aortic, mitral, pulmonic, tricuspid valves** | RT planning CT (non-contrast, non-gated, 1–3 mm) | **Yes, geometrically**: nnU-Net whole heart → multi-atlas chambers → valves modelled as fixed-size discs at the chamber interfaces (identical size parameters for all patients) | Apache-2.0 | github.com/pyplati/platipy, docs pyplati.github.io/platipy/cardiac.html |
| TAVRP-PL (Zöllner et al., arXiv 2507.16573, 2025) | aorta, aortic root, **valve, annulus**, LV, iliac arteries | 578 TotalSegmentator-dataset CTs (mixed contrast) with pseudo-labels | Yes (aortic only) | CC-BY-4.0 data on Zenodo 10.5281/zenodo.16274176; model weights not confirmed [unverified] | zenodo record |
| MIA-laboratory/AorticValveExtraction (Inomata 2025) | aortic valve on contrast cardiac CT | contrast, gated | Yes (aortic) | repo is README-only, no code/weights at time of check | github.com/MIA-laboratory/AorticValveExtraction |
| AortaPlane (QMUL MSc 2026) | aortic annulus plane regression | gated CT | annulus plane only | unstated; checkpoints included | github.com/vandhanaanatarajan-oss/AortaPlane |
| BAMNet | aortic root + landmarks | **fluoroscopy**, not CT | n/a | weights on Zenodo | github.com/Nikita75699/BAMNet |
| Wang M et al., eBioMedicine 2023;96:104794 (pre-TAVR aortic valvular complex) | annulus, leaflets, calcification, risk features | gated CTA | Yes | code not public [unverified] | — |
| "Whole-CCT"/unified cardiac CT (Mohammadi Kazaj, Shiri et al., arXiv 2607.11287, 2026) | 14 cardiac structures; 1,598 labelled + 60k pre-training | cardiac CT | not itemised; valves unlikely [unverified] | authors state code/weights/data to be released (CTAug library is public: github.com/AI-in-Cardiovascular-Medicine/CTAug) | arXiv |
| MOOSE 2 (`clin_ct_cardiac`) | myocardium, 4 chambers, aorta, PA, IVC, iliacs | CT (PET/CT) | No | Apache-2.0 code, CC-BY-4.0 weights | github.com/ENHANCE-PET/MOOSE |
| MONAI `wholeBody_ct_segmentation` bundle | 104 TotalSegmentator-v1 classes incl. chambers | CT | No | Apache-2.0 | Project-MONAI/model-zoo |
| VISTA-3D (label_dict.json, 127 classes) | `heart` (115), `left atrial appendage` (108), `aorta` (6), SVC, pulmonary vein, carotids, subclavians | CT | No chambers, no valves; point/class prompts can be used interactively | Apache-2.0 (MONAI) | Project-MONAI/VISTA |
| MM-WHS 2017 challenge (Zhuang) | 7 whole-heart structures (chambers, myocardium, aorta, PA) | contrast CTA, 20 train cases | No | per-repo | zmiclab.github.io/zxh/0/mmwhs |
| SegTHOR | heart (whole), aorta, trachea, oesophagus | non-contrast RT CT | No | challenge | litislab.fr/en/articles/segthor |
| ImageCAS | coronary arteries (1,000 CCTAs) | contrast, gated | No | Kaggle licence | github.com/XiaoweiXu/ImageCAS-... |
| CT-FM (project-lighter) | SSL foundation model (148k CTs); whole-body seg head | CT | No | open | github.com/project-lighter/CT-FM |
| TotalVibeSegmentator | 71 torso structures on **MRI (VIBE)** | MRI | No | open | github.com/robert-graf/TotalVibeSegmentator |
| qurAI calcium-scoring (Lessmann) | CAC/TAC/valve calcium classifier | low-dose chest CT | valve calcium *labels* in paper; training code only, no weights | GPL-3 | github.com/qurAI-amsterdam/calcium-scoring |
| AI-CAC (Hagopian) | CAC on non-gated non-contrast chest CT | non-contrast | No | weights public | github.com/Raffi-Hagopian/AI-CAC |

Bottom line: **no open model outputs a mitral annulus on CT**; the only open tools that output valves are PlatiPy (geometric, all four valves, Apache-2.0, trained on non-contrast RT CT) and the TAVR pseudo-label dataset (aortic valve/annulus, data only). TotalSegmentator `aortic_sinuses` + `heartchambers_highres` give enough geometry to construct both the aortic annulus (LVOT–cusp interface) and the mitral annulus (LA–LV interface ring, excluding the LVOT/aortomitral continuity) yourself.

## 6. Practical considerations on contrast CTA

- **Threshold selection.** Use a patient-specific blood-pool-relative threshold measured in the ascending aorta/LV cavity on the same scan: mean + 3–4 SD (Eberhard 2017; Laohachewin 2024; Dong 2025) or LA×1.25–1.5 (Bettinger 2017). Dong's data give a feel for scale: aorta 195 ± 87 HU, SD 14 ± 5 → threshold ≈250 HU; neck-CTA injection timing often leaves the heart in a mixed arterial/venous phase with *inhomogeneous* LA/LV enhancement, so sample the blood pool locally (aortic root for AVC, LA + LV for MAC) and record it as a covariate. Jiang 2026 shows a fixed 850 HU is unbiased but imprecise and 450 HU over-counts; a learned segmentation beats both.
- **Scores.** Report volume (mm³) and mass-like scores rather than Agatston: Agatston assumes 3 mm slices and 130 HU, is density-weighted (contrast-dependent), and shows large bias on non-gated CT (Ouchi 2026). Store the raw threshold-exceeding volume at several thresholds (mean+3SD, +4SD, 850 HU) to allow later calibration against gated non-contrast data; Laohachewin's conversion formula shows calibration is feasible.
- **Blooming / partial volume.** Blooming (partial volume + motion + beam hardening) inflates calcium apparent volume (Pack et al., Vis Comput Ind Biomed Art 2022; DOI 10.1186/s42492-022-00125-0). At 0.5 mm isotropic the partial-volume component is smaller than at 3 mm, but lesion edges will bloom into contrast-filled blood; partial-volume-corrected scoring (Šprem et al., PLoS One 2018;13:e0209318) or an erosion/min-island filter (your `fat_islands`-style min-volume rule) reduces noise. Higher thresholds (mean+4SD) shrink blooming at the cost of missing low-density (early) calcification — report sensitivity to k.
- **Motion in non-gated scans.** Mitral annulus moves 10–15 mm per cycle; aortic cusps similarly. Non-gated acquisition smears calcium (Agatston under-/over-estimation, Ouchi 2026) and can duplicate dense MAC. Mitigations used in the literature: volume rather than Agatston (Ouchi), visual ordinal grade as a robustness check (Gunning 2025; BSCI consensus), thick-slab MIP for presence/extent, and geometric MAC descriptors (thickness, arc of annulus involved, trigone/leaflet involvement — Guerrero 2020) which are less sensitive to HU than mass. Flag scans with step/stair-step artefact via TotalSegmentator chamber-surface discontinuity.
- **Anatomical exclusions.** Exclude coronary ostia/calcified proximal RCA/LM (use `coronary_arteries` task) and aortic-wall calcium above the sinotubular junction from AVC; exclude posterior-leaflet calcium vs annulus if you want MAC strictly (or report both, as Guerrero does). Caseous MAC: rim-only calcium around a hypodense core — flag by hole-filling ratio.
- **Dual phase.** If any patients have a non-contrast head CT that covers the heart apex it will not; so no in-house gold standard — plan a small external validation set with paired gated non-contrast CT (e.g., TAVI work-up patients) or use the Laohachewin/Dong conversion.

## Recommended approach

Reuse rather than build for anatomy: run TotalSegmentator `heartchambers_highres` + `aortic_sinuses` (+ `coronary_arteries` for exclusion) on the cardiac crop of each CTA, then construct annuli geometrically the way PlatiPy does — aortic annulus = boundary surface between LVOT and the union of the three cusp labels, dilated ±3–4 mm along the LVOT axis; mitral annulus = ring where LA and LV labels meet, minus the LVOT/aortomitral continuity, dilated into a 5–8 mm tube; optionally also run PlatiPy's valve module as a second opinion (Apache-2.0, but trained on non-contrast RT CT, so expect it to need the whole-heart mask rather than its own nnU-Net). Inside each valve ROI apply a patient-specific threshold (aortic-root mean + 3 SD and + 4 SD; separately LA/LV blood-pool for MAC) with a minimum-island rule, and export: calcium volume at each threshold, mass-like score (volume × mean excess HU), MAC geometric descriptors (max thickness, circumferential arc, trigone/leaflet involvement → Guerrero score bins), and a visual ordinal grade for a QC subset. Do not use VISTA-3D for anatomy (its `heart` label is a single blob) except as an interactive prompt tool for correcting chambers in hard cases.

Build only two things: (1) a small nnU-Net (or VISTA-3D fine-tune) for valve-calcium voxels on CTA, trained from the thresholded masks after manual cleaning of ~100–150 cases, because Jiang 2026 shows learned segmentation beats any fixed threshold on contrast CT; (2) a calibration/validation arm against gated non-contrast Agatston in an external sample. The clinical analyses that are genuinely open: CT-quantified MAC and AVC in an acute-stroke / ESUS cohort versus AF detection, LAA morphology/thrombus, and recurrence — MESA (Calicchio 2023; Fashanu 2020) suggests MAC rather than AVC is the stroke-relevant signal, and no group has published this on stroke CTA.

## References (verified unless flagged)

1. Clavel MA et al. JACC 2013;62:2329–38. DOI 10.1016/j.jacc.2013.08.1621
2. Pawade T et al. Circ Cardiovasc Imaging 2018;11:e007146. DOI 10.1161/CIRCIMAGING.117.007146; PMID 29555836
3. Pawade T et al. JACC Cardiovasc Imaging 2019. DOI 10.1016/j.jcmg.2019.01.045
4. Vahanian A et al. 2021 ESC/EACTS VHD guidelines. Eur Heart J 2022;43:561. (2025 update: Eur Heart J VHD 2026, xwag001)
5. Budoff MJ et al. Acad Radiol 2006;13:166–72
6. van Hamersvelt RW et al. Eur Radiol 2014. DOI 10.1007/s00330-014-3191-0
7. Ouchi K et al. J Cardiovasc Comput Tomogr 2024;18:352–62. PMID 38556394
8. Ouchi K et al. Coron Artery Dis 2026;37(6). PMID 41705777
9. Gunning SGS et al. Eur Radiol 2025;35:6291–301. DOI 10.1007/s00330-025-11553-w
10. Williams MC et al. Br J Radiol 2021;94:20200894. DOI 10.1259/bjr.20200894
11. Dong T et al. Circ Cardiovasc Imaging 2025;18:e017373. DOI 10.1161/CIRCIMAGING.124.017373; PMID 40376780
12. Laohachewin D et al. J Clin Med 2024;13:2386. DOI 10.3390/jcm13082386
13. Eberhard M et al. EuroIntervention 2017;13:921–7
14. Bettinger N et al. J Cardiovasc Comput Tomogr 2017;11(4). https://www.sciencedirect.com/science/article/abs/pii/S1934592517301077
15. Cartlidge TRG et al. Heart 2021;107:1905–11. DOI 10.1136/heartjnl-2020-318556
16. Williams MC et al. Eur Heart J Cardiovasc Imaging 2021;22:262–70. DOI 10.1093/ehjci/jeaa263
17. Jiang D et al. BMC Cardiovasc Disord 2026. DOI 10.1186/s12872-026-06108-w; PMID 42323538
18. Park D et al. Sci Rep 2025. DOI 10.1038/s41598-025-93744-5
19. Inomata S et al. J Cardiovasc Dev Dis 2025;12:3. DOI 10.3390/jcdd12010003
20. Guerrero M et al. JACC Cardiovasc Imaging 2020;13:1945–57. DOI 10.1016/j.jcmg.2020.03.013
21. Guerrero M et al. (Heart Valve Collaboratory) JACC Cardiovasc Interv 2023;16(18). DOI 10.1016/j.jcin.2023.06.044
22. Hou J et al. Clin Cardiol 2023;46:1310–8. DOI 10.1002/clc.24110
23. Kanjanauthai S et al. Atherosclerosis 2010 (MESA MAC risk factors) [volume unverified]
24. Benjamin EJ et al. N Engl J Med 1992;327:374–9. PMID 1625711
25. Fox CS et al. Circulation 2003;107:1492–6. DOI 10.1161/01.CIR.0000058168.26163.BC
26. Kohsaka S et al. JACC Cardiovasc Imaging 2008;1:617–23. DOI 10.1016/j.jcmg.2008.07.006
27. Kizer JR et al. Stroke 2005;36:2533–7. DOI 10.1161/01.STR.0000190005.09442.ad
28. Rodriguez CJ et al. JACC 2011;57:2172–80. DOI 10.1016/j.jacc.2011.01.034
29. O'Neal WT et al. Europace 2015;17:358–63
30. O'Neal WT et al. Eur Heart J Cardiovasc Imaging 2018;19:279. DOI 10.1093/ehjci/jex093
31. Fashanu OE et al. Atherosclerosis 2020;307:32–8. DOI 10.1016/j.atherosclerosis.2020.06.009
32. Calicchio F et al. J Cardiovasc Med 2023;24(9). DOI 10.2459/JCM.0000000000001524
33. Marrero N et al. Atherosclerosis 2024. PMID 38890039
34. Bos D et al. Stroke 2016;47:2859–61. DOI 10.1161/STROKEAHA.116.015200
35. Wolters FJ et al. J Alzheimers Dis 2017. DOI 10.3233/JAD-160871
36. Ntaios G et al. Eur Heart J 2024;45:1701–15. DOI 10.1093/eurheartj/ehae150
37. Lessmann N et al. IEEE Trans Med Imaging 2018;37:615–25. DOI 10.1109/TMI.2017.2769839
38. van Velzen SGM et al. Radiology 2020;295:66–79. DOI 10.1148/radiol.2020191621
39. Zeleznik R et al. Nat Commun 2021;12:715. DOI 10.1038/s41467-021-20966-2
40. Eng D et al. npj Digit Med 2021;4:88. DOI 10.1038/s41746-021-00460-1
41. Hagopian R et al. NEJM AI 2025;2(6). DOI 10.1056/AIoa2400937
42. Zheng J et al. JACC Adv 2026. DOI 10.1016/j.jacadv.2026.103277
43. Subramaniam A, Wu H, … Wilson DL. Sci Rep 2025. DOI 10.1038/s41598-025-34818-2; PMID 40502794
44. Nalepa J et al. arXiv 2501.11428 (2025)
45. Finnegan RN et al. Phys Eng Sci Med 2023. DOI 10.1007/s13246-023-01231-w; PlatiPy JOSS 2023 DOI 10.21105/joss.05374
46. Zöllner C et al. arXiv 2507.16573 (2025); data Zenodo 10.5281/zenodo.16274176
47. Wang M et al. eBioMedicine 2023;96:104794 [code availability unverified]
48. Mohammadi Kazaj P, … Shiri I. arXiv 2607.11287 (2026)
49. Pack JD et al. Vis Comput Ind Biomed Art 2022. DOI 10.1186/s42492-022-00125-0
50. Šprem J et al. PLoS One 2018;13:e0209318. DOI 10.1371/journal.pone.0209318
51. Wasserthal J et al. TotalSegmentator (Radiol Artif Intell 2023); task list github.com/wasserth/TotalSegmentator
52. He Y et al. VISTA3D, CVPR 2025; label list Project-MONAI/VISTA `label_dict.json`
53. Zhuang X et al. MM-WHS, Med Image Anal 2019 [volume unverified]
54. Lambert Z et al. SegTHOR, arXiv 1912.05950
55. Zeng A et al. ImageCAS, Comput Med Imaging Graph 2023
56. Shiyam Sundar LK et al. MOOSE, J Nucl Med 2022 [unverified]; repo ENHANCE-PET/MOOSE
57. Pai S et al. CT-FM, arXiv 2025; repo project-lighter/CT-FM [unverified]
