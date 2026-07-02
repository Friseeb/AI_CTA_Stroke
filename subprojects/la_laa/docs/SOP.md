# SOP — LAA Completion & SLAAO Type 1 Annotation

Standard operating procedure for the Phase-0 pilot and reproducibility study.
The objective is to establish a **reproducible** annotation procedure, measure
feasibility / time / interobserver agreement, and produce training-ready data —
**before** any model fine-tuning.

Tooling: `LAA Completion Assistant` 3D Slicer module (see
`../slicer_module/README.md`).

---

## 1. Annotation targets

### Label 1 — Whole LAA (primary)
From the **ostial boundary** to the most distal appendage tip.

- **Include:** dominant lobe, secondary lobes, distal hypoattenuated regions.
- **Exclude:** LA body, pulmonary veins, coronary arteries, myocardium, aorta,
  pulmonary artery.

**Ostial boundary definition.** Use the plane connecting the circumflex artery /
left superior pulmonary vein ridge ("warfarin ridge") to the opposing
mitral-annulus side of the orifice. When ambiguous, prefer the LA-side limit so
the appendage is captured completely; note the uncertainty in the case notes.

### Label 2 — SLAAO Type 1 region (nested, may be empty)
The **geometric** distal hypoattenuated subregion, fully inside Label 1.

- The reader identifies it **geometrically** (where contrast fails to fill /
  the appendage is truncated by AI). Do **not** apply HU thresholds — quantitative
  HU metrics are computed downstream.
- If no distinct Type 1 region exists, leave Label 2 empty and set Type 1
  confidence accordingly.

### Labels 3–7 (optional during pilot)
LA body, pulmonary veins, coronary artery, aorta/PA, other hard-negatives.
Annotate only if time permits; they support future training / error analysis.

---

## 2. Per-case procedure

1. **Load** the CTA and set Case ID + Reader ID + output folder. **Start timer.**
2. **Import the AI candidate** (VISTA-3D / NUDF / MONAI) or create an empty
   segmentation. Treat the candidate as a draft, not ground truth.
3. **Set up the long-axis workspace.** Align the long-axis view down the LAA.
   Annotate primarily in the **long-axis** view, cross-checking axial.
4. **Place prompts** marking what the candidate got wrong:
   - Positive: `distal_tip`, `distal_lobe`, `missed_appendage`, `type1_region`.
   - Negative: `la_body`, `pulmonary_vein`, `coronary_artery`, `aorta`,
     `pulmonary_artery`, `myocardium`, `artifact`.
5. *(Optional)* **MONAILabel update** to propose a refined mask from the prompts.
6. **Manually correct** Label 1 with the Segment Editor until the whole LAA is
   captured to the ostial boundary.
7. **Paint Label 2** for the geometric Type 1 region (if present).
8. **Finalize**: rate segmentation confidence, Type 1 confidence (0–1), and image
   quality (1–5), add notes, then finalize. The module writes masks, prompt log,
   pilot metrics, and the session log.

---

## 3. Rating scales

- **Segmentation confidence (0–1):** reader's confidence the whole-LAA mask is
  complete and correct.
- **Type 1 confidence (0–1):** confidence in the presence/extent of the Type 1
  region (use a low value when leaving Label 2 empty due to uncertainty).
- **Image quality (1–5):** 1 = non-diagnostic, 2 = poor, 3 = adequate,
  4 = good, 5 = excellent.

---

## 4. Pilot study (Phase 0a)

Target **10–15 cases**. The module records per case: annotation time, correction
time, prompt counts (total / positive / negative), edit count, both confidences,
image quality, and whether a Type 1 region was present. Review the aggregated
`logs/*_session.csv` to assess feasibility and time per case.

## 5. Reproducibility study (Phase 0b)

Readers **A, B, C** annotate the same cases independently. Set Reader ID to
`readerA` / `readerB` / `readerC` so outputs nest per reader. Then compute
agreement:

```bash
python scripts/run_laa_reproducibility.py \
  --manifest derivatives/laa_pilot/repro_manifest.csv \
  --out-dir  derivatives/laa_pilot/metrics
```

Reported per case pair and aggregated: **Dice**, **Surface Dice** (1 mm
tolerance), **HD95** (mm). Establish acceptable reproducibility before scaling
annotation or starting model fine-tuning.

---

## 6. What this SOP is **not**

- Not generic left-atrium segmentation (the LA is already segmented robustly).
- Not HU thresholding by the reader (Type 1 is geometric; HU is downstream).
- Not a model-training step — training begins only after Phase 0 succeeds.
