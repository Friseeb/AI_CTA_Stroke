# LAA Completion & SLAAO Annotation Assistant (3D Slicer)

Annotator-facing module for recovering the **complete** left atrial appendage
(LAA) from CTA — including the distal hypoattenuated region that VISTA-3D /
TotalSegmentator / NUDF routinely truncate, which overlaps the **SLAAO Type 1**
phenotype of interest.

Pilot-first: the goal is a reproducible annotation SOP, feasibility, timing, and
interobserver reproducibility (Phase 0) before any model fine-tuning (Phase 2+).

## Files

- `laa_annotation_core.py` — pure-Python core (no Slicer imports): label
  contract, prompt schema, pilot/reproducibility metrics, session logging,
  output layout, MONAILabel request builder. Unit-tested in `../tests/`.
- `LAACompletionAssistant.py` — `ScriptedLoadableModule` GUI (category
  `CTA-in-AI`) that drives the 7-step workflow and delegates logic to the core.

## Install (Slicer)

1. `Edit → Application Settings → Modules → Additional module paths` → add this
   `slicer_module/` folder. Restart Slicer.
2. Open `Modules → CTA-in-AI → LAA Completion Assistant`.

## Label contract (`laa_completion_v1`)

| Label | Name | Notes |
| ----- | ---- | ----- |
| 1 | Whole LAA | **Primary target** — ostium to distal tip, all lobes + distal hypoattenuated regions |
| 2 | SLAAO Type 1 region | Geometric distal subregion, nested in 1; may be empty |
| 3 | LA body | Optional (training / error analysis) |
| 4 | Pulmonary veins | Optional |
| 5 | Coronary artery | Optional |
| 6 | Aorta / pulmonary artery | Optional |
| 7 | Other hard-negative | Optional |

## 7-step workflow

1. **Case** — load CTA (NIfTI/DICOM), set Case ID / Reader ID / output folder.
2. **LAA candidate** — import VISTA-3D / NUDF / external MONAI output, or create
   an empty segmentation. The AI output is never assumed correct.
3. **Long-axis workspace** — axial/sagittal/coronal + an LAA long-axis reference;
   annotate in long-axis, not pure axial.
4. **Prompts** — place positive (distal_tip, distal_lobe, missed_appendage,
   type1_region) / negative (la_body, pulmonary_vein, coronary_artery, aorta,
   pulmonary_artery, myocardium, artifact) point prompts. Each is logged.
5. **MONAILabel update (optional)** — send CTA + current mask + prompts to a
   MONAILabel server and load the result. The module is fully usable without it.
6. **Manual correction** — Segment Editor (Paint/Erase/Scissors/Islands/Smoothing).
7. **Type 1** — paint the geometric distal hypoattenuated subregion as label 2.
   Do **not** apply HU thresholds — HU analysis is downstream.

**Finalize** writes the `laa_annotation/` output tree (see below), pilot metrics,
prompt log, and per-case session log (JSON + appended CSV).

## Modes

- **Pilot** — leave Reader ID blank or use initials; collect timing, prompt
  counts, confidence, and image quality for 10–15 cases.
- **Reproducibility** — set Reader ID to `readerA` / `readerB` / `readerC`;
  outputs nest per-reader so masks stay separate. Then run the reproducibility
  script (below) for Dice / Surface Dice / HD95.

## Output tree

```
<case_dir>/laa_annotation/[<reader_id>/]
  candidate_masks/  manual_masks/  type1_masks/
  iterations/       logs/          screenshots/  metrics/
```

## Reproducibility metrics (offline)

```bash
python scripts/run_laa_reproducibility.py \
  --manifest derivatives/laa_pilot/repro_manifest.csv \
  --out-dir  derivatives/laa_pilot/metrics
```
Manifest columns: `case_id,reader_id,mask_path` (`case_id` inferred if blank).

## Tests

```bash
cd subprojects/la_laa && python -m pytest tests/test_laa_annotation_core.py -q
```

See `../docs/SOP.md` for the annotation standard operating procedure.
