# stroke_cta_osa: Evidence-tiered CTA phenotyping of OSA-related airway, tongue, and neck adiposity in stroke

`stroke_cta_osa` is a **research feature-extraction pipeline**. It quantifies
CT/CTA-derived upper-airway, tongue, cervical and parapharyngeal fat, skeletal,
and optional vascular/metabolic anatomy from routine acute-stroke head/neck CT
angiograms, and it organises every output by the **strength of prior OSA imaging
evidence**. It is designed for acute stroke CTA but can reuse dental/CBCT airway
outputs through adapters. **It does not diagnose obstructive sleep apnoea (OSA).**
It creates evidence-tiered outputs for primary, secondary, cardiometabolic, and
exploratory analyses so that downstream work cannot accidentally mix
OSA-backed features with exploratory ones.

> The pipeline does not diagnose OSA. CTA-derived markers should trigger
> confirmatory sleep evaluation or research hypotheses — they do not replace
> polysomnography (PSG) or home sleep apnoea testing (HSAT).

## Why this exists

Obstructive sleep apnoea is mechanistically relevant to stroke onset and
outcome: nocturnal hypoxaemia, intrathoracic-pressure swings, autonomic surges,
and atrial remodelling connect OSA to wake-up stroke, atrial fibrillation and
AF-detected-after-stroke (AFDAS), PFO/right-to-left-shunt physiology,
small-vessel disease burden, recurrent stroke, and major adverse cardiovascular
events (MACE). Routine stroke CTA already images the upper airway and neck soft
tissues that govern pharyngeal collapsibility. The goal here is **not
opportunistic screening alone** — it is to identify an *OSA-related nocturnal
vascular-risk phenotype* from anatomy that is already acquired, and to express
it as reproducible, provenance-tracked features. Any positive signal is a
prompt for confirmatory sleep testing or a research hypothesis, never a
diagnosis.

## Evidence-tiered design

Every feature is assigned an **evidence tier** describing how strongly prior
adult OSA imaging literature supports it. This is orthogonal to the engineering
"analysis tier" used inside the metric registry.

| Tier | Name | Evidence meaning | Intended use | Examples |
|---|---|---|---|---|
| Tier 1 | Core OSA-backed | Prior adult OSA imaging support (CT/CBCT/MRI) | Primary CTA-OSA phenotype | airway min CSA, retropalatal/retroglossal CSA, tongue/mandible ratio, cervical fat, parapharyngeal fat, MP-H distance |
| Tier 2 | OSA-plausible CT anatomy | Anatomically grounded but not an established CT-OSA biomarker | Secondary/mechanistic analysis | retropharyngeal fat, submandibular fat, periairway shells, soft palate/lateral wall |
| Tier 3 | CT cardiometabolic/vascular | Established in cardiometabolic/vascular CT literature, not direct OSA anatomy | AF/MACE/stroke-risk models | C5 NAT compartments, pericarotid fat |
| Tier 4 | Novel stroke-CTA exploratory | New engineered/radiomic/model features | Hypothesis generation | fat-to-airway ratios, untrained scores, radiomics |

The single most important invariant: **Tier 1 stays clean.** The core feature
set contains *only* features with prior OSA imaging support. Novel CTA features
are valuable, but they are labelled Tier 2/3/4 and never enter the primary set.

## Core features (Tier 1)

**Airway** — minimum cross-sectional area; retropalatal CSA/volume;
retroglossal/retrolingual CSA/volume; airway volume; airway length; AP/lateral
diameters at min CSA; eccentricity.

**Tongue / mandible** — tongue volume; mandible volume; tongue/mandible volume
ratio; posterior tongue attenuation / low-HU fraction; tongue-base-to-
retroglossal-airway ratio; tongue-to-skeletal-enclosure ratio.

**Fat** — cervical fat volume and mean HU; internal vs subcutaneous neck-fat
proxy; pharyngeal/peripharyngeal fat; parapharyngeal fat-pad volume/area;
level-specific parapharyngeal fat at retropalatal, retroglossal, and
subglosso-supraglottic levels; parapharyngeal-to-airway ratios.

**Skeletal** — mandibular-plane-to-hyoid (MP-H) distance; hyoid position;
cervicomandibular ring area; hyoid-to-posterior-pharyngeal-wall distance.

## Secondary / anatomic extension features (Tier 2)

Plausible and CT-anatomy-based, but not the primary evidence-backed OSA set:
retropharyngeal fat; submandibular/submental fat; surface-distance shell fat;
supraplatysmal/subplatysmal **proxy** fat; periairway distance-shell fat; soft
palate length/thickness/volume; lateral pharyngeal wall thickness/asymmetry;
uvula and tonsil volumes. Use these for mechanism discovery, not the primary
phenotype.

## Cardiometabolic and vascular extension features (Tier 3)

Separated from the core OSA-backed set: C5 compartmental neck adipose tissue
(subcutaneous, posterior intermuscular, perivertebral, internal); pericarotid
fat; and thoracic/cardiac fat **only if** the relevant mask is supplied. These
are most appropriate for stroke / MACE / AF / AFDAS / metabolic-risk modelling
rather than direct OSA-anatomy claims.

## Inputs

- DICOM CTA folder, or NIfTI CT/CTA
- Optional masks: airway, tongue, mandible, hyoid, soft palate, fat
  compartments, carotid
- Optional landmarks JSON/CSV
- Optional dental/CBCT pipeline outputs (airway mask, landmarks, features,
  jawbone mask) — discovered via `--dental-artifacts-dir`
- Optional clinical/outcome CSV (merged separately, never required)

## Outputs

| File | Contents |
|---|---|
| `features.csv` | all stable columns, one row per CTA |
| `features_core_osa_backed.csv` | Tier 1 + identifiers/QC |
| `features_core_plus_anatomic_extensions.csv` | Tier 1 + Tier 2 |
| `features_core_plus_cardiometabolic_ct.csv` | Tier 1 + Tier 3 |
| `features_all_exploratory.csv` | all implemented features (Tier 1-4) |
| `qc.csv` | per-case QC flags + coverage score |
| `feature_metadata.json` | full metric + evidence registry metadata |
| `feature_evidence_summary.csv` | per-feature evidence tier/class/role/references |
| `feature_missingness_by_tier.csv` | per-tier availability/missingness |
| optional `<case>/mask_*.nii.gz`, `<case>/qc_*.png` | masks, QC overlays |

Missing optional features appear as **NA/null** in the tiered CSVs, not as
absent columns — the schema is stable across cohorts.

## Example CLI usage

```bash
stroke-cta-osa extract \
  --input /path/to/case \
  --output-dir /path/to/output \
  --config configs/stroke_cta_osa_default.yaml \
  --feature-set core_osa_backed

stroke-cta-osa batch \
  --manifest /path/to/manifest.csv \
  --output-dir /path/to/output \
  --feature-set core_plus_anatomic_extensions \
  --save-qc-images

stroke-cta-osa list-features \
  --feature-set core_osa_backed \
  --output feature_dictionary_core.csv
```

(The installed CLI uses `--out` for the output directory with a positional
input path; run `stroke-cta-osa extract --help` for the exact flags in your
build.)

## Recommended analysis strategy

**Primary.** Clinical variables only, then `core_osa_backed` features. Test
against sleep-study OSA where available: AHI/REI ≥ 15, AHI/REI ≥ 30, ODI,
hypoxic burden, minimum SpO₂, T90.

**Secondary.** Wake-up vs non-wake-up stroke; AF/AFDAS; ESUS/cardioembolic
phenotype; PFO/right-to-left-shunt interaction; small-vessel-disease burden;
recurrent stroke / MACE / mortality.

**Exploratory.** Tier 2-4 features, radiomics, untrained composite scores,
pericarotid/cardiometabolic CT features.

## Limitations

- CTA is awake/static imaging and does **not** capture sleep-state dynamic
  airway collapse.
- CTA contrast phase shifts HU-based tongue, soft-tissue, vessel, and fat
  attenuation metrics; contrast sensitivity is flagged per feature.
- Dental artifact degrades tongue-base, parapharyngeal, submandibular, and
  airway measurements.
- OSA diagnosis still requires PSG, HSAT, or clinically accepted sleep testing.
- Feature confidence and missingness must be considered in analysis.
- Some features are **proxies** unless true segmentation masks exist (e.g.
  supra-/subplatysmal fat without a platysma mask).
- **AirwayNet-MM-H pretrained weights are not assumed to be available**; the
  pipeline references it only as a benchmark concept and trains nothing.

## Evidence references

| Reference tag | Domain | Why it matters | Citation / link |
|---|---|---|---|
| `Barkdull_2008_CT_OSA` | CT airway, retrolingual airway, MP-H, cervicomandibular ring, posterior tongue HU | Foundational CT cephalometric/airway OSA markers | https://pubmed.ncbi.nlm.nih.gov/18528305/ |
| `Shigeta_2011_Tongue_Mandible_CT` | 3D CT tongue volume, mandible volume, tongue/mandible ratio | Tongue-to-mandible volume ratio and OSA | https://pmc.ncbi.nlm.nih.gov/articles/PMC3026324/ · https://pubmed.ncbi.nlm.nih.gov/21237441/ |
| `Chen_2019_Parapharyngeal_Fat_DI_SLEEP_CT` | parapharyngeal fat-pad areas, subglosso-supraglottic level, lateral-wall collapse, AHI | Level-specific parapharyngeal fat and AHI | https://www.nature.com/articles/s41598-019-53515-5 · https://pubmed.ncbi.nlm.nih.gov/31776365/ |
| `Ernst_2023_Cervical_Fat_Tissue_Volume` | CT cervical fat tissue volume and moderate-to-severe OSA | Cervical fat volume as an OSA-severity marker | https://pmc.ncbi.nlm.nih.gov/articles/PMC10773506/ · https://pubmed.ncbi.nlm.nih.gov/38196763/ |
| `Shelton_1993_Pharyngeal_Fat_OSA` | adipose tissue adjacent to pharyngeal airway and OSA severity | Pharyngeal adipose tissue ↔ OSA | https://pubmed.ncbi.nlm.nih.gov/8342912/ |
| `Torriani_2014_C5_Neck_Adipose_Tissue` | C5 compartmental neck adipose tissue and metabolic/CVD risk | Cardiometabolic neck-fat compartments | https://pubmed.ncbi.nlm.nih.gov/25332322/ |
| `AirwayNet_MMH_2024` | deep learning OSA prediction from CT | Benchmark concept only; no local pretrained model assumed | https://pubmed.ncbi.nlm.nih.gov/38471111/ |
| `Zhang_2022_Upper_Airway_CT_DL` | 3D upper-airway CT deep learning for moderate-to-severe OSA | Upper-airway CT supports OSA prediction | https://jtd.amegroups.org/article/view/70727/html |

## Repository status / maturity

- Research prototype.
- **Not** FDA / Health Canada / CE cleared.
- Not for clinical decision making.
- Requires local validation.
- Use only with IRB/REB-approved data governance.

## Developer notes

To add a new feature:

1. Add a `MetricSpec` to `metric_registry.py` (the column contract: name,
   unit, missingness behaviour).
2. Add an `EvidenceSpec` to `evidence_registry.py` and assign an
   `evidence_tier` **and** `evidence_class` — this places it in a feature set.
3. Document its unit and missingness behaviour; never rename published columns.
4. Add tests (registry completeness, feature-set membership, output presence).
5. Never log PHI; identifiers in CSV are hashed.

## Document index

| Doc | Purpose |
| --- | --- |
| [EVIDENCE_TIERS.md](EVIDENCE_TIERS.md) | The four evidence tiers, classes, and analysis roles |
| [FEATURE_SETS.md](FEATURE_SETS.md) | The four canonical feature sets and how to select them |
| [FAT_COMPARTMENTS.md](FAT_COMPARTMENTS.md) | Fat compartments by evidence tier; proxy vs anatomic |
| [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) | Primary / secondary / exploratory analysis plan |
| [CT_OSA_METRICS.md](CT_OSA_METRICS.md) | High-level map of metric families + the registry contract |
| [FEATURES.md](FEATURES.md) | Per-column reference: type, unit, method, missing-value behaviour |
| [LANDMARKS.md](LANDMARKS.md) | Landmark schema, provider chain, validation |
| [TONGUE_METRICS.md](TONGUE_METRICS.md) | Tongue volume + HU stats + posterior ROI + tongue-base encroachment |
| [SOFT_TISSUE_METRICS.md](SOFT_TISSUE_METRICS.md) | Soft palate, uvula, palatine tonsils, lateral pharyngeal wall |
| [SKELETAL_METRICS.md](SKELETAL_METRICS.md) | Hyoid distances, neck length, laryngeal descent, cervicomandibular ring |
| [FAT_METRICS.md](FAT_METRICS.md) | Global + level-anchored + per-side fat areas |
| [COMPOSITES.md](COMPOSITES.md) | Exploratory `_untrained` composite indices |
| [CLI.md](CLI.md) | Subcommands + mask/landmark CLI options |
| [QC.md](QC.md) | Quality-control checks, coverage score, artefact heuristics |
| [DENTAL_PIPELINE_INTEGRATION.md](DENTAL_PIPELINE_INTEGRATION.md) | What is shared with the dental subproject, what is CTA-specific |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Known unknowns, calibration TODOs, validation gates |

## Research-only disclaimer

All feature values, composite scores, and adapter outputs are **research
prototypes**. None of them — including the composite `*_untrained` columns —
should be used for patient-care decisions. The pipeline is designed for
retrospective cohort analysis where imaging features are entered into
statistical models alongside formal sleep-study and clinical-outcome data.
