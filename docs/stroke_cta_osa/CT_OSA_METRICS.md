# CT OSA Metrics — overview

The `stroke_cta_osa` pipeline exports a single canonical schema described by the
`metric_registry`. The registry is the source of truth for feature names,
units, missingness behaviour, tier, and maturity; the CSV and JSON outputs are
both derived from it.

This document gives a high-level map of the metric families and points to the
per-family docs for the geometry details. Use it to plan a phenotyping
analysis or to verify that a downstream consumer is reading the columns it
expects.

## Where the registry lives

| File | Purpose |
|---|---|
| [stroke_cta_osa/metric_registry.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/metric_registry.py) | Builds the immutable tuple of `MetricSpec`s |
| [stroke_cta_osa/output.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/output.py) | Uses `empty_row()` to start every row from the registry's defaults |
| [tests/test_metric_registry.py](../../subprojects/stroke-cta-osa/tests/test_metric_registry.py) | Pins down the contract |

The CLI subcommand `list-features` dumps the registry in CSV / JSON / table
form — see [CLI.md](CLI.md).

## Metric families

Each row in the output CSV/JSON belongs to exactly one family. Families are
grouped by anatomical region and method maturity:

| Family | Doc | Description |
|---|---|---|
| `identifiers` / `qc` | [QC.md](QC.md) | study/scan IDs, QC pass flags, per-region coverage |
| `airway` (global) | [FEATURES.md](FEATURES.md) | volume, length, min CSA, AP/lateral, eccentricity |
| `airway_regions` | this doc | five compartments + standard-level CSAs + shape at min CSA |
| `tongue` | [TONGUE_METRICS.md](TONGUE_METRICS.md) | volume, HU stats, posterior-tongue ROI, tongue-base encroachment |
| `mandible` / `oral_cavity` | this doc | mandible mask volume + mandibular-plane geometry |
| `soft_palate` / `uvula` / `tonsil` | [SOFT_TISSUE_METRICS.md](SOFT_TISSUE_METRICS.md) | masks-or-landmarks length/thickness; lateral wall thickness |
| `skeletal` | [SKELETAL_METRICS.md](SKELETAL_METRICS.md) | hyoid distances, neck length, laryngeal descent, cervicomandibular ring |
| `fat_*` | [FAT_METRICS.md](FAT_METRICS.md) | global + level-anchored + per-side fat areas |
| `composite` | [COMPOSITES.md](COMPOSITES.md) | exploratory `_untrained` indices (off by default) |
| `optional` / `radiomics` | [FEATURES.md](FEATURES.md) | per-ROI radiomics, gated on PyRadiomics availability |

## Tier and maturity

Each `MetricSpec` carries two flags that downstream callers can filter on:

* `Tier.TIER1` — high-value, well-defined, expected for every case;
* `Tier.TIER2` — anatomically motivated but more sensitive to mask quality;
* `Tier.EXPLORATORY` — composites and experimental ratios — always treat as
  unvalidated.

* `Maturity.STABLE` — geometry/code unlikely to change;
* `Maturity.HEURISTIC` — works by rules of thumb (e.g. body silhouette);
* `Maturity.EXPERIMENTAL` — pre-validation; may rename or vanish.

## Missingness contract

Every column appears in every row. Missingness is encoded by *type-stable*
default values controlled by the `MetricSpec.missingness_behaviour`:

| Behaviour | Default value | Used for |
|---|---|---|
| `nan_float` | `float('nan')` | volumes, distances, HU stats, ratios |
| `bool_False` | `False` | availability flags |
| `empty_str` | `""` | provenance/method strings |
| `-1_int` | `-1` | counts (so 0 has meaning) |

Consumers must treat NaN as **missing**, not as **zero**. A `0.0` in the
output means we measured the structure and it was actually zero.

## Shared with the dental subproject

A subset of features is intentionally compatible with our sibling dental/CBCT
pipeline so cross-cohort comparisons are exactly that — comparisons of the
same number, not similar numbers under similar names. The `shared_with_dental`
flag in the registry marks every such column, and the CLI `compare-dental`
subcommand walks both schemas side by side. See
[DENTAL_PIPELINE_INTEGRATION.md](DENTAL_PIPELINE_INTEGRATION.md).
