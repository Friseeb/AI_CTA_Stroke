# CLI

The `stroke_cta_osa` CLI is the canonical way to run the pipeline. The Python
API is available too (`from stroke_cta_osa import extract_case`), but the CLI
is what the unit tests, dental-integration scripts, and IRB documentation
assume.

## Subcommands

| Command | Purpose |
|---|---|
| `extract` | Process one CTA → row + masks + JSON |
| `batch` | Process a CSV manifest of cases |
| `qc` | Run QC only (no feature extraction), emit `qc.csv` row |
| `summarize` | Aggregate a directory of per-case rows into one CSV |
| `compare-dental` | Side-by-side comparison of CTA vs dental-pipeline shared columns |
| `list-features` | Export the metric registry as CSV / JSON / table |
| `validate-landmarks` | Lint a landmark JSON against the canonical schema |

Run `stroke_cta_osa <cmd> --help` for full options.

## Parallel batch (`--workers`)

`batch` runs cases in independent processes. Worker count is **memory-aware**,
because even after the memory fixes a single large head/neck CTA peaks ~8 GB
(≈9× its raw voxel bytes) during the body-silhouette / fat stages — it is
RAM-bound, not CPU-bound. (It peaked ~21 GB before the fixes; the worker
estimator is calibrated to the current ~9× figure.)

```
stroke-cta-osa batch /data/cohort --out outputs/run --glob "*.nii.gz" --workers auto
stroke-cta-osa batch /data/cohort --out outputs/run --workers 4   # explicit cap
stroke-cta-osa batch /data/cohort --out outputs/run --workers 1   # force sequential
```

- **`auto` (default):** estimates each input's peak from its *raw voxel count*
  (read from the NIfTI/DICOM header, so compression doesn't fool it), takes the
  largest case, and picks `workers = min(usable_RAM / largest_peak, CPU,
  n_cases)`. Usable RAM is 85% of available.
- **Explicit `N`:** still clamped down if RAM can't hold `N` copies of the
  largest case — it never over-commits memory.
- **Thread oversubscription** is avoided: cores are divided across workers and
  per-worker BLAS/ITK thread caps (`OMP_NUM_THREADS`, `ITK_…`, etc.) are
  exported before the pool spawns.
- The pool uses the `spawn` start method to avoid fork + threaded-ITK deadlocks;
  one bad case is reported and skipped without killing the run.

The chosen plan is printed at startup, e.g.
`workers: 4 × 4 thread(s)  ·  auto: memory-bound; largest-case peak≈18.0GB …`.

### `--precompute-airway` (two-pass)

```
stroke-cta-osa batch /data/cohort --out outputs/run --precompute-airway
```

Runs the airway segmentation **first** for all cases, caching each mask under
`out/_airway_cache/`, then runs feature extraction reusing those masks. Its
value is **reuse and decoupling**: feature re-runs and config sweeps don't
recompute the airway segmentation, and the airway stage can later be swapped
for a different segmenter. It does *not* change worker count — after the memory
fixes the in-process airway path and the supplied-mask path both peak ~8 GB, so
single-pass runs are already as wide as two-pass ones.

## Common mask + landmark options

The single-case `extract` and the manifest-driven `batch` accept the same
mask / landmark hand-offs:

```
--external-airway-mask PATH
--external-tongue-mask PATH
--external-mandible-mask PATH
--external-soft-palate-mask PATH
--external-oral-cavity-mask PATH
--landmarks PATH               # JSON landmark bundle
--dental-landmarks PATH        # dental adapter JSON (lower priority)
```

When both `--landmarks` and `--dental-landmarks` are supplied, the explicit
file wins on every key it sets.

## `list-features`

```
stroke_cta_osa list-features --format csv  > metrics.csv
stroke_cta_osa list-features --format json > metrics.json
stroke_cta_osa list-features --format table         # human-readable
```

Filters:
```
--tier {tier1,tier2,exploratory}
--family <name>                # e.g. tongue, fat, airway_regions
--shared-with-dental
```

Use these to verify the schema downstream before you commit to consuming a
column.

## `validate-landmarks`

```
stroke_cta_osa validate-landmarks --landmarks landmarks.json [--image CTA.nii.gz]
```

Prints:

1. Counts of populated points, z-levels, planes vs the canonical schema.
2. Any warnings from `validate_landmarks()` (shape mismatch, out-of-bounds,
   degenerate planes).

Always non-fatal — the warnings tell you what's wrong, but the command exits
0 unless the JSON itself can't be loaded.

## Tongue + posterior ROI fallback

By default `compute_tongue_features` refuses to invent a landmark-only
posterior tongue ROI. Pass `--allow-tongue-posterior-roi-fallback` (or set
`tongue.allow_posterior_roi_fallback=true` in YAML config) to enable it. The
ROI will be flagged `tongue_roi_confidence='low'` and the source string
records whether the anchor was a landmark or the airway's min-CSA slice.

## Reading the YAML config

Every CLI option corresponds to a key under one of the per-module config
blocks. See [stroke_cta_osa/config.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/config.py)
for the Pydantic schema. Module-specific defaults are documented in the
respective metric docs ([TONGUE_METRICS.md](TONGUE_METRICS.md),
[FAT_METRICS.md](FAT_METRICS.md), etc).
