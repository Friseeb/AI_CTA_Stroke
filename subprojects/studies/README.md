# Studies Subproject

This area holds study-specific orchestration and reporting presets.

## Current Studies

- `la_laa` (`subprojects/studies/la_laa/`, canonical workflow in `subprojects/la_laa/`)
- `carotid` (`subprojects/studies/carotid/`)
- `intracranial` (`subprojects/studies/intracranial/`)

## Purpose

A study should primarily define:

- selected ROIs
- selected analysis modules
- output artifacts and QC thresholds

Study folders should avoid copying core segmentation/analysis logic.
