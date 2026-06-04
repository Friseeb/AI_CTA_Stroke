# Protocol Index

Current protocol documents:

- `docs/protocols/laa_highres_dataset_setup.md`
- `docs/protocols/script_setup.md`
- `docs/protocols/topcow_claim_inference.md`

Primary workflow entrypoint:

- `README.md` (master DICOM -> NIfTI -> deface -> substudies flow)
- `subprojects/la_laa/README.md` (LA/LAA substudy home)
- `scripts/README.md` (script map + naming rubric)
- `docs/architecture/ORGANIZATION_BLUEPRINT.md` (target organization)
- `docs/architecture/MIGRATION_MAP.md` (incremental transition map)
- `configs/profiles/README.md` (study/analysis profiles)

Path policy:

- Use placeholders like `<PROJECT_ROOT>`, `<DATA_ROOT>`, `<BIDS_ROOT>`, `<CASE_ID>`
- Avoid machine-specific absolute paths in docs and scripts
