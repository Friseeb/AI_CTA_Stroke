"""stroke_cta_osa — CTA-derived airway/adiposity feature extraction for acute
stroke cohorts (research prototype).

Goal: produce reproducible, interpretable imaging features from routine head/neck
CTA that may represent an OSA-related upper-airway + cervical-adiposity phenotype.
The pipeline does NOT diagnose OSA. Downstream statistical association with
sleep-study outcomes (AHI/ODI/T90), wake-up stroke, AF/AFDAS, PFO, recurrence
and MACE is left to separate analysis code.

The package can run independently on CTA DICOM/NIfTI or reuse upper-airway
outputs from sibling pipelines (e.g. the dental/CBCT subproject) via adapters.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS. "
    "Features are exploratory imaging biomarkers; OSA cannot be inferred from CTA alone."
)

PIPELINE_NAME = "stroke_cta_osa"
