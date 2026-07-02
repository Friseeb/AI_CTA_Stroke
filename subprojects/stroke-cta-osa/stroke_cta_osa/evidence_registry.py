"""Evidence-tiered feature ontology for stroke_cta_osa.

This module sits **orthogonal** to :mod:`stroke_cta_osa.metric_registry`.

* ``metric_registry`` answers *"what columns can the pipeline emit, in what
  unit, with what extraction maturity?"* — it is the column contract.
* ``evidence_registry`` answers *"how strong is the prior OSA-imaging evidence
  for this feature, and which analysis is it allowed to drive?"* — it is the
  scientific provenance contract.

The two axes are deliberately separate. ``metric_registry.Tier`` is an
*engineering/analysis* tier (tier1 = robust extraction, exploratory = research
hook). ``evidence_registry.EvidenceTier`` is a *prior-literature* tier
(TIER_1_CORE_OSA_BACKED = has direct adult-OSA imaging support).

The single most important invariant of the whole subproject:

    **Keep Tier 1 clean.** ``TIER_1_CORE_OSA_BACKED`` must contain *only*
    features with prior adult OSA imaging support (CT / CBCT / MRI). Novel
    CTA/stroke features are valuable but must be labelled Tier 2/3/4 so a
    downstream analyst cannot accidentally mix them into the primary phenotype.

Every entry carries the metadata required to (a) build the four canonical
feature sets in :mod:`stroke_cta_osa.feature_sets`, (b) emit
``feature_evidence_summary.csv`` and ``feature_metadata.json``, and (c) drive
the documentation tables. Some entries describe *planned* feature names that
the pipeline does not emit yet (``implemented=False``); these still appear in
the ontology and as NA columns in the tiered subset CSVs so that downstream
schemas are stable as the pipeline grows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from . import metric_registry as mr


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class EvidenceTier(str, Enum):
    """Strength of prior OSA-imaging evidence for a feature.

    * ``TIER_1_CORE_OSA_BACKED`` — direct adult OSA imaging support
      (CT/CBCT/MRI). Eligible for the *primary* CTA-OSA phenotype.
    * ``TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC`` — anatomically grounded CT/CTA
      features plausible for OSA but not established CT-OSA biomarkers.
    * ``TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR`` — CT adiposity / vascular
      features documented in cardiometabolic / cardiovascular literature, not
      primarily OSA-anatomy metrics.
    * ``TIER_4_STROKE_CTA_NOVEL_EXPLORATORY`` — novel engineered CTA/stroke
      ratios, scores, asymmetries, radiomics, or model outputs.
    """
    TIER_1_CORE_OSA_BACKED = "TIER_1_CORE_OSA_BACKED"
    TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC = "TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC"
    TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR = "TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR"
    TIER_4_STROKE_CTA_NOVEL_EXPLORATORY = "TIER_4_STROKE_CTA_NOVEL_EXPLORATORY"


class EvidenceClass(str, Enum):
    """What *kind* of prior evidence (or lack thereof) backs a feature."""
    OSA_CT_DIRECT = "OSA_CT_DIRECT"
    OSA_CBCT_DIRECT = "OSA_CBCT_DIRECT"
    OSA_MRI_DIRECT = "OSA_MRI_DIRECT"
    OSA_IMAGING_INDIRECT = "OSA_IMAGING_INDIRECT"
    CT_ANATOMY_DIRECT_NO_OSA = "CT_ANATOMY_DIRECT_NO_OSA"
    CT_CARDIOMETABOLIC_DIRECT_NO_OSA = "CT_CARDIOMETABOLIC_DIRECT_NO_OSA"
    CTA_STROKE_NOVEL = "CTA_STROKE_NOVEL"
    ENGINEERED_PROXY = "ENGINEERED_PROXY"
    RADIOMICS_EXPLORATORY = "RADIOMICS_EXPLORATORY"
    MODEL_OUTPUT_EXPLORATORY = "MODEL_OUTPUT_EXPLORATORY"


class AnalysisRole(str, Enum):
    """How a feature is *allowed* to be used in analysis."""
    PRIMARY_CANDIDATE = "primary_candidate"
    SECONDARY_CANDIDATE = "secondary_candidate"
    MECHANISTIC_SECONDARY = "mechanistic_secondary"
    CARDIOMETABOLIC_SECONDARY = "cardiometabolic_secondary"
    EXPLORATORY = "exploratory"
    DO_NOT_MODEL_WITHOUT_VALIDATION = "do_not_model_without_validation"


# Canonical feature-set names (mirrored in feature_sets.py).
FS_CORE = "core_osa_backed"
FS_CORE_PLUS_ANATOMIC = "core_plus_anatomic_extensions"
FS_CORE_PLUS_CARDIOMETABOLIC = "core_plus_cardiometabolic_ct"
FS_ALL = "all_features_exploratory"


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSpec:
    """Scientific-provenance metadata for one feature.

    See the module docstring for the design intent. ``required_masks`` /
    ``optional_masks`` / ``required_landmarks`` mirror (but are coarser than)
    the engineering ``required_inputs`` on :class:`metric_registry.MetricSpec`;
    they exist so the evidence table reads cleanly for a clinical reviewer.
    """
    feature_name: str
    feature_family: str
    anatomical_region: str
    unit: str
    evidence_tier: EvidenceTier
    evidence_class: EvidenceClass
    analysis_role: AnalysisRole
    true_anatomic_vs_proxy: str  # "anatomic" | "proxy" | "mixed"
    prior_osa_link: str = "no"           # "yes" | "indirect" | "no"
    prior_ct_link: str = "no"
    prior_cbct_link: str = "no"
    prior_mri_link: str = "no"
    prior_anatomic_link: str = "no"
    prior_cardiometabolic_link: str = "no"
    stroke_cta_novelty: str = "low"      # "low" | "moderate" | "high"
    extraction_method: str = ""
    confidence_field_name: str = ""
    contrast_sensitive: bool = False
    artifact_sensitive: bool = False
    missingness_behavior: str = "NA"     # "NA" | "bool_False" | "empty_str" | "-1_int"
    required_masks: tuple[str, ...] = ()
    optional_masks: tuple[str, ...] = ()
    required_landmarks: tuple[str, ...] = ()
    reference_tags: tuple[str, ...] = ()
    comments: str = ""
    implemented: bool = True
    aliases: tuple[str, ...] = ()

    @property
    def recommended_feature_set(self) -> str:
        return _TIER_TO_SET[self.evidence_tier]


_TIER_TO_SET = {
    EvidenceTier.TIER_1_CORE_OSA_BACKED: FS_CORE,
    EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC: FS_CORE_PLUS_ANATOMIC,
    EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR: FS_CORE_PLUS_CARDIOMETABOLIC,
    EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY: FS_ALL,
}


# ---------------------------------------------------------------------------
# Compact builders
# ---------------------------------------------------------------------------

def _unit_for(name: str, fallback: str) -> str:
    """Pull the canonical unit from the metric_registry when the name exists."""
    spec = mr.find(name)
    return spec.unit if spec is not None else fallback


def _spec(
    name: str, family: str, region: str, *,
    tier: EvidenceTier, evidence_class: EvidenceClass, role: AnalysisRole,
    anatomic: str, unit: str = "", references: Iterable[str] = (),
    osa: str = "no", ct: str = "no", cbct: str = "no", mri: str = "no",
    anatomic_link: str = "no", cardiometabolic: str = "no",
    novelty: str = "low", method: str = "", confidence_field: str = "",
    contrast: bool = False, artifact: bool = False,
    required_masks: Iterable[str] = (), optional_masks: Iterable[str] = (),
    required_landmarks: Iterable[str] = (),
    implemented: bool = True, aliases: Iterable[str] = (),
    comments: str = "",
) -> EvidenceSpec:
    resolved_unit = unit or _unit_for(name, "ratio")
    base = mr.find(name)
    miss = base.missingness_behaviour if base is not None else "NA"
    return EvidenceSpec(
        feature_name=name, feature_family=family, anatomical_region=region,
        unit=resolved_unit, evidence_tier=tier, evidence_class=evidence_class,
        analysis_role=role, true_anatomic_vs_proxy=anatomic,
        prior_osa_link=osa, prior_ct_link=ct, prior_cbct_link=cbct,
        prior_mri_link=mri, prior_anatomic_link=anatomic_link,
        prior_cardiometabolic_link=cardiometabolic, stroke_cta_novelty=novelty,
        extraction_method=method or (base.extraction_method if base else ""),
        confidence_field_name=confidence_field,
        contrast_sensitive=contrast, artifact_sensitive=artifact,
        missingness_behavior=miss,
        required_masks=tuple(required_masks), optional_masks=tuple(optional_masks),
        required_landmarks=tuple(required_landmarks),
        reference_tags=tuple(references), comments=comments,
        implemented=implemented, aliases=tuple(aliases),
    )


# Reference-tag shorthands -------------------------------------------------
R_BARKDULL = "Barkdull_2008_CT_OSA"
R_SHIGETA = "Shigeta_2011_Tongue_Mandible_CT"
R_CHEN = "Chen_2019_Parapharyngeal_Fat_DI_SLEEP_CT"
R_ERNST = "Ernst_2023_Cervical_Fat_Tissue_Volume"
R_SHELTON = "Shelton_1993_Pharyngeal_Fat_OSA"
R_TORRIANI = "Torriani_2014_C5_Neck_Adipose_Tissue"
R_AIRWAYNET = "AirwayNet_MMH_2024"
R_ZHANG = "Zhang_2022_Upper_Airway_CT_DL"


def _build() -> tuple[EvidenceSpec, ...]:
    T1 = EvidenceTier.TIER_1_CORE_OSA_BACKED
    T2 = EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC
    T3 = EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR
    T4 = EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY

    PRIMARY = AnalysisRole.PRIMARY_CANDIDATE
    SECONDARY = AnalysisRole.SECONDARY_CANDIDATE
    MECH = AnalysisRole.MECHANISTIC_SECONDARY
    CARDIO = AnalysisRole.CARDIOMETABOLIC_SECONDARY
    EXPL = AnalysisRole.EXPLORATORY
    NOVAL = AnalysisRole.DO_NOT_MODEL_WITHOUT_VALIDATION

    DIRECT = EvidenceClass.OSA_CT_DIRECT
    INDIRECT = EvidenceClass.OSA_IMAGING_INDIRECT
    CT_ANAT = EvidenceClass.CT_ANATOMY_DIRECT_NO_OSA
    CARDIOMETAB = EvidenceClass.CT_CARDIOMETABOLIC_DIRECT_NO_OSA
    NOVEL = EvidenceClass.CTA_STROKE_NOVEL
    PROXY = EvidenceClass.ENGINEERED_PROXY
    RAD = EvidenceClass.RADIOMICS_EXPLORATORY
    MODEL = EvidenceClass.MODEL_OUTPUT_EXPLORATORY

    E: list[EvidenceSpec] = []

    # ===================================================================
    # TIER 1 — CORE OSA-BACKED
    # ===================================================================

    # ----- Airway (Barkdull 2008; Zhang 2022; AirwayNet 2024) -----
    airway_refs = (R_BARKDULL, R_ZHANG, R_AIRWAYNET)
    for name, unit, region in (
        ("airway_min_csa_mm2", "mm2", "airway"),
        ("airway_volume_ml", "ml", "airway"),
        ("airway_length_mm", "mm", "airway"),
        ("airway_ap_diameter_at_min_csa_mm", "mm", "airway"),
        ("airway_lateral_diameter_at_min_csa_mm", "mm", "airway"),
        ("airway_ap_to_lateral_ratio_at_min_csa", "ratio", "airway"),
        ("airway_eccentricity_at_min_csa", "ratio", "airway"),
        ("retropalatal_min_csa_mm2", "mm2", "retropalatal"),
        ("retropalatal_volume_ml", "ml", "retropalatal"),
        ("retroglossal_min_csa_mm2", "mm2", "retroglossal"),
        ("retroglossal_volume_ml", "ml", "retroglossal"),
        ("retrolingual_min_csa_mm2", "mm2", "retrolingual"),
        ("retrolingual_volume_ml", "ml", "retrolingual"),
    ):
        E.append(_spec(
            name, "airway", region, tier=T1, evidence_class=DIRECT, role=PRIMARY,
            anatomic="anatomic", unit=unit, references=airway_refs,
            osa="yes", ct="yes", mri="yes",
            method="airway mask geometry; axial CSA / EDT",
            confidence_field="airway_confidence",
            required_masks=("airway_mask",),
            required_landmarks=() if name.startswith("airway_") else ("hyoid", "soft_palate_inferior"),
        ))
    E.append(_spec(
        "airway_min_csa_region", "airway", "airway", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit="str",
        references=airway_refs, osa="yes", ct="yes",
        confidence_field="airway_confidence", required_masks=("airway_mask",),
        comments="Labelled compartment of the minimum-CSA slice.",
    ))

    # ----- Tongue / mandible (Shigeta 2011; Barkdull 2008) -----
    for name, unit, region, refs, contrast in (
        ("tongue_volume_ml", "ml", "tongue", (R_SHIGETA,), False),
        ("tongue_posterior_mean_hu", "HU", "tongue_posterior", (R_BARKDULL,), True),
        ("tongue_posterior_low_hu_fraction", "fraction", "tongue_posterior", (R_BARKDULL,), True),
        ("tongue_base_volume_ml", "ml", "tongue_base", (R_BARKDULL,), False),
        ("mandible_volume_ml", "ml", "mandible", (R_SHIGETA,), False),
    ):
        E.append(_spec(
            name, "tongue" if "tongue" in name else "mandible", region, tier=T1,
            evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit=unit,
            references=refs, osa="yes", ct="yes",
            contrast=contrast, artifact=True,
            confidence_field="tongue_confidence" if "tongue" in name else "mandible_mask_method",
            required_masks=("tongue_mask",) if "tongue" in name else ("mandible_mask",),
            comments="Posterior-tongue attenuation is a low-HU fat surrogate; CTA contrast shifts it."
            if "posterior" in name else "",
        ))
    E.append(_spec(
        "tongue_to_mandible_volume_ratio", "tongue", "tongue", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic",
        references=(R_SHIGETA,), osa="yes", ct="yes",
        confidence_field="tongue_confidence",
        required_masks=("tongue_mask", "mandible_mask"),
        comments="Shigeta 2011 tongue/mandible volume ratio.",
    ))
    E.append(_spec(
        "tongue_base_to_retroglossal_airway_ratio", "tongue", "tongue_base",
        tier=T1, evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic",
        references=(R_BARKDULL,), osa="yes", ct="yes",
        confidence_field="tongue_confidence",
        required_masks=("tongue_mask", "airway_mask"),
        required_landmarks=("hyoid",),
    ))
    E.append(_spec(
        "tongue_to_skeletal_enclosure_ratio", "tongue", "tongue", tier=T1,
        evidence_class=INDIRECT, role=PRIMARY, anatomic="proxy",
        references=(R_BARKDULL, R_SHIGETA), osa="indirect", ct="yes",
        anatomic_link="yes", novelty="moderate",
        confidence_field="tongue_confidence",
        comments="Tongue crowding within the skeletal enclosure; proxy for soft-tissue/bony mismatch.",
    ))

    # ----- Cervical fat (Ernst 2023; Shelton 1993) -----
    E.append(_spec(
        "fat_cervical_total_volume_ml", "fat", "fat_cervical", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic",
        references=(R_ERNST, R_SHELTON), osa="yes", ct="yes",
        confidence_field="fat_cervical_confidence", required_masks=(),
        comments="Total cervical fat tissue volume (Ernst 2023).",
    ))
    E.append(_spec(
        "fat_cervical_mean_hu", "fat", "fat_cervical", tier=T1,
        evidence_class=INDIRECT, role=PRIMARY, anatomic="anatomic", unit="HU",
        references=(R_ERNST,), osa="indirect", ct="yes", contrast=True,
        confidence_field="fat_cervical_confidence",
    ))
    # Internal vs subcutaneous neck-fat proxy (planned canonical names + aliases).
    E.append(_spec(
        "fat_neck_subcutaneous_proxy_volume_ml", "fat", "fat_subcutaneous", tier=T1,
        evidence_class=INDIRECT, role=PRIMARY, anatomic="proxy",
        references=(R_ERNST,), osa="indirect", ct="yes", novelty="moderate",
        confidence_field="fat_cervical_confidence", implemented=False,
        aliases=("fat_subcutaneous_cervical_volume_ml",),
        comments="Subcutaneous neck fat by surface-distance proxy; alias of fat_subcutaneous_cervical_volume_ml.",
    ))
    E.append(_spec(
        "fat_neck_internal_proxy_volume_ml", "fat", "fat_deep", tier=T1,
        evidence_class=INDIRECT, role=PRIMARY, anatomic="proxy",
        references=(R_ERNST, R_SHELTON), osa="indirect", ct="yes", novelty="moderate",
        confidence_field="fat_cervical_confidence", implemented=False,
        aliases=("fat_deep_cervical_volume_ml",),
        comments="Internal/deep cervical fat by surface-distance proxy; alias of fat_deep_cervical_volume_ml.",
    ))
    E.append(_spec(
        "fat_internal_to_subcutaneous_ratio", "fat", "fat_deep", tier=T1,
        evidence_class=INDIRECT, role=PRIMARY, anatomic="proxy",
        references=(R_ERNST,), osa="indirect", ct="yes", novelty="moderate",
        confidence_field="fat_cervical_confidence", implemented=False,
        aliases=("fat_deep_to_subcutaneous_ratio",),
    ))
    # FOV-robust anchored neck slab — the fix for cervical-volume inflation on
    # tall CTAs. Fixed-height slab anchored on the airway min-CSA; dimensionless
    # fractions are the preferred, scan-length-invariant cervical adiposity marks.
    for name, unit, anatomic, note in (
        ("fat_neck_slab_volume_ml", "ml", "anatomic",
         "Fat in a fixed ~8cm slab anchored on the airway min-CSA (FOV-robust)."),
        ("fat_neck_slab_fat_fraction", "fraction", "anatomic",
         "Slab fat / slab neck volume — dimensionless, scan-length invariant."),
        ("fat_neck_area_fraction_at_min_csa", "fraction", "anatomic",
         "Fat area / neck area at the min-CSA slice — dimensionless."),
        ("fat_neck_slab_to_airway_volume_ratio", "ratio", "proxy",
         "Neck-slab fat normalised to airway volume."),
    ):
        E.append(_spec(
            name, "fat", "fat_cervical", tier=T1, evidence_class=INDIRECT,
            role=PRIMARY, anatomic=anatomic, unit=unit,
            references=(R_ERNST, R_TORRIANI), osa="indirect", ct="yes",
            novelty="moderate", contrast=False,
            confidence_field="fat_cervical_confidence",
            required_masks=("airway_mask",), comments=note,
        ))

    # ----- Parapharyngeal fat (Chen 2019; Shelton 1993) -----
    pp_refs = (R_CHEN, R_SHELTON)
    for name, unit in (
        ("fat_parapharyngeal_left_volume_ml", "ml"),
        ("fat_parapharyngeal_right_volume_ml", "ml"),
        ("fat_parapharyngeal_total_volume_ml", "ml"),
        ("fat_parapharyngeal_asymmetry_index", "ratio"),
        ("fat_parapharyngeal_area_retropalatal_total_mm2", "mm2"),
        ("fat_parapharyngeal_area_retroglossal_total_mm2", "mm2"),
        ("fat_parapharyngeal_area_subglosso_supraglottic_total_mm2", "mm2"),
        ("fat_parapharyngeal_to_airway_ratio_retropalatal", "ratio"),
        ("fat_parapharyngeal_to_airway_ratio_retroglossal", "ratio"),
        ("fat_parapharyngeal_to_airway_ratio_min_csa", "ratio"),
    ):
        E.append(_spec(
            name, "fat", "fat_parapharyngeal", tier=T1, evidence_class=DIRECT,
            role=PRIMARY, anatomic="anatomic", unit=unit, references=pp_refs,
            osa="yes", ct="yes", confidence_field="fat_parapharyngeal_confidence",
            required_masks=("airway_mask",), artifact=True,
            required_landmarks=("soft_palate_inferior", "hyoid")
            if ("retropalatal" in name or "retroglossal" in name
                or "subglosso" in name) else (),
            comments="Chen 2019 level-specific parapharyngeal fat-pad areas."
            if "area_" in name else "",
        ))

    # ----- Skeletal / hyoid (Barkdull 2008) -----
    E.append(_spec(
        "mandibular_plane_to_hyoid_distance_mm", "skeletal", "mandible", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit="mm",
        references=(R_BARKDULL,), osa="yes", ct="yes", cbct="yes",
        confidence_field="mandibular_plane_method",
        required_landmarks=("hyoid", "mandibular_plane_z"),
        comments="MP-H distance — classic cephalometric/CT OSA marker.",
    ))
    E.append(_spec(
        "hyoid_to_mandibular_plane_distance_mm", "skeletal", "mandible", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit="mm",
        references=(R_BARKDULL,), osa="yes", ct="yes", cbct="yes",
        confidence_field="mandibular_plane_method", implemented=False,
        aliases=("hyoid_to_mandible_distance_mm", "mandibular_plane_to_hyoid_distance_mm"),
        required_landmarks=("hyoid", "mandibular_plane_z"),
        comments="Alias of mandibular_plane_to_hyoid_distance_mm via hyoid anchor.",
    ))
    E.append(_spec(
        "cervicomandibular_ring_area_mm2", "skeletal", "skeletal", tier=T1,
        evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit="mm2",
        references=(R_BARKDULL,), osa="yes", ct="yes",
        confidence_field="cervicomandibular_ring_method",
        required_landmarks=("hyoid",),
        comments="Cervicomandibular ring area at hyoid level (Barkdull 2008).",
    ))
    E.append(_spec(
        "hyoid_to_posterior_pharyngeal_wall_distance_mm", "skeletal", "skeletal",
        tier=T1, evidence_class=DIRECT, role=PRIMARY, anatomic="anatomic", unit="mm",
        references=(R_BARKDULL,), osa="yes", ct="yes",
        confidence_field="airway_confidence",
        required_landmarks=("hyoid",),
    ))

    # ===================================================================
    # TIER 2 — OSA-PLAUSIBLE CT ANATOMY
    # ===================================================================

    # ----- Retropharyngeal fat -----
    for name, unit, implemented in (
        ("fat_retropharyngeal_volume_ml", "ml", True),
        ("fat_retropharyngeal_mean_hu", "HU", True),
        ("fat_retropharyngeal_max_thickness_mm", "mm", True),
        ("fat_retropharyngeal_mean_thickness_mm", "mm", True),
        ("fat_retropharyngeal_area_at_retropalatal_level_mm2", "mm2", True),
        ("fat_retropharyngeal_area_at_retroglossal_level_mm2", "mm2", True),
        ("fat_retropharyngeal_to_airway_ratio_retropalatal", "ratio", False),
        ("fat_retropharyngeal_to_airway_ratio_retroglossal", "ratio", False),
        ("fat_retropharyngeal_to_airway_ratio_min_csa", "ratio", False),
    ):
        E.append(_spec(
            name, "fat", "fat_retropharyngeal", tier=T2,
            evidence_class=CT_ANAT, role=MECH, anatomic="anatomic", unit=unit,
            references=(R_SHELTON,), osa="indirect", ct="yes", anatomic_link="yes",
            novelty="moderate", contrast=("hu" in name),
            confidence_field="fat_retropharyngeal_confidence",
            required_masks=("airway_mask",), optional_masks=("prevertebral_mask",),
            implemented=implemented,
        ))

    # ----- Surface / layer proxy fat -----
    for name in (
        "fat_surface_shell_0_5mm_volume_ml", "fat_surface_shell_5_10mm_volume_ml",
        "fat_surface_shell_10_20mm_volume_ml", "fat_surface_shell_20_30mm_volume_ml",
        "fat_internal_beyond_30mm_volume_ml", "fat_surface_to_internal_ratio",
        "fat_supraplatysmal_proxy_volume_ml", "fat_subplatysmal_proxy_volume_ml",
        "fat_supraplatysmal_to_subplatysmal_ratio",
    ):
        unit = "ratio" if name.endswith("_ratio") else "ml"
        is_platysma = "platysmal" in name
        E.append(_spec(
            name, "fat", "fat_surface_shell", tier=T2, evidence_class=PROXY,
            role=MECH, anatomic="proxy", unit=unit, references=(R_ERNST,),
            osa="indirect", ct="indirect", novelty="moderate",
            confidence_field="fat_cervical_confidence",
            optional_masks=("platysma_mask",) if is_platysma else (),
            implemented=False,
            comments="Supra-/subplatysmal split is a surface-distance PROXY; not a true platysma "
                     "partition unless a platysma mask is supplied." if is_platysma else
                     "Surface-distance shell decomposition of neck fat.",
        ))

    # ----- Submental / submandibular fat -----
    for name, unit in (
        ("fat_submental_total_volume_ml", "ml"),
        ("fat_interdigastric_submental_volume_ml", "ml"),
        ("fat_submandibular_space_left_volume_ml", "ml"),
        ("fat_submandibular_space_right_volume_ml", "ml"),
        ("fat_submandibular_space_total_volume_ml", "ml"),
        ("fat_submandibular_space_asymmetry_index", "ratio"),
    ):
        E.append(_spec(
            name, "fat", "fat_submandibular", tier=T2, evidence_class=CT_ANAT,
            role=MECH, anatomic="proxy", unit=unit, references=(R_SHELTON,),
            osa="indirect", ct="indirect", anatomic_link="yes", novelty="moderate",
            confidence_field="fat_submandibular_confidence",
            optional_masks=("submandibular_gland_mask",), implemented=False,
            comments="Submandibular-space fat; gland is only excluded when a gland mask is supplied.",
        ))
    E.append(_spec(
        "fat_submandibular_gland_excluded_flag", "fat", "fat_submandibular",
        tier=T2, evidence_class=CT_ANAT, role=MECH, anatomic="anatomic",
        unit="bool", references=(R_SHELTON,), confidence_field="fat_submandibular_confidence",
        implemented=False,
        comments="True only when a submandibular-gland mask was available to subtract.",
    ))

    # ----- Periairway distance-shell fat -----
    for name, unit, implemented in (
        ("fat_periairway_shell_0_5mm_volume_ml", "ml", True),
        ("fat_periairway_shell_5_10mm_volume_ml", "ml", True),
        ("fat_periairway_shell_10_20mm_volume_ml", "ml", True),
        ("fat_periairway_shell_20_30mm_volume_ml", "ml", True),
        ("fat_periairway_to_airway_volume_ratio", "ratio", False),
        ("fat_periairway_left_right_asymmetry", "ratio", False),
        ("fat_periairway_anterior_posterior_ratio", "ratio", False),
    ):
        E.append(_spec(
            name, "fat", "fat_periairway", tier=T2, evidence_class=CT_ANAT,
            role=MECH, anatomic="proxy", unit=unit, references=(R_SHELTON,),
            osa="indirect", ct="indirect", anatomic_link="yes", novelty="high",
            confidence_field="fat_periairway_confidence",
            required_masks=("airway_mask",), implemented=implemented,
            comments="Fat within a Euclidean-distance shell around the airway.",
        ))

    # ----- Soft-tissue extensions -----
    for name, unit in (
        ("soft_palate_length_mm", "mm"), ("soft_palate_thickness_max_mm", "mm"),
        ("soft_palate_thickness_mean_mm", "mm"), ("soft_palate_volume_ml", "ml"),
        ("soft_palate_to_retropalatal_airway_ratio", "ratio"),
        ("lateral_pharyngeal_wall_left_thickness_mm", "mm"),
        ("lateral_pharyngeal_wall_right_thickness_mm", "mm"),
        ("lateral_pharyngeal_wall_asymmetry_index", "ratio"),
        ("uvula_length_mm", "mm"), ("palatine_tonsil_total_volume_ml", "ml"),
        ("lingual_tonsil_volume_ml", "ml"),
    ):
        fam = ("soft_palate" if name.startswith(("soft_palate", "uvula"))
               else "lateral_wall" if name.startswith("lateral") else "tonsil")
        E.append(_spec(
            name, fam, fam, tier=T2, evidence_class=CT_ANAT, role=MECH,
            anatomic="anatomic", unit=unit, references=(R_SHELTON,),
            osa="indirect", ct="indirect", anatomic_link="yes", novelty="moderate",
            artifact=True,
            required_landmarks=("soft_palate_inferior",) if "soft_palate" in name else (),
        ))

    # ===================================================================
    # TIER 3 — CT CARDIOMETABOLIC / VASCULAR
    # ===================================================================
    for name, unit in (
        ("fat_c5_nat_total_area_mm2", "mm2"),
        ("fat_c5_nat_subcutaneous_area_mm2", "mm2"),
        ("fat_c5_nat_posterior_area_mm2", "mm2"),
        ("fat_c5_nat_perivertebral_area_mm2", "mm2"),
        ("fat_c5_nat_internal_area_mm2", "mm2"),
        ("fat_c5_nat_subcutaneous_fraction", "fraction"),
        ("fat_c5_nat_posterior_fraction", "fraction"),
        ("fat_c5_nat_perivertebral_fraction", "fraction"),
        ("fat_c5_nat_posterior_to_subcutaneous_ratio", "ratio"),
        ("fat_c5_nat_perivertebral_to_subcutaneous_ratio", "ratio"),
    ):
        E.append(_spec(
            name, "fat", "fat_c5_nat", tier=T3, evidence_class=CARDIOMETAB,
            role=CARDIO, anatomic="anatomic", unit=unit, references=(R_TORRIANI,),
            osa="no", ct="yes", cardiometabolic="yes", novelty="moderate",
            confidence_field="fat_c5_nat_confidence",
            required_landmarks=("c5_level",), implemented=False,
            comments="C5 compartmental neck adipose tissue (Torriani 2014) — metabolic/CVD risk, not OSA anatomy.",
        ))
    for name, unit in (
        ("fat_pericarotid_left_volume_ml", "ml"),
        ("fat_pericarotid_right_volume_ml", "ml"),
        ("fat_pericarotid_left_mean_hu", "HU"),
        ("fat_pericarotid_right_mean_hu", "HU"),
        ("fat_pericarotid_asymmetry_index", "ratio"),
    ):
        E.append(_spec(
            name, "fat", "fat_pericarotid", tier=T3, evidence_class=CARDIOMETAB,
            role=CARDIO, anatomic="anatomic", unit=unit, references=(R_TORRIANI,),
            osa="no", ct="yes", cardiometabolic="yes", novelty="moderate",
            contrast=("hu" in name), confidence_field="fat_pericarotid_confidence",
            required_masks=("carotid_mask",), implemented=False,
            comments="Perivascular (pericarotid) fat — requires a carotid mask; vascular-risk feature.",
        ))

    # ===================================================================
    # TIER 4 — NOVEL STROKE-CTA / EXPLORATORY
    # ===================================================================
    for name in (
        "fat_periairway_to_min_csa_ratio",
        "fat_parapharyngeal_to_tongue_base_ratio",
        "fat_retropharyngeal_to_retroglossal_airway_ratio",
        "fat_left_right_asymmetry_near_airway",
    ):
        E.append(_spec(
            name, "fat", "fat_engineered", tier=T4, evidence_class=NOVEL,
            role=EXPL, anatomic="proxy", unit="ratio", references=(R_AIRWAYNET, R_CHEN),
            osa="no", ct="no", novelty="high",
            confidence_field="fat_periairway_confidence", implemented=False,
            comments="Engineered CTA fat-to-airway ratio; hypothesis generation only.",
        ))
    for name in (
        "cta_osa_airway_narrowing_index_untrained",
        "cta_osa_tongue_crowding_index_untrained",
        "cta_osa_fat_burden_index_untrained",
        "cta_osa_skeletal_restriction_index_untrained",
        "cta_osa_combined_anatomy_index_untrained",
        "cta_osa_nocturnal_stroke_endotype_score_untrained",
    ):
        E.append(_spec(
            name, "composite", "case", tier=T4, evidence_class=MODEL,
            role=NOVAL, anatomic="proxy", unit="ratio", references=(R_AIRWAYNET,),
            osa="no", ct="no", novelty="high",
            implemented=(name in {m.feature_name for m in mr.all_metrics()}),
            comments="UNTRAINED composite — name suffix is intentional; not for clinical use.",
        ))

    return tuple(E)


def _finalize(specs: tuple[EvidenceSpec, ...]) -> tuple[EvidenceSpec, ...]:
    """Make ``implemented`` truthful: a spec is implemented iff its canonical
    name or any alias is a real metric-registry column. This keeps hand-written
    flags from drifting away from what the pipeline actually emits."""
    from dataclasses import replace
    registry = set(mr.feature_names())
    out: list[EvidenceSpec] = []
    for e in specs:
        impl = e.feature_name in registry or any(a in registry for a in e.aliases)
        out.append(e if e.implemented == impl else replace(e, implemented=impl))
    return tuple(out)


_EVIDENCE: tuple[EvidenceSpec, ...] = _finalize(_build())
_BY_NAME: dict[str, EvidenceSpec] = {e.feature_name: e for e in _EVIDENCE}
# Alias map: idealized/aspirational name → implemented registry name.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _e in _EVIDENCE:
    for _a in _e.aliases:
        _ALIAS_TO_CANONICAL.setdefault(_a, _e.feature_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def all_evidence() -> tuple[EvidenceSpec, ...]:
    """Every evidence-classified feature, in registration order."""
    return _EVIDENCE


def evidence_for(name: str) -> Optional[EvidenceSpec]:
    """Look up an evidence spec by feature name or by a known alias."""
    if name in _BY_NAME:
        return _BY_NAME[name]
    canon = _ALIAS_TO_CANONICAL.get(name)
    return _BY_NAME.get(canon) if canon else None


def by_tier(tier: EvidenceTier) -> list[EvidenceSpec]:
    return [e for e in _EVIDENCE if e.evidence_tier == tier]


def feature_names_for_tier(tier: EvidenceTier, *, implemented_only: bool = False
                           ) -> list[str]:
    return [e.feature_name for e in _EVIDENCE
            if e.evidence_tier == tier and (e.implemented or not implemented_only)]


def implemented_names() -> set[str]:
    """Evidence features that the pipeline currently emits as columns.

    An evidence spec is treated as implemented if its own name OR any of its
    aliases is present in the metric registry.
    """
    registry = set(mr.feature_names())
    out: set[str] = set()
    for e in _EVIDENCE:
        if e.feature_name in registry or any(a in registry for a in e.aliases):
            out.add(e.feature_name)
    return out


def resolve_to_columns(name: str, available: Iterable[str]) -> Optional[str]:
    """Return the column name to read for an evidence feature.

    Prefers the canonical name, then any alias that is actually present in
    ``available``. Returns None if neither is available.
    """
    avail = set(available)
    if name in avail:
        return name
    e = _BY_NAME.get(name)
    if e is not None:
        for a in e.aliases:
            if a in avail:
                return a
    return None


def to_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in _EVIDENCE:
        rows.append({
            "feature_name": e.feature_name,
            "feature_family": e.feature_family,
            "anatomical_region": e.anatomical_region,
            "unit": e.unit,
            "evidence_tier": e.evidence_tier.value,
            "evidence_class": e.evidence_class.value,
            "prior_osa_link": e.prior_osa_link,
            "prior_ct_link": e.prior_ct_link,
            "prior_cbct_link": e.prior_cbct_link,
            "prior_mri_link": e.prior_mri_link,
            "prior_anatomic_link": e.prior_anatomic_link,
            "prior_cardiometabolic_link": e.prior_cardiometabolic_link,
            "stroke_cta_novelty": e.stroke_cta_novelty,
            "recommended_feature_set": e.recommended_feature_set,
            "analysis_role": e.analysis_role.value,
            "true_anatomic_vs_proxy": e.true_anatomic_vs_proxy,
            "required_masks": ";".join(e.required_masks),
            "optional_masks": ";".join(e.optional_masks),
            "required_landmarks": ";".join(e.required_landmarks),
            "extraction_method": e.extraction_method,
            "confidence_field_name": e.confidence_field_name,
            "contrast_sensitive": e.contrast_sensitive,
            "artifact_sensitive": e.artifact_sensitive,
            "missingness_behavior": e.missingness_behavior,
            "reference_tags": ";".join(e.reference_tags),
            "implemented": e.implemented,
            "aliases": ";".join(e.aliases),
            "comments": e.comments,
        })
    return rows


def to_json(path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_records(), indent=2))
    return Path(path)


def to_csv(path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = to_records()
    fieldnames = list(records[0].keys()) if records else []
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return Path(path)


def merged_records() -> list[dict[str, Any]]:
    """One row per metric-registry feature, annotated with evidence metadata.

    Columns the metric registry already owns (unit, family, maturity, ...) are
    joined with the evidence axis (tier, class, analysis role, references). For
    features without an evidence spec — identifiers, QC, method/confidence
    helper columns — the evidence fields are left blank. This is what
    ``list-features`` exports so a single CSV documents both axes.
    """
    rows: list[dict[str, Any]] = []
    for m in mr.all_metrics():
        e = evidence_for(m.feature_name)
        rows.append({
            "feature_name": m.feature_name,
            "family": m.family,
            "region": m.region,
            "unit": m.unit,
            "analysis_tier": m.tier.value,
            "maturity": m.maturity.value,
            "evidence_tier": e.evidence_tier.value if e else "",
            "evidence_class": e.evidence_class.value if e else "",
            "analysis_role": e.analysis_role.value if e else "",
            "recommended_feature_set": e.recommended_feature_set if e else "",
            "true_anatomic_vs_proxy": e.true_anatomic_vs_proxy if e else "",
            "prior_osa_link": e.prior_osa_link if e else "",
            "confidence_field_name": e.confidence_field_name if e else "",
            "reference_tags": ";".join(e.reference_tags) if e else "",
        })
    return rows


def evidence_summary_records() -> list[dict[str, Any]]:
    """Condensed rows for ``feature_evidence_summary.csv``."""
    rows: list[dict[str, Any]] = []
    for e in _EVIDENCE:
        rows.append({
            "feature_name": e.feature_name,
            "evidence_tier": e.evidence_tier.value,
            "evidence_class": e.evidence_class.value,
            "feature_set": e.recommended_feature_set,
            "analysis_role": e.analysis_role.value,
            "prior_osa_link": e.prior_osa_link,
            "prior_ct_link": e.prior_ct_link,
            "true_anatomic_vs_proxy": e.true_anatomic_vs_proxy,
            "contrast_sensitive": e.contrast_sensitive,
            "confidence_field_name": e.confidence_field_name,
            "reference_tags": ";".join(e.reference_tags),
        })
    return rows
