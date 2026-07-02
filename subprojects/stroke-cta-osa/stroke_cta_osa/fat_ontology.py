"""Evidence-tiered ontology for fat *compartments*.

The fat modules (:mod:`fat`, :mod:`fat_regions`) emit many compartments. This
module groups those compartments by :class:`evidence_registry.EvidenceTier`,
records whether each is a *true anatomic* region or a geometric *proxy*, and
lists the masks/landmarks required to upgrade a proxy to an anatomic label.

It is the fat-specific companion to :mod:`evidence_registry` and exists so the
fat code, the docs (``FAT_COMPARTMENTS.md``), and the QC/confidence logic share
one source of truth about what a compartment *is* and how strongly it is
backed by prior OSA imaging literature.

Naming rule enforced here:

* A compartment is ``anatomic`` only if real segmentation masks (or validated
  landmarks) define it; otherwise it is ``proxy`` and its feature names must
  carry ``_proxy`` (or the spec's ``true_anatomic_vs_proxy='proxy'``).
* Surface-shell fat must NOT be labelled supraplatysmal/subplatysmal unless a
  platysma mask is available.
* Submandibular fat must NOT be labelled gland-excluded unless a gland mask is
  available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evidence_registry import EvidenceTier


@dataclass(frozen=True)
class FatCompartment:
    """One fat compartment in the ontology."""
    key: str                       # internal compartment key
    label: str                     # human-readable label
    evidence_tier: EvidenceTier
    true_anatomic_vs_proxy: str    # "anatomic" | "proxy"
    confidence_field: str          # which *_confidence column governs it
    required_masks: tuple[str, ...] = ()
    optional_masks: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    contrast_sensitive: bool = False
    artifact_sensitive: bool = True
    notes: str = ""


T1 = EvidenceTier.TIER_1_CORE_OSA_BACKED
T2 = EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC
T3 = EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR
T4 = EvidenceTier.TIER_4_STROKE_CTA_NOVEL_EXPLORATORY


_FAT_COMPARTMENTS: tuple[FatCompartment, ...] = (
    # ---------------- Tier 1: core OSA-backed ----------------
    FatCompartment(
        "cervical_total", "Total cervical fat", T1, "anatomic",
        "fat_cervical_confidence",
        feature_names=("fat_cervical_total_volume_ml", "fat_cervical_mean_hu"),
        contrast_sensitive=True,
        notes="Ernst 2023 cervical fat tissue volume; moderate-to-severe OSA.",
    ),
    FatCompartment(
        "cervical_subcutaneous_proxy", "Subcutaneous neck fat (proxy)", T1, "proxy",
        "fat_cervical_confidence",
        feature_names=("fat_neck_subcutaneous_proxy_volume_ml",
                       "fat_subcutaneous_cervical_volume_ml"),
        notes="Surface-distance subcutaneous split; proxy unless skin/platysma masks exist.",
    ),
    FatCompartment(
        "cervical_internal_proxy", "Internal/deep neck fat (proxy)", T1, "proxy",
        "fat_cervical_confidence",
        feature_names=("fat_neck_internal_proxy_volume_ml",
                       "fat_deep_cervical_volume_ml",
                       "fat_internal_to_subcutaneous_ratio"),
    ),
    FatCompartment(
        "pharyngeal_airway_adjacent_fat", "Peripharyngeal airway-adjacent fat", T1,
        "anatomic", "fat_parapharyngeal_confidence",
        required_masks=("airway_mask",),
        feature_names=("fat_deep_peripharyngeal_volume_ml",),
        notes="Shelton 1993 pharyngeal adipose tissue.",
    ),
    FatCompartment(
        "parapharyngeal_fat_pad_left", "Left parapharyngeal fat pad", T1, "anatomic",
        "fat_parapharyngeal_confidence", required_masks=("airway_mask",),
        feature_names=("fat_parapharyngeal_left_volume_ml",),
    ),
    FatCompartment(
        "parapharyngeal_fat_pad_right", "Right parapharyngeal fat pad", T1, "anatomic",
        "fat_parapharyngeal_confidence", required_masks=("airway_mask",),
        feature_names=("fat_parapharyngeal_right_volume_ml",),
    ),
    FatCompartment(
        "parapharyngeal_fat_pad_retropalatal", "Parapharyngeal fat @ retropalatal", T1,
        "anatomic", "fat_parapharyngeal_confidence", required_masks=("airway_mask",),
        feature_names=("fat_parapharyngeal_area_retropalatal_total_mm2",),
        notes="Chen 2019 level-specific parapharyngeal fat-pad area.",
    ),
    FatCompartment(
        "parapharyngeal_fat_pad_retroglossal", "Parapharyngeal fat @ retroglossal", T1,
        "anatomic", "fat_parapharyngeal_confidence", required_masks=("airway_mask",),
        feature_names=("fat_parapharyngeal_area_retroglossal_total_mm2",),
    ),
    FatCompartment(
        "parapharyngeal_fat_pad_subglosso_supraglottic",
        "Parapharyngeal fat @ subglosso-supraglottic", T1, "anatomic",
        "fat_parapharyngeal_confidence", required_masks=("airway_mask",),
        feature_names=("fat_parapharyngeal_area_subglosso_supraglottic_total_mm2",),
    ),

    # ---------------- Tier 2: OSA-plausible CT anatomy ----------------
    FatCompartment(
        "retropharyngeal_fat", "Retropharyngeal fat", T2, "anatomic",
        "fat_retropharyngeal_confidence", required_masks=("airway_mask",),
        optional_masks=("prevertebral_mask",),
        feature_names=("fat_retropharyngeal_volume_ml", "fat_retropharyngeal_mean_hu",
                       "fat_retropharyngeal_max_thickness_mm",
                       "fat_retropharyngeal_mean_thickness_mm"),
        contrast_sensitive=True,
    ),
    FatCompartment(
        "submandibular_space_fat", "Submandibular-space fat", T2, "proxy",
        "fat_submandibular_confidence", optional_masks=("submandibular_gland_mask",),
        feature_names=("fat_submandibular_space_left_volume_ml",
                       "fat_submandibular_space_right_volume_ml",
                       "fat_submandibular_space_total_volume_ml"),
        notes="Gland excluded only if a gland mask is supplied.",
    ),
    FatCompartment(
        "submental_interdigastric_fat", "Submental / interdigastric fat", T2, "proxy",
        "fat_submandibular_confidence",
        feature_names=("fat_submental_total_volume_ml",
                       "fat_interdigastric_submental_volume_ml"),
    ),
    FatCompartment(
        "sublingual_space_fat", "Sublingual-space fat", T2, "proxy",
        "fat_submandibular_confidence",
        feature_names=(),
    ),
    FatCompartment(
        "surface_shell_fat", "Surface-distance shell fat", T2, "proxy",
        "fat_cervical_confidence",
        feature_names=("fat_surface_shell_0_5mm_volume_ml",
                       "fat_surface_shell_5_10mm_volume_ml",
                       "fat_surface_shell_10_20mm_volume_ml",
                       "fat_surface_shell_20_30mm_volume_ml",
                       "fat_internal_beyond_30mm_volume_ml"),
    ),
    FatCompartment(
        "supraplatysmal_proxy_fat", "Supraplatysmal fat (proxy)", T2, "proxy",
        "fat_cervical_confidence", optional_masks=("platysma_mask",),
        feature_names=("fat_supraplatysmal_proxy_volume_ml",),
        notes="PROXY — not a true platysma partition unless a platysma mask exists.",
    ),
    FatCompartment(
        "subplatysmal_proxy_fat", "Subplatysmal fat (proxy)", T2, "proxy",
        "fat_cervical_confidence", optional_masks=("platysma_mask",),
        feature_names=("fat_subplatysmal_proxy_volume_ml",),
    ),
    FatCompartment(
        "periairway_distance_shell_fat", "Periairway distance-shell fat", T2, "proxy",
        "fat_periairway_confidence", required_masks=("airway_mask",),
        feature_names=("fat_periairway_shell_0_5mm_volume_ml",
                       "fat_periairway_shell_5_10mm_volume_ml",
                       "fat_periairway_shell_10_20mm_volume_ml",
                       "fat_periairway_shell_20_30mm_volume_ml"),
    ),
    FatCompartment(
        "masticator_space_fat", "Masticator-space fat", T2, "proxy",
        "fat_cervical_confidence", feature_names=(),
    ),
    FatCompartment(
        "buccal_space_fat", "Buccal-space fat", T2, "proxy",
        "fat_cervical_confidence",
        feature_names=("fat_buccal_left_volume_ml", "fat_buccal_right_volume_ml"),
    ),

    # ---------------- Tier 3: cardiometabolic / vascular ----------------
    FatCompartment(
        "c5_nat_subcutaneous", "C5 subcutaneous NAT", T3, "anatomic",
        "fat_c5_nat_confidence",
        feature_names=("fat_c5_nat_subcutaneous_area_mm2",
                       "fat_c5_nat_subcutaneous_fraction"),
        notes="Torriani 2014 C5 neck adipose tissue.",
    ),
    FatCompartment(
        "c5_nat_posterior", "C5 posterior intermuscular NAT", T3, "anatomic",
        "fat_c5_nat_confidence",
        feature_names=("fat_c5_nat_posterior_area_mm2", "fat_c5_nat_posterior_fraction"),
    ),
    FatCompartment(
        "c5_nat_perivertebral", "C5 perivertebral NAT", T3, "anatomic",
        "fat_c5_nat_confidence",
        feature_names=("fat_c5_nat_perivertebral_area_mm2",
                       "fat_c5_nat_perivertebral_fraction"),
    ),
    FatCompartment(
        "c5_nat_internal", "C5 internal NAT", T3, "anatomic",
        "fat_c5_nat_confidence", feature_names=("fat_c5_nat_internal_area_mm2",),
    ),
    FatCompartment(
        "pericarotid_fat", "Pericarotid fat", T3, "anatomic",
        "fat_pericarotid_confidence", required_masks=("carotid_mask",),
        feature_names=("fat_pericarotid_left_volume_ml",
                       "fat_pericarotid_right_volume_ml",
                       "fat_pericarotid_left_mean_hu", "fat_pericarotid_right_mean_hu"),
        contrast_sensitive=True,
    ),
    FatCompartment(
        "thoracic_mediastinal_fat", "Thoracic mediastinal fat", T3, "anatomic",
        "fat_cervical_confidence", required_masks=("mediastinal_mask",),
        feature_names=("fat_mediastinal_volume_ml",),
    ),
    FatCompartment(
        "epicardial_fat", "Epicardial fat", T3, "anatomic",
        "fat_cervical_confidence", required_masks=("epicardial_mask",),
        feature_names=("fat_epicardial_volume_ml", "fat_epicardial_mean_hu"),
    ),
    FatCompartment(
        "pericoronary_fat", "Pericoronary fat", T3, "anatomic",
        "fat_cervical_confidence", required_masks=("pericoronary_mask",),
        feature_names=("fat_pericoronary_mean_hu",),
    ),

    # ---------------- Tier 4: novel stroke-CTA exploratory ----------------
    FatCompartment(
        "engineered_fat_ratios", "Engineered fat ratios", T4, "proxy",
        "fat_periairway_confidence",
        feature_names=("fat_periairway_to_min_csa_ratio",
                       "fat_parapharyngeal_to_tongue_base_ratio",
                       "fat_retropharyngeal_to_retroglossal_airway_ratio",
                       "fat_left_right_asymmetry_near_airway"),
        notes="Hypothesis-generation only.",
    ),
    FatCompartment(
        "untrained_fat_composites", "Untrained fat composites", T4, "proxy",
        "fat_periairway_confidence",
        feature_names=("cta_osa_fat_burden_index_untrained",
                       "cta_osa_nocturnal_stroke_endotype_score_untrained"),
    ),
)

_BY_KEY: dict[str, FatCompartment] = {c.key: c for c in _FAT_COMPARTMENTS}


def all_compartments() -> tuple[FatCompartment, ...]:
    return _FAT_COMPARTMENTS


def compartment(key: str) -> Optional[FatCompartment]:
    return _BY_KEY.get(key)


def compartments_for_tier(tier: EvidenceTier) -> list[FatCompartment]:
    return [c for c in _FAT_COMPARTMENTS if c.evidence_tier == tier]


def tier_for_compartment(key: str) -> Optional[EvidenceTier]:
    c = _BY_KEY.get(key)
    return c.evidence_tier if c else None


def proxy_compartments() -> list[FatCompartment]:
    return [c for c in _FAT_COMPARTMENTS if c.true_anatomic_vs_proxy == "proxy"]


def confidence_for_masks(
    required_masks: tuple[str, ...],
    available_masks: set[str],
    *,
    is_proxy: bool,
) -> str:
    """Map mask availability to the confidence scale.

    * ``high``    — all required masks present and the compartment is anatomic.
    * ``moderate``— partial masks or a robust landmark-based method.
    * ``low``     — geometric proxy only.
    * ``missing`` — a required mask is absent.
    """
    if required_masks and not set(required_masks).issubset(available_masks):
        return "missing"
    if is_proxy:
        return "low"
    if required_masks and set(required_masks).issubset(available_masks):
        return "high"
    return "moderate"
