"""Canonical metric registry for stroke_cta_osa.

This file is the **single source of truth** for every per-case feature column
the pipeline can emit. Each entry carries enough metadata to:

  * generate a stable feature dictionary CSV (``list-features`` CLI),
  * decide at runtime whether a feature should appear in features.csv even
    when its value is missing (it always should — the column persists),
  * tell downstream analysis which features are landmark-dependent /
    contrast-sensitive / mask-required,
  * drive the documentation page in ``docs/stroke_cta_osa/FEATURES.md``.

Adding a new feature: append a ``MetricSpec`` instance to ``_REGISTRY``.
No other module should hard-code a feature name that isn't in this file.

The registry is *interpreted Python* on purpose — keeping it in YAML adds an
extra parse step at startup and makes IDE refactors lose track of names.
The CLI ``list-features`` command exports the registry to CSV/JSON for the
analysis pipeline and the docs build.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    """Recommended analysis tier.

    * ``TIER1`` — core, interpretable, should be extracted whenever coverage
      allows. These are the columns suitable for primary cohort modelling.
    * ``TIER2`` — useful but mask- or landmark-dependent enough that a fair
      fraction of real cohorts will have missingness.
    * ``EXPLORATORY`` — radiomics, composite indices, hooks for DL models.
      Treat as research-only; downstream analyses should filter for
      cohort-level completeness before using.
    """
    TIER1 = "tier1"
    TIER2 = "tier2"
    EXPLORATORY = "exploratory"


class Maturity(str, Enum):
    """How well-validated the underlying extraction method is in this pipeline.

    * ``STABLE`` — implemented from a literal voxel-counting / EDT / mask
      operation; behaviour is deterministic; tests assert physical units.
    * ``HEURISTIC`` — uses a landmark / morphology heuristic that works in
      common anatomy but can fail; surface a per-case method string.
    * ``EXPERIMENTAL`` — placeholder or fallback; values should be treated
      as exploratory only.
    """
    STABLE = "stable"
    HEURISTIC = "heuristic"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class MetricSpec:
    """One feature row in the registry.

    Attributes:
        feature_name: snake_case column name; immutable once published.
        family: high-level grouping for tables and the doc page.
            One of: identifiers / qc / airway / tongue / mandible / oral_cavity
            / soft_palate / lateral_wall / tonsil / skeletal / fat /
            composite / radiomics / optional.
        region: anatomical region the value summarises (e.g. ``retroglossal``).
        unit: explicit physical unit (``mm``, ``mm2``, ``mm3``, ``ml``, ``HU``,
            ``fraction``, ``ratio``, ``count``, ``bool``, ``str``).
        tier: analysis tier — see :class:`Tier`.
        maturity: extraction-method maturity — see :class:`Maturity`.
        required_inputs: list of inputs that MUST be available for the feature
            to be non-missing (e.g. ``["airway_mask"]``).
        optional_inputs: inputs that improve the feature but aren't required.
        landmark_dependent: True if a missing landmark would emit NaN.
        contrast_sensitive: True if HU-based values shift with CTA contrast.
        cta_specific: True for features the dental/CBCT pipeline cannot
            compute (e.g. cervical fat below the mandible plane).
        shared_with_dental: True if both pipelines may emit this column;
            governs the ``compare-dental`` join surface.
        extraction_method: one-line description of how the value is computed.
        missingness_behaviour: how a missing value is represented.
            One of: ``"NaN"``, ``"bool_False"``, ``"empty_str"``,
            ``"-1_int"``.
        notes: free-form research notes; safe to render into docs.
    """
    feature_name: str
    family: str
    region: str
    unit: str
    tier: Tier
    maturity: Maturity
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    landmark_dependent: bool = False
    contrast_sensitive: bool = False
    cta_specific: bool = False
    shared_with_dental: bool = False
    extraction_method: str = ""
    missingness_behaviour: Literal["NaN", "bool_False", "empty_str", "-1_int"] = "NaN"
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper factories — used to keep registry entries compact and readable
# ---------------------------------------------------------------------------

def _ml(name: str, region: str, *, tier: Tier = Tier.TIER1,
        maturity: Maturity = Maturity.STABLE,
        required: Iterable[str] = (), landmark_dependent: bool = False,
        shared_with_dental: bool = False, cta_specific: bool = False,
        contrast_sensitive: bool = False, method: str = "",
        notes: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="ml", tier=tier, maturity=maturity,
        required_inputs=tuple(required), landmark_dependent=landmark_dependent,
        shared_with_dental=shared_with_dental,
        cta_specific=cta_specific, contrast_sensitive=contrast_sensitive,
        extraction_method=method, notes=notes,
    )


def _mm(name: str, region: str, *, tier: Tier = Tier.TIER1,
        maturity: Maturity = Maturity.STABLE,
        required: Iterable[str] = (), landmark_dependent: bool = False,
        shared_with_dental: bool = False, cta_specific: bool = False,
        method: str = "", notes: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="mm", tier=tier, maturity=maturity,
        required_inputs=tuple(required), landmark_dependent=landmark_dependent,
        shared_with_dental=shared_with_dental, cta_specific=cta_specific,
        extraction_method=method, notes=notes,
    )


def _mm2(name: str, region: str, *, tier: Tier = Tier.TIER1,
         maturity: Maturity = Maturity.STABLE,
         required: Iterable[str] = (), landmark_dependent: bool = False,
         shared_with_dental: bool = False, cta_specific: bool = False,
         method: str = "", notes: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="mm2", tier=tier, maturity=maturity,
        required_inputs=tuple(required), landmark_dependent=landmark_dependent,
        shared_with_dental=shared_with_dental, cta_specific=cta_specific,
        extraction_method=method, notes=notes,
    )


def _hu(name: str, region: str, *, tier: Tier = Tier.TIER1,
        maturity: Maturity = Maturity.STABLE,
        required: Iterable[str] = (), contrast_sensitive: bool = True,
        landmark_dependent: bool = False, cta_specific: bool = False,
        method: str = "", notes: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="HU", tier=tier, maturity=maturity,
        required_inputs=tuple(required),
        landmark_dependent=landmark_dependent,
        contrast_sensitive=contrast_sensitive,
        cta_specific=cta_specific,
        extraction_method=method, notes=notes,
    )


def _frac(name: str, region: str, *, tier: Tier = Tier.TIER2,
          maturity: Maturity = Maturity.STABLE,
          required: Iterable[str] = (), landmark_dependent: bool = False,
          method: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="fraction", tier=tier, maturity=maturity,
        required_inputs=tuple(required), landmark_dependent=landmark_dependent,
        extraction_method=method,
    )


def _ratio(name: str, region: str, *, tier: Tier = Tier.TIER1,
           maturity: Maturity = Maturity.STABLE,
           required: Iterable[str] = (), landmark_dependent: bool = False,
           shared_with_dental: bool = False, method: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="ratio", tier=tier, maturity=maturity,
        required_inputs=tuple(required), landmark_dependent=landmark_dependent,
        shared_with_dental=shared_with_dental,
        extraction_method=method,
    )


def _bool(name: str, region: str, *, tier: Tier = Tier.TIER1,
          maturity: Maturity = Maturity.STABLE,
          method: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="bool", tier=tier, maturity=maturity,
        missingness_behaviour="bool_False",
        extraction_method=method,
    )


def _str(name: str, region: str, *, tier: Tier = Tier.TIER1,
         method: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="str", tier=tier, maturity=Maturity.STABLE,
        missingness_behaviour="empty_str",
        extraction_method=method,
    )


def _count(name: str, region: str, *, tier: Tier = Tier.TIER2,
           maturity: Maturity = Maturity.STABLE,
           method: str = "") -> MetricSpec:
    return MetricSpec(
        feature_name=name, family=_family_from_name(name),
        region=region, unit="count", tier=tier, maturity=maturity,
        extraction_method=method,
    )


def _family_from_name(name: str) -> str:
    """Best-effort family inference from a feature name prefix.

    Used so we don't repeat the family in every helper call. Falls back to
    ``other`` for un-recognised prefixes; explicit registry edits can
    override by setting `family` directly on the MetricSpec.
    """
    for prefix, family in (
        ("qc_", "qc"),
        ("airway_", "airway"),
        ("nasopharyngeal_", "airway"),
        ("retropalatal_", "airway"),
        ("retroglossal_", "airway"),
        ("retrolingual_", "airway"),
        ("hypopharyngeal_", "airway"),
        ("tongue_", "tongue"),
        ("lingual_tonsil_", "tongue"),
        ("mandible_", "mandible"),
        ("mandibular_", "mandible"),
        ("oral_cavity_", "oral_cavity"),
        ("soft_palate_", "soft_palate"),
        ("uvula_", "soft_palate"),
        ("palatine_tonsil_", "tonsil"),
        ("lateral_pharyngeal_", "lateral_wall"),
        ("lateral_wall_", "lateral_wall"),
        ("hyoid_", "skeletal"),
        ("neck_", "skeletal"),
        ("laryngeal_", "skeletal"),
        ("hard_palate_", "skeletal"),
        ("posterior_nasal_spine_", "skeletal"),
        ("cervicomandibular_", "skeletal"),
        ("skeletal_", "skeletal"),
        ("fat_", "fat"),
        ("cta_osa_", "composite"),
        ("rad_", "radiomics"),
        ("pericarotid_", "optional"),
        ("carotid_", "optional"),
        ("thoracic_", "optional"),
        ("epicardial_", "optional"),
        ("mediastinal_", "optional"),
        ("pericardial_", "optional"),
        ("tongue_to_", "tongue"),
        ("composite_", "composite"),
        ("patient_", "identifiers"),
        ("study_", "identifiers"),
        ("scan_", "identifiers"),
        ("pipeline", "identifiers"),
        ("config_", "identifiers"),
        ("input_", "identifiers"),
        ("processing_", "identifiers"),
        ("slicer_", "identifiers"),
    ):
        if name.startswith(prefix):
            return family
    return "other"


# ---------------------------------------------------------------------------
# Registry — append-only; keep groups visually together
# ---------------------------------------------------------------------------

def _build_registry() -> tuple[MetricSpec, ...]:
    R: list[MetricSpec] = []

    # ----- Identifiers -----
    for n in (
        "pipeline", "pipeline_version", "config_hash",
        "processing_timestamp", "patient_id", "study_id", "scan_id",
        "input_path_hash", "input_kind", "airway_source",
        "airway_provider_notes", "slicer_loader_script",
    ):
        R.append(_str(n, region="case", tier=Tier.TIER1))

    # ----- QC -----
    R.append(_bool("qc_pass", "case"))
    R.append(_count("qc_warning_count", "case", tier=Tier.TIER1))
    R.append(_str("qc_failure_reasons", "case"))
    R.append(MetricSpec(
        feature_name="qc_coverage_score", family="qc", region="case",
        unit="fraction", tier=Tier.TIER1, maturity=Maturity.HEURISTIC,
        extraction_method="weighted sum of presence flags",
    ))
    for flag in (
        "qc_has_upper_airway", "qc_has_cervical_soft_tissue",
        "qc_has_hard_palate_region", "qc_has_retropalatal_region",
        "qc_has_retroglossal_region", "qc_has_tongue_region",
        "qc_has_tongue_base_region", "qc_has_soft_palate_region",
        "qc_has_hyoid_region", "qc_has_epiglottis_region",
        "qc_has_mandible_region", "qc_has_parapharyngeal_region",
        "qc_has_retropharyngeal_region",
        "qc_dental_artifact_flag", "qc_motion_artifact_flag",
        "qc_swallow_artifact_flag", "qc_truncation_flag",
        "qc_low_fov_flag", "qc_contrast_phase_flag",
        "qc_streak_artifact_near_tongue_flag",
        "qc_airway_mask_available", "qc_tongue_mask_available",
        "qc_mandible_mask_available", "qc_hyoid_landmark_available",
        "qc_soft_palate_mask_available", "qc_fat_mask_available",
    ):
        R.append(_bool(flag, "case"))
    R.append(MetricSpec(
        feature_name="qc_dental_artifact_score", family="qc", region="case",
        unit="fraction", tier=Tier.TIER1, maturity=Maturity.HEURISTIC,
        extraction_method="fraction of voxels above bone-HU threshold",
    ))
    R.append(_mm("qc_spacing_x_mm", "case"))
    R.append(_mm("qc_spacing_y_mm", "case"))
    R.append(_mm("qc_spacing_z_mm", "case"))
    R.append(_mm("qc_z_extent_mm", "case"))
    R.append(_bool("qc_contrast_enhanced", "case",
                   method="inferred from DICOM tags / image-type"))

    # Feature-level reliability flags
    for flag in (
        "airway_features_reliable", "tongue_features_reliable",
        "posterior_tongue_hu_reliable", "soft_palate_features_reliable",
        "skeletal_features_reliable", "parapharyngeal_fat_features_reliable",
        "cervical_fat_features_reliable",
    ):
        R.append(_bool(flag, "case",
                       method="set by per-module QC; default True if module ran cleanly"))

    # ----- Airway, global -----
    R.append(_bool("airway_mask_available", "airway"))
    R.append(_str("airway_method", "airway"))
    R.append(_str("airway_confidence", "airway"))
    R.append(_str("airway_measurement_plane", "airway",
                  method="axial_approximation | centerline_orthogonal"))
    R.append(_bool("airway_centerline_available", "airway"))

    for n, region in (
        ("airway_volume_mm3", "airway"),
        ("airway_volume_ml", "airway"),
    ):
        m = _ml(n, region, shared_with_dental=True,
                method="voxel count * voxel volume", required=("airway_mask",))
        R.append(m)
    R.append(_mm("airway_length_mm", "airway", shared_with_dental=True,
                 method="non-zero z-slice count × dz (vertical extent v1)",
                 required=("airway_mask",)))
    R.append(_mm2("airway_min_csa_mm2", "airway", shared_with_dental=True,
                  method="min(non-zero per-slice voxel count × dx*dy)",
                  required=("airway_mask",)))
    R.append(MetricSpec(
        feature_name="airway_min_csa_slice_index", family="airway",
        region="airway", unit="count", tier=Tier.TIER1, maturity=Maturity.STABLE,
        missingness_behaviour="-1_int",
        required_inputs=("airway_mask",),
    ))
    R.append(_mm("airway_min_csa_z_mm", "airway", shared_with_dental=True,
                 method="physical z of min CSA slice",
                 required=("airway_mask",)))
    R.append(_str("airway_min_csa_region", "airway",
                  method="labelled compartment containing the min CSA slice"))
    for pct in (1, 5, 10, 25, 50):
        R.append(_mm2(
            f"airway_csa_p{pct:02d}_mm2" if pct < 50 else "airway_csa_median_mm2",
            "airway",
            method=f"percentile{pct} of per-slice CSA",
            required=("airway_mask",),
        ))
    R.append(_str("airway_csa_profile_json_path", "airway",
                  tier=Tier.TIER2,
                  method="path to per-slice CSA JSON if save_csa_profile=true"))
    R.append(_count("airway_area_profile_n_slices", "airway"))

    # Diameters / shape at min CSA
    R.append(_mm("airway_ap_diameter_at_min_csa_mm", "airway",
                 shared_with_dental=True,
                 method="axis-aligned A-P bbox extent at min-CSA slice"))
    R.append(_mm("airway_lateral_diameter_at_min_csa_mm", "airway",
                 shared_with_dental=True,
                 method="axis-aligned L-R bbox extent at min-CSA slice"))
    R.append(_ratio("airway_ap_to_lateral_ratio_at_min_csa", "airway",
                    method="AP/LAT"))
    R.append(_ratio("airway_eccentricity_at_min_csa", "airway",
                    shared_with_dental=True,
                    method="sqrt(1 - (min(L,AP)/max(L,AP))²)"))
    R.append(_ratio("airway_circularity_at_min_csa", "airway",
                    tier=Tier.TIER2,
                    method="4π·area / perimeter² of largest slice CC"))
    R.append(_ratio("airway_lateral_narrowing_index", "airway",
                    tier=Tier.TIER2,
                    method="median(LAT)/median(AP) across nonzero slices"))
    R.append(_ratio("airway_concentricity_index", "airway",
                    tier=Tier.TIER2, maturity=Maturity.EXPERIMENTAL,
                    method="placeholder; future centerline-orthogonal definition"))

    # Regional airway compartments
    for compartment in (
        "nasopharyngeal", "retropalatal", "retroglossal",
        "retrolingual", "hypopharyngeal",
    ):
        R.append(_ml(f"{compartment}_volume_ml", compartment,
                     tier=Tier.TIER1, landmark_dependent=True,
                     shared_with_dental=compartment in ("retropalatal", "retroglossal"),
                     method=f"airway voxels within {compartment} z band × voxel volume",
                     required=("airway_mask",)))
        R.append(_mm2(f"{compartment}_min_csa_mm2", compartment,
                      tier=Tier.TIER1, landmark_dependent=True,
                      shared_with_dental=compartment in ("retropalatal", "retroglossal"),
                      method=f"min per-slice CSA inside {compartment} band",
                      required=("airway_mask",)))
    R.append(_mm2("retropalatal_csa_at_standard_level_mm2", "retropalatal",
                  landmark_dependent=True,
                  method="CSA at exact PNS-equivalent z"))
    R.append(_mm2("retroglossal_csa_at_standard_level_mm2", "retroglossal",
                  landmark_dependent=True,
                  method="CSA at exact epiglottis/hyoid z"))
    R.append(_str("airway_region_method", "airway",
                  method="how regional z bounds were derived (landmark or fallback)"))

    # Airway-tongue cross-features
    R.append(_ratio("retroglossal_airway_to_tongue_base_area_ratio",
                    "retroglossal", tier=Tier.TIER1,
                    landmark_dependent=True,
                    method="airway slice area / tongue-base slice area at RG level",
                    required=("airway_mask", "tongue_mask")))
    R.append(_ratio("retroglossal_airway_to_tongue_volume_ratio", "retroglossal",
                    tier=Tier.TIER1,
                    method="RG airway volume / tongue volume",
                    required=("airway_mask", "tongue_mask")))
    R.append(_bool("airway_min_csa_adjacent_to_tongue_base_flag", "airway",
                   method="True if min-CSA slice ∈ tongue-base z band"))

    # ----- Tongue -----
    R.append(_bool("tongue_mask_available", "tongue"))
    R.append(_str("tongue_mask_method", "tongue"))
    R.append(_bool("tongue_qc_pass", "tongue"))
    R.append(_str("tongue_qc_failure_reasons", "tongue"))
    R.append(_bool("tongue_artifact_warning", "tongue"))
    R.append(_bool("tongue_contrast_sensitive", "tongue"))
    R.append(_str("tongue_mask_source", "tongue"))
    R.append(_str("tongue_roi_confidence", "tongue"))
    R.append(_str("tongue_coverage_warning", "tongue"))

    for n in ("tongue_volume_mm3", "tongue_volume_ml"):
        R.append(_ml(n, "tongue", shared_with_dental=True,
                     method="tongue mask voxel count × voxel volume",
                     required=("tongue_mask",)))
    R.append(_hu("tongue_mean_hu", "tongue", required=("tongue_mask",)))
    R.append(_hu("tongue_median_hu", "tongue", required=("tongue_mask",)))
    R.append(_hu("tongue_std_hu", "tongue", required=("tongue_mask",)))
    R.append(_hu("tongue_p10_hu", "tongue", required=("tongue_mask",)))
    R.append(_hu("tongue_p90_hu", "tongue", required=("tongue_mask",)))
    R.append(_frac("tongue_low_hu_fraction", "tongue",
                   method="voxels with HU < low_hu_threshold_used / total",
                   required=("tongue_mask",)))
    R.append(_hu("tongue_low_hu_threshold_used", "tongue",
                 contrast_sensitive=True,
                 method="config: tongue.low_hu_threshold (default 30 HU)"))

    # Posterior tongue
    R.append(_bool("tongue_posterior_roi_available", "tongue_posterior"))
    R.append(_str("tongue_posterior_roi_method", "tongue_posterior"))
    R.append(_ml("tongue_posterior_volume_ml", "tongue_posterior",
                 tier=Tier.TIER1, maturity=Maturity.HEURISTIC,
                 method="posterior-1/3 of tongue mask OR landmark box"))
    R.append(_hu("tongue_posterior_mean_hu", "tongue_posterior",
                 tier=Tier.TIER1,
                 method="mean HU within posterior tongue ROI"))
    R.append(_hu("tongue_posterior_median_hu", "tongue_posterior"))
    R.append(_hu("tongue_posterior_std_hu", "tongue_posterior"))
    R.append(_hu("tongue_posterior_p10_hu", "tongue_posterior"))
    R.append(_hu("tongue_posterior_p90_hu", "tongue_posterior"))
    R.append(_frac("tongue_posterior_low_hu_fraction", "tongue_posterior",
                   tier=Tier.TIER1,
                   method="low-HU surrogate for tongue-fat; CTA-contrast sensitive"))

    # Tongue base / encroachment
    R.append(_ml("tongue_base_volume_ml", "tongue_base", tier=Tier.TIER1,
                 maturity=Maturity.HEURISTIC,
                 method="tongue voxels within tongue-base z band"))
    R.append(_mm2("tongue_base_area_at_retroglossal_level_mm2", "tongue_base",
                  tier=Tier.TIER1, landmark_dependent=True))
    R.append(_ratio("tongue_base_to_retroglossal_airway_ratio", "tongue_base",
                    tier=Tier.TIER1, landmark_dependent=True))
    R.append(_mm("tongue_base_posterior_displacement_mm", "tongue_base",
                 tier=Tier.TIER2, landmark_dependent=True,
                 method="distance from posterior pharyngeal wall to tongue posterior point"))
    R.append(_mm("tongue_base_inferior_displacement_mm", "tongue_base",
                 tier=Tier.TIER2, landmark_dependent=True))
    R.append(_mm2("retroglossal_airway_area_adjacent_to_tongue_base_mm2",
                  "retroglossal", landmark_dependent=True))
    R.append(_mm("tongue_base_airway_contact_length_mm", "tongue_base",
                 tier=Tier.TIER2, maturity=Maturity.EXPERIMENTAL,
                 landmark_dependent=True))

    # Tongue ratios
    R.append(_ratio("tongue_to_mandible_volume_ratio", "tongue", tier=Tier.TIER1,
                    shared_with_dental=True,
                    method="tongue_volume_ml / mandible_volume_ml",
                    required=("tongue_mask", "mandible_mask")))
    R.append(_ratio("tongue_to_oral_cavity_volume_ratio", "tongue", tier=Tier.TIER2,
                    method="tongue_volume_ml / oral_cavity_volume_ml",
                    required=("tongue_mask", "oral_cavity_mask")))
    R.append(_ratio("tongue_to_skeletal_enclosure_ratio", "tongue", tier=Tier.TIER2))

    # Lingual tonsil (kept in tongue family per registry policy)
    R.append(_bool("lingual_tonsil_roi_available", "tongue"))
    R.append(_ml("lingual_tonsil_volume_ml", "tongue", tier=Tier.TIER2,
                 maturity=Maturity.EXPERIMENTAL))
    R.append(_hu("lingual_tonsil_mean_hu", "tongue", tier=Tier.TIER2,
                 maturity=Maturity.EXPERIMENTAL))
    R.append(_ratio("lingual_tonsil_to_retroglossal_airway_ratio", "tongue",
                    tier=Tier.TIER2, maturity=Maturity.EXPERIMENTAL))

    # ----- Mandible / Oral cavity -----
    R.append(_bool("mandible_mask_available", "mandible"))
    R.append(_str("mandible_mask_method", "mandible"))
    for n in ("mandible_volume_mm3", "mandible_volume_ml"):
        R.append(_ml(n, "mandible", shared_with_dental=True,
                     method="mandible mask voxel count × voxel volume",
                     required=("mandible_mask",)))
    R.append(_bool("mandibular_plane_available", "mandible"))
    R.append(_str("mandibular_plane_method", "mandible"))
    R.append(_mm("mandibular_plane_to_hyoid_distance_mm", "mandible",
                 shared_with_dental=True, landmark_dependent=True,
                 method="perpendicular distance from mandibular plane to hyoid centroid"))
    R.append(_mm("hyoid_to_mandible_distance_mm", "mandible",
                 landmark_dependent=True,
                 method="alias of mandibular_plane_to_hyoid_distance_mm via different anchor"))
    R.append(_mm2("cervicomandibular_ring_area_mm2", "skeletal",
                  tier=Tier.TIER2, landmark_dependent=True,
                  method="axial slice area enclosed by mandible ramus + hyoid + spine at hyoid level"))
    R.append(_str("cervicomandibular_ring_method", "skeletal"))
    R.append(_bool("oral_cavity_mask_available", "oral_cavity"))
    R.append(_ml("oral_cavity_volume_ml", "oral_cavity", tier=Tier.TIER2))
    R.append(_str("oral_cavity_method", "oral_cavity"))

    # ----- Soft palate / Uvula / Lateral wall / Tonsil -----
    R.append(_bool("soft_palate_mask_available", "soft_palate"))
    R.append(_mm("soft_palate_length_mm", "soft_palate", tier=Tier.TIER2,
                 landmark_dependent=True))
    R.append(_mm("soft_palate_thickness_max_mm", "soft_palate", tier=Tier.TIER2))
    R.append(_mm("soft_palate_thickness_mean_mm", "soft_palate", tier=Tier.TIER2))
    R.append(_ml("soft_palate_volume_ml", "soft_palate", tier=Tier.TIER2))
    R.append(_hu("soft_palate_mean_hu", "soft_palate", tier=Tier.TIER2))
    R.append(_mm("soft_palate_inferior_tip_z_mm", "soft_palate", tier=Tier.TIER2))
    R.append(_mm("soft_palate_to_posterior_pharyngeal_wall_distance_mm",
                 "soft_palate", tier=Tier.TIER2, landmark_dependent=True))
    R.append(_ratio("soft_palate_to_retropalatal_airway_ratio", "soft_palate",
                    tier=Tier.TIER2, landmark_dependent=True))
    R.append(_bool("uvula_visible", "soft_palate"))
    R.append(_mm("uvula_length_mm", "soft_palate", tier=Tier.TIER2))
    R.append(_mm("uvula_width_mm", "soft_palate", tier=Tier.TIER2))
    R.append(_ml("uvula_volume_ml", "soft_palate", tier=Tier.TIER2))
    R.append(_mm("lateral_pharyngeal_wall_left_thickness_mm", "lateral_wall",
                 tier=Tier.TIER2, landmark_dependent=True))
    R.append(_mm("lateral_pharyngeal_wall_right_thickness_mm", "lateral_wall",
                 tier=Tier.TIER2, landmark_dependent=True))
    R.append(_mm("lateral_pharyngeal_wall_mean_thickness_mm", "lateral_wall",
                 tier=Tier.TIER2))
    R.append(_ratio("lateral_pharyngeal_wall_asymmetry_index", "lateral_wall",
                    tier=Tier.TIER2,
                    method="(R-L)/(R+L) of mean wall thickness"))
    R.append(_ratio("lateral_wall_to_airway_ratio_at_retropalatal_level",
                    "lateral_wall", tier=Tier.TIER2, landmark_dependent=True))
    R.append(_ratio("lateral_wall_to_airway_ratio_at_retroglossal_level",
                    "lateral_wall", tier=Tier.TIER2, landmark_dependent=True))
    R.append(_bool("palatine_tonsil_left_visible", "tonsil"))
    R.append(_bool("palatine_tonsil_right_visible", "tonsil"))
    R.append(_ml("palatine_tonsil_left_volume_ml", "tonsil", tier=Tier.TIER2,
                 maturity=Maturity.EXPERIMENTAL))
    R.append(_ml("palatine_tonsil_right_volume_ml", "tonsil", tier=Tier.TIER2,
                 maturity=Maturity.EXPERIMENTAL))
    R.append(_ml("palatine_tonsil_total_volume_ml", "tonsil", tier=Tier.TIER2,
                 maturity=Maturity.EXPERIMENTAL))
    R.append(_ratio("tonsil_to_retropalatal_airway_ratio", "tonsil",
                    tier=Tier.TIER2, maturity=Maturity.EXPERIMENTAL))

    # ----- Skeletal / Hyoid -----
    R.append(_bool("hyoid_detected", "skeletal"))
    R.append(_bool("hyoid_mask_available", "skeletal"))
    R.append(_mm("hyoid_centroid_x_mm", "skeletal", landmark_dependent=True))
    R.append(_mm("hyoid_centroid_y_mm", "skeletal", landmark_dependent=True))
    R.append(_mm("hyoid_centroid_z_mm", "skeletal", landmark_dependent=True))
    for n in (
        "hyoid_to_posterior_pharyngeal_wall_distance_mm",
        "hyoid_to_c2_distance_mm", "hyoid_to_c3_distance_mm",
        "hyoid_to_c4_distance_mm", "hyoid_to_epiglottis_distance_mm",
        "hyoid_vertical_position_relative_to_mandible_mm",
        "hyoid_ap_position_relative_to_cervical_spine_mm",
        "neck_length_mm", "laryngeal_descent_mm",
        "hard_palate_to_hyoid_distance_mm",
        "posterior_nasal_spine_to_epiglottis_distance_mm",
    ):
        R.append(_mm(n, "skeletal", tier=Tier.TIER1 if n in (
            "hyoid_to_c3_distance_mm",
            "hard_palate_to_hyoid_distance_mm",
        ) else Tier.TIER2,
            landmark_dependent=True,
            shared_with_dental=n.startswith("hyoid_to_c") or
                                n.startswith("hard_palate_"),
        ))
    R.append(_ratio("skeletal_enclosure_index", "skeletal", tier=Tier.TIER2,
                    maturity=Maturity.HEURISTIC))

    # ----- Fat -----
    R.append(_hu("fat_hu_min_used", "fat",
                 contrast_sensitive=False,
                 method="config: fat.fat_hu_min"))
    R.append(_hu("fat_hu_max_used", "fat",
                 contrast_sensitive=False,
                 method="config: fat.fat_hu_max"))
    R.append(_str("fat_roi_method", "fat"))
    R.append(_bool("contrast_phase_sensitive_flag", "fat",
                   method="True if CTA contrast might shift HU stats"))

    # Cervical
    R.append(_ml("fat_cervical_volume_mm3", "fat_cervical"))
    R.append(_ml("fat_cervical_volume_ml", "fat_cervical", tier=Tier.TIER1,
                 cta_specific=True))
    R.append(_ml("fat_cervical_total_volume_ml", "fat_cervical", tier=Tier.TIER1,
                 cta_specific=True))
    R.append(_hu("fat_cervical_mean_hu", "fat_cervical", contrast_sensitive=True,
                 cta_specific=True))
    R.append(_hu("fat_cervical_median_hu", "fat_cervical"))
    R.append(_hu("fat_cervical_p10_hu", "fat_cervical"))
    R.append(_hu("fat_cervical_p90_hu", "fat_cervical"))
    R.append(_hu("fat_cervical_std_hu", "fat_cervical"))
    R.append(_mm2("fat_cervical_area_at_hyoid_level_mm2", "fat_cervical",
                  tier=Tier.TIER2, landmark_dependent=True))
    R.append(_mm2("fat_cervical_area_at_retropalatal_level_mm2", "fat_cervical",
                  tier=Tier.TIER2, landmark_dependent=True))
    R.append(_mm2("fat_cervical_area_at_retroglossal_level_mm2", "fat_cervical",
                  tier=Tier.TIER2, landmark_dependent=True))

    # Sub-cutaneous / deep
    R.append(_ml("fat_subcutaneous_cervical_volume_ml", "fat_subcutaneous",
                 tier=Tier.TIER1, cta_specific=True))
    R.append(_hu("fat_subcutaneous_cervical_mean_hu", "fat_subcutaneous"))
    R.append(_frac("fat_subcutaneous_fraction_of_neck_area", "fat_subcutaneous"))
    R.append(_ml("fat_deep_cervical_volume_ml", "fat_deep", tier=Tier.TIER1,
                 cta_specific=True))
    R.append(_hu("fat_deep_cervical_mean_hu", "fat_deep"))
    R.append(_ratio("fat_deep_to_subcutaneous_ratio", "fat_deep"))

    # Anatomically-constrained neck slab (FOV-robust cervical adiposity).
    # A fixed-height slab anchored on the airway min-CSA slice, plus dimensionless
    # fractions/ratios that do not scale with the imaged z-extent — these fix the
    # plain cervical-volume inflation on tall head-to-chest CTAs.
    R.append(_str("fat_neck_anchor_method", "fat_cervical",
                  method="min_csa | cervical_zrange | unavailable_no_anchor"))
    R.append(_mm("fat_neck_roi_radius_mm", "fat_cervical",
                 method="in-plane containment radius around the airway centroid"))
    R.append(_mm("fat_neck_slab_height_mm", "fat_cervical", tier=Tier.TIER1,
                 method="physical height of the anchored neck slab"))
    R.append(_ml("fat_neck_slab_volume_ml", "fat_cervical", tier=Tier.TIER1,
                 cta_specific=True,
                 method="fat within a fixed ±slab_mm window around the airway "
                        "min-CSA slice ∩ body ∩ fat-HU (FOV-robust)"))
    R.append(_ml("fat_neck_slab_subcutaneous_volume_ml", "fat_cervical",
                 tier=Tier.TIER1, cta_specific=True))
    R.append(_ml("fat_neck_slab_deep_volume_ml", "fat_cervical", tier=Tier.TIER1,
                 cta_specific=True))
    R.append(_hu("fat_neck_slab_mean_hu", "fat_cervical", contrast_sensitive=True))
    R.append(_frac("fat_neck_slab_fat_fraction", "fat_cervical", tier=Tier.TIER1,
                   method="slab fat voxels / slab body voxels (dimensionless, FOV-invariant)"))
    R.append(_ratio("fat_neck_slab_deep_to_subcutaneous_ratio", "fat_cervical",
                    tier=Tier.TIER1))
    R.append(_ratio("fat_neck_slab_to_airway_volume_ratio", "fat_cervical",
                    tier=Tier.TIER1, method="slab fat volume / airway volume"))
    R.append(_mm2("fat_neck_area_at_min_csa_mm2", "fat_cervical", tier=Tier.TIER1,
                  method="cervical fat area on the airway min-CSA slice"))
    R.append(_mm2("fat_neck_body_area_at_min_csa_mm2", "fat_cervical",
                  tier=Tier.TIER1, method="neck (body) area on the min-CSA slice"))
    R.append(_frac("fat_neck_area_fraction_at_min_csa", "fat_cervical",
                   tier=Tier.TIER1,
                   method="fat area / neck area at min-CSA slice (dimensionless)"))
    R.append(_ml("fat_deep_peripharyngeal_volume_ml", "fat_deep_peripharyngeal",
                 tier=Tier.TIER1, cta_specific=True,
                 method="Deep cervical fat within a physical-distance band around the airway"))
    R.append(_hu("fat_deep_peripharyngeal_mean_hu", "fat_deep_peripharyngeal"))
    R.append(_str("fat_deep_peripharyngeal_roi_method",
                  "fat_deep_peripharyngeal"))

    # Parapharyngeal
    for side in ("left", "right", "total"):
        R.append(_ml(f"fat_parapharyngeal_{side}_volume_ml",
                     "fat_parapharyngeal", tier=Tier.TIER1, cta_specific=True))
    for side in ("left", "right"):
        R.append(_hu(f"fat_parapharyngeal_{side}_mean_hu",
                     "fat_parapharyngeal"))
    R.append(_ratio("fat_parapharyngeal_asymmetry_index", "fat_parapharyngeal"))
    R.append(_ratio("fat_parapharyngeal_to_airway_ratio", "fat_parapharyngeal",
                    tier=Tier.TIER1))
    for level in ("retropalatal", "retroglossal", "subglosso_supraglottic"):
        for side in ("left", "right", "total"):
            R.append(_mm2(
                f"fat_parapharyngeal_area_{level}_{side}_mm2",
                "fat_parapharyngeal",
                landmark_dependent=True, tier=Tier.TIER2,
            ))
    for level in ("retropalatal", "retroglossal", "min_csa"):
        R.append(_ratio(f"fat_parapharyngeal_to_airway_ratio_{level}",
                        "fat_parapharyngeal", landmark_dependent=level != "min_csa",
                        tier=Tier.TIER1))
    R.append(_mm2("fat_parapharyngeal_area_at_min_airway_csa_mm2",
                  "fat_parapharyngeal"))
    R.append(_str("fat_parapharyngeal_roi_method", "fat_parapharyngeal"))
    R.append(_str("fat_anatomy_prior_masks_used", "fat"))
    R.append(_str("fat_regional_anatomy_prior_masks_used", "fat"))
    R.append(_str("fat_regional_parapharyngeal_roi_method",
                  "fat_parapharyngeal"))

    # Retropharyngeal
    R.append(_ml("fat_retropharyngeal_volume_ml", "fat_retropharyngeal",
                 tier=Tier.TIER1, cta_specific=True))
    R.append(_hu("fat_retropharyngeal_mean_hu", "fat_retropharyngeal"))
    R.append(_mm("fat_retropharyngeal_max_thickness_mm", "fat_retropharyngeal"))
    R.append(_mm("fat_retropharyngeal_mean_thickness_mm", "fat_retropharyngeal"))
    R.append(_mm2("fat_retropharyngeal_area_at_retropalatal_level_mm2",
                  "fat_retropharyngeal", landmark_dependent=True))
    R.append(_mm2("fat_retropharyngeal_area_at_retroglossal_level_mm2",
                  "fat_retropharyngeal", landmark_dependent=True))
    R.append(_str("fat_retropharyngeal_roi_method", "fat_retropharyngeal"))

    # Facial / buccal (only when FOV covers face)
    R.append(_ml("fat_facial_total_volume_ml", "fat_facial",
                 tier=Tier.TIER2, maturity=Maturity.HEURISTIC))
    R.append(_ml("fat_buccal_left_volume_ml", "fat_facial",
                 tier=Tier.TIER2, maturity=Maturity.HEURISTIC))
    R.append(_ml("fat_buccal_right_volume_ml", "fat_facial",
                 tier=Tier.TIER2, maturity=Maturity.HEURISTIC))
    R.append(_ratio("fat_facial_to_parapharyngeal_ratio", "fat_facial",
                    tier=Tier.TIER2))

    # ----- Evidence-aware method + confidence fields -----
    # These describe HOW each feature family was produced and a coarse
    # confidence label (high|moderate|low|missing). They are referenced by
    # evidence_registry.EvidenceSpec.confidence_field_name. Adding them as
    # stable string columns lets downstream analysis filter by provenance.
    for n in (
        "tongue_confidence",
        "fat_cervical_method", "fat_cervical_confidence",
        "fat_parapharyngeal_method", "fat_parapharyngeal_confidence",
        "fat_retropharyngeal_method", "fat_retropharyngeal_confidence",
        "fat_submandibular_method", "fat_submandibular_confidence",
        "fat_periairway_method", "fat_periairway_confidence",
        "fat_c5_nat_method", "fat_c5_nat_confidence",
        "fat_pericarotid_method", "fat_pericarotid_confidence",
    ):
        R.append(_str(n, region="fat" if n.startswith("fat_") else "tongue",
                      method="provenance/confidence label: high|moderate|low|missing"))

    # ----- Composites (exploratory) -----
    for n in (
        "cta_osa_airway_narrowing_index_untrained",
        "cta_osa_tongue_crowding_index_untrained",
        "cta_osa_fat_burden_index_untrained",
        "cta_osa_skeletal_restriction_index_untrained",
        "cta_osa_combined_anatomy_index_untrained",
    ):
        R.append(MetricSpec(
            feature_name=n, family="composite", region="case",
            unit="ratio", tier=Tier.EXPLORATORY, maturity=Maturity.EXPERIMENTAL,
            extraction_method=(
                "unstandardised weighted sum of component features; only "
                "computed when ALL components present"
            ),
            notes="DO NOT USE FOR CLINICAL DECISION — name suffix _untrained "
                  "is intentional.",
        ))
    R.append(_str("composite_score_method", "case", tier=Tier.EXPLORATORY))
    R.append(_str("composite_score_disclaimer", "case", tier=Tier.EXPLORATORY))

    # ----- Optional carotid / thoracic / radiomics presence flags -----
    R.append(_bool("perivascular_available", "optional"))
    R.append(_bool("thoracic_available", "optional"))
    R.append(_bool("radiomics_available", "optional"))

    return tuple(R)


_REGISTRY: tuple[MetricSpec, ...] = _build_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def all_metrics() -> tuple[MetricSpec, ...]:
    """Every metric the pipeline can emit, in registration order."""
    return _REGISTRY


def feature_names() -> list[str]:
    """Stable list of `feature_name` strings, in canonical order.

    Use this to seed empty feature rows so every case has the same columns
    regardless of which masks/landmarks were available.
    """
    return [m.feature_name for m in _REGISTRY]


def by_family(family: str) -> list[MetricSpec]:
    return [m for m in _REGISTRY if m.family == family]


def by_tier(tier: Tier) -> list[MetricSpec]:
    return [m for m in _REGISTRY if m.tier == tier]


def shared_feature_names() -> list[str]:
    """Subset that BOTH stroke-CTA and dental/CBCT pipelines can emit.

    Replaces the legacy hard-coded list in :mod:`shared_schema`.
    """
    return [m.feature_name for m in _REGISTRY if m.shared_with_dental]


def landmark_dependent_features() -> list[str]:
    return [m.feature_name for m in _REGISTRY if m.landmark_dependent]


def find(name: str) -> Optional[MetricSpec]:
    for m in _REGISTRY:
        if m.feature_name == name:
            return m
    return None


def empty_row() -> dict[str, Any]:
    """Default-valued row with one key per metric.

    The defaults respect each metric's :attr:`MetricSpec.missingness_behaviour`,
    so an empty case still produces a CSV row with stable columns and
    sensible nulls (NaN for numerics, "" for strings, False for bools).
    """
    out: dict[str, Any] = {}
    for m in _REGISTRY:
        out[m.feature_name] = _default_for(m)
    return out


def _default_for(m: MetricSpec) -> Any:
    if m.missingness_behaviour == "bool_False":
        return False
    if m.missingness_behaviour == "empty_str":
        return ""
    if m.missingness_behaviour == "-1_int":
        return -1
    return float("nan")


def to_records() -> list[dict[str, Any]]:
    """Serialise the registry as a list of plain dicts (for CSV export)."""
    return [{
        "feature_name": m.feature_name,
        "family": m.family,
        "region": m.region,
        "unit": m.unit,
        "tier": m.tier.value,
        "maturity": m.maturity.value,
        "required_inputs": ";".join(m.required_inputs),
        "optional_inputs": ";".join(m.optional_inputs),
        "landmark_dependent": m.landmark_dependent,
        "contrast_sensitive": m.contrast_sensitive,
        "cta_specific": m.cta_specific,
        "shared_with_dental": m.shared_with_dental,
        "extraction_method": m.extraction_method,
        "missingness_behaviour": m.missingness_behaviour,
        "notes": m.notes,
    } for m in _REGISTRY]


def to_json(path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_records(), indent=2))
    return Path(path)


def to_csv(path: Path) -> Path:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = to_records()
    fieldnames = list(records[0].keys()) if records else []
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return Path(path)
