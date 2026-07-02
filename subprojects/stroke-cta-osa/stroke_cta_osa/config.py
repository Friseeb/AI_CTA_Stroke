"""Pydantic-validated configuration. All numeric defaults are spelt out so
researchers can see them without reading code, and the YAML default at
configs/default.yaml mirrors them 1:1 for documentation purposes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class CoverageRequirements(BaseModel):
    include_hard_palate: Literal["required", "optional", "ignore"] = "optional"
    include_hyoid: Literal["required", "optional", "ignore"] = "optional"
    include_epiglottis: Literal["required", "optional", "ignore"] = "optional"
    include_cervical_soft_tissues: Literal["required", "optional", "ignore"] = "required"


class HUConfig(BaseModel):
    fat_hu_min: float = -190.0
    fat_hu_max: float = -30.0
    air_hu_max: float = -500.0
    bone_hu_min: float = 250.0


class IngestionConfig(BaseModel):
    input_type: Literal["dicom", "nifti", "auto"] = "auto"
    orientation: Literal["RAS", "LPS", "native"] = "RAS"
    resample_spacing_mm: Optional[float] = None
    age_floor_years: int = 18


class AirwayConfig(BaseModel):
    use_existing_dental_airway_outputs: bool = False
    dental_airway_mask_path: Optional[str] = None
    dental_landmarks_path: Optional[str] = None
    dental_features_path: Optional[str] = None
    fallback_method: Literal[
        "threshold_connected_component",
        "external_mask_only",
        "none",
    ] = "threshold_connected_component"
    external_mask_path: Optional[str] = None
    min_component_volume_ml: float = 1.0
    morphology_closing_mm: float = 1.0
    # Fallback airway can leak pharynx→trachea→bronchi→LUNGS into one component
    # on tall CTAs. Any axial slice of the chosen air component exceeding this
    # cross-section is lung-scale and is dropped, breaking the connection so only
    # the pharyngeal/tracheal column survives. 0 disables the cap.
    fallback_max_airway_slice_area_mm2: float = 3000.0
    centerline_orthogonal_csa: bool = False  # axial-CSA only in v1
    retropalatal_window_mm: float = 15.0
    retroglossal_window_mm: float = 15.0
    retrolingual_window_mm: float = 10.0


class FatConfig(BaseModel):
    parapharyngeal_lateral_band_mm: float = 25.0
    parapharyngeal_axial_window_mm: float = 30.0
    retropharyngeal_posterior_band_mm: float = 15.0
    retropharyngeal_axial_window_mm: float = 30.0
    subcutaneous_erosion_mm: float = 6.0
    body_air_threshold_hu: float = -250.0
    exclude_bone_for_fat: bool = True
    exclude_vessels_hu_min: float = 120.0  # contrast-enhanced vessel exclusion
    use_anatomy_priors: bool = True
    anatomy_prior_dilation_mm: float = 1.0
    parapharyngeal_sector_min_lateral_fraction: float = 0.75
    prevertebral_mask_paths: list[str] = Field(default_factory=list)
    retropharyngeal_use_oropharyngeal_window: bool = True
    retropharyngeal_prevertebral_margin_mm: float = 1.0
    retropharyngeal_lateral_margin_mm: float = 5.0
    # --- Anatomically-constrained neck slab (FOV-robust cervical fat) ---
    # The plain cervical volume scales with the imaged z-extent, which on tall
    # head-to-chest CTAs badly inflates it. Anchoring a fixed-height slab on the
    # airway min-CSA slice and reporting fractions/ratios removes that FOV
    # dependence. See stroke_cta_osa.fat._anchored_neck_features.
    neck_slab_enabled: bool = True
    neck_slab_half_height_mm: float = 40.0   # ±40 mm ⇒ ~8 cm neck slab
    neck_slab_anchor: Literal["min_csa", "cervical_zrange"] = "min_csa"
    # In-plane containment: a cylinder of this radius around the airway centroid,
    # so shoulders / arms / immobilisation padding (all low-HU, else miscounted
    # as fat) are excluded. 0 disables in-plane containment.
    neck_slab_radius_mm: float = 75.0
    # The airway min-CSA anchor sits at the tongue-base / retroglossal level,
    # which is fat-rich (floor of mouth, submandibular) and abuts the air-filled
    # oropharynx. Shift the slab centre inferiorly by this much so it sits over
    # the true mid-cervical neck (hyoid/thyroid level). Assumes RAS orientation
    # (the ingestion default); clamped to stay within the imaged airway span.
    # Default 0: testing showed it did not reduce the high-fat tail (that was an
    # airway-segmentation leak, fixed via fallback_max_airway_slice_area_mm2).
    neck_slab_inferior_offset_mm: float = 0.0


class RadiomicsConfig(BaseModel):
    enabled: bool = False
    rois: list[Literal[
        "airway", "tongue", "posterior_tongue", "cervical_fat",
        "parapharyngeal_fat", "retropharyngeal_fat", "soft_palate",
        "lateral_wall", "combined_airway_soft_tissue",
    ]] = Field(default_factory=lambda: [
        "airway", "tongue", "parapharyngeal_fat", "soft_palate",
    ])
    bin_width_hu: float = 25.0
    label_value: int = 1


class TongueConfig(BaseModel):
    """Tongue-module configuration; see `stroke_cta_osa.tongue`."""
    enabled: bool = True
    require_mask_for_volume: bool = True
    allow_posterior_roi_fallback: bool = False
    low_hu_threshold: float = 30.0
    low_hu_threshold_mode: Literal["absolute", "relative"] = "absolute"
    record_contrast_sensitivity: bool = True
    external_mask_path: Optional[str] = None


class MandibleConfig(BaseModel):
    enabled: bool = True
    allow_bone_threshold_fallback: bool = False
    bone_hu_min: float = 250.0
    require_mask_for_volume: bool = True
    bone_min_volume_ml: float = 5.0
    external_mask_path: Optional[str] = None
    dental_mandible_mask_path: Optional[str] = None


class OralCavityConfig(BaseModel):
    enabled: bool = True
    external_mask_path: Optional[str] = None


class SoftTissueConfig(BaseModel):
    enabled: bool = True
    require_masks_for_volumes: bool = True
    allow_landmark_length_fallback: bool = True
    lateral_wall_band_mm: float = 15.0
    lateral_wall_axial_window_mm: float = 20.0
    body_air_threshold_hu: float = -250.0
    soft_palate_mask_path: Optional[str] = None
    uvula_mask_path: Optional[str] = None
    palatine_tonsil_left_mask_path: Optional[str] = None
    palatine_tonsil_right_mask_path: Optional[str] = None


class SkeletalConfig(BaseModel):
    enabled: bool = True
    allow_landmark_only_distances: bool = True
    allow_hyoid_threshold_fallback: bool = False
    hyoid_mask_path: Optional[str] = None


class AirwayRegionConfig(BaseModel):
    enabled: bool = True
    prefer_landmark_defined_regions: bool = True
    allow_axial_approximation: bool = True
    save_csa_profile: bool = False


class FatRegionConfig(BaseModel):
    enabled: bool = True
    parapharyngeal_roi_method: Literal[
        "airway_relative", "atlas", "external"
    ] = "airway_relative"
    retropharyngeal_roi_method: Literal[
        "airway_spine_relative", "airway_relative", "external"
    ] = "airway_spine_relative"
    enable_facial_fat: bool = False


class LandmarkConfig(BaseModel):
    explicit_path: Optional[str] = None
    dental_landmarks_path: Optional[str] = None
    allow_heuristic_fallback: bool = True


class CompositesConfig(BaseModel):
    enabled: bool = False
    require_batch_standardization: bool = True
    suffix: Literal["untrained"] = "untrained"
    cohort_stats_path: Optional[str] = None


class PerivascularConfig(BaseModel):
    enabled: bool = False
    carotid_mask_path: Optional[str] = None
    plaque_mask_path: Optional[str] = None
    pericarotid_shell_mm: float = 3.0


class ThoracicConfig(BaseModel):
    enabled: bool = False
    epicardial_mask_path: Optional[str] = None
    mediastinal_mask_path: Optional[str] = None


class ClinicalMergeConfig(BaseModel):
    clinical_csv_path: Optional[str] = None
    patient_id_column: str = "patient_id"
    scan_id_column: str = "scan_id"


class FeatureSelectionConfig(BaseModel):
    """Which evidence-gated feature sets to emit and which is the default.

    The pipeline always computes every implemented feature; this block only
    governs the *subset* CSVs and the default modelling set recorded per run.
    """
    output_all_features: bool = True
    output_feature_sets: bool = True
    default_modeling_feature_set: Literal[
        "core_osa_backed", "core_plus_anatomic_extensions",
        "core_plus_cardiometabolic_ct", "all_features_exploratory",
    ] = "core_osa_backed"
    allowed_feature_sets: list[str] = Field(default_factory=lambda: [
        "core_osa_backed", "core_plus_anatomic_extensions",
        "core_plus_cardiometabolic_ct", "all_features_exploratory",
    ])


class EvidenceTiersConfig(BaseModel):
    """Toggle whole evidence tiers on/off. Tier 1 cannot be disabled here —
    the core OSA-backed set must always be available."""
    include_tier_1_core_osa_backed: bool = True
    include_tier_2_osa_plausible_ct_anatomic: bool = True
    include_tier_3_ct_cardiometabolic_or_vascular: bool = True
    include_tier_4_stroke_cta_novel_exploratory: bool = True


class OutputConfig(BaseModel):
    save_masks: bool = False
    save_qc_images: bool = True
    save_overlays: bool = False


class QCConfig(BaseModel):
    min_z_extent_mm: float = 60.0
    max_slice_thickness_mm: float = 3.0
    dental_artifact_hu_threshold: float = 2500.0
    dental_artifact_voxel_fraction_warn: float = 0.001


class PipelineConfig(BaseModel):
    """Top-level config object materialised from YAML."""
    disclaimer: str = "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS"
    output_dir: Optional[str] = None
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    hu: HUConfig = Field(default_factory=HUConfig)
    airway: AirwayConfig = Field(default_factory=AirwayConfig)
    fat: FatConfig = Field(default_factory=FatConfig)
    coverage: CoverageRequirements = Field(default_factory=CoverageRequirements)
    qc: QCConfig = Field(default_factory=QCConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    radiomics: RadiomicsConfig = Field(default_factory=RadiomicsConfig)
    perivascular: PerivascularConfig = Field(default_factory=PerivascularConfig)
    thoracic: ThoracicConfig = Field(default_factory=ThoracicConfig)
    clinical: ClinicalMergeConfig = Field(default_factory=ClinicalMergeConfig)

    # ---- new module configs (additive) ----
    tongue: TongueConfig = Field(default_factory=TongueConfig)
    mandible: MandibleConfig = Field(default_factory=MandibleConfig)
    oral_cavity: OralCavityConfig = Field(default_factory=OralCavityConfig)
    soft_tissue: SoftTissueConfig = Field(default_factory=SoftTissueConfig)
    skeletal: SkeletalConfig = Field(default_factory=SkeletalConfig)
    airway_regions: AirwayRegionConfig = Field(default_factory=AirwayRegionConfig)
    fat_regions: FatRegionConfig = Field(default_factory=FatRegionConfig)
    landmarks: LandmarkConfig = Field(default_factory=LandmarkConfig)
    composites: CompositesConfig = Field(default_factory=CompositesConfig)

    # ---- evidence-aware feature selection (additive) ----
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig)
    evidence_tiers: EvidenceTiersConfig = Field(
        default_factory=EvidenceTiersConfig)

    def hash(self) -> str:
        """SHA-1 of the JSON-serialised config. Recorded per case so feature
        rows can be filtered by the config that produced them."""
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    if not Path(path).exists():
        return PipelineConfig()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig.model_validate(raw)


def apply_overrides(cfg: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    """Apply dotted overrides (e.g. {'airway.fallback_method': 'none'})."""
    data = cfg.model_dump()
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return PipelineConfig.model_validate(data)
