"""Pydantic config models and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PreprocessingConfig(BaseModel):
    target_spacing_mm: float = 0.5
    orientation: str = "RAS"
    hu_clip_min: float = -1000.0
    hu_clip_max: float = 3000.0
    min_patient_age_years: int = 18


class ROIConfig(BaseModel):
    method: Literal[
        "totalseg_teeth",
        "totalseg_craniofacial",
        "dentalsegmentator_coarse",
        "threshold_fallback",
    ] = "totalseg_teeth"
    margin_mm: float = 20.0
    threshold_fallback_hu: float = 700.0
    # Sanity gate: a single tooth label cannot plausibly span more than this.
    # On out-of-domain contrast CTA, TotalSegmentator-teeth sometimes smears a
    # tooth across the whole scan; such cases are failed instead of yielding a
    # whole-volume "dentition ROI". Set <= 0 to disable.
    max_tooth_extent_mm: float = 40.0


class DefaceConfig(BaseModel):
    mode: Literal["none", "mask_only", "posthoc", "pre"] = "mask_only"
    executable: Optional[str] = None


class TotalSegmentatorConfig(BaseModel):
    task: str = "teeth"
    fast: bool = False
    device: str = "cpu"
    weights_dir: Optional[str] = None


class DentalSegmentatorConfig(BaseModel):
    weights_path: Optional[str] = None
    nnunet_results_dir: Optional[str] = None


class OralSegConfig(BaseModel):
    model_path: Optional[str] = None


class RAILConfig(BaseModel):
    model_path: Optional[str] = None


class SegmentationConfig(BaseModel):
    backend: Literal[
        "totalseg_teeth", "dentalsegmentator", "oralseg", "rail", "none"
    ] = "totalseg_teeth"
    totalsegmentator: TotalSegmentatorConfig = Field(default_factory=TotalSegmentatorConfig)
    dentalsegmentator: DentalSegmentatorConfig = Field(default_factory=DentalSegmentatorConfig)
    oralseg: OralSegConfig = Field(default_factory=OralSegConfig)
    rail: RAILConfig = Field(default_factory=RAILConfig)


class FeaturesConfig(BaseModel):
    # Minimum segmented volume for an implant candidate. TotalSegmentator writes a
    # label file per class even when empty, so without this every case gets a
    # 0-volume "implant" candidate. A real implant is >100 mm3.
    candidate_min_volume_mm3: float = 20.0
    # Crowns/bridges over-label on CTA (TS "crown" class can't be cleanly separated
    # from dense natural enamel), so require a substantial restoration volume AND a
    # supra-enamel median density: enamel/saturation tops out ~3000 HU on this
    # cohort while metal/ceramic crowns read far higher (median ~4700-11600 HU).
    crown_min_volume_mm3: float = 200.0
    crown_min_median_hu: float = 3000.0
    periapical_search_radius_mm: float = 5.0
    # Periapical lucency = a focal soft-tissue/fluid density (granuloma/cyst/abscess,
    # ≈ 0–60 HU) where alveolar bone (>~200 HU) should be. The candidate HU band is
    # therefore (lesion_hu_min, lesion_hu_max): above fatty marrow (≈ −50 to −120 HU,
    # normal) and below bone. (The previous band of (−300, −50) targeted fat and
    # flagged normal marrow in ~100% of cases.)
    periapical_lesion_hu_min: float = -20.0
    periapical_lesion_hu_max: float = 80.0
    periapical_min_volume_mm3: float = 50.0
    # A true periapical lucency is INTRA-OSSEOUS: a focal low-density defect encased
    # in alveolar bone. Without this, any soft-tissue-density blob near a tooth apex
    # (gingiva, PDL space, vessels, partial-volume edges) is flagged, inflating
    # prevalence to ~96%. Require a candidate's surrounding shell to be mostly
    # jawbone, so only intra-bony lesions pass.
    periapical_bone_shell_mm: float = 2.0
    periapical_min_bone_encasement: float = 0.6
    # mm dilation applied to anatomical exclusion labels (sinuses, pharynx,
    # neurovascular canals). A small buffer absorbs segmentation edge error
    # so the air–bone interface itself doesn't register as a "lucency".
    periapical_anatomy_exclusion_mm: float = 2.0
    periodontal_bone_shell_mm: float = 3.0
    periodontal_min_bone_coverage: float = 0.15
    allow_threshold_fallback_features: bool = False


class QCConfig(BaseModel):
    dpi: int = 150
    window_center_hu: float = 400.0
    window_width_hu: float = 1500.0
    mip_slab_mm: float = 20.0
    overlay_alpha: float = 0.4


class ReportConfig(BaseModel):
    include_dicom_series_uid: bool = True
    include_reconstruction_kernel: bool = True


class PipelineConfig(BaseModel):
    disclaimer: str = "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS"
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    deface: DefaceConfig = Field(default_factory=DefaceConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: Path | None = None) -> PipelineConfig:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not path.exists():
        return PipelineConfig()
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return PipelineConfig.model_validate(raw)


def merge_cli_overrides(cfg: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """Apply flat CLI overrides on top of a loaded config."""
    data = cfg.model_dump()
    for key, val in overrides.items():
        if val is None:
            continue
        parts = key.split(".")
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return PipelineConfig.model_validate(data)
