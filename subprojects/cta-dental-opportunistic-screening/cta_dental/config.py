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
    periapical_search_radius_mm: float = 5.0
    periapical_low_hu_threshold: float = -50.0
    # Below this HU the voxel is treated as air (sinus, airway, oral cavity),
    # not periapical pathology. Granuloma / cyst / abscess sit at ≈ 0–50 HU,
    # mucus at ≈ 0–40 HU, pure air near −1000 HU. Anything < −300 HU inside
    # the periapical search shell is almost certainly an air pocket.
    periapical_air_hu_threshold: float = -300.0
    periapical_min_volume_mm3: float = 10.0
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
