"""Shared lightweight dataclasses for cross-module communication.

Every result type that flows between modules lives here so import cycles
between e.g. airway.py and fat.py are impossible. These are deliberately
plain dataclasses (not pydantic): they describe *runtime values*, not user
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# --- Image container --------------------------------------------------------

@dataclass
class CTAImage:
    """Loaded CTA volume plus the metadata feature code actually needs.

    `array` is in numpy (z, y, x) order. `spacing_xyz_mm` follows ITK (x, y, z).
    `direction_3x3` and `origin_xyz_mm` are the affine components needed to
    map voxel ↔ physical and to write derived masks back out in the same frame.
    """
    array: np.ndarray
    spacing_xyz_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float]
    direction_3x3: tuple[float, ...]
    source_path: Path
    study_id: str
    scan_id: str
    orientation_code: str = "RAS"
    is_contrast_enhanced: Optional[bool] = None
    sidecar: dict[str, Any] = field(default_factory=dict)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing_xyz_mm
        return float(sx) * float(sy) * float(sz)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.array.shape)  # type: ignore[return-value]


# --- Airway -----------------------------------------------------------------

@dataclass
class AirwayLandmarks:
    """Anatomical reference points used to slice the airway into regions.

    Each landmark is a `(z, y, x)` numpy index into the parent image. Any
    landmark may be None if it could not be detected; downstream code must
    treat those as optional and emit missing-flag features.
    """
    posterior_nasal_spine: Optional[tuple[int, int, int]] = None
    soft_palate_inferior: Optional[tuple[int, int, int]] = None
    hyoid: Optional[tuple[int, int, int]] = None
    epiglottis_tip: Optional[tuple[int, int, int]] = None
    mandibular_plane_z: Optional[int] = None
    c2_dens: Optional[tuple[int, int, int]] = None
    c3_anterior: Optional[tuple[int, int, int]] = None
    source: str = "unknown"


@dataclass
class AirwayMaskInfo:
    """Binary airway mask + provenance.

    `method` records how the mask was obtained; this is preserved in feature
    output so radiomic/geometric values can be filtered by provenance during
    analysis (e.g. drop fallback masks for primary analysis, keep them for
    sensitivity).
    """
    mask_zyx: np.ndarray  # bool
    method: str           # 'dental_adapter' | 'threshold_connected_component'
                          # | 'external_mask' | 'null'
    confidence: str = "low"   # 'low' | 'medium' | 'high'
    notes: str = ""

    @property
    def is_present(self) -> bool:
        return bool(self.mask_zyx.any())


# --- QC ---------------------------------------------------------------------

@dataclass
class QCResult:
    qc_pass: bool
    qc_warning_count: int
    qc_failure_reasons: list[str]
    qc_coverage_score: float           # 0..1
    qc_artifact_score: Optional[float] # 0..1, None if not computable
    has_upper_airway_region: bool
    has_cervical_soft_tissue: bool
    has_hyoid_region: bool
    has_epiglottis_region: bool
    truncation_flag: bool
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float
    contrast_enhanced: Optional[bool] = None
    extra: dict[str, Any] = field(default_factory=dict)


# --- Feature row ------------------------------------------------------------

@dataclass
class CaseResult:
    """One row of features.csv plus a separate QC row for qc.csv."""
    identifiers: dict[str, Any]
    qc: dict[str, Any]
    airway: dict[str, Any]
    fat: dict[str, Any]
    optional: dict[str, Any] = field(default_factory=dict)
    radiomics: dict[str, Any] = field(default_factory=dict)
    composite: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_feature_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for block in (
            self.identifiers, self.qc, self.airway, self.fat,
            self.optional, self.radiomics, self.composite,
        ):
            row.update(block)
        return row

    def to_qc_row(self) -> dict[str, Any]:
        return {**self.identifiers, **self.qc}
