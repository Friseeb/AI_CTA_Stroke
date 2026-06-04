"""SLAAO multi-label schema.

Each filling-state feature is an independent yes/no boolean.
This is NOT a mutually-exclusive category system — features overlap and coexist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SLAAOLabels:
    """Multi-label representation of LAA filling states."""

    # --- filling-state features ---
    dark_thrombus_component: Optional[bool] = None
    contrast_stagnation: Optional[bool] = None
    rim_pattern: Optional[bool] = None
    whole_laa_involvement: Optional[bool] = None
    regional_pooling: Optional[bool] = None
    distal_tip_involvement: Optional[bool] = None
    mixed_pattern: Optional[bool] = None
    uncertain_artifact: Optional[bool] = None

    # --- metadata ---
    case_id: str = ""
    rater_id: str = ""
    annotation_date: str = ""
    annotation_source: str = "manual"   # manual | monailabel | model
    confidence: Optional[float] = None  # 0–1
    notes: str = ""

    # --- correction map reference ---
    correction_map_path: str = ""       # relative path to correction_map.nii.gz
    uncertainty_map_path: str = ""      # relative path to uncertainty_map.nii.gz
    corrected_laa_mask_path: str = ""   # relative path to corrected LAA mask

    def __post_init__(self):
        if not self.annotation_date:
            self.annotation_date = datetime.utcnow().isoformat()

    @property
    def feature_flags(self) -> dict[str, Optional[bool]]:
        return {
            "dark_thrombus_component": self.dark_thrombus_component,
            "contrast_stagnation": self.contrast_stagnation,
            "rim_pattern": self.rim_pattern,
            "whole_laa_involvement": self.whole_laa_involvement,
            "regional_pooling": self.regional_pooling,
            "distal_tip_involvement": self.distal_tip_involvement,
            "mixed_pattern": self.mixed_pattern,
            "uncertain_artifact": self.uncertain_artifact,
        }

    @property
    def any_positive(self) -> bool:
        return any(v is True for v in self.feature_flags.values())

    @property
    def is_complete(self) -> bool:
        return all(v is not None for v in self.feature_flags.values())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, d: dict) -> "SLAAOLabels":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "SLAAOLabels":
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def blank(cls, case_id: str, rater_id: str = "") -> "SLAAOLabels":
        """Return an unannotated template for a new case."""
        return cls(case_id=case_id, rater_id=rater_id)

    def validate(self) -> list[str]:
        """Return list of validation warnings (empty = valid)."""
        warnings = []
        if not self.case_id:
            warnings.append("case_id is empty")
        if not self.is_complete:
            missing = [k for k, v in self.feature_flags.items() if v is None]
            warnings.append(f"Unannotated features: {missing}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            warnings.append(f"confidence {self.confidence} out of range [0, 1]")
        return warnings
