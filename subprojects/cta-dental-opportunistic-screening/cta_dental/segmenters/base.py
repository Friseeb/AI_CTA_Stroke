"""Abstract base segmenter interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import SimpleITK as sitk


@dataclass
class SegmentationResult:
    success: bool
    label_map: Optional[sitk.Image]         # single multi-label NIfTI if available
    label_files: dict[str, Path]            # label_name -> NIfTI path (binary masks)
    labels_json: Optional[Path]             # JSON label manifest
    domain_warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "label_files": {k: str(v) for k, v in self.label_files.items()},
            "labels_json": str(self.labels_json) if self.labels_json else None,
            "domain_warnings": self.domain_warnings,
            "errors": self.errors,
            "meta": self.meta,
        }


class BaseSegmenter(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check_available(self) -> bool:
        """Return True if this segmenter's external dependencies are installed."""
        ...

    @abstractmethod
    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
        config: dict,
    ) -> SegmentationResult: ...

    @abstractmethod
    def labels(self) -> dict[str, int]:
        """Return label_name -> integer_id mapping."""
        ...

    def expected_spacing(self) -> Optional[float]:
        """Preferred isotropic spacing in mm for this model, or None."""
        return None

    def domain_notes(self) -> str:
        return ""

    def _write_labels_json(self, output_dir: Path, label_files: dict[str, Path]) -> Path:
        manifest = {
            "segmenter": self.name,
            "domain_notes": self.domain_notes(),
            "labels": {k: str(v) for k, v in label_files.items()},
            "label_ids": self.labels(),
        }
        path = output_dir / "labels.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path
