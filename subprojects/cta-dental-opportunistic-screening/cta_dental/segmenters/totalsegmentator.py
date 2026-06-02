"""TotalSegmentator teeth/craniofacial segmentation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from ..external_tools import find_totalsegmentator, run_totalsegmentator
from ..logging_utils import get_logger
from .base import BaseSegmenter, SegmentationResult

log = get_logger("segmenters.totalseg")

# Known label file names produced by TotalSegmentator teeth task.
# TotalSegmentator may produce individual binary NIfTIs per label,
# or a combined multi-label file. We handle both.
_TEETH_LABELS = {
    "teeth_upper": 1,
    "teeth_lower": 2,
    "jawbone_upper": 3,
    "jawbone_lower": 4,
    "tooth_canal": 5,
    "implant": 6,
    "crown": 7,
    "bridge": 8,
    "root_canal": 9,
    "maxillary_sinus": 10,
    "mandibular_canal": 11,
}

_CRANIOFACIAL_DENTAL_LABELS = {
    "skull": 1,
    "mandible": 2,
    "teeth_upper": 3,
    "teeth_lower": 4,
    "maxillary_sinus_left": 5,
    "maxillary_sinus_right": 6,
}


class TotalSegmentatorTeethSegmenter(BaseSegmenter):

    def __init__(self, task: str = "teeth"):
        self._task = task

    @property
    def name(self) -> str:
        return f"totalsegmentator_{self._task}"

    def check_available(self) -> bool:
        return find_totalsegmentator() is not None

    def labels(self) -> dict[str, int]:
        return _TEETH_LABELS if self._task == "teeth" else _CRANIOFACIAL_DENTAL_LABELS

    def expected_spacing(self) -> Optional[float]:
        return 1.5  # TotalSegmentator default; resampling is handled internally

    def domain_notes(self) -> str:
        return (
            "TotalSegmentator teeth task trained on CT data. "
            "Head/neck CTA is broadly compatible. "
            "Contrast enhancement may affect soft-tissue labels slightly."
        )

    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
        config: dict,
    ) -> SegmentationResult:
        if not self.check_available():
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                errors=[
                    "TotalSegmentator not found. Install it with: pip install TotalSegmentator\n"
                    "or choose --roi-method threshold_fallback for degraded ROI only."
                ],
            )

        task_cfg = config.get("totalsegmentator", {})
        try:
            run_totalsegmentator(
                input_nifti=input_nifti,
                output_dir=output_dir,
                task=self._task,
                fast=task_cfg.get("fast", False),
                device=task_cfg.get("device", "cpu"),
                weights_dir=task_cfg.get("weights_dir"),
            )
        except (RuntimeError, Exception) as exc:
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                errors=[str(exc)],
            )

        label_files = self._collect_outputs(output_dir)
        label_map = self._try_load_multilabel(output_dir)
        labels_json = self._write_labels_json(output_dir, label_files)

        log.info("TotalSegmentator %s produced %d label files.", self._task, len(label_files))
        return SegmentationResult(
            success=True,
            label_map=label_map,
            label_files=label_files,
            labels_json=labels_json,
            meta={"task": self._task},
        )

    def _collect_outputs(self, output_dir: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for f in sorted(output_dir.glob("*.nii.gz")):
            stem = f.name.replace(".nii.gz", "")
            files[stem] = f
        return files

    def _try_load_multilabel(self, output_dir: Path) -> Optional[sitk.Image]:
        for candidate in ["segmentation.nii.gz", "combined.nii.gz"]:
            p = output_dir / candidate
            if p.exists():
                try:
                    return sitk.ReadImage(str(p))
                except Exception:
                    pass
        return None
