"""RAIL / RailNet optional experimental adapter.

Source: https://github.com/Tournesol-Saturday/RAIL
HuggingFace: https://huggingface.co/Tournesol-Saturday/railNet-tooth-segmentation-in-CBCT-image

CBCT-domain tooth/jaw/canal segmentation model.
NOT validated on CTA.
All outputs tagged cbct_to_cta_unvalidated.

Expected input format: CBCT NIfTI, typically ~0.3–0.4 mm isotropic.
Domain shift when applied to head/neck CTA (0.5–1.0 mm, contrast-enhanced) is expected.

Installation:
  1. pip install torch torchvision huggingface_hub
  2. Clone https://github.com/Tournesol-Saturday/RAIL
  3. Follow repository install instructions
  4. Set --rail-model-path /path/to/model_weights
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from .base import BaseSegmenter, SegmentationResult

log = get_logger("segmenters.rail")

_INSTALL_MSG = (
    "RAIL dependencies not available. To enable:\n"
    "  pip install torch torchvision huggingface_hub\n"
    "Clone: https://github.com/Tournesol-Saturday/RAIL\n"
    "Set --rail-model-path /path/to/weights"
)

LABELS = {
    "tooth_instances": 1,  # RAIL produces instance-level tooth labels
    "mandible": 2,
    "mandibular_canal": 3,
}

DOMAIN_WARNING = (
    "cbct_to_cta_unvalidated: RAIL/RailNet was trained on CBCT images for tooth instance segmentation. "
    "Application to head/neck CTA involves significant domain shift. "
    "All outputs are highly experimental."
)


class RAILSegmenter(BaseSegmenter):

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path

    @property
    def name(self) -> str:
        return "rail"

    def check_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import huggingface_hub  # noqa: F401
            return True
        except ImportError:
            return False

    def labels(self) -> dict[str, int]:
        return LABELS

    def expected_spacing(self) -> Optional[float]:
        return 0.3  # CBCT domain

    def domain_notes(self) -> str:
        return DOMAIN_WARNING

    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
        config: dict,
    ) -> SegmentationResult:
        domain_warnings = [DOMAIN_WARNING]

        if not self.check_available():
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                domain_warnings=domain_warnings,
                errors=[_INSTALL_MSG],
            )

        log.warning(
            "RAIL inference scaffold invoked. "
            "Clone https://github.com/Tournesol-Saturday/RAIL and integrate "
            "inference pipeline with this wrapper."
        )
        return SegmentationResult(
            success=False,
            label_map=None,
            label_files={},
            labels_json=None,
            domain_warnings=domain_warnings,
            errors=[
                "RAIL inference not yet integrated. "
                "Clone https://github.com/Tournesol-Saturday/RAIL, "
                "follow install instructions, and wire the inference into "
                "RAILSegmenter._run_inference(). "
                "See segmenters/rail.py for integration scaffold."
            ],
        )
