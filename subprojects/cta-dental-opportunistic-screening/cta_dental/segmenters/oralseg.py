"""OralSeg optional experimental adapter.

Source: https://github.com/OttoYouZhou/oralseg
HuggingFace: https://huggingface.co/aiadir/OralSeg

CBCT-domain model. NOT validated on CTA.
All outputs tagged cbct_to_cta_unvalidated.

Installation steps (required before use):
  1. pip install torch torchvision
  2. pip install huggingface_hub
  3. Either:
     - Set --oralseg-model-path /local/path to pre-downloaded weights
     - Or allow auto-download from HuggingFace (requires internet at first run)

This segmenter is NOT the default for CTA workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from .base import BaseSegmenter, SegmentationResult

log = get_logger("segmenters.oralseg")

_INSTALL_MSG = (
    "OralSeg dependencies not available. To enable:\n"
    "  pip install torch torchvision huggingface_hub\n"
    "Then provide --oralseg-model-path or allow HuggingFace auto-download.\n"
    "See: https://github.com/OttoYouZhou/oralseg"
)

LABELS = {
    "maxilla": 1,
    "mandible": 2,
    "upper_teeth": 3,
    "lower_teeth": 4,
    "mandibular_canal": 5,
}

DOMAIN_WARNING = (
    "cbct_to_cta_unvalidated: OralSeg was trained on CBCT images. "
    "Application to head/neck CTA involves significant domain shift. "
    "All outputs are highly experimental and should not inform clinical decisions."
)


class OralSegSegmenter(BaseSegmenter):

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path

    @property
    def name(self) -> str:
        return "oralseg"

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
        return 0.4  # Typical CBCT voxel size; domain-shifted for CTA

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

        try:
            return self._run_inference(input_nifti, output_dir, config, domain_warnings)
        except Exception as exc:
            log.error("OralSeg inference failed: %s", exc)
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                domain_warnings=domain_warnings,
                errors=[
                    f"OralSeg inference error: {exc}\n"
                    "This is an optional experimental backend. "
                    "Verify model weights and dependencies are correctly installed."
                ],
            )

    def _run_inference(
        self,
        input_nifti: Path,
        output_dir: Path,
        config: dict,
        domain_warnings: list[str],
    ) -> SegmentationResult:
        # Scaffold: actual inference requires OralSeg internals.
        # The OralSeg repository does not publish a pip-installable package.
        # Users must clone the repository and adapt the inference script.
        # This scaffold provides the integration point.
        log.warning(
            "OralSeg inference scaffold invoked. "
            "Clone https://github.com/OttoYouZhou/oralseg and integrate "
            "run_inference.py with this wrapper. Returning not-implemented."
        )
        return SegmentationResult(
            success=False,
            label_map=None,
            label_files={},
            labels_json=None,
            domain_warnings=domain_warnings,
            errors=[
                "OralSeg inference not yet integrated. "
                "Clone https://github.com/OttoYouZhou/oralseg, adapt inference script, "
                "and call it from OralSegSegmenter._run_inference(). "
                "See segmenters/oralseg.py for integration scaffold."
            ],
        )
