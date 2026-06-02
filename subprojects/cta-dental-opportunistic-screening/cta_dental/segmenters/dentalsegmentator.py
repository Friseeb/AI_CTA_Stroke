"""DentalSegmentator wrapper using local nnU-Net v2 weights.

Model: Dataset112_DentalSegmentator_v100
Source: https://github.com/gaudot/SlicerDentalSegmentator
Weights: https://zenodo.org/records/10829675

Label map (5 classes):
  1 — upper skull / maxilla
  2 — mandible
  3 — upper teeth
  4 — lower teeth
  5 — mandibular canal

Requires:
  - nnUNetv2 installed: pip install nnunetv2
  - Weights downloaded from Zenodo and either:
      (a) pointed to via --dentalseg-weights /path/to/Dataset112_DentalSegmentator_v100.zip
      (b) already unpacked into nnUNet_results folder

Usage:
  cta-dental segment --segmenter dentalsegmentator \
    --dentalseg-weights /data/weights/Dataset112_DentalSegmentator_v100.zip
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from ..external_tools import find_nnunet_predict, run_nnunet_predict
from ..logging_utils import get_logger
from .base import BaseSegmenter, SegmentationResult

log = get_logger("segmenters.dentalseg")

DATASET_ID = 112
CONFIGURATION = "3d_fullres"
FOLD = "all"
TRAINER = "nnUNetTrainer"

LABELS = {
    "upper_skull_maxilla": 1,
    "mandible": 2,
    "upper_teeth": 3,
    "lower_teeth": 4,
    "mandibular_canal": 5,
}


class DentalSegmentatorSegmenter(BaseSegmenter):

    def __init__(
        self,
        weights_path: Optional[str] = None,
        nnunet_results_dir: Optional[str] = None,
    ):
        self._weights_path = Path(weights_path) if weights_path else None
        self._nnunet_results_dir = Path(nnunet_results_dir) if nnunet_results_dir else None

    @property
    def name(self) -> str:
        return "dentalsegmentator"

    def check_available(self) -> bool:
        if find_nnunet_predict() is None:
            return False
        if self._nnunet_results_dir and self._nnunet_results_dir.exists():
            return True
        if self._weights_path and self._weights_path.exists():
            return True
        return False

    def labels(self) -> dict[str, int]:
        return LABELS

    def expected_spacing(self) -> Optional[float]:
        return 0.3  # DentalSegmentator trained on high-res CBCT; CTA will be domain-shifted

    def domain_notes(self) -> str:
        return (
            "DentalSegmentator trained on dental CBCT (high-resolution, no contrast). "
            "Application to head/neck CTA involves domain shift. "
            "Results are cbct_to_cta_unvalidated and should be treated as experimental candidates only."
        )

    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
        config: dict,
    ) -> SegmentationResult:
        domain_warnings = [
            "cbct_to_cta_unvalidated: DentalSegmentator was trained on dental CBCT. "
            "CTA domain shift is expected. All outputs are experimental."
        ]

        if not check_available_detailed(self):
            missing = []
            if find_nnunet_predict() is None:
                missing.append("nnUNetv2 not installed (pip install nnunetv2)")
            if not (self._weights_path or self._nnunet_results_dir):
                missing.append(
                    "No weights configured. Download from https://zenodo.org/records/10829675 "
                    "and pass --dentalseg-weights /path/to/Dataset112_DentalSegmentator_v100.zip"
                )
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                domain_warnings=domain_warnings,
                errors=missing,
            )

        results_dir = self._resolve_results_dir(output_dir)
        if results_dir is None:
            return SegmentationResult(
                success=False,
                label_map=None,
                label_files={},
                labels_json=None,
                domain_warnings=domain_warnings,
                errors=["Could not resolve nnUNet results directory from weights path."],
            )

        # nnUNet expects input as a folder with files named <case>_0000.nii.gz
        with tempfile.TemporaryDirectory() as tmp_in:
            tmp_in_path = Path(tmp_in)
            nnunet_input = tmp_in_path / "case_0000.nii.gz"
            shutil.copy2(input_nifti, nnunet_input)

            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_nnunet_predict(
                    input_dir=tmp_in_path,
                    output_dir=output_dir,
                    dataset_id=DATASET_ID,
                    configuration=CONFIGURATION,
                    fold=FOLD,
                    trainer=TRAINER,
                    results_folder=results_dir,
                )
            except RuntimeError as exc:
                return SegmentationResult(
                    success=False,
                    label_map=None,
                    label_files={},
                    labels_json=None,
                    domain_warnings=domain_warnings,
                    errors=[str(exc)],
                )

        label_map, label_files = self._collect_outputs(output_dir)
        labels_json = self._write_labels_json(output_dir, label_files)

        return SegmentationResult(
            success=True,
            label_map=label_map,
            label_files=label_files,
            labels_json=labels_json,
            domain_warnings=domain_warnings,
            meta={"dataset_id": DATASET_ID, "configuration": CONFIGURATION},
        )

    def _resolve_results_dir(self, output_dir: Path) -> Optional[Path]:
        if self._nnunet_results_dir and self._nnunet_results_dir.exists():
            return self._nnunet_results_dir
        if self._weights_path and self._weights_path.exists():
            return _unpack_weights(self._weights_path, output_dir / "_nnunet_weights")
        return None

    def _collect_outputs(self, output_dir: Path) -> tuple[Optional[sitk.Image], dict[str, Path]]:
        label_files: dict[str, Path] = {}
        label_map: Optional[sitk.Image] = None

        pred_files = list(output_dir.glob("*.nii.gz"))
        if not pred_files:
            return None, {}

        # nnUNet produces a single multi-label prediction per case
        seg_path = pred_files[0]
        try:
            label_map = sitk.ReadImage(str(seg_path))
            import numpy as np
            arr = sitk.GetArrayFromImage(label_map)
            for label_name, label_id in LABELS.items():
                mask_arr = (arr == label_id).astype(np.uint8)
                mask_img = sitk.GetImageFromArray(mask_arr)
                mask_img.CopyInformation(label_map)
                mask_path = output_dir / f"{label_name}.nii.gz"
                sitk.WriteImage(mask_img, str(mask_path), useCompression=True)
                label_files[label_name] = mask_path
        except Exception as exc:
            log.error("Failed to split DentalSegmentator labels: %s", exc)

        return label_map, label_files


def check_available_detailed(seg: DentalSegmentatorSegmenter) -> bool:
    return seg.check_available()


def _unpack_weights(zip_path: Path, target_dir: Path) -> Optional[Path]:
    """Unpack Zenodo zip into target_dir if not already done. Returns results dir root."""
    if target_dir.exists() and any(target_dir.iterdir()):
        log.info("Using existing unpacked weights at %s", target_dir)
        return target_dir
    log.info("Unpacking DentalSegmentator weights from %s → %s", zip_path, target_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        return target_dir
    except Exception as exc:
        log.error("Failed to unpack weights: %s", exc)
        return None
