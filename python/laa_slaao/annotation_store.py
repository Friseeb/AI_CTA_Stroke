"""Annotation store: manages correction maps, SLAAO labels, and annotation sessions.

Directory layout per case:
  <store_root>/<case_id>/
    prior_fusion/          <- outputs of prior_fusion.py
    annotation/
      corrected_LAA_mask.nii.gz
      filling_defect_map.nii.gz
      uncertainty_map.nii.gz
      correction_map.nii.gz    <- delta: corrected - consensus_laa
      SLAAO_labels.json
      SLAAO_labels.<rater_id>.json   <- per-rater copies
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from .slaao_schema import SLAAOLabels


class AnnotationStore:
    """Manages per-case annotation artifacts."""

    ANNOTATION_SUBDIR = "annotation"
    PRIOR_FUSION_SUBDIR = "prior_fusion"

    def __init__(self, store_root: Path | str):
        self.root = Path(store_root)

    def case_dir(self, case_id: str) -> Path:
        return self.root / case_id

    def annotation_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / self.ANNOTATION_SUBDIR

    def prior_fusion_dir(self, case_id: str) -> Path:
        return self.case_dir(case_id) / self.PRIOR_FUSION_SUBDIR

    # ------------------------------------------------------------------
    # Correction map
    # ------------------------------------------------------------------

    def save_correction_map(
        self,
        case_id: str,
        corrected_mask: np.ndarray,
        reference_mask: np.ndarray,
        affine: np.ndarray,
    ) -> Path:
        """Save a correction map = corrected_mask - reference_mask (signed int8).

        Values:
          +1 = added by expert (false negative in prior)
          -1 = removed by expert (false positive in prior)
           0 = no change
        """
        ann_dir = self.annotation_dir(case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)

        corrected = (corrected_mask > 0).astype(np.int8)
        reference = (reference_mask > 0).astype(np.int8)
        correction = corrected - reference  # -1, 0, +1

        correction_path = ann_dir / "correction_map.nii.gz"
        nib.save(nib.Nifti1Image(correction, affine), str(correction_path))
        return correction_path

    def save_corrected_laa_mask(
        self, case_id: str, mask: np.ndarray, affine: np.ndarray
    ) -> Path:
        ann_dir = self.annotation_dir(case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)
        path = ann_dir / "corrected_LAA_mask.nii.gz"
        nib.save(nib.Nifti1Image((mask > 0).astype(np.uint8), affine), str(path))
        return path

    def save_filling_defect_map(
        self, case_id: str, defect_map: np.ndarray, affine: np.ndarray
    ) -> Path:
        ann_dir = self.annotation_dir(case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)
        path = ann_dir / "filling_defect_map.nii.gz"
        nib.save(nib.Nifti1Image(defect_map.astype(np.float32), affine), str(path))
        return path

    def save_uncertainty_map(
        self, case_id: str, uncertainty: np.ndarray, affine: np.ndarray
    ) -> Path:
        ann_dir = self.annotation_dir(case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)
        path = ann_dir / "uncertainty_map.nii.gz"
        nib.save(nib.Nifti1Image(uncertainty.astype(np.float32), affine), str(path))
        return path

    # ------------------------------------------------------------------
    # SLAAO labels
    # ------------------------------------------------------------------

    def save_slaao_labels(self, labels: SLAAOLabels, rater_id: Optional[str] = None) -> Path:
        ann_dir = self.annotation_dir(labels.case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)

        if rater_id:
            path = ann_dir / f"SLAAO_labels.{rater_id}.json"
        else:
            path = ann_dir / "SLAAO_labels.json"
        labels.save(path)
        return path

    def load_slaao_labels(
        self, case_id: str, rater_id: Optional[str] = None
    ) -> Optional[SLAAOLabels]:
        ann_dir = self.annotation_dir(case_id)
        if rater_id:
            path = ann_dir / f"SLAAO_labels.{rater_id}.json"
        else:
            path = ann_dir / "SLAAO_labels.json"
        if not path.exists():
            return None
        return SLAAOLabels.load(path)

    def list_rater_labels(self, case_id: str) -> list[str]:
        """Return list of rater IDs that have submitted labels for this case."""
        ann_dir = self.annotation_dir(case_id)
        raters = []
        for p in ann_dir.glob("SLAAO_labels.*.json"):
            rater = p.stem.replace("SLAAO_labels.", "")
            raters.append(rater)
        return sorted(raters)

    # ------------------------------------------------------------------
    # Annotation package (for 3D Slicer / MONAILabel)
    # ------------------------------------------------------------------

    def init_annotation_package(
        self,
        case_id: str,
        ct_path: Path,
        consensus_laa_path: Optional[Path] = None,
        positive_prior_path: Optional[Path] = None,
        negative_prior_path: Optional[Path] = None,
        peri_laa_fat_labels_path: Optional[Path] = None,
        peri_laa_fat_metrics_path: Optional[Path] = None,
    ) -> Path:
        """Prepare a per-case annotation folder ready for 3D Slicer / MONAILabel.

        Copies CT and prior masks into <store_root>/<case_id>/annotation/.
        Writes a blank SLAAO_labels.json template.

        Optionally stages a peri-LAA fat multi-label NIfTI + metrics JSON
        produced by `laa_slaao.peri_laa_fat`. The fat shells live next to the
        priors so 3D Slicer / MONAILabel can show them as additional
        Segmentation nodes during expert correction.

        Returns the annotation directory.
        """
        ann_dir = self.annotation_dir(case_id)
        ann_dir.mkdir(parents=True, exist_ok=True)

        def _copy(src: Optional[Path], dest_name: str):
            if src is not None and src.exists():
                shutil.copy2(src, ann_dir / dest_name)

        _copy(ct_path, f"{case_id}_ct.nii.gz")
        _copy(consensus_laa_path, "consensus_laa.nii.gz")
        _copy(positive_prior_path, "positive_prior.nii.gz")
        _copy(negative_prior_path, "negative_prior.nii.gz")
        _copy(peri_laa_fat_labels_path, "peri_laa_fat_labels.nii.gz")
        _copy(peri_laa_fat_metrics_path, "peri_laa_fat_metrics.json")

        # Write blank SLAAO template if none exists
        slaao_path = ann_dir / "SLAAO_labels.json"
        if not slaao_path.exists():
            blank = SLAAOLabels.blank(case_id=case_id)
            blank.save(slaao_path)

        # Write session metadata
        meta = {
            "case_id": case_id,
            "created": datetime.utcnow().isoformat(),
            "files": [p.name for p in ann_dir.iterdir() if p.is_file()],
            "status": "pending",
            "peri_laa_fat_staged": (
                peri_laa_fat_labels_path is not None
                and Path(peri_laa_fat_labels_path).exists()
            ),
        }
        (ann_dir / "session.json").write_text(json.dumps(meta, indent=2))

        return ann_dir

    # ------------------------------------------------------------------
    # Batch summary
    # ------------------------------------------------------------------

    def list_cases(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def batch_summary(self) -> list[dict]:
        rows = []
        for case_id in self.list_cases():
            ann_dir = self.annotation_dir(case_id)
            slaao = self.load_slaao_labels(case_id)
            correction_exists = (ann_dir / "correction_map.nii.gz").exists()
            corrected_mask_exists = (ann_dir / "corrected_LAA_mask.nii.gz").exists()
            rows.append({
                "case_id": case_id,
                "slaao_complete": slaao.is_complete if slaao else False,
                "any_positive": slaao.any_positive if slaao else None,
                "correction_map": correction_exists,
                "corrected_mask": corrected_mask_exists,
                "raters": self.list_rater_labels(case_id),
            })
        return rows
