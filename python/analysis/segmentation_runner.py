"""Utilities to run vessel segmentation methods in batch with QC outputs.

This module provides a thin orchestration layer around a callable that performs
segmentation. It handles I/O, QC metric computation, and summary logging.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import nibabel as nib

from .qc_metrics import QCResult, QCThresholds, compute_qc_metrics

SegmentationFn = Callable[[nib.Nifti1Image], "SegmentationOutputs"]


@dataclass
class SegmentationOutputs:
    """Container for a single segmentation run."""

    mask: nib.Nifti1Image
    centerline: Optional[nib.Nifti1Image] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class RunResult:
    """Paths and QC summary for a processed case."""

    case_id: str
    method_name: str
    mask_path: Path
    centerline_path: Optional[Path]
    qc: QCResult
    metadata: Optional[Dict[str, str]] = None

    def to_summary_row(self) -> Dict[str, str]:
        row: Dict[str, str] = {
            "case_id": self.case_id,
            "method": self.method_name,
            "mask_path": str(self.mask_path),
            "centerline_path": str(self.centerline_path) if self.centerline_path else "",
        }
        row.update(self.qc.to_flat_dict())
        if self.metadata:
            for key, value in self.metadata.items():
                row[f"meta_{key}"] = str(value)
        return row


class VesselSegmentationRunner:
    """Run a segmentation callable on one or many CTA volumes."""

    def __init__(self, output_root: Path, thresholds: Optional[QCThresholds] = None):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or QCThresholds()

    def run_case(
        self,
        case_id: str,
        image_path: Path,
        method_name: str,
        segmentation_fn: SegmentationFn,
        save_centerline: bool = True,
    ) -> RunResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing input CTA: {image_path}")

        img = nib.load(str(image_path))
        outputs = segmentation_fn(img)

        case_dir = self.output_root / case_id / method_name
        case_dir.mkdir(parents=True, exist_ok=True)

        mask_path = case_dir / "mask.nii.gz"
        nib.save(outputs.mask, mask_path)

        centerline_path: Optional[Path] = None
        if save_centerline and outputs.centerline is not None:
            centerline_path = case_dir / "centerline.nii.gz"
            nib.save(outputs.centerline, centerline_path)

        qc = compute_qc_metrics(outputs.mask, outputs.centerline, thresholds=self.thresholds)
        qc_path = case_dir / "qc.json"
        qc_path.write_text(json.dumps(qc.to_dict(), indent=2))

        return RunResult(
            case_id=case_id,
            method_name=method_name,
            mask_path=mask_path,
            centerline_path=centerline_path,
            qc=qc,
            metadata=outputs.metadata,
        )

    def run_manifest(
        self,
        manifest_csv: Path,
        method_name: str,
        segmentation_fn: SegmentationFn,
        id_field: str = "case_id",
        path_field: str = "nifti_path",
        limit: Optional[int] = None,
    ) -> Path:
        """Process a CSV manifest. Returns the path to the summary CSV."""
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_csv}")

        summary_rows = []
        with manifest_csv.open() as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    break
                case_id = str(row[id_field])
                image_path = Path(row[path_field])
                result = self.run_case(case_id, image_path, method_name, segmentation_fn)
                summary_rows.append(result.to_summary_row())

        summary_path = self.output_root / f"qc_summary_{method_name}.csv"
        with summary_path.open("w", newline="") as f:
            fieldnames = list(summary_rows[0].keys()) if summary_rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_path

    def dry_run_manifest(
        self,
        manifest_csv: Path,
        id_field: str = "case_id",
        path_field: str = "nifti_path",
    ) -> Dict[str, int]:
        """Validate manifest paths without running segmentation."""
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_csv}")

        total = 0
        missing = 0
        with manifest_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if not Path(row[path_field]).exists():
                    missing += 1
        return {"total": total, "missing": missing}


def load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def save_nifti(img: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
