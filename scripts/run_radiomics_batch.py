#!/usr/bin/env python3
"""
Batch PyRadiomics extraction for CTA segments.

Manifest CSV must include at least:
  - case_id
  - cta_path
  - aorta_mask
  - la_mask
  - laa_mask

Outputs:
  - one CSV with one row per (case_id, segment)
  - one JSON sidecar describing preprocessing/feature settings

Notes:
  - `--ibsi-preset` configures an IBSI-like extraction profile:
      Original image type + standard first/shape/texture classes,
      isotropic resampling, fixed bin width, no intensity normalization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch PyRadiomics for CTA segments")
    p.add_argument("--manifest", required=True, help="CSV manifest")
    p.add_argument("--output-csv", required=True, help="Output CSV path")

    p.add_argument("--id-field", default="case_id", help="Case ID column")
    p.add_argument("--cta-field", default="cta_path", help="CTA NIfTI column")
    p.add_argument("--aorta-field", default="aorta_mask", help="Aorta mask column")
    p.add_argument("--la-field", default="la_mask", help="Left atrium mask column")
    p.add_argument("--laa-field", default="laa_mask", help="LAA mask column")

    p.add_argument("--binwidth", type=float, default=25.0, help="HU bin width")
    p.add_argument("--spacing", default="1.0,1.0,1.0", help="Resample spacing (comma-separated)")
    p.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.0,
        help="LoG sigma value for LoG image type (0 disables extra LoG sigma setting).",
    )
    p.add_argument(
        "--resegment-range",
        default="",
        help="Optional HU range for voxel resegmentation within ROI, e.g. -200,1200",
    )
    p.add_argument(
        "--remove-outliers",
        type=float,
        default=None,
        help="Optional outlier removal in SD units (PyRadiomics setting).",
    )
    p.add_argument(
        "--interpolator",
        default="sitkBSpline",
        help="SimpleITK interpolator name for resampling (default: sitkBSpline).",
    )
    p.add_argument(
        "--correct-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable PyRadiomics mask correction.",
    )
    p.add_argument(
        "--pre-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable pre-cropping around ROI before feature extraction.",
    )
    p.add_argument(
        "--geometry-tolerance",
        type=float,
        default=None,
        help="Optional geometry tolerance for mask/image mismatch handling.",
    )
    p.add_argument(
        "--min-voxels",
        type=int,
        default=20,
        help="Skip ROIs with fewer than this many non-zero voxels.",
    )
    p.add_argument("--disable-diagnostics", action="store_true", help="Drop diagnostic_* fields")
    p.add_argument("--limit", type=int, default=None, help="Limit rows")
    p.add_argument("--progress", action="store_true", help="Show progress bar")
    p.add_argument(
        "--ibsi-preset",
        action="store_true",
        help="Use IBSI-like profile (Original image only + standard radiomics classes).",
    )
    p.add_argument(
        "--image-types",
        default="all",
        help="Comma-separated image types to enable (default: all). Example: Original",
    )
    p.add_argument(
        "--feature-classes",
        default="all",
        help="Comma-separated feature classes to enable (default: all). Example: firstorder,shape,glcm",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast preset: Original image only + firstorder,shape,glcm,glrlm,glszm,gldm,ngtdm",
    )
    p.add_argument(
        "--settings-json",
        default=None,
        help="Optional preprocessing/settings JSON output (default: output-csv with .settings.json suffix).",
    )
    return p.parse_args()


def _parse_spacing(spacing: str) -> List[float]:
    parts = [s.strip() for s in spacing.split(",")]
    if len(parts) != 3:
        raise ValueError("spacing must be three comma-separated values")
    return [float(x) for x in parts]


def _parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_resegment_range(value: str) -> List[float] | None:
    value = value.strip()
    if not value:
        return None
    parts = _parse_list(value)
    if len(parts) != 2:
        raise ValueError("resegment-range must be 'min,max'")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise ValueError("resegment-range requires min < max")
    return [lo, hi]


def _resolve_modes(
    image_types: str,
    feature_classes: str,
    fast: bool,
    ibsi_preset: bool,
) -> Tuple[str, str]:
    if ibsi_preset:
        return "Original", "firstorder,shape,glcm,glrlm,glszm,gldm,ngtdm"
    if fast:
        return "Original", "firstorder,shape,glcm,glrlm,glszm,gldm,ngtdm"
    return image_types, feature_classes


def _build_extractor(
    binwidth: float,
    spacing: List[float],
    smoothing_sigma: float,
    resegment_range: List[float] | None,
    remove_outliers: float | None,
    interpolator: str,
    correct_mask: bool,
    pre_crop: bool,
    geometry_tolerance: float | None,
    image_types: str,
    feature_classes: str,
    fast: bool,
    ibsi_preset: bool,
):
    from radiomics import featureextractor

    image_types, feature_classes = _resolve_modes(image_types, feature_classes, fast, ibsi_preset)

    # Keep IBSI-style profile deterministic: no extra LoG sigma override.
    if ibsi_preset and smoothing_sigma > 0:
        print("IBSI preset active: ignoring --smoothing-sigma to keep Original-only extraction.")
        smoothing_sigma = 0.0

    settings = {
        "binWidth": float(binwidth),
        "resampledPixelSpacing": spacing,
        "interpolator": interpolator,
        "normalize": False,
        "correctMask": bool(correct_mask),
        "preCrop": bool(pre_crop),
        "verbose": False,
    }
    if geometry_tolerance is not None:
        settings["geometryTolerance"] = float(geometry_tolerance)
    if resegment_range is not None:
        settings["resegmentRange"] = [float(resegment_range[0]), float(resegment_range[1])]
    if remove_outliers is not None:
        settings["removeOutliers"] = float(remove_outliers)
    if smoothing_sigma > 0:
        settings["sigma"] = [float(smoothing_sigma)]
        settings["enableCExtensions"] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    if feature_classes.lower() == "all":
        extractor.enableAllFeatures()
        selected_feature_classes = ["all"]
    else:
        extractor.disableAllFeatures()
        selected_feature_classes = []
        for fc in _parse_list(feature_classes):
            extractor.enableFeatureClassByName(fc)
            selected_feature_classes.append(fc)

    if image_types.lower() == "all":
        extractor.enableAllImageTypes()
        selected_image_types = ["all"]
    else:
        # Disable all image types, then enable selected ones
        extractor.disableAllImageTypes()
        selected_image_types = []
        for itype in _parse_list(image_types):
            extractor.enableImageTypeByName(itype)
            selected_image_types.append(itype)

    extraction_plan = {
        "binWidth": float(binwidth),
        "resampledPixelSpacing": spacing,
        "interpolator": interpolator,
        "normalize": False,
        "correctMask": bool(correct_mask),
        "preCrop": bool(pre_crop),
        "geometryTolerance": geometry_tolerance,
        "resegmentRange": resegment_range,
        "removeOutliers": remove_outliers,
        "smoothingSigma": float(smoothing_sigma),
        "ibsiPreset": bool(ibsi_preset),
        "selectedImageTypes": selected_image_types,
        "selectedFeatureClasses": selected_feature_classes,
    }
    return extractor, extraction_plan


def _run_case(
    extractor,
    case_id: str,
    cta_path: Path,
    segment_name: str,
    mask_path: Path,
    disable_diagnostics: bool,
    min_voxels: int,
) -> Dict[str, str] | None:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.ReadImage(str(cta_path))
    mask_image = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_image)
    if mask_arr is None:
        print(f"⚠ Skipping unreadable mask: {case_id} / {segment_name} -> {mask_path}")
        return None

    # Robust ROI handling: treat any non-zero voxel as label=1.
    mask_bin = (mask_arr > 0).astype(np.uint8)
    voxel_count = int(np.sum(mask_bin))
    if voxel_count == 0:
        print(f"⚠ Skipping empty mask: {case_id} / {segment_name} -> {mask_path}")
        return None
    if voxel_count < min_voxels:
        print(
            f"⚠ Skipping tiny mask: {case_id} / {segment_name} "
            f"({voxel_count} voxels < {min_voxels}) -> {mask_path}"
        )
        return None

    if not np.array_equal(mask_arr, mask_bin):
        mask = sitk.GetImageFromArray(mask_bin)
        mask.CopyInformation(mask_image)
    else:
        mask = mask_image

    result = extractor.execute(image, mask, label=1)

    row: Dict[str, str] = {
        "case_id": case_id,
        "segment": segment_name,
        "cta_path": str(cta_path),
        "mask_path": str(mask_path),
        "roi_voxels": str(voxel_count),
    }
    for k, v in result.items():
        if disable_diagnostics and k.startswith("diagnostics_"):
            continue
        row[k] = str(v)
    return row


def main() -> int:
    args = _parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    spacing = _parse_spacing(args.spacing)
    resegment_range = _parse_resegment_range(args.resegment_range)
    extractor, extraction_plan = _build_extractor(
        args.binwidth,
        spacing,
        args.smoothing_sigma,
        resegment_range,
        args.remove_outliers,
        args.interpolator,
        args.correct_mask,
        args.pre_crop,
        args.geometry_tolerance,
        args.image_types,
        args.feature_classes,
        args.fast,
        args.ibsi_preset,
    )

    tasks = []
    with manifest.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            case_id = row.get(args.id_field, "").strip()
            cta_path = Path(row.get(args.cta_field, "").strip())
            if not case_id or not cta_path.exists():
                continue

            segments = [
                ("aorta", row.get(args.aorta_field, "").strip()),
                ("la", row.get(args.la_field, "").strip()),
                ("laa", row.get(args.laa_field, "").strip()),
            ]
            for seg_name, mask_path_str in segments:
                if not mask_path_str:
                    continue
                mask_path = Path(mask_path_str)
                if not mask_path.exists():
                    continue
                tasks.append((case_id, cta_path, seg_name, mask_path))

    if not tasks:
        print("No valid tasks found; check manifest paths.")
        return 1

    # Optional progress bar (tqdm if available)
    iterator = tasks
    if args.progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(tasks, total=len(tasks), desc="Radiomics", unit="seg")
        except Exception:
            print("tqdm not available; falling back to basic progress.")

    rows: List[Dict[str, str]] = []
    for i, (case_id, cta_path, seg_name, mask_path) in enumerate(iterator, start=1):
        result = _run_case(
            extractor=extractor,
            case_id=case_id,
            cta_path=cta_path,
            segment_name=seg_name,
            mask_path=mask_path,
            disable_diagnostics=args.disable_diagnostics,
            min_voxels=args.min_voxels,
        )
        if result is not None:
            rows.append(result)
        if args.progress and "tqdm" not in str(type(iterator)) and i % 25 == 0:
            print(f"Processed {i}/{len(tasks)} segmentations...")

    if not rows:
        print("No rows produced; check manifest paths.")
        return 1

    # Write CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    settings_path = (
        Path(args.settings_json)
        if args.settings_json
        else output_csv.with_suffix(".settings.json")
    )
    settings_doc = {
        "manifest": str(manifest),
        "output_csv": str(output_csv),
        "task_count": len(tasks),
        "row_count": len(rows),
        "min_voxels": int(args.min_voxels),
        "disable_diagnostics": bool(args.disable_diagnostics),
        "extraction_plan": extraction_plan,
    }
    settings_path.write_text(json.dumps(settings_doc, indent=2))

    print(f"Wrote {len(rows)} rows -> {output_csv}")
    print(f"Wrote settings -> {settings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
