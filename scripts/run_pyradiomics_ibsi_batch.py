#!/usr/bin/env python3
"""Batch PyRadiomics extraction with explicit IBSI-oriented preprocessing.

Pipeline highlights:
- Match mask geometry to image geometry (nearest-neighbor)
- Binarize masks and optionally keep largest connected component
- Resample image+mask to isotropic voxels before extraction
- Optional CT intensity clipping in HU
- Export one CSV row per (case, region)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


@dataclass
class CaseRegion:
    case_id: str
    image_path: Path
    region: str
    mask_path: Path


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _scan_id_from_image(path: Path) -> str:
    return _strip_nii_suffix(path.name)


def _parse_intensity_range(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    txt = raw.strip()
    if txt == "" or txt.lower() in {"none", "null"}:
        return None
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 2:
        raise ValueError("--intensity-range must be two comma-separated numbers, e.g. -1024,3071")
    lo, hi = float(parts[0]), float(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_float_list(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one numeric value.")
    return [float(p) for p in parts]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch PyRadiomics with IBSI-oriented preprocessing and isotropic voxels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image-dir",
        default="./data/daylightbids",
        help="Directory containing sub-*_acq-CTA_ct.nii.gz images",
    )
    p.add_argument("--image-glob", default="sub-*_acq-CTA_ct.nii.gz", help="Input image glob")
    p.add_argument(
        "--mask-root",
        default="./data/daylightbids/derivatives/nudf_la",
        help="Root containing <case>/<case>_<region>.nii.gz masks",
    )
    p.add_argument(
        "--mask-suffix",
        action="append",
        default=[],
        help=(
            "Mask suffix(es) under each case dir. Repeatable. "
            "Examples: laa_nudf, left_atrium_highres, aorta_highres"
        ),
    )
    p.add_argument(
        "--output-csv",
        default="./data/daylightbids/derivatives/radiomics/pyradiomics_ibsi_batch.csv",
        help="Output CSV path",
    )
    p.add_argument("--subject", action="append", default=[], help="Optional subject IDs (repeatable)")

    p.add_argument("--isotropic-mm", type=float, default=1.0, help="Target isotropic voxel spacing in mm")
    p.add_argument(
        "--image-interpolator",
        choices=["linear", "bspline"],
        default="linear",
        help="Image interpolation used for isotropic resampling",
    )
    p.add_argument(
        "--intensity-range",
        default="-1024,3071",
        help="CT clipping range in HU. Use 'none' to disable clipping.",
    )
    p.add_argument("--bin-width", type=float, default=25.0, help="PyRadiomics discretization bin width")
    p.add_argument("--label", type=int, default=1, help="Label value used for feature extraction")
    p.add_argument(
        "--image-type",
        action="append",
        choices=["original", "wavelet", "log-sigma", "square"],
        default=None,
        help=(
            "Image type(s) to extract. Repeatable. "
            "Example: --image-type wavelet --image-type log-sigma --image-type square"
        ),
    )
    p.add_argument(
        "--log-sigma-mm",
        default="1.0,2.0,3.0,4.0,5.0",
        help="Comma-separated sigma values (mm) for LoG/log-sigma image type.",
    )
    p.add_argument("--min-mask-voxels", type=int, default=100, help="Skip ROIs smaller than this")
    p.add_argument(
        "--keep-largest-component",
        action="store_true",
        default=True,
        help="Keep only largest connected component in ROI mask",
    )
    p.add_argument(
        "--drop-diagnostics",
        action="store_true",
        default=True,
        help="Drop diagnostic_* keys from final CSV",
    )
    p.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed image/mask NIfTIs")
    p.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Output folder for preprocessed files (default: <output-csv-dir>/preprocessed)",
    )
    p.add_argument("--check-env", action="store_true", help="Check required packages and exit")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume into existing output CSV. Only process case/region rows missing selected "
            "image-type feature families, then merge new features into existing rows."
        ),
    )
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar if available")
    p.add_argument(
        "--pyradiomics-log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="ERROR",
        help="PyRadiomics logger verbosity",
    )
    return p.parse_args()


def _check_env() -> None:
    import sys

    details = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "SimpleITK": sitk.Version_VersionString(),
    }
    try:
        import radiomics

        details["pyradiomics"] = radiomics.__version__
    except Exception as exc:  # noqa: BLE001
        details["pyradiomics_error"] = str(exc)

    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _iter_case_regions(
    image_paths: Iterable[Path],
    mask_root: Path,
    mask_suffixes: list[str],
    wanted_subjects: set[str],
) -> Iterable[CaseRegion]:
    for image_path in sorted(image_paths):
        case_id = _scan_id_from_image(image_path)
        sid = case_id.split("_")[0].replace("sub-", "")
        if wanted_subjects and sid.isdigit() and str(int(sid)) not in wanted_subjects:
            continue
        for suffix in mask_suffixes:
            mask_path = mask_root / case_id / f"{case_id}_{suffix}.nii.gz"
            yield CaseRegion(case_id=case_id, image_path=image_path, region=suffix, mask_path=mask_path)


def _is_geometry_match(img: sitk.Image, mask: sitk.Image, tol: float = 1e-5) -> bool:
    if img.GetSize() != mask.GetSize():
        return False
    if not np.allclose(img.GetSpacing(), mask.GetSpacing(), atol=tol):
        return False
    if not np.allclose(img.GetOrigin(), mask.GetOrigin(), atol=tol):
        return False
    if not np.allclose(img.GetDirection(), mask.GetDirection(), atol=tol):
        return False
    return True


def _resample_to_reference(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    return sitk.Cast(rs.Execute(mask), sitk.sitkUInt8)


def _binarize(mask: sitk.Image) -> sitk.Image:
    mask_u8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return mask_u8


def _largest_component(mask_u8: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask_u8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        return sitk.Image(mask_u8.GetSize(), sitk.sitkUInt8)
    largest = max(labels, key=lambda lab: stats.GetNumberOfPixels(lab))
    out = sitk.Cast(cc == largest, sitk.sitkUInt8)
    out.CopyInformation(mask_u8)
    return out


def _clip_hu(image: sitk.Image, clip_range: tuple[float, float] | None) -> sitk.Image:
    img = sitk.Cast(image, sitk.sitkFloat32)
    if clip_range is None:
        return img
    lo, hi = clip_range
    arr = sitk.GetArrayFromImage(img)
    arr = np.clip(arr, lo, hi).astype(np.float32)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def _resample_isotropic(
    image: sitk.Image,
    spacing_mm: float,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    old_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    old_size = np.array(image.GetSize(), dtype=np.int64)
    new_spacing = np.array([spacing_mm, spacing_mm, spacing_mm], dtype=np.float64)
    new_size = np.maximum(1, np.round(old_size * (old_spacing / new_spacing)).astype(np.int64))

    rs = sitk.ResampleImageFilter()
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(interpolator)
    rs.SetOutputSpacing(tuple(float(x) for x in new_spacing))
    rs.SetSize([int(x) for x in new_size])
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetOutputDirection(image.GetDirection())
    rs.SetDefaultPixelValue(float(default_value))
    out = rs.Execute(image)
    return out


def _to_builtin(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [ _to_builtin(x) for x in v ]
    return v


def _feature_prefixes_for_image_types(image_types: list[str]) -> list[str]:
    prefixes: list[str] = []
    if "original" in image_types:
        prefixes.append("original_")
    if "wavelet" in image_types:
        prefixes.append("wavelet-")
    if "log-sigma" in image_types:
        prefixes.append("log-sigma-")
    if "square" in image_types:
        prefixes.append("square_")
    return prefixes


def _value_is_present(v: object) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    if s.lower() in {"nan", "none", "null"}:
        return False
    return True


def _row_has_feature_prefix(row: dict[str, object], prefix: str) -> bool:
    return any(k.startswith(prefix) and _value_is_present(v) for k, v in row.items())


def _row_has_required_prefixes(row: dict[str, object], prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    return all(_row_has_feature_prefix(row, p) for p in prefixes)


def _row_key_from_dict(row: dict[str, object]) -> tuple[str, str] | None:
    case_id = str(row.get("case_id", "")).strip()
    region = str(row.get("region", "")).strip()
    if case_id and region:
        return case_id, region
    return None


def _load_existing_rows(path: Path) -> tuple[list[dict[str, object]], dict[tuple[str, str], int]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], {}
    rows: list[dict[str, object]] = []
    key_to_index: dict[tuple[str, str], int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row_obj: dict[str, object] = dict(row)
            rows.append(row_obj)
            key = _row_key_from_dict(row_obj)
            if key is not None:
                key_to_index[key] = i
    return rows, key_to_index


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    try:
        import radiomics
        from radiomics import featureextractor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyRadiomics is not available in this environment. "
            "Install pyradiomics in the active environment first."
        ) from exc

    log_level = getattr(logging, args.pyradiomics_log_level, logging.ERROR)
    radiomics.setVerbosity(log_level)
    logging.getLogger("radiomics").setLevel(log_level)

    mask_suffixes = args.mask_suffix[:] if args.mask_suffix else ["laa_nudf"]
    wanted_subjects = {str(int(s)) for s in args.subject if str(s).isdigit()}
    clip_range = _parse_intensity_range(args.intensity_range)

    image_dir = Path(args.image_dir)
    mask_root = Path(args.mask_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    preproc_dir = Path(args.preprocessed_dir) if args.preprocessed_dir else output_csv.parent / "preprocessed"
    if args.save_preprocessed:
        preproc_dir.mkdir(parents=True, exist_ok=True)

    img_interp = sitk.sitkLinear if args.image_interpolator == "linear" else sitk.sitkBSpline

    selected_image_types = args.image_type[:] if args.image_type else ["original"]
    log_sigma_values = _parse_float_list(args.log_sigma_mm)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=args.bin_width,
        label=args.label,
        correctMask=True,
        geometryTolerance=1e-5,
        force2D=False,
    )
    extractor.disableAllImageTypes()
    if "original" in selected_image_types:
        extractor.enableImageTypeByName("Original")
    if "wavelet" in selected_image_types:
        extractor.enableImageTypeByName("Wavelet")
    if "log-sigma" in selected_image_types:
        extractor.enableImageTypeByName("LoG", customArgs={"sigma": log_sigma_values})
    if "square" in selected_image_types:
        extractor.enableImageTypeByName("Square")

    extractor.enableFeatureClassByName("firstorder")
    if "original" in selected_image_types:
        extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    existing_rows: list[dict[str, object]] = []
    existing_key_to_index: dict[tuple[str, str], int] = {}
    if args.resume:
        existing_rows, existing_key_to_index = _load_existing_rows(output_csv)
        print(f"Resume mode: loaded {len(existing_rows)} existing rows from {output_csv}")

    required_prefixes = _feature_prefixes_for_image_types(selected_image_types)
    rows: list[dict[str, object]] = []
    image_paths = list(image_dir.glob(args.image_glob))
    work_items = list(_iter_case_regions(image_paths, mask_root, mask_suffixes, wanted_subjects))
    skipped_already_done = 0
    if args.resume and existing_rows:
        filtered_items: list[CaseRegion] = []
        for item in work_items:
            key = (item.case_id, item.region)
            idx = existing_key_to_index.get(key)
            if idx is None:
                filtered_items.append(item)
                continue
            old_row = existing_rows[idx]
            old_status = str(old_row.get("status", "")).lower()
            if old_status == "success" and _row_has_required_prefixes(old_row, required_prefixes):
                skipped_already_done += 1
                continue
            filtered_items.append(item)
        work_items = filtered_items
        print(
            f"Resume mode: {skipped_already_done} rows already had required feature families; "
            f"{len(work_items)} rows need processing."
        )

    work_iter = work_items
    if args.progress:
        try:
            from tqdm import tqdm

            work_iter = tqdm(work_items, total=len(work_items), unit="roi", desc="PyRadiomics")
        except Exception:
            print("tqdm not available; running without progress bar.")

    for item in work_iter:
        key = (item.case_id, item.region)
        existing_idx = existing_key_to_index.get(key)
        existing_row = existing_rows[existing_idx] if existing_idx is not None else None

        row: dict[str, object]
        if args.resume and existing_row is not None:
            row = dict(existing_row)
        else:
            row = {}

        row.update({
            "case_id": item.case_id,
            "subject_id": item.case_id.split("_")[0].replace("sub-", ""),
            "region": item.region,
            "image_path": str(item.image_path),
            "mask_path": str(item.mask_path),
            "status": "pending",
            "error": "",
            "isotropic_mm": args.isotropic_mm,
            "bin_width": args.bin_width,
        })

        if not item.mask_path.exists():
            if args.resume and existing_row is not None and str(existing_row.get("status", "")).lower() == "success":
                continue
            row["status"] = "skip_missing_mask"
            rows.append(row)
            continue

        try:
            image = sitk.ReadImage(str(item.image_path))
            mask = sitk.ReadImage(str(item.mask_path))

            if not _is_geometry_match(image, mask):
                mask = _resample_to_reference(mask, image)

            image = _clip_hu(image, clip_range)
            mask = _binarize(mask)
            if args.keep_largest_component:
                mask = _largest_component(mask)

            image_iso = _resample_isotropic(
                image=image,
                spacing_mm=float(args.isotropic_mm),
                interpolator=img_interp,
                default_value=-1024.0,
            )
            mask_iso = _resample_isotropic(
                image=mask,
                spacing_mm=float(args.isotropic_mm),
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0.0,
            )
            mask_iso = _binarize(mask_iso)
            if args.keep_largest_component:
                mask_iso = _largest_component(mask_iso)

            roi_vox = int(sitk.GetArrayViewFromImage(mask_iso).sum())
            row["roi_voxels_iso"] = roi_vox
            if roi_vox < int(args.min_mask_voxels):
                if args.resume and existing_row is not None and str(existing_row.get("status", "")).lower() == "success":
                    continue
                row["status"] = "skip_small_roi"
                rows.append(row)
                continue

            if args.save_preprocessed:
                base = f"{item.case_id}_{item.region}"
                out_img = preproc_dir / f"{base}_img_iso.nii.gz"
                out_msk = preproc_dir / f"{base}_msk_iso.nii.gz"
                sitk.WriteImage(image_iso, str(out_img), useCompression=True)
                sitk.WriteImage(mask_iso, str(out_msk), useCompression=True)
                row["preprocessed_image_path"] = str(out_img)
                row["preprocessed_mask_path"] = str(out_msk)

            features = extractor.execute(image_iso, mask_iso, label=int(args.label))
            for key, val in features.items():
                if args.drop_diagnostics and key.startswith("diagnostics_"):
                    continue
                row[key] = _to_builtin(val)
            row["status"] = "success"
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001
            if args.resume and existing_row is not None and str(existing_row.get("status", "")).lower() == "success":
                continue
            row["status"] = "failure"
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    final_rows: list[dict[str, object]]
    if args.resume and existing_rows:
        final_rows = list(existing_rows)
        key_to_index = dict(existing_key_to_index)
        for row in rows:
            key = _row_key_from_dict(row)
            if key is None:
                final_rows.append(row)
                continue
            idx = key_to_index.get(key)
            if idx is None:
                key_to_index[key] = len(final_rows)
                final_rows.append(row)
            else:
                final_rows[idx] = row
    else:
        final_rows = rows

    all_fields = set()
    for r in final_rows:
        all_fields.update(r.keys())
    front = [
        "case_id",
        "subject_id",
        "region",
        "status",
        "error",
        "image_path",
        "mask_path",
        "roi_voxels_iso",
        "isotropic_mm",
        "bin_width",
        "preprocessed_image_path",
        "preprocessed_mask_path",
    ]
    feature_fields = sorted(f for f in all_fields if f not in set(front))
    fieldnames = [f for f in front if f in all_fields] + feature_fields

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    ok = sum(1 for r in final_rows if r.get("status") == "success")
    print(
        "Image types extracted: "
        + ", ".join(selected_image_types)
        + (f" | LoG sigma mm: {log_sigma_values}" if "log-sigma" in selected_image_types else "")
    )
    print(f"Saved radiomics batch CSV: {output_csv}")
    print(f"Rows: {len(final_rows)} | success: {ok} | non-success: {len(final_rows) - ok}")
    if args.resume:
        print(f"Resume updates applied: {len(rows)} | skipped already complete: {skipped_already_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
