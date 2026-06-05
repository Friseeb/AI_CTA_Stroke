#!/usr/bin/env python
"""Analyze wall/lumen HU labels and write strict lumen threshold candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

from aorta_cta_radiomics.fat_wall import _keep_components_touching, _slice_centerline_core_mask
from aorta_cta_radiomics.shells import external_shell


def main() -> None:
    args = _build_parser().parse_args()
    case_id = args.case_id
    outdir = Path(args.outdir)
    mask_dir = outdir / "masks" / case_id
    feature_dir = outdir / "features"
    mask_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    image_itk = sitk.ReadImage(str(args.image))
    image = sitk.GetArrayFromImage(image_itk).astype(float)
    aorta = _read_mask(args.aorta_mask, image.shape)
    lumen = _read_mask(args.lumen_mask, image.shape) if args.lumen_mask else np.zeros_like(aorta)
    wall = _read_mask(args.wall_mask, image.shape) if args.wall_mask else np.zeros_like(aorta)
    fat_0_2 = _read_mask(args.fat_0_2_mask, image.shape) if args.fat_0_2_mask else np.zeros_like(aorta)
    fat_2_5 = _read_mask(args.fat_2_5_mask, image.shape) if args.fat_2_5_mask else np.zeros_like(aorta)

    spacing_xyz = tuple(float(value) for value in image_itk.GetSpacing())
    voxel_volume = float(np.prod(np.asarray(spacing_xyz, dtype=float)))
    core = _slice_centerline_core_mask(
        aorta,
        spacing_xyz,
        centerline_core_radius_mm=float(args.centerline_core_radius_mm),
    )
    if not core.any():
        raise ValueError("Could not estimate a non-empty centerline core from the aorta mask.")

    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    dist_inside = ndi.distance_transform_edt(aorta, sampling=spacing_zyx)
    inner_rim = aorta & (dist_inside <= float(args.inner_rim_mm))
    mid_noncore = aorta & (dist_inside > float(args.inner_rim_mm)) & ~core
    outer_shell = external_shell(
        aorta,
        spacing_xyz,
        inner_mm=0.0,
        outer_mm=float(args.outer_search_mm),
    )
    outside_roi = outer_shell & ~aorta
    seed = aorta & (image < float(args.calcium_hu))
    label_lumen = lumen & (image < float(args.calcium_hu))
    label_wall = wall & (image < float(args.calcium_hu))
    high_conf_lumen = _high_confidence_lumen_from_labels(
        label_lumen,
        label_wall,
        spacing_xyz,
        wall_exclusion_mm=float(args.wall_exclusion_mm),
    )

    summary = pd.DataFrame(
        [
            _summarize_region("label_lumen_all_lt_calcium", image, label_lumen, voxel_volume),
            _summarize_region("label_lumen_high_confidence_lt_calcium", image, high_conf_lumen, voxel_volume),
            _summarize_region("label_wall_lt_calcium", image, label_wall, voxel_volume),
            _summarize_region("center_core_lumen_reference", image, core & (image < args.calcium_hu), voxel_volume),
            _summarize_region("input_aorta_all_lt_calcium", image, seed, voxel_volume),
            _summarize_region("input_aorta_inner_rim_lt_calcium", image, inner_rim & seed, voxel_volume),
            _summarize_region("input_aorta_mid_noncore_lt_calcium", image, mid_noncore & seed, voxel_volume),
            _summarize_region("current_wall_candidate", image, wall, voxel_volume),
            _summarize_region("fat_0_2", image, fat_0_2, voxel_volume),
            _summarize_region("fat_2_5", image, fat_2_5, voxel_volume),
            _summarize_region("outside_search_shell_lt_calcium", image, outside_roi & (image < args.calcium_hu), voxel_volume),
        ]
    )
    summary_path = feature_dir / "hu_range_summary.csv"
    summary.to_csv(summary_path, index=False)

    label_thresholds = _write_label_threshold_masks(
        image=image,
        image_itk=image_itk,
        lumen=label_lumen,
        high_conf_lumen=high_conf_lumen,
        wall=label_wall,
        outside_roi=outside_roi,
        spacing_xyz=spacing_xyz,
        voxel_volume=voxel_volume,
        case_id=case_id,
        mask_dir=mask_dir,
        calcium_hu=float(args.calcium_hu),
    )
    if args.write_legacy_aorta_plus:
        experiments = _write_threshold_experiment_masks(
            image=image,
            image_itk=image_itk,
            aorta=aorta,
            seed=seed,
            wall=wall,
            core=core,
            outside_roi=outside_roi,
            spacing_xyz=spacing_xyz,
            voxel_volume=voxel_volume,
            case_id=case_id,
            mask_dir=mask_dir,
            calcium_hu=float(args.calcium_hu),
        )
        experiments_path = feature_dir / "aorta_plus_threshold_experiments.csv"
        experiments.to_csv(experiments_path, index=False)
    else:
        experiments = pd.DataFrame()
        experiments_path = None
    label_thresholds_path = feature_dir / "lumen_wall_label_hu_thresholds.csv"
    label_thresholds.to_csv(label_thresholds_path, index=False)

    print(f"HU range summary: {summary_path}")
    print(summary[["region", "voxels", "volume_mm3", "hu_p1", "hu_p5", "hu_p25", "hu_p50", "hu_p75", "hu_p95", "hu_p99", "hu_mean", "hu_sd"]].to_string(index=False))
    if experiments_path is not None:
        print(f"\nLegacy aorta-plus threshold experiments: {experiments_path}")
        print(experiments[["method", "lower_hu", "upper_hu", "added_outside_aorta_volume_mm3", "added_hu_min", "added_hu_median", "added_hu_max"]].to_string(index=False))
    print(f"\nLabel-based lumen HU thresholds: {label_thresholds_path}")
    print(label_thresholds[["method", "lower_hu", "upper_hu", "lumen_hu_volume_mm3", "outside_added_volume_mm3", "mask_path"]].to_string(index=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--aorta-mask", required=True, type=Path)
    parser.add_argument("--lumen-mask", type=Path)
    parser.add_argument("--wall-mask", type=Path)
    parser.add_argument("--fat-0-2-mask", type=Path)
    parser.add_argument("--fat-2-5-mask", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--outer-search-mm", type=float, default=1.0)
    parser.add_argument("--inner-rim-mm", type=float, default=2.0)
    parser.add_argument("--centerline-core-radius-mm", type=float, default=2.0)
    parser.add_argument("--wall-exclusion-mm", type=float, default=1.0)
    parser.add_argument("--calcium-hu", type=float, default=500.0)
    parser.add_argument("--write-legacy-aorta-plus", action="store_true")
    return parser


def _read_mask(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch for {path}: {mask.shape} != {expected_shape}")
    return mask


def _summarize_region(
    name: str,
    image: np.ndarray,
    mask: np.ndarray,
    voxel_volume: float,
) -> dict[str, float | int | str]:
    values = image[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    row: dict[str, float | int | str] = {
        "region": name,
        "voxels": int(values.size),
        "volume_mm3": float(values.size * voxel_volume),
    }
    if values.size == 0:
        for quantile in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
            row[f"hu_p{quantile}"] = float("nan")
        row["hu_mean"] = float("nan")
        row["hu_sd"] = float("nan")
        row["hu_ge_500_voxels"] = 0
        return row
    for quantile in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        row[f"hu_p{quantile}"] = float(np.percentile(values, quantile))
    row["hu_mean"] = float(np.mean(values))
    row["hu_sd"] = float(np.std(values))
    row["hu_ge_500_voxels"] = int((values >= 500.0).sum())
    return row


def _write_threshold_experiment_masks(
    image: np.ndarray,
    image_itk: sitk.Image,
    aorta: np.ndarray,
    seed: np.ndarray,
    wall: np.ndarray,
    core: np.ndarray,
    outside_roi: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    voxel_volume: float,
    case_id: str,
    mask_dir: Path,
    calcium_hu: float,
) -> pd.DataFrame:
    core_values = image[core & (image < calcium_hu)]
    if core_values.size == 0:
        raise ValueError("Centerline core has no non-calcium voxels for HU calibration.")
    wall_values = image[wall & (image < calcium_hu)]
    core_mean = float(np.mean(core_values))
    core_sd = float(np.std(core_values))
    core_p25 = float(np.percentile(core_values, 25))
    core_p50 = float(np.percentile(core_values, 50))
    core_p75 = float(np.percentile(core_values, 75))
    wall_p99 = float(np.percentile(wall_values, 99)) if wall_values.size else float("nan")
    methods = [
        ("thr400_500", 400.0, calcium_hu),
        ("core_p25_500", core_p25, calcium_hu),
        ("core_p50_500", core_p50, calcium_hu),
        ("core_p75_500", core_p75, calcium_hu),
        ("core_mean_500", core_mean, calcium_hu),
        ("core_mean_plus_halfsd_500", core_mean + 0.5 * core_sd, calcium_hu),
        ("wall_p99_plus25_500", max(wall_p99 + 25.0, 400.0), calcium_hu),
    ]
    rows: list[dict[str, object]] = []
    for method, lower, upper in methods:
        candidate = outside_roi & (image >= lower) & (image < upper)
        connected = _keep_components_touching(seed | candidate, seed)
        added = connected & ~aorta
        aorta_plus = seed | added
        path = mask_dir / f"{case_id}_aorta_plus_{method}.nii.gz"
        out_img = sitk.GetImageFromArray(aorta_plus.astype(np.uint8))
        out_img.CopyInformation(image_itk)
        sitk.WriteImage(out_img, str(path))
        rows.append(
            {
                "method": method,
                "lower_hu": float(lower),
                "upper_hu": float(upper),
                "aorta_plus_volume_mm3": float(aorta_plus.sum() * voxel_volume),
                "added_outside_aorta_voxels": int(added.sum()),
                "added_outside_aorta_volume_mm3": float(added.sum() * voxel_volume),
                "added_hu_min": float(np.min(image[added])) if added.any() else float("nan"),
                "added_hu_median": float(np.median(image[added])) if added.any() else float("nan"),
                "added_hu_mean": float(np.mean(image[added])) if added.any() else float("nan"),
                "added_hu_max": float(np.max(image[added])) if added.any() else float("nan"),
                "mask_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _high_confidence_lumen_from_labels(
    lumen: np.ndarray,
    wall: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    wall_exclusion_mm: float,
) -> np.ndarray:
    """Use the wall/lumen labels to keep lumen voxels away from the labeled wall."""
    if not lumen.any() or not wall.any() or wall_exclusion_mm <= 0:
        return np.asarray(lumen, dtype=bool)
    sampling_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    distance_to_wall = ndi.distance_transform_edt(~np.asarray(wall, dtype=bool), sampling=sampling_zyx)
    high_confidence = np.asarray(lumen, dtype=bool) & (distance_to_wall > float(wall_exclusion_mm))
    return high_confidence if high_confidence.any() else np.asarray(lumen, dtype=bool)


def _write_label_threshold_masks(
    image: np.ndarray,
    image_itk: sitk.Image,
    lumen: np.ndarray,
    high_conf_lumen: np.ndarray,
    wall: np.ndarray,
    outside_roi: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    voxel_volume: float,
    case_id: str,
    mask_dir: Path,
    calcium_hu: float,
) -> pd.DataFrame:
    lumen_values = image[high_conf_lumen & (image < calcium_hu)]
    wall_values = image[wall & (image < calcium_hu)]
    if lumen_values.size == 0:
        return pd.DataFrame()
    lumen_p25 = float(np.percentile(lumen_values, 25))
    lumen_p50 = float(np.percentile(lumen_values, 50))
    lumen_p75 = float(np.percentile(lumen_values, 75))
    lumen_p90 = float(np.percentile(lumen_values, 90))
    wall_p99 = float(np.percentile(wall_values, 99)) if wall_values.size else float("nan")
    youden = _youden_threshold(wall_values, lumen_values, lower=float(np.percentile(lumen_values, 5)), upper=calcium_hu)
    methods = [
        ("label_lumen_p25_500", lumen_p25, calcium_hu),
        ("label_lumen_p50_500", lumen_p50, calcium_hu),
        ("label_lumen_p75_500", lumen_p75, calcium_hu),
        ("label_lumen_p90_500", lumen_p90, calcium_hu),
        ("label_youden_500", youden, calcium_hu),
        ("label_wall_p99_500", max(wall_p99, lumen_p25), calcium_hu),
    ]
    rows: list[dict[str, object]] = []
    for method, lower, upper in methods:
        candidate_roi = np.asarray(lumen, dtype=bool) | np.asarray(outside_roi, dtype=bool)
        candidate = candidate_roi & (image >= lower) & (image < upper)
        seed = np.asarray(high_conf_lumen, dtype=bool) & candidate
        if seed.any():
            candidate = _keep_components_touching(candidate, seed)
        outside_added = candidate & np.asarray(outside_roi, dtype=bool)
        path = mask_dir / f"{case_id}_lumen_hu_{method}.nii.gz"
        out_img = sitk.GetImageFromArray(candidate.astype(np.uint8))
        out_img.CopyInformation(image_itk)
        sitk.WriteImage(out_img, str(path))
        rows.append(
            {
                "method": method,
                "lower_hu": float(lower),
                "upper_hu": float(upper),
                "lumen_hu_voxels": int(candidate.sum()),
                "lumen_hu_volume_mm3": float(candidate.sum() * voxel_volume),
                "inside_label_lumen_volume_mm3": float((candidate & lumen).sum() * voxel_volume),
                "outside_added_volume_mm3": float(outside_added.sum() * voxel_volume),
                "mask_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _youden_threshold(wall_values: np.ndarray, lumen_values: np.ndarray, lower: float, upper: float) -> float:
    """Find a high-HU cutoff separating labeled lumen from labeled wall."""
    if wall_values.size == 0:
        return float(np.percentile(lumen_values, 50))
    grid = np.linspace(float(lower), float(min(upper, np.percentile(lumen_values, 99))), num=256)
    best_threshold = float(grid[0])
    best_score = -np.inf
    for threshold in grid:
        sensitivity = float((lumen_values >= threshold).mean())
        specificity = float((wall_values < threshold).mean())
        score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


if __name__ == "__main__":
    main()
