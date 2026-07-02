#!/usr/bin/env python
"""Measure mask-derived aortic wall thickness for one case."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aorta_cta_radiomics.io import read_mask, read_volume, write_label_like, write_mask_like
from aorta_cta_radiomics.crop import crop_region_for_mask
from aorta_cta_radiomics.wall_thickness import (
    measure_wall_thickness,
    thickness_threshold_summary,
    wall_thickness_threshold_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Reference CTA NIfTI path.")
    parser.add_argument("--lumen-mask", required=True, help="Contrast lumen/aorta mask NIfTI.")
    parser.add_argument("--wall-mask", required=True, help="Aortic wall mask NIfTI.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--calcium-mask", help="Optional calcium mask to move from lumen into wall.")
    parser.add_argument("--subtract-calcium-from-lumen", action="store_true")
    parser.add_argument("--add-calcium-to-wall", action="store_true")
    parser.add_argument(
        "--crop-margin-mm",
        type=float,
        default=0.0,
        help="Crop around lumen/wall/calcium before distance transforms, then paste outputs back.",
    )
    parser.add_argument("--risk-thickness-threshold-mm", type=float, default=4.0)
    args = parser.parse_args()

    image = read_volume(args.image)
    lumen = read_mask(args.lumen_mask).array.astype(bool)
    wall = read_mask(args.wall_mask).array.astype(bool)
    if lumen.shape != image.array.shape or wall.shape != image.array.shape:
        raise ValueError("Image, lumen mask, and wall mask must have the same shape.")

    calcium = None
    calcium_in_lumen = np.zeros(lumen.shape, dtype=bool)
    if args.calcium_mask:
        calcium = read_mask(args.calcium_mask).array.astype(bool)
        if calcium.shape != image.array.shape:
            raise ValueError("Calcium mask must have the same shape as the reference image.")
        calcium_in_lumen = calcium & lumen
        if args.subtract_calcium_from_lumen:
            lumen = lumen & ~calcium
        if args.add_calcium_to_wall:
            wall = wall | calcium
    wall = wall & ~lumen
    full_shape = lumen.shape
    crop_region = None
    if args.crop_margin_mm > 0:
        crop_seed = lumen | wall
        if calcium is not None:
            crop_seed |= calcium
        crop_region = crop_region_for_mask(crop_seed, image.spacing_xyz, margin_mm=float(args.crop_margin_mm))
        lumen = crop_region.crop(lumen)
        wall = crop_region.crop(wall)
        if calcium is not None:
            calcium = crop_region.crop(calcium)
        calcium_in_lumen = crop_region.crop(calcium_in_lumen)

    outdir = Path(args.outdir)
    masks_dir = outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    result = measure_wall_thickness(
        lumen_mask=lumen,
        wall_mask=wall,
        spacing_xyz=image.spacing_xyz,
        case_id=args.case_id,
    )
    if crop_region is not None:
        result = _paste_result_to_full_shape(result, crop_region, full_shape)
        lumen = crop_region.paste(lumen)
        wall = crop_region.paste(wall)
        calcium_in_lumen = crop_region.paste(calcium_in_lumen)
        if calcium is not None:
            calcium = crop_region.paste(calcium)

    write_mask_like(result.lumen_mask, image.image, masks_dir / f"{args.case_id}_lumen_for_wall_thickness.nii.gz")
    write_mask_like(result.wall_mask, image.image, masks_dir / f"{args.case_id}_wall_for_thickness.nii.gz")
    write_mask_like(result.inner_surface_mask, image.image, masks_dir / f"{args.case_id}_wall_inner_surface.nii.gz")
    write_mask_like(result.outer_surface_mask, image.image, masks_dir / f"{args.case_id}_wall_outer_surface.nii.gz")
    write_label_like(result.thickness_bin_labelmap, image.image, masks_dir / f"{args.case_id}_wall_thickness_bins_labels.nii.gz")
    _write_scalar_like(result.thickness_map_mm, image.image, masks_dir / f"{args.case_id}_wall_thickness_mm.nii.gz")
    _write_scalar_like(
        result.outer_surface_thickness_map_mm,
        image.image,
        masks_dir / f"{args.case_id}_wall_outer_surface_thickness_mm.nii.gz",
    )
    risk_mask = wall_thickness_threshold_mask(
        result.thickness_map_mm,
        result.wall_mask,
        threshold_mm=args.risk_thickness_threshold_mm,
        inclusive=False,
    )
    risk_suffix = _threshold_suffix(args.risk_thickness_threshold_mm)
    write_mask_like(
        risk_mask,
        image.image,
        masks_dir / f"{args.case_id}_wall_thickness_gt_{risk_suffix}mm_TEE_analogue.nii.gz",
    )
    write_label_like(
        risk_mask.astype(np.uint16),
        image.image,
        masks_dir / f"{args.case_id}_wall_thickness_gt_{risk_suffix}mm_TEE_analogue_labels.nii.gz",
    )
    if calcium is not None:
        write_mask_like(calcium, image.image, masks_dir / f"{args.case_id}_calcium_input.nii.gz")
        write_mask_like(calcium_in_lumen, image.image, masks_dir / f"{args.case_id}_calcium_removed_from_lumen.nii.gz")

    risk_summary = thickness_threshold_summary(
        case_id=args.case_id,
        threshold_mask=risk_mask,
        wall_mask=result.wall_mask,
        spacing_xyz=image.spacing_xyz,
        threshold_mm=args.risk_thickness_threshold_mm,
    )
    result.summary.to_csv(features_dir / "wall_thickness_summary.csv", index=False)
    risk_summary.to_csv(features_dir / f"wall_thickness_gt_{risk_suffix}mm_TEE_analogue_summary.csv", index=False)
    pd.concat([result.summary, risk_summary], ignore_index=True).to_csv(
        features_dir / "wall_thickness_summary_with_thresholds.csv",
        index=False,
    )
    _write_run_notes(
        features_dir / "wall_thickness_notes.txt",
        args=args,
        calcium_in_lumen_voxels=int(calcium_in_lumen.sum()),
        wall_voxels=int(result.wall_mask.sum()),
    )
    print(pd.concat([result.summary, risk_summary], ignore_index=True).to_string(index=False))
    print(f"Saved wall thickness outputs to {outdir.resolve()}")


def _write_scalar_like(array: np.ndarray, reference_image: object, output_path: str | Path) -> Path:
    import SimpleITK as sitk

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(path))
    return path


def _paste_result_to_full_shape(result, crop_region, full_shape: tuple[int, int, int]):
    from aorta_cta_radiomics.wall_thickness import WallThicknessResult

    if crop_region.full_shape != full_shape:
        raise ValueError("Crop region shape does not match requested full shape.")
    return WallThicknessResult(
        lumen_mask=crop_region.paste(result.lumen_mask),
        wall_mask=crop_region.paste(result.wall_mask),
        inner_surface_mask=crop_region.paste(result.inner_surface_mask),
        outer_surface_mask=crop_region.paste(result.outer_surface_mask),
        thickness_map_mm=crop_region.paste(result.thickness_map_mm, fill_value=0),
        inner_surface_thickness_map_mm=crop_region.paste(result.inner_surface_thickness_map_mm, fill_value=0),
        outer_surface_thickness_map_mm=crop_region.paste(result.outer_surface_thickness_map_mm, fill_value=0),
        thickness_bin_labelmap=crop_region.paste(result.thickness_bin_labelmap, fill_value=0),
        summary=result.summary,
    )


def _write_run_notes(path: Path, args: argparse.Namespace, calcium_in_lumen_voxels: int, wall_voxels: int) -> None:
    path.write_text(
        "\n".join(
            [
                "Mask-derived aortic wall thickness.",
                "This is a geometry measurement from binary masks, not histologic wall thickness.",
                f"case_id: {args.case_id}",
                f"image: {args.image}",
                f"lumen_mask: {args.lumen_mask}",
                f"wall_mask: {args.wall_mask}",
                f"calcium_mask: {args.calcium_mask or ''}",
                f"subtract_calcium_from_lumen: {args.subtract_calcium_from_lumen}",
                f"add_calcium_to_wall: {args.add_calcium_to_wall}",
                f"risk_thickness_threshold_mm: > {args.risk_thickness_threshold_mm:g}",
                f"calcium_in_lumen_voxels: {calcium_in_lumen_voxels}",
                f"wall_voxels_for_thickness: {wall_voxels}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _threshold_suffix(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


if __name__ == "__main__":
    main()
