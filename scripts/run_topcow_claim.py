#!/usr/bin/env python3
"""
TopCoW 2024 (CLAIM) inference wrapper.

Based on CLAIM's public winning solution:
https://github.com/claim-berlin/TopCoW_2024_MRA_winning_solution
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib


TOPCOW_LABELS = {
    0: "background",
    1: "BA",
    2: "R-PCA",
    3: "L-PCA",
    4: "R-ICA",
    5: "R-MCA",
    6: "L-ICA",
    7: "L-MCA",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-ACA",
    12: "L-ACA",
    15: "3rd-A2",
}


def _require_deps(require_yolo: bool = True) -> None:
    missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    if require_yolo:
        try:
            import cv2  # noqa: F401
        except Exception:
            missing.append("opencv-python")
        try:
            import ultralytics  # noqa: F401
        except Exception:
            missing.append("ultralytics")
    try:
        import nnunetv2  # noqa: F401
    except Exception:
        missing.append("nnunetv2")
    try:
        import SimpleITK  # noqa: F401
    except Exception:
        missing.append("SimpleITK")
    try:
        import batchgenerators  # noqa: F401
    except Exception:
        missing.append("batchgenerators")

    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install per CLAIM TopCoW README."
        )


def _load_nifti(nifti_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj)
    return data, img


def _save_nifti(data: np.ndarray, output_path: Path, reference_path: Path) -> None:
    ref = nib.load(str(reference_path))
    out = nib.Nifti1Image(data, ref.affine, ref.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(output_path))


def _window_cta(volume: np.ndarray, wl: float = 300, ww: float = 1000) -> np.ndarray:
    lower = wl - (ww / 2)
    upper = wl + (ww / 2)
    return np.clip(volume, lower, upper)


def _process_nifti_full_volume(
    nifti_path: Path,
    output_crop_path: Path,
) -> tuple[np.ndarray, dict, tuple[int, int, int]]:
    nifti_data, _ = _load_nifti(nifti_path)
    nifti_data = np.squeeze(nifti_data)
    if "ct" in nifti_path.name.lower():
        nifti_data = _window_cta(nifti_data)
    _save_nifti(nifti_data, output_crop_path, nifti_path)
    roi_dict = {"size": list(nifti_data.shape), "location": [0, 0, 0]}
    return nifti_data, roi_dict, nifti_data.shape


def _process_nifti_with_yolo(
    model,
    nifti_path: Path,
    output_crop_path: Path,
    margin_xy: int,
    margin_z: int,
    scale_xy: float,
    scale_z: float,
    crop_mode: str,
):
    import cv2

    nifti_data, _ = _load_nifti(nifti_path)
    nifti_data = np.squeeze(nifti_data)

    bbox_centers = []
    bbox_widths = []
    bbox_heights = []
    bbox_bounds = []
    slices_with_roi = []

    for slice_idx in range(nifti_data.shape[2]):
        slice_data = nifti_data[:, :, slice_idx]
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            normalized = (slice_data - slice_min) / (slice_max - slice_min) * 255.0
        else:
            normalized = np.zeros_like(slice_data)
        normalized = normalized.astype(np.uint8)
        slice_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

        results = model.predict(source=slice_rgb, verbose=False)
        for result in results:
            if len(result.boxes.xyxy) > 0:
                slices_with_roi.append(slice_idx)
            for bbox in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, bbox)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                bbox_centers.append((center_x, center_y, slice_idx))
                bbox_widths.append(x_max - x_min)
                bbox_heights.append(y_max - y_min)
                bbox_bounds.append((x_min, y_min, x_max, y_max, slice_idx))

    if "ct" in nifti_path.name.lower():
        nifti_data = _window_cta(nifti_data)

    if not bbox_centers:
        cropped = nifti_data
        _save_nifti(cropped, output_crop_path, nifti_path)
        crop_dict = {"size": list(cropped.shape), "location": [0, 0, 0]}
        return cropped, crop_dict, nifti_data.shape

    crop_mode = (crop_mode or "median").lower()
    if crop_mode == "union":
        x_min = min(b[0] for b in bbox_bounds)
        y_min = min(b[1] for b in bbox_bounds)
        x_max = max(b[2] for b in bbox_bounds)
        y_max = max(b[3] for b in bbox_bounds)
        z_min = min(b[4] for b in bbox_bounds)
        z_max = max(b[4] for b in bbox_bounds)
        median_center_x = int(round((x_min + x_max) / 2))
        median_center_y = int(round((y_min + y_max) / 2))
        median_center_z = int(round((z_min + z_max) / 2))
        base_x = max(1, x_max - x_min)
        base_y = max(1, y_max - y_min)
        base_z = max(1, z_max - z_min + 1)
        crop_x_size = int(round(base_x * max(scale_xy, 1.0)))
        crop_y_size = int(round(base_y * max(scale_xy, 1.0)))
        crop_z_size = int(round(base_z * max(scale_z, 1.0)))
    else:
        median_center_x = int(np.median([c[0] for c in bbox_centers]))
        median_center_y = int(np.median([c[1] for c in bbox_centers]))
        median_center_z = int(np.median([c[2] for c in bbox_centers]))
        median_width = int(np.median(bbox_widths))
        median_height = int(np.median(bbox_heights))
        crop_x_size = int(round(median_width * max(scale_xy, 1.0)))
        crop_y_size = int(round(median_height * max(scale_xy, 1.0)))
        crop_z_size = int(round(len(set(slices_with_roi)) * max(scale_z, 1.0)))

    crop_x_min = max(0, median_center_x - crop_x_size // 2)
    crop_x_max = min(nifti_data.shape[0], median_center_x + crop_x_size // 2)
    crop_y_min = max(0, median_center_y - crop_y_size // 2)
    crop_y_max = min(nifti_data.shape[1], median_center_y + crop_y_size // 2)
    crop_z_min = max(0, median_center_z - crop_z_size // 2)
    crop_z_max = min(nifti_data.shape[2], median_center_z + crop_z_size // 2)

    f_xy = int(margin_xy)
    f_z = int(margin_z)
    y0 = max(0, crop_y_min - f_xy)
    y1 = min(nifti_data.shape[0], crop_y_max + f_xy)
    x0 = max(0, crop_x_min - f_xy)
    x1 = min(nifti_data.shape[1], crop_x_max + f_xy)
    z0 = max(0, crop_z_min - f_z)
    z1 = min(nifti_data.shape[2], crop_z_max + f_z)

    cropped = nifti_data[y0:y1, x0:x1, z0:z1]
    _save_nifti(cropped, output_crop_path, nifti_path)

    crop_dict = {
        "size": [y1 - y0, x1 - x0, z1 - z0],
        "location": [y0, x0, z0],
    }

    return cropped, crop_dict, nifti_data.shape


def _segment_nifti(
    img_file: Path,
    yolo_model,
    segmodel_dir: Path,
    device: str,
    work_dir: Path,
    convert_label_13_to_15: bool,
    keep_temp: bool,
    margin_xy: int,
    margin_z: int,
    use_yolo: bool,
    scale_xy: float,
    scale_z: float,
    crop_mode: str,
) -> np.ndarray:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.ensembling.ensemble import ensemble_folders
    temp_root = work_dir / f"temp_topcow_{img_file.stem}"
    temp_save = temp_root / "temp_save"
    temp_best = temp_root / "temp_save_seg_best"
    temp_final = temp_root / "temp_save_seg_final"
    temp_ensemble = temp_root / "temp_save_seg_ensemble"

    shutil.rmtree(temp_root, ignore_errors=True)
    temp_save.mkdir(parents=True, exist_ok=True)
    temp_best.mkdir(parents=True, exist_ok=True)
    temp_final.mkdir(parents=True, exist_ok=True)
    temp_ensemble.mkdir(parents=True, exist_ok=True)

    roi_path = temp_save / "ROI_0000.nii.gz"
    if use_yolo:
        _, roi_dict, original_size = _process_nifti_with_yolo(
            yolo_model,
            img_file,
            roi_path,
            margin_xy=margin_xy,
            margin_z=margin_z,
            scale_xy=scale_xy,
            scale_z=scale_z,
            crop_mode=crop_mode,
        )
    else:
        _, roi_dict, original_size = _process_nifti_full_volume(
            img_file,
            roi_path,
        )

    torch_device = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")

    predictor_best = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor_final = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor_best.initialize_from_trained_model_folder(
        str(segmodel_dir),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_best.pth",
    )
    predictor_final.initialize_from_trained_model_folder(
        str(segmodel_dir),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_final.pth",
    )

    predictor_best.predict_from_files(
        str(temp_save),
        str(temp_best),
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
    )
    predictor_final.predict_from_files(
        str(temp_save),
        str(temp_final),
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
    )

    ensemble_folders([str(temp_best), str(temp_final)], str(temp_ensemble), num_processes=4)

    pred_array, _ = _load_nifti(temp_ensemble / "ROI.nii.gz")
    pred_array = np.round(pred_array).astype(np.uint8)

    pred_resized = np.zeros(original_size, dtype=np.uint8)
    loc = roi_dict["location"]
    size = roi_dict["size"]
    pred_resized[
        loc[0] : loc[0] + size[0],
        loc[1] : loc[1] + size[1],
        loc[2] : loc[2] + size[2],
    ] = pred_array
    pred_array = pred_resized

    if convert_label_13_to_15:
        pred_array[pred_array == 13] = 15

    if not keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    return pred_array


def _write_labels_json(path: Path, include_label_13: bool) -> None:
    labels = dict(TOPCOW_LABELS)
    if include_label_13:
        labels[13] = "3rd-A2"
    labels_out = {str(k): v for k, v in sorted(labels.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels_out}, indent=2))


def run_inference(
    input_path: Path,
    output_dir: Path,
    yolo_model_path: Path | None,
    nnunet_model_dir: Path,
    device: str = "auto",
    work_dir: Path | None = None,
    keep_temp: bool = False,
    overwrite: bool = False,
    labels_json_path: Path | None = None,
    keep_label_13: bool = False,
    margin_xy: int = 10,
    margin_z: int = 5,
    no_yolo_crop: bool = False,
    scale_xy: float = 1.0,
    scale_z: float = 1.0,
    crop_mode: str = "median",
) -> list[Path]:
    use_yolo = not no_yolo_crop
    _require_deps(require_yolo=use_yolo)

    if use_yolo:
        if yolo_model_path is None:
            raise ValueError("yolo_model_path is required unless --no-yolo-crop is set.")
        if not yolo_model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
    if not nnunet_model_dir.exists():
        raise FileNotFoundError(f"Segmentation model dir not found: {nnunet_model_dir}")

    import torch
    import numpy as np

    # Ensure nnUNet v2 paths exist (required by nnunetv2)
    nnunet_raw = os.environ.get("nnUNet_raw") or str(Path.home() / "nnUNet_raw")
    nnunet_pre = os.environ.get("nnUNet_preprocessed") or str(Path.home() / "nnUNet_preprocessed")
    nnunet_res = os.environ.get("nnUNet_results") or str(Path.home() / "nnUNet_results")
    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_pre
    os.environ["nnUNet_results"] = nnunet_res
    Path(nnunet_raw).mkdir(parents=True, exist_ok=True)
    Path(nnunet_pre).mkdir(parents=True, exist_ok=True)
    Path(nnunet_res).mkdir(parents=True, exist_ok=True)

    # PyTorch >=2.6 sets weights_only=True by default; allow numpy scalar for TopCoW checkpoints.
    try:
        torch.serialization.add_safe_globals([
            np._core.multiarray.scalar,
            np.dtype,
            type(np.dtype("float32")),
            type(np.dtype("int64")),
        ])
    except Exception:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir if work_dir is not None else output_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    yolo_model = None
    if use_yolo:
        from ultralytics import YOLO
        yolo_model = YOLO(str(yolo_model_path))

    inputs = _collect_inputs(input_path)
    if not inputs:
        raise FileNotFoundError(f"No NIfTI files found in {input_path}")

    outputs: list[Path] = []
    for img_file in inputs:
        out_name = img_file.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{out_name}_topcow_seg.nii.gz"
        if out_path.exists() and not overwrite:
            print(f"Skipping existing: {out_path}")
            outputs.append(out_path)
            continue

        print(f"Processing {img_file.name} -> {out_path.name}")
        pred = _segment_nifti(
            img_file=img_file,
            yolo_model=yolo_model,
            segmodel_dir=nnunet_model_dir,
            device=device,
            work_dir=work_dir,
            convert_label_13_to_15=not keep_label_13,
            keep_temp=keep_temp,
            margin_xy=margin_xy,
            margin_z=margin_z,
            use_yolo=use_yolo,
            scale_xy=scale_xy,
            scale_z=scale_z,
            crop_mode=crop_mode,
        )
        _save_nifti(pred.astype(np.uint8), out_path, img_file)
        print(f"Saved: {out_path}")
        outputs.append(out_path)

    if labels_json_path:
        _write_labels_json(Path(labels_json_path), include_label_13=keep_label_13)
        print(f"Wrote labels JSON: {labels_json_path}")

    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TopCoW CLAIM inference")
    p.add_argument("--input", required=True, help="Input CTA/MRA NIfTI file or folder")
    p.add_argument("--output", required=True, help="Output folder for segmentations")
    p.add_argument("--yolo-model", default=None, help="Path to yolo-cow-detection.pt")
    p.add_argument("--nnunet-model-dir", required=True, help="Path to topcow-claim-models folder")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device")
    p.add_argument("--work-dir", default=None, help="Temporary working directory")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary inference files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--labels-json", default=None, help="Write TopCoW label mapping JSON")
    p.add_argument(
        "--keep-label-13",
        action="store_true",
        help="Keep label 13 for 3rd-A2 (default converts 13 -> 15)",
    )
    p.add_argument("--crop-margin-xy", type=int, default=10, help="Extra margin added to YOLO ROI crop (x/y)")
    p.add_argument("--crop-margin-z", type=int, default=5, help="Extra margin added to YOLO ROI crop (z)")
    p.add_argument("--crop-scale-xy", type=float, default=1.0, help="Scale factor applied to YOLO ROI size (x/y)")
    p.add_argument("--crop-scale-z", type=float, default=1.0, help="Scale factor applied to YOLO ROI size (z)")
    p.add_argument("--crop-mode", choices=["median", "union"], default="median", help="How to compute YOLO ROI box")
    p.add_argument("--no-yolo-crop", action="store_true", help="Skip YOLO cropping; run full-volume inference")
    return p.parse_args()


def _collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.nii.gz")) + sorted(input_path.glob("*.nii"))
        return files
    raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> int:
    args = parse_args()
    run_inference(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
        nnunet_model_dir=Path(args.nnunet_model_dir),
        device=args.device,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        keep_temp=args.keep_temp,
        overwrite=args.overwrite,
        labels_json_path=Path(args.labels_json) if args.labels_json else None,
        keep_label_13=args.keep_label_13,
        margin_xy=args.crop_margin_xy,
        margin_z=args.crop_margin_z,
        no_yolo_crop=args.no_yolo_crop,
        scale_xy=args.crop_scale_xy,
        scale_z=args.crop_scale_z,
        crop_mode=args.crop_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
