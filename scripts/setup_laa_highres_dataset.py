#!/usr/bin/env python3
"""
High-res LAA dataset setup from the STACOM 2025 ImageCAS labels.

This script prepares training-ready datasets for LAA (label=8) using:
  1) ImageCAS CCTA volumes (Kaggle)
  2) STACOM 2025 segmentation labels (zip from DTU link)

Outputs:
  - Extracted segmentation labels
  - Optional LAA-only binary masks
  - Optional nnUNetv2 dataset folder (imagesTr/labelsTr + dataset.json)

Typical usage:
  # 1) Extract the segmentation zip
  python setup_laa_highres_dataset.py extract \
    --seg-zip /path/to/ImageCAS-STACOM2025-02-10-2025.zip \
    --out-dir /path/to/LAADATA

  # 2) Create LAA-only masks (label 8)
  python setup_laa_highres_dataset.py laa-only \
    --dataset-dir /path/to/LAADATA \
    --out-dir /path/to/LAADATA/labels_laa \
    --full-only

  # 3) Build nnUNet dataset (LAA-only)
  python setup_laa_highres_dataset.py nnunet \
    --images-dir /path/to/ImageCAS/images \
    --labels-dir /path/to/LAADATA/labels_laa \
    --out-dir /path/to/nnunetv2_raw/Dataset901_LAA \
    --laa-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_LAA_LABEL = 8


def _find_dataset_root(root: Path) -> Path:
    """Find extracted dataset root containing segmentations/ and info/."""
    if (root / "segmentations").exists():
        return root
    for path in root.rglob("segmentations"):
        candidate = path.parent
        if (candidate / "info").exists():
            return candidate
    return root


def _load_ids(info_dir: Path, full_only: bool) -> list[str]:
    if full_only:
        list_path = info_dir / "all_full_laa_segmentations_id.txt"
    else:
        list_path = info_dir / "all_segmentations_id.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing id list: {list_path}")
    ids = np.loadtxt(list_path, dtype=str).tolist()
    if isinstance(ids, str):
        ids = [ids]
    return ids


def _iter_label_files(segmentations_dir: Path, ids: list[str] | None) -> list[Path]:
    if ids is None:
        return sorted(segmentations_dir.glob("*.nii.gz"))
    paths = []
    for case_id in ids:
        p = segmentations_dir / f"{case_id}.nii.gz"
        if p.exists():
            paths.append(p)
    return paths


def extract_zip(seg_zip: Path, out_dir: Path, force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return _find_dataset_root(out_dir)
    with zipfile.ZipFile(seg_zip, "r") as zf:
        zf.extractall(out_dir)
    return _find_dataset_root(out_dir)


def build_laa_only(
    dataset_dir: Path,
    out_dir: Path,
    laa_label: int,
    full_only: bool,
    overwrite: bool,
) -> int:
    dataset_dir = _find_dataset_root(dataset_dir)
    segmentations_dir = dataset_dir / "segmentations"
    info_dir = dataset_dir / "info"
    if not segmentations_dir.exists():
        raise FileNotFoundError(f"segmentations dir not found: {segmentations_dir}")
    if not info_dir.exists():
        raise FileNotFoundError(f"info dir not found: {info_dir}")

    ids = _load_ids(info_dir, full_only=full_only)
    files = _iter_label_files(segmentations_dir, ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for seg_path in files:
        out_path = out_dir / seg_path.name
        if out_path.exists() and not overwrite:
            continue
        img = nib.load(str(seg_path))
        data = np.asarray(img.dataobj)
        laa_mask = (data == laa_label).astype(np.uint8)
        nib.save(nib.Nifti1Image(laa_mask, img.affine, img.header), str(out_path))
        count += 1
    return count


def _match_images(images_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        case_id = img_path.name.replace(".nii.gz", "")
        mapping[case_id] = img_path
    return mapping


def build_nnunet_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    laa_only: bool,
    overwrite: bool,
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_map = _match_images(images_dir)
    label_paths = sorted(labels_dir.glob("*.nii.gz"))

    copied = 0
    missing = 0
    for lab_path in label_paths:
        case_id = lab_path.name.replace(".nii.gz", "")
        img_path = image_map.get(case_id)
        if img_path is None:
            missing += 1
            continue

        img_target = images_tr / f"{case_id}_0000.nii.gz"
        lab_target = labels_tr / f"{case_id}.nii.gz"
        if (img_target.exists() or lab_target.exists()) and not overwrite:
            continue
        shutil.copy2(img_path, img_target)
        shutil.copy2(lab_path, lab_target)
        copied += 1

    labels = {"background": 0, "LAA": 1} if laa_only else {
        "background": 0,
        "myocardium": 1,
        "LA": 2,
        "LV": 3,
        "RA": 4,
        "RV": 5,
        "aorta": 6,
        "PA": 7,
        "LAA": 8,
        "coronary": 9,
        "PV": 10,
    }

    dataset_json = {
        "name": out_dir.name,
        "description": "STACOM 2025 ImageCAS LAA dataset",
        "tensorImageSize": "3D",
        "reference": "https://arxiv.org/abs/2510.06090",
        "licence": "Check ImageCAS and STACOM 2025 label licenses",
        "channel_names": {"0": "CCTA"},
        "labels": labels,
        "numTraining": len(list(labels_tr.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    return {"copied": copied, "missing_images": missing, "dataset_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare high-res LAA dataset from STACOM 2025 labels")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract segmentation zip")
    p_extract.add_argument("--seg-zip", required=True, help="Path to ImageCAS-STACOM2025-*.zip")
    p_extract.add_argument("--out-dir", required=True, help="Output directory")
    p_extract.add_argument("--force", action="store_true", help="Overwrite existing output")

    p_laa = sub.add_parser("laa-only", help="Create LAA-only binary masks (label 8)")
    p_laa.add_argument("--dataset-dir", required=True, help="Extracted dataset directory")
    p_laa.add_argument("--out-dir", required=True, help="Output directory for LAA masks")
    p_laa.add_argument("--laa-label", type=int, default=DEFAULT_LAA_LABEL, help="Label id for LAA (default=8)")
    p_laa.add_argument("--full-only", action="store_true", help="Use only full-LAA cases")
    p_laa.add_argument("--overwrite", action="store_true", help="Overwrite outputs if present")

    p_nnunet = sub.add_parser("nnunet", help="Build nnUNetv2 dataset structure")
    p_nnunet.add_argument("--images-dir", required=True, help="ImageCAS images directory (NIfTI)")
    p_nnunet.add_argument("--labels-dir", required=True, help="Labels directory (NIfTI)")
    p_nnunet.add_argument("--out-dir", required=True, help="Output nnUNet dataset folder")
    p_nnunet.add_argument("--laa-only", action="store_true", help="Labels are LAA-only (binary)")
    p_nnunet.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "extract":
        root = extract_zip(Path(args.seg_zip), Path(args.out_dir), force=args.force)
        print(f"Extracted dataset root: {root}")
        return 0
    if args.cmd == "laa-only":
        count = build_laa_only(
            dataset_dir=Path(args.dataset_dir),
            out_dir=Path(args.out_dir),
            laa_label=args.laa_label,
            full_only=args.full_only,
            overwrite=args.overwrite,
        )
        print(f"Wrote {count} LAA masks to {args.out_dir}")
        return 0
    if args.cmd == "nnunet":
        res = build_nnunet_dataset(
            images_dir=Path(args.images_dir),
            labels_dir=Path(args.labels_dir),
            out_dir=Path(args.out_dir),
            laa_only=args.laa_only,
            overwrite=args.overwrite,
        )
        print(json.dumps(res, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
