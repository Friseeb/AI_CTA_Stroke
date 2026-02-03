#!/usr/bin/env python3
"""
Build a multi-label NIfTI from per-structure masks (TotalSegmentator outputs)
and optionally merge an intracranial TopCoW label map.

Default labels:
  1 aorta
  2 subclavian_artery_left
  3 subclavian_artery_right
  4 common_carotid_artery_left
  5 common_carotid_artery_right
  6 internal_carotid_artery_left
  7 internal_carotid_artery_right
  8 left_atrium
  9 left_atrial_appendage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_LABEL_SPECS = [
    {"label_name": "aorta", "mask_name": "aorta", "source": "total", "id": 1},
    {"label_name": "subclavian_artery_left", "mask_name": "subclavian_artery_left", "source": "total", "id": 2},
    {"label_name": "subclavian_artery_right", "mask_name": "subclavian_artery_right", "source": "total", "id": 3},
    {"label_name": "common_carotid_artery_left", "mask_name": "common_carotid_artery_left", "source": "total", "id": 4},
    {"label_name": "common_carotid_artery_right", "mask_name": "common_carotid_artery_right", "source": "total", "id": 5},
    {"label_name": "internal_carotid_artery_left", "mask_name": "internal_carotid_artery_left", "source": "headneck", "id": 6},
    {"label_name": "internal_carotid_artery_right", "mask_name": "internal_carotid_artery_right", "source": "headneck", "id": 7},
    {"label_name": "left_atrium", "mask_name": "heart_atrium_left", "source": "heartchambers", "id": 8},
    {"label_name": "left_atrial_appendage", "mask_name": "atrial_appendage_left", "source": "total", "id": 9},
]

TOPCOW_LABELS = {
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
    13: "3rd-A2",
    15: "3rd-A2",
}


def _mask_path(dir_path: Path | None, name: str) -> Path | None:
    if dir_path is None:
        return None
    if (dir_path / f"{name}.nii.gz").exists():
        return dir_path / f"{name}.nii.gz"
    if (dir_path / name).exists():
        return dir_path / name
    return None


def _load_binary_mask(path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.shape != reference_shape:
        raise ValueError(f"Shape mismatch for {path}: {data.shape} vs {reference_shape}")
    return data > 0


def _load_label_specs(config_path: Path | None) -> list[dict]:
    if config_path is None:
        return DEFAULT_LABEL_SPECS
    data = json.loads(config_path.read_text())
    if isinstance(data, dict) and "labels" in data:
        data = data["labels"]
    if not isinstance(data, list):
        raise ValueError("Label config must be a list or a dict with key 'labels'.")
    required = {"label_name", "mask_name", "source", "id"}
    for item in data:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Label config entry missing keys: {missing}")
    return data


def _apply_mask(
    label_map: np.ndarray,
    mask: np.ndarray,
    label_id: int,
    overwrite: bool,
) -> int:
    if overwrite:
        label_map[mask] = label_id
        return int(mask.sum())
    free = mask & (label_map == 0)
    label_map[free] = label_id
    return int(free.sum())


def _parse_int_mapping(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("TopCoW remap must be a JSON object of {old_id: new_id}.")
    mapping: dict[int, int] = {}
    for k, v in data.items():
        try:
            mapping[int(k)] = int(v)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid remap entry: {k}:{v}") from exc
    return mapping


def _merge_topcow(
    label_map: np.ndarray,
    topcow_path: Path,
    reference_shape: tuple[int, int, int],
    mode: str,
    binary_label: int,
    offset: int,
    remap: dict[int, int] | None,
    overwrite: bool,
) -> dict[str, int]:
    img = nib.load(str(topcow_path))
    data = np.asarray(img.dataobj)
    if data.shape != reference_shape:
        raise ValueError(f"TopCoW shape mismatch: {data.shape} vs {reference_shape}")

    stats: dict[str, int] = {}
    if mode == "auto":
        unique_vals = np.unique(data)
        if unique_vals.size <= 2 and unique_vals.min() >= 0 and unique_vals.max() <= 1:
            mode = "binary"
        else:
            mode = "label"

    if mode == "binary":
        mask = data > 0
        stats["topcow_binary_voxels"] = _apply_mask(label_map, mask, binary_label, overwrite)
        return stats

    labels = data.astype(np.int32, copy=False)
    if remap:
        remapped = np.zeros_like(labels, dtype=np.int32)
        for old_id, new_id in remap.items():
            if old_id <= 0:
                continue
            remapped[labels == old_id] = int(new_id)
        labels = remapped
    elif offset:
        labels = np.where(labels > 0, labels + int(offset), 0)

    mask = labels > 0
    if overwrite:
        label_map[mask] = labels[mask]
        stats["topcow_label_voxels"] = int(mask.sum())
    else:
        free = mask & (label_map == 0)
        label_map[free] = labels[free]
        stats["topcow_label_voxels"] = int(free.sum())
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-label vascular map from TotalSegmentator masks")
    p.add_argument("--reference", required=True, help="Reference NIfTI (CTA or mask) for shape/affine/header")
    p.add_argument("--output", required=True, help="Output multi-label NIfTI (.nii.gz)")
    p.add_argument("--totalseg-dir", required=True, help="TotalSegmentator total-task output directory")
    p.add_argument("--headneck-dir", default=None, help="TotalSegmentator headneck output directory (defaults to --totalseg-dir)")
    p.add_argument("--heartchambers-dir", default=None, help="TotalSegmentator heartchambers_highres output directory")
    p.add_argument("--label-config", default=None, help="Optional JSON config overriding default labels")
    p.add_argument("--labels-json", default=None, help="Optional JSON output for label ID mapping")
    p.add_argument("--overwrite", action="store_true", help="Allow later labels to overwrite earlier labels")

    p.add_argument("--topcow", default=None, help="Optional TopCoW label map NIfTI to merge")
    p.add_argument("--topcow-mode", choices=["auto", "binary", "label"], default="auto", help="TopCoW input type")
    p.add_argument("--topcow-label", type=int, default=50, help="Label ID to use for TopCoW binary mask")
    p.add_argument("--topcow-offset", type=int, default=100, help="Offset to add to TopCoW label map values")
    p.add_argument("--topcow-remap", default=None, help="JSON mapping for TopCoW label remap (old_id -> new_id)")
    return p.parse_args()


def build_multilabel_map(
    reference_path: Path,
    output_path: Path,
    totalseg_dir: Path,
    headneck_dir: Path | None = None,
    heartchambers_dir: Path | None = None,
    label_config: Path | None = None,
    labels_json_path: Path | None = None,
    overwrite: bool = False,
    topcow_path: Path | None = None,
    topcow_mode: str = "auto",
    topcow_label: int = 50,
    topcow_offset: int = 100,
    topcow_remap: Path | None = None,
) -> dict:
    reference_img = nib.load(str(reference_path))
    reference_shape = reference_img.shape

    headneck_dir = headneck_dir if headneck_dir is not None else totalseg_dir
    label_specs = _load_label_specs(label_config) if label_config else DEFAULT_LABEL_SPECS

    label_map = np.zeros(reference_shape, dtype=np.int32)
    labels_json: dict[str, str] = {}

    source_dirs = {
        "total": totalseg_dir,
        "headneck": headneck_dir,
        "heartchambers": heartchambers_dir,
    }

    print("Building label map...")
    for spec in label_specs:
        label_name = spec["label_name"]
        mask_name = spec["mask_name"]
        source = spec["source"]
        label_id = int(spec["id"])

        src_dir = source_dirs.get(source)
        if src_dir is None:
            print(f"  ⚠ Missing directory for source '{source}' ({label_name}); skipping")
            continue

        mask_path = _mask_path(src_dir, mask_name)
        if mask_path is None:
            print(f"  ⚠ Missing mask for {label_name}: {mask_name} in {src_dir}")
            continue

        mask = _load_binary_mask(mask_path, reference_shape)
        voxels = _apply_mask(label_map, mask, label_id, overwrite=overwrite)
        if voxels > 0:
            labels_json[str(label_id)] = label_name
        print(f"  ✓ {label_name} -> {label_id} ({voxels:,} voxels)")

    topcow_meta: dict[str, int] = {}
    topcow_label_map: dict[str, str] = {}
    if topcow_path:
        remap = _parse_int_mapping(topcow_remap) if topcow_remap else None
        topcow_meta = _merge_topcow(
            label_map=label_map,
            topcow_path=topcow_path,
            reference_shape=reference_shape,
            mode=topcow_mode,
            binary_label=int(topcow_label),
            offset=int(topcow_offset),
            remap=remap,
            overwrite=overwrite,
        )
        print(f"  ✓ TopCoW merged: {topcow_meta}")

        if topcow_mode == "binary":
            topcow_label_map[str(int(topcow_label))] = "TopCoW"
        else:
            if remap:
                for old_id, new_id in remap.items():
                    name = TOPCOW_LABELS.get(int(old_id))
                    if name:
                        topcow_label_map[str(int(new_id))] = name
            elif topcow_offset:
                for old_id, name in TOPCOW_LABELS.items():
                    if int(old_id) <= 0:
                        continue
                    topcow_label_map[str(int(old_id) + int(topcow_offset))] = name
            else:
                for old_id, name in TOPCOW_LABELS.items():
                    if int(old_id) <= 0:
                        continue
                    topcow_label_map[str(int(old_id))] = name

    out_dtype = np.int16 if label_map.max() <= np.iinfo(np.int16).max else np.int32
    label_map = label_map.astype(out_dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(label_map, reference_img.affine, reference_img.header), str(output_path))
    print(f"✓ Saved label map: {output_path}")

    labels_path_written = None
    if labels_json_path:
        labels_out = {
            "labels": {**labels_json, **topcow_label_map},
            "topcow": {
                "mode": topcow_mode,
                "label": int(topcow_label),
                "offset": int(topcow_offset),
                "stats": topcow_meta,
            },
        }
        labels_json_path.parent.mkdir(parents=True, exist_ok=True)
        labels_json_path.write_text(json.dumps(labels_out, indent=2))
        labels_path_written = str(labels_json_path)
        print(f"✓ Saved labels JSON: {labels_json_path}")

    return {
        "label_map_path": str(output_path),
        "labels_json_path": labels_path_written,
        "topcow_stats": topcow_meta,
    }


def main() -> int:
    args = parse_args()
    result = build_multilabel_map(
        reference_path=Path(args.reference),
        output_path=Path(args.output),
        totalseg_dir=Path(args.totalseg_dir),
        headneck_dir=Path(args.headneck_dir) if args.headneck_dir else None,
        heartchambers_dir=Path(args.heartchambers_dir) if args.heartchambers_dir else None,
        label_config=Path(args.label_config) if args.label_config else None,
        labels_json_path=Path(args.labels_json) if args.labels_json else None,
        overwrite=args.overwrite,
        topcow_path=Path(args.topcow) if args.topcow else None,
        topcow_mode=args.topcow_mode,
        topcow_label=args.topcow_label,
        topcow_offset=args.topcow_offset,
        topcow_remap=Path(args.topcow_remap) if args.topcow_remap else None,
    )
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
