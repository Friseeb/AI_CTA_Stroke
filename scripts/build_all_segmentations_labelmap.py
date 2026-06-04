#!/usr/bin/env python3
"""
Build a single multi-label NIfTI containing key vascular/cardiac segmentations.

Includes:
  - High-res aorta (TotalSegmentator heartchambers_highres if available)
  - Subclavian (L/R), CCA (L/R), ICA (L/R)
  - Left atrium (high-res)
  - LAA from TotalSegmentator, NV-Segment-CT, and NUDF
  - Optional TopCoW (Circle of Willis) label map

Notes:
  - Overlaps are resolved by label order; later labels can overwrite earlier ones
    when --overwrite is set.
  - External LAA masks must be in the same space as the reference image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-label labelmap from multiple segmentation sources")
    p.add_argument("--reference", required=True, help="Reference NIfTI for shape/affine/header")
    p.add_argument("--output", required=True, help="Output multi-label NIfTI (.nii.gz)")
    p.add_argument("--labels-json", default=None, help="Optional JSON output for label mapping")

    p.add_argument("--totalseg-total", required=True, help="TotalSegmentator total-task output directory")
    p.add_argument("--totalseg-headneck", required=True, help="TotalSegmentator headneck output directory")
    p.add_argument("--totalseg-heartchambers", required=True, help="TotalSegmentator heartchambers_highres directory")

    p.add_argument("--laa-nv", default=None, help="NV-Segment-CT LAA mask NIfTI")
    p.add_argument("--laa-nudf", default=None, help="NUDF LAA mask NIfTI")
    p.add_argument("--laa-nv-id", type=int, default=10, help="Label ID for NV LAA")
    p.add_argument("--laa-nudf-id", type=int, default=11, help="Label ID for NUDF LAA")
    p.add_argument("--aorta-nv", default=None, help="NV-Segment-CT aorta mask NIfTI")
    p.add_argument("--aorta-nv-id", type=int, default=12, help="Label ID for NV aorta")

    p.add_argument("--topcow", default=None, help="Optional TopCoW label map NIfTI")
    p.add_argument("--topcow-mode", choices=["auto", "binary", "label"], default="auto", help="TopCoW input type")
    p.add_argument("--topcow-label", type=int, default=50, help="Label ID for TopCoW binary mask")
    p.add_argument("--topcow-offset", type=int, default=100, help="Offset to add to TopCoW label map values")
    p.add_argument("--topcow-remap", default=None, help="JSON mapping for TopCoW remap (old_id -> new_id)")

    p.add_argument("--overwrite", action="store_true", help="Allow later labels to overwrite earlier labels")
    return p.parse_args()


def _mask_path(dir_path: Path, name: str) -> Path | None:
    if (dir_path / f"{name}.nii.gz").exists():
        return dir_path / f"{name}.nii.gz"
    if (dir_path / name).exists():
        return dir_path / name
    return None


def _load_mask_or_warn(
    path: Path | None,
    reference_shape: tuple[int, int, int],
    label_name: str,
) -> np.ndarray | None:
    if path is None:
        print(f"  ⚠ Missing mask for {label_name}")
        return None
    return _load_binary_mask(path, reference_shape)


def _load_preferred_mask(
    candidates: list[tuple[str, Path | None]],
    reference_shape: tuple[int, int, int],
    label_name: str,
) -> np.ndarray | None:
    """Load the first available mask from a prioritized list of sources."""
    for source_name, path in candidates:
        if path is None:
            continue
        try:
            mask = _load_binary_mask(path, reference_shape)
            print(f"  ✓ {label_name} from {source_name}")
            return mask
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ Failed {label_name} from {source_name}: {exc}")
    print(f"  ⚠ Missing mask for {label_name}")
    return None


def _load_binary_mask(path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.shape != reference_shape:
        raise ValueError(f"Shape mismatch for {path}: {data.shape} vs {reference_shape}")
    return data > 0


def _apply_mask(label_map: np.ndarray, mask: np.ndarray, label_id: int, overwrite: bool) -> int:
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


def main() -> int:
    args = parse_args()
    reference_img = nib.load(str(args.reference))
    reference_shape = reference_img.shape

    label_map = np.zeros(reference_shape, dtype=np.int32)
    labels_json: dict[str, str] = {}

    total_dir = Path(args.totalseg_total)
    headneck_dir = Path(args.totalseg_headneck)
    heart_dir = Path(args.totalseg_heartchambers)

    print("Building label map...")

    # High-res aorta (prefer heartchambers_highres)
    aorta_mask = _load_preferred_mask(
        [
            ("heartchambers_highres", _mask_path(heart_dir, "aorta")),
            ("total", _mask_path(total_dir, "aorta")),
        ],
        reference_shape,
        "aorta_highres",
    )
    if aorta_mask is not None:
        vox = _apply_mask(label_map, aorta_mask, 1, args.overwrite)
        labels_json["1"] = "aorta_highres"
        print(f"  ✓ aorta_highres -> 1 ({vox:,} voxels)")

    # Subclavians (prefer headneck if available)
    for label_id, name in [(2, "subclavian_artery_left"), (3, "subclavian_artery_right")]:
        mask = _load_preferred_mask(
            [
                ("headneck", _mask_path(headneck_dir, name)),
                ("total", _mask_path(total_dir, name)),
            ],
            reference_shape,
            name,
        )
        if mask is not None:
            vox = _apply_mask(label_map, mask, label_id, args.overwrite)
            labels_json[str(label_id)] = name
            print(f"  ✓ {name} -> {label_id} ({vox:,} voxels)")

    # Common carotids (prefer headneck if available)
    for label_id, name in [(4, "common_carotid_artery_left"), (5, "common_carotid_artery_right")]:
        mask = _load_preferred_mask(
            [
                ("headneck", _mask_path(headneck_dir, name)),
                ("total", _mask_path(total_dir, name)),
            ],
            reference_shape,
            name,
        )
        if mask is not None:
            vox = _apply_mask(label_map, mask, label_id, args.overwrite)
            labels_json[str(label_id)] = name
            print(f"  ✓ {name} -> {label_id} ({vox:,} voxels)")

    # Internal carotids (headneck)
    for label_id, name in [(6, "internal_carotid_artery_left"), (7, "internal_carotid_artery_right")]:
        mask = _load_preferred_mask(
            [
                ("headneck", _mask_path(headneck_dir, name)),
                ("total", _mask_path(total_dir, name)),
            ],
            reference_shape,
            name,
        )
        if mask is not None:
            vox = _apply_mask(label_map, mask, label_id, args.overwrite)
            labels_json[str(label_id)] = name
            print(f"  ✓ {name} -> {label_id} ({vox:,} voxels)")

    # Left atrium high-res
    la_mask = _load_mask_or_warn(_mask_path(heart_dir, "heart_atrium_left"), reference_shape, "left_atrium_highres")
    if la_mask is not None:
        vox = _apply_mask(label_map, la_mask, 8, args.overwrite)
        labels_json["8"] = "left_atrium_highres"
        print(f"  ✓ left_atrium_highres -> 8 ({vox:,} voxels)")

    # LAA by TotalSegmentator (total task)
    laa_ts = _load_mask_or_warn(_mask_path(total_dir, "atrial_appendage_left"), reference_shape, "laa_totalseg")
    if laa_ts is not None:
        vox = _apply_mask(label_map, laa_ts, 9, args.overwrite)
        labels_json["9"] = "laa_totalseg"
        print(f"  ✓ laa_totalseg -> 9 ({vox:,} voxels)")

    # LAA by NV-Segment-CT
    if args.laa_nv:
        nv_path = Path(args.laa_nv)
        nv_mask = _load_mask_or_warn(nv_path, reference_shape, "laa_nv_segment_ct")
        if nv_mask is not None:
            vox = _apply_mask(label_map, nv_mask, int(args.laa_nv_id), args.overwrite)
            labels_json[str(int(args.laa_nv_id))] = "laa_nv_segment_ct"
            print(f"  ✓ laa_nv_segment_ct -> {args.laa_nv_id} ({vox:,} voxels)")

    # LAA by NUDF
    if args.laa_nudf:
        nudf_path = Path(args.laa_nudf)
        nudf_mask = _load_mask_or_warn(nudf_path, reference_shape, "laa_nudf")
        if nudf_mask is not None:
            vox = _apply_mask(label_map, nudf_mask, int(args.laa_nudf_id), args.overwrite)
            labels_json[str(int(args.laa_nudf_id))] = "laa_nudf"
            print(f"  ✓ laa_nudf -> {args.laa_nudf_id} ({vox:,} voxels)")

    # Aorta by NV-Segment-CT (optional)
    if args.aorta_nv:
        nv_aorta_path = Path(args.aorta_nv)
        nv_aorta_mask = _load_mask_or_warn(nv_aorta_path, reference_shape, "aorta_nv_segment_ct")
        if nv_aorta_mask is not None:
            vox = _apply_mask(label_map, nv_aorta_mask, int(args.aorta_nv_id), args.overwrite)
            labels_json[str(int(args.aorta_nv_id))] = "aorta_nv_segment_ct"
            print(f"  ✓ aorta_nv_segment_ct -> {args.aorta_nv_id} ({vox:,} voxels)")

    # TopCoW merge
    topcow_meta: dict[str, int] = {}
    topcow_label_map: dict[str, str] = {}
    if args.topcow:
        remap = _parse_int_mapping(Path(args.topcow_remap)) if args.topcow_remap else None
        topcow_meta = _merge_topcow(
            label_map=label_map,
            topcow_path=Path(args.topcow),
            reference_shape=reference_shape,
            mode=args.topcow_mode,
            binary_label=int(args.topcow_label),
            offset=int(args.topcow_offset),
            remap=remap,
            overwrite=args.overwrite,
        )
        print(f"  ✓ TopCoW merged: {topcow_meta}")
        if args.topcow_mode == "binary":
            topcow_label_map[str(int(args.topcow_label))] = "TopCoW"
        else:
            if remap:
                for old_id, new_id in remap.items():
                    name = TOPCOW_LABELS.get(int(old_id), f"TopCoW_{old_id}")
                    topcow_label_map[str(int(new_id))] = name
            elif args.topcow_offset:
                for old_id, name in TOPCOW_LABELS.items():
                    if int(old_id) <= 0:
                        continue
                    topcow_label_map[str(int(old_id) + int(args.topcow_offset))] = name
            else:
                for old_id, name in TOPCOW_LABELS.items():
                    if int(old_id) <= 0:
                        continue
                    topcow_label_map[str(int(old_id))] = name

    out_dtype = np.int16 if label_map.max() <= np.iinfo(np.int16).max else np.int32
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(label_map.astype(out_dtype), reference_img.affine, reference_img.header), str(output_path))
    print(f"✓ Saved label map: {output_path}")

    if args.labels_json:
        labels_out = {
            "labels": {**labels_json, **topcow_label_map},
            "topcow": {
                "mode": args.topcow_mode,
                "label": int(args.topcow_label),
                "offset": int(args.topcow_offset),
                "stats": topcow_meta,
            },
        }
        labels_path = Path(args.labels_json)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels_out, indent=2))
        print(f"✓ Saved labels JSON: {labels_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
