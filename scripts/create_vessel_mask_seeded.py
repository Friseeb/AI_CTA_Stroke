#!/usr/bin/env python3
"""
Seeded HU vessel mask creation.

Keeps only HU-band voxels that are connected to arterial seeds. Seeds can come from:
1) TotalSegmentator multi-label output (segmentator.nii), or
2) A provided binary seed mask, or
3) High-HU voxels (fallback).

This avoids aggressive bone masking and reduces venous leakage by enforcing connectivity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_propagation,
    distance_transform_edt,
    find_objects,
    label,
)


HEAD_NECK_SEED_LABELS = [
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
]

ARTERY_LABELS = [
    "aorta",
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
    "internal_carotid_artery_left",
    "internal_carotid_artery_right",
    "vertebral_artery_left",
    "vertebral_artery_right",
    "basilar_artery",
]

VEIN_LABELS = [
    "internal_jugular_vein_left",
    "internal_jugular_vein_right",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "pulmonary_vein",
    "portal_vein_and_splenic_vein",
    "inferior_vena_cava",
    "superior_vena_cava",
    "vena_cava_inferior",
    "vena_cava_superior",
    "iliac_vein_left",
    "iliac_vein_right",
]

def _load_class_map() -> dict:
    try:
        import totalsegmentator.map_to_binary as m
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("totalsegmentator not available for label mapping") from exc

    return m.class_map


def _load_label_map(class_map: dict, task: str) -> dict:
    label_map = class_map.get(task, {})
    if not label_map:
        raise RuntimeError(f"Missing TotalSegmentator label map for task: {task}")
    return label_map


def _select_totalseg_task(seg: np.ndarray, class_map: dict) -> str:
    max_label = int(seg.max())
    if max_label > 104 and "total" in class_map:
        return "total"
    return "total_v1"


def _build_seed_from_seg(
    seg: np.ndarray,
    label_names: list[str],
    label_map: dict,
    task: str,
) -> tuple[np.ndarray | None, dict]:
    inv = {v: k for k, v in label_map.items()}
    label_ids = [inv[name] for name in label_names if name in inv]

    meta = {
        "task": task,
        "label_ids": label_ids,
        "label_names": [name for name in label_names if name in inv],
    }

    if not label_ids:
        return None, meta

    seed = np.isin(seg, label_ids)
    return seed, meta


def _mask_from_label_names(
    seg: np.ndarray,
    label_map: dict,
    label_names: list[str],
) -> tuple[np.ndarray | None, list[str]]:
    inv = {v: k for k, v in label_map.items()}
    label_ids = [inv[name] for name in label_names if name in inv]
    found = [name for name in label_names if name in inv]
    if not label_ids:
        return None, found
    return np.isin(seg, label_ids), found


def _load_seed_mask(seed_path: Path) -> np.ndarray:
    img = nib.load(str(seed_path))
    data = np.asanyarray(img.dataobj)
    return data > 0


def _sample_values(vals: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or vals.size <= max_samples:
        return vals
    step = max(1, int(vals.size // max_samples))
    return vals[::step]


def _compute_percentiles_from_mask(
    data: np.ndarray,
    mask: np.ndarray,
    percentiles: list[float],
    max_samples: int | None,
) -> tuple[dict[float, float], int] | None:
    vals = data[mask]
    if vals.size == 0:
        return None
    vals = _sample_values(vals.astype(np.int32, copy=False), max_samples)
    perc_vals = np.percentile(vals, percentiles)
    stats = {float(p): float(v) for p, v in zip(percentiles, perc_vals)}
    return stats, int(vals.size)


def _compute_hu_window_from_mask(
    data: np.ndarray,
    mask: np.ndarray,
    low_pct: float,
    high_pct: float,
    pad: float,
    max_samples: int | None,
) -> tuple[int, int, int] | None:
    stats = _compute_percentiles_from_mask(data, mask, [low_pct, high_pct], max_samples)
    if stats is None:
        return None
    pct_vals, used = stats
    low_val = pct_vals[float(low_pct)]
    high_val = pct_vals[float(high_pct)]
    low = int(np.floor(low_val - pad))
    high = int(np.ceil(high_val + pad))
    if high <= low:
        return None
    return low, high, used


def _select_components_by_seeds(
    mask: np.ndarray,
    artery_seed: np.ndarray,
    vein_seed: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    """Select components connected to artery seeds and optionally split mixed components."""
    labeled, num = label(mask)
    if num == 0:
        return mask

    artery_labels = np.unique(labeled[artery_seed]) if artery_seed.any() else np.array([], dtype=int)
    artery_set = {int(x) for x in artery_labels if x > 0}

    vein_set: set[int] = set()
    if vein_seed is not None and vein_seed.any():
        vein_labels = np.unique(labeled[vein_seed])
        vein_set = {int(x) for x in vein_labels if x > 0}

    if not artery_set:
        return np.zeros_like(mask, dtype=bool)

    keep = np.zeros_like(mask, dtype=bool)

    if mode in ("component", "drop"):
        for comp_id in artery_set:
            if mode == "drop" and comp_id in vein_set:
                continue
            keep[labeled == comp_id] = True
        return keep

    if mode != "watershed" or not vein_set:
        for comp_id in artery_set:
            keep[labeled == comp_id] = True
        return keep

    try:
        from skimage.segmentation import watershed
    except Exception:
        for comp_id in artery_set:
            keep[labeled == comp_id] = True
        return keep

    slices = find_objects(labeled)
    for comp_id in artery_set:
        has_vein = comp_id in vein_set
        slc = slices[comp_id - 1]
        if slc is None:
            continue
        comp_mask = labeled[slc] == comp_id
        if not has_vein:
            keep_view = keep[slc]
            keep_view[comp_mask] = True
            keep[slc] = keep_view
            continue

        artery_local = artery_seed[slc] & comp_mask
        vein_local = vein_seed[slc] & comp_mask if vein_seed is not None else None
        if vein_local is None or not vein_local.any() or not artery_local.any():
            keep_view = keep[slc]
            keep_view[comp_mask] = True
            keep[slc] = keep_view
            continue

        markers = np.zeros(comp_mask.shape, dtype=np.int16)
        markers[artery_local] = 1
        markers[vein_local] = 2
        dist = distance_transform_edt(comp_mask)
        labels_local = watershed(-dist, markers, mask=comp_mask)
        keep_local = labels_local == 1
        keep_view = keep[slc]
        keep_view[keep_local] = True
        keep[slc] = keep_view

    return keep


def create_seeded_vessel_mask(
    cta_path: Path,
    output_path: Path,
    segmentator_path: Path | None = None,
    seed_mask_path: Path | None = None,
    hu_low: int = 150,
    hu_high: int = 600,
    hu_auto: str = "off",
    hu_low_pct: float = 5.0,
    hu_high_pct: float = 95.0,
    hu_auto_pad: float = 0.0,
    hu_auto_max_samples: int | None = 1_000_000,
    hu_vein_gap: float = 20.0,
    seed_hu: int | None = 250,
    add_seed_hu: bool = False,
    seed_erosion: int = 0,
    min_component_size: int = 300,
    vein_mode: str = "off",
    save_intermediates: bool = False,
) -> Path:
    cta_img = nib.load(str(cta_path))
    data = np.asanyarray(cta_img.dataobj)

    seg = None
    label_map = None
    seed = None
    seed_meta = None
    artery_mask = None
    vein_mask = None
    if segmentator_path:
        seg_path = Path(segmentator_path)
        if seg_path.exists() and seg_path.is_file():
            seg = np.asanyarray(nib.load(str(seg_path)).dataobj)
            class_map = _load_class_map()
            task = _select_totalseg_task(seg, class_map)
            label_map = _load_label_map(class_map, task)
            seed, seed_meta = _build_seed_from_seg(seg, HEAD_NECK_SEED_LABELS, label_map, task)
            artery_mask, _ = _mask_from_label_names(seg, label_map, ARTERY_LABELS)
            vein_mask, _ = _mask_from_label_names(seg, label_map, VEIN_LABELS)
        else:
            raise FileNotFoundError(f"segmentator file not found: {seg_path}")

    if seed_mask_path:
        seed_mask = _load_seed_mask(seed_mask_path)
        seed = seed_mask if seed is None else (seed | seed_mask)
        artery_mask = seed if artery_mask is None else (artery_mask | seed)

    hu_auto_mode = (hu_auto or "off").lower()
    if hu_auto_mode != "off":
        window = None
        window_source = None
        if hu_auto_mode == "aorta":
            if seg is not None and label_map is not None:
                inv = {v: k for k, v in label_map.items()}
                aorta_id = inv.get("aorta")
                if aorta_id is not None:
                    window = _compute_hu_window_from_mask(
                        data,
                        seg == aorta_id,
                        hu_low_pct,
                        hu_high_pct,
                        hu_auto_pad,
                        hu_auto_max_samples,
                    )
                    window_source = "aorta"
        elif hu_auto_mode == "seed" and seed is not None:
            window = _compute_hu_window_from_mask(
                data,
                seed,
                hu_low_pct,
                hu_high_pct,
                hu_auto_pad,
                hu_auto_max_samples,
            )
            window_source = "seed"
        elif hu_auto_mode in ("artery", "artery_vein"):
            if artery_mask is not None:
                window = _compute_hu_window_from_mask(
                    data,
                    artery_mask,
                    hu_low_pct,
                    hu_high_pct,
                    hu_auto_pad,
                    hu_auto_max_samples,
                )
                window_source = "artery"

            if window is not None and hu_auto_mode == "artery_vein" and vein_mask is not None:
                artery_low, artery_high, _ = window
                vein_stats = _compute_percentiles_from_mask(
                    data,
                    vein_mask,
                    [hu_low_pct, hu_high_pct],
                    hu_auto_max_samples,
                )
                if vein_stats is not None:
                    vein_pct, _ = vein_stats
                    vein_low = vein_pct[float(hu_low_pct)]
                    vein_high = vein_pct[float(hu_high_pct)]
                    new_low = artery_low
                    new_high = artery_high

                    if vein_low - hu_vein_gap > artery_high:
                        new_high = int(np.floor(vein_low - hu_vein_gap))
                    if vein_high + hu_vein_gap < artery_low:
                        new_low = int(np.ceil(vein_high + hu_vein_gap))

                    if new_low < new_high and (new_low != artery_low or new_high != artery_high):
                        window = (new_low, new_high, window[2])
                        print(
                            f"Adjusted HU window using veins: {new_low}-{new_high} "
                            f"(artery {artery_low}-{artery_high}, vein {vein_low:.1f}-{vein_high:.1f})"
                        )

        if window is not None:
            hu_low, hu_high, used = window
            print(
                f"Auto HU window from {window_source}: "
                f"{hu_low}-{hu_high} (p{hu_low_pct:g}/p{hu_high_pct:g}, n={used})"
            )
        else:
            print("Auto HU window unavailable; using provided hu_low/hu_high.")

    candidate = (data >= hu_low) & (data <= hu_high)

    if seed is not None:
        seed = seed & candidate

    if seed is None or not bool(seed.any()):
        if seed_hu is None:
            raise RuntimeError("No seeds found and seed HU fallback disabled.")
        seed = (data >= seed_hu) & (data <= hu_high)
        seed = seed & candidate
        seed_meta = {"task": "hu_seed", "label_ids": [], "label_names": []}
    elif add_seed_hu and seed_hu is not None:
        high_hu = (data >= seed_hu) & (data <= hu_high)
        seed = seed | (high_hu & candidate)

    # Free CTA/seg data to reduce memory before propagation
    del data
    if seg is not None:
        del seg

    mask_for_prop = candidate
    if seed_erosion > 0:
        mask_for_prop = binary_erosion(candidate, iterations=int(seed_erosion))

    artery_seed = artery_mask if artery_mask is not None and artery_mask.any() else seed
    vein_seed = vein_mask if vein_mask is not None and vein_mask.any() else None

    if vein_mode != "off":
        propagated = _select_components_by_seeds(
            mask_for_prop,
            artery_seed=artery_seed,
            vein_seed=vein_seed,
            mode=vein_mode,
        )
    else:
        propagated = binary_propagation(seed, mask=mask_for_prop)

    if seed_erosion > 0:
        propagated = binary_propagation(propagated, mask=candidate)

    if min_component_size > 0:
        labeled, num = label(propagated)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            keep = np.where(counts >= int(min_component_size))[0]
            keep = keep[keep > 0]
            propagated = np.isin(labeled, keep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(propagated.astype(np.uint8), cta_img.affine, cta_img.header), str(output_path))

    if save_intermediates:
        stem = output_path.with_suffix("").name
        inter_dir = output_path.parent / f"{stem}_intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(candidate.astype(np.uint8), cta_img.affine, cta_img.header), str(inter_dir / "candidate_mask.nii.gz"))
        nib.save(nib.Nifti1Image(seed.astype(np.uint8), cta_img.affine, cta_img.header), str(inter_dir / "seed_mask.nii.gz"))
        nib.save(nib.Nifti1Image(propagated.astype(np.uint8), cta_img.affine, cta_img.header), str(inter_dir / "propagated_mask.nii.gz"))
        if seed_meta is not None:
            meta_path = inter_dir / "seed_meta.txt"
            meta_lines = [
                f"seed_task: {seed_meta.get('task')}",
                f"seed_label_ids: {seed_meta.get('label_ids')}",
                f"seed_label_names: {seed_meta.get('label_names')}",
                f"hu_auto: {hu_auto_mode}",
                f"hu_low: {hu_low}",
                f"hu_high: {hu_high}",
                f"hu_vein_gap: {hu_vein_gap}",
                f"vein_mode: {vein_mode}",
            ]
            meta_path.write_text("\n".join(meta_lines) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded HU vessel mask creation")
    parser.add_argument("--cta", required=True, help="Input CTA NIfTI")
    parser.add_argument("--output", required=True, help="Output vessel mask NIfTI")
    parser.add_argument("--segmentator", default=None, help="TotalSegmentator multi-label NIfTI (segmentator.nii)")
    parser.add_argument("--seed-mask", default=None, help="Binary seed mask NIfTI (optional)")
    parser.add_argument("--hu-low", type=int, default=150, help="Lower HU threshold")
    parser.add_argument("--hu-high", type=int, default=600, help="Upper HU threshold")
    parser.add_argument(
        "--hu-auto",
        type=str,
        default="off",
        choices=["off", "aorta", "seed", "artery", "artery_vein"],
        help="Auto-tune HU window per patient",
    )
    parser.add_argument("--hu-low-percentile", type=float, default=5.0, help="Low percentile for auto HU window")
    parser.add_argument("--hu-high-percentile", type=float, default=95.0, help="High percentile for auto HU window")
    parser.add_argument("--hu-auto-pad", type=float, default=0.0, help="Pad HU window by this amount")
    parser.add_argument("--hu-auto-max-samples", type=int, default=1_000_000, help="Max samples for HU auto window")
    parser.add_argument("--hu-vein-gap", type=float, default=20.0, help="Gap required to trim veins from HU window")
    parser.add_argument("--seed-hu", type=int, default=250, help="Fallback high-HU seed threshold")
    parser.add_argument("--add-seed-hu", action="store_true", help="OR high-HU seeds with TS/seed-mask")
    parser.add_argument("--seed-erosion", type=int, default=0, help="Erode candidate mask before propagation")
    parser.add_argument("--min-component", type=int, default=300, help="Minimum component size to keep")
    parser.add_argument(
        "--vein-mode",
        type=str,
        default="off",
        choices=["off", "component", "drop", "watershed"],
        help="Use venous seeds to exclude veins (component or watershed split)",
    )
    parser.add_argument("--save-intermediates", action="store_true", help="Write candidate/seed/prop masks")
    args = parser.parse_args()

    create_seeded_vessel_mask(
        cta_path=Path(args.cta),
        output_path=Path(args.output),
        segmentator_path=Path(args.segmentator) if args.segmentator else None,
        seed_mask_path=Path(args.seed_mask) if args.seed_mask else None,
        hu_low=args.hu_low,
        hu_high=args.hu_high,
        hu_auto=args.hu_auto,
        hu_low_pct=args.hu_low_percentile,
        hu_high_pct=args.hu_high_percentile,
        hu_auto_pad=args.hu_auto_pad,
        hu_auto_max_samples=args.hu_auto_max_samples,
        hu_vein_gap=args.hu_vein_gap,
        seed_hu=args.seed_hu,
        add_seed_hu=args.add_seed_hu,
        seed_erosion=args.seed_erosion,
        min_component_size=args.min_component,
        vein_mode=args.vein_mode,
        save_intermediates=args.save_intermediates,
    )


if __name__ == "__main__":
    main()
