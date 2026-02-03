#!/usr/bin/env python3
"""
Split centerline extraction into two processes to avoid Julia/Torch conflicts.

Stage A (EDT): uses distance_transforms (Julia/Metal) to compute distance map + extremal points.
Stage B (Trace): uses torch MPS for path tracing and graph export.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    class _NoTqdm:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

        def set_description(self, *args, **kwargs) -> None:
            return None

        def set_postfix_str(self, *args, **kwargs) -> None:
            return None

    def tqdm(*args, **kwargs):  # type: ignore[no-redef]
        return _NoTqdm()

import nibabel as nib
import numpy as np

# Ensure repo root on import path (for python.analysis.* modules)
REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_binary_mask(data: np.ndarray) -> bool:
    finite = np.isfinite(data)
    if not finite.all():
        data = np.nan_to_num(data)
    unique_vals = np.unique(data)
    return bool(unique_vals.size <= 3 and unique_vals.min() >= 0 and unique_vals.max() <= 1)


def _crop_and_downsample(
    mask: np.ndarray,
    margin_vox: int | tuple[int, int, int] = 8,
    downsample_factor: int = 1,
) -> tuple[np.ndarray, tuple[int, int, int], int]:
    """Crop mask to bbox + margin; optional max-pool downsample."""
    def downsample_max_pool(vol: np.ndarray, ds: int) -> np.ndarray:
        if ds <= 1:
            return vol
        pad_x = (-vol.shape[0]) % ds
        pad_y = (-vol.shape[1]) % ds
        pad_z = (-vol.shape[2]) % ds
        if pad_x or pad_y or pad_z:
            vol = np.pad(
                vol,
                ((0, pad_x), (0, pad_y), (0, pad_z)),
                mode="constant",
                constant_values=False,
            )
        sx, sy, sz = vol.shape
        vol = vol.reshape(sx // ds, ds, sy // ds, ds, sz // ds, ds)
        return vol.max(axis=(1, 3, 5))

    coords = np.where(mask)
    if len(coords[0]) == 0:
        return mask, (0, 0, 0), 1

    if isinstance(margin_vox, (tuple, list)) and len(margin_vox) == 3:
        mx, my, mz = [int(m) for m in margin_vox]
    else:
        mx = my = mz = int(margin_vox)

    x0 = max(int(coords[0].min()) - mx, 0)
    x1 = min(int(coords[0].max()) + mx + 1, mask.shape[0])
    y0 = max(int(coords[1].min()) - my, 0)
    y1 = min(int(coords[1].max()) + my + 1, mask.shape[1])
    z0 = max(int(coords[2].min()) - mz, 0)
    z1 = min(int(coords[2].max()) + mz + 1, mask.shape[2])

    cropped = mask[x0:x1, y0:y1, z0:z1]

    ds = max(1, int(downsample_factor))
    if ds > 1:
        cropped = downsample_max_pool(cropped.astype(bool), ds)

    return cropped, (x0, y0, z0), ds


def _rescale_centerlines(centerlines: dict, offset: tuple[int, int, int], scale: int) -> dict:
    """Rescale centerline paths/radii back to original voxel space."""
    if scale == 1 and offset == (0, 0, 0):
        return centerlines

    offset_arr = np.array(offset, dtype=float)
    scale_f = float(scale)

    for seg_id, seg_data in centerlines.items():
        path = np.array(seg_data["path"], dtype=float)
        path = path * scale_f + offset_arr
        seg_data["path"] = path
        if len(path) > 1:
            diffs = np.diff(path, axis=0)
            seg_data["length"] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            seg_data["length"] = 0.0
        if "radii" in seg_data:
            seg_data["radii"] = np.array(seg_data["radii"], dtype=float) * scale_f
        centerlines[seg_id] = seg_data

    return centerlines


def _save_centerline_mask(
    centerlines: dict,
    reference_img: nib.Nifti1Image,
    output_path: Path,
    step: float = 0.5,
    dilation_iters: int = 0,
) -> int:
    """Rasterize centerlines into a binary mask aligned to reference_img."""
    mask = np.zeros(reference_img.shape, dtype=np.uint8)
    bounds = np.array(reference_img.shape) - 1

    def stamp(points: np.ndarray) -> None:
        if points.size == 0:
            return
        pts = np.round(points).astype(int)
        pts = np.clip(pts, 0, bounds)
        mask[pts[:, 0], pts[:, 1], pts[:, 2]] = 1

    for seg_data in centerlines.values():
        pts = np.asarray(seg_data["path"], dtype=float)
        if pts.shape[0] == 0:
            continue
        stamp(pts)

        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n_steps = max(2, int(np.ceil(seg_len / max(step, 1e-3))))
            interp = np.linspace(a, b, n_steps)
            stamp(interp)

    if dilation_iters > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilation_iters).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, reference_img.affine, reference_img.header), str(output_path))
    return int(mask.sum())


def _run_edt_stage(args: argparse.Namespace) -> Path:
    from python.analysis.centerline_antiga_2008.stage1_surface_extraction import extract_surface
    from python.analysis.centerline_antiga_2008.stage2_extremal_points import detect_extremal_points

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_file = Path(args.work_file) if args.work_file else output_dir / "centerline_stage12.npz"

    progress = tqdm(total=2, desc="EDT stage (distance transform)", unit="step", disable=args.no_progress)
    progress.set_postfix_str(f"requested={args.edt_backend}")

    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj)
    mask = (data > 0) if _is_binary_mask(data) else (data > 0)

    proc_mask, offset, scale = _crop_and_downsample(
        mask,
        margin_vox=args.crop_margin_vox,
        downsample_factor=args.downsample_factor,
    )

    stage1 = extract_surface(
        proc_mask,
        min_component_size=args.min_component_size,
        erosion_iterations=args.erosion_iterations,
        dilation_iterations=args.dilation_iterations,
        thick_component_max_radius=args.thick_component_max_radius if args.thick_component_max_radius > 0 else None,
        edt_backend=args.edt_backend,
        gpu_backend=None,
        allow_cpu_edt=not args.no_cpu_edt,
    )
    progress.update(1)
    progress.set_description("EDT stage (extremal points)")

    min_distance_value = float(args.min_distance_value)
    if scale > 1:
        min_distance_value = min_distance_value / float(scale)

    stage2 = detect_extremal_points(
        stage1["cleaned_mask"],
        min_distance_value=min_distance_value,
        distance_map=stage1["distance_map"],
        edt_backend=args.edt_backend,
        gpu_backend=None,
        max_filter_backend=args.max_filter_backend,
        retry_if_empty=True,
        skeleton_fallback=True,
        allow_cpu_edt=not args.no_cpu_edt,
    )
    progress.update(1)
    progress.close()

    extremal_points = stage2["extremal_points"]
    if args.max_extremal_points and len(extremal_points) > args.max_extremal_points:
        extremal_points = sorted(
            extremal_points,
            key=lambda ep: float(ep.get("distance_value", 0.0)),
            reverse=True,
        )[: args.max_extremal_points]

    positions = np.array([ep["position"] for ep in extremal_points], dtype=np.float32)
    ids = np.array(
        [int(ep.get("id", idx + 1)) for idx, ep in enumerate(extremal_points)],
        dtype=np.int32,
    )

    if positions.size == 0:
        raise RuntimeError("No extremal points detected; check vessel mask quality.")

    distance_backend = stage1.get("distance_backend", "unknown")
    mps_enabled = "metal" in str(distance_backend).lower()
    print(
        "EDT backend requested: "
        f"{args.edt_backend}; used: {distance_backend}; MPS: {'on' if mps_enabled else 'off'}"
    )
    print(f"Extremal max-filter backend used: {stage2.get('max_filter_backend', 'unknown')}")

    np.savez_compressed(
        work_file,
        distance_map=stage1["distance_map"].astype(np.float32),
        extremal_positions=positions,
        extremal_ids=ids,
        offset=np.array(offset, dtype=np.int32),
        scale=np.array([scale], dtype=np.int32),
        mask_shape=np.array(mask.shape, dtype=np.int32),
    )

    meta = {
        "input": str(input_path),
        "distance_backend": distance_backend,
        "edt_backend_requested": str(args.edt_backend),
        "edt_backend_used": distance_backend,
        "mps_enabled": bool(mps_enabled),
        "no_cpu_edt": bool(args.no_cpu_edt),
        "max_filter_backend_requested": str(args.max_filter_backend),
        "max_filter_backend_used": stage2.get("max_filter_backend", "unknown"),
        "num_extremal_points": int(len(positions)),
        "offset": list(offset),
        "scale": int(scale),
    }
    meta_path = work_file.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"EDT stage complete. Saved: {work_file}")
    return work_file


def _run_trace_stage(args: argparse.Namespace) -> None:
    from python.analysis.centerline_antiga_2008.stage4_eikonal import extract_centerlines_via_eikonal
    from python.analysis.centerline_antiga_2008.stage5_radius import compute_radii
    from python.analysis.centerline_antiga_2008.stage6_bifurcations import detect_bifurcations
    from python.analysis.centerline_antiga_2008.stage7_graph import build_centerline_graph, export_graph

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_file = Path(args.work_file) if args.work_file else output_dir / "centerline_stage12.npz"

    data = np.load(work_file)
    distance_map = data["distance_map"]
    positions = data["extremal_positions"]
    ids = data["extremal_ids"] if "extremal_ids" in data else np.arange(len(positions)) + 1

    extremal_points = []
    for idx, pos in enumerate(positions):
        extremal_points.append({
            "id": int(ids[idx]),
            "position": pos,
        })

    gpu_backend = args.device if args.device in ("mps", "cuda") else None
    stage4 = extract_centerlines_via_eikonal(
        distance_map=distance_map,
        extremal_points=extremal_points,
        step_size=args.step_size,
        max_iterations=args.max_iterations,
        gpu_backend=gpu_backend,
        k_nearest=args.k_nearest,
        max_pair_distance=args.max_pair_distance,
    )

    stage5 = compute_radii(
        centerlines=stage4["centerlines"],
        distance_map=distance_map,
    )

    offset = tuple(data["offset"].tolist())
    scale = int(data["scale"][0])
    if scale != 1 or offset != (0, 0, 0):
        stage4["centerlines"] = _rescale_centerlines(stage4["centerlines"], offset, scale)
        stage5["centerlines"] = _rescale_centerlines(stage5["centerlines"], offset, scale)

    stage6 = detect_bifurcations(
        stage5["centerlines"],
        contact_distance_threshold=args.contact_distance_threshold,
    )

    stage7 = build_centerline_graph(
        stage5["centerlines"],
        stage6,
    )

    centerline_dir = output_dir / "centerline"
    centerline_dir.mkdir(parents=True, exist_ok=True)
    export_graph(stage7, centerline_dir, basename="centerline")

    if args.export_centerline_mask:
        ref_img = nib.load(str(input_path))
        mask_path = centerline_dir / "centerline_mask.nii.gz"
        voxels = _save_centerline_mask(
            stage5["centerlines"],
            ref_img,
            mask_path,
            step=args.centerline_step,
            dilation_iters=args.centerline_dilation,
        )
        print(f"Centerline mask saved ({voxels:,} voxels): {mask_path}")

    print(f"Trace stage complete. Outputs in: {centerline_dir}")


def _run_full(args: argparse.Namespace) -> None:
    work_file = Path(args.work_file) if args.work_file else Path(args.output) / "centerline_stage12.npz"

    edt_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "edt",
        "--input",
        args.input,
        "--output",
        args.output,
        "--work-file",
        str(work_file),
        "--crop-margin-vox",
        str(args.crop_margin_vox),
        "--downsample-factor",
        str(args.downsample_factor),
        "--min-component-size",
        str(args.min_component_size),
        "--min-distance-value",
        str(args.min_distance_value),
        "--erosion-iterations",
        str(args.erosion_iterations),
        "--dilation-iterations",
        str(args.dilation_iterations),
        "--thick-component-max-radius",
        str(args.thick_component_max_radius),
        "--edt-backend",
        args.edt_backend,
        "--max-filter-backend",
        args.max_filter_backend,
    ]

    if args.no_cpu_edt:
        edt_cmd.append("--no-cpu-edt")
    if args.max_extremal_points:
        edt_cmd.extend(["--max-extremal-points", str(args.max_extremal_points)])
    if args.no_progress:
        edt_cmd.append("--no-progress")

    subprocess.run(edt_cmd, check=True)

    trace_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "trace",
        "--input",
        args.input,
        "--output",
        args.output,
        "--work-file",
        str(work_file),
        "--device",
        args.device,
        "--step-size",
        str(args.step_size),
        "--max-iterations",
        str(args.max_iterations),
        "--contact-distance-threshold",
        str(args.contact_distance_threshold),
        "--k-nearest",
        str(args.k_nearest) if args.k_nearest is not None else "0",
        "--max-pair-distance",
        str(args.max_pair_distance) if args.max_pair_distance is not None else "0",
        "--centerline-step",
        str(args.centerline_step),
        "--centerline-dilation",
        str(args.centerline_dilation),
    ]

    if args.export_centerline_mask:
        trace_cmd.append("--export-centerline-mask")

    subprocess.run(trace_cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split MPS centerline extraction (Julia EDT + torch MPS)")
    p.add_argument("--input", required=True, help="Input vessel mask NIfTI (.nii/.nii.gz)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--stage", choices=["full", "edt", "trace"], default="full")
    p.add_argument("--work-file", default=None, help="Path for stage-1/2 intermediate .npz")

    p.add_argument("--crop-margin-vox", type=int, default=8, help="Crop margin (voxels)")
    p.add_argument("--downsample-factor", type=int, default=1, help="Downsample factor for EDT/path tracing")

    p.add_argument("--min-component-size", type=int, default=300, help="Minimum connected component size")
    p.add_argument("--min-distance-value", type=float, default=1.5, help="Minimum distance value for extremal points")
    p.add_argument("--max-extremal-points", type=int, default=None, help="Cap extremal points (keep largest radii)")
    p.add_argument("--erosion-iterations", type=int, default=1, help="Erosion iterations in Stage1 cleaning")
    p.add_argument("--dilation-iterations", type=int, default=1, help="Dilation iterations in Stage1 cleaning")
    p.add_argument("--thick-component-max-radius", type=float, default=0.0, help="Remove overly thick components (mm), 0=disable")

    p.add_argument("--edt-backend", type=str, default="metal", choices=["auto", "metal", "cuda"], help="EDT backend (metal uses distance_transforms)")
    p.add_argument("--no-cpu-edt", action="store_true", help="Disable CPU EDT fallback")
    p.add_argument(
        "--max-filter-backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Max-filter backend for extremal detection",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    p.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device for path tracing")
    p.add_argument("--step-size", type=float, default=0.5, help="Gradient descent step size")
    p.add_argument("--max-iterations", type=int, default=8000, help="Max iterations per path")
    p.add_argument("--contact-distance-threshold", type=float, default=1.5, help="Bifurcation contact threshold (mm)")
    p.add_argument("--k-nearest", type=int, default=None, help="Limit tracing to k nearest extremal points")
    p.add_argument("--max-pair-distance", type=float, default=None, help="Limit tracing to pairs within this distance (voxels)")

    p.add_argument("--export-centerline-mask", action="store_true", help="Write centerline_mask.nii.gz")
    p.add_argument("--centerline-step", type=float, default=0.5, help="Interpolation step for mask rasterization")
    p.add_argument("--centerline-dilation", type=int, default=0, help="Optional dilation iterations on centerline mask")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "edt":
        _run_edt_stage(args)
    elif args.stage == "trace":
        _run_trace_stage(args)
    else:
        _run_full(args)


if __name__ == "__main__":
    main()
