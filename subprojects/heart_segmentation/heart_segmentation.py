#!/usr/bin/env python3
"""Whole-heart segmentation of a CTA with the HolOrama CT model (14 structures, one labelmap).

Input : a CTA NIfTI in Hounsfield units (any orientation) and, normally, a binary crop mask
        (e.g. TotalSegmentator ``heart.nii.gz``) that limits inference to the heart. Stroke CTAs
        run from the aortic arch to the vertex; the model was trained on cardiac fields of view
        and a full head-neck-chest volume is 10x the tiles and 10x the memory for no gain.
Model : HolOrama ``ct_seg_u2net_small.onnx`` (nnU-Net 3d_fullres, U2Net-small trainer, no
        test-time mirroring), run by ``nnunet_onnx.OnnxNnUNet3D`` on onnxruntime. Weights:
        ``external/holorama/`` (``fetch_holorama_weights.py``).
Output: ``<case>_holorama_heart_seg.nii.gz``       uint8 labelmap on the CTA grid (values: holorama_labels)
        ``structures/<structure>.nii.gz``            one binary mask per structure (TotalSegmentator style)
        ``<case>_holorama_heart_volumes.csv``       long format: volume per structure + field-of-view flags
        ``<case>_holorama_heart_labels.json``       label table (ids, names, colours) for viewers
        ``<case>_holorama_heart_qc.png``            three orthogonal slices with the overlay
        ``<case>_holorama_heart_manifest.json``     provenance: inputs (sha256), model, crop, timings

Everything is computed in the model's orientation (LAS) and written back on the CTA's own grid, so
the labelmap overlays the input in any viewer without resampling.
Research use only; the model is not a medical device.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata as _md
from pathlib import Path
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from holorama_labels import LABELS, check_against_model_config, label_table  # noqa: E402
from nnunet_onnx import (  # noqa: E402
    ModelConfig,
    OnnxNnUNet3D,
    SegmentationUnavailable,
    estimate_inference_bytes,
    network_activation_bytes,
    physical_memory_bytes,
)

__version__ = "0.1.0"
REPO = HERE.parents[1]
DEFAULT_WEIGHTS_DIR = REPO / "external" / "holorama"
MODEL_FILE = "ct_seg_u2net_small.onnx"
SUFFIX = "holorama_heart"
EDGE_FLAGS = ("touches_scan_inferior", "touches_scan_superior", "touches_scan_inplane", "touches_crop_box")


@dataclass
class Params:
    crop_margin_mm: float = 15.0  # added around the crop mask's bounding box on every side
    device: str = "cpu"  # nnunet_onnx.DEVICES; CoreML cannot run this model (5-D InstanceNorm)
    precision: str = "fp32"  # fp16 only on a GPU, uses <model>_fp16.onnx
    num_threads: int = 0  # 0 = onnxruntime default (physical cores)
    tile_step_size: float | None = None  # None = the model's config.json (0.75)
    write_structures: bool = True
    write_qc: bool = True
    max_memory_fraction: float = 0.5  # refuse when the estimated peak exceeds this share of physical RAM
    allow_large: bool = False  # ... unless this is set


# --------------------------------------------------------------------------------------- helpers
def log_print(msg: str) -> None:
    """Default logger: stdout, flushed, so a redirected batch log streams while tiles run."""
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def orientation_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(img.GetDirection())


def same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def mask_bbox(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Half-open bounding box of a boolean array, None when empty. Per-axis any(): no np.where copy."""
    slices = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        if not hits.size:
            return None
        slices.append(slice(int(hits[0]), int(hits[-1]) + 1))
    return tuple(slices)


def crop_box_from_mask(mask_img: sitk.Image, ref: sitk.Image, margin_mm: float) -> tuple[slice, ...]:
    """Bounding box (z, y, x slices on ``ref``'s grid) of a binary mask, padded by ``margin_mm``.
    The mask is brought onto ``ref``'s grid with nearest-neighbour resampling, which also absorbs
    an orientation difference; a mask on the same physical grid is unchanged."""
    if not same_grid(mask_img, ref):
        mask_img = sitk.Resample(mask_img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(mask_img) > 0
    box = mask_bbox(arr)
    if box is None:
        raise ValueError("The crop mask has no voxels on the CTA grid")
    spacing_zyx = ref.GetSpacing()[::-1]
    shape_zyx = ref.GetSize()[::-1]
    out = []
    for sl, sp, n in zip(box, spacing_zyx, shape_zyx):
        pad = int(np.ceil(margin_mm / sp))
        out.append(slice(max(sl.start - pad, 0), min(sl.stop + pad, int(n))))
    return tuple(out)


def git_commit(repo: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance never breaks the run
        return None


def package_versions(names=("numpy", "scipy", "scikit-image", "SimpleITK", "onnxruntime", "onnxruntime-gpu")) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = _md.version(n)
        except Exception:  # noqa: BLE001
            out[n] = None
    return out


def load_segmenter(weights_dir: Path | None, params: Params, log: Callable[[str], None]) -> OnnxNnUNet3D:
    weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    model_path = weights_dir / MODEL_FILE
    if not model_path.is_file():
        raise SegmentationUnavailable(
            f"HolOrama weights not found: {model_path}\n"
            f"Download them with: python {HERE / 'fetch_holorama_weights.py'}"
        )
    seg = OnnxNnUNet3D(
        model_path,
        device=params.device,
        precision=params.precision,
        num_threads=params.num_threads,
        tile_step_size=params.tile_step_size,
        log=log,
    )
    seg.load()
    check_against_model_config(seg.config.labels)
    return seg


# --------------------------------------------------------------------------------------- stats
def structure_stats(labels_sub: np.ndarray, box: tuple[slice, ...], shape_zyx, spacing_zyx, config: ModelConfig) -> list[dict]:
    """Volume per structure plus field-of-view flags. ``labels_sub`` is the cropped label array in
    the model's orientation (LAS: z index 0 is inferior, the last index superior). A structure on a
    crop face that is also a scan face is truncated by the scan; on a crop face that is not, the
    crop margin was too small."""
    voxel_ml = float(np.prod(spacing_zyx)) / 1000.0
    counts = np.bincount(labels_sub.ravel(), minlength=config.num_classes)
    faces = {
        "z0": labels_sub[0], "z1": labels_sub[-1],
        "y0": labels_sub[:, 0], "y1": labels_sub[:, -1],
        "x0": labels_sub[:, :, 0], "x1": labels_sub[:, :, -1],
    }
    scan_face = {
        "z0": box[0].start == 0, "z1": box[0].stop == shape_zyx[0],
        "y0": box[1].start == 0, "y1": box[1].stop == shape_zyx[1],
        "x0": box[2].start == 0, "x1": box[2].stop == shape_zyx[2],
    }
    face_labels = {name: set(np.unique(face).tolist()) for name, face in faces.items()}
    rows = []
    for lab in LABELS:
        on = {name: lab.id in vals for name, vals in face_labels.items()}
        rows.append({
            "label_id": lab.id,
            "structure": lab.key,
            "voxels": int(counts[lab.id]) if lab.id < counts.size else 0,
            "volume_ml": round(float(counts[lab.id]) * voxel_ml, 3) if lab.id < counts.size else 0.0,
            "present": bool(lab.id < counts.size and counts[lab.id] > 0),
            "touches_scan_inferior": bool(on["z0"] and scan_face["z0"]),
            "touches_scan_superior": bool(on["z1"] and scan_face["z1"]),
            "touches_scan_inplane": bool(any(on[f] and scan_face[f] for f in ("y0", "y1", "x0", "x1"))),
            "touches_crop_box": bool(any(on[f] and not scan_face[f] for f in faces)),
        })
    return rows


def write_volumes_csv(rows: list[dict], case_id: str, path: Path) -> None:
    cols = ["case_id", "label_id", "structure", "voxels", "volume_ml", "present", *EDGE_FLAGS]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({"case_id": case_id, **{c: r[c] for c in cols if c != "case_id"}})


# --------------------------------------------------------------------------------------- QC
def write_qc_png(cta_sub: np.ndarray, labels_sub: np.ndarray, spacing_zyx, path: Path, title: str) -> bool:
    """Axial / coronal / sagittal slices through the centroid of the segmentation (LAS array:
    radiological view, anterior up, patient right on the image's left). False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        return False
    fg = labels_sub > 0
    if fg.any():
        centre = [int(round(np.mean(idx))) for idx in np.nonzero(fg)]
    else:
        centre = [s // 2 for s in labels_sub.shape]
    lut = np.zeros((max(l.id for l in LABELS) + 1, 4), dtype=float)
    for lab in LABELS:
        lut[lab.id, :3] = lab.rgb
        lut[lab.id, 3] = 0.55
    cmap = ListedColormap(lut)
    dz, dy, dx = spacing_zyx
    views = [
        ("axial", cta_sub[centre[0]], labels_sub[centre[0]], dy / dx),
        ("coronal", cta_sub[:, centre[1], :], labels_sub[:, centre[1], :], dz / dx),
        ("sagittal", cta_sub[:, :, centre[2]], labels_sub[:, :, centre[2]], dz / dy),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (name, img, lab, aspect) in zip(axes, views):
        ax.imshow(img, cmap="gray", vmin=-200, vmax=700, origin="lower", aspect=aspect, interpolation="nearest")
        ax.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, vmin=0, vmax=lut.shape[0] - 1, origin="lower", aspect=aspect, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    present = sorted(set(np.unique(labels_sub).tolist()) - {0})
    handles = [Patch(color=lut[i, :3], label=f"{i} {next(l.name for l in LABELS if l.id == i)}") for i in present if i < lut.shape[0]]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------- main entry
def segment_case(
    cta_path: Path,
    out_dir: Path,
    case_id: str,
    params: Params | None = None,
    crop_mask_path: Path | None = None,
    segmenter: Any | None = None,
    weights_dir: Path | None = None,
    log: Callable[[str], None] = log_print,
) -> dict:
    """Segment one CTA and write every output into ``out_dir``. Returns the manifest (also written).
    ``segmenter`` may be any object with ``config``, ``predict(volume, spacing, orientation)`` and
    ``describe()`` (tests pass a stub); by default the HolOrama model is loaded from ``weights_dir``."""
    params = params or Params()
    cta_path, out_dir = Path(cta_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    seg = segmenter or load_segmenter(weights_dir, params, log)
    config: ModelConfig = seg.config
    model_code = config.orientation

    tic = time.perf_counter()
    img = sitk.ReadImage(str(cta_path))
    if img.GetNumberOfComponentsPerPixel() != 1 or img.GetDimension() != 3:
        raise ValueError(f"{cta_path.name}: expected a 3D scalar volume")
    src_code = orientation_code(img)
    oriented = sitk.DICOMOrient(img, model_code) if src_code != model_code else img
    spacing_zyx = tuple(float(s) for s in oriented.GetSpacing()[::-1])
    shape_zyx = tuple(int(s) for s in oriented.GetSize()[::-1])
    timings["read_s"] = round(time.perf_counter() - tic, 2)
    log(f"{case_id}: {cta_path.name} {shape_zyx} (z, y, x) at {[round(s, 3) for s in spacing_zyx]} mm, orientation {src_code} -> {model_code}")

    # -- crop
    if crop_mask_path is not None:
        mask_img = sitk.ReadImage(str(crop_mask_path))
        box = crop_box_from_mask(mask_img, oriented, params.crop_margin_mm)
        log(f"crop from {Path(crop_mask_path).name} + {params.crop_margin_mm} mm: z {box[0].start}:{box[0].stop}, y {box[1].start}:{box[1].stop}, x {box[2].start}:{box[2].stop}")
    else:
        box = tuple(slice(0, s) for s in shape_zyx)
        log("no crop mask: segmenting the full volume")
    sub_shape = tuple(s.stop - s.start for s in box)

    # -- memory guard
    est_bytes, resampled_shape = estimate_inference_bytes(sub_shape, spacing_zyx, config)
    net_bytes = network_activation_bytes(config, getattr(seg, "batch_size", 1))
    phys = physical_memory_bytes()
    log(
        f"inference volume {sub_shape} -> {resampled_shape} at model spacing; estimated arrays "
        f"{est_bytes / 2**30:.1f} GB + network activations ~{net_bytes / 2**30:.0f} GB (fixed per tile)"
    )
    if phys and est_bytes > params.max_memory_fraction * phys and not params.allow_large:
        raise MemoryError(
            f"Estimated array memory {est_bytes / 2**30:.1f} GB exceeds {params.max_memory_fraction:.0%} of the "
            f"{phys / 2**30:.0f} GB installed. Pass a crop mask (TotalSegmentator heart) or --allow-large."
        )

    # -- inference
    index_xyz = [box[2].start, box[1].start, box[0].start]
    size_xyz = [sub_shape[2], sub_shape[1], sub_shape[0]]
    sub_img = sitk.RegionOfInterest(oriented, size_xyz, index_xyz)
    cta_sub = sitk.GetArrayFromImage(sub_img)
    tic = time.perf_counter()
    labels_sub = seg.predict(cta_sub, spacing_zyx, orientation=model_code)
    if labels_sub is None:
        raise RuntimeError("segmentation was cancelled")
    labels_sub = np.asarray(labels_sub, dtype=np.uint8)
    timings["inference_s"] = round(time.perf_counter() - tic, 1)
    log(f"inference {timings['inference_s']} s")

    # -- back onto the CTA grid
    tic = time.perf_counter()
    full = np.zeros(shape_zyx, dtype=np.uint8)
    full[box] = labels_sub
    label_img = sitk.GetImageFromArray(full)
    label_img.CopyInformation(oriented)
    del full
    if src_code != model_code:
        label_img = sitk.DICOMOrient(label_img, src_code)
    resampled_back = False
    if not same_grid(label_img, img):
        label_img = sitk.Resample(label_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        resampled_back = True
    label_img = sitk.Cast(label_img, sitk.sitkUInt8)
    seg_path = out_dir / f"{case_id}_{SUFFIX}_seg.nii.gz"
    sitk.WriteImage(label_img, str(seg_path), True)
    outputs = {"labelmap": seg_path.name}

    if params.write_structures:
        sdir = out_dir / "structures"
        sdir.mkdir(exist_ok=True)
        src_arr = sitk.GetArrayFromImage(label_img)
        for lab in LABELS:
            m = sitk.GetImageFromArray((src_arr == lab.id).astype(np.uint8))
            m.CopyInformation(label_img)
            sitk.WriteImage(m, str(sdir / f"{lab.key}.nii.gz"), True)
        del src_arr
        outputs["structures_dir"] = "structures"
    timings["write_s"] = round(time.perf_counter() - tic, 1)

    # -- stats, labels, QC
    rows = structure_stats(labels_sub, box, shape_zyx, spacing_zyx, config)
    vol_path = out_dir / f"{case_id}_{SUFFIX}_volumes.csv"
    write_volumes_csv(rows, case_id, vol_path)
    outputs["volumes_csv"] = vol_path.name
    labels_path = out_dir / f"{case_id}_{SUFFIX}_labels.json"
    labels_path.write_text(json.dumps({"labelmap": seg_path.name, "labels": label_table()}, indent=2))
    outputs["labels_json"] = labels_path.name
    if params.write_qc:
        qc_path = out_dir / f"{case_id}_{SUFFIX}_qc.png"
        title = f"{case_id} | HolOrama heart | {timings['inference_s']} s on {getattr(seg, 'active_device', '?')}"
        if write_qc_png(cta_sub, labels_sub, spacing_zyx, qc_path, title):
            outputs["qc_png"] = qc_path.name
    present = [r["structure"] for r in rows if r["present"]]
    truncated = [r["structure"] for r in rows if r["present"] and (r["touches_scan_inferior"] or r["touches_scan_superior"] or r["touches_scan_inplane"])]
    crop_tight = [r["structure"] for r in rows if r["present"] and r["touches_crop_box"]]
    log(f"{len(present)}/14 structures present; truncated by the scan: {truncated or 'none'}; touching the crop box: {crop_tight or 'none'}")

    timings["total_s"] = round(time.perf_counter() - t_all, 1)
    manifest = {
        "module": "heart_segmentation",
        "module_version": __version__,
        "pipeline": "holorama_heart",
        "case_id": case_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "cta": str(cta_path.resolve()),
            "cta_sha256": sha256(cta_path),
            "crop_mask": str(Path(crop_mask_path).resolve()) if crop_mask_path is not None else None,
            "crop_mask_sha256": sha256(crop_mask_path) if crop_mask_path is not None else None,
        },
        "cta": {
            "source_orientation": src_code,
            "size_xyz": list(img.GetSize()),
            "spacing_xyz_mm": [float(s) for s in img.GetSpacing()],
        },
        "model_orientation": model_code,
        "crop_box_zyx": [[s.start, s.stop] for s in box],
        "crop_extent_mm_zyx": [round((s.stop - s.start) * sp, 1) for s, sp in zip(box, spacing_zyx)],
        "resampled_shape_zyx": list(resampled_shape),
        "estimated_array_gb": round(est_bytes / 2**30, 2),
        "estimated_network_activation_gb": round(net_bytes / 2**30, 1),
        "resampled_back_to_source_grid": resampled_back,
        "params": asdict(params),
        "model": seg.describe(),
        "structures_present": present,
        "structures_truncated_by_scan": truncated,
        "structures_touching_crop_box": crop_tight,
        "timings": timings,
        "outputs": outputs,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(REPO),
            "packages": package_versions(),
        },
    }
    (out_dir / f"{case_id}_{SUFFIX}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"done in {timings['total_s']} s -> {out_dir}")
    return manifest
