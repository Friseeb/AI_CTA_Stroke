"""nnU-Net 3D full-resolution inference on numpy + onnxruntime (no torch, no nnunetv2).

Runs an nnU-Net model that was exported to ONNX together with a ``config.json`` describing its
plans (target spacing, CT normalisation statistics, patch size, labels, training orientation).
The HolOrama whole-heart CT model (``external/holorama/ct_seg_u2net_small.onnx``) is such an
export; see ``fetch_holorama_weights.py``.

The steps reproduce ``nnUNetPredictor.predict_single_npy_array`` for a 3d_fullres configuration
with identity ``transpose_forward``:

1. bring the (z, y, x) volume into the orientation the model was trained in
2. crop to the bounding box of the non-zero region (``crop_to_nonzero``)
3. CT normalisation: clip to the dataset's 0.5 / 99.5 percentiles, z-score with its mean / std
4. resample to the model's target spacing (order 3; separate z only when anisotropic)
5. sliding window over the padded volume with nnU-Net's Gaussian importance weighting
6. resample the logits back to the cropped shape (order 1) and argmax
7. undo the crop and the reorientation

Provenance of this file
-----------------------
Adapted from HolOrama ``src/segmentation/onnx_inference.py`` and ``src/segmentation/utils.py``
(https://github.com/AI-in-Cardiovascular-Medicine/HolOrama, MIT License,
Copyright (c) 2023 Sebastian Balzer), commit e6ce0460249be20df058cd204df332f9e01aa594.
HolOrama verified its re-implementation against ``nnUNetPredictor`` on its own test data; the
only intended numerical difference is float32 instead of float16 logit accumulation.
Changes here: the IVUS 2D path, Qt coupling, DirectML/float16 auto-selection and the pandas
dependency were removed; the model configuration became a dataclass; CoreML was added as an
optional provider; a memory estimate was added for the caller's guard.

Several functions are copied or adapted from MIC-DKFZ projects under the Apache License 2.0
(Copyright Division of Medical Image Computing, German Cancer Research Center (DKFZ)):
``compute_new_shape``, ``get_lowres_axis``, ``get_do_separate_z``, ``resample_data_to_shape``
(nnunetv2/preprocessing/resampling/default_resampling.py), ``compute_steps``, ``compute_gaussian``
(nnunetv2/inference/sliding_window_prediction.py) and ``ct_normalization``
(nnunetv2/preprocessing/normalization/default_normalization_schemes.py).
"""
from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import resize

__all__ = [
    "ModelConfig",
    "OnnxNnUNet3D",
    "SegmentationUnavailable",
    "compute_gaussian",
    "compute_new_shape",
    "compute_steps",
    "ct_normalization",
    "estimate_inference_bytes",
    "network_activation_bytes",
    "nonzero_bbox",
    "physical_memory_bytes",
    "reorient_array",
    "resample_data_to_shape",
    "resample_logits_to_labels",
]

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
DEVICES = ("auto", *PROVIDERS)  # 'auto': cuda, then directml, else cpu (CoreML only on request)
_AUTO_ORDER = ("cuda", "directml")
_OOM_MARKERS = ("failed to allocate memory", "out of memory", "e_outofmemory", "bad_alloc")


class SegmentationUnavailable(RuntimeError):
    """The model cannot run on this installation (runtime missing, model/config missing or unreadable,
    requested device not working). The message is meant for the user as is."""


@dataclass(frozen=True)
class ModelConfig:
    """What the runtime needs to know about an exported model: its ``config.json`` next to the
    ``.onnx`` file (plans) plus the metadata stored in the ONNX file by HolOrama's export script."""

    patch_size: tuple[int, int, int]
    spacing: tuple[float, float, float]  # target spacing, (dz, dy, dx) in mm
    mean: float
    std: float
    percentile_00_5: float
    percentile_99_5: float
    labels: dict[str, int]  # name -> value, includes background 0
    orientation: str = "LAS"  # SimpleITK/DICOM code of the training data (letters in x, y, z order)
    tile_step_size: float = 0.5
    batch_size: int = 1
    mirror_axes: tuple[int, ...] = ()
    source: str = ""  # free text from the ONNX metadata (trainer / plans / fold)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return max(int(v) for v in self.labels.values()) + 1

    @property
    def value_to_name(self) -> dict[int, str]:
        return {int(v): k for k, v in self.labels.items()}

    @classmethod
    def from_files(cls, config_path: Path, onnx_metadata: dict[str, str] | None = None) -> "ModelConfig":
        try:
            raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SegmentationUnavailable(f"The model's config.json is unreadable: {config_path}\n{e}") from e
        required = ("spacing", "mean", "std", "percentile_00_5", "percentile_99_5", "patch_size", "labels")
        missing = [k for k in required if k not in raw]
        if missing:
            raise SegmentationUnavailable(f"The model's config.json lacks {missing}: {config_path}")
        meta = onnx_metadata or {}
        mirror = tuple(int(a) for a in meta.get("mirror_axes", "").split(",") if a.strip() != "")
        known = set(required) | {"orientation", "tile_step_size", "batch_size"}
        return cls(
            patch_size=tuple(int(v) for v in raw["patch_size"]),  # type: ignore[arg-type]
            spacing=tuple(float(v) for v in raw["spacing"]),  # type: ignore[arg-type]
            mean=float(raw["mean"]),
            std=float(raw["std"]),
            percentile_00_5=float(raw["percentile_00_5"]),
            percentile_99_5=float(raw["percentile_99_5"]),
            labels={str(k): int(v) for k, v in raw["labels"].items()},
            orientation=str(raw.get("orientation", "LAS")).upper(),
            tile_step_size=float(raw.get("tile_step_size", 0.5)),
            batch_size=max(1, int(raw.get("batch_size", 1))),
            mirror_axes=mirror,
            source=str(meta.get("source", "")),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_size": list(self.patch_size),
            "spacing_zyx_mm": list(self.spacing),
            "mean": self.mean,
            "std": self.std,
            "percentile_00_5": self.percentile_00_5,
            "percentile_99_5": self.percentile_99_5,
            "labels": dict(self.labels),
            "orientation": self.orientation,
            "tile_step_size": self.tile_step_size,
            "batch_size": self.batch_size,
            "mirror_axes": list(self.mirror_axes),
            "source": self.source,
        }


class OnnxNnUNet3D:
    """One exported nnU-Net 3D model. ``predict`` takes a (z, y, x) HU volume with its spacing and
    returns a uint8 label volume on the same grid."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        device: str = "cpu",
        precision: str = "fp32",
        num_threads: int = 0,
        tile_step_size: float | None = None,
        batch_size: int | None = None,
        use_mirroring: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = str(device or "cpu").lower()
        self.precision = str(precision or "fp32").lower()
        self.num_threads = int(num_threads or 0)
        self._tile_step_size = tile_step_size
        self._batch_size = batch_size
        self.use_mirroring = use_mirroring
        self.log = log or (lambda msg: None)
        self._session = None
        self._config: ModelConfig | None = None
        self.active_device = ""
        self.active_precision = ""
        self.active_model_path: Path = self.model_path

    # -- loading -------------------------------------------------------------------------------
    @property
    def config(self) -> ModelConfig:
        self.load()
        assert self._config is not None
        return self._config

    @property
    def tile_step_size(self) -> float:
        return self._tile_step_size if self._tile_step_size is not None else self.config.tile_step_size

    @property
    def batch_size(self) -> int:
        return max(1, int(self._batch_size if self._batch_size is not None else self.config.batch_size))

    def load(self) -> None:
        if self._session is not None:
            return
        if self.device not in DEVICES:
            raise SegmentationUnavailable(f"device must be one of {', '.join(DEVICES)}, not {self.device!r}")
        if self.precision not in ("fp32", "fp16"):
            raise SegmentationUnavailable(f"precision must be 'fp32' or 'fp16', not {self.precision!r}")
        if not self.model_path.is_file():
            raise SegmentationUnavailable(f"Model not found: {self.model_path}")
        config_path = self.model_path.with_name("config.json")
        if not config_path.is_file():
            raise SegmentationUnavailable(f"The model's config.json was not found next to it: {config_path}")
        ort = _import_onnxruntime()
        session = self._open_session(ort)
        meta = session.get_modelmeta().custom_metadata_map
        config = ModelConfig.from_files(config_path, meta)
        if "num_classes" in meta and int(meta["num_classes"]) != config.num_classes:
            raise SegmentationUnavailable(
                f"{self.model_path.name} has {meta['num_classes']} classes but config.json lists "
                f"{config.num_classes} label values"
            )
        if "patch_size" in meta:
            meta_patch = tuple(int(s) for s in meta["patch_size"].split(","))
            if meta_patch != config.patch_size:
                raise SegmentationUnavailable(
                    f"patch_size differs between the ONNX metadata {meta_patch} and config.json {config.patch_size}"
                )
        self._config = config
        self._session = session
        self.log(
            f"model {self.active_model_path.name} loaded on {self.active_device} ({self.active_precision}); "
            f"patch {config.patch_size}, spacing {config.spacing} mm, {config.num_classes} classes"
        )

    def _open_session(self, ort):
        candidates: list[str]
        if self.device == "auto":
            candidates = [d for d in _AUTO_ORDER if PROVIDERS[d] in ort.get_available_providers()] + ["cpu"]
        else:
            candidates = [self.device]
        errors: list[str] = []
        for device in candidates:
            provider = PROVIDERS[device]
            if provider not in ort.get_available_providers():
                errors.append(f"{device}: {provider} is not available in this onnxruntime build")
                continue
            model_path = self.model_path
            fp16 = self.precision == "fp16" and device != "cpu"
            if fp16:
                candidate = self.model_path.with_name(f"{self.model_path.stem}_fp16{self.model_path.suffix}")
                if candidate.is_file():
                    model_path = candidate
                else:
                    self.log(f"precision fp16 requested but {candidate.name} is missing, using fp32")
                    fp16 = False
            options = ort.SessionOptions()
            options.log_severity_level = 3
            if self.num_threads:
                options.intra_op_num_threads = self.num_threads
            providers: list = [provider] if device == "cpu" else [provider, "CPUExecutionProvider"]
            if device == "coreml":
                providers[0] = (provider, {"ModelFormat": "MLProgram"})
            try:
                session = ort.InferenceSession(str(model_path), options, providers=providers)
            except Exception as e:  # noqa: BLE001 - reported per device, next candidate is tried
                errors.append(f"{device}: could not create a session ({str(e).splitlines()[0][:200]})")
                continue
            if session.get_providers()[0] != provider:
                errors.append(f"{device}: onnxruntime fell back to {session.get_providers()[0]}")
                continue
            self.active_device = device
            self.active_precision = "fp16" if fp16 else "fp32"
            self.active_model_path = model_path
            if errors:
                self.log("device fallback: " + "; ".join(errors))
            return session
        raise SegmentationUnavailable("No usable inference device:\n" + "\n".join(errors))

    def describe(self) -> dict[str, Any]:
        """Model provenance for a manifest."""
        self.load()
        return {
            "model_path": str(self.active_model_path.resolve()),
            "model_sha256": _sha256(self.active_model_path),
            "config_path": str(self.model_path.with_name("config.json").resolve()),
            "config_sha256": _sha256(self.model_path.with_name("config.json")),
            "device": self.active_device,
            "precision": self.active_precision,
            "num_threads": self.num_threads,
            "tile_step_size": self.tile_step_size,
            "batch_size": self.batch_size,
            "use_mirroring": bool(self.use_mirroring and self.config.mirror_axes),
            "config": self.config.to_dict(),
        }

    # -- inference -----------------------------------------------------------------------------
    def predict(
        self,
        volume: np.ndarray,
        spacing: Sequence[float],
        orientation: str | None = None,
        progress_cb: Callable[[int, int], bool | None] | None = None,
    ) -> np.ndarray | None:
        """Segment a (z, y, x) volume in Hounsfield units with voxel ``spacing`` (dz, dy, dx) in mm.
        ``orientation`` is the volume's SimpleITK/DICOM code (default: the model's own, i.e. the
        caller already oriented it). Returns a uint8 label volume of ``volume.shape``, or None if
        ``progress_cb(tiles_done, tiles_total)`` returned False."""
        self.load()
        config = self.config
        if volume.ndim != 3:
            raise ValueError(f"Expected a (z, y, x) volume, got shape {volume.shape}")
        if len(spacing) != 3:
            raise ValueError(f"spacing must be (dz, dy, dx), got {spacing}")
        orientation = (orientation or config.orientation).upper()
        try:
            return self._predict(volume, [float(s) for s in spacing], orientation, progress_cb)
        except Exception as e:
            if isinstance(e, MemoryError) or any(m in str(e).lower() for m in _OOM_MARKERS):
                raise MemoryError(f"Out of memory on {self.active_device} running {self.model_path.name}") from e
            raise

    def _predict(self, app_volume, spacing, orientation, progress_cb):
        config = self.config
        volume, perm = reorient_array(app_volume, orientation, config.orientation)
        original_spacing = [spacing[p] for p in perm]
        target_spacing = list(config.spacing)

        tic = time.perf_counter()
        crop = nonzero_bbox(volume)
        data = volume[crop][None].astype(np.float32)  # (1, z, y, x)
        shape_after_cropping = data.shape[1:]
        data = ct_normalization(data, config.mean, config.std, config.percentile_00_5, config.percentile_99_5)
        new_shape = compute_new_shape(shape_after_cropping, original_spacing, target_spacing)
        data = resample_data_to_shape(data, new_shape, original_spacing, target_spacing, order=3, order_z=0)
        self.log(
            f"preprocessing {time.perf_counter() - tic:.1f} s: {tuple(volume.shape)} at "
            f"{[round(s, 3) for s in original_spacing]} mm -> {tuple(data.shape[1:])} at {target_spacing} mm"
        )

        tic = time.perf_counter()
        logits = self._sliding_window_3d(data[0], progress_cb)
        if logits is None:
            return None
        del data
        self.log(f"sliding window {time.perf_counter() - tic:.1f} s")

        tic = time.perf_counter()
        labels = resample_logits_to_labels(logits, shape_after_cropping, target_spacing, original_spacing)
        del logits
        out = np.zeros(volume.shape, dtype=np.uint8)
        out[crop] = labels
        out, _ = reorient_array(out, config.orientation, orientation)
        out = np.ascontiguousarray(out)
        assert out.shape == app_volume.shape
        self.log(f"postprocessing {time.perf_counter() - tic:.1f} s")
        return out

    def _sliding_window_3d(self, image, progress_cb):
        """nnU-Net's predict_sliding_window_return_logits for one (z, y, x) image: centre-pad to at
        least the patch size, run every tile, weight its logits by the Gaussian importance map,
        divide by the summed weights, undo the padding. Returns (classes, z, y, x) float32 logits."""
        config = self.config
        patch = config.patch_size
        padded_shape = tuple(max(p, s) for p, s in zip(patch, image.shape))
        pad = [((p - s) // 2, (p - s) - (p - s) // 2) for p, s in zip(padded_shape, image.shape)]
        if any(lo or hi for lo, hi in pad):
            image = np.pad(image, pad, mode="constant", constant_values=0)
        unpad = tuple(slice(lo, size - hi) for (lo, hi), size in zip(pad, padded_shape))
        steps = compute_steps(padded_shape, patch, self.tile_step_size)
        tile_slicers = [tuple(slice(s, s + p) for s, p in zip(start, patch)) for start in itertools.product(*steps)]
        gaussian = compute_gaussian(patch)
        logits = np.zeros((config.num_classes, *padded_shape), dtype=np.float32)
        weight = np.zeros(padded_shape, dtype=np.float32)

        batch_size = self.batch_size
        batch = np.zeros((batch_size, 1, *patch), dtype=np.float32)
        total = len(tile_slicers)
        self.log(f"{total} tiles of {patch} (step {self.tile_step_size}) over {padded_shape}")
        tic = time.perf_counter()
        for start in range(0, total, batch_size):
            chunk = tile_slicers[start : start + batch_size]
            batch[:] = 0
            for i, sl in enumerate(chunk):
                batch[i, 0] = image[sl]
            tile_logits = self._run(batch)
            for i, sl in enumerate(chunk):
                logits[(slice(None), *sl)] += tile_logits[i] * gaussian
                weight[sl] += gaussian
            done = start + len(chunk)
            self.log(f"  tile {done}/{total} ({time.perf_counter() - tic:.0f} s elapsed)")
            if progress_cb is not None and progress_cb(done, total) is False:
                return None
        logits /= weight
        return logits[(slice(None), *unpad)]

    def _run(self, tiles: np.ndarray) -> np.ndarray:
        """Logits for a (n, 1, pz, py, px) stack of tiles, mirror-averaged when the model allows it."""
        assert self._session is not None
        combos = self._mirror_combinations()
        result = None
        for axes in combos:
            batch = np.ascontiguousarray(np.flip(tiles, axes) if axes else tiles)
            logits = self._session.run(None, {"image": batch})[0]
            logits = np.flip(logits, axes) if axes else logits
            result = logits if result is None else result + logits
        assert result is not None
        return result / len(combos)

    def _mirror_combinations(self) -> list[tuple[int, ...]]:
        axes_meta = self.config.mirror_axes
        if not self.use_mirroring or not axes_meta:
            return [()]
        axes = [a + 2 for a in axes_meta]  # (batch, channel, z, y, x)
        return [()] + [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]


# ------------------------------------------------------------------------------ nnU-Net pieces
def compute_new_shape(old_shape, old_spacing, new_spacing) -> np.ndarray:
    """Shape of an image of old_shape at old_spacing after resampling to new_spacing
    (nnunetv2.preprocessing.resampling.default_resampling.compute_new_shape, unchanged)."""
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


def get_lowres_axis(new_spacing) -> np.ndarray:
    """nnunetv2 get_lowres_axis: the axes whose spacing equals the largest one."""
    return np.where(np.array(new_spacing) / max(new_spacing) == 1)[0]


def get_do_separate_z(spacing, anisotropy_threshold: float = 3) -> bool:
    """nnunetv2 get_do_separate_z (anisotropy_threshold = nnunetv2.configuration.ANISO_THRESHOLD)."""
    return bool((np.max(spacing) / np.min(spacing)) > anisotropy_threshold)


def resample_data_to_shape(
    data: np.ndarray,
    new_shape,
    current_spacing,
    new_spacing,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: bool | None = None,
    separate_z_anisotropy_threshold: float = 3,
) -> np.ndarray:
    """Resample (c, z, y, x) image data to new_shape as nnU-Net does (resample_data_or_seg_to_shape
    + resample_data_or_seg for ``is_seg=False``): skimage ``resize`` with ``mode='edge'`` and no
    anti-aliasing, and, for anisotropic spacing, in-plane resampling followed by order-``order_z``
    interpolation along the low-resolution axis."""
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        axis = get_lowres_axis(current_spacing) if force_separate_z else None
    elif get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(current_spacing)
    elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
        do_separate_z, axis = True, get_lowres_axis(new_spacing)
    else:
        do_separate_z, axis = False, None
    if axis is not None:
        if len(axis) != 1:  # all axes equal, or two low-res axes: nnU-Net then resamples in 3D
            do_separate_z, axis = False, None
        else:
            axis = int(axis[0])

    assert data.ndim == 4, "data must be (c, z, y, x)"
    assert len(new_shape) == 3
    shape = np.array(data[0].shape)
    new_shape = np.array([int(s) for s in new_shape])
    if not np.any(shape != new_shape):
        return data
    data = data.astype(np.float32, copy=False)
    kwargs = {"mode": "edge", "anti_aliasing": False}
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=np.float32)
    if not do_separate_z:
        for c in range(data.shape[0]):
            reshaped_final[c] = resize(data[c], new_shape, order, **kwargs)
        return reshaped_final

    assert axis is not None
    new_shape_2d = np.delete(new_shape, axis)
    for c in range(data.shape[0]):
        tmp = deepcopy(new_shape)
        tmp[axis] = shape[axis]
        reshaped_here = np.zeros(tmp, dtype=np.float32)
        for slice_id in range(shape[axis]):
            sl_in = [slice(None)] * 3
            sl_in[axis] = slice_id
            reshaped_here[tuple(sl_in)] = resize(data[c][tuple(sl_in)], new_shape_2d, order, **kwargs)
        if shape[axis] != new_shape[axis]:
            rows, cols, dim = new_shape
            orig_rows, orig_cols, orig_dim = reshaped_here.shape
            map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
            map_rows = float(orig_rows) / rows * (map_rows + 0.5) - 0.5
            map_cols = float(orig_cols) / cols * (map_cols + 0.5) - 0.5
            map_dims = float(orig_dim) / dim * (map_dims + 0.5) - 0.5
            coord_map = np.array([map_rows, map_cols, map_dims])
            reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode="nearest")
        else:
            reshaped_final[c] = reshaped_here
    return reshaped_final


def resample_logits_to_labels(logits: np.ndarray, new_shape, current_spacing, new_spacing) -> np.ndarray:
    """argmax over classes of nnU-Net's resampling_fn_probabilities applied to (C, z, y, x) logits
    (order 1, order 0 along z for anisotropic spacing), as a uint8 label map of new_shape. One
    channel at a time with a running argmax (strict > keeps the lower class on ties, as np.argmax)."""
    best: np.ndarray | None = None
    labels = np.zeros(tuple(int(s) for s in new_shape), dtype=np.uint8)
    for c in range(logits.shape[0]):
        channel = resample_data_to_shape(logits[c : c + 1], new_shape, current_spacing, new_spacing, order=1, order_z=0)[0]
        if best is None:
            best = np.array(channel, dtype=np.float32, copy=True)
            continue
        better = channel > best
        labels[better] = c
        np.copyto(best, channel, where=better)
        del channel, better
    return labels


def compute_steps(image_size, tile_size, tile_step_size: float) -> list[list[int]]:
    """nnunetv2 compute_steps_for_sliding_window, condensed."""
    steps = []
    for size, tile in zip(image_size, tile_size):
        num_steps = int(np.ceil((size - tile) / (tile * tile_step_size))) + 1
        actual = (size - tile) / (num_steps - 1) if num_steps > 1 else 0
        steps.append([int(np.round(actual * i)) for i in range(num_steps)])
    return steps


def compute_gaussian(tile_size, sigma_scale: float = 1.0 / 8, value_scaling_factor: float = 10) -> np.ndarray:
    """nnU-Net's Gaussian importance map (numpy float32; value_scaling_factor 10 as nnUNetPredictor passes)."""
    tmp = np.zeros(tile_size)
    tmp[tuple(i // 2 for i in tile_size)] = 1
    gaussian = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode="constant", cval=0)
    gaussian = (gaussian / gaussian.max() * value_scaling_factor).astype(np.float32)
    gaussian[gaussian == 0] = gaussian[gaussian != 0].min()
    return gaussian


def ct_normalization(image: np.ndarray, mean_intensity, std_intensity, percentile_00_5, percentile_99_5) -> np.ndarray:
    """nnunetv2 CTNormalization.run with the intensity properties passed in."""
    image = np.clip(image, percentile_00_5, percentile_99_5)
    return (image - mean_intensity) / max(std_intensity, 1e-8)


def nonzero_bbox(volume: np.ndarray) -> tuple[slice, ...]:
    """Half-open bounding box of the non-zero region, the box nnU-Net's crop_to_nonzero crops to.
    An all-zero volume keeps its full extent."""
    mask = volume != 0
    slicer = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        slicer.append(slice(int(hits[0]), int(hits[-1]) + 1) if hits.size else slice(0, mask.shape[axis]))
    return tuple(slicer)


_OPPOSITE_DIRECTION = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def reorient_array(array: np.ndarray, source: str, target: str) -> tuple[np.ndarray, list[int]]:
    """A (z, y, x) array of an image in orientation ``source`` (SimpleITK/DICOM code such as 'LAS',
    letters in x, y, z order) as SimpleITK gives it after ``sitk.DICOMOrient(image, target)``:
    only axis permutations and flips. Also returns the axis permutation, to reorder a (dz, dy, dx)
    spacing the same way: ``[spacing[p] for p in perm]``. A view when no copy is needed."""
    source, target = source.upper(), target.upper()
    if len(source) != 3 or len(target) != 3:
        raise ValueError(f"orientation codes must have three letters, got {source!r} and {target!r}")
    perm, flips = [], []
    for k in range(3):  # numpy axis k is SimpleITK axis 2 - k
        letter = target[2 - k]
        j = next((i for i, s in enumerate(source) if s in (letter, _OPPOSITE_DIRECTION[letter])), None)
        if j is None:
            raise ValueError(f"Cannot reorient {source!r} to {target!r}")
        perm.append(2 - j)
        flips.append(source[j] != letter)
    out = np.transpose(array, perm)
    axes = tuple(k for k, flip in enumerate(flips) if flip)
    return (np.flip(out, axes) if axes else out), perm


# ------------------------------------------------------------------------------ memory guard
def estimate_inference_bytes(shape_zyx, spacing_zyx, config: ModelConfig) -> tuple[int, tuple[int, ...]]:
    """Rough peak memory of ``predict`` for a volume of this shape and spacing: the logits
    (num_classes x resampled volume, float32) dominate; the weight map, the resampled data and the
    float64 temporaries of skimage's resize add about four more volumes. Returns (bytes, resampled shape)."""
    new_shape = compute_new_shape(shape_zyx, spacing_zyx, config.spacing)
    padded = tuple(int(max(p, s)) for p, s in zip(config.patch_size, new_shape))
    voxels = int(np.prod(padded))
    return int((config.num_classes + 4) * 4 * voxels), tuple(int(s) for s in new_shape)


# Measured on the HolOrama CT model (U2Net-small, patch 128 x 256 x 224, batch 1) on an Apple M-series CPU:
# ~23 GB of onnxruntime activations on top of the arrays, i.e. ~3.2 kB per patch voxel. A fixed cost per
# tile, independent of the volume size; a crop cannot reduce it. Float16 on a GPU roughly halves it.
ACTIVATION_BYTES_PER_PATCH_VOXEL = 3200


def network_activation_bytes(config: ModelConfig, batch_size: int = 1) -> int:
    """Rough peak memory of one network call (onnxruntime activations), see ACTIVATION_BYTES_PER_PATCH_VOXEL."""
    return int(ACTIVATION_BYTES_PER_PATCH_VOXEL * batch_size * np.prod(config.patch_size))


def physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _import_onnxruntime():
    try:
        import onnxruntime  # noqa: PLC0415 - optional dependency
    except ImportError as e:
        raise SegmentationUnavailable(
            "onnxruntime is not installed in this Python environment.\n"
            "    pip install onnxruntime        # CPU / CoreML (macOS)\n"
            "    pip install onnxruntime-gpu    # CUDA 12 (Linux / Windows)"
        ) from e
    return onnxruntime


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
