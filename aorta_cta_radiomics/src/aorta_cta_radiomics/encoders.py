"""Optional CT-trained encoder embeddings for local aortic patches.

The core pipeline remains handcrafted and deterministic. This module adds an
optional research extension for CT foundation-model embeddings around
predefined aortic regions, such as 500 HU calcification. Irregularity/adaptive
ROI patch sources have been removed from the active pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .features import feature_row


def extract_encoder_features_from_masks(
    image: np.ndarray,
    source_masks: dict[str, np.ndarray],
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    config: dict[str, Any],
    software_version: str = __version__,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract optional CT encoder embeddings and return features plus patch manifest."""
    encoder_config = config.get("encoders", {})
    if not bool(encoder_config.get("enabled", False)):
        return pd.DataFrame(), pd.DataFrame()

    patch_manifest = build_patch_manifest(
        source_masks=source_masks,
        spacing_xyz=spacing_xyz,
        case_id=case_id,
        max_patches_per_source=int(encoder_config.get("max_patches_per_source", 8)),
        patch_size_mm_zyx=tuple(float(v) for v in encoder_config.get("patch_size_mm_zyx", [48, 96, 96])),
    )
    if patch_manifest.empty:
        return pd.DataFrame(), patch_manifest

    backend_features: list[pd.DataFrame] = []
    for backend_config in _encoder_backend_configs(encoder_config):
        try:
            backend_features.append(
                _extract_embeddings_for_backend(
                    image=image,
                    patch_manifest=patch_manifest,
                    encoder_config=backend_config,
                    case_id=case_id,
                    software_version=software_version,
                )
            )
        except Exception as exc:
            backend_features.append(_encoder_error_frame(case_id, _backend_label(backend_config), str(exc), software_version))

    if not backend_features:
        return pd.DataFrame(), patch_manifest
    return pd.concat(backend_features, ignore_index=True), patch_manifest


def _encoder_backend_configs(encoder_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy single-backend config and new multi-backend config."""
    shared = {
        key: value
        for key, value in encoder_config.items()
        if key not in {"backend", "backends", "model_name", "name", "enabled"}
    }
    raw_backends = encoder_config.get("backends")
    if "backend" in encoder_config or "model_name" in encoder_config or not raw_backends:
        legacy = {
            **shared,
            "backend": encoder_config.get("backend", "tap_ct_hf"),
            "model_name": encoder_config.get("model_name", "fomofo/tap-ct-b-3d"),
            "name": encoder_config.get("name", encoder_config.get("backend", "tap_ct_hf")),
        }
        return [legacy]

    normalized: list[dict[str, Any]] = []
    if isinstance(raw_backends, dict):
        iterable = []
        for name, values in raw_backends.items():
            values = values or {}
            if not isinstance(values, dict):
                raise TypeError("Each encoder backend entry must be a mapping.")
            iterable.append({"name": name, **values})
    else:
        iterable = list(raw_backends)

    for item in iterable:
        if not isinstance(item, dict):
            raise TypeError("Each encoder backend entry must be a mapping.")
        if not bool(item.get("enabled", True)):
            continue
        merged = {**shared, **item}
        merged.setdefault("backend", merged.get("name", "tap_ct_hf"))
        merged.setdefault("name", merged["backend"])
        normalized.append(merged)
    return normalized


def _extract_embeddings_for_backend(
    image: np.ndarray,
    patch_manifest: pd.DataFrame,
    encoder_config: dict[str, Any],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    backend = str(encoder_config.get("backend", "tap_ct_hf"))
    if backend == "tap_ct_hf":
        return _extract_tap_ct_embeddings(
            image=image,
            patch_manifest=patch_manifest,
            encoder_config=encoder_config,
            case_id=case_id,
            software_version=software_version,
        )
    if backend == "ct_fm_lighter_zoo":
        return _extract_ct_fm_lighter_zoo_embeddings(
            image=image,
            patch_manifest=patch_manifest,
            encoder_config=encoder_config,
            case_id=case_id,
            software_version=software_version,
        )
    if backend in {"hf_auto_model_3d", "voxelfm_hf"}:
        return _extract_hf_auto_model_3d_embeddings(
            image=image,
            patch_manifest=patch_manifest,
            encoder_config=encoder_config,
            case_id=case_id,
            software_version=software_version,
        )
    raise ValueError(f"Unsupported encoder backend: {backend}")


def build_patch_manifest(
    source_masks: dict[str, np.ndarray],
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    max_patches_per_source: int,
    patch_size_mm_zyx: tuple[float, float, float],
) -> pd.DataFrame:
    """Create deterministic patch centers from occupied source-mask voxels."""
    rows: list[dict[str, object]] = []
    patch_id = 0
    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    half_size_voxels = np.ceil(np.asarray(patch_size_mm_zyx, dtype=float) / spacing_zyx / 2.0).astype(int)

    for source_name, source_mask in source_masks.items():
        if source_name == "wall_surface_grid":
            wall_manifest = build_wall_surface_patch_manifest(
                aorta_mask=source_mask,
                spacing_xyz=spacing_xyz,
                case_id=case_id,
                patch_size_mm_zyx=patch_size_mm_zyx,
                axial_step_mm=5.0,
                angular_bins=24,
                max_patches=max_patches_per_source,
                starting_patch_index=patch_id,
            )
            rows.extend(wall_manifest.to_dict("records"))
            patch_id += len(wall_manifest)
            continue

        coords = np.argwhere(np.asarray(source_mask, dtype=bool))
        if coords.size == 0:
            continue
        coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))]
        chunks = np.array_split(coords, min(max_patches_per_source, len(coords)))
        for chunk_index, chunk in enumerate(chunks):
            if chunk.size == 0:
                continue
            center_zyx = np.rint(np.median(chunk, axis=0)).astype(int)
            start = np.maximum(center_zyx - half_size_voxels, 0)
            stop = center_zyx + half_size_voxels + 1
            rows.append(
                {
                    "case_id": case_id,
                    "patch_id": f"{case_id}_patch_{patch_id:04d}",
                    "patch_index": patch_id,
                    "source": source_name,
                    "source_chunk_index": int(chunk_index),
                    "source_voxel_count": int(len(chunk)),
                    "center_z_voxel": int(center_zyx[0]),
                    "center_y_voxel": int(center_zyx[1]),
                    "center_x_voxel": int(center_zyx[2]),
                    "center_z_mm": float(center_zyx[0] * spacing_xyz[2]),
                    "center_y_mm": float(center_zyx[1] * spacing_xyz[1]),
                    "center_x_mm": float(center_zyx[2] * spacing_xyz[0]),
                    "start_z_voxel": int(start[0]),
                    "start_y_voxel": int(start[1]),
                    "start_x_voxel": int(start[2]),
                    "stop_z_voxel": int(stop[0]),
                    "stop_y_voxel": int(stop[1]),
                    "stop_x_voxel": int(stop[2]),
                    "patch_size_z_mm": float(patch_size_mm_zyx[0]),
                    "patch_size_y_mm": float(patch_size_mm_zyx[1]),
                    "patch_size_x_mm": float(patch_size_mm_zyx[2]),
                }
            )
            patch_id += 1
    return pd.DataFrame(rows)


def build_wall_surface_patch_manifest(
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    patch_size_mm_zyx: tuple[float, float, float],
    axial_step_mm: float = 5.0,
    angular_bins: int = 24,
    max_patches: int = 64,
    starting_patch_index: int = 0,
) -> pd.DataFrame:
    """Sample small local patches on the aortic wall surface.

    The goal is to avoid treating the arch as an irregular object. Each patch is
    centered on a local wall sector within one axial component.
    """
    binary = np.asarray(aorta_mask, dtype=bool)
    if not binary.any():
        return pd.DataFrame()

    spacing_zyx = np.asarray([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)
    half_size_voxels = np.ceil(np.asarray(patch_size_mm_zyx, dtype=float) / spacing_zyx / 2.0).astype(int)
    z_step = max(1, int(round(axial_step_mm / spacing_xyz[2])))
    occupied_slices = np.where(binary.any(axis=(1, 2)))[0]
    sampled_slices = occupied_slices[::z_step]

    rows: list[dict[str, object]] = []
    patch_index = starting_patch_index
    for z in sampled_slices:
        labels, n_labels = _label_2d(binary[z])
        for component_rank, component_label in enumerate(_component_labels_by_size(labels, n_labels, max_components=4)):
            component = labels == component_label
            boundary = component ^ _binary_erosion_2d(component)
            bys, bxs = np.where(boundary)
            if bys.size < angular_bins:
                continue
            ys, xs = np.where(component)
            centroid_y = float(ys.mean())
            centroid_x = float(xs.mean())
            angles = (np.arctan2(bys - centroid_y, bxs - centroid_x) + 2.0 * np.pi) % (2.0 * np.pi)
            bins = np.floor(angles / (2.0 * np.pi / angular_bins)).astype(int)
            for angle_bin in range(angular_bins):
                in_bin = bins == angle_bin
                if not np.any(in_bin):
                    continue
                center_y = int(round(float(np.median(bys[in_bin]))))
                center_x = int(round(float(np.median(bxs[in_bin]))))
                center_zyx = np.asarray([int(z), center_y, center_x], dtype=int)
                start = np.maximum(center_zyx - half_size_voxels, 0)
                stop = center_zyx + half_size_voxels + 1
                rows.append(
                    {
                        "case_id": case_id,
                        "patch_id": f"{case_id}_patch_{patch_index:04d}",
                        "patch_index": patch_index,
                        "source": "wall_surface_grid",
                        "source_chunk_index": int(angle_bin),
                        "source_voxel_count": int(in_bin.sum()),
                        "wall_slice_z": int(z),
                        "wall_component_rank_in_slice": int(component_rank),
                        "wall_component_label_2d": int(component_label),
                        "wall_angle_bin": int(angle_bin),
                        "center_z_voxel": int(center_zyx[0]),
                        "center_y_voxel": int(center_zyx[1]),
                        "center_x_voxel": int(center_zyx[2]),
                        "center_z_mm": float(center_zyx[0] * spacing_xyz[2]),
                        "center_y_mm": float(center_zyx[1] * spacing_xyz[1]),
                        "center_x_mm": float(center_zyx[2] * spacing_xyz[0]),
                        "start_z_voxel": int(start[0]),
                        "start_y_voxel": int(start[1]),
                        "start_x_voxel": int(start[2]),
                        "stop_z_voxel": int(stop[0]),
                        "stop_y_voxel": int(stop[1]),
                        "stop_x_voxel": int(stop[2]),
                        "patch_size_z_mm": float(patch_size_mm_zyx[0]),
                        "patch_size_y_mm": float(patch_size_mm_zyx[1]),
                        "patch_size_x_mm": float(patch_size_mm_zyx[2]),
                    }
                )
                patch_index += 1

    frame = pd.DataFrame(rows)
    if frame.empty or len(frame) <= max_patches:
        return frame
    keep = np.linspace(0, len(frame) - 1, max_patches).round().astype(int)
    return frame.iloc[keep].reset_index(drop=True)


def _extract_tap_ct_embeddings(
    image: np.ndarray,
    patch_manifest: pd.DataFrame,
    encoder_config: dict[str, Any],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "TAP-CT embeddings require optional dependencies: torch and transformers. "
            "Install with `pip install -e .[encoders]` or install those packages in this environment."
        ) from exc

    model_name = str(encoder_config.get("model_name", "fomofo/tap-ct-b-3d"))
    trust_remote_code = bool(encoder_config.get("trust_remote_code", True))
    device = _select_device(str(encoder_config.get("device", "auto")), torch)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    feature_group = f"encoder_{_backend_label(encoder_config)}"
    target_shape = tuple(int(v) for v in encoder_config.get("target_shape_zyx", [12, 224, 224]))
    hu_min, hu_max = [float(v) for v in encoder_config.get("hu_clip", [-1008, 822])]
    norm_mean = float(encoder_config.get("normalization_mean", -86.8086))
    norm_std = float(encoder_config.get("normalization_std", 322.6347))

    for patch in patch_manifest.itertuples(index=False):
        patch_array = _extract_patch_array(image, patch)
        model_input = _prepare_tap_ct_tensor(patch_array, target_shape, hu_min, hu_max, norm_mean, norm_std, torch)
        model_input = model_input.to(device)
        with torch.no_grad():
            output = _forward_tap_ct(model, model_input)
            embedding = _pool_model_output(output).detach().cpu().numpy().astype(float).ravel()

        for idx, value in enumerate(embedding):
            rows.append(
                feature_row(
                    case_id=case_id,
                    region=str(patch.source),
                    feature_group=feature_group,
                    feature_name=f"{patch.patch_id}_embedding_{idx:04d}",
                    feature_value=float(value),
                    units="",
                    threshold_if_applicable="",
                    mask_name=str(patch.source),
                    software_version=software_version,
                )
            )
        rows.append(
            feature_row(
                case_id=case_id,
                region=str(patch.source),
                feature_group=feature_group,
                feature_name=f"{patch.patch_id}_embedding_l2_norm",
                feature_value=float(np.linalg.norm(embedding)),
                units="",
                mask_name=str(patch.source),
                software_version=software_version,
            )
        )
    return pd.DataFrame(rows)


def _extract_ct_fm_lighter_zoo_embeddings(
    image: np.ndarray,
    patch_manifest: pd.DataFrame,
    encoder_config: dict[str, Any],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        from lighter_zoo import SegResEncoder
    except ImportError as exc:
        raise ImportError(
            "CT-FM embeddings require optional dependencies: torch and lighter_zoo. "
            "Install with `pip install -e .[ct-fm]` or install lighter_zoo in this environment."
        ) from exc

    model_name = str(encoder_config.get("model_name", "project-lighter/ct_fm_feature_extractor"))
    device = _select_device(str(encoder_config.get("device", "auto")), torch)
    model = SegResEncoder.from_pretrained(model_name)
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    feature_group = f"encoder_{_backend_label(encoder_config)}"
    target_shape = tuple(int(v) for v in encoder_config.get("target_shape_zyx", [64, 64, 64]))
    hu_min, hu_max = [float(v) for v in encoder_config.get("hu_clip", [-1024, 2048])]

    for patch in patch_manifest.itertuples(index=False):
        patch_array = _extract_patch_array(image, patch)
        model_input = _prepare_scaled_ct_tensor(patch_array, target_shape, hu_min, hu_max, torch).to(device)
        with torch.no_grad():
            output = model(model_input)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            embedding = _pool_spatial_tensor(output, torch).detach().cpu().numpy().astype(float).ravel()
        rows.extend(_embedding_feature_rows(case_id, patch, feature_group, embedding, software_version))
    return pd.DataFrame(rows)


def _extract_hf_auto_model_3d_embeddings(
    image: np.ndarray,
    patch_manifest: pd.DataFrame,
    encoder_config: dict[str, Any],
    case_id: str,
    software_version: str,
) -> pd.DataFrame:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Generic Hugging Face 3D encoder embeddings require optional dependencies: torch and transformers. "
            "Install with `pip install -e .[encoders]` or install those packages in this environment."
        ) from exc

    model_name = str(encoder_config.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(
            "hf_auto_model_3d/voxelfm_hf requires `model_name`. Set it to the released VoxelFM Hugging Face "
            "identifier or local model path once the exact checkpoint/API is selected."
        )

    trust_remote_code = bool(encoder_config.get("trust_remote_code", True))
    device = _select_device(str(encoder_config.get("device", "auto")), torch)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    feature_group = f"encoder_{_backend_label(encoder_config)}"
    target_shape = tuple(int(v) for v in encoder_config.get("target_shape_zyx", [12, 224, 224]))
    hu_min, hu_max = [float(v) for v in encoder_config.get("hu_clip", [-1008, 822])]
    norm_mean = float(encoder_config.get("normalization_mean", -86.8086))
    norm_std = float(encoder_config.get("normalization_std", 322.6347))

    for patch in patch_manifest.itertuples(index=False):
        patch_array = _extract_patch_array(image, patch)
        model_input = _prepare_tap_ct_tensor(patch_array, target_shape, hu_min, hu_max, norm_mean, norm_std, torch)
        model_input = model_input.to(device)
        with torch.no_grad():
            output = _forward_tap_ct(model, model_input)
            embedding = _pool_model_output(output).detach().cpu().numpy().astype(float).ravel()
        rows.extend(_embedding_feature_rows(case_id, patch, feature_group, embedding, software_version))
    return pd.DataFrame(rows)


def _embedding_feature_rows(
    case_id: str,
    patch: object,
    feature_group: str,
    embedding: np.ndarray,
    software_version: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, value in enumerate(embedding):
        rows.append(
            feature_row(
                case_id=case_id,
                region=str(patch.source),
                feature_group=feature_group,
                feature_name=f"{patch.patch_id}_embedding_{idx:04d}",
                feature_value=float(value),
                units="",
                threshold_if_applicable="",
                mask_name=str(patch.source),
                software_version=software_version,
            )
        )
    rows.append(
        feature_row(
            case_id=case_id,
            region=str(patch.source),
            feature_group=feature_group,
            feature_name=f"{patch.patch_id}_embedding_l2_norm",
            feature_value=float(np.linalg.norm(embedding)),
            units="",
            mask_name=str(patch.source),
            software_version=software_version,
        )
    )
    return rows


def _extract_patch_array(image: np.ndarray, patch: object) -> np.ndarray:
    start = np.asarray([patch.start_z_voxel, patch.start_y_voxel, patch.start_x_voxel], dtype=int)
    stop = np.asarray([patch.stop_z_voxel, patch.stop_y_voxel, patch.stop_x_voxel], dtype=int)
    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(stop - np.asarray(image.shape), 0)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, np.asarray(image.shape))
    patch_array = image[
        clipped_start[0] : clipped_stop[0],
        clipped_start[1] : clipped_stop[1],
        clipped_start[2] : clipped_stop[2],
    ]
    if pad_before.any() or pad_after.any():
        patch_array = np.pad(
            patch_array,
            tuple((int(pad_before[i]), int(pad_after[i])) for i in range(3)),
            mode="constant",
            constant_values=-1024,
        )
    return patch_array.astype(np.float32)


def _prepare_tap_ct_tensor(
    patch_array: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    hu_min: float,
    hu_max: float,
    norm_mean: float,
    norm_std: float,
    torch: Any,
) -> Any:
    from scipy import ndimage as ndi

    patch = np.clip(patch_array.astype(np.float32), hu_min, hu_max)
    zoom = [target_shape_zyx[axis] / patch.shape[axis] for axis in range(3)]
    patch = ndi.zoom(patch, zoom=zoom, order=1)
    patch = (patch - norm_mean) / norm_std
    return torch.from_numpy(patch[None, None, ...].astype(np.float32))


def _prepare_scaled_ct_tensor(
    patch_array: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    hu_min: float,
    hu_max: float,
    torch: Any,
) -> Any:
    from scipy import ndimage as ndi

    patch = np.clip(patch_array.astype(np.float32), hu_min, hu_max)
    zoom = [target_shape_zyx[axis] / patch.shape[axis] for axis in range(3)]
    patch = ndi.zoom(patch, zoom=zoom, order=1)
    denom = max(hu_max - hu_min, 1.0)
    patch = (patch - hu_min) / denom
    return torch.from_numpy(patch[None, None, ...].astype(np.float32))


def _forward_tap_ct(model: object, model_input: object) -> object:
    try:
        return model.forward(model_input)
    except TypeError:
        try:
            return model(model_input)
        except TypeError:
            return model(pixel_values=model_input)


def _pool_model_output(output: object) -> object:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output[0]
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
        while hidden.ndim > 2:
            hidden = hidden.mean(dim=1)
        return hidden[0]
    if isinstance(output, tuple):
        return _pool_model_output(output[0])
    if hasattr(output, "ndim"):
        return _pool_spatial_tensor(output, None)
    raise ValueError("Could not find a usable embedding tensor in encoder output.")


def _pool_spatial_tensor(tensor: object, torch: Any | None) -> object:
    if not hasattr(tensor, "ndim"):
        raise ValueError("Encoder output is not tensor-like.")
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2:
        return tensor[0]
    if tensor.ndim == 5 and torch is not None:
        return torch.nn.functional.adaptive_avg_pool3d(tensor, 1).reshape(tensor.shape[0], -1)[0]
    while tensor.ndim > 2:
        tensor = tensor.mean(dim=tuple(range(2, tensor.ndim)))
    return tensor[0]


def _select_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _backend_label(encoder_config: dict[str, Any]) -> str:
    raw = str(encoder_config.get("name") or encoder_config.get("backend") or "encoder")
    return "".join(char if char.isalnum() else "_" for char in raw).strip("_") or "encoder"


def _encoder_error_frame(case_id: str, backend: str, message: str, software_version: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            feature_row(
                case_id=case_id,
                region="encoder",
                feature_group=f"encoder_{backend}_status",
                feature_name="extraction_error",
                feature_value=message,
                software_version=software_version,
            )
        ]
    )


def _label_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage as ndi

        return ndi.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=int))
    except Exception:
        from .lumen_geometry import _label_2d as fallback_label_2d

        return fallback_label_2d(mask)


def _binary_erosion_2d(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi

        return ndi.binary_erosion(mask)
    except Exception:
        from .lumen_geometry import _binary_erosion_2d as fallback_binary_erosion_2d

        return fallback_binary_erosion_2d(mask)


def _component_labels_by_size(labels: np.ndarray, n_labels: int, max_components: int) -> list[int]:
    if n_labels <= 0:
        return []
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    ordered = np.argsort(counts)[::-1]
    return [int(label) for label in ordered[:max_components] if counts[label] > 0]
