"""Radiomics extractor integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .features import feature_row


def extract_radiomics_features(
    image_path: str | Path,
    mask_path: str | Path,
    case_id: str,
    region: str,
    settings_path: str | Path | None = None,
    include_diagnostics: bool = False,
    software_version: str = __version__,
    backend: str = "pyradiomics",
    device: str = "auto",
) -> pd.DataFrame:
    """Extract radiomics features for one image/mask pair."""
    backend = str(backend or "pyradiomics").lower()
    if backend in {"pyradiomics", "py radiomics"}:
        result = _execute_pyradiomics(image_path, mask_path, settings_path)
        backend_name = "pyradiomics"
    elif backend in {"fastrad", "fast-rad", "fast_radiomics", "fastradiomics"}:
        result = _execute_fastrad(image_path, mask_path, settings_path, device=device)
        backend_name = "fastrad"
    else:
        raise ValueError(f"Unsupported radiomics backend: {backend!r}. Use 'pyradiomics' or 'fastrad'.")

    rows: list[dict[str, object]] = []
    for name, value in result.items():
        if name.startswith("diagnostics_") and not include_diagnostics:
            continue
        group = _feature_group(name)
        rows.append(
            feature_row(
                case_id=case_id,
                region=region,
                feature_group=f"radiomics_{group}",
                feature_name=name,
                feature_value=_coerce_value(value),
                units="",
                mask_name=Path(mask_path).name,
                software_version=f"{software_version}+{backend_name}",
            )
        )
    return pd.DataFrame(rows)


def _execute_pyradiomics(
    image_path: str | Path,
    mask_path: str | Path,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the reference PyRadiomics extractor."""
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "PyRadiomics is enabled but not installed. Install the provided conda "
            "environment, install `pyradiomics`, or set radiomics.enabled=false."
        ) from exc

    extractor = (
        featureextractor.RadiomicsFeatureExtractor(str(settings_path))
        if settings_path is not None
        else featureextractor.RadiomicsFeatureExtractor()
    )
    return dict(extractor.execute(str(image_path), str(mask_path)))


def _execute_fastrad(
    image_path: str | Path,
    mask_path: str | Path,
    settings_path: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Run the optional fastrad PyTorch-native extractor.

    The installed fastrad API uses tensor-backed ``MedicalImage`` and ``Mask``
    objects rather than reading paths directly, so this wrapper translates the
    PyRadiomics settings subset used by this project and loads NIfTI data via
    SimpleITK.
    """
    try:
        import torch
        import SimpleITK as sitk
        from fastrad import FeatureExtractor, FeatureSettings, Mask, MedicalImage
        from fastrad.extractor import _FEATURE_MAP
    except ImportError as exc:
        raise ImportError(
            "fastrad radiomics backend requested but not installed. Install it "
            "with `pip install fastrad` in an environment that also has torch, "
            "or set radiomics.backend=pyradiomics."
        ) from exc

    settings_dict = _read_radiomics_settings(settings_path)
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image, mask = _preprocess_for_fastrad(image, mask, settings_dict, sitk)
    image_arr = sitk.GetArrayFromImage(image).astype("float32")
    mask_arr = sitk.GetArrayFromImage(mask).astype("float32")
    image_arr, mask_arr = _crop_to_mask_bbox(image_arr, mask_arr, margin_voxels=3)
    spacing_zyx = tuple(float(v) for v in image.GetSpacing()[::-1])

    feature_classes = _fastrad_feature_classes(settings_dict)
    settings = FeatureSettings(
        feature_classes=feature_classes,
        bin_width=float(settings_dict.get("setting", {}).get("binWidth", 25.0)),
        device=_resolve_torch_device(device, torch),
        spacing=spacing_zyx,
        force2D=bool(settings_dict.get("setting", {}).get("force2D", False)),
        force2Ddimension=int(settings_dict.get("setting", {}).get("force2Ddimension", 0)),
    )
    image_tensor = torch.from_numpy(_pad_to_even_shape(image_arr))
    mask_tensor = torch.from_numpy(_pad_to_even_shape(mask_arr))
    result = _extract_fastrad_feature_classes(
        extractor_class=FeatureExtractor,
        feature_map=_FEATURE_MAP,
        settings=settings,
        image=MedicalImage(image_tensor, spacing=spacing_zyx),
        mask=Mask(mask_tensor, spacing=spacing_zyx),
        torch_module=torch,
    )
    return {f"original_{name}": value for name, value in result.items()}


def _extract_fastrad_feature_classes(
    extractor_class: Any,
    feature_map: dict[str, Any],
    settings: Any,
    image: Any,
    mask: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Extract fastrad features class-by-class so one texture bug is nonfatal."""
    requested_classes = list(settings.feature_classes)
    result: dict[str, Any] = {}
    failed: list[str] = []
    for feature_class in requested_classes:
        if feature_class not in feature_map:
            failed.append(f"{feature_class}:unknown_feature_class")
            continue
        single_settings = _copy_fastrad_settings(settings, feature_class)
        extractor = extractor_class(single_settings)
        try:
            result.update(extractor.extract(image, mask))
        except Exception as exc:
            if (
                isinstance(exc, torch_module.cuda.OutOfMemoryError)
                and getattr(single_settings, "device", None) is not None
                and str(single_settings.device).startswith("cuda")
            ):
                try:
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
                cpu_settings = _copy_fastrad_settings(settings, feature_class, device="cpu")
                try:
                    result.update(extractor_class(cpu_settings).extract(image, mask))
                    continue
                except Exception as cpu_exc:
                    failed.append(f"{feature_class}:{type(cpu_exc).__name__}:{cpu_exc}")
            else:
                failed.append(f"{feature_class}:{type(exc).__name__}:{exc}")
    if failed:
        result["diagnostics_fastrad_failed_feature_classes"] = "; ".join(failed)
    return result


def _copy_fastrad_settings(settings: Any, feature_class: str, device: str | None = None) -> Any:
    from fastrad import FeatureSettings

    return FeatureSettings(
        feature_classes=[feature_class],
        bin_width=settings.bin_width,
        device=device or settings.device,
        spacing=settings.spacing,
        force2D=settings.force2D,
        force2Ddimension=settings.force2Ddimension,
    )


def _pad_to_even_shape(array: Any) -> Any:
    """Pad trailing edges to even extents for fastrad texture kernels."""
    import numpy as np

    arr = np.asarray(array)
    pad_width = [(0, int(size % 2)) for size in arr.shape]
    if not any(after for _, after in pad_width):
        return arr
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _preprocess_for_fastrad(image: Any, mask: Any, settings: dict[str, Any], sitk_module: Any) -> tuple[Any, Any]:
    """Apply the PyRadiomics preprocessing subset needed before fastrad extraction."""
    setting = settings.get("setting", {}) if isinstance(settings, dict) else {}
    spacing = _resampled_spacing_xyz(setting.get("resampledPixelSpacing"))
    if spacing is not None:
        image = _resample_sitk(
            image,
            spacing_xyz=spacing,
            interpolator=_sitk_interpolator(str(setting.get("interpolator", "sitkBSpline")), sitk_module),
            default_value=0.0,
            pixel_id=image.GetPixelID(),
            sitk_module=sitk_module,
        )
        mask = _resample_sitk(
            mask,
            spacing_xyz=spacing,
            interpolator=sitk_module.sitkNearestNeighbor,
            default_value=0,
            pixel_id=sitk_module.sitkUInt8,
            sitk_module=sitk_module,
        )
    return image, mask


def _resampled_spacing_xyz(value: Any) -> tuple[float, float, float] | None:
    if value in (None, "", []):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    spacing = tuple(float(v) for v in value)
    if any(v <= 0 for v in spacing):
        return None
    return spacing


def _resample_sitk(
    image: Any,
    spacing_xyz: tuple[float, float, float],
    interpolator: int,
    default_value: float,
    pixel_id: int,
    sitk_module: Any,
) -> Any:
    original_spacing = tuple(float(v) for v in image.GetSpacing())
    original_size = tuple(int(v) for v in image.GetSize())
    if all(abs(original_spacing[i] - spacing_xyz[i]) < 1e-6 for i in range(3)):
        return image
    new_size = [
        max(1, int(round(original_size[i] * (original_spacing[i] / spacing_xyz[i]))))
        for i in range(3)
    ]
    return sitk_module.Resample(
        image,
        new_size,
        sitk_module.Transform(),
        interpolator,
        image.GetOrigin(),
        spacing_xyz,
        image.GetDirection(),
        default_value,
        pixel_id,
    )


def _sitk_interpolator(name: str, sitk_module: Any) -> int:
    mapping = {
        "sitknearestneighbor": sitk_module.sitkNearestNeighbor,
        "sitklinear": sitk_module.sitkLinear,
        "sitkbspline": sitk_module.sitkBSpline,
        "sitkgaussian": sitk_module.sitkGaussian,
        "sitklabelgaussian": sitk_module.sitkLabelGaussian,
    }
    return mapping.get(name.lower(), sitk_module.sitkBSpline)


def _crop_to_mask_bbox(
    image_arr: Any,
    mask_arr: Any,
    margin_voxels: int = 3,
) -> tuple[Any, Any]:
    import numpy as np

    mask_binary = np.asarray(mask_arr) > 0
    if not mask_binary.any():
        return image_arr, mask_arr
    coords = np.argwhere(mask_binary)
    mins = np.maximum(coords.min(axis=0) - int(margin_voxels), 0)
    maxs = np.minimum(coords.max(axis=0) + int(margin_voxels) + 1, np.asarray(mask_binary.shape))
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(mask_binary.ndim))
    return image_arr[slices], mask_arr[slices]


def _read_radiomics_settings(settings_path: str | Path | None) -> dict[str, Any]:
    if settings_path is None:
        return {}
    import yaml

    with Path(settings_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _fastrad_feature_classes(settings: dict[str, Any]) -> list[str]:
    feature_classes = settings.get("featureClass") or {}
    unsupported = {"shape", "shape2d"}
    if not isinstance(feature_classes, dict) or not feature_classes:
        return ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    return [str(name) for name in feature_classes if str(name).lower() not in unsupported]


def _resolve_torch_device(device: str, torch_module: Any) -> str:
    """Resolve config shorthand to a fastrad-supported device string."""
    requested = str(device or "auto").lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "gpu":
        return "cuda"
    return requested


def _feature_group(name: str) -> str:
    if ":" in name:
        prefix = name.split(":", 1)[0]
        parts = prefix.split("_")
        if len(parts) >= 2:
            return parts[1]
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] in {"original", "wavelet", "logarithm", "gradient", "square"}:
        return parts[1]
    if name.startswith("diagnostics_"):
        return "diagnostics"
    return "unknown"


def _coerce_value(value: object) -> object:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value
