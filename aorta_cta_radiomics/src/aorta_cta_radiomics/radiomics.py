"""PyRadiomics integration."""

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
) -> pd.DataFrame:
    """Extract PyRadiomics features for one image/mask pair."""
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
    result: dict[str, Any] = extractor.execute(str(image_path), str(mask_path))

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
                software_version=software_version,
            )
        )
    return pd.DataFrame(rows)


def _feature_group(name: str) -> str:
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
