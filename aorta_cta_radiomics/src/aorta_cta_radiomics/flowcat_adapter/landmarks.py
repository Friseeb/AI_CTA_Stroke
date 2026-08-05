"""Safe JSON loading for FLOWCAT anatomical landmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .exceptions import FlowcatSchemaError


def load_landmarks_ras_mm(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Load FLOWCAT's data-only ``{label: [r, a, s]}`` landmarks JSON."""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"FLOWCAT landmarks JSON not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FlowcatSchemaError(f"Could not read FLOWCAT landmarks JSON {file_path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise FlowcatSchemaError("FLOWCAT landmarks JSON must be a non-empty object")
    landmarks: dict[str, tuple[float, float, float]] = {}
    for raw_label, raw_position in payload.items():
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise FlowcatSchemaError(f"Invalid FLOWCAT landmark label: {raw_label!r}")
        try:
            position = np.asarray(raw_position, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FlowcatSchemaError(f"Landmark {raw_label!r} is not a numeric coordinate") from exc
        if position.shape != (3,) or not np.isfinite(position).all():
            raise FlowcatSchemaError(
                f"Landmark {raw_label!r} must be a finite RAS coordinate with shape (3,)"
            )
        landmarks[raw_label] = tuple(float(value) for value in position)
    return landmarks
