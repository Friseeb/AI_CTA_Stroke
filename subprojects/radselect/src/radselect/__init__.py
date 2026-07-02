"""Leakage-safe feature selection for already-extracted radiomic and clinical tables."""

from .config import RunConfig
from .core import (
    RadselectResult,
    apply_composite_score_parameters,
    apply_projection_parameters,
    run_selection,
    write_output_manifest,
)

__all__ = [
    "RadselectResult",
    "RunConfig",
    "apply_composite_score_parameters",
    "apply_projection_parameters",
    "run_selection",
    "write_output_manifest",
]

__version__ = "0.1.0"
