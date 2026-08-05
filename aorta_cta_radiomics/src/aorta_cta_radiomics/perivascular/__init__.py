"""Carotid-first perivascular phenotyping primitives.

This package is deliberately independent of segmentation inference.  It
consumes typed lumen/centreline/anatomical inputs and produces deterministic,
physical-space measurements that an adapter or CLI can orchestrate.
"""

from .composition import (
    CompositionResult,
    TissueLabel,
    TissueThresholds,
    classify_tissue_composition,
)
from .contact import SurfaceContactResult, calculate_surface_contacts
from .coordinates import VesselCoordinateSystem, VesselCoordinates
from .sectors import SectorAssignment, assign_angular_sectors
from .shells import (
    AdaptiveShellLaw,
    RadialBand,
    ShellBandResult,
    ShellSetResult,
    generate_adaptive_shell,
    generate_radial_shells,
)

__all__ = [
    "AdaptiveShellLaw",
    "CompositionResult",
    "RadialBand",
    "SectorAssignment",
    "ShellBandResult",
    "ShellSetResult",
    "SurfaceContactResult",
    "TissueLabel",
    "TissueThresholds",
    "VesselCoordinateSystem",
    "VesselCoordinates",
    "assign_angular_sectors",
    "calculate_surface_contacts",
    "classify_tissue_composition",
    "generate_adaptive_shell",
    "generate_radial_shells",
]
