"""Canonical landmark names + dataclasses.

A landmark is one of:

  * a **point** — voxel index ``(z, y, x)`` plus optional physical mm position;
  * a **z-level** — a single axial slice index marking an anatomical level
    (e.g. the retropalatal level);
  * a **plane** — three points or a normal + point pair, defining a plane
    in physical space (e.g. the mandibular plane).

This file does NOT compute landmarks — it just describes their shapes so
external CSV/JSON, dental-pipeline outputs, and our own heuristic
estimators can all serialise to the same contract.

Coordinates: every landmark records position in *both* voxel index and
physical mm whenever possible. Voxel indices are in array order ``(z, y, x)``
to match SimpleITK / nibabel conventions used throughout the pipeline.
Physical mm coordinates are in the image's native physical frame (LPS for
ITK / DICOM-derived NIfTI, RAS for some external NIfTI). The landmark file
records ``coord_system`` so consumers can convert if needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Optional


# Canonical landmark identifiers — used as JSON keys + as registry names.
POINT_LANDMARKS: tuple[str, ...] = (
    "posterior_nasal_spine",
    "anterior_nasal_spine",
    "soft_palate_tip",
    "uvula_tip",
    "epiglottis_tip",
    "tongue_anterior_point",
    "tongue_posterior_point",
    "tongue_base_point",
    "tongue_dorsum_point",
    "floor_of_mouth_point",
    "menton",
    "gonion_left",
    "gonion_right",
    "hyoid_centroid",
    "hyoid_superior_point",
    "hyoid_inferior_point",
    "c2_centroid",
    "c3_centroid",
    "c4_centroid",
)

Z_LEVEL_LANDMARKS: tuple[str, ...] = (
    "hard_palate_plane",
    "retropalatal_level",
    "retroglossal_level",
    "retrolingual_level",
    "tongue_base_level",
    "laryngeal_inlet_level",
)

PLANE_LANDMARKS: tuple[str, ...] = (
    "mandibular_plane",
    "cervical_spine_axis",
    "oral_cavity_boundary",
)


@dataclass
class LandmarkPoint:
    """One 3D anatomical point.

    `voxel_zyx` is the array-index representation; `physical_mm` is the
    same point mapped through the image affine. Either can be missing
    individually (e.g. when a landmark is supplied as voxel-only and we
    haven't been given the image yet).
    """
    name: str
    voxel_zyx: Optional[tuple[int, int, int]] = None
    physical_mm: Optional[tuple[float, float, float]] = None
    source: str = "unknown"       # 'external_json', 'dental_adapter', 'heuristic'
    confidence: float = 0.0       # 0..1 — provider's own confidence

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "voxel_zyx": list(self.voxel_zyx) if self.voxel_zyx else None,
            "physical_mm": list(self.physical_mm) if self.physical_mm else None,
            "source": self.source,
            "confidence": float(self.confidence),
        }


@dataclass
class LandmarkZLevel:
    """A single axial slice marking an anatomical level."""
    name: str
    z_voxel: Optional[int] = None
    z_physical_mm: Optional[float] = None
    source: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "z_voxel": int(self.z_voxel) if self.z_voxel is not None else None,
            "z_physical_mm": (float(self.z_physical_mm)
                              if self.z_physical_mm is not None else None),
            "source": self.source,
            "confidence": float(self.confidence),
        }


@dataclass
class LandmarkPlane:
    """A plane in physical mm — represented as 3 points OR (point, normal).

    Either tuple of three `LandmarkPoint` references OR an explicit
    (point_phys_mm, normal_phys_mm) pair. The validator enforces that
    exactly one of the two representations is populated.
    """
    name: str
    point_names: Optional[tuple[str, str, str]] = None
    point_phys_mm: Optional[tuple[float, float, float]] = None
    normal_phys_mm: Optional[tuple[float, float, float]] = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "point_names": list(self.point_names) if self.point_names else None,
            "point_phys_mm": list(self.point_phys_mm) if self.point_phys_mm else None,
            "normal_phys_mm": list(self.normal_phys_mm) if self.normal_phys_mm else None,
            "source": self.source,
        }


@dataclass
class LandmarkBundle:
    """Full landmark set for one case.

    Stored on disk as a single JSON file with `coord_system`, `image_shape_zyx`,
    `image_affine`, and per-landmark dicts. The affine is recorded so that
    downstream consumers can convert between voxel and physical mm without
    re-loading the image.
    """
    coord_system: Literal["voxel_zyx", "lps_mm", "ras_mm", "mixed"] = "mixed"
    image_shape_zyx: Optional[tuple[int, int, int]] = None
    image_affine: Optional[list[list[float]]] = None
    points: dict[str, LandmarkPoint] = field(default_factory=dict)
    z_levels: dict[str, LandmarkZLevel] = field(default_factory=dict)
    planes: dict[str, LandmarkPlane] = field(default_factory=dict)
    case_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "coord_system": self.coord_system,
            "image_shape_zyx": (list(self.image_shape_zyx)
                                if self.image_shape_zyx else None),
            "image_affine": self.image_affine,
            "points": {k: v.to_dict() for k, v in self.points.items()},
            "z_levels": {k: v.to_dict() for k, v in self.z_levels.items()},
            "planes": {k: v.to_dict() for k, v in self.planes.items()},
            "notes": self.notes,
        }
