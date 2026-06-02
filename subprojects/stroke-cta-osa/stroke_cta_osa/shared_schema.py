"""Schemas every airway provider must speak.

This module is intentionally *interface-only*: any pipeline that wants to
contribute an upper-airway mask + landmarks for downstream OSA-style feature
extraction can satisfy these dataclasses without importing the rest of
stroke_cta_osa. The dental subproject, a future deep-learning model, or a
manual annotation toolchain can all be adapted.

If/when the dental pipeline grows real airway outputs they should be saved
as JSON matching :class:`SharedAirwayPayload.to_json` so the
:class:`adapters.DentalAirwayAdapter` can consume them without code change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SHARED_FEATURE_NAMES: list[str] = [
    "airway_volume_ml",
    "airway_min_csa_mm2",
    "airway_min_csa_z_mm",
    "airway_csa_p05_mm2",
    "airway_csa_p10_mm2",
    "airway_csa_p25_mm2",
    "airway_csa_median_mm2",
    "airway_length_mm",
    "airway_lateral_diameter_min_mm",
    "airway_ap_diameter_min_mm",
    "airway_eccentricity_at_min_csa",
    "retropalatal_csa_mm2",
    "retroglossal_csa_mm2",
    "retrolingual_csa_mm2",
    "retropalatal_volume_ml",
    "retroglossal_volume_ml",
]
"""Features that BOTH pipelines (dental/CBCT and stroke CTA) may emit. Names
are stable across both — when you join the two CSVs on patient_id you can
diff these columns directly. See docs/stroke_cta_osa/DENTAL_PIPELINE_INTEGRATION.md.
"""


@dataclass
class SharedAirwayLandmarks:
    """Anatomical landmarks expressed as voxel indices in the *consuming
    image's* frame. Adapters must convert if they receive a different frame.
    """
    posterior_nasal_spine: Optional[tuple[int, int, int]] = None
    soft_palate_inferior: Optional[tuple[int, int, int]] = None
    hyoid: Optional[tuple[int, int, int]] = None
    epiglottis_tip: Optional[tuple[int, int, int]] = None
    mandibular_plane_z: Optional[int] = None


@dataclass
class SharedAirwayFeatures:
    """Pre-computed shared features, if a provider already has them."""
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> Optional[float]:
        v = self.values.get(name)
        return float(v) if v is not None else None


@dataclass
class SharedAirwayPayload:
    """Full handoff object an adapter returns to the orchestrator.

    Two cases are explicitly supported:

    1. Provider has a raw mask  → ``mask_path`` set, ``features`` may be empty.
       The stroke pipeline will compute features from the mask itself.
    2. Provider only has derived numbers → ``features`` populated, ``mask_path``
       is None. The stroke pipeline records the numbers under shared columns
       but won't recompute them.
    """
    mask_path: Optional[str] = None
    landmarks: SharedAirwayLandmarks = field(default_factory=SharedAirwayLandmarks)
    features: SharedAirwayFeatures = field(default_factory=SharedAirwayFeatures)
    source: str = "unknown"
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "mask_path": self.mask_path,
            "landmarks": asdict(self.landmarks),
            "features": dict(self.features.values),
            "source": self.source,
            "notes": self.notes,
        }
