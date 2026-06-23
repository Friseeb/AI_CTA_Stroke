"""Metal / beam-hardening artifact detection and burden scoring.

Image-domain detection of metal (valves, stents, implants, wires) plus its
blooming and an in-plane streak estimate, and a per-case / per-ROI artifact
*burden* for flagging and HU-feature exclusion. This deliberately does NOT
restore/inpaint HU — fabricated HU biases quantitative endpoints (calcium, FAI
fat HU). Downstream code should EXCLUDE the artifact mask from HU statistics and
FLAG high-burden cases/ROIs.

Arrays are numpy in array order (axis 0,1,2 ≈ z,y,x for SimpleITK volumes), and
``spacing`` is given in the same array order. From a SimpleITK image use
``cta_common.geometry.compute_spacing_from_sitk(img)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Defaults (HU). Implant-grade threshold: 3000 HU excludes cortical bone and
# ordinary dense calcium (which sit ~1500-2500), keeping metal prostheses/wires
# and severely calcified valves that actually cause beam-hardening streaks.
DEFAULT_METAL_HU = 3000.0
DEFAULT_BLOOM_MM = 3.0
DEFAULT_STREAK_MM = 40.0
DEFAULT_BRIGHT_HU = 1500.0   # bright streak / over-correction near metal
DEFAULT_DARK_HU = -150.0     # dark streak / under-correction (below soft tissue)
DEFAULT_AIR_HU = -300.0      # body vs background air separation


def _ndi():
    from scipy import ndimage as ndi  # lazy: scipy optional at import time
    return ndi


def body_mask(hu: np.ndarray, air_hu: float = DEFAULT_AIR_HU) -> np.ndarray:
    """Largest filled connected component above ``air_hu`` (the patient body)."""
    ndi = _ndi()
    filled = ndi.binary_fill_holes(hu > air_hu)
    lbl, n = ndi.label(filled)
    if n == 0:
        return filled
    largest = 1 + int(np.argmax(np.bincount(lbl.ravel())[1:]))
    return lbl == largest


def detect_metal(hu: np.ndarray, metal_hu: float = DEFAULT_METAL_HU) -> np.ndarray:
    """Boolean mask of metal-grade voxels (HU >= ``metal_hu``)."""
    return np.asarray(hu) >= float(metal_hu)


@dataclass
class ArtifactMasks:
    metal: np.ndarray
    bloom: np.ndarray
    streak: np.ndarray
    body: np.ndarray

    @property
    def artifact(self) -> np.ndarray:
        """Full artifact region (metal + bloom + streak), inside the body."""
        return (self.metal | self.bloom | self.streak) & self.body

    @property
    def core(self) -> np.ndarray:
        """High-confidence region (metal + bloom) for conservative exclusion."""
        return (self.metal | self.bloom) & self.body


def artifact_masks(
    hu: np.ndarray,
    spacing: Sequence[float],
    *,
    metal_hu: float = DEFAULT_METAL_HU,
    bloom_mm: float = DEFAULT_BLOOM_MM,
    streak_mm: float = DEFAULT_STREAK_MM,
    bright_hu: float = DEFAULT_BRIGHT_HU,
    dark_hu: float = DEFAULT_DARK_HU,
    air_hu: float = DEFAULT_AIR_HU,
) -> ArtifactMasks:
    """Detect metal, bloom (dilation), and an in-plane streak estimate.

    Streaks are flagged as bright/dark anomalies within ``streak_mm`` of metal,
    restricted to the body mask (so background air is not mistaken for a dark
    streak). The streak term is a heuristic — pure image-domain streak detection
    is approximate; ``core`` (metal+bloom) is the high-confidence subset.
    """
    ndi = _ndi()
    hu = np.asarray(hu, dtype=np.float32)
    sp = [float(s) for s in spacing]
    body = body_mask(hu, air_hu)

    metal = detect_metal(hu, metal_hu) & body
    if not metal.any():
        empty = np.zeros_like(metal)
        return ArtifactMasks(metal=metal, bloom=empty, streak=empty.copy(), body=body)

    bloom_it = max(1, int(round(bloom_mm / min(sp))))
    bloom = ndi.binary_dilation(metal, iterations=bloom_it) & ~metal & body

    streak = np.zeros_like(metal)
    in_plane = min(sp[1], sp[2]) if len(sp) >= 3 else min(sp)
    rad = max(1, int(round(streak_mm / in_plane)))
    for z in np.where(metal.any(axis=(1, 2)))[0]:
        near = ndi.binary_dilation(metal[z], iterations=rad)
        anomalous = (hu[z] >= bright_hu) | (hu[z] <= dark_hu)
        streak[z] = near & anomalous & body[z] & ~metal[z] & ~bloom[z]

    return ArtifactMasks(metal=metal, bloom=bloom, streak=streak, body=body)


def artifact_burden(
    masks: ArtifactMasks,
    spacing: Sequence[float],
    roi_mask: np.ndarray | None = None,
) -> dict:
    """Volumes (mL) and ROI-affected fractions for the detected artifact."""
    ndi = _ndi()
    vox_ml = float(np.prod([float(s) for s in spacing])) / 1000.0

    def ml(m):
        return float(m.sum()) * vox_ml

    metal, artifact, core = masks.metal, masks.artifact, masks.core
    out = {
        "metal_ml": ml(metal),
        "bloom_ml": ml(masks.bloom),
        "streak_ml": ml(masks.streak),
        "core_ml": ml(core),
        "artifact_ml": ml(artifact),
        "n_metal_components": int(ndi.label(metal)[1]) if metal.any() else 0,
        "n_slices_with_metal": int(np.count_nonzero(metal.any(axis=(1, 2)))) if metal.ndim == 3 else 0,
        "has_metal": bool(metal.any()),
    }
    if roi_mask is not None:
        rm = np.asarray(roi_mask) > 0
        n = int(rm.sum())
        out["roi_voxels"] = n
        # ROI-scoped metal: global HU thresholds also catch dense skull-base/
        # petrous bone in head/neck CTA, so metal presence is judged within the ROI.
        out["roi_has_metal"] = bool((metal & rm).any())
        out["roi_artifact_ml"] = ml(artifact & rm)
        out["roi_core_ml"] = ml(core & rm)
        out["roi_artifact_fraction"] = float((artifact & rm).sum() / n) if n else 0.0
        out["roi_core_fraction"] = float((core & rm).sum() / n) if n else 0.0
    return out


def classify_burden(
    burden: dict,
    *,
    key: str = "roi_artifact_fraction",
    low: float = 0.01,
    moderate: float = 0.05,
) -> str:
    """Map a burden metric to none|low|moderate|high.

    Presence is judged ROI-scoped (``roi_has_metal``) when available — a global
    ``has_metal`` is confounded by dense skull-base bone in head/neck CTA. Cases
    flagged ``moderate``/``high`` should have their HU features caveated/excluded.
    """
    has_metal = burden.get("roi_has_metal", burden.get("has_metal"))
    if not has_metal:
        return "none"
    frac = burden.get(key)
    if frac is None:
        return "low"  # metal present but no ROI to quantify against
    if frac >= moderate:
        return "high"
    if frac >= low:
        return "moderate"
    return "low" if frac > 0 else "none"


__all__ = [
    "ArtifactMasks",
    "artifact_masks",
    "artifact_burden",
    "body_mask",
    "classify_burden",
    "detect_metal",
    "DEFAULT_METAL_HU",
    "DEFAULT_BLOOM_MM",
    "DEFAULT_STREAK_MM",
]
