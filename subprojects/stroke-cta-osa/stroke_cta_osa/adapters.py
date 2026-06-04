"""Airway providers.

Each adapter implements the same minimal interface and decides at construction
time whether it can serve a given case. The orchestrator tries them in
priority order and accepts the first that returns ``is_available()`` True.

Importantly: the stroke pipeline never imports the dental package. The
DentalAirwayAdapter reads files (mask NIfTI + landmarks JSON + optional
features JSON) — that's the only contract. If the dental subproject grows
real airway outputs it must save them at the configured paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import SimpleITK as sitk

from .logging_utils import get_logger
from .shared_schema import SharedAirwayLandmarks, SharedAirwayFeatures, SharedAirwayPayload
from .types import AirwayMaskInfo, CTAImage

log = get_logger("adapters")


class AirwayProvider(Protocol):
    name: str

    def is_available(self) -> bool: ...
    def get_payload(self, image: CTAImage) -> SharedAirwayPayload: ...
    def get_airway_mask(self, image: CTAImage) -> Optional[AirwayMaskInfo]: ...
    def get_landmarks(self, image: CTAImage) -> SharedAirwayLandmarks: ...
    def get_existing_features(self) -> SharedAirwayFeatures: ...


# ---------------------------------------------------------------------------
# DentalAirwayAdapter — read JSON/NIfTI artefacts produced by the dental
# subproject (or any compatible tool). No dependency on cta_dental.
# ---------------------------------------------------------------------------

class DentalAirwayAdapter:
    name = "dental_adapter"

    def __init__(
        self,
        mask_path: Optional[str | Path],
        landmarks_path: Optional[str | Path],
        features_path: Optional[str | Path],
    ) -> None:
        self.mask_path = Path(mask_path) if mask_path else None
        self.landmarks_path = Path(landmarks_path) if landmarks_path else None
        self.features_path = Path(features_path) if features_path else None

    def is_available(self) -> bool:
        return bool(
            (self.mask_path and self.mask_path.is_file())
            or (self.features_path and self.features_path.is_file())
        )

    def get_payload(self, image: CTAImage) -> SharedAirwayPayload:
        return SharedAirwayPayload(
            mask_path=str(self.mask_path) if self.mask_path and self.mask_path.is_file() else None,
            landmarks=self.get_landmarks(image),
            features=self.get_existing_features(),
            source="dental_subproject",
            notes="Reused upper-airway artefacts from dental/CBCT pipeline.",
        )

    def get_airway_mask(self, image: CTAImage) -> Optional[AirwayMaskInfo]:
        if not (self.mask_path and self.mask_path.is_file()):
            return None
        try:
            img = sitk.ReadImage(str(self.mask_path))
            arr = sitk.GetArrayFromImage(img).astype(bool)
        except Exception as exc:
            log.warning("Could not read dental airway mask: %s", exc)
            return None
        if arr.shape != image.shape_zyx:
            # Resample to match the consuming image's geometry. We accept
            # that this can erode thin airway walls; the alternative is to
            # silently emit wrong CSAs.
            try:
                img = sitk.Resample(
                    img,
                    _reference_from_cta(image),
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    img.GetPixelID(),
                )
                arr = sitk.GetArrayFromImage(img).astype(bool)
            except Exception as exc:
                log.warning("Failed to resample dental mask: %s", exc)
                return None
        if not arr.any():
            return None
        return AirwayMaskInfo(
            mask_zyx=arr,
            method=self.name,
            confidence="medium",
            notes="From dental pipeline.",
        )

    def get_landmarks(self, image: CTAImage) -> SharedAirwayLandmarks:
        if not (self.landmarks_path and self.landmarks_path.is_file()):
            return SharedAirwayLandmarks()
        try:
            data = json.loads(self.landmarks_path.read_text())
        except Exception as exc:
            log.warning("Could not parse landmarks JSON: %s", exc)
            return SharedAirwayLandmarks()
        out = SharedAirwayLandmarks()
        for fld in ("posterior_nasal_spine", "soft_palate_inferior",
                    "hyoid", "epiglottis_tip"):
            v = data.get(fld)
            if isinstance(v, (list, tuple)) and len(v) == 3:
                out.__setattr__(fld, tuple(int(x) for x in v))
        mp = data.get("mandibular_plane_z")
        if isinstance(mp, int):
            out.mandibular_plane_z = mp
        return out

    def get_existing_features(self) -> SharedAirwayFeatures:
        if not (self.features_path and self.features_path.is_file()):
            return SharedAirwayFeatures()
        try:
            data = json.loads(self.features_path.read_text())
        except Exception as exc:
            log.warning("Could not parse dental features JSON: %s", exc)
            return SharedAirwayFeatures()
        vals: dict[str, float] = {}
        for k, v in (data or {}).items():
            if isinstance(v, (int, float)):
                vals[k] = float(v)
        return SharedAirwayFeatures(values=vals)


# ---------------------------------------------------------------------------
# ExternalMaskAdapter — user-provided mask NIfTI.
# ---------------------------------------------------------------------------

class ExternalMaskAdapter:
    name = "external_mask"

    def __init__(self, mask_path: Optional[str | Path]) -> None:
        self.mask_path = Path(mask_path) if mask_path else None

    def is_available(self) -> bool:
        return bool(self.mask_path and self.mask_path.is_file())

    def get_payload(self, image: CTAImage) -> SharedAirwayPayload:
        return SharedAirwayPayload(
            mask_path=str(self.mask_path) if self.is_available() else None,
            landmarks=SharedAirwayLandmarks(),
            features=SharedAirwayFeatures(),
            source="external_mask",
            notes="User-supplied airway mask.",
        )

    def get_airway_mask(self, image: CTAImage) -> Optional[AirwayMaskInfo]:
        if not self.is_available():
            return None
        try:
            img = sitk.ReadImage(str(self.mask_path))
        except Exception as exc:
            log.warning("External mask read failed: %s", exc)
            return None
        if img.GetSize() != tuple(reversed(image.shape_zyx)):
            try:
                img = sitk.Resample(
                    img, _reference_from_cta(image), sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0, img.GetPixelID(),
                )
            except Exception as exc:
                log.warning("External mask resample failed: %s", exc)
                return None
        arr = sitk.GetArrayFromImage(img).astype(bool)
        if not arr.any():
            return None
        return AirwayMaskInfo(
            mask_zyx=arr,
            method=self.name,
            confidence="medium",
            notes="External user-provided mask.",
        )

    def get_landmarks(self, image: CTAImage) -> SharedAirwayLandmarks:
        return SharedAirwayLandmarks()

    def get_existing_features(self) -> SharedAirwayFeatures:
        return SharedAirwayFeatures()


# ---------------------------------------------------------------------------
# CTAFallbackAirwayAdapter — produce a mask from the CTA itself.
# ---------------------------------------------------------------------------

class CTAFallbackAirwayAdapter:
    """Threshold + connected-component fallback airway segmentation.

    Approach:
      1. Threshold the CTA below `air_hu_max` (default −500 HU).
      2. Erase external air by clearing voxels that touch the image border
         in the axial plane.
      3. Connected components — keep the component whose centroid is closest
         to the cranio-caudal mid-axis of the upper image. This selects the
         pharyngeal column, not the trachea (when present) or mastoid air.
      4. Light morphological closing to bridge sub-millimetre gaps.

    Flagged as `threshold_connected_component` in every downstream output;
    not a substitute for a trained airway model.
    """
    name = "threshold_connected_component"

    def __init__(self, air_hu_max: float, min_component_volume_ml: float,
                 closing_mm: float, min_vertical_extent_mm: float = 60.0) -> None:
        self.air_hu_max = float(air_hu_max)
        self.min_component_volume_ml = float(min_component_volume_ml)
        self.closing_mm = float(closing_mm)
        # The pharynx + trachea tree is always many cm tall. Anything shorter
        # is a sinus pocket or sub-glottic remnant: reject it.
        self.min_vertical_extent_mm = float(min_vertical_extent_mm)

    def is_available(self) -> bool:
        return True

    def get_payload(self, image: CTAImage) -> SharedAirwayPayload:
        return SharedAirwayPayload(
            mask_path=None,
            landmarks=SharedAirwayLandmarks(),
            features=SharedAirwayFeatures(),
            source=self.name,
            notes="Fallback threshold + connected-component airway.",
        )

    def get_airway_mask(self, image: CTAImage) -> Optional[AirwayMaskInfo]:
        """Body-envelope airway extraction.

        On a real head/neck CTA the entire external-air + nasopharynx + mouth
        + trachea + lungs is one giant connected component. Lateral-border
        clearing wipes the pharynx along with it. The fix:

          1. Compute the patient body silhouette (soft-tissue threshold,
             largest CC, fill-holes per axial slice — same routine used by
             the fat module). This silhouette by construction excludes
             external air but INCLUDES the airway lumen because
             `binary_fill_holes` patches it shut.
          2. Subtract the silhouette's "solid tissue" voxels: what remains is
             internal-air = airway + sinuses + ears + (lungs if in FOV).
          3. Connected components of internal-air; pick the largest one with
             vertical extent above `min_vertical_extent_mm`. On a head-neck
             CTA the pharynx + trachea is reliably the tallest internal air
             tube; sinuses are short.
        """
        from scipy import ndimage
        arr = image.array
        soft = arr > -250  # soft-tissue threshold — matches `body_air_threshold_hu` default
        if not soft.any():
            return None
        labeled, n = ndimage.label(soft)
        if n == 0:
            return None
        sizes = ndimage.sum_labels(np.ones_like(soft), labeled, range(1, n + 1))
        body_id = int(np.argmax(sizes)) + 1
        body = (labeled == body_id)
        # Fill the body silhouette per axial slice so internal air becomes part
        # of "body". The actual air voxels we want are then body ∩ HU<air_hu_max.
        filled = np.zeros_like(body)
        for z in range(body.shape[0]):
            filled[z] = ndimage.binary_fill_holes(body[z])
        internal_air = filled & (arr < self.air_hu_max)
        if not internal_air.any():
            return None

        labeled2, n2 = ndimage.label(internal_air)
        if n2 == 0:
            return None
        vox_ml = image.voxel_volume_mm3 / 1000.0
        min_size = max(1, int(self.min_component_volume_ml / max(vox_ml, 1e-6)))
        dz_mm = float(image.spacing_xyz_mm[2])
        min_vox_extent = max(1, int(self.min_vertical_extent_mm / max(dz_mm, 1e-6)))

        best_id = None
        best_score = -1
        fallback_id = None
        fallback_size = -1
        comp_sizes = ndimage.sum_labels(
            np.ones_like(internal_air), labeled2, range(1, n2 + 1)
        )
        # Iterate largest-first so we can early-exit on the airway tree.
        for comp_id in (int(np.argsort(-comp_sizes)[k]) + 1 for k in range(n2)):
            size = int(comp_sizes[comp_id - 1])
            if size < min_size:
                break  # remaining components are even smaller
            if size > fallback_size:
                fallback_size = size
                fallback_id = comp_id
            zs = np.where((labeled2 == comp_id).any(axis=(1, 2)))[0]
            extent = int(zs.max() - zs.min() + 1)
            if extent < min_vox_extent:
                continue
            score = size * extent
            if score > best_score:
                best_score = score
                best_id = comp_id

        if best_id is None:
            if fallback_id is None:
                return None
            best_id = fallback_id
            extent_note = (
                f" (no component met min vertical extent "
                f"{self.min_vertical_extent_mm:.0f} mm; used largest survivor)"
            )
        else:
            extent_note = ""

        mask = (labeled2 == best_id)
        if self.closing_mm > 0:
            sx_mm, sy_mm, _ = image.spacing_xyz_mm
            r = max(1, int(round(self.closing_mm / max(min(sx_mm, sy_mm), 1e-6))))
            mask = ndimage.binary_closing(mask, iterations=r)
        return AirwayMaskInfo(
            mask_zyx=mask.astype(bool),
            method=self.name,
            confidence="low",
            notes=(
                f"Body silhouette ∩ HU<{self.air_hu_max:.0f}; "
                f"largest internal-air connected component by size×z-extent, "
                f"min vertical extent {self.min_vertical_extent_mm:.0f} mm"
                f"{extent_note}."
            ),
        )

    def get_landmarks(self, image: CTAImage) -> SharedAirwayLandmarks:
        return SharedAirwayLandmarks()

    def get_existing_features(self) -> SharedAirwayFeatures:
        return SharedAirwayFeatures()


# ---------------------------------------------------------------------------
# NullAirwayAdapter — for cases where no airway segmentation is possible.
# ---------------------------------------------------------------------------

class NullAirwayAdapter:
    name = "null"

    def is_available(self) -> bool:
        return True

    def get_payload(self, image: CTAImage) -> SharedAirwayPayload:
        return SharedAirwayPayload(source=self.name, notes="No airway mask available.")

    def get_airway_mask(self, image: CTAImage) -> Optional[AirwayMaskInfo]:
        return None

    def get_landmarks(self, image: CTAImage) -> SharedAirwayLandmarks:
        return SharedAirwayLandmarks()

    def get_existing_features(self) -> SharedAirwayFeatures:
        return SharedAirwayFeatures()


# ---------------------------------------------------------------------------
# Factory: pick the first available adapter following user config priority.
# ---------------------------------------------------------------------------

def build_airway_provider_chain(cfg) -> list[AirwayProvider]:
    """Return providers in priority order, given a PipelineConfig.airway block."""
    providers: list[AirwayProvider] = []
    a = cfg.airway
    if a.use_existing_dental_airway_outputs:
        providers.append(DentalAirwayAdapter(
            mask_path=a.dental_airway_mask_path,
            landmarks_path=a.dental_landmarks_path,
            features_path=a.dental_features_path,
        ))
    if a.fallback_method == "external_mask_only" and a.external_mask_path:
        providers.append(ExternalMaskAdapter(a.external_mask_path))
    if a.fallback_method == "threshold_connected_component":
        providers.append(CTAFallbackAirwayAdapter(
            air_hu_max=cfg.hu.air_hu_max,
            min_component_volume_ml=a.min_component_volume_ml,
            closing_mm=a.morphology_closing_mm,
        ))
    providers.append(NullAirwayAdapter())
    return providers


def first_available(
    providers: list[AirwayProvider],
    image: CTAImage,
) -> tuple[Optional[AirwayMaskInfo], SharedAirwayPayload]:
    """Walk providers; return (mask, payload). mask may be None if every
    provider declined, in which case the payload's source records the last
    attempt."""
    for p in providers:
        if not p.is_available():
            continue
        mask = p.get_airway_mask(image)
        payload = p.get_payload(image)
        if mask is not None:
            return mask, payload
        if isinstance(p, NullAirwayAdapter):
            return None, payload
    return None, SharedAirwayPayload(source="none")


def _reference_from_cta(image: CTAImage) -> "sitk.Image":
    """Empty sitk image with the consuming CTA's geometry for resampling."""
    arr = np.zeros_like(image.array, dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(image.spacing_xyz_mm)
    img.SetOrigin(image.origin_xyz_mm)
    img.SetDirection(image.direction_3x3)
    return img
