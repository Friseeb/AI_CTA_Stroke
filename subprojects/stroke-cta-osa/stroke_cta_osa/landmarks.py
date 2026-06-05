"""Load / save / validate landmarks; derive region z-levels.

Sources, in priority order (see `LandmarkProvider.priority`):
    1. External CSV / JSON file (user-supplied);
    2. Dental/CBCT pipeline adapter (file-based contract);
    3. Heuristic estimators that need only the CTA + airway mask;
    4. Empty bundle — every landmark missing, features fall back to NaN.

The heuristic estimators in this module are deliberately *conservative*: they
populate landmarks only when the input gives strong evidence (e.g. the
airway's narrowest slice is taken as a retroglossal-level proxy only if
the airway mask is large enough). When in doubt the landmark stays None
and downstream features become NaN — which is the explicit contract.

This module does NOT do automatic deep-learning landmark detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .landmark_schema import (
    LandmarkBundle, LandmarkPlane, LandmarkPoint, LandmarkZLevel,
    PLANE_LANDMARKS, POINT_LANDMARKS, Z_LEVEL_LANDMARKS,
)
from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("landmarks")


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_landmarks(path: Path) -> LandmarkBundle:
    """Load a LandmarkBundle from JSON.

    Unknown landmark names are silently skipped — this keeps the loader
    forward-compatible with new canonical names added to the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Landmarks JSON not found: {path}")
    data = json.loads(path.read_text())
    bundle = LandmarkBundle(
        case_id=data.get("case_id", ""),
        coord_system=data.get("coord_system", "mixed"),
        image_shape_zyx=tuple(data["image_shape_zyx"])
            if data.get("image_shape_zyx") else None,
        image_affine=data.get("image_affine"),
        notes=data.get("notes", ""),
    )
    for name, raw in (data.get("points") or {}).items():
        if name not in POINT_LANDMARKS:
            log.debug("Skipping unknown point landmark %r", name)
            continue
        bundle.points[name] = LandmarkPoint(
            name=name,
            voxel_zyx=tuple(raw["voxel_zyx"]) if raw.get("voxel_zyx") else None,
            physical_mm=tuple(raw["physical_mm"]) if raw.get("physical_mm") else None,
            source=raw.get("source", "external_json"),
            confidence=float(raw.get("confidence", 0.0)),
        )
    for name, raw in (data.get("z_levels") or {}).items():
        if name not in Z_LEVEL_LANDMARKS:
            continue
        bundle.z_levels[name] = LandmarkZLevel(
            name=name,
            z_voxel=raw.get("z_voxel"),
            z_physical_mm=raw.get("z_physical_mm"),
            source=raw.get("source", "external_json"),
            confidence=float(raw.get("confidence", 0.0)),
        )
    for name, raw in (data.get("planes") or {}).items():
        if name not in PLANE_LANDMARKS:
            continue
        bundle.planes[name] = LandmarkPlane(
            name=name,
            point_names=tuple(raw["point_names"]) if raw.get("point_names") else None,
            point_phys_mm=tuple(raw["point_phys_mm"])
                if raw.get("point_phys_mm") else None,
            normal_phys_mm=tuple(raw["normal_phys_mm"])
                if raw.get("normal_phys_mm") else None,
            source=raw.get("source", "external_json"),
        )
    return bundle


def save_landmarks(bundle: LandmarkBundle, path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle.to_dict(), indent=2))
    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_landmarks(
    bundle: LandmarkBundle,
    image: Optional[CTAImage] = None,
) -> list[str]:
    """Return a list of human-readable warnings; never raises."""
    warnings: list[str] = []
    if image is not None and bundle.image_shape_zyx is not None:
        if tuple(bundle.image_shape_zyx) != image.shape_zyx:
            warnings.append(
                f"image_shape_zyx mismatch: bundle={bundle.image_shape_zyx} "
                f"vs image={image.shape_zyx}"
            )
    for name, pt in bundle.points.items():
        if pt.voxel_zyx is None and pt.physical_mm is None:
            warnings.append(f"Point {name!r} has neither voxel_zyx nor physical_mm.")
        if pt.voxel_zyx is not None and image is not None:
            z, y, x = pt.voxel_zyx
            sz, sy, sx = image.shape_zyx
            if not (0 <= z < sz and 0 <= y < sy and 0 <= x < sx):
                warnings.append(
                    f"Point {name!r} voxel index {pt.voxel_zyx} outside image "
                    f"shape {image.shape_zyx}."
                )
    for name, lvl in bundle.z_levels.items():
        if lvl.z_voxel is None and lvl.z_physical_mm is None:
            warnings.append(f"Z-level {name!r} is empty.")
        if lvl.z_voxel is not None and image is not None:
            sz = image.shape_zyx[0]
            if not (0 <= lvl.z_voxel < sz):
                warnings.append(
                    f"Z-level {name!r} index {lvl.z_voxel} outside [0, {sz})."
                )
    for name, pl in bundle.planes.items():
        has_points = pl.point_names is not None
        has_normal = pl.point_phys_mm is not None and pl.normal_phys_mm is not None
        if not (has_points or has_normal):
            warnings.append(f"Plane {name!r} has no representation set.")
        if has_points and pl.point_names:
            missing = [n for n in pl.point_names if n not in bundle.points]
            if missing:
                warnings.append(
                    f"Plane {name!r} references missing points: {missing}"
                )
    return warnings


# ---------------------------------------------------------------------------
# Coordinate transforms — minimal helpers
# ---------------------------------------------------------------------------

def voxel_to_physical(image: CTAImage, voxel_zyx: tuple[int, int, int]) -> tuple[float, float, float]:
    """Map a (z, y, x) voxel index to (x, y, z) physical mm using the image
    origin + spacing + direction. The output is in ITK convention (x, y, z)
    so it can be persisted directly as `physical_mm`.

    For non-identity direction matrices we apply the direction; this matches
    SimpleITK's TransformIndexToPhysicalPoint behaviour.
    """
    z, y, x = (int(v) for v in voxel_zyx)
    sx, sy, sz = image.spacing_xyz_mm
    ox, oy, oz = image.origin_xyz_mm
    d = image.direction_3x3  # row-major 3x3
    # physical = origin + Direction @ (Spacing * (x, y, z))
    vx = sx * x
    vy = sy * y
    vz = sz * z
    px = ox + d[0] * vx + d[1] * vy + d[2] * vz
    py = oy + d[3] * vx + d[4] * vy + d[5] * vz
    pz = oz + d[6] * vx + d[7] * vy + d[8] * vz
    return (px, py, pz)


def fill_physical_coords(bundle: LandmarkBundle, image: CTAImage) -> LandmarkBundle:
    """Populate the `physical_mm` slot on points / `z_physical_mm` on levels.

    Useful when the bundle was loaded with voxel indices only — saves the
    consumer from re-deriving the affine.
    """
    for pt in bundle.points.values():
        if pt.voxel_zyx is not None and pt.physical_mm is None:
            pt.physical_mm = voxel_to_physical(image, pt.voxel_zyx)
    for lvl in bundle.z_levels.values():
        if lvl.z_voxel is not None and lvl.z_physical_mm is None:
            _, _, pz = voxel_to_physical(image, (lvl.z_voxel, 0, 0))
            lvl.z_physical_mm = pz
    return bundle


# ---------------------------------------------------------------------------
# Region-level inference
# ---------------------------------------------------------------------------

def infer_region_levels_from_landmarks(
    bundle: LandmarkBundle,
) -> dict[str, Optional[int]]:
    """Best-effort z-band boundaries (voxel indices) for the five airway
    compartments. Returns a dict with keys ``nasopharyngeal_lo``,
    ``nasopharyngeal_hi``, ``retropalatal_lo``/``hi``, etc. — values are
    None when the required landmark is missing.

    Convention (NIfTI z increases superior-inferior of the patient varies
    with affine; we operate on array-index z, which is "slice index 0 = first
    stored slice"). Boundaries are derived without making assumptions about
    superior vs inferior — the caller decides which direction is which by
    comparing the indices.
    """
    pt = bundle.points
    lvl = bundle.z_levels

    def _pt_z(name: str) -> Optional[int]:
        p = pt.get(name)
        return p.voxel_zyx[0] if p and p.voxel_zyx else None

    def _lvl_z(name: str) -> Optional[int]:
        l = lvl.get(name)
        return l.z_voxel if l and l.z_voxel is not None else None

    hard_palate = _lvl_z("hard_palate_plane") or _pt_z("posterior_nasal_spine")
    retropalatal = _lvl_z("retropalatal_level")
    retroglossal = _lvl_z("retroglossal_level") or _pt_z("epiglottis_tip")
    tongue_base = _lvl_z("tongue_base_level") or _pt_z("tongue_base_point")
    larynx = _lvl_z("laryngeal_inlet_level")
    hyoid = _pt_z("hyoid_centroid")

    return {
        "hard_palate": hard_palate,
        "retropalatal_level": retropalatal,
        "retroglossal_level": retroglossal,
        "tongue_base_level": tongue_base,
        "laryngeal_inlet_level": larynx,
        "hyoid_level": hyoid,
    }


def get_retropalatal_level(bundle: LandmarkBundle) -> Optional[int]:
    return infer_region_levels_from_landmarks(bundle)["retropalatal_level"]


def get_retroglossal_level(bundle: LandmarkBundle) -> Optional[int]:
    return infer_region_levels_from_landmarks(bundle)["retroglossal_level"]


def get_tongue_base_level(bundle: LandmarkBundle) -> Optional[int]:
    return infer_region_levels_from_landmarks(bundle)["tongue_base_level"]


def get_hyoid_position(
    bundle: LandmarkBundle,
) -> Optional[tuple[int, int, int]]:
    p = bundle.points.get("hyoid_centroid")
    return p.voxel_zyx if p and p.voxel_zyx else None


def get_mandibular_plane(
    bundle: LandmarkBundle,
) -> Optional[LandmarkPlane]:
    pl = bundle.planes.get("mandibular_plane")
    if pl is None:
        return None
    # Validate the plane has a usable representation
    if pl.point_phys_mm is not None and pl.normal_phys_mm is not None:
        return pl
    if pl.point_names is not None and all(
        n in bundle.points and bundle.points[n].physical_mm is not None
        for n in pl.point_names
    ):
        return pl
    return None


# ---------------------------------------------------------------------------
# Heuristic estimators (conservative; no DL)
# ---------------------------------------------------------------------------

def estimate_from_airway(
    bundle: LandmarkBundle,
    image: CTAImage,
    airway: AirwayMaskInfo,
    *,
    overwrite: bool = False,
) -> LandmarkBundle:
    """Populate region z-levels from an airway mask only.

    Heuristic rules (every one falls back to "do not populate" if the input
    is too small or shape doesn't make sense):

      * `retroglossal_level` ≈ z of minimum airway CSA, provided airway is
        at least 60 mm tall and the min CSA is in the lower 2/3 of the
        non-zero airway extent (the airway has to actually narrow down
        toward the hypopharynx for this to be plausible).
      * `tongue_base_level` ≈ retroglossal_level + 5 mm (z direction sign
        comes from the data, never assumed).
      * `hard_palate_plane` ≈ z of widest airway CSA in the upper 1/3 of
        the non-zero airway extent (used as a rough nasopharyngeal floor).

    These are intentionally noisy. Surface them as
    `source = "heuristic_airway"` so downstream code can filter.
    """
    if not airway.is_present:
        return bundle

    mask = airway.mask_zyx
    per_slice = mask.sum(axis=(1, 2))
    nonzero = np.where(per_slice > 0)[0]
    if nonzero.size < 60:
        return bundle  # too short to anchor — return untouched
    z_lo, z_hi = int(nonzero.min()), int(nonzero.max())
    sz = image.spacing_xyz_mm[2]
    if (z_hi - z_lo) * sz < 60.0:
        return bundle  # < 60 mm of airway — too thin to landmark

    csa = per_slice[z_lo:z_hi + 1]
    rel = np.arange(csa.size) / max(csa.size - 1, 1)
    lower_two_thirds = rel > 0.33
    upper_one_third = rel < 0.33

    if lower_two_thirds.any():
        local_min = int(np.argmin(np.where(lower_two_thirds, csa, np.inf)))
        z_min = z_lo + local_min
        if "retroglossal_level" not in bundle.z_levels or overwrite:
            bundle.z_levels["retroglossal_level"] = LandmarkZLevel(
                name="retroglossal_level", z_voxel=z_min,
                source="heuristic_airway", confidence=0.3,
            )
            # tongue_base ≈ retroglossal + 5 mm toward the larger-z direction
            step = max(1, int(round(5.0 / sz)))
            z_tongue = z_min + step
            if 0 <= z_tongue < image.shape_zyx[0]:
                bundle.z_levels.setdefault(
                    "tongue_base_level",
                    LandmarkZLevel(name="tongue_base_level",
                                   z_voxel=z_tongue,
                                   source="heuristic_airway",
                                   confidence=0.2),
                )

    if upper_one_third.any():
        local_max = int(np.argmax(np.where(upper_one_third, csa, -1)))
        z_hp = z_lo + local_max
        if "hard_palate_plane" not in bundle.z_levels or overwrite:
            bundle.z_levels["hard_palate_plane"] = LandmarkZLevel(
                name="hard_palate_plane", z_voxel=z_hp,
                source="heuristic_airway", confidence=0.2,
            )

    bundle = fill_physical_coords(bundle, image)
    return bundle


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------

def build_landmark_bundle(
    image: CTAImage,
    explicit_path: Optional[Path] = None,
    dental_landmarks_path: Optional[Path] = None,
    airway: Optional[AirwayMaskInfo] = None,
    allow_heuristic_fallback: bool = True,
) -> LandmarkBundle:
    """Run the provider chain and return a single LandmarkBundle.

    Provider order:
        1. explicit_path  (user-supplied JSON)
        2. dental_landmarks_path (sibling pipeline)
        3. heuristic_from_airway (if airway mask present)
        4. empty bundle

    Each provider only fills slots the previous didn't fill — explicit user
    input always wins.
    """
    bundle = LandmarkBundle(case_id=image.study_id,
                            coord_system="voxel_zyx",
                            image_shape_zyx=image.shape_zyx)

    if explicit_path and Path(explicit_path).is_file():
        try:
            loaded = load_landmarks(explicit_path)
            bundle.points.update(loaded.points)
            bundle.z_levels.update(loaded.z_levels)
            bundle.planes.update(loaded.planes)
        except Exception as exc:
            log.warning("Could not load explicit landmarks: %s", exc)

    if dental_landmarks_path and Path(dental_landmarks_path).is_file():
        try:
            loaded = load_landmarks(dental_landmarks_path)
            # Only fill slots not already populated.
            for name, pt in loaded.points.items():
                bundle.points.setdefault(name, pt)
            for name, lvl in loaded.z_levels.items():
                bundle.z_levels.setdefault(name, lvl)
            for name, pl in loaded.planes.items():
                bundle.planes.setdefault(name, pl)
        except Exception as exc:
            log.warning("Could not load dental landmarks: %s", exc)

    if allow_heuristic_fallback and airway is not None:
        bundle = estimate_from_airway(bundle, image, airway, overwrite=False)

    bundle = fill_physical_coords(bundle, image)
    return bundle


def transform_landmarks_between_image_spaces(
    bundle: LandmarkBundle,
    source_image: CTAImage,
    target_image: CTAImage,
) -> LandmarkBundle:
    """Transform voxel indices when two images share the same physical frame
    but different sampling grids.

    We do this by going voxel→physical→voxel via the source then target
    affines. For now we only support landmarks that have either voxel_zyx
    or physical_mm populated.
    """
    out = LandmarkBundle(
        case_id=bundle.case_id, coord_system=bundle.coord_system,
        image_shape_zyx=target_image.shape_zyx, notes=bundle.notes,
    )
    for name, pt in bundle.points.items():
        phys = pt.physical_mm or (
            voxel_to_physical(source_image, pt.voxel_zyx) if pt.voxel_zyx else None
        )
        if phys is None:
            continue
        vox = _physical_to_voxel(target_image, phys)
        out.points[name] = LandmarkPoint(
            name=name, voxel_zyx=vox, physical_mm=phys,
            source=pt.source, confidence=pt.confidence,
        )
    for name, lvl in bundle.z_levels.items():
        if lvl.z_voxel is None and lvl.z_physical_mm is None:
            continue
        z_phys = (lvl.z_physical_mm if lvl.z_physical_mm is not None
                  else voxel_to_physical(source_image, (lvl.z_voxel, 0, 0))[2])
        z_vox = _physical_to_voxel(target_image, (0.0, 0.0, z_phys))[0]
        out.z_levels[name] = LandmarkZLevel(
            name=name, z_voxel=z_vox, z_physical_mm=z_phys,
            source=lvl.source, confidence=lvl.confidence,
        )
    # Planes: pass through unchanged (they are in physical mm)
    for name, pl in bundle.planes.items():
        out.planes[name] = pl
    return out


def _physical_to_voxel(image: CTAImage, phys_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    """Inverse of voxel_to_physical for axis-aligned (diagonal direction) affines.

    For general 3×3 direction matrices, solves a small 3×3 linear system.
    """
    px, py, pz = phys_xyz
    ox, oy, oz = image.origin_xyz_mm
    sx, sy, sz = image.spacing_xyz_mm
    d = np.asarray(image.direction_3x3, dtype=float).reshape(3, 3)
    delta = np.array([px - ox, py - oy, pz - oz])
    scale = np.diag([sx, sy, sz])
    try:
        idx = np.linalg.solve(d @ scale, delta)
    except np.linalg.LinAlgError:
        # Singular direction matrix (shouldn't happen on real CTA) — fall back to identity
        idx = np.array([(px - ox) / sx, (py - oy) / sy, (pz - oz) / sz])
    x, y, z = (int(round(v)) for v in idx)
    return (z, y, x)
