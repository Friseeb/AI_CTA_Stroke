"""QC: decide whether a CTA can plausibly support OSA-style features.

This is intentionally permissive: QC never raises. It returns flags and a
coverage score so the orchestrator can still extract the features that ARE
possible. Cases that fully fail QC are still recorded — they just carry
`qc_pass=False` and `qc_failure_reasons=[…]` for transparency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CoverageRequirements, QCConfig
from .logging_utils import get_logger
from .shared_schema import SharedAirwayLandmarks
from .types import AirwayMaskInfo, CTAImage, QCResult

log = get_logger("qc")


def run_qc(
    image: CTAImage,
    coverage_cfg: CoverageRequirements,
    qc_cfg: QCConfig,
    airway: Optional[AirwayMaskInfo],
    landmarks: SharedAirwayLandmarks,
) -> QCResult:
    failures: list[str] = []
    warnings: list[str] = []
    sx, sy, sz = image.spacing_xyz_mm
    sz_image = image.shape_zyx[0]

    # ---- Spacing / thickness ----
    if sz > qc_cfg.max_slice_thickness_mm:
        warnings.append(
            f"Slice thickness {sz:.2f}mm exceeds max {qc_cfg.max_slice_thickness_mm}mm"
        )

    # ---- Z extent ----
    z_extent_mm = sz * sz_image
    if z_extent_mm < qc_cfg.min_z_extent_mm:
        failures.append(
            f"Z extent {z_extent_mm:.1f}mm below min {qc_cfg.min_z_extent_mm}mm"
        )

    # ---- Coverage signals (heuristic; no atlas) ----
    has_upper_airway = airway is not None and airway.is_present
    has_hyoid_region = landmarks.hyoid is not None
    has_epiglottis_region = landmarks.epiglottis_tip is not None
    has_cervical_soft = _has_cervical_soft_tissue(image)
    if coverage_cfg.include_hard_palate == "required" and landmarks.posterior_nasal_spine is None:
        failures.append("Required hard palate / PNS landmark missing.")
    if coverage_cfg.include_hyoid == "required" and not has_hyoid_region:
        failures.append("Required hyoid landmark missing.")
    if coverage_cfg.include_epiglottis == "required" and not has_epiglottis_region:
        failures.append("Required epiglottis landmark missing.")
    if coverage_cfg.include_cervical_soft_tissues == "required" and not has_cervical_soft:
        failures.append("Required cervical soft-tissue coverage not detected.")

    if not has_upper_airway:
        warnings.append("No airway mask produced; airway features will be missing.")

    # ---- Truncation: any axial slice where body silhouette touches L/R borders ----
    truncation_flag = _detect_lateral_truncation(image, qc_cfg.dental_artifact_hu_threshold)
    if truncation_flag:
        warnings.append("Likely lateral truncation detected on some axial slices.")

    # ---- Dental artifact estimate ----
    artifact_score = _dental_artifact_score(image, qc_cfg.dental_artifact_hu_threshold)
    if artifact_score is not None and artifact_score > qc_cfg.dental_artifact_voxel_fraction_warn:
        warnings.append(f"Dental artifact voxel fraction high: {artifact_score:.4f}")

    # ---- Coverage score (rough composite) ----
    coverage_score = _coverage_score(
        has_upper_airway, has_cervical_soft,
        has_hyoid_region, has_epiglottis_region, truncation_flag,
    )

    return QCResult(
        qc_pass=not failures,
        qc_warning_count=len(warnings),
        qc_failure_reasons=failures,
        qc_coverage_score=coverage_score,
        qc_artifact_score=artifact_score,
        has_upper_airway_region=has_upper_airway,
        has_cervical_soft_tissue=has_cervical_soft,
        has_hyoid_region=has_hyoid_region,
        has_epiglottis_region=has_epiglottis_region,
        truncation_flag=truncation_flag,
        spacing_x_mm=float(sx),
        spacing_y_mm=float(sy),
        spacing_z_mm=float(sz),
        contrast_enhanced=image.is_contrast_enhanced,
        extra={"warnings": warnings, "z_extent_mm": round(z_extent_mm, 1)},
    )


def enrich_qc_row(
    row: dict,
    *,
    landmarks,
    masks_present: dict[str, bool],
    feature_rows: dict[str, dict],
) -> dict:
    """Add v2 per-region coverage + reliability flags to an existing QC row.

    Inputs are intentionally loose to keep this callable from anywhere the
    orchestrator has already gathered the pieces. Every flag is False
    unless we can affirmatively prove the region is present.
    """
    pts = getattr(landmarks, "points", {}) if landmarks else {}
    z_levels = getattr(landmarks, "z_levels", {}) if landmarks else {}

    def _has_point(name: str) -> bool:
        p = pts.get(name)
        return bool(p and (p.voxel_zyx or p.physical_mm))

    def _has_level(name: str) -> bool:
        l = z_levels.get(name)
        return bool(l and (l.z_voxel is not None or l.z_physical_mm is not None))

    row.setdefault("qc_has_hard_palate_region",
                   _has_level("hard_palate_plane") or _has_point("posterior_nasal_spine"))
    row.setdefault("qc_has_retropalatal_region", _has_level("retropalatal_level"))
    row.setdefault("qc_has_retroglossal_region",
                   _has_level("retroglossal_level") or _has_point("epiglottis_tip"))
    row.setdefault("qc_has_tongue_region", bool(masks_present.get("tongue", False)))
    row.setdefault("qc_has_tongue_base_region",
                   _has_level("tongue_base_level") or _has_point("tongue_base_point"))
    row.setdefault("qc_has_soft_palate_region",
                   bool(masks_present.get("soft_palate", False))
                   or _has_point("uvula_tip"))
    row.setdefault("qc_has_mandible_region",
                   bool(masks_present.get("mandible", False)))
    row.setdefault("qc_has_parapharyngeal_region",
                   bool(masks_present.get("airway", False)))
    row.setdefault("qc_has_retropharyngeal_region",
                   bool(masks_present.get("airway", False)))

    # Mask availability mirrors masks_present
    row["qc_airway_mask_available"] = bool(masks_present.get("airway", False))
    row["qc_tongue_mask_available"] = bool(masks_present.get("tongue", False))
    row["qc_mandible_mask_available"] = bool(masks_present.get("mandible", False))
    row["qc_hyoid_landmark_available"] = _has_point("hyoid_centroid")
    row["qc_soft_palate_mask_available"] = bool(masks_present.get("soft_palate", False))
    row["qc_fat_mask_available"] = bool(masks_present.get("fat", False))

    # Artefact flag is derived from the existing dental_artifact_score
    art_score = row.get("qc_dental_artifact_score")
    row["qc_dental_artifact_flag"] = (isinstance(art_score, float)
                                       and art_score == art_score
                                       and art_score > 0.001)
    row.setdefault("qc_motion_artifact_flag", False)
    row.setdefault("qc_swallow_artifact_flag", False)
    row.setdefault("qc_low_fov_flag", False)
    row.setdefault("qc_contrast_phase_flag",
                   bool(row.get("qc_contrast_enhanced", False)))
    row.setdefault("qc_streak_artifact_near_tongue_flag", False)

    # Feature-level reliability — default True if the module wrote data,
    # False if it wrote only empties.
    def _ok(d: dict, key: str) -> bool:
        v = d.get(key)
        return isinstance(v, (int, float)) and v == v
    row["airway_features_reliable"] = _ok(feature_rows.get("airway", {}),
                                           "airway_volume_ml")
    row["tongue_features_reliable"] = _ok(feature_rows.get("tongue", {}),
                                           "tongue_volume_ml") or \
        _ok(feature_rows.get("tongue", {}), "tongue_posterior_volume_ml")
    row["posterior_tongue_hu_reliable"] = _ok(feature_rows.get("tongue", {}),
                                                "tongue_posterior_mean_hu")
    row["soft_palate_features_reliable"] = _ok(feature_rows.get("soft_tissue", {}),
                                                 "soft_palate_volume_ml") or \
        _ok(feature_rows.get("soft_tissue", {}), "soft_palate_length_mm")
    row["skeletal_features_reliable"] = _ok(feature_rows.get("skeletal", {}),
                                              "hyoid_to_c3_distance_mm")
    row["parapharyngeal_fat_features_reliable"] = _ok(feature_rows.get("fat", {}),
                                                       "fat_parapharyngeal_total_volume_ml")
    row["cervical_fat_features_reliable"] = _ok(feature_rows.get("fat", {}),
                                                  "fat_cervical_volume_ml")
    return row


def qc_to_row(qc: QCResult) -> dict:
    """Flatten a QCResult into the per-case feature row."""
    return {
        "qc_pass": qc.qc_pass,
        "qc_warning_count": qc.qc_warning_count,
        "qc_failure_reasons": ";".join(qc.qc_failure_reasons) if qc.qc_failure_reasons else "",
        "qc_coverage_score": round(float(qc.qc_coverage_score), 3),
        "qc_dental_artifact_score": (round(float(qc.qc_artifact_score), 5)
                                     if qc.qc_artifact_score is not None else float("nan")),
        "qc_has_upper_airway": bool(qc.has_upper_airway_region),
        "qc_has_cervical_soft_tissue": bool(qc.has_cervical_soft_tissue),
        "qc_has_hyoid_region": bool(qc.has_hyoid_region),
        "qc_has_epiglottis_region": bool(qc.has_epiglottis_region),
        "qc_truncation_flag": bool(qc.truncation_flag),
        "qc_spacing_x_mm": qc.spacing_x_mm,
        "qc_spacing_y_mm": qc.spacing_y_mm,
        "qc_spacing_z_mm": qc.spacing_z_mm,
        "qc_contrast_enhanced": qc.contrast_enhanced if qc.contrast_enhanced is not None else False,
        "qc_z_extent_mm": qc.extra.get("z_extent_mm"),
    }


# --- internals --------------------------------------------------------------

def _has_cervical_soft_tissue(image: CTAImage) -> bool:
    """A scan has cervical soft tissue if a meaningful fraction of voxels in
    the lower 2/3 of the image are in muscle / fat HU range."""
    sz = image.shape_zyx[0]
    lo = int(sz * 0.33)
    sub = image.array[lo:]
    soft = (sub > -250) & (sub < 200)
    return float(soft.mean()) > 0.05


def _detect_lateral_truncation(image: CTAImage, bone_hu: float) -> bool:
    """Truncation heuristic: high-HU voxels (bone) reaching the lateral image
    border on many axial slices implies the body is cropped.
    """
    arr = image.array
    border_l = arr[:, :, 0]
    border_r = arr[:, :, -1]
    hit_l = (border_l > bone_hu / 4).any(axis=1)
    hit_r = (border_r > bone_hu / 4).any(axis=1)
    fraction = float((hit_l | hit_r).mean())
    return fraction > 0.20


def _dental_artifact_score(image: CTAImage, hu_threshold: float) -> Optional[float]:
    """Fraction of voxels above an unusually high HU threshold (e.g. > 2500).
    Indicative of metallic dental artifact / streak. Not perfect."""
    arr = image.array
    if arr.size == 0:
        return None
    return float((arr > hu_threshold).mean())


def _coverage_score(has_airway: bool, has_soft: bool, has_hyoid: bool,
                    has_epi: bool, truncation: bool) -> float:
    """Composite 0–1 coverage score; weighted toward airway + soft tissue."""
    score = 0.0
    if has_airway: score += 0.40
    if has_soft:   score += 0.30
    if has_hyoid:  score += 0.10
    if has_epi:    score += 0.10
    if not truncation: score += 0.10
    return float(score)
