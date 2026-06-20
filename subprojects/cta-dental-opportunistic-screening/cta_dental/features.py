"""Candidate feature extraction from segmentation outputs.

DISCLAIMER: All outputs are experimental candidate markers, NOT clinical diagnoses.
CTA is not the primary modality for dental disease detection.
Subtle caries, gingivitis, and plaque are NOT reliable CTA endpoints and are not
implemented in this module.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .config import FeaturesConfig
from .geometry import voxel_volume_mm3
from .logging_utils import get_logger

log = get_logger("features")


@functools.lru_cache(maxsize=None)
def _read_label_image_array(path_str: str) -> np.ndarray:
    """Read a label NIfTI as a numpy array, memoized by path.

    Several detectors read the same per-tooth/jaw label files, so caching avoids
    re-reading each file from disk 5+ times per case. The cache is cleared at the
    start of every ``extract_features`` call so it stays per-case (no growth or
    staleness across a batch). Callers copy via ``.astype(...)`` before use, so
    the cached array is never mutated.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path_str))


def _label_array(path) -> np.ndarray:
    return _read_label_image_array(str(path))


NOT_IMPLEMENTED = [
    "subtle caries (CTA evidence limited; not implemented in v1)",
    "gingivitis (not visible on CTA)",
    "dental plaque (not visible on CTA)",
    "mucosal disease (requires clinical/MRI/CT soft-tissue protocol)",
    "pain assessment (clinical finding only)",
    "halitosis (clinical finding only)",
    "mild/moderate periodontal staging (insufficient CTA resolution in v1)",
]

DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS. "
    "All candidate markers are experimental and unvalidated on CTA. "
    "gross caries not implemented in v1; CTA evidence for caries is limited/moderate. "
    "Do not use for patient care decisions."
)

# Full permanent dentition by FDI quadrant (2-digit IDs only)
_UPPER_RIGHT_FDI = frozenset(str(i) for i in range(11, 19))
_UPPER_LEFT_FDI  = frozenset(str(i) for i in range(21, 29))
_LOWER_LEFT_FDI  = frozenset(str(i) for i in range(31, 39))
_LOWER_RIGHT_FDI = frozenset(str(i) for i in range(41, 49))
_UPPER_FDI  = _UPPER_RIGHT_FDI | _UPPER_LEFT_FDI
_LOWER_FDI  = _LOWER_LEFT_FDI | _LOWER_RIGHT_FDI
_WISDOM_FDI = frozenset({"18", "28", "38", "48"})


@dataclass
class CandidateMarker:
    tooth_id: Optional[str]
    location_voxel: Optional[list]
    location_mm: Optional[list]
    volume_mm3: Optional[float]
    mean_hu: Optional[float]
    confidence: str  # "low" | "moderate" — always low in v1
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            k: v for k, v in {
                "tooth_id": self.tooth_id,
                "location_voxel": self.location_voxel,
                "location_mm": self.location_mm,
                "volume_mm3": round(self.volume_mm3, 2) if self.volume_mm3 else None,
                "mean_hu": round(self.mean_hu, 1) if self.mean_hu else None,
                "confidence": self.confidence,
                "notes": self.notes,
            }.items() if v is not None
        }


@dataclass
class FeatureResult:
    case_id: str
    assessability: dict
    candidate_markers: dict
    not_implemented_or_not_reliable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "disclaimer": self.disclaimer,
            "assessability": self.assessability,
            "candidate_markers": self.candidate_markers,
            "not_implemented_or_not_reliable": self.not_implemented_or_not_reliable,
            "warnings": self.warnings,
        }


def extract_features(
    case_id: str,
    hu_image: sitk.Image,
    label_files: dict[str, Path],
    cfg: FeaturesConfig,
    roi_quality: str = "unknown",
    roi_method: str = "unknown",
    domain_warnings: Optional[list[str]] = None,
) -> FeatureResult:
    warnings_out: list[str] = list(domain_warnings or [])
    _read_label_image_array.cache_clear()  # per-case label-array cache
    spacing_xyz = hu_image.GetSpacing()
    spacing_ijk = tuple(reversed(spacing_xyz))
    vox_vol = voxel_volume_mm3(spacing_ijk)

    assessability = _compute_assessability(label_files, roi_quality)

    markers: dict[str, list] = {
        "teeth_present": [],
        "teeth_missing_candidate": [],
        "implants_candidate": [],
        "crowns_or_bridges_candidate": [],
        "root_remnant_candidate": [],
        "periapical_lucency_candidate": [],
        "severe_periodontal_bone_loss_candidate": [],
    }

    if roi_quality == "poor" and not cfg.allow_threshold_fallback_features:
        warnings_out.append(
            "Candidate disease features disabled: ROI was derived from threshold_fallback (poor quality). "
            "Set allow_threshold_fallback_features=true to override."
        )
        return FeatureResult(
            case_id=case_id,
            assessability=assessability,
            candidate_markers=markers,
            not_implemented_or_not_reliable=NOT_IMPLEMENTED,
            warnings=warnings_out,
        )

    hu_arr = sitk.GetArrayFromImage(hu_image).astype(np.float32)

    markers["teeth_present"] = _detect_teeth_present(label_files, voxel_volume_mm3=vox_vol)
    markers["implants_candidate"] = _detect_implants(label_files, hu_arr, vox_vol)
    markers["crowns_or_bridges_candidate"] = _detect_crowns_bridges(label_files, hu_arr, vox_vol)
    markers["periapical_lucency_candidate"] = _detect_periapical_lucency(
        label_files, hu_arr, spacing_ijk, vox_vol, cfg
    )
    markers["severe_periodontal_bone_loss_candidate"] = _detect_periodontal_bone_loss(
        label_files, hu_arr, spacing_ijk, cfg
    )

    if assessability.get("dentition_fov") == "complete":
        markers["teeth_missing_candidate"] = _estimate_missing_teeth(
            label_files, markers["teeth_present"]
        )
    else:
        markers["teeth_missing_candidate"] = [
            {"status": "not_assessable", "reason": "FOV incomplete or unknown"}
        ]

    return FeatureResult(
        case_id=case_id,
        assessability=assessability,
        candidate_markers={k: [m.to_dict() if isinstance(m, CandidateMarker) else m for m in v]
                          for k, v in markers.items()},
        not_implemented_or_not_reliable=NOT_IMPLEMENTED,
        warnings=warnings_out,
    )


def _compute_assessability(label_files: dict[str, Path], roi_quality: str) -> dict:
    names = set(label_files.keys())
    has_upper = any("upper" in n or "maxilla" in n for n in names)
    has_lower = any("lower" in n or "mandible" in n for n in names)

    if has_upper and has_lower:
        fov = "complete"
    elif has_upper or has_lower:
        fov = "partial"
    elif not names:
        fov = "absent"
    else:
        fov = "unknown"

    seg_quality_map = {"good": "good", "fair": "fair", "poor": "poor", "failed": "failed"}
    return {
        "dentition_fov": fov,
        "upper_dentition": has_upper,
        "lower_dentition": has_lower,
        "metal_artifact_severity": "unknown",
        "segmentation_quality": seg_quality_map.get(roi_quality, "unknown"),
    }


def _is_tooth_label(name: str) -> bool:
    """True if label is an individual FDI-numbered tooth (2-digit FDI, not pulp/canal)."""
    if "fdi" not in name:
        return False
    if "pulp" in name or "canal" in name or "sinus" in name:
        return False
    fdi_tag = name.split("fdi")[-1]
    return fdi_tag.isdigit() and len(fdi_tag) == 2


_TOOTH_PRESENT_MIN_MM3 = 30.0
"""A tooth crown is on the order of 100s of mm³; <30 mm³ is noise or an empty
TotalSegmentator label file (the task emits a .nii.gz per class even when no
tooth is detected). Tooth roots alone are also usually well above this floor."""


def _detect_teeth_present(
    label_files: dict[str, Path],
    voxel_volume_mm3: Optional[float] = None,
) -> list[dict]:
    """Detect teeth from segmentation labels.

    A label is counted as present only if its mask contains a non-trivial
    number of voxels — TotalSegmentator's teeth task writes one NIfTI per FDI
    class regardless of whether the tooth actually exists in the image, so
    treating file existence as presence over-reports the dentition.
    """
    result = []

    # Aggregate masks (DentalSegmentator / OralSeg style) — keep file-existence
    # check since those backends only emit the file if something was segmented.
    for label_name in ("teeth_upper", "upper_teeth", "teeth_lower", "lower_teeth"):
        if label_name in label_files:
            jaw = "upper" if "upper" in label_name else "lower"
            result.append({"jaw": jaw, "aggregate_mask": label_name, "fdi_ids": "aggregate_only"})

    # FDI-numbered individual tooth labels — exactly 2-digit FDI (regular teeth only)
    fdi_upper: list[tuple[str, float]] = []
    fdi_lower: list[tuple[str, float]] = []
    empty_labels: list[str] = []

    for label_name, label_path in label_files.items():
        if not _is_tooth_label(label_name):
            continue
        try:
            mask = _label_array(label_path).astype(bool)
        except Exception as exc:
            log.warning("Tooth label read failed (%s): %s", label_name, exc)
            continue
        n_voxels = int(mask.sum())
        volume_mm3 = n_voxels * voxel_volume_mm3 if voxel_volume_mm3 else float(n_voxels)
        if voxel_volume_mm3 and volume_mm3 < _TOOTH_PRESENT_MIN_MM3:
            empty_labels.append(label_name)
            continue
        if n_voxels == 0:
            empty_labels.append(label_name)
            continue
        fdi_tag = label_name.split("fdi")[-1]
        bucket = fdi_upper if ("upper" in label_name or fdi_tag[0] in "12") else fdi_lower
        bucket.append((label_name, round(volume_mm3, 1)))

    def _summarize(jaw: str, entries: list[tuple[str, float]]) -> dict:
        names = [n for n, _ in entries]
        ids = sorted({n.split("fdi")[-1] for n in names}, key=int)
        volumes = {n.split("fdi")[-1]: v for n, v in entries}
        return {
            "jaw": jaw,
            "fdi_labels": names,
            "fdi_ids": ids,
            "volumes_mm3": volumes,
            "count": len(entries),
        }

    if fdi_upper:
        result.append(_summarize("upper", fdi_upper))
    if fdi_lower:
        result.append(_summarize("lower", fdi_lower))
    if empty_labels:
        result.append({
            "note": "FDI labels emitted by segmenter but mask empty / below "
                    f"{_TOOTH_PRESENT_MIN_MM3:.0f} mm³ — treated as not present.",
            "empty_or_sparse_fdi_labels": sorted(empty_labels),
        })

    if not result and label_files:
        result.append({"note": "No recognisable tooth labels found."})
    return result


def _detect_implants(label_files: dict[str, Path], hu_arr: np.ndarray, vox_vol: float) -> list[dict]:
    result = []
    for label_name in ("implant", "implants"):
        if label_name in label_files:
            try:
                mask = _label_array(label_files[label_name]).astype(bool)
                volume = mask.sum() * vox_vol
                mean_hu = float(hu_arr[mask].mean()) if mask.any() else None
                result.append({
                    "label": label_name,
                    "volume_mm3": round(volume, 1),
                    "mean_hu": round(mean_hu, 1) if mean_hu else None,
                    "confidence": "low",
                    "notes": "Implant candidate from segmentation label. Verify radiographically.",
                })
            except Exception as exc:
                log.warning("Implant label read failed: %s", exc)
    return result


def _detect_crowns_bridges(label_files: dict[str, Path], hu_arr: np.ndarray, vox_vol: float) -> list[dict]:
    result = []
    for label_name in ("crown", "bridge", "crowns", "bridges", "crown_or_bridge"):
        if label_name in label_files:
            try:
                mask = _label_array(label_files[label_name]).astype(bool)
                volume = mask.sum() * vox_vol
                result.append({
                    "label": label_name,
                    "volume_mm3": round(volume, 1),
                    "confidence": "low",
                    "notes": "Crown/bridge candidate from segmentation label.",
                })
            except Exception as exc:
                log.warning("Crown/bridge label read failed: %s", exc)
    return result


_PERIAPICAL_EXCLUDE_SUBSTRINGS = (
    "sinus",            # left/right maxillary sinus
    "pharynx",
    "nasal_cavity",
    "oral_cavity",
    "airway",
    "canal",            # inferior alveolar, lingual, incisive canals
)


def _build_periapical_exclusion_mask(
    label_files: dict[str, Path],
    shape: tuple,
    spacing_ijk: tuple,
    dilation_mm: float,
) -> tuple[np.ndarray, list[str]]:
    """Union of anatomical air-space / neurovascular labels, optionally dilated.

    Voxels inside this mask are excluded from the periapical lucency search
    because the low-HU values there reflect anatomy, not pathology.
    """
    exclude = np.zeros(shape, dtype=bool)
    used: list[str] = []
    for label_name, label_path in label_files.items():
        if not any(tag in label_name for tag in _PERIAPICAL_EXCLUDE_SUBSTRINGS):
            continue
        try:
            arr = _label_array(label_path).astype(bool)
        except Exception as exc:
            log.warning("Could not load exclusion label %s: %s", label_name, exc)
            continue
        if arr.shape != shape or not arr.any():
            continue
        exclude |= arr
        used.append(label_name)
    if exclude.any() and dilation_mm > 0:
        dil_vox = max(1, int(round(dilation_mm / min(spacing_ijk))))
        exclude = ndimage.binary_dilation(exclude, iterations=dil_vox)
    return exclude, used


def _detect_periapical_lucency(
    label_files: dict[str, Path],
    hu_arr: np.ndarray,
    spacing_ijk: tuple,
    vox_vol: float,
    cfg: FeaturesConfig,
) -> list[dict]:
    """Experimental periapical lucency detection around tooth apex regions.

    The search region for each tooth = (dilated tooth mask) − (tooth itself)
    − (anatomical exclusion mask: sinuses / pharynx / canals).
    Inside that region a voxel is a lucency candidate iff its HU lies in the
    band (air_threshold, low_hu_threshold) — i.e. lower than soft tissue but
    not as low as pure air, ruling out gas pockets.
    """
    result = []
    tooth_labels = {k: v for k, v in label_files.items()
                    if any(t in k for t in ("teeth", "tooth", "upper_teeth", "lower_teeth", "fdi"))
                    and "pulp" not in k and "canal" not in k and "sinus" not in k}

    if not tooth_labels:
        return [{"status": "not_assessable", "reason": "No tooth labels available for apex localisation."}]

    exclusion_mask, exclusion_labels = _build_periapical_exclusion_mask(
        label_files=label_files,
        shape=hu_arr.shape,
        spacing_ijk=spacing_ijk,
        dilation_mm=cfg.periapical_anatomy_exclusion_mm,
    )
    if exclusion_labels:
        log.info(
            "Periapical anatomical exclusion built from %d labels (%.1f mm dilation): %s",
            len(exclusion_labels), cfg.periapical_anatomy_exclusion_mm,
            ", ".join(sorted(exclusion_labels)),
        )

    search_radius_vox = [max(1, int(cfg.periapical_search_radius_mm / s)) for s in spacing_ijk]
    hu_band = (hu_arr < cfg.periapical_low_hu_threshold) & (hu_arr > cfg.periapical_air_hu_threshold)

    for label_name, label_path in tooth_labels.items():
        try:
            tooth_mask = _label_array(label_path).astype(bool)
            if not tooth_mask.any():
                continue

            dilated = ndimage.binary_dilation(tooth_mask, iterations=max(search_radius_vox))
            search_region = dilated & ~tooth_mask
            if exclusion_mask.any():
                search_region &= ~exclusion_mask
            lucency_voxels = search_region & hu_band

            if not lucency_voxels.any():
                continue

            labeled, n = ndimage.label(lucency_voxels)
            for comp_id in range(1, n + 1):
                comp_mask = labeled == comp_id
                volume = comp_mask.sum() * vox_vol
                if volume < cfg.periapical_min_volume_mm3:
                    continue
                centroid = np.array(ndimage.center_of_mass(comp_mask)).tolist()
                mean_hu = float(hu_arr[comp_mask].mean())
                result.append(CandidateMarker(
                    tooth_id=None,
                    location_voxel=[round(c) for c in centroid],
                    location_mm=None,
                    volume_mm3=volume,
                    mean_hu=mean_hu,
                    confidence="low",
                    notes=(
                        f"Experimental periapical lucency candidate adjacent to {label_name}. "
                        f"HU band ({cfg.periapical_air_hu_threshold:.0f}, "
                        f"{cfg.periapical_low_hu_threshold:.0f}); anatomical air-spaces "
                        "(sinus/pharynx/canals) excluded. NOT a diagnosis — verify with "
                        "dental radiographs."
                    ),
                ).to_dict())
        except Exception as exc:
            log.warning("Periapical lucency detection failed for %s: %s", label_name, exc)

    if not result:
        result.append({
            "status": "none_detected",
            "confidence": "low",
            "exclusion_labels_used": sorted(exclusion_labels),
        })
    return result


def _detect_periodontal_bone_loss(
    label_files: dict[str, Path],
    hu_arr: np.ndarray,
    spacing_ijk: tuple,
    cfg: FeaturesConfig,
) -> list[dict]:
    """Experimental severe periodontal bone-loss candidate detection via periradicular shell coverage."""
    tooth_labels = {k: v for k, v in label_files.items() if _is_tooth_label(k)}
    upper_bone_path = label_files.get("upper_jawbone")
    lower_bone_path = label_files.get("lower_jawbone")

    if not tooth_labels:
        return [{"status": "not_assessable", "reason": "No FDI tooth labels found."}]
    if not upper_bone_path and not lower_bone_path:
        return [{"status": "not_assessable",
                 "reason": "No jawbone labels (upper_jawbone, lower_jawbone) found."}]

    # Load bone masks once
    upper_bone: Optional[np.ndarray] = None
    lower_bone: Optional[np.ndarray] = None
    try:
        if upper_bone_path:
            upper_bone = _label_array(upper_bone_path).astype(bool)
    except Exception as exc:
        log.warning("Failed to load upper_jawbone: %s", exc)
    try:
        if lower_bone_path:
            lower_bone = _label_array(lower_bone_path).astype(bool)
    except Exception as exc:
        log.warning("Failed to load lower_jawbone: %s", exc)

    # Shell thickness in voxels (isotropic assumption: use smallest spacing)
    shell_vox = max(1, int(cfg.periodontal_bone_shell_mm / min(spacing_ijk)))
    struct = ndimage.generate_binary_structure(3, 1)

    result = []
    for label_name, label_path in sorted(tooth_labels.items()):
        fdi_tag = label_name.split("fdi")[-1]
        if fdi_tag[0] in "12":
            bone_mask = upper_bone
            jaw = "upper"
        else:
            bone_mask = lower_bone
            jaw = "lower"

        if bone_mask is None:
            continue

        try:
            tooth_mask = _label_array(label_path).astype(bool)
            if not tooth_mask.any():
                continue

            # Periradicular shell = dilated tooth minus tooth itself
            dilated = ndimage.binary_dilation(tooth_mask, structure=struct, iterations=shell_vox)
            shell = dilated & ~tooth_mask
            shell_size = int(shell.sum())
            if shell_size == 0:
                continue

            coverage = float((shell & bone_mask).sum()) / shell_size

            if coverage < cfg.periodontal_min_bone_coverage:
                centroid = [round(c) for c in ndimage.center_of_mass(tooth_mask)]
                result.append({
                    "fdi_label": label_name,
                    "fdi_id": fdi_tag,
                    "jaw": jaw,
                    "periradicular_bone_coverage": round(coverage, 3),
                    "location_voxel": centroid,
                    "confidence": "low",
                    "notes": (
                        f"Experimental: {coverage * 100:.0f}% of the {cfg.periodontal_bone_shell_mm:.0f}mm "
                        f"periradicular shell overlaps the jawbone label. "
                        "Low coverage may indicate severe bone loss OR segmentation edge effect "
                        "(particularly for third molars). "
                        "Verify with periodontal radiographs. NOT a diagnosis."
                    ),
                })
        except Exception as exc:
            log.warning("Bone loss check failed for %s: %s", label_name, exc)

    if not result:
        return [{
            "status": "no_candidates",
            "confidence": "low",
            "notes": (
                f"No teeth with periradicular bone coverage below "
                f"{cfg.periodontal_min_bone_coverage * 100:.0f}% threshold detected."
            ),
        }]
    return result


def _estimate_missing_teeth(label_files: dict[str, Path], teeth_present: list) -> list[dict]:
    """Identify FDI positions absent from segmentation output."""
    present_upper: set[str] = set()
    present_lower: set[str] = set()
    has_fdi = False

    for entry in teeth_present:
        fdi_ids = entry.get("fdi_ids")
        if not fdi_ids or fdi_ids == "aggregate_only":
            continue
        has_fdi = True
        jaw = entry.get("jaw", "")
        for fdi_id in fdi_ids:
            sid = str(fdi_id)
            if not sid.isdigit() or len(sid) != 2:
                continue
            if jaw == "upper":
                present_upper.add(sid)
            elif jaw == "lower":
                present_lower.add(sid)

    if not has_fdi:
        return [{"status": "not_assessable",
                 "reason": "FDI-level tooth labels not available from current segmentation backend."}]

    result = []
    for expected, present, jaw in [
        (_UPPER_FDI, present_upper, "upper"),
        (_LOWER_FDI, present_lower, "lower"),
    ]:
        if not present:
            # FOV may genuinely not cover this jaw
            continue
        for fdi_id in sorted(expected - present, key=int):
            is_wisdom = fdi_id in _WISDOM_FDI
            result.append({
                "fdi_id": fdi_id,
                "jaw": jaw,
                "confidence": "low",
                "notes": (
                    "Third molar — commonly absent due to agenesis, impaction, or extraction. "
                    "Verify clinically."
                    if is_wisdom else
                    "Missing tooth candidate — consider extraction, agenesis, or impaction. "
                    "Verify clinically and with dedicated dental radiographs."
                ),
            })

    if not result:
        return [{
            "status": "none_missing",
            "note": "All expected FDI positions detected by segmentation.",
            "confidence": "low",
        }]
    return result


def write_features_json(result: FeatureResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    log.info("Candidate features written to %s", path)
