"""Top-level case orchestrator.

extract_case() runs one CTA → CaseResult. It exists so the CLI, the batch
runner, and the test suite all hit exactly the same code path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np

from . import PIPELINE_NAME, __version__
from .adapters import build_airway_provider_chain, first_available
from .airway import compute_airway_features
from .airway_regions import AirwayRegionConfig as _AirwayRegionCfg, compute_regional_airway_features
from .composites import CohortStats, CompositeConfig, compute_composites
from .config import PipelineConfig
from .dicom_utils import safe_hash
from .fat import compute_fat_features
from .fat_regions import FatRegionConfig as _FatRegionCfg, compute_regional_fat_features
from .io import load_input, save_mask
from .landmarks import build_landmark_bundle
from .logging_utils import get_logger
from .mandible import (
    MandibleConfig as _MandibleCfg, OralCavityConfig as _OralCfg,
    compute_mandible_features,
)
from .metric_registry import empty_row
from .perivascular import compute_perivascular_features
from .qc import enrich_qc_row, qc_to_row, run_qc
from .qc_slicer import write_slicer_loader
from .radiomics import compute_radiomics
from .rois import body_mask
from .skeletal import SkeletalConfig as _SkeletalCfg, compute_skeletal_features
from .soft_palate import SoftTissueConfig as _SoftTissueCfg, compute_soft_palate_features
from .thoracic import compute_thoracic_features
from .tongue import TongueConfig as _TongueCfg, compute_tongue_features
from .types import CaseResult, CTAImage

log = get_logger("features")


def extract_case(
    input_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    sidecar_path: Optional[Path] = None,
    patient_id: Optional[str] = None,
    scan_id_override: Optional[str] = None,
    *,
    external_airway_mask_path: Optional[Path] = None,
    external_tongue_mask_path: Optional[Path] = None,
    external_mandible_mask_path: Optional[Path] = None,
    external_soft_palate_mask_path: Optional[Path] = None,
    external_oral_cavity_mask_path: Optional[Path] = None,
    external_landmarks_path: Optional[Path] = None,
) -> CaseResult:
    """Run the full feature extraction for one CTA.

    `patient_id` is the human-facing study identifier (e.g. 'sub-547'). If
    omitted, the I/O layer derives an opaque study_id from path/UID hashes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / safe_hash(str(Path(input_path).resolve()))
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    try:
        image, scrubbed_meta = load_input(
            Path(input_path), age_floor_years=cfg.ingestion.age_floor_years,
            sidecar_path=sidecar_path,
        )
    except Exception as exc:
        return _failed_load_result(
            input_path, patient_id, str(exc), cfg=cfg, out_dir=case_dir,
        )

    # 2. Airway adapter chain — explicit external mask wins
    if external_airway_mask_path is not None:
        from .adapters import ExternalMaskAdapter
        airway_info = ExternalMaskAdapter(str(external_airway_mask_path)).get_airway_mask(image)
        payload_source = "external_mask"
        payload_notes = f"explicit --external-airway-mask: {external_airway_mask_path}"
        from .shared_schema import (SharedAirwayPayload, SharedAirwayFeatures,
                                     SharedAirwayLandmarks)
        payload = SharedAirwayPayload(
            mask_path=str(external_airway_mask_path),
            landmarks=SharedAirwayLandmarks(),
            features=SharedAirwayFeatures(),
            source=payload_source, notes=payload_notes,
        )
    else:
        providers = build_airway_provider_chain(cfg)
        airway_info, payload = first_available(providers, image)

    # 3. Landmarks — provider chain: explicit JSON → dental → heuristic → empty
    landmarks_bundle = build_landmark_bundle(
        image=image,
        explicit_path=(external_landmarks_path
                       or (Path(cfg.landmarks.explicit_path)
                           if cfg.landmarks.explicit_path else None)),
        dental_landmarks_path=(Path(cfg.landmarks.dental_landmarks_path)
                                if cfg.landmarks.dental_landmarks_path else None),
        airway=airway_info,
        allow_heuristic_fallback=cfg.landmarks.allow_heuristic_fallback,
    )

    # Legacy shared-schema landmarks (back-compat for compute_airway_features)
    landmarks = payload.landmarks

    # 4. QC
    qc_result = run_qc(image, cfg.coverage, cfg.qc, airway_info, landmarks)
    qc_row = qc_to_row(qc_result)

    # 5. Airway features
    airway_geom = compute_airway_features(
        image=image, mask_info=airway_info, landmarks=landmarks,
        retropalatal_window_mm=cfg.airway.retropalatal_window_mm,
        retroglossal_window_mm=cfg.airway.retroglossal_window_mm,
        retrolingual_window_mm=cfg.airway.retrolingual_window_mm,
    )
    airway_features = dict(airway_geom.features)

    # Pre-computed (shared) features take priority for columns the dental
    # pipeline also produced — recorded with a separate "_from_dental" name
    # so the caller can audit reuse without losing the CTA-side recompute.
    if payload.features.values:
        for k, v in payload.features.values.items():
            airway_features[f"{k}_from_dental"] = float(v)

    # Determine the min CSA z-index used as anchor by fat ROIs
    min_csa_z = airway_features.get("airway_min_csa_slice_index")
    if isinstance(min_csa_z, int) and min_csa_z >= 0:
        anchor_z = min_csa_z
    else:
        anchor_z = None

    # 6. Fat features
    masks_for_radiomics: dict[str, np.ndarray] = {}
    def _save_mask(name: str, mask: np.ndarray) -> None:
        if cfg.output.save_masks:
            try:
                save_mask(mask, image, case_dir / f"mask_{name}.nii.gz")
            except Exception as exc:
                log.warning("Could not save mask %s: %s", name, exc)
        if name in ("fat_cervical_total", "fat_parapharyngeal_total"):
            masks_for_radiomics[name.replace("fat_", "").replace("_total", "")] = mask
        if name == "fat_parapharyngeal_total":
            masks_for_radiomics["parapharyngeal_fat"] = mask
        if name == "fat_cervical_total":
            masks_for_radiomics["cervical_fat"] = mask

    fat_features = compute_fat_features(
        image=image, airway=airway_info, landmarks=landmarks,
        hu_cfg=cfg.hu, fat_cfg=cfg.fat,
        airway_min_csa_z_index=anchor_z,
        save_masks_callback=_save_mask,
    )

    # Save airway mask for radiomics too
    if airway_info is not None and airway_info.is_present:
        masks_for_radiomics["airway"] = airway_info.mask_zyx
        if cfg.output.save_masks:
            try:
                save_mask(airway_info.mask_zyx, image, case_dir / "mask_airway.nii.gz")
            except Exception as exc:
                log.warning("Could not save airway mask: %s", exc)
        if "airway" in cfg.radiomics.rois and "airway" not in masks_for_radiomics:
            masks_for_radiomics["airway"] = airway_info.mask_zyx
        if ("combined_airway_soft_tissue" in cfg.radiomics.rois
                and "fat_parapharyngeal_total" in masks_for_radiomics):
            masks_for_radiomics["combined_airway_soft_tissue"] = (
                airway_info.mask_zyx | masks_for_radiomics["fat_parapharyngeal_total"]
            )

    # ---- New modules (registry-driven; each is an additive dict) ----

    # 6.1 New-module masks should also be available to PyRadiomics. The
    # `_save_mask` closure already populates masks_for_radiomics via name
    # prefixes — extend it here to recognise the new module's mask names.
    _orig_save = _save_mask
    def _save_mask_v2(name: str, mask: np.ndarray) -> None:
        _orig_save(name, mask)
        roi_map = {
            "tongue": "tongue",
            "tongue_posterior": "posterior_tongue",
            "soft_palate": "soft_palate",
            "fat_retropharyngeal_regional": "retropharyngeal_fat",
        }
        roi = roi_map.get(name)
        if roi is not None:
            masks_for_radiomics[roi] = mask
        # Lateral wall: synthesize a combined band from L/R parapharyngeal
        # if both arrive.
        if name in ("fat_parapharyngeal_retropalatal_left",
                    "fat_parapharyngeal_retropalatal_right"):
            existing = masks_for_radiomics.get("lateral_wall")
            masks_for_radiomics["lateral_wall"] = (
                mask if existing is None else (existing | mask)
            )
    _save_mask = _save_mask_v2  # noqa: F841 — keep symbol name to reduce diff

    def _load_external_mask(path: Optional[Path | str]) -> Optional[np.ndarray]:
        if path is None:
            return None
        p = Path(path)
        if not p.is_file():
            log.warning("External mask not found: %s", p)
            return None
        try:
            import SimpleITK as sitk
            m = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(bool)
            if m.shape != image.shape_zyx:
                log.warning("External mask shape %s != image %s — skipping",
                            m.shape, image.shape_zyx)
                return None
            return m
        except Exception as exc:
            log.warning("Could not load external mask %s: %s", p, exc)
            return None

    tongue_mask = _load_external_mask(external_tongue_mask_path
                                       or cfg.tongue.external_mask_path)
    mandible_mask = _load_external_mask(external_mandible_mask_path
                                         or cfg.mandible.external_mask_path)
    oral_cavity_mask = _load_external_mask(external_oral_cavity_mask_path
                                            or cfg.oral_cavity.external_mask_path)
    soft_palate_mask = _load_external_mask(external_soft_palate_mask_path
                                            or cfg.soft_tissue.soft_palate_mask_path)
    uvula_mask = _load_external_mask(cfg.soft_tissue.uvula_mask_path)
    tonsil_l = _load_external_mask(cfg.soft_tissue.palatine_tonsil_left_mask_path)
    tonsil_r = _load_external_mask(cfg.soft_tissue.palatine_tonsil_right_mask_path)

    # 6.2 Mandible (provides volume needed by tongue ratios)
    mandible_features = compute_mandible_features(
        image=image,
        cfg=_MandibleCfg(
            enabled=cfg.mandible.enabled,
            allow_bone_threshold_fallback=cfg.mandible.allow_bone_threshold_fallback,
            bone_hu_min=cfg.mandible.bone_hu_min,
            require_mask_for_volume=cfg.mandible.require_mask_for_volume,
            bone_min_volume_ml=cfg.mandible.bone_min_volume_ml,
        ),
        mandible_mask=mandible_mask,
        landmarks=landmarks_bundle,
        oral_cavity_mask=oral_cavity_mask,
        oral_cavity_cfg=_OralCfg(enabled=cfg.oral_cavity.enabled),
        save_masks_callback=_save_mask,
    )
    mandible_volume_ml = mandible_features.get("mandible_volume_ml")
    if isinstance(mandible_volume_ml, float) and mandible_volume_ml != mandible_volume_ml:
        mandible_volume_ml = None
    oral_volume_ml = mandible_features.get("oral_cavity_volume_ml")
    if isinstance(oral_volume_ml, float) and oral_volume_ml != oral_volume_ml:
        oral_volume_ml = None

    # 6.3 Tongue
    tongue_features = compute_tongue_features(
        image=image,
        cfg=_TongueCfg(
            enabled=cfg.tongue.enabled,
            require_mask_for_volume=cfg.tongue.require_mask_for_volume,
            allow_posterior_roi_fallback=cfg.tongue.allow_posterior_roi_fallback,
            low_hu_threshold=cfg.tongue.low_hu_threshold,
            low_hu_threshold_mode=cfg.tongue.low_hu_threshold_mode,
            record_contrast_sensitivity=cfg.tongue.record_contrast_sensitivity,
        ),
        tongue_mask=tongue_mask,
        landmarks=landmarks_bundle,
        airway=airway_info,
        mandible_volume_ml=mandible_volume_ml,
        oral_cavity_volume_ml=oral_volume_ml,
        save_masks_callback=_save_mask,
    )
    tongue_volume_ml = tongue_features.get("tongue_volume_ml")
    if isinstance(tongue_volume_ml, float) and tongue_volume_ml != tongue_volume_ml:
        tongue_volume_ml = None

    # 6.4 Body silhouette for soft palate + regional fat (compute once)
    body_arr = body_mask(image, cfg.fat.body_air_threshold_hu) \
        if airway_info is not None and airway_info.is_present else None

    # 6.5 Soft palate / Uvula / Lateral wall / Tonsils
    soft_features = compute_soft_palate_features(
        image=image,
        cfg=_SoftTissueCfg(
            enabled=cfg.soft_tissue.enabled,
            require_masks_for_volumes=cfg.soft_tissue.require_masks_for_volumes,
            allow_landmark_length_fallback=cfg.soft_tissue.allow_landmark_length_fallback,
            lateral_wall_band_mm=cfg.soft_tissue.lateral_wall_band_mm,
            lateral_wall_axial_window_mm=cfg.soft_tissue.lateral_wall_axial_window_mm,
            body_air_threshold_hu=cfg.soft_tissue.body_air_threshold_hu,
        ),
        soft_palate_mask=soft_palate_mask,
        uvula_mask=uvula_mask,
        palatine_tonsil_left_mask=tonsil_l,
        palatine_tonsil_right_mask=tonsil_r,
        landmarks=landmarks_bundle,
        airway=airway_info,
        body_mask=body_arr,
        save_masks_callback=_save_mask,
    )

    # 6.6 Skeletal
    skeletal_features = compute_skeletal_features(
        image=image,
        cfg=_SkeletalCfg(
            enabled=cfg.skeletal.enabled,
            allow_landmark_only_distances=cfg.skeletal.allow_landmark_only_distances,
            allow_hyoid_threshold_fallback=cfg.skeletal.allow_hyoid_threshold_fallback,
        ),
        landmarks=landmarks_bundle,
        airway=airway_info,
        mandible_mask=mandible_mask if mandible_mask is not None else None,
        mandibular_plane_to_hyoid_distance_mm=(
            mandible_features.get("mandibular_plane_to_hyoid_distance_mm")
            if isinstance(mandible_features.get("mandibular_plane_to_hyoid_distance_mm"),
                          (int, float)) else None
        ),
    )

    # 6.7 Regional airway
    airway_region_features = compute_regional_airway_features(
        image=image,
        cfg=_AirwayRegionCfg(
            enabled=cfg.airway_regions.enabled,
            prefer_landmark_defined_regions=cfg.airway_regions.prefer_landmark_defined_regions,
            allow_axial_approximation=cfg.airway_regions.allow_axial_approximation,
            save_csa_profile=cfg.airway_regions.save_csa_profile,
        ),
        airway=airway_info,
        landmarks=landmarks_bundle,
        tongue_mask=tongue_mask,
        tongue_volume_ml=tongue_volume_ml,
        csa_profile_path=str(case_dir / "airway_csa_profile.json")
            if cfg.airway_regions.save_csa_profile else None,
    )

    # 6.8 Regional fat (level-anchored)
    fat_region_features = compute_regional_fat_features(
        image=image,
        cfg=_FatRegionCfg(
            enabled=cfg.fat_regions.enabled,
            fat_hu_min=cfg.hu.fat_hu_min,
            fat_hu_max=cfg.hu.fat_hu_max,
            parapharyngeal_lateral_band_mm=cfg.fat.parapharyngeal_lateral_band_mm,
            parapharyngeal_axial_window_mm=cfg.fat.parapharyngeal_axial_window_mm,
            retropharyngeal_posterior_band_mm=cfg.fat.retropharyngeal_posterior_band_mm,
            retropharyngeal_axial_window_mm=cfg.fat.retropharyngeal_axial_window_mm,
            body_air_threshold_hu=cfg.fat.body_air_threshold_hu,
            enable_facial_fat=cfg.fat_regions.enable_facial_fat,
        ),
        airway=airway_info,
        body_mask=body_arr,
        landmarks=landmarks_bundle,
        save_masks_callback=_save_mask,
    )

    # 7. Optional modules
    perivascular = compute_perivascular_features(
        image, cfg.perivascular, cfg.hu.fat_hu_min, cfg.hu.fat_hu_max
    )
    thoracic = compute_thoracic_features(
        image, cfg.thoracic, cfg.hu.fat_hu_min, cfg.hu.fat_hu_max
    )
    optional = {**perivascular, **thoracic}

    # 8. Radiomics
    radiomics = compute_radiomics(image, cfg.radiomics, masks_for_radiomics)

    # 8.5 Enrich QC with per-region and reliability flags
    qc_row = enrich_qc_row(
        qc_row,
        landmarks=landmarks_bundle,
        masks_present={
            "airway": airway_info is not None and airway_info.is_present,
            "tongue": tongue_mask is not None and tongue_mask.any(),
            "mandible": mandible_mask is not None and mandible_mask.any(),
            "soft_palate": soft_palate_mask is not None and soft_palate_mask.any(),
            "fat": body_arr is not None,
        },
        feature_rows={
            "airway": airway_features, "tongue": tongue_features,
            "mandible": mandible_features, "soft_tissue": soft_features,
            "skeletal": skeletal_features, "fat": fat_features,
        },
    )

    # 9. Composite exploratory scores (UNVALIDATED — names end with _untrained).
    # New composites consume the *combined* registry-shaped row. We build the
    # combined row first so component features from every module are visible.
    composite_input: dict[str, object] = {
        **airway_features, **airway_region_features,
        **tongue_features, **mandible_features, **soft_features,
        **skeletal_features, **fat_features, **fat_region_features,
    }
    composite_cfg = CompositeConfig(
        enabled=cfg.composites.enabled,
        require_batch_standardization=cfg.composites.require_batch_standardization,
        suffix=cfg.composites.suffix,
    )
    cohort_stats = _load_cohort_stats(cfg.composites.cohort_stats_path)
    composite = compute_composites(composite_input, composite_cfg, cohort_stats)
    # Back-compat: keep the legacy two-score helper alive for any caller that
    # depended on it. The new composites supersede it.
    legacy_composite = _composite_scores(airway_features, fat_features)
    for k, v in legacy_composite.items():
        composite.setdefault(k, v)

    # 10. Slicer QC scene (only when masks were materialised — the loader
    # references them by absolute path, so it's useless without files on disk)
    slicer_script_path: Optional[Path] = None
    if cfg.output.save_masks:
        try:
            slicer_script_path = write_slicer_loader(
                case_id=patient_id or image.study_id,
                image_path=Path(input_path),
                case_dir=case_dir,
                out_script=case_dir / f"{patient_id or image.study_id}_load_qc_in_slicer.py",
            )
        except Exception as exc:
            log.warning("Could not write Slicer loader: %s", exc)

    # 11. Identifiers
    identifiers = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": __version__,
        "config_hash": cfg.hash(),
        "processing_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "patient_id": patient_id or image.study_id,
        "study_id": image.study_id,
        "scan_id": scan_id_override or image.scan_id,
        "input_path_hash": safe_hash(str(Path(input_path).resolve())),
        "input_kind": (image.sidecar or {}).get("input_kind", "unknown"),
        "airway_source": payload.source,
        "airway_provider_notes": payload.notes,
        "slicer_loader_script": str(slicer_script_path) if slicer_script_path else "",
    }

    warnings: list[str] = list(qc_result.extra.get("warnings", []))

    return CaseResult(
        identifiers=identifiers,
        qc=qc_row,
        airway=airway_features,
        airway_regions=airway_region_features,
        tongue=tongue_features,
        mandible=mandible_features,
        soft_tissue=soft_features,
        skeletal=skeletal_features,
        fat=fat_features,
        fat_regions=fat_region_features,
        optional=optional,
        radiomics=radiomics,
        composite=composite,
        warnings=warnings,
        errors=[],
    )


# --- helpers ----------------------------------------------------------------

def _load_cohort_stats(path: Optional[str]) -> Optional[CohortStats]:
    """Read a 3-column CSV (feature_name, mean, std) into a CohortStats.

    Returns None if path is missing/empty so the composites module falls
    back to its "require_batch_standardization gates emission" rule.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        import csv
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        with p.open() as fh:
            for row in csv.DictReader(fh):
                name = row.get("feature_name")
                if not name:
                    continue
                try:
                    means[name] = float(row.get("mean", "nan"))
                    stds[name] = float(row.get("std", "nan"))
                except ValueError:
                    continue
        return CohortStats(means=means, stds=stds)
    except Exception as exc:
        log.warning("Could not parse cohort stats CSV %s: %s", p, exc)
        return None


def _composite_scores(airway: dict, fat: dict) -> dict:
    """Simple, **unvalidated** exploratory composites.

    Marked `_untrained` because they are NOT cohort-standardized here — they
    are only the raw sum of feature signals likely to track with OSA from
    the literature. Standardization (z-score against the analysis cohort)
    is left for downstream code that has access to age/sex/BMI distributions.
    """
    out: dict = {
        "cta_osa_anatomy_score_untrained": float("nan"),
        "cta_osa_fat_score_untrained": float("nan"),
        "cta_osa_combined_score_untrained": float("nan"),
        "composite_score_method": "raw_linear_unstandardized_v1",
        "composite_score_disclaimer":
            "EXPLORATORY — not standardized against any cohort. Do NOT use clinically.",
    }
    min_csa = airway.get("airway_min_csa_mm2")
    rg = airway.get("retroglossal_csa_mm2")
    if isinstance(min_csa, float) and min_csa == min_csa and min_csa > 0:
        # Smaller airway → larger score (inverse).
        out["cta_osa_anatomy_score_untrained"] = round(100.0 / float(min_csa), 4)

    pp_total = fat.get("fat_parapharyngeal_total_volume_ml")
    rp = fat.get("fat_retropharyngeal_volume_ml")
    parts = [v for v in (pp_total, rp) if isinstance(v, float) and v == v]
    if parts:
        out["cta_osa_fat_score_untrained"] = round(float(sum(parts)), 3)

    if (isinstance(out["cta_osa_anatomy_score_untrained"], float)
            and isinstance(out["cta_osa_fat_score_untrained"], float)
            and out["cta_osa_anatomy_score_untrained"] == out["cta_osa_anatomy_score_untrained"]
            and out["cta_osa_fat_score_untrained"] == out["cta_osa_fat_score_untrained"]):
        out["cta_osa_combined_score_untrained"] = round(
            out["cta_osa_anatomy_score_untrained"]
            + out["cta_osa_fat_score_untrained"], 3
        )
    return out


def _failed_load_result(
    input_path: Path, patient_id: Optional[str], reason: str,
    cfg: PipelineConfig, out_dir: Path,
) -> CaseResult:
    identifiers = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": __version__,
        "config_hash": cfg.hash(),
        "processing_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "patient_id": patient_id or "unknown",
        "study_id": "unknown",
        "scan_id": "unknown",
        "input_path_hash": safe_hash(str(Path(input_path).resolve())),
        "input_kind": "unknown",
        "airway_source": "none",
        "airway_provider_notes": "",
    }
    qc_row = {
        "qc_pass": False,
        "qc_warning_count": 0,
        "qc_failure_reasons": f"load_failed: {reason[:200]}",
        "qc_coverage_score": 0.0,
        "qc_dental_artifact_score": float("nan"),
        "qc_has_upper_airway": False,
        "qc_has_cervical_soft_tissue": False,
        "qc_has_hyoid_region": False,
        "qc_has_epiglottis_region": False,
        "qc_truncation_flag": False,
        "qc_spacing_x_mm": float("nan"),
        "qc_spacing_y_mm": float("nan"),
        "qc_spacing_z_mm": float("nan"),
        "qc_contrast_enhanced": False,
        "qc_z_extent_mm": float("nan"),
    }
    return CaseResult(identifiers=identifiers, qc=qc_row, airway={}, fat={},
                      optional={}, radiomics={}, composite={},
                      warnings=[], errors=[reason])
