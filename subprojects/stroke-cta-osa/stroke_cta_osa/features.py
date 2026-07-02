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
from .io import load_input, save_mask, to_sitk_image
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

    # Optional anatomy masks from TotalSegmentator/VISTA/manual/dental outputs.
    # These are consumed by the anatomy modules and by fat ROI priors.
    tongue_mask = _load_external_mask_array(
        external_tongue_mask_path or cfg.tongue.external_mask_path, image)
    mandible_mask_method = "external_mask"
    mandible_mask = _load_external_mask_array(
        external_mandible_mask_path or cfg.mandible.external_mask_path, image)
    if (mandible_mask is None or not mandible_mask.any()) \
            and cfg.mandible.dental_mandible_mask_path:
        mandible_mask = _load_external_mask_array(
            cfg.mandible.dental_mandible_mask_path, image)
        mandible_mask_method = "dental_mandible_mask"
    oral_cavity_mask = _load_external_mask_array(
        external_oral_cavity_mask_path or cfg.oral_cavity.external_mask_path, image)
    soft_palate_mask = _load_external_mask_array(
        external_soft_palate_mask_path or cfg.soft_tissue.soft_palate_mask_path,
        image)
    uvula_mask = _load_external_mask_array(cfg.soft_tissue.uvula_mask_path, image)
    tonsil_l = _load_external_mask_array(
        cfg.soft_tissue.palatine_tonsil_left_mask_path, image)
    tonsil_r = _load_external_mask_array(
        cfg.soft_tissue.palatine_tonsil_right_mask_path, image)
    prevertebral_mask = _load_external_mask_union(
        cfg.fat.prevertebral_mask_paths, image)
    anatomy_masks = {
        "tongue": tongue_mask,
        "mandible": mandible_mask,
        "oral_cavity": oral_cavity_mask,
        "soft_palate": soft_palate_mask,
        "uvula": uvula_mask,
        "palatine_tonsil_left": tonsil_l,
        "palatine_tonsil_right": tonsil_r,
        "prevertebral": prevertebral_mask,
    }

    # 6. Fat features
    # Only retain full-volume masks for radiomics when radiomics is actually
    # enabled — otherwise these bool volumes stay resident for nothing and
    # inflate the per-case memory peak (which gates batch worker count).
    masks_for_radiomics: dict[str, np.ndarray] = {}
    _keep_radiomics_masks = cfg.radiomics.enabled
    def _save_mask(name: str, mask: np.ndarray) -> None:
        if cfg.output.save_masks:
            try:
                save_mask(mask, image, case_dir / f"mask_{name}.nii.gz")
            except Exception as exc:
                log.warning("Could not save mask %s: %s", name, exc)
        if not _keep_radiomics_masks:
            return
        if name in ("fat_cervical_total", "fat_parapharyngeal_total"):
            masks_for_radiomics[name.replace("fat_", "").replace("_total", "")] = mask
        if name == "fat_parapharyngeal_total":
            masks_for_radiomics["parapharyngeal_fat"] = mask
        if name == "fat_cervical_total":
            masks_for_radiomics["cervical_fat"] = mask

    if prevertebral_mask is not None and prevertebral_mask.any():
        _save_mask("prevertebral", prevertebral_mask)

    # Compute the body silhouette ONCE and share it across the fat, regional-fat
    # and soft-palate modules. It's a full-volume connected-component pass — the
    # single most expensive op after airway — so reusing it both speeds the case
    # and removes a large duplicate memory spike (previously body_mask ran twice).
    body_silhouette = body_mask(image, cfg.fat.body_air_threshold_hu)

    fat_features = compute_fat_features(
        image=image, airway=airway_info, landmarks=landmarks,
        hu_cfg=cfg.hu, fat_cfg=cfg.fat,
        airway_min_csa_z_index=anchor_z,
        save_masks_callback=_save_mask,
        anatomy_masks=anatomy_masks,
        precomputed_body_mask=body_silhouette,
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
        mandible_mask_method=mandible_mask_method,
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

    # 6.4 Body silhouette for soft palate + regional fat. Reuse the one already
    # computed above; preserve the prior contract of None when no airway is
    # present (downstream soft-palate / regional-fat / QC depend on that).
    body_arr = body_silhouette \
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
            subcutaneous_erosion_mm=cfg.fat.subcutaneous_erosion_mm,
            enable_facial_fat=cfg.fat_regions.enable_facial_fat,
            use_anatomy_priors=cfg.fat.use_anatomy_priors,
            anatomy_prior_dilation_mm=cfg.fat.anatomy_prior_dilation_mm,
            parapharyngeal_sector_min_lateral_fraction=(
                cfg.fat.parapharyngeal_sector_min_lateral_fraction
            ),
        ),
        airway=airway_info,
        body_mask=body_arr,
        landmarks=landmarks_bundle,
        save_masks_callback=_save_mask,
        anatomy_masks=anatomy_masks,
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

def _load_external_mask_array(
    path: Optional[Path | str],
    image: CTAImage,
) -> Optional[np.ndarray]:
    """Load an optional binary mask and resample it to the CTA geometry."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        log.warning("External mask not found: %s", p)
        return None
    try:
        import SimpleITK as sitk
        mask_img = sitk.ReadImage(str(p))
        if not _sitk_geometry_matches(mask_img, image):
            mask_img = sitk.Resample(
                mask_img,
                to_sitk_image(image),
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                mask_img.GetPixelID(),
            )
        mask = sitk.GetArrayFromImage(mask_img).astype(bool)
        if mask.shape != image.shape_zyx:
            log.warning("External mask resampled shape %s != image %s - skipping",
                        mask.shape, image.shape_zyx)
            return None
        return mask
    except Exception as exc:
        log.warning("Could not load external mask %s: %s", p, exc)
        return None


def _load_external_mask_union(
    paths: list[str],
    image: CTAImage,
) -> Optional[np.ndarray]:
    """Load and union multiple optional masks in CTA geometry."""
    masks = [
        m for p in paths
        if (m := _load_external_mask_array(p, image)) is not None and m.any()
    ]
    if not masks:
        return None
    out = np.zeros(image.shape_zyx, dtype=bool)
    for mask in masks:
        out |= mask
    return out


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


def _sitk_geometry_matches(mask_img: object, image: CTAImage) -> bool:
    """True when an optional mask is already in the consuming CTA geometry."""
    try:
        size = tuple(int(v) for v in mask_img.GetSize())
        spacing = tuple(float(v) for v in mask_img.GetSpacing())
        origin = tuple(float(v) for v in mask_img.GetOrigin())
        direction = tuple(float(v) for v in mask_img.GetDirection())
    except AttributeError:
        return False
    return (
        size == tuple(reversed(image.shape_zyx))
        and np.allclose(spacing, image.spacing_xyz_mm, atol=1e-4)
        and np.allclose(origin, image.origin_xyz_mm, atol=1e-3)
        and np.allclose(direction, image.direction_3x3, atol=1e-5)
    )


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
