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
from .config import PipelineConfig
from .dicom_utils import safe_hash
from .fat import compute_fat_features
from .io import load_input, save_mask
from .logging_utils import get_logger
from .perivascular import compute_perivascular_features
from .qc import qc_to_row, run_qc
from .qc_slicer import write_slicer_loader
from .radiomics import compute_radiomics
from .thoracic import compute_thoracic_features
from .types import CaseResult, CTAImage

log = get_logger("features")


def extract_case(
    input_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    sidecar_path: Optional[Path] = None,
    patient_id: Optional[str] = None,
    scan_id_override: Optional[str] = None,
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

    # 2. Airway adapter chain
    providers = build_airway_provider_chain(cfg)
    airway_info, payload = first_available(providers, image)

    # 3. Landmarks
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

    # 9. Composite exploratory scores (UNVALIDATED — names end with _untrained)
    composite = _composite_scores(airway_features, fat_features)

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
        fat=fat_features,
        optional=optional,
        radiomics=radiomics,
        composite=composite,
        warnings=warnings,
        errors=[],
    )


# --- helpers ----------------------------------------------------------------

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
