"""Headless commands for the FLOWCAT-backed carotid milestone.

The command never runs a substitute centreline or anatomical labeller. A case
with incomplete native FLOWCAT products receives a manifest and failure record,
then exits with status 2.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import yaml

from .carotid_analysis import (
    CarotidAnalysisConfig,
    CoarseAnatomyMasksIJK,
    analyze_carotid_case,
    perivascular_zyx_to_nibabel_ijk,
)
from .flowcat_adapter import (
    DEFAULT_REQUIRED_OUTPUTS,
    ProcessingProvenance,
    assert_geometry_compatible,
    build_output_manifest,
    collect_processing_provenance,
    geometry_from_nifti,
    validate_flowcat_case,
    write_output_manifest,
)
from .output_schema import sha256_file, write_output_bundle
from .perivascular import AdaptiveShellLaw, RadialBand, TissueThresholds


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _failure(outdir: Path, *, stage: str, exc: Exception) -> int:
    _write_json(
        outdir / "failure.json",
        {
            "schema_version": 1,
            "status": "failed",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "recorded_at": _utc_now(),
        },
    )
    return 2


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Carotid configuration must be a YAML mapping.")
    return payload


def _analysis_config(
    payload: dict[str, Any],
    spacing_xyz_mm: tuple[float, float, float],
) -> CarotidAnalysisConfig:
    flowcat = payload.get("flowcat", {})
    coordinate = payload.get("coordinate_system", {})
    propagation = payload.get("label_propagation", {})
    shells = payload.get("radial_shells", {})
    adaptive = payload.get("adaptive_envelope", {}).get("fat", {})
    tissue = payload.get("tissue_composition", {})
    sectors = payload.get("sectors", {})
    contacts = payload.get("surface_contact", {})
    bands = tuple(
        RadialBand(
            float(item["inner_mm"]),
            float(item["outer_mm"]),
            str(item.get("name")) if item.get("name") is not None else None,
        )
        for item in shells.get("bands_mm", [])
    )
    if not bands:
        raise ValueError("radial_shells.bands_mm must define at least one physical band.")
    adaptive_minimum = adaptive.get("minimum_mm", 1.0)
    if adaptive_minimum == "resolution_floor":
        adaptive_minimum = 2.0 * max(spacing_xyz_mm)
    fat_windows = tissue.get("fat_windows_hu", {})
    thresholds = TissueThresholds(
        primary_fat_hu=tuple(fat_windows.get("primary", (-190.0, -30.0))),
        narrow_fat_hu=tuple(fat_windows.get("narrow_sensitivity", (-150.0, -50.0))),
        wide_fat_hu=tuple(fat_windows.get("wide_sensitivity", (-200.0, -20.0))),
        high_density_min_hu=float(tissue.get("high_density_hu", 200.0)),
    )
    return CarotidAnalysisConfig(
        required_flowcat_codes=tuple(flowcat.get("target_labels", ("RCCA", "LCCA", "RICA", "LICA"))),
        coordinate_interval_mm=float(coordinate.get("sample_spacing_mm", 1.0)),
        fixed_radial_bands=bands,
        adaptive_law=AdaptiveShellLaw(
            alpha=float(adaptive.get("alpha", 1.0)),
            minimum_mm=float(adaptive_minimum),
            maximum_mm=float(adaptive.get("maximum_mm", 5.0)),
        ),
        partial_volume_boundary_mm=float(shells.get("partial_volume_exclusion_mm", 0.0)),
        bifurcation_radius_mm=float(coordinate.get("bifurcation_exclusion_mm", 3.0)),
        bifurcation_shell_exclusion_mm=float(
            coordinate.get("bifurcation_exclusion_mm", 3.0)
        ),
        propagation_minimum_confidence=float(propagation.get("minimum_confidence", 0.05)),
        sector_count=int(sectors.get("primary_count", 16)),
        contact_distances_mm=tuple(float(value) for value in contacts.get("distances_mm", (0.5, 1, 2))),
        primary_contact_distance_mm=float(contacts.get("primary_distance_mm", 1.0)),
        longitudinal_bin_mm=float(coordinate.get("sample_spacing_mm", 1.0)),
        tissue_thresholds=thresholds,
        acquisition_metadata={},
    )


def _load_array(path: str | Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(Path(path).expanduser().resolve()))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a three-dimensional NIfTI, got shape {image.shape}.")
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"NIfTI contains non-finite values: {path}")
    return image, data


def _load_mask(
    path: str | Path | None,
    *,
    reference_geometry: Any,
) -> np.ndarray | None:
    if path is None:
        return None
    image, data = _load_array(path)
    assert_geometry_compatible(geometry_from_nifti(image), reference_geometry)
    values = np.unique(data)
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"Coarse anatomy mask must be binary 0/1: {path}")
    return np.asarray(data != 0, dtype=bool)


def _save_nifti(
    path: Path,
    array_ijk: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    dtype: np.dtype[Any] | type,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(array_ijk, dtype=dtype), reference.affine, header=header)
    temporary = path.with_name(f".{path.stem}.tmp.nii.gz")
    nib.save(image, str(temporary))
    temporary.replace(path)
    return path


def _write_analysis_volumes(result: Any, reference: Any, outdir: Path) -> list[dict[str, Any]]:
    masks = outdir / "masks"
    case_id = result.case_id
    volume = result.anatomical_artery_volume_ijk
    outputs: list[tuple[str, Path]] = []
    outputs.append(
        (
            "anatomical_artery_labels",
            _save_nifti(
                masks / f"{case_id}_anatomical_artery_labels.nii.gz",
                volume.labels,
                reference,
                dtype=np.uint8,
            ),
        )
    )
    for name, data, dtype in (
        ("artery_assignment_confidence", volume.confidence, np.float32),
        ("artery_assignment_uncertainty", volume.uncertainty_mask, np.uint8),
        ("artery_bifurcation_exclusion", volume.bifurcation_exclusion_mask, np.uint8),
        ("artery_labels_nearest", volume.nearest_labels, np.uint8),
        ("artery_labels_branch_aware", volume.connected_labels, np.uint8),
        ("artery_label_strategy_disagreement", volume.disagreement_mask, np.uint8),
    ):
        outputs.append(
            (
                name,
                _save_nifti(
                    masks / f"{case_id}_{name}.nii.gz",
                    data,
                    reference,
                    dtype=dtype,
                ),
            )
        )
    for code, branch in result.branches.items():
        target = perivascular_zyx_to_nibabel_ijk(branch.target_lumen_zyx)
        outputs.append(
            (
                f"{code}.target_lumen",
                _save_nifti(
                    masks / f"{case_id}_{code}_target_lumen.nii.gz",
                    target,
                    reference,
                    dtype=np.uint8,
                ),
            )
        )
        for band_name, band in branch.fixed_shells.bands.items():
            for state, data in (("raw", band.raw_mask), ("filtered", band.filtered_mask)):
                output_name = f"{code}.{band_name}.{state}"
                outputs.append(
                    (
                        output_name,
                        _save_nifti(
                            masks / f"{case_id}_{code}_{band_name}_{state}.nii.gz",
                            perivascular_zyx_to_nibabel_ijk(data),
                            reference,
                            dtype=np.uint8,
                        ),
                    )
                )
        adaptive_band = next(iter(branch.adaptive_shell.bands.values()))
        for state, data in (
            ("raw", adaptive_band.raw_mask),
            ("filtered", adaptive_band.filtered_mask),
        ):
            outputs.append(
                (
                    f"{code}.adaptive.{state}",
                    _save_nifti(
                        masks / f"{case_id}_{code}_adaptive_{state}.nii.gz",
                        perivascular_zyx_to_nibabel_ijk(data),
                        reference,
                        dtype=np.uint8,
                    ),
                )
            )
        for name, data, dtype in (
            ("tissue_classes", branch.composition.labels, np.uint8),
            ("angular_sectors", branch.sector_assignment.labels, np.uint8),
        ):
            outputs.append(
                (
                    f"{code}.{name}",
                    _save_nifti(
                        masks / f"{case_id}_{code}_{name}.nii.gz",
                        perivascular_zyx_to_nibabel_ijk(data),
                        reference,
                        dtype=dtype,
                    ),
                )
            )
    return [
        {"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in outputs
    ]


def _model_payloads(repository: str | Path) -> dict[str, Path]:
    root = Path(repository).expanduser().resolve()
    candidates = sorted((root / "arterial").glob("**/models/**/*.pth"))
    return {
        str(path.relative_to(root)): path
        for path in candidates
        if path.is_file()
    }


def _provenance(repository: str | Path, *, parameters: dict[str, Any], status: str) -> ProcessingProvenance:
    return collect_processing_provenance(
        repository,
        model_files=_model_payloads(repository),
        command_line_parameters=parameters,
        processing_status=status,
    )


def command_manifest(args: argparse.Namespace) -> int:
    provenance = _provenance(
        args.flowcat_repo,
        parameters={"case_id": args.case_id, "mode": args.mode, "operation": "manifest"},
        status="output_inventory",
    )
    manifest = build_output_manifest(
        args.case_dir,
        case_id=args.case_id,
        mode=args.mode,
        provenance=provenance,
    )
    write_output_manifest(manifest, args.output)
    return 0 if not manifest.missing_required_outputs else 2


def command_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stage = "preflight"
    try:
        if not args.trusted_flowcat_artifacts:
            raise ValueError(
                "Refusing pickle/object-NPY loading without --trusted-flowcat-artifacts."
            )
        configuration = _load_yaml(args.config)
        provenance = _provenance(
            args.flowcat_repo,
            parameters={
                "case_id": args.case_id,
                "mode": args.mode,
                "config": str(Path(args.config).expanduser().resolve()),
                "trusted_flowcat_artifacts": True,
            },
            status="started",
        )
        inventory = build_output_manifest(
            args.case_dir,
            case_id=args.case_id,
            mode=args.mode,
            provenance=provenance,
        )
        write_output_manifest(inventory, outdir / "flowcat_output_manifest.json")
        if inventory.missing_required_outputs:
            raise FileNotFoundError(
                "Missing required FLOWCAT outputs: "
                + ", ".join(inventory.missing_required_outputs)
            )
        stage = "flowcat_validation"
        validated = validate_flowcat_case(
            args.case_dir,
            case_id=args.case_id,
            cta_path=args.cta,
            provenance=provenance,
            mode=args.mode,
            trusted_source=True,
        )
        write_output_manifest(validated.manifest, outdir / "flowcat_output_manifest.json")
        stage = "input_loading"
        cta_image, cta = _load_array(args.cta)
        geometry = validated.cta_geometry
        assert geometry is not None
        analysis_config = _analysis_config(configuration, geometry.spacing_mm)
        anatomy = CoarseAnatomyMasksIJK(
            vein=_load_mask(args.vein_mask, reference_geometry=geometry),
            bone=_load_mask(args.bone_mask, reference_geometry=geometry),
            airway=_load_mask(args.airway_mask, reference_geometry=geometry),
            thyroid_gland=_load_mask(args.thyroid_mask, reference_geometry=geometry),
            skeletal_muscle=_load_mask(args.muscle_mask, reference_geometry=geometry),
            uncertain=_load_mask(args.uncertain_mask, reference_geometry=geometry),
            high_density=_load_mask(args.high_density_mask, reference_geometry=geometry),
        )
        stage = "carotid_analysis"
        result = analyze_carotid_case(
            validated,
            cta,
            config=analysis_config,
            coarse_anatomy_ijk=anatomy,
        )
        result.case_output.input_sha256 = sha256_file(args.cta)
        stage = "output_writing"
        tables = write_output_bundle(
            outdir / "tables",
            result.case_output,
            result.vessel_outputs,
            result.sample_outputs,
        )
        volumes = _write_analysis_volumes(result, cta_image, outdir)
        _write_json(outdir / "artery_label_summary.json", result.anatomical_artery_volume_ijk.summary)
        _write_json(
            outdir / "analysis_manifest.json",
            {
                "schema_version": 1,
                "case_id": result.case_id,
                "status": result.case_output.processing_status,
                "configuration": configuration,
                "configuration_sha256": result.case_output.configuration_sha256,
                "array_order_contract": result.array_order_contract,
                "tables": {key: str(value) for key, value in asdict(tables).items()},
                "volumes": volumes,
                "manual_review_required": any(
                    vessel.manual_review_required for vessel in result.vessel_outputs
                ),
                "completed_at": _utc_now(),
            },
        )
        return 0
    except Exception as exc:
        return _failure(outdir, stage=stage, exc=exc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest", help="Inventory and hash FLOWCAT outputs")
    manifest.add_argument("--case-dir", required=True)
    manifest.add_argument("--case-id", required=True)
    manifest.add_argument("--flowcat-repo", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--mode", default="extracranial_vessels")
    manifest.set_defaults(handler=command_manifest)

    run = subparsers.add_parser("run", help="Run the fail-closed carotid milestone")
    run.add_argument("--case-dir", required=True)
    run.add_argument("--case-id", required=True)
    run.add_argument("--cta", required=True)
    run.add_argument("--flowcat-repo", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--mode", default="extracranial_vessels")
    run.add_argument("--trusted-flowcat-artifacts", action="store_true")
    run.add_argument("--vein-mask")
    run.add_argument("--bone-mask")
    run.add_argument("--airway-mask")
    run.add_argument("--thyroid-mask")
    run.add_argument("--muscle-mask")
    run.add_argument("--uncertain-mask")
    run.add_argument("--high-density-mask")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
