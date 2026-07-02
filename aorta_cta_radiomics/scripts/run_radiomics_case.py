#!/usr/bin/env python
"""Run radiomics only from masks already written by a previous case stage."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aorta_cta_radiomics import __version__
from aorta_cta_radiomics.config import load_config, resolve_project_path
from aorta_cta_radiomics.features import feature_row, write_csv
from aorta_cta_radiomics.radiomics import extract_radiomics_features
from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide


logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path, help="Per-case output directory.")
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument(
        "--masks-dir",
        default=None,
        type=Path,
        help="Optional masks directory. Defaults to <outdir>/masks/<case_id>.",
    )
    parser.add_argument(
        "--region",
        action="append",
        dest="regions",
        default=None,
        help="Restrict extraction to one radiomics region. Repeat for multiple regions.",
    )
    parser.add_argument(
        "--output-name",
        default="radiomics_features.csv",
        help="Feature CSV filename written under <outdir>/features. Must be a filename, not a path.",
    )
    parser.add_argument(
        "--no-rebuild-wide",
        action="store_true",
        help="Do not rebuild modeling_wide_features.csv. Used by split-region runners.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    radiomics_config = dict(config.get("radiomics", {}))
    outdir = args.outdir
    masks_dir = args.masks_dir or outdir / "masks" / args.case_id
    features_dir = outdir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    output_name = _validate_output_name(args.output_name)

    if not bool(radiomics_config.get("enabled", False)):
        frame = pd.DataFrame(
            [
                feature_row(
                    case_id=args.case_id,
                    region="radiomics",
                    feature_group="radiomics_status",
                    feature_name="disabled",
                    feature_value=True,
                    software_version=str(config.get("outputs", {}).get("software_version", __version__)),
                )
            ]
        )
        write_csv(frame, features_dir / output_name)
        if not args.no_rebuild_wide:
            rebuild_modeling_wide(features_dir)
        return

    project_root = Path(__file__).resolve().parents[1]
    settings_path = resolve_project_path(str(radiomics_config.get("settings_path", "")), project_root)
    if not str(radiomics_config.get("settings_path", "")).strip() or not settings_path.exists():
        logger.warning("Radiomics settings file not found; using backend defaults: %s", settings_path)
        settings_path_or_none = None
    else:
        settings_path_or_none = settings_path
        shutil.copy2(settings_path, masks_dir / f"{args.case_id}_{radiomics_config.get('backend', 'pyradiomics')}_settings.yaml")

    frames: list[pd.DataFrame] = []
    regions = [str(region) for region in args.regions] if args.regions else list(radiomics_config.get("regions", []))
    for region in regions:
        mask_path = _region_mask_path(masks_dir, args.case_id, str(region))
        if not mask_path.exists():
            logger.warning("Skipping radiomics region with missing mask: %s (%s)", region, mask_path)
            frames.append(
                pd.DataFrame(
                    [
                        feature_row(
                            case_id=args.case_id,
                            region=str(region),
                            feature_group="radiomics_status",
                            feature_name="missing_mask",
                            feature_value=str(mask_path),
                            mask_name=mask_path.name,
                            software_version=str(config.get("outputs", {}).get("software_version", __version__)),
                        )
                    ]
                )
            )
            continue
        try:
            frames.append(
                extract_radiomics_features(
                    image_path=args.image,
                    mask_path=mask_path,
                    case_id=args.case_id,
                    region=str(region),
                    settings_path=settings_path_or_none,
                    include_diagnostics=bool(radiomics_config.get("include_diagnostics", False)),
                    software_version=str(config.get("outputs", {}).get("software_version", __version__)),
                    backend=str(radiomics_config.get("backend", "pyradiomics")),
                    device=str(radiomics_config.get("device", "cpu")),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Radiomics extraction failed for %s/%s: %s", args.case_id, region, exc)
            frames.append(
                pd.DataFrame(
                    [
                        feature_row(
                            case_id=args.case_id,
                            region=str(region),
                            feature_group="radiomics_status",
                            feature_name="extraction_error",
                            feature_value=str(exc),
                            mask_name=mask_path.name,
                            software_version=str(config.get("outputs", {}).get("software_version", __version__)),
                        )
                    ]
                )
            )

    radiomics_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(radiomics_frame, features_dir / output_name)
    if not args.no_rebuild_wide:
        rebuild_modeling_wide(features_dir)
    print(f"Wrote radiomics rows: {len(radiomics_frame)}")
    print(f"Saved radiomics outputs to {outdir.resolve()}")


def _validate_output_name(output_name: str) -> str:
    path = Path(output_name)
    if path.is_absolute() or path.name != output_name:
        raise ValueError("--output-name must be a filename under the case features directory, not a path.")
    return output_name


def _region_mask_path(masks_dir: Path, case_id: str, region: str) -> Path:
    if region == "aorta_mask":
        return masks_dir / f"{case_id}_aorta_mask_cleaned.nii.gz"
    return masks_dir / f"{case_id}_{region}.nii.gz"


if __name__ == "__main__":
    main()
