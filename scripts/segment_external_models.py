#!/usr/bin/env python3
"""
Run segmentation using external pretrained models (TotalSegmentator or MONAI bundle).

No HU thresholding or custom segmentation fallback is performed. This script only
invokes TotalSegmentator and/or MONAI models.

Examples:
  # TotalSegmentator (total + head/neck + heartchambers)
  python -u scripts/segment_external_models.py \
    --input data/sub-547_0000.nii.gz \
    --output outputs/seg_547 \
    --totalseg-task total \
    --totalseg-task headneck_bones_vessels \
    --totalseg-task heartchambers_highres \
    --totalseg-export atrial_appendage_left=left_atrial_appendage.nii.gz \
    --totalseg-export heart_atrium_left=left_atrium.nii.gz

  # MONAI bundle (override input/output keys in config)
  python -u scripts/segment_external_models.py \
    --input data/sub-547_0000.nii.gz \
    --output outputs/seg_547 \
    --monai-bundle /path/to/monai_bundle \
    --monai-config configs/inference.json \
    --monai-meta configs/metadata.json \
    --monai-input-key input \
    --monai-output-key output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib


def _parse_key_value(pair: str) -> tuple[str, Any]:
    if "=" not in pair:
        raise ValueError(f"Expected key=value, got: {pair}")
    key, raw = pair.split("=", 1)
    val = raw.strip()
    if val.lower() in {"true", "false"}:
        return key.strip(), val.lower() == "true"
    if val.lower() in {"none", "null"}:
        return key.strip(), None
    try:
        if "." in val:
            return key.strip(), float(val)
        return key.strip(), int(val)
    except ValueError:
        return key.strip(), val


def _resolve_path(root: Path | None, path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute() and root is not None:
        path = root / path
    return str(path)


def _totalseg_label_path(totalseg_dir: Path, label: str) -> Path | None:
    if (totalseg_dir / f"{label}.nii.gz").exists():
        return totalseg_dir / f"{label}.nii.gz"
    if (totalseg_dir / label).exists():
        return totalseg_dir / label
    return None


def _export_totalseg_label(
    totalseg_dir: Path,
    label: str,
    output_path: Path,
    reference_shape: tuple[int, int, int] | None = None,
) -> bool:
    path = _totalseg_label_path(totalseg_dir, label)
    if path is None:
        return False
    img = nib.load(str(path))
    data = img.get_fdata() > 0
    if reference_shape is not None and data.shape != reference_shape:
        print(f"  ⚠ Shape mismatch for {label}: {data.shape} vs {reference_shape}")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype("uint8"), img.affine, img.header), str(output_path))
    print(f"✓ Saved {label} -> {output_path}")
    return True


def _check_env(require_monai: bool, require_totalseg: bool) -> None:
    details = {}
    try:
        import numpy as np  # noqa: F401

        details["numpy"] = np.__version__
    except Exception as exc:  # noqa: BLE001
        details["numpy_error"] = str(exc)
    try:
        import scipy  # noqa: F401

        details["scipy"] = scipy.__version__
    except Exception as exc:  # noqa: BLE001
        details["scipy_error"] = str(exc)
    if require_monai:
        try:
            import monai  # noqa: F401

            details["monai"] = monai.__version__
        except Exception as exc:  # noqa: BLE001
            details["monai_error"] = str(exc)
    if require_totalseg:
        try:
            import totalsegmentator  # noqa: F401

            details["totalsegmentator"] = "ok"
        except Exception as exc:  # noqa: BLE001
            details["totalsegmentator_error"] = str(exc)

    if any("error" in k for k in details):
        print("Environment check failed:")
        print(json.dumps(details, indent=2))
        raise SystemExit(2)
    print("Environment OK:")
    print(json.dumps(details, indent=2))


def run_totalseg(
    input_path: Path,
    output_dir: Path,
    task: str,
    roi_subset: list[str] | None,
    fast: bool,
    ml: bool,
) -> Path:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TotalSegmentator not available; install it in the active env.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    if fast and task in {"headneck_bones_vessels", "heartchambers_highres"}:
        print(f"  ⚠ Task {task} does not support --fast; switching to full resolution.")
        fast = False
    print(f"Running TotalSegmentator task={task} (fast={fast}, ml={ml}) -> {output_dir}")
    totalsegmentator(
        input=str(input_path),
        output=str(output_dir),
        task=task,
        fast=fast,
        ml=ml,
        roi_subset=roi_subset,
    )
    return output_dir


def run_monai_bundle(
    bundle_root: Path,
    config_files: list[str],
    meta_file: str | None,
    run_id: str | None,
    init_id: str | None,
    final_id: str | None,
    args_file: str | None,
    overrides: dict[str, Any],
) -> None:
    try:
        from monai.bundle import run as bundle_run
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MONAI bundle API not available; install monai in the active env.") from exc

    print("Running MONAI bundle...")
    bundle_run(
        run_id=run_id,
        init_id=init_id,
        final_id=final_id,
        meta_file=meta_file,
        config_file=config_files,
        args_file=args_file,
        **overrides,
    )
    print("✓ MONAI bundle run complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment using external pretrained MONAI/TotalSegmentator models")
    p.add_argument("--input", required=True, help="Input CTA NIfTI (.nii/.nii.gz)")
    p.add_argument("--output", required=True, help="Output directory for segmentations")

    # Environment
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")

    # TotalSegmentator options
    p.add_argument("--totalseg-task", action="append", default=[], help="TotalSegmentator task name (repeatable)")
    p.add_argument("--totalseg-out", type=str, default=None, help="Output dir for TotalSegmentator (default: output/totalseg_TASK)")
    p.add_argument("--totalseg-roi-subset", type=str, default=None, help="Comma-separated ROI subset for TotalSegmentator")
    p.add_argument("--totalseg-fast", action="store_true", default=True, help="Use TotalSegmentator fast mode")
    p.add_argument("--totalseg-fullres", dest="totalseg_fast", action="store_false", help="Disable TotalSegmentator fast mode")
    p.add_argument("--totalseg-ml", action="store_true", default=False, help="Write multi-label segmentator.nii.gz output")
    p.add_argument(
        "--totalseg-export",
        action="append",
        default=[],
        help="Export label mask: label=filename.nii.gz (repeatable)",
    )

    # MONAI bundle options
    p.add_argument("--monai-bundle", type=str, default=None, help="Path to MONAI bundle root")
    p.add_argument("--monai-config", action="append", default=[], help="Config file(s) relative to bundle root (repeatable)")
    p.add_argument("--monai-meta", type=str, default=None, help="Metadata JSON relative to bundle root")
    p.add_argument("--monai-run-id", type=str, default=None, help="MONAI bundle run_id")
    p.add_argument("--monai-init-id", type=str, default=None, help="MONAI bundle init_id")
    p.add_argument("--monai-final-id", type=str, default=None, help="MONAI bundle final_id")
    p.add_argument("--monai-args-file", type=str, default=None, help="Args file for MONAI bundle")
    p.add_argument("--monai-input-key", type=str, default="input", help="Override key for input path")
    p.add_argument("--monai-output-key", type=str, default="output", help="Override key for output path")
    p.add_argument("--monai-output", type=str, default=None, help="Output path passed to MONAI bundle")
    p.add_argument("--monai-override", action="append", default=[], help="Override config key=value (repeatable)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    require_monai = args.monai_bundle is not None
    require_totalseg = bool(args.totalseg_task)
    if args.check_env:
        _check_env(require_monai=require_monai, require_totalseg=require_totalseg)
        return 0

    if not require_monai and not require_totalseg:
        print("ERROR: Specify at least one of --totalseg-task or --monai-bundle")
        return 2

    if require_totalseg:
        roi_subset = None
        if args.totalseg_roi_subset:
            roi_subset = [s.strip() for s in args.totalseg_roi_subset.split(",") if s.strip()]

        for task in args.totalseg_task:
            task_dir = Path(args.totalseg_out) if args.totalseg_out else output_dir / f"totalseg_{task}"
            run_totalseg(
                input_path=input_path,
                output_dir=task_dir,
                task=task,
                roi_subset=roi_subset,
                fast=args.totalseg_fast,
                ml=args.totalseg_ml,
            )

            if args.totalseg_export and not args.totalseg_ml:
                ref_shape = nib.load(str(input_path)).shape
                export_dir = output_dir / "label_exports"
                supported_labels = None
                try:
                    import totalsegmentator.map_to_binary as m

                    label_map = m.class_map.get(task, {})
                    if label_map:
                        supported_labels = set(label_map.values())
                except Exception:
                    supported_labels = None
                for entry in args.totalseg_export:
                    label, out_name = _parse_key_value(entry)
                    if not isinstance(out_name, str):
                        raise ValueError(f"Invalid export entry: {entry}")
                    if supported_labels is not None and label not in supported_labels:
                        print(f"  ℹ Skipping export for {label}; not in task '{task}' label map.")
                        continue
                    ok = _export_totalseg_label(
                        task_dir,
                        label,
                        export_dir / out_name,
                        reference_shape=ref_shape,
                    )
                    if not ok:
                        print(f"  ⚠ Missing label in {task}: {label}")
            elif args.totalseg_export and args.totalseg_ml:
                print("  ⚠ --totalseg-export ignored with --totalseg-ml (no per-structure files).")

    if require_monai:
        bundle_root = Path(args.monai_bundle)
        config_files = args.monai_config or []
        if not config_files:
            print("ERROR: --monai-config is required when --monai-bundle is set")
            return 2
        config_paths = [_resolve_path(bundle_root, cfg) for cfg in config_files]
        meta_path = _resolve_path(bundle_root, args.monai_meta)
        args_path = _resolve_path(bundle_root, args.monai_args_file)
        monai_output = Path(args.monai_output) if args.monai_output else output_dir / "monai_output"

        overrides: dict[str, Any] = {}
        for pair in args.monai_override:
            key, val = _parse_key_value(pair)
            overrides[key] = val

        overrides.setdefault(args.monai_input_key, str(input_path))
        overrides.setdefault(args.monai_output_key, str(monai_output))
        overrides.setdefault("bundle_root", str(bundle_root))

        run_monai_bundle(
            bundle_root=bundle_root,
            config_files=config_paths,
            meta_file=meta_path,
            run_id=args.monai_run_id,
            init_id=args.monai_init_id,
            final_id=args.monai_final_id,
            args_file=args_path,
            overrides=overrides,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
