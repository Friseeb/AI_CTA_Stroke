#!/usr/bin/env python3
"""
Run only the NUDF LAA step from CardiacCTExplorer using existing TotalSegmentator outputs.

This script expects multi-label TotalSegmentator outputs:
  - total.nii.gz
  - heartchambers_highres.nii.gz

If they are missing, you can pass --run-totalseg to generate them.

Example:
  python scripts/run_cardiac_ct_explorer_nudf_only.py \
    --input /path/to/sub-547_defaced.nii.gz \
    --output-dir /path/to/outputs/cardiac_ct_explorer_547 \
    --laa-output /path/to/outputs/cardiac_ct_explorer_547/sub-547_defaced_laa8.nii.gz \
    --run-totalseg --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CardiacCTExplorer NUDF-only LAA segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input NIfTI (.nii/.nii.gz)")
    p.add_argument("--output-dir", required=True, help="Output directory for CardiacCTExplorer")
    p.add_argument("--laa-output", required=True, help="Output LAA mask NIfTI (.nii/.nii.gz)")

    p.add_argument("--device", default="auto", help="auto|cpu|gpu")
    p.add_argument("--run-totalseg", action="store_true", help="Run TotalSegmentator if outputs are missing")
    p.add_argument("--totalseg-device", default=None, help="Override TotalSegmentator device (cpu|gpu)")
    p.add_argument("--roi-subset-total", default=None, help="Comma-separated ROI subset for TotalSegmentator total task")
    p.add_argument(
        "--roi-subset-heartchambers",
        default=None,
        help="Comma-separated ROI subset for TotalSegmentator heartchambers_highres task",
    )
    p.add_argument("--skip-coronary", action="store_true", help="Skip coronary_arteries task (faster)")
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")
    return p.parse_args()


def _check_env() -> None:
    details = {}
    for name in ("cardiacctexplorer", "totalsegmentator", "nibabel", "numpy", "SimpleITK"):
        try:
            module = __import__(name)
            details[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            details[f"{name}_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(key.endswith("_error") for key in details):
        raise SystemExit(2)


def _scan_id(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return input_path.stem


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
        return "cpu"
    if device == "gpu":
        try:
            import torch

            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
        return "cpu"
    return device


def _parse_roi_subset(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() in {"none", "null"}:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _ensure_totalseg_outputs(input_path: Path, ts_folder: Path, device: str, run_totalseg: bool) -> None:
    total_out = ts_folder / "total.nii.gz"
    hc_out = ts_folder / "heartchambers_highres.nii.gz"
    if total_out.exists() and hc_out.exists():
        return
    if not run_totalseg:
        missing = [str(p) for p in (total_out, hc_out) if not p.exists()]
        raise FileNotFoundError(
            "Missing TotalSegmentator outputs: " + ", ".join(missing) +
            ". Run with --run-totalseg or place the files in the expected folder."
        )

    from totalsegmentator.python_api import totalsegmentator

    ts_folder.mkdir(parents=True, exist_ok=True)
    if not total_out.exists():
        totalsegmentator(
            input=str(input_path),
            output=str(total_out),
            task="total",
            ml=True,
            fast=False,
            device=device,
        )
    if not hc_out.exists():
        totalsegmentator(
            input=str(input_path),
            output=str(hc_out),
            task="heartchambers_highres",
            ml=True,
            fast=False,
            device=device,
        )


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    laa_output = Path(args.laa_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    laa_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cardiacctexplorer.python_api import get_default_parameters
        from cardiacctexplorer.nudf_laa_utils import nudf_laa_analysis
        import cardiacctexplorer.general_utils as gu
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CardiacCTExplorer not installed in this env. "
            "Install it or activate the correct env."
        ) from exc

    params = get_default_parameters()
    params["device"] = _resolve_device(args.device)
    if args.totalseg_device:
        params["device_totalsegmentator"] = args.totalseg_device
    params["roi_subset_total"] = _parse_roi_subset(args.roi_subset_total)
    params["roi_subset_heartchambers"] = _parse_roi_subset(args.roi_subset_heartchambers)
    if args.skip_coronary:
        params["skip_coronary"] = True
    # Extra safety: if this torch build has no CUDA, force NUDF to CPU.
    try:
        import torch

        if not torch.cuda.is_available():
            params["device"] = "cpu"
            # Guard against any CUDA selection inside CardiacCTExplorer.
            import cardiacctexplorer.nudf_laa_utils as nudf_utils

            def _safe_select_device(_device: str) -> str:
                return "cpu"

            nudf_utils.select_device = _safe_select_device
    except Exception:
        pass

    # Set folders and ensure TotalSegmentator outputs are present
    output_dir_str = str(output_dir)
    if not output_dir_str.endswith(os.sep):
        output_dir_str = output_dir_str + os.sep

    params = gu.set_and_create_folders(str(input_path), output_dir_str, params)
    ts_folder = Path(params["ts_folder"])
    ts_device = params.get("device_totalsegmentator", params["device"])
    _ensure_totalseg_outputs(input_path, ts_folder, ts_device, args.run_totalseg)

    ok = nudf_laa_analysis([str(input_path)], output_dir_str, params)
    if ok is False:
        raise RuntimeError("NUDF LAA analysis failed; check logs in the output folder.")

    # Copy final LAA label to requested output location
    scan_id = _scan_id(input_path)
    default_laa = output_dir / scan_id / "segmentations" / "laa_nudf_label.nii.gz"
    if not default_laa.exists():
        raise FileNotFoundError(f"Expected LAA output not found: {default_laa}")
    shutil.copy2(default_laa, laa_output)
    print(f"Saved: {laa_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
