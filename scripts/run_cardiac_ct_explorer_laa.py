#!/usr/bin/env python3
"""
Run CardiacCTExplorer on a CTA NIfTI and export the LAA (label 8) as a mask.

This script uses the CardiacCTExplorer Python API, then extracts the LAA label
from the combined segmentation output.

Example:
  python scripts/run_cardiac_ct_explorer_laa.py \
    --input /path/to/<CASE_ID>_defaced.nii.gz \
    --output-dir /path/to/outputs/cardiac_ct_explorer_<CASE_ID> \
    --laa-output /path/to/outputs/cardiac_ct_explorer_<CASE_ID>/<CASE_ID>_defaced_laa8.nii.gz \
    --device auto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CardiacCTExplorer and export LAA label (8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input NIfTI (.nii/.nii.gz)")
    p.add_argument("--output-dir", required=True, help="Output directory for CardiacCTExplorer")
    p.add_argument("--laa-output", required=True, help="Output LAA mask NIfTI (.nii/.nii.gz)")

    p.add_argument("--label-id", type=int, default=8, help="Label ID for LAA in CardiacCTExplorer output")
    p.add_argument("--binary", action="store_true", help="Save LAA as binary mask (1/0)")
    p.add_argument("--skip-run", action="store_true", help="Skip CardiacCTExplorer run; only extract LAA")

    p.add_argument("--device", default="auto", help="auto|cpu|gpu")
    p.add_argument("--ts-procs", type=int, default=None, help="TotalSegmentator processes")
    p.add_argument("--nudf-procs", type=int, default=None, help="NUDF processes")
    p.add_argument("--general-procs", type=int, default=None, help="General processes")
    p.add_argument("--image-cas", action="store_true", help="Enable ImageCAS mode")
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")

    return p.parse_args()


def _check_env() -> None:
    details = {}
    for name in ("cardiacctexplorer", "totalsegmentator", "nibabel", "numpy"):
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


def _find_segmentation(output_dir: Path, scan_id: str) -> Path:
    # Preferred: combined segmentations copied to all_segmentations
    candidate = output_dir / "all_segmentations" / f"{scan_id}_cardiac_segmentations.nii.gz"
    if candidate.exists():
        return candidate
    # Fallback: per-scan folder
    candidate = output_dir / scan_id / "segmentations" / "cardiac_combined_segmentation.nii.gz"
    if candidate.exists():
        return candidate
    # Last resort: glob
    matches = list(output_dir.rglob(f"{scan_id}*cardiac*segmentation*.nii*"))
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError("Could not locate CardiacCTExplorer segmentation output")


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
        return "cpu"
    return device


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

    if not args.skip_run:
        try:
            from cardiacctexplorer.python_api import cardiacctexplorer, get_default_parameters
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "CardiacCTExplorer not installed in this env. "
                "Install it or activate the correct env."
            ) from exc

        params = get_default_parameters()
        params["device"] = _resolve_device(args.device)
        params["image_cas_mode"] = args.image_cas
        if args.ts_procs is not None:
            params["num_proc_total_segmentator"] = args.ts_procs
        if args.nudf_procs is not None:
            params["num_proc_nudf"] = args.nudf_procs
        if args.general_procs is not None:
            params["num_proc_general"] = args.general_procs

        ok = cardiacctexplorer(str(input_path), str(output_dir), params)
        if not ok:
            raise RuntimeError("CardiacCTExplorer reported failure; check its logs.")

    seg_path = _find_segmentation(output_dir, _scan_id(input_path))
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nibabel/numpy missing; install them in the active env.") from exc

    img = nib.load(str(seg_path))
    data = img.get_fdata()
    label_id = int(args.label_id)
    if args.binary:
        mask = (data == label_id).astype("uint8")
    else:
        mask = (data == label_id).astype("uint16") * np.uint16(label_id)
    header = img.header.copy()
    header.set_data_dtype(mask.dtype)
    nib.save(nib.Nifti1Image(mask, img.affine, header), str(laa_output))

    print(f"Saved: {laa_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
