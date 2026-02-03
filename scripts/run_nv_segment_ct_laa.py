#!/usr/bin/env python3
"""
Run NV-Segment-CT (VISTA3D) to segment a single class from a CTA volume.

This script downloads the NV-Segment-CT Hugging Face repo (if missing),
runs inference with a label prompt, and writes a single NIfTI mask.

Example:
  python scripts/run_nv_segment_ct_laa.py \
    --input /path/to/sub-547_defaced.nii.gz \
    --output /path/to/sub-547_defaced_laa108.nii.gz \
    --model-dir /path/to/external/nv_segment_ct \
    --device auto

Notes:
  - Default label-id is 108 (Left Atrial Appendage).
  - By default the output mask is binarized and set to the label-id value.
    Use --keep-values to preserve the model's native output values.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path



def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment NV-Segment-CT class 108 (LAA) from a CTA NIfTI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=False, help="Input NIfTI (.nii/.nii.gz)")
    p.add_argument("--output", required=False, help="Output mask NIfTI (.nii/.nii.gz)")

    p.add_argument("--label-id", type=int, default=108, help="Label ID to segment")
    p.add_argument(
        "--model-dir",
        default=None,
        help="Local NV-Segment-CT repo directory (downloaded if missing)",
    )
    p.add_argument(
        "--hf-repo",
        default="nvidia/NV-Segment-CT",
        help="Hugging Face repo to download",
    )
    p.add_argument("--revision", default=None, help="Optional HF revision (commit/tag)")
    p.add_argument("--cache-dir", default=None, help="Optional HF cache dir")

    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cpu|mps|cuda:0 (auto uses cuda if available, else cpu)",
    )
    p.add_argument(
        "--allow-mps",
        action="store_true",
        help="Allow MPS when --device auto (MPS may fail with float64 ops).",
    )
    p.add_argument(
        "--force-mps",
        action="store_true",
        help="Force MPS even if known-unsupported ops are present (may crash).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Work dir for pipeline outputs (auto temp if omitted)",
    )
    p.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the auto-created work dir",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the NV-Segment-CT repo",
    )
    p.add_argument(
        "--keep-values",
        action="store_true",
        help="Keep native output values (do not remap to label-id)",
    )
    p.add_argument(
        "--check-env",
        action="store_true",
        help="Print environment versions and exit",
    )
    return p.parse_args()


def _check_env() -> None:
    details = {}
    for name in ("torch", "monai", "transformers", "huggingface_hub", "nibabel", "numpy"):
        try:
            module = __import__(name)
            details[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            details[f"{name}_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(key.endswith("_error") for key in details):
        raise SystemExit(2)


def _ensure_repo(
    model_dir: Path,
    hf_repo: str,
    cache_dir: Path | None,
    revision: str | None,
    force: bool,
) -> None:
    marker = model_dir / "vista3d_pretrained_model"
    if marker.exists() and not force:
        return
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
    )


def _resolve_device(device: str, allow_mps: bool):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _find_output_file(output_dir: Path, postfix: str) -> Path:
    candidates = sorted(output_dir.glob(f"*_{postfix}.nii*"))
    if not candidates:
        candidates = sorted(output_dir.glob("*.nii*"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*.nii*"))
    if not candidates:
        raise FileNotFoundError(f"No NIfTI outputs found in {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _remap_to_label(output_file: Path, output_path: Path, label_id: int) -> None:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(output_file))
    data = img.get_fdata()
    mask = (data > 0).astype(np.uint16) * np.uint16(label_id)
    header = img.header.copy()
    header.set_data_dtype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine, header), str(output_path))


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    if not args.input or not args.output:
        print("Missing required arguments: --input and --output")
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    model_dir = Path(args.model_dir) if args.model_dir else _repo_root() / "external" / "nv_segment_ct"
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    _ensure_repo(model_dir, args.hf_repo, cache_dir, args.revision, args.force_download)

    work_dir: Path | None = None
    cleanup_work_dir = False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="nv_segment_ct_", dir=output_path.parent))
        cleanup_work_dir = not args.keep_work_dir

    sys.path.insert(0, str(model_dir))
    from hugging_face_pipeline import HuggingFacePipelineHelper  # noqa: E402
    import torch  # noqa: E402

    device = _resolve_device(args.device, args.allow_mps)
    if device.type == "mps" and not args.force_mps:
        print("MPS backend lacks ConvTranspose3D support for this model; falling back to CPU.")
        device = torch.device("cpu")
    pipeline_helper = HuggingFacePipelineHelper("vista3d")
    pipeline = pipeline_helper.init_pipeline(
        str(model_dir / "vista3d_pretrained_model"),
        device=device,
    )

    label_id = int(args.label_id)
    inputs = [{"image": str(input_path), "label_prompt": [label_id]}]
    output_postfix = f"label{label_id}"
    try:
        pipeline(
            inputs,
            output_dir=str(work_dir),
            output_postfix=output_postfix,
            output_ext=".nii.gz",
            separate_folder=False,
            amp=device.type == "cuda",
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "mps" in msg.lower() and "float64" in msg.lower():
            print("MPS float64 unsupported in MONAI transforms; retrying on CPU.")
            device = torch.device("cpu")
            pipeline = pipeline_helper.init_pipeline(
                str(model_dir / "vista3d_pretrained_model"),
                device=device,
            )
            pipeline(
                inputs,
                output_dir=str(work_dir),
                output_postfix=output_postfix,
                output_ext=".nii.gz",
                separate_folder=False,
                amp=False,
            )
        elif "label prompt" in msg.lower() or "label_prompt" in msg.lower():
            inputs = [{"image": str(input_path), "label_prompt": [torch.tensor([label_id])]}]
            pipeline(
                inputs,
                output_dir=str(work_dir),
                output_postfix=output_postfix,
                output_ext=".nii.gz",
                separate_folder=False,
                amp=device.type == "cuda",
            )
        else:
            raise

    output_file = _find_output_file(work_dir, output_postfix)
    if args.keep_values:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_file, output_path)
    else:
        _remap_to_label(output_file, output_path, label_id)

    if cleanup_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
