"""Wrappers for external CLI tools: TotalSegmentator, nnUNetv2."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

log = get_logger("external_tools")


def find_totalsegmentator() -> Optional[str]:
    return shutil.which("TotalSegmentator")


def require_totalsegmentator() -> str:
    exe = find_totalsegmentator()
    if exe is None:
        raise RuntimeError(
            "TotalSegmentator not found. Install it with:\n"
            "  pip install TotalSegmentator\n"
            "or choose --roi-method threshold_fallback for degraded ROI only."
        )
    return exe


def run_totalsegmentator(
    input_nifti: Path,
    output_dir: Path,
    task: str = "teeth",
    fast: bool = False,
    device: str = "cpu",
    weights_dir: Optional[str] = None,
    timeout: int = 3600,
) -> None:
    """Run TotalSegmentator via Python API (preferred) or CLI fallback."""
    require_totalsegmentator()  # ensures it's installed
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_totalseg_python_api(input_nifti, output_dir, task, fast, device, weights_dir)
    except Exception as exc:
        log.warning("TotalSegmentator Python API failed (%s); falling back to CLI.", exc)
        _run_totalseg_cli(input_nifti, output_dir, task, fast, device, weights_dir, timeout)


def _run_totalseg_python_api(
    input_nifti: Path,
    output_dir: Path,
    task: str,
    fast: bool,
    device: str,
    weights_dir: Optional[str],
) -> None:
    import inspect
    import totalsegmentator.python_api as ts_mod
    from totalsegmentator.python_api import totalsegmentator as ts_api

    # Workaround for TotalSegmentator ≥2.12 bug (CPU-only systems):
    # The `teeth` task calls totalsegmentator() recursively for its crop model
    # without forwarding `device`. When no GPU is present, select_device() returns
    # the string "cpu"; convert_device_to_string("cpu") then returns None (it only
    # handles torch.device objects); validate_device_type_api(None) raises TypeError;
    # and select_device(None) raises AttributeError in convert_device_to_cuda.
    # We patch all three functions to treat None as "cpu" for the duration of the call.
    _patches: dict = {}
    for fname, impl in [
        ("validate_device_type_api", lambda v: (None if v is None else None) or
            ts_mod.__dict__.get("_orig_validate_device_type_api", lambda x: x)(v if v is not None else "cpu")),
        ("convert_device_to_string", None),
        ("select_device", None),
    ]:
        pass  # handled below

    orig_validate = getattr(ts_mod, "validate_device_type_api", None)
    orig_convert = getattr(ts_mod, "convert_device_to_string", None)
    orig_select = getattr(ts_mod, "select_device", None)

    if orig_validate:
        def _safe_validate(value, _orig=orig_validate):
            return _orig("cpu" if value is None else value)
        ts_mod.validate_device_type_api = _safe_validate

    if orig_convert:
        def _safe_convert(dev, _orig=orig_convert):
            if dev is None:
                return "cpu"
            if isinstance(dev, str):
                return dev  # already a string ("cpu", "gpu", "mps") — pass through
            return _orig(dev)
        ts_mod.convert_device_to_string = _safe_convert

    if orig_select:
        def _safe_select(dev, _orig=orig_select):
            return _orig("cpu" if dev is None else dev)
        ts_mod.select_device = _safe_select

    try:
        sig = inspect.signature(ts_api)
        kwargs: dict = dict(task=task, fast=fast)
        if "device" in sig.parameters:
            kwargs["device"] = device
        if weights_dir and "weights_dir" in sig.parameters:
            kwargs["weights_dir"] = weights_dir

        log.info("Running TotalSegmentator (Python API): task=%s device=%s", task, device)
        ts_api(str(input_nifti), str(output_dir), **kwargs)
        log.info("TotalSegmentator (Python API) completed successfully.")
    finally:
        if orig_validate:
            ts_mod.validate_device_type_api = orig_validate
        if orig_convert:
            ts_mod.convert_device_to_string = orig_convert
        if orig_select:
            ts_mod.select_device = orig_select


def _run_totalseg_cli(
    input_nifti: Path,
    output_dir: Path,
    task: str,
    fast: bool,
    device: str,
    weights_dir: Optional[str],
    timeout: int,
) -> None:
    exe = require_totalsegmentator()
    cmd = [exe, "-i", str(input_nifti), "-o", str(output_dir), "--task", task]
    if fast:
        cmd.append("--fast")
    if weights_dir:
        cmd.extend(["--weights_dir", weights_dir])
    log.info("Running TotalSegmentator (CLI): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("TotalSegmentator CLI failed (rc=%d):\n%s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(
            f"TotalSegmentator exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[-1000:]}"
        )
    log.info("TotalSegmentator (CLI) completed successfully.")


def find_nnunet_predict() -> Optional[str]:
    for name in ("nnUNetv2_predict", "nnunetv2_predict"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def run_nnunet_predict(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int,
    configuration: str,
    fold: str,
    trainer: str,
    results_folder: Path,
    timeout: int = 7200,
) -> subprocess.CompletedProcess:
    exe = find_nnunet_predict()
    if exe is None:
        raise RuntimeError(
            "nnUNetv2_predict not found. Install nnU-Net v2:\n"
            "  pip install nnunetv2"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", fold,
        "-tr", trainer,
        "-i", str(input_dir),
        "-o", str(output_dir),
    ]
    import os
    env = os.environ.copy()
    env["nnUNet_results"] = str(results_folder)
    log.info("Running nnUNetv2_predict: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode != 0:
        log.error("nnUNetv2_predict failed (rc=%d):\n%s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(
            f"nnUNetv2_predict exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[-1000:]}"
        )
    log.info("nnUNetv2_predict completed successfully.")
    return result
