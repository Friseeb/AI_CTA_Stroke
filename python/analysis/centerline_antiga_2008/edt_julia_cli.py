"""Julia CLI-based EDT helper (avoids juliacall in-process crashes)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _find_julia_exe() -> str | None:
    env_exe = os.environ.get("JULIA_EXE")
    if env_exe and Path(env_exe).exists():
        return env_exe

    env_bindir = os.environ.get("JULIA_BINDIR")
    if env_bindir:
        candidate = Path(env_bindir) / "julia"
        if candidate.exists():
            return str(candidate)

    env_root = Path(sys.executable).resolve().parents[1]
    candidate = env_root / "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia"
    if candidate.exists():
        return str(candidate)

    return shutil.which("julia")


def _run_julia_transform(mask: np.ndarray, backend: str) -> np.ndarray | None:
    julia_exe = _find_julia_exe()
    if not julia_exe:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mask_path = tmpdir_path / "mask.npz"
        dist_path = tmpdir_path / "dist.npz"
        script_path = tmpdir_path / "compute_dt.jl"
        np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

        script = f"""
        using NPZ
        using DistanceTransforms
        backend = "{backend}"
        data = npzread(raw"{mask_path}")
        mask = data["mask"] .> 0
        if backend == "metal"
            using Metal
            if !Metal.functional()
                error("Metal not functional")
            end
            mask_gpu = Metal.MtlArray(mask)
            dist_gpu = DistanceTransforms.transform(DistanceTransforms.boolean_indicator(mask_gpu))
            dist = Array(dist_gpu)
        else
            dist = DistanceTransforms.transform(DistanceTransforms.boolean_indicator(mask))
        end
        npzwrite(raw"{dist_path}", Dict("dist" => dist))
        """
        script_path.write_text(script)

        result = subprocess.run(
            [julia_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None

        if not dist_path.exists():
            return None

        dist_sq = np.load(dist_path)["dist"]
        return np.asarray(dist_sq)


def distance_transform_julia_cli(
    mask: np.ndarray,
    backend: str = "auto",
) -> np.ndarray | None:
    """Return EDT from Julia DistanceTransforms, or None on failure."""
    backend = (backend or "auto").lower()

    dist_sq = None
    if backend in ("auto", "metal"):
        dist_sq = _run_julia_transform(mask, backend="metal")
        if dist_sq is None and backend == "metal":
            return None

    if dist_sq is None:
        dist_sq = _run_julia_transform(mask, backend="cpu")
    if dist_sq is None:
        return None

    if np.max(dist_sq) < 1e-6:
        return None

    if np.any(mask > 0):
        dist_inside_max = float(dist_sq[mask > 0].max())
        if dist_inside_max < 1e-6:
            retry_backend = "metal" if backend == "metal" else "cpu"
            dist_sq = _run_julia_transform(1 - mask.astype(np.uint8), backend=retry_backend)
            if dist_sq is None:
                return None

    return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)
