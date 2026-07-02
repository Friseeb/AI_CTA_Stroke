"""Pipeline subprocess runner with live log streaming."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
_CONDA_BIN = Path(sys.executable).parent  # bin dir of the active Python


def _env_with_conda_bin() -> dict[str, str]:
    """Return os.environ with the active conda env's bin prepended to PATH.

    This makes cta-dental, TotalSegmentator, dcm2niix, etc. findable as CLI
    tools by subprocesses even when the conda env is not 'activated'."""
    env = dict(os.environ)
    env["PATH"] = str(_CONDA_BIN) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_streaming(cmd: list[str], cwd: Path | None = None) -> Iterator[str]:
    """Run a subprocess and yield stdout+stderr lines as they arrive."""
    env = _env_with_conda_bin()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(cwd or REPO_ROOT),
        env=env,
    )
    assert proc.stdout is not None

    q: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            q.put(line.rstrip())
        q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    while True:
        line = q.get()
        if line is None:
            break
        yield line

    proc.wait()


def dicom_to_nifti(dicom_dir: Path, out_dir: Path, case_id: str) -> Path:
    """Convert a DICOM folder to NIfTI using dcm2niix. Returns the NIfTI path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dcm2niix = shutil.which("dcm2niix") or str(_CONDA_BIN / "dcm2niix")
    cmd = [dcm2niix, "-z", "y", "-f", case_id, "-o", str(out_dir), str(dicom_dir)]
    for line in run_streaming(cmd):
        yield line  # type: ignore[misc]
    nifti = out_dir / f"{case_id}.nii.gz"
    if not nifti.exists():
        raise FileNotFoundError(f"dcm2niix did not produce {nifti}")
    return nifti  # type: ignore[return-value]


def run_dental(
    nifti: Path,
    out_dir: Path,
    case_id: str,
    device: str = "auto",
) -> Iterator[str]:
    cmd = [
        sys.executable,
        "-m", "cta_dental.cli",  # avoids PATH dependency for the cta-dental script
        "run",
        str(nifti),
        "--out", str(out_dir),
        "--case-id", case_id,
        "--segmenter", "totalseg_teeth",
        "--roi-method", "totalseg_teeth",
        "--reuse-roi-seg",
        "--verbose",
    ]
    yield from run_streaming(cmd)


def run_laa(
    nifti: Path,
    out_dir: Path,
    case_id: str,
    device: str = "gpu",
    skip_vista3d: bool = False,
) -> Iterator[str]:
    totalseg_out = out_dir / "totalseg_total"
    vista_out = out_dir / "nv_segment_ct" / f"{case_id}_laa108.nii.gz"
    fusion_out = out_dir / "prior_fusion" / case_id

    # Stage 1: TotalSegmentator
    yield "--- Stage 1: TotalSegmentator ---"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "total_segmentator.py"),
        "--input", str(nifti),
        "--output", str(totalseg_out),
        "--device", device,
        "--fast",
    ]
    yield from run_streaming(cmd)

    # Stage 2: VISTA3D
    if not skip_vista3d:
        yield "--- Stage 2: VISTA3D (NV-Segment-CT) ---"
        vista_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "run_nv_segment_ct_laa.py"),
            "--input", str(nifti),
            "--output", str(vista_out),
            "--device", "auto",
        ]
        yield from run_streaming(cmd)

    # Stage 3: Prior fusion
    yield "--- Stage 3: Prior fusion ---"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_prior_fusion.py"),
        "--case-id", case_id,
        "--input", str(nifti),
        "--totalseg-total-dir", str(totalseg_out),
        "--out-dir", str(fusion_out),
        "--device", device,
    ]
    if vista_out.exists():
        cmd += ["--vista3d-combined", str(vista_out)]
    yield from run_streaming(cmd)


def run_aortic(
    nifti: Path,
    out_dir: Path,
    case_id: str,
    tasks: list[str] | None = None,
    device: str = "gpu",
) -> Iterator[str]:
    tasks = tasks or ["Calcium", "Fat", "Wall"]
    yield f"--- Aortic pipeline: {', '.join(tasks)} ---"
    script = REPO_ROOT / "scripts" / "run_aortic.py"
    if not script.exists():
        yield "[Aortic] Not yet implemented — add scripts/run_aortic.py to enable"
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--input", str(nifti),
        "--out", str(out_dir),
        "--case-id", case_id,
        "--device", device,
        "--tasks", *[t.lower() for t in tasks],
    ]
    yield from run_streaming(cmd)


def run_sleep_apnea(
    nifti: Path,
    out_dir: Path,
    case_id: str,
    device: str = "gpu",
) -> Iterator[str]:
    yield "--- Sleep Apnea pipeline ---"
    script = REPO_ROOT / "scripts" / "run_sleep_apnea.py"
    if not script.exists():
        yield "[Sleep Apnea] Not yet implemented — add scripts/run_sleep_apnea.py to enable"
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--input", str(nifti),
        "--out", str(out_dir),
        "--case-id", case_id,
        "--device", device,
    ]
    yield from run_streaming(cmd)


def run_batch_dental(
    nifti_paths: list[Path],
    out_root: Path,
    device: str = "auto",
) -> Iterator[tuple[str, str]]:
    """Yield (case_id, log_line) tuples for a batch dental run."""
    for nifti in nifti_paths:
        case_id = nifti.name.replace(".nii.gz", "").replace(".nii", "")
        case_out = out_root / case_id
        yield case_id, f"=== Starting {case_id} ==="
        for line in run_dental(nifti, case_out, case_id, device):
            yield case_id, line
        yield case_id, f"=== Done {case_id} ==="
