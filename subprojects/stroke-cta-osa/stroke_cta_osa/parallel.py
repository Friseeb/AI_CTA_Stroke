"""Memory-aware parallel batch execution.

Each case is fully independent (its own ``<case_hash>/`` output sub-dir), so a
batch parallelises trivially across processes. The binding constraint is **RAM,
not CPU**: a single large head/neck CTA peaks well over 20 GB during the airway
connected-component fallback and the distance-transform fat ROIs. So the worker
count is chosen from *available memory* and the *estimated peak of the largest
input*, then capped by CPU count and the number of cases.

Calibration anchor (measured on this codebase), for a 512×512×1819 int16
volume (~0.95 GB of raw voxels): the whole pipeline peaks **~8.3 GB (≈9× raw)**
after the memory fixes — shared single body silhouette, slim ``body_mask``,
free-at-last-use in ``compute_fat_features``, single parapharyngeal union, and
an airway fallback that reuses ``body_mask`` and frees its label volumes. The
peak is the same whether the airway is computed in-process or supplied via a
mask (both are fat-bound now), so a single multiplier with a little headroom
covers both. (Before these fixes it was ~22×.)

A second optimisation prevents BLAS/ITK **thread oversubscription**: with N
worker processes each spawning one thread per core you get N×cores threads
thrashing. We divide cores across workers and export the per-worker thread caps
*before* the pool spawns, so children inherit them at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PipelineConfig

# --- calibration constants --------------------------------------------------

# Peak RSS ≈ PEAK_MULTIPLIER × raw voxel bytes. Observed ≈9× post-optimisation
# (same for in-process and supplied-mask paths — both are fat-bound); 11× adds
# headroom for heavier-than-average cases.
PEAK_MULTIPLIER = 11.0
# Never assume a case needs less than this (header probe failures, small FOV).
PEAK_FLOOR_GB = 2.5
# Fraction of *available* RAM we are willing to commit to workers.
DEFAULT_RESERVE_FRACTION = 0.85
# Assumed gzip decompression ratio when we can only see the compressed size.
GZIP_DECOMP_RATIO = 3.0
# Bytes per voxel when the header doesn't tell us (CT is typically int16).
ASSUMED_BYTES_PER_VOXEL = 2


@dataclass(frozen=True)
class WorkerPlan:
    """Result of :func:`auto_worker_count` — carries the rationale for logging."""
    workers: int
    threads_per_worker: int
    per_case_peak_gb: float
    available_gb: float
    cpu_count: int
    n_cases: int
    reason: str


@dataclass
class CaseJob:
    """One unit of work dispatched to a worker. Must be picklable (it is:
    ``PipelineConfig`` is a pydantic model and the rest are plain types)."""
    input_path: str
    out_dir: str
    pid: str
    cfg: PipelineConfig
    verbose: bool = False
    external_airway_mask: Optional[str] = None  # cached mask from pass 1


@dataclass
class CaseOutcome:
    pid: str
    input_path: str
    result: Optional[object]   # CaseResult, or None on failure
    error: Optional[str]


@dataclass
class AirwayJob:
    """Pass-1 unit: compute and cache an airway mask only."""
    input_path: str
    cache_path: str
    pid: str
    cfg: PipelineConfig
    verbose: bool = False


@dataclass
class AirwayOutcome:
    pid: str
    input_path: str
    mask_path: Optional[str]   # cached NIfTI path, or None if no airway / failed
    source: str
    error: Optional[str]


# --- memory probing ---------------------------------------------------------

def available_ram_gb() -> float:
    """Best-effort available (not just free) RAM in GB.

    Prefers psutil; falls back to POSIX sysconf; finally a conservative default.
    """
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        pass
    try:  # Linux exposes available physical pages
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:  # total physical as a last-resort upper bound
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0  # safe-ish default for a modern workstation


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def input_raw_bytes(path: Path) -> float:
    """Estimate the *raw* (uncompressed) voxel byte count for one input.

    NIfTI: read the header lazily (no pixel load). DICOM dir / zip: use on-disk
    size as a proxy (uncompressed DICOM ≈ raw; gz/zip get a decompression bump).
    """
    p = Path(path)
    name = p.name.lower()

    if p.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz")):
        # Try a header-only read first — exact and cheap.
        try:
            import nibabel as nib  # type: ignore
            img = nib.load(str(p))  # lazy: header only until dataobj is touched
            shape = img.shape
            try:
                itemsize = int(img.header.get_data_dtype().itemsize)
            except Exception:
                itemsize = ASSUMED_BYTES_PER_VOXEL
            n = 1
            for d in shape:
                n *= int(d)
            return float(n * max(itemsize, 1))
        except Exception:
            pass
        try:
            import SimpleITK as sitk  # type: ignore
            r = sitk.ImageFileReader()
            r.SetFileName(str(p))
            r.ReadImageInformation()
            n = 1
            for d in r.GetSize():
                n *= int(d)
            return float(n * ASSUMED_BYTES_PER_VOXEL)
        except Exception:
            pass
        # Last resort: scale compressed file size.
        size = p.stat().st_size
        return float(size * (GZIP_DECOMP_RATIO if name.endswith(".gz") else 1.0))

    if p.is_dir():
        return float(_dir_bytes(p))  # DICOM series ≈ raw bytes
    if p.is_file() and name.endswith(".zip"):
        return float(p.stat().st_size * GZIP_DECOMP_RATIO)
    if p.is_file():
        return float(p.stat().st_size)
    return float(PEAK_FLOOR_GB / PEAK_MULTIPLIER * 1e9)  # unknown → floor peak


def estimate_peak_gb(
    path: Path,
    *,
    multiplier: float = PEAK_MULTIPLIER,
    floor_gb: float = PEAK_FLOOR_GB,
) -> float:
    """Estimated peak RSS (GB) for processing one case."""
    raw_gb = input_raw_bytes(path) / 1e9
    return max(floor_gb, raw_gb * multiplier)


# --- worker-count selection -------------------------------------------------

def auto_worker_count(
    paths: list[Path],
    *,
    requested: Optional[int] = None,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    multiplier: float = PEAK_MULTIPLIER,
    floor_gb: float = PEAK_FLOOR_GB,
    cpu_cap: Optional[int] = None,
) -> WorkerPlan:
    """Choose a safe worker count.

    Sizes memory by the **largest** input so that any of the N concurrent slots
    can hold the biggest case without OOM. ``requested`` (an explicit
    ``--workers N``) is still clamped by memory headroom — we never let an
    explicit value exceed what RAM can hold for the largest case.
    """
    n_cases = max(1, len(paths))
    cpu = os.cpu_count() or 1
    cap = cpu_cap or cpu

    peak = max((estimate_peak_gb(p, multiplier=multiplier, floor_gb=floor_gb)
                for p in paths), default=floor_gb)
    avail = available_ram_gb() * reserve_fraction
    by_mem = max(1, int(avail // peak))

    if requested is not None and requested > 0:
        workers = min(requested, by_mem, n_cases)
        why = "explicit --workers, clamped by memory headroom" \
            if workers < requested else "explicit --workers"
    else:
        workers = max(1, min(by_mem, cap, n_cases))
        why = "auto: memory-bound" if by_mem <= min(cap, n_cases) else \
              ("auto: cpu-bound" if cap <= n_cases else "auto: case-bound")

    workers = max(1, workers)
    threads = max(1, cpu // workers)
    reason = (
        f"{why}; largest-case peak≈{peak:.1f}GB, "
        f"usable RAM≈{avail:.1f}GB ⇒ {by_mem} by-memory, "
        f"{cpu} cpu, {n_cases} cases ⇒ {workers} workers × {threads} threads"
    )
    return WorkerPlan(
        workers=workers, threads_per_worker=threads, per_case_peak_gb=peak,
        available_gb=avail, cpu_count=cpu, n_cases=n_cases, reason=reason,
    )


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def apply_thread_limits(threads_per_worker: int) -> None:
    """Export per-worker thread caps so spawned children inherit them.

    Must be called in the parent *before* the pool is created — numpy/BLAS/ITK
    read these at import time, which in a spawned child happens before any
    initializer runs.
    """
    val = str(max(1, threads_per_worker))
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, val)


# --- the worker -------------------------------------------------------------

def run_case(job: CaseJob) -> CaseOutcome:
    """Process one case in a worker process. Top-level so it is picklable."""
    from .features import extract_case
    from .logging_utils import configure_logging
    configure_logging(verbose=job.verbose)
    try:
        mask = Path(job.external_airway_mask) if job.external_airway_mask else None
        result = extract_case(
            input_path=Path(job.input_path), out_dir=Path(job.out_dir),
            cfg=job.cfg, patient_id=job.pid,
            external_airway_mask_path=mask,
        )
        return CaseOutcome(job.pid, job.input_path, result, None)
    except Exception as exc:  # never let one bad case kill the pool
        return CaseOutcome(job.pid, job.input_path, None, repr(exc))


def run_airway_precompute(job: AirwayJob) -> AirwayOutcome:
    """Pass-1 worker: build the airway mask via the provider chain and cache it.

    Reusing this cache across feature re-runs / config sweeps avoids recomputing
    the connected-component airway segmentation, the second-heaviest stage.
    """
    from .adapters import build_airway_provider_chain, first_available
    from .io import load_input, save_mask
    from .logging_utils import configure_logging
    configure_logging(verbose=job.verbose)
    try:
        image, _ = load_input(Path(job.input_path))
        info, payload = first_available(build_airway_provider_chain(job.cfg), image)
        if info is not None and info.is_present:
            out = Path(job.cache_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            save_mask(info.mask_zyx, image, out)
            return AirwayOutcome(job.pid, job.input_path, str(out),
                                 payload.source, None)
        return AirwayOutcome(job.pid, job.input_path, None, payload.source,
                             "no airway mask produced")
    except Exception as exc:
        return AirwayOutcome(job.pid, job.input_path, None, "none", repr(exc))
