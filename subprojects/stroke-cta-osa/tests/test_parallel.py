"""Memory-aware worker selection + parallel batch execution."""

import pickle
from pathlib import Path

import pytest
import SimpleITK as sitk
from typer.testing import CliRunner

from stroke_cta_osa import parallel
from stroke_cta_osa.cli import app

runner = CliRunner()


# --- worker-count math (deterministic, no real compute) --------------------

def test_estimate_peak_has_floor(tmp_path):
    # A tiny/unknown input still reserves at least the floor.
    p = tmp_path / "tiny.txt"
    p.write_text("x")
    assert parallel.estimate_peak_gb(p) >= parallel.PEAK_FLOOR_GB


def test_auto_worker_count_memory_bound(monkeypatch, tmp_path):
    paths = [tmp_path / f"c{i}.nii.gz" for i in range(8)]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 32.0)
    monkeypatch.setattr(parallel, "estimate_peak_gb", lambda p, **k: 20.0)
    # usable = 32*0.85 = 27.2 ; 27.2 // 20 = 1
    plan = parallel.auto_worker_count(paths, cpu_cap=16)
    assert plan.workers == 1
    assert "memory" in plan.reason


def test_auto_worker_count_cpu_bound(monkeypatch, tmp_path):
    paths = [tmp_path / f"c{i}.nii.gz" for i in range(20)]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 256.0)
    monkeypatch.setattr(parallel, "estimate_peak_gb", lambda p, **k: 2.0)
    plan = parallel.auto_worker_count(paths, cpu_cap=4)
    assert plan.workers == 4  # capped by cpu, not memory or cases


def test_auto_worker_count_case_bound(monkeypatch, tmp_path):
    paths = [tmp_path / "c0.nii.gz", tmp_path / "c1.nii.gz"]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 256.0)
    monkeypatch.setattr(parallel, "estimate_peak_gb", lambda p, **k: 2.0)
    plan = parallel.auto_worker_count(paths, cpu_cap=32)
    assert plan.workers == 2  # only two cases


def test_explicit_workers_clamped_by_memory(monkeypatch, tmp_path):
    paths = [tmp_path / f"c{i}.nii.gz" for i in range(8)]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 30.0)
    monkeypatch.setattr(parallel, "estimate_peak_gb", lambda p, **k: 20.0)
    # user asked 8 but 30*0.85//20 = 1
    plan = parallel.auto_worker_count(paths, requested=8, cpu_cap=16)
    assert plan.workers == 1
    assert "clamped" in plan.reason


def test_threads_per_worker_divides_cpu(monkeypatch, tmp_path):
    paths = [tmp_path / f"c{i}.nii.gz" for i in range(4)]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 256.0)
    monkeypatch.setattr(parallel, "estimate_peak_gb", lambda p, **k: 1.0)
    plan = parallel.auto_worker_count(paths, cpu_cap=8)
    assert plan.workers >= 1
    assert plan.threads_per_worker == max(1, plan.cpu_count // plan.workers)


def test_peak_multiplier_calibrated_low(monkeypatch, tmp_path):
    """Post-optimisation the pipeline is fat-bound (~9× raw); the multiplier
    should reflect that so auto picks a sensible worker count."""
    assert 8.0 <= parallel.PEAK_MULTIPLIER <= 13.0
    paths = [tmp_path / f"c{i}.nii.gz" for i in range(16)]
    monkeypatch.setattr(parallel, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(parallel, "input_raw_bytes", lambda p: 1.0e9)  # 1 GB raw
    plan = parallel.auto_worker_count(paths, cpu_cap=32)
    # 64*0.85 = 54.4 usable / (1GB*11) ≈ 4 workers
    assert plan.workers >= 4


def test_apply_thread_limits_sets_env(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    parallel.apply_thread_limits(3)
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3"


# --- worker round-trip ------------------------------------------------------

def test_case_outcome_is_picklable(synth_nifti_path, tmp_path):
    from stroke_cta_osa.config import PipelineConfig
    from stroke_cta_osa.parallel import CaseJob, run_case
    job = CaseJob(input_path=str(synth_nifti_path), out_dir=str(tmp_path),
                  pid="synth", cfg=PipelineConfig())
    assert pickle.loads(pickle.dumps(job)).pid == "synth"
    outcome = run_case(job)
    assert outcome.error is None, outcome.error
    # the returned CaseResult must survive process boundaries
    assert pickle.loads(pickle.dumps(outcome)).pid == "synth"


# --- CLI batch with workers -------------------------------------------------

def _write_synth(arr, path: Path) -> None:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path), useCompression=True)


def test_batch_workers_two_matches_sequential(synth_array, tmp_path):
    indir = tmp_path / "cases"
    indir.mkdir()
    _write_synth(synth_array, indir / "caseA.nii.gz")
    _write_synth(synth_array, indir / "caseB.nii.gz")

    out_par = tmp_path / "par"
    result = runner.invoke(app, [
        "batch", str(indir), "--out", str(out_par),
        "--glob", "*.nii.gz", "--workers", "2",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    import pandas as pd
    df = pd.read_csv(out_par / "features.csv")
    assert len(df) == 2
    assert set(df["patient_id"]) == {"caseA.nii.gz", "caseB.nii.gz"}


def test_batch_workers_auto_and_one(synth_array, tmp_path):
    indir = tmp_path / "cases"
    indir.mkdir()
    _write_synth(synth_array, indir / "only.nii.gz")
    for w in ("auto", "1"):
        out = tmp_path / f"out_{w}"
        result = runner.invoke(app, [
            "batch", str(indir), "--out", str(out),
            "--glob", "*.nii.gz", "--workers", w,
        ])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert (out / "features.csv").is_file()


def test_case_job_carries_external_mask(synth_nifti_path, synth_airway_mask_path, tmp_path):
    """A CaseJob with a cached airway mask routes it into extract_case."""
    from stroke_cta_osa.config import PipelineConfig
    from stroke_cta_osa.parallel import CaseJob, run_case
    job = CaseJob(input_path=str(synth_nifti_path), out_dir=str(tmp_path),
                  pid="synth", cfg=PipelineConfig(),
                  external_airway_mask=str(synth_airway_mask_path))
    outcome = run_case(job)
    assert outcome.error is None, outcome.error
    assert outcome.result.identifiers.get("airway_source") == "external_mask"


def test_airway_precompute_worker(synth_nifti_path, tmp_path):
    from stroke_cta_osa.config import PipelineConfig
    from stroke_cta_osa.parallel import AirwayJob, run_airway_precompute
    cache = tmp_path / "cache" / "synth.airway.nii.gz"
    job = AirwayJob(input_path=str(synth_nifti_path), cache_path=str(cache),
                    pid="synth", cfg=PipelineConfig())
    outcome = run_airway_precompute(job)
    assert outcome.error is None, outcome.error
    # synthetic case has a patent column → mask is produced and cached
    if outcome.mask_path is not None:
        assert Path(outcome.mask_path).is_file()


def test_batch_precompute_airway(synth_array, tmp_path):
    indir = tmp_path / "cases"
    indir.mkdir()
    _write_synth(synth_array, indir / "caseA.nii.gz")
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "batch", str(indir), "--out", str(out),
        "--glob", "*.nii.gz", "--workers", "1", "--precompute-airway",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (out / "features.csv").is_file()
    assert (out / "_airway_cache").is_dir()


def test_batch_bad_workers(synth_array, tmp_path):
    indir = tmp_path / "cases"
    indir.mkdir()
    _write_synth(synth_array, indir / "only.nii.gz")
    result = runner.invoke(app, [
        "batch", str(indir), "--out", str(tmp_path / "o"),
        "--glob", "*.nii.gz", "--workers", "0",
    ])
    assert result.exit_code != 0
