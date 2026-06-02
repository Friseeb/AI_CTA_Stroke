"""CLI smoke tests using typer's CliRunner.

These do not check feature *values* — they only check that subcommands run,
write the expected files, and exit with code 0 on a clean synthetic case.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stroke_cta_osa.cli import app


runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stroke-cta-osa" in result.stdout.lower() or "Usage" in result.stdout


def test_extract_subcommand(synth_nifti_path, tmp_path):
    result = runner.invoke(app, [
        "extract", str(synth_nifti_path),
        "--out", str(tmp_path),
        "--patient-id", "synth_cli",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    feat = pd.read_csv(tmp_path / "features.csv")
    qc = pd.read_csv(tmp_path / "qc.csv")
    assert len(feat) == 1
    assert len(qc) == 1
    assert feat.loc[0, "patient_id"] == "synth_cli"


def test_qc_subcommand(synth_nifti_path, tmp_path):
    result = runner.invoke(app, [
        "qc", str(synth_nifti_path), "--out", str(tmp_path),
    ])
    # Synthetic image easily passes QC
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "qc.csv").exists()


def test_batch_subcommand(synth_nifti_path, tmp_path):
    # Build a tiny directory of two copies of the synthetic NIfTI
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for i in (1, 2):
        p = case_dir / f"synth_{i}.nii.gz"
        p.write_bytes(synth_nifti_path.read_bytes())
    out_dir = tmp_path / "out"

    result = runner.invoke(app, [
        "batch", str(case_dir), "--out", str(out_dir),
        "--glob", "*.nii.gz",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    feat = pd.read_csv(out_dir / "features.csv")
    assert len(feat) == 2


def test_summarize_subcommand(synth_nifti_path, tmp_path):
    # Build a tiny features.csv via the extract command
    runner.invoke(app, [
        "extract", str(synth_nifti_path),
        "--out", str(tmp_path), "--patient-id", "sum_001",
    ])
    feats = tmp_path / "features.csv"
    assert feats.exists()
    result = runner.invoke(app, ["summarize", str(feats)])
    assert result.exit_code == 0
    # rich.Console(stderr=True) writes to stderr; check combined output
    assert "Rows" in (result.output or "") or "Rows" in (result.stderr or "")
