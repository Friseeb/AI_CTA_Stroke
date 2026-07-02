"""CLI: list-features evidence export + --feature-set selection."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stroke_cta_osa.cli import app

runner = CliRunner()


def test_list_features_exports_evidence_metadata(tmp_path):
    out = tmp_path / "dict.csv"
    result = runner.invoke(app, ["list-features", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    df = pd.read_csv(out)
    assert "evidence_tier" in df.columns
    assert "evidence_class" in df.columns
    assert "reference_tags" in df.columns


def test_list_features_feature_set_filter(tmp_path):
    out = tmp_path / "core.csv"
    result = runner.invoke(app, [
        "list-features", "--feature-set", "core_osa_backed", "--out", str(out),
    ])
    assert result.exit_code == 0, result.stdout
    df = pd.read_csv(out)
    assert (df["evidence_tier"] == "TIER_1_CORE_OSA_BACKED").all()


def test_list_features_evidence_tier_filter(tmp_path):
    out = tmp_path / "t3.json"
    result = runner.invoke(app, [
        "list-features", "--evidence-tier",
        "TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR", "--out", str(out),
    ])
    assert result.exit_code == 0, result.stdout
    recs = json.loads(out.read_text())
    assert recs
    assert all(r["evidence_tier"] == "TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR"
               for r in recs)


def test_list_features_bad_feature_set():
    result = runner.invoke(app, ["list-features", "--feature-set", "nope"])
    assert result.exit_code != 0


def test_extract_with_feature_set(synth_nifti_path, tmp_path):
    result = runner.invoke(app, [
        "extract", str(synth_nifti_path), "--out", str(tmp_path),
        "--patient-id", "synth", "--feature-set", "core_osa_backed",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    # all tiered subset CSVs are written regardless of the chosen set
    for f in ("features_core_osa_backed.csv",
              "features_core_plus_anatomic_extensions.csv",
              "features_all_exploratory.csv"):
        assert (tmp_path / f).is_file(), f


def test_extract_bad_feature_set(synth_nifti_path, tmp_path):
    result = runner.invoke(app, [
        "extract", str(synth_nifti_path), "--out", str(tmp_path),
        "--feature-set", "bogus",
    ])
    assert result.exit_code != 0


def test_summarize_by_evidence_tier(synth_nifti_path, tmp_path):
    runner.invoke(app, ["extract", str(synth_nifti_path), "--out", str(tmp_path),
                        "--patient-id", "synth"])
    result = runner.invoke(app, [
        "summarize", str(tmp_path / "features.csv"), "--by-evidence-tier",
    ])
    assert result.exit_code == 0, result.stdout
    # summarize prints to the stderr console; combine both streams.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "TIER_1_CORE_OSA_BACKED" in combined
