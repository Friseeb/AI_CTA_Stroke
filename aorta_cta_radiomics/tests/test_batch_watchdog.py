from argparse import Namespace
from pathlib import Path

import pandas as pd

from aorta_cta_radiomics.batch_watchdog import (
    build_staged_command,
    generate_ntfy_topic,
    ntfy_endpoint,
    resolve_ntfy_config,
    summarize_run,
    validate_inputs,
)


def test_build_staged_command_includes_neuro_cta_filter_and_watchdog_defaults(tmp_path: Path):
    args = Namespace(
        python="/env/bin/python",
        staged_script=tmp_path / "run_manifest_staged.py",
        manifest=tmp_path / "manifest.csv",
        outdir=tmp_path / "out",
        stages="vista,base,calcium",
        config=tmp_path / "config.yaml",
        metadata_filter="neuro-cta",
        vista_workers=1,
        base_workers=2,
        calcium_workers=2,
        fat_wall_workers=1,
        protrusion_workers=1,
        wall_thickness_workers=2,
        radiomics_workers=1,
        nv_python="/envs/nv/bin/python",
        nv_device="mps",
        no_skip_existing=False,
        keep_going=True,
        dry_run=False,
        allow_missing_metadata=False,
        radiomics_split_by_region=True,
        radiomics_region_workers=4,
        metadata_include_keyword=["hyperacute"],
        metadata_exclude_keyword=["coronary"],
    )

    command = build_staged_command(args)

    assert command[:2] == ["/env/bin/python", str(tmp_path / "run_manifest_staged.py")]
    assert "--metadata-filter" in command
    assert "neuro-cta" in command
    assert "--skip-existing" in command
    assert "--keep-going" in command
    assert "--radiomics-split-by-region" in command
    assert "--metadata-include-keyword" in command
    assert "--metadata-exclude-keyword" in command
    assert "mps" in command


def test_summarize_run_reports_metadata_and_stage_counts(tmp_path: Path):
    pd.DataFrame(
        {
            "case_id": ["A", "B", "C"],
            "eligible": [True, False, True],
            "reason": ["eligible_neuro_cta", "missing_brain_neck_stroke_term", "eligible_neuro_cta"],
        }
    ).to_csv(tmp_path / "metadata_eligibility.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["A", "B", "A"],
            "stage": ["base", "base", "calcium"],
            "status": ["ok", "failed", "ok"],
        }
    ).to_csv(tmp_path / "stage_status.csv", index=False)

    summary = summarize_run(tmp_path)

    assert "metadata: kept 2/3" in summary
    assert "current:" in summary
    assert "ETA" in summary
    assert "processes:" in summary
    assert "artifacts:" in summary
    assert "base 2/2" in summary
    assert "calcium 1/2" in summary
    assert "failures: B:base" in summary


def test_ntfy_endpoint_accepts_topic_or_full_url():
    assert ntfy_endpoint("https://ntfy.sh", "my-topic") == "https://ntfy.sh/my-topic"
    assert ntfy_endpoint("https://ntfy.sh/", "/my-topic") == "https://ntfy.sh/my-topic"
    assert ntfy_endpoint("https://ntfy.sh", "https://example.org/custom") == "https://example.org/custom"


def test_generate_ntfy_topic_uses_run_label_when_provided(tmp_path: Path):
    topic = generate_ntfy_topic(
        prefix="aorta cta",
        manifest=tmp_path / "manifest.csv",
        outdir=tmp_path / "aorta_batch_run",
        run_label="SLAOBIDS full cohort",
    )

    assert topic == "aorta-cta-slaobids-full-cohort"


def test_generate_ntfy_topic_uses_manifest_and_outdir_without_randomness(tmp_path: Path):
    manifest = tmp_path / "slaobids" / "manifest.csv"

    topic_1 = generate_ntfy_topic(prefix="aorta-cta", manifest=manifest, outdir=tmp_path / "aorta_batch_run")
    topic_2 = generate_ntfy_topic(prefix="aorta-cta", manifest=manifest, outdir=tmp_path / "aorta_batch_run")

    assert topic_1 == "aorta-cta-slaobids-manifest-aorta-batch-run"
    assert topic_1 == topic_2


def test_resolve_ntfy_config_auto_writes_topic_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    args = Namespace(
        ntfy_topic="auto",
        ntfy_topic_prefix="aorta-cta",
        ntfy_url="https://ntfy.sh",
        ntfy_token="",
        ntfy_priority="3",
        manifest=tmp_path / "manifest.csv",
        run_label="demo run",
    )

    config = resolve_ntfy_config(args, tmp_path / "out")

    assert config.topic == "aorta-cta-demo-run"
    assert (tmp_path / "out" / "ntfy_topic.txt").read_text(encoding="utf-8").splitlines() == [
        "aorta-cta-demo-run",
        "https://ntfy.sh/aorta-cta-demo-run",
    ]


def test_validate_inputs_fails_before_start_for_missing_manifest(tmp_path: Path):
    args = Namespace(
        manifest=tmp_path / "missing_manifest.csv",
        config=tmp_path / "config.yaml",
        staged_script=tmp_path / "run_manifest_staged.py",
        python="python",
    )
    args.config.touch()
    args.staged_script.touch()

    try:
        validate_inputs(args)
    except SystemExit as exc:
        assert "missing_manifest.csv" in str(exc)
    else:
        raise AssertionError("validate_inputs should fail for a missing manifest")
