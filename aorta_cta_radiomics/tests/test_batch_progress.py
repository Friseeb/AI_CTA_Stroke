from pathlib import Path

import pandas as pd

from aorta_cta_radiomics.batch_progress import compact_ntfy_summary, summarize_progress


def test_summarize_progress_reports_metadata_stage_counts_and_runner_tail(tmp_path: Path):
    pd.DataFrame(
        {
            "case_id": ["A", "B", "C"],
            "eligible": [True, False, True],
            "reason": ["eligible_neuro_cta", "excluded_non_neuro_protocol", "eligible_neuro_cta"],
        }
    ).to_csv(tmp_path / "metadata_eligibility.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["A", "B", "A"],
            "stage": ["base", "base", "calcium"],
            "detail": ["", "", ""],
            "status": ["ok", "failed", "ok"],
            "end_time_utc": ["2026-01-01T00:00:01", "2026-01-01T00:00:02", "2026-01-01T00:00:03"],
            "log_path": ["base/A.log", "base/B.log", "calcium/A.log"],
        }
    ).to_csv(tmp_path / "stage_status.csv", index=False)
    log_path = tmp_path / "logs" / "batch_watchdog" / "staged_runner.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("line one\nline two\nline three\n", encoding="utf-8")

    summary = summarize_progress(tmp_path, tail_lines=2)

    assert "metadata: kept 2/3" in summary
    assert "base: 2/2" in summary
    assert "ok=1" in summary
    assert "failed=1" in summary
    assert "calcium: 1/2" in summary
    assert "failures:" in summary
    assert "base/B.log" in summary
    assert "line two" in summary
    assert "line three" in summary

    ntfy_summary = compact_ntfy_summary(tmp_path)

    assert "current:" in ntfy_summary
    assert "ETA" in ntfy_summary
    assert "processes:" in ntfy_summary
    assert "artifacts:" in ntfy_summary
    assert "failures: B:base" in ntfy_summary
