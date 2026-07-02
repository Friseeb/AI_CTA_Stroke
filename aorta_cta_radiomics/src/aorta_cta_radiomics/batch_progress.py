"""Read-only monitor for staged aorta CTA batch runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import socketserver
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from statistics import median
from urllib.parse import urlparse

import pandas as pd


STAGE_ORDER = [
    "vista",
    "base",
    "calcium",
    "fat-wall",
    "protrusions",
    "wall-thickness",
    "radiomics",
    "radiomics-region",
    "qc",
]


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    cpu_percent: float
    memory_percent: float
    rss_mb: float
    elapsed: str
    command: str


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir).resolve()
    if args.serve:
        serve_dashboard(
            outdir=outdir,
            host=args.host,
            port=args.port,
            refresh_seconds=args.interval_seconds,
            tail_lines=args.tail_lines,
            failure_limit=args.failure_limit,
            open_browser=not args.no_open_browser,
        )
        return
    if args.watch:
        while True:
            if os.isatty(1):
                print("\033[2J\033[H", end="")
            print(summarize_progress(outdir, tail_lines=args.tail_lines, failure_limit=args.failure_limit), flush=True)
            time.sleep(max(float(args.interval_seconds), 2.0))
        return
    print(summarize_progress(outdir, tail_lines=args.tail_lines, failure_limit=args.failure_limit))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--watch", action="store_true", help="Refresh the terminal progress view until interrupted.")
    parser.add_argument("--serve", action="store_true", help="Serve a local browser dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--tail-lines", type=int, default=8)
    parser.add_argument("--failure-limit", type=int, default=8)
    return parser


def summarize_progress(outdir: Path, tail_lines: int = 8, failure_limit: int = 8) -> str:
    payload = progress_payload(outdir, tail_lines=tail_lines, failure_limit=failure_limit)
    lines = [
        f"outdir: {payload['outdir']}",
        f"generated: {payload['generated_at']}",
    ]
    lines.extend(_metadata_text(payload["metadata"]))
    lines.extend(_artifact_text(payload["artifacts"]))
    lines.extend(_stage_text(payload["stage"]))
    lines.extend(_process_text(payload["processes"]))
    lines.append(
        f"latest activity: {payload['latest_activity'] or 'none'} "
        f"({_format_minutes(payload['latest_activity_age_minutes'])})"
    )
    if payload["runner_tail"]:
        lines.append("runner log tail:")
        lines.extend(f"  {line}" for line in payload["runner_tail"])
    else:
        lines.append("runner log tail: not written yet")
    return "\n".join(lines)


def serve_dashboard(
    outdir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    refresh_seconds: float = 30.0,
    tail_lines: int = 8,
    failure_limit: int = 8,
    open_browser: bool = True,
) -> None:
    """Serve a dependency-free local HTML dashboard for a running batch."""

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/data.json":
                payload = json.dumps(progress_payload(outdir, tail_lines, failure_limit), indent=2).encode("utf-8")
                self._send(payload, "application/json; charset=utf-8")
                return
            if route in {"/", "/index.html"}:
                payload = render_dashboard_html(outdir, refresh_seconds, tail_lines, failure_limit).encode("utf-8")
                self._send(payload, "text/html; charset=utf-8")
                return
            self.send_error(404)

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((host, int(port)), DashboardHandler) as server:
        url = f"http://{host}:{port}/"
        print(f"Aorta batch dashboard: {url}")
        print(f"outdir: {outdir}")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


def progress_payload(outdir: Path, tail_lines: int = 8, failure_limit: int = 8) -> dict[str, object]:
    metadata = _read_metadata_summary(outdir / "metadata_eligibility.csv")
    artifacts = _artifact_summary(outdir)
    processes = _process_summary(outdir, exclude_self_tree=True)
    stage = _read_stage_summary(
        outdir / "stage_status.csv",
        eligible_cases=int(metadata.get("eligible", 0) or 0),
        artifacts=artifacts,
        processes=processes,
        failure_limit=failure_limit,
    )
    latest_activity = _latest_activity_time(outdir)
    latest_activity_age_minutes = max((time.time() - latest_activity) / 60.0, 0.0) if latest_activity else None
    return {
        "outdir": str(outdir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "artifacts": artifacts,
        "stage": stage,
        "processes": processes,
        "latest_activity": datetime.fromtimestamp(latest_activity).isoformat(timespec="seconds") if latest_activity else "",
        "latest_activity_age_minutes": latest_activity_age_minutes,
        "runner_tail": _read_tail(outdir / "logs" / "batch_watchdog" / "staged_runner.log", tail_lines=tail_lines),
    }


def compact_ntfy_summary(outdir: str | Path, failure_limit: int = 5) -> str:
    """Return a phone-readable progress summary for ntfy notifications."""
    outdir = Path(outdir)
    metadata = _read_metadata_summary(outdir / "metadata_eligibility.csv")
    artifacts = _artifact_summary(outdir)
    processes = _process_summary(outdir, exclude_self_tree=False)
    stage = _read_stage_summary(
        outdir / "stage_status.csv",
        eligible_cases=int(metadata.get("eligible", 0) or 0),
        artifacts=artifacts,
        processes=processes,
        failure_limit=failure_limit,
    )
    latest_activity = _latest_activity_time(outdir)
    latest_age = max((time.time() - latest_activity) / 60.0, 0.0) if latest_activity else None
    lines = [
        f"outdir: {outdir}",
        f"metadata: kept {metadata.get('eligible', 0)}/{metadata.get('rows', 0)}",
        f"current: {stage.get('current_stage', 'unknown')} ETA {_format_seconds(stage.get('current_stage_eta_seconds'))}",
        (
            f"processes: {processes.get('count', 0)} | "
            f"CPU {processes.get('cpu_core_percent', 0)}% core-equivalent "
            f"({processes.get('cpu_machine_percent', 0)}% of machine) | "
            f"RSS {processes.get('rss_mb', 0)} MB"
        ),
        f"latest activity: {_format_minutes(latest_age)}",
        "artifacts: "
        f"vista={artifacts.get('vista_aorta_masks', 0)}, "
        f"base={artifacts.get('cleaned_aorta_masks', 0)}, "
        f"calcium={artifacts.get('calcium_masks', 0)}, "
        f"fat-wall={artifacts.get('fat_wall_masks', 0)}, "
        f"thickness={artifacts.get('wall_thickness_maps', 0)}, "
        f"radiomics={artifacts.get('radiomics_csvs', 0)}",
    ]
    stages = stage.get("stages", [])
    if isinstance(stages, list) and stages:
        stage_parts = []
        for item in stages:
            if isinstance(item, dict):
                stage_parts.append(
                    f"{item.get('stage')} {item.get('done', 0)}/{item.get('total_expected', 0)}"
                    f" ({float(item.get('percent') or 0):.0f}%)"
                )
        lines.append("stages: " + "; ".join(stage_parts))
    failures = stage.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.append("failures: " + ", ".join(f"{row.get('case_id')}:{row.get('stage')}" for row in failures if isinstance(row, dict)))
    return "\n".join(lines)


def render_dashboard_html(
    outdir: Path,
    refresh_seconds: float = 30.0,
    tail_lines: int = 8,
    failure_limit: int = 8,
) -> str:
    payload = progress_payload(outdir, tail_lines=tail_lines, failure_limit=failure_limit)
    metadata = payload["metadata"]
    artifacts = payload["artifacts"]
    stage = payload["stage"]
    processes = payload["processes"]
    latest_age = _format_minutes(payload["latest_activity_age_minutes"])
    refresh = max(int(refresh_seconds), 5)
    current_stage = stage.get("current_stage") or "unknown"
    eta_text = _format_seconds(stage.get("current_stage_eta_seconds"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{refresh}">
  <title>Aorta CTA Batch Monitor</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5b6876;
      --line: #d8e0e8;
      --ok: #0f7b4f;
      --fail: #b42318;
      --accent: #1f5ea8;
      --warn: #925400;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 22px 28px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 650; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    main {{ padding: 20px 28px 34px; display: grid; gap: 18px; }}
    .path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .value {{
      margin-top: 4px;
      font-size: 23px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .subvalue {{ margin-top: 4px; font-size: 12px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      text-align: left;
      padding: 8px 7px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .ok {{ color: var(--ok); font-weight: 650; }}
    .failed, .fail {{ color: var(--fail); font-weight: 650; }}
    .warn {{ color: var(--warn); font-weight: 650; }}
    .muted {{ color: var(--muted); }}
    .progress {{
      height: 9px;
      background: #e9eef4;
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }}
    .bar {{ height: 100%; background: var(--accent); }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Aorta CTA Batch Monitor</h1>
    <div class="path">{_esc(str(payload["outdir"]))}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Metadata Kept</div><div class="value">{_esc(_metadata_value(metadata))}</div></div>
      <div class="card"><div class="label">Current Stage</div><div class="value">{_esc(str(current_stage))}</div><div class="subvalue">ETA {eta_text}</div></div>
      <div class="card"><div class="label">Processes</div><div class="value">{_esc(str(processes["count"]))}</div><div class="subvalue">{_esc(str(processes["rss_mb"]))} MB RSS, {_esc(str(processes["cpu_core_percent"]))}% CPU cores, {_esc(str(processes["cpu_machine_percent"]))}% machine</div></div>
      <div class="card"><div class="label">VISTA Masks</div><div class="value">{_esc(str(artifacts["vista_aorta_masks"]))}</div></div>
      <div class="card"><div class="label">Latest Activity</div><div class="value">{_esc(latest_age)}</div><div class="subvalue">{_esc(str(payload["latest_activity"] or "none"))}</div></div>
    </div>
    <section>
      <h2>Stage Progress and ETA</h2>
      {_stage_table(stage)}
    </section>
    <section>
      <h2>Running Processes</h2>
      {_process_table(processes)}
    </section>
    <section>
      <h2>Artifact Counts</h2>
      {_counts_table(artifacts)}
    </section>
    <section>
      <h2>Metadata Reasons</h2>
      {_counts_table(metadata.get("reason_counts", {}))}
    </section>
    <section>
      <h2>Latest Completed</h2>
      {_simple_list(stage.get("recent", []))}
    </section>
    <section>
      <h2>Failures</h2>
      {_failure_table(stage.get("failures", []))}
    </section>
    <section>
      <h2>Runner Log Tail</h2>
      <pre>{_esc(chr(10).join(str(line) for line in payload["runner_tail"]))}</pre>
    </section>
  </main>
</body>
</html>
"""


def _read_metadata_summary(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {"rows": 0, "eligible": 0, "reason_counts": {}, "status": "not written"}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return {"rows": 0, "eligible": 0, "reason_counts": {}, "status": "unreadable"}
    eligible = int(frame["eligible"].astype(bool).sum()) if "eligible" in frame.columns else 0
    reasons = frame["reason"].fillna("").value_counts().to_dict() if "reason" in frame.columns else {}
    return {"rows": int(len(frame)), "eligible": eligible, "reason_counts": reasons, "status": "ok"}


def _read_stage_summary(
    path: Path,
    eligible_cases: int,
    artifacts: dict[str, int],
    processes: dict[str, object],
    failure_limit: int = 8,
) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        stages = _artifact_only_stage_estimates(eligible_cases, artifacts)
        return {
            "rows": 0,
            "cases_with_status": 0,
            "stages": stages,
            "current_stage": _current_stage_from_artifacts(stages, processes),
            "current_stage_eta_seconds": None,
            "recent": [],
            "failures": [],
            "status": "not written",
        }
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return {
            "rows": 0,
            "cases_with_status": 0,
            "stages": [],
            "current_stage": "unreadable",
            "current_stage_eta_seconds": None,
            "recent": [],
            "failures": [],
            "status": "unreadable",
        }
    if frame.empty:
        return {
            "rows": 0,
            "cases_with_status": 0,
            "stages": _artifact_only_stage_estimates(eligible_cases, artifacts),
            "current_stage": "empty",
            "current_stage_eta_seconds": None,
            "recent": [],
            "failures": [],
            "status": "empty",
        }

    stages = _stage_estimates_from_status(frame, eligible_cases)
    current = _current_stage_from_status(stages, processes)
    current_eta = None
    for item in stages:
        if item["stage"] == current:
            current_eta = item.get("eta_seconds")
            break
    failures = []
    if {"case_id", "stage", "status"}.issubset(frame.columns):
        failed = frame[frame["status"] != "ok"].tail(max(int(failure_limit), 1))
        for row in failed.itertuples(index=False):
            failures.append(
                {
                    "case_id": str(getattr(row, "case_id")),
                    "stage": str(getattr(row, "stage")),
                    "status": str(getattr(row, "status")),
                    "log_path": str(getattr(row, "log_path", "")),
                }
            )
    return {
        "rows": int(len(frame)),
        "cases_with_status": int(frame["case_id"].nunique()) if "case_id" in frame.columns else 0,
        "stages": stages,
        "current_stage": current,
        "current_stage_eta_seconds": current_eta,
        "recent": _recent_stage_dicts(frame),
        "failures": failures,
        "status": "ok",
    }


def _stage_estimates_from_status(frame: pd.DataFrame, eligible_cases: int) -> list[dict[str, object]]:
    if "stage" not in frame.columns:
        return []
    stages: list[dict[str, object]] = []
    ordered_names = sorted(frame["stage"].astype(str).unique(), key=_stage_sort_key)
    for stage in ordered_names:
        stage_frame = frame[frame["stage"].astype(str) == stage].copy()
        done = int(len(stage_frame))
        counts = stage_frame["status"].fillna("").value_counts().to_dict() if "status" in stage_frame.columns else {}
        total_expected = _expected_total(stage, stage_frame, eligible_cases)
        remaining = max(total_expected - done, 0) if total_expected else 0
        durations = _durations_seconds(stage_frame)
        mean_seconds = sum(durations) / len(durations) if durations else None
        median_seconds = median(durations) if durations else None
        eta_seconds = _stage_eta_seconds(stage_frame, total_expected, done, mean_seconds)
        stages.append(
            {
                "stage": stage,
                "done": done,
                "total_expected": total_expected,
                "remaining": remaining,
                "percent": _percent(done, total_expected),
                "counts": counts,
                "mean_seconds": mean_seconds,
                "median_seconds": median_seconds,
                "eta_seconds": eta_seconds,
                "first_start": _min_datetime_text(stage_frame, "start_time_utc"),
                "last_end": _max_datetime_text(stage_frame, "end_time_utc"),
            }
        )
    return stages


def _artifact_only_stage_estimates(eligible_cases: int, artifacts: dict[str, int]) -> list[dict[str, object]]:
    specs = [
        ("vista", "vista_aorta_masks"),
        ("base", "cleaned_aorta_masks"),
        ("calcium", "calcium_masks"),
        ("fat-wall", "fat_wall_masks"),
        ("protrusions", "protrusion_csvs"),
        ("wall-thickness", "wall_thickness_maps"),
        ("radiomics", "radiomics_csvs"),
    ]
    stages = []
    for stage, key in specs:
        done = int(artifacts.get(key, 0))
        total = eligible_cases or max(done, 0)
        stages.append(
            {
                "stage": stage,
                "done": done,
                "total_expected": total,
                "remaining": max(total - done, 0) if total else 0,
                "percent": _percent(done, total),
                "counts": {"artifact": done},
                "mean_seconds": None,
                "median_seconds": None,
                "eta_seconds": None,
                "first_start": "",
                "last_end": "",
            }
        )
    return stages


def _expected_total(stage: str, stage_frame: pd.DataFrame, eligible_cases: int) -> int:
    if stage == "radiomics-region":
        observed_regions = stage_frame["detail"].dropna().astype(str).replace("", pd.NA).dropna().nunique()
        if observed_regions and eligible_cases:
            return int(observed_regions * eligible_cases)
    if eligible_cases:
        return int(eligible_cases)
    return int(len(stage_frame))


def _stage_eta_seconds(
    stage_frame: pd.DataFrame,
    total_expected: int,
    done: int,
    mean_seconds: float | None,
) -> float | None:
    if not total_expected or done >= total_expected:
        return 0.0
    remaining = max(total_expected - done, 0)
    first_start = _parse_datetime(_min_datetime_text(stage_frame, "start_time_utc"))
    if first_start and done:
        elapsed = max((datetime.now(timezone.utc) - first_start).total_seconds(), 1.0)
        return elapsed / done * remaining
    if mean_seconds is not None:
        return mean_seconds * remaining
    return None


def _durations_seconds(frame: pd.DataFrame) -> list[float]:
    if not {"start_time_utc", "end_time_utc"}.issubset(frame.columns):
        return []
    durations = []
    for row in frame.itertuples(index=False):
        start = _parse_datetime(str(getattr(row, "start_time_utc", "")))
        end = _parse_datetime(str(getattr(row, "end_time_utc", "")))
        if start and end:
            durations.append(max((end - start).total_seconds(), 0.0))
    return durations


def _current_stage_from_status(stages: list[dict[str, object]], processes: dict[str, object]) -> str:
    running_stage = _stage_from_processes(processes)
    if running_stage:
        return running_stage
    for stage in sorted(stages, key=lambda item: _stage_sort_key(str(item["stage"]))):
        total = int(stage.get("total_expected", 0) or 0)
        done = int(stage.get("done", 0) or 0)
        if total and done < total:
            return str(stage["stage"])
    if stages:
        return "complete"
    return "unknown"


def _current_stage_from_artifacts(stages: list[dict[str, object]], processes: dict[str, object]) -> str:
    running_stage = _stage_from_processes(processes)
    if running_stage:
        return running_stage
    for stage in stages:
        total = int(stage.get("total_expected", 0) or 0)
        done = int(stage.get("done", 0) or 0)
        if total and done < total:
            return str(stage["stage"])
    return "unknown"


def _stage_from_processes(processes: dict[str, object]) -> str:
    commands = " ".join(str(row.get("command", "")) for row in processes.get("rows", []))
    stage_markers = [
        ("vista", "run_nv_segment_ct_laa.py"),
        ("base", "run_base_case.py"),
        ("calcium", "run_calcium_case.py"),
        ("fat-wall", "run_fat_wall_case.py"),
        ("protrusions", "run_protrusions_case.py"),
        ("wall-thickness", "measure_wall_thickness.py"),
        ("radiomics", "run_radiomics_case.py"),
        ("staged", "run_manifest_staged.py"),
    ]
    for stage, marker in stage_markers:
        if marker in commands:
            return stage
    return ""


def _artifact_summary(outdir: Path) -> dict[str, int]:
    return {
        "vista_aorta_masks": _count_files(outdir / "vista_aorta", "*_aorta6.nii.gz"),
        "case_dirs": _count_dirs(outdir / "cases"),
        "cleaned_aorta_masks": _count_files(outdir / "cases", "*_aorta_mask_cleaned.nii.gz"),
        "calcium_masks": _count_files(outdir / "cases", "*_calcification_aorta_wall_dynamic_seed500HU.nii.gz"),
        "fat_wall_masks": _count_files(outdir / "cases", "*_aortic_wall_candidate_from_fat_lumen.nii.gz"),
        "protrusion_csvs": _count_files(outdir / "cases", "lumen_protrusion_candidates.csv"),
        "wall_thickness_maps": _count_files(outdir / "cases", "*_wall_thickness_gt_*mm_TEE_analogue_labels.nii.gz"),
        "radiomics_csvs": _count_files(outdir / "cases", "radiomics_features.csv"),
    }


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def _count_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.iterdir() if path.is_dir())


def _process_summary(outdir: Path, exclude_self_tree: bool = True) -> dict[str, object]:
    rows = _matching_processes(outdir, exclude_self_tree=exclude_self_tree)
    total_rss = round(sum(row.rss_mb for row in rows), 1)
    total_cpu = round(sum(row.cpu_percent for row in rows), 1)
    logical_cpus = max(int(os.cpu_count() or 1), 1)
    machine_cpu = round(total_cpu / logical_cpus, 1)
    return {
        "count": len(rows),
        "rss_mb": total_rss,
        "cpu_percent": total_cpu,
        "cpu_core_percent": total_cpu,
        "cpu_machine_percent": machine_cpu,
        "logical_cpus": logical_cpus,
        "rows": [
            {
                "pid": row.pid,
                "ppid": row.ppid,
                "cpu_percent": row.cpu_percent,
                "cpu_machine_percent": round(row.cpu_percent / logical_cpus, 1),
                "memory_percent": row.memory_percent,
                "rss_mb": row.rss_mb,
                "elapsed": row.elapsed,
                "command": row.command,
            }
            for row in rows
        ],
    }


def _matching_processes(outdir: Path, exclude_self_tree: bool = True) -> list[ProcessInfo]:
    all_processes = _read_processes()
    outdir_text = str(outdir)
    markers = [
        outdir_text,
        "run_batch_with_watchdog.py",
        "run_manifest_staged.py",
        "run_nv_segment_ct_laa.py",
        "run_base_case.py",
        "run_calcium_case.py",
        "run_fat_wall_case.py",
        "run_protrusions_case.py",
        "measure_wall_thickness.py",
        "run_radiomics_case.py",
    ]
    monitor_markers = [
        "aorta_cta_radiomics.batch_progress",
        "batch_progress.py",
        "batch_progress --outdir",
    ]
    seed_pids = {
        process.pid
        for process in all_processes
        if any(marker in process.command for marker in markers)
        and not any(marker in process.command for marker in monitor_markers)
    }
    selected = set(seed_pids)
    changed = True
    while changed:
        changed = False
        for process in all_processes:
            if process.ppid in selected and process.pid not in selected:
                selected.add(process.pid)
                changed = True
    if exclude_self_tree:
        selected -= _descendant_pids(all_processes, os.getpid())
    return [process for process in all_processes if process.pid in selected and process.pid != os.getpid()]


def _descendant_pids(processes: list[ProcessInfo], root_pid: int) -> set[int]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for process in processes:
            if process.ppid in selected and process.pid not in selected:
                selected.add(process.pid)
                changed = True
    return selected


def _read_processes() -> list[ProcessInfo]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,%cpu=,%mem=,rss=,etime=,command="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    processes: list[ProcessInfo] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu = float(parts[2])
            mem = float(parts[3])
            rss_mb = round(float(parts[4]) / 1024.0, 1)
        except ValueError:
            continue
        processes.append(
            ProcessInfo(
                pid=pid,
                ppid=ppid,
                cpu_percent=cpu,
                memory_percent=mem,
                rss_mb=rss_mb,
                elapsed=parts[5],
                command=parts[6],
            )
        )
    return processes


def _metadata_text(metadata: object) -> list[str]:
    data = dict(metadata) if isinstance(metadata, dict) else {}
    lines = [f"metadata: kept {data.get('eligible', 0)}/{data.get('rows', 0)} ({data.get('status', 'unknown')})"]
    counts = data.get("reason_counts", {})
    if isinstance(counts, dict) and counts:
        lines.append("metadata reasons: " + _format_counts(counts))
    return lines


def _artifact_text(artifacts: object) -> list[str]:
    data = dict(artifacts) if isinstance(artifacts, dict) else {}
    return ["artifacts: " + _format_counts(data)]


def _stage_text(stage: object) -> list[str]:
    data = dict(stage) if isinstance(stage, dict) else {}
    lines = [
        f"stage rows: {data.get('rows', 0)} ({data.get('status', 'unknown')})",
        f"current stage: {data.get('current_stage', 'unknown')} ETA {_format_seconds(data.get('current_stage_eta_seconds'))}",
    ]
    stages = data.get("stages", [])
    if isinstance(stages, list) and stages:
        lines.append("stage progress:")
        for item in stages:
            if not isinstance(item, dict):
                continue
            count_text = _format_counts(item.get("counts", {}))
            lines.append(
                "  "
                f"{item.get('stage')}: {item.get('done', 0)}/{item.get('total_expected', 0)} "
                f"({float(item.get('percent') or 0):.1f}%) "
                f"ETA {_format_seconds(item.get('eta_seconds'))} "
                f"mean {_format_seconds(item.get('mean_seconds'))} "
                f"{count_text}"
            )
    recent = data.get("recent", [])
    if isinstance(recent, list) and recent:
        lines.append("latest completed:")
        for row in recent[-5:]:
            if isinstance(row, dict):
                detail = f" {row.get('detail')}" if row.get("detail") else ""
                lines.append(
                    f"  {row.get('stage')}{detail} {row.get('case_id')}: "
                    f"{row.get('status')} at {row.get('end_time_utc')}"
                )
    failures = data.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.append("failures:")
        for row in failures:
            if isinstance(row, dict):
                lines.append(f"  {row.get('case_id')} {row.get('stage')}: {row.get('log_path')}")
    return lines


def _process_text(processes: object) -> list[str]:
    data = dict(processes) if isinstance(processes, dict) else {}
    lines = [
        f"processes: {data.get('count', 0)} running, "
        f"{data.get('rss_mb', 0)} MB RSS, "
        f"{data.get('cpu_core_percent', data.get('cpu_percent', 0))}% CPU core-equivalent, "
        f"{data.get('cpu_machine_percent', 0)}% estimated machine CPU "
        f"({data.get('logical_cpus', '?')} logical CPUs)"
    ]
    rows = data.get("rows", [])
    if isinstance(rows, list) and rows:
        lines.append("process table:")
        for row in rows[:12]:
            if isinstance(row, dict):
                lines.append(
                    f"  pid={row.get('pid')} cpu={row.get('cpu_percent')}% cores "
                    f"({row.get('cpu_machine_percent', 0)}% machine) "
                    f"rss={row.get('rss_mb')}MB elapsed={row.get('elapsed')} "
                    f"{_short_command(str(row.get('command', '')))}"
                )
    return lines


def _stage_table(stage: object) -> str:
    data = dict(stage) if isinstance(stage, dict) else {}
    stages = data.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return '<div class="muted">No stage rows yet.</div>'
    rows = []
    for item in stages:
        if not isinstance(item, dict):
            continue
        percent = float(item.get("percent") or 0.0)
        rows.append(
            "<tr>"
            f"<td>{_esc(str(item.get('stage', '')))}</td>"
            f"<td>{_esc(str(item.get('done', 0)))}/{_esc(str(item.get('total_expected', 0)))}</td>"
            f"<td><div class=\"progress\"><div class=\"bar\" style=\"width:{percent:.1f}%\"></div></div></td>"
            f"<td>{percent:.1f}%</td>"
            f"<td>{_esc(_format_seconds(item.get('eta_seconds')))}</td>"
            f"<td>{_esc(_format_seconds(item.get('mean_seconds')))}</td>"
            f"<td>{_esc(_format_counts(item.get('counts', {})))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>Done</th><th>Progress</th><th>%</th>"
        "<th>ETA</th><th>Mean/task</th><th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _process_table(processes: object) -> str:
    data = dict(processes) if isinstance(processes, dict) else {}
    rows = data.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return '<div class="muted">No matching running batch processes.</div>'
    html_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        html_rows.append(
            "<tr>"
            f"<td>{_esc(str(row.get('pid')))}</td>"
            f"<td>{_esc(str(row.get('cpu_percent')))}%</td>"
            f"<td>{_esc(str(row.get('cpu_machine_percent', 0)))}%</td>"
            f"<td>{_esc(str(row.get('rss_mb')))} MB</td>"
            f"<td>{_esc(str(row.get('elapsed')))}</td>"
            f"<td><code>{_esc(_short_command(str(row.get('command', '')), limit=150))}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>PID</th><th>CPU Cores</th><th>Machine CPU</th><th>RSS</th><th>Elapsed</th><th>Command</th></tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def _counts_table(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return '<div class="muted">None.</div>'
    rows = "".join(
        f"<tr><td>{_esc(str(key))}</td><td>{_esc(str(value))}</td></tr>" for key, value in counts.items()
    )
    return f"<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"


def _simple_list(items: object) -> str:
    if not isinstance(items, list) or not items:
        return '<div class="muted">None.</div>'
    rows = []
    for item in items[-8:]:
        if isinstance(item, dict):
            detail = f" {item.get('detail')}" if item.get("detail") else ""
            rows.append(
                "<tr>"
                f"<td>{_esc(str(item.get('stage', '')) + detail)}</td>"
                f"<td>{_esc(str(item.get('case_id', '')))}</td>"
                f"<td>{_esc(str(item.get('status', '')))}</td>"
                f"<td>{_esc(str(item.get('end_time_utc', '')))}</td>"
                "</tr>"
            )
        else:
            rows.append(f"<tr><td colspan=\"4\">{_esc(str(item))}</td></tr>")
    return (
        "<table><thead><tr><th>Stage</th><th>Case</th><th>Status</th><th>Finished</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _failure_table(items: object) -> str:
    if not isinstance(items, list) or not items:
        return '<div class="muted">None.</div>'
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_esc(str(item.get('case_id', '')))}</td>"
            f"<td>{_esc(str(item.get('stage', '')))}</td>"
            f"<td class=\"failed\">{_esc(str(item.get('status', '')))}</td>"
            f"<td><code>{_esc(str(item.get('log_path', '')))}</code></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Case</th><th>Stage</th><th>Status</th><th>Log</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _recent_stage_dicts(frame: pd.DataFrame, limit: int = 8) -> list[dict[str, str]]:
    if not {"case_id", "stage", "status"}.issubset(frame.columns):
        return []
    sort_column = "end_time_utc" if "end_time_utc" in frame.columns else None
    recent = frame.sort_values(sort_column).tail(limit) if sort_column else frame.tail(limit)
    rows = []
    for row in recent.itertuples(index=False):
        detail_value = getattr(row, "detail", "")
        detail = "" if not detail_value or pd.isna(detail_value) else str(detail_value)
        rows.append(
            {
                "case_id": str(getattr(row, "case_id", "")),
                "stage": str(getattr(row, "stage", "")),
                "detail": detail,
                "status": str(getattr(row, "status", "")),
                "end_time_utc": str(getattr(row, "end_time_utc", "")),
            }
        )
    return rows


def _latest_activity_time(outdir: Path) -> float:
    candidates = [
        outdir / "stage_status.csv",
        outdir / "metadata_eligibility.csv",
        outdir / "features",
        outdir / "qc",
        outdir / "logs",
        outdir / "vista_aorta",
        outdir / "cases",
    ]
    return max((_path_mtime(path) for path in candidates), default=0.0)


def _path_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest


def _read_tail(path: Path, tail_lines: int) -> list[str]:
    if tail_lines <= 0 or not path.exists() or path.stat().st_size == 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-tail_lines:]


def _min_datetime_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = [value for value in frame[column].dropna().astype(str).tolist() if value]
    return min(values) if values else ""


def _max_datetime_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = [value for value in frame[column].dropna().astype(str).tolist() if value]
    return max(values) if values else ""


def _parse_datetime(text: str) -> datetime | None:
    if not text or text.lower() in {"nan", "nat"}:
        return None
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return (STAGE_ORDER.index(stage), stage)
    except ValueError:
        return (len(STAGE_ORDER), stage)


def _percent(done: int, total: int) -> float:
    if not total:
        return 0.0
    return min(max(100.0 * done / total, 0.0), 100.0)


def _format_counts(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _format_seconds(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return "unknown"
    if seconds <= 0:
        return "0s"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_minutes(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return "unknown"
    return f"{minutes:.1f} min ago"


def _metadata_value(metadata: object) -> str:
    if not isinstance(metadata, dict):
        return "0/0"
    return f"{metadata.get('eligible', 0)}/{metadata.get('rows', 0)}"


def _short_command(command: str, limit: int = 120) -> str:
    command = " ".join(command.split())
    if len(command) <= limit:
        return command
    return command[: limit - 1] + "..."


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


if __name__ == "__main__":
    main()
