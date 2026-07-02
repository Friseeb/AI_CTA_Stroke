"""Run a staged aorta batch with neuro-CTA filtering, ntfy, and a watchdog."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


AORTA_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = AORTA_ROOT / "configs" / "calcium_dynamic_500hu.yaml"
DEFAULT_STAGED_SCRIPT = AORTA_ROOT / "scripts" / "run_manifest_staged.py"
DEFAULT_NV_PYTHON = Path("/opt/anaconda3/envs/nv-segment-ct/bin/python")


@dataclass(frozen=True)
class NtfyConfig:
    topic: str
    url: str
    token: str
    priority: str

    @property
    def enabled(self) -> bool:
        return bool(self.topic)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_inputs(args)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = outdir / "logs" / "batch_watchdog"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log = log_dir / "staged_runner.log"
    watchdog_log = log_dir / "watchdog.log"

    ntfy = resolve_ntfy_config(args, outdir)

    command = build_staged_command(args)
    command_text = " ".join(command)
    watchdog_log.write_text(
        f"[{utc_now()}] starting batch watchdog\ncommand: {command_text}\n",
        encoding="utf-8",
    )
    if ntfy.enabled:
        subscribe_url = ntfy_endpoint(ntfy.url, ntfy.topic)
        print(f"ntfy topic: {ntfy.topic}")
        print(f"ntfy subscribe URL: {subscribe_url}")
        _append_log(watchdog_log, f"ntfy topic: {ntfy.topic}\nntfy subscribe URL: {subscribe_url}\n")
    notify(
        ntfy,
        "Aorta batch started",
        f"Started staged aorta batch\noutdir: {outdir}\nstages: {args.stages}",
        tags="hourglass_flowing_sand,cta",
    )

    process = subprocess.Popen(
        command,
        cwd=str(AORTA_ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    line_queue: queue.Queue[str] = queue.Queue()
    pump = threading.Thread(
        target=_pump_process_output,
        args=(process, line_queue, runner_log),
        daemon=True,
    )
    pump.start()

    last_notice = time.monotonic()
    last_activity_seen = latest_activity_time(outdir)
    last_stall_notice = 0.0
    poll_seconds = max(float(args.poll_seconds), 5.0)
    notify_every_seconds = max(float(args.notify_every_minutes), 0.0) * 60.0
    stall_seconds = max(float(args.stall_minutes), 0.0) * 60.0

    try:
        while process.poll() is None:
            _drain_print_queue(line_queue)
            now = time.monotonic()
            activity = latest_activity_time(outdir)
            if activity > last_activity_seen:
                last_activity_seen = activity
                last_stall_notice = 0.0

            if notify_every_seconds and now - last_notice >= notify_every_seconds:
                summary = summarize_run(outdir)
                notify(ntfy, "Aorta batch heartbeat", summary, tags="bar_chart,cta")
                _append_log(watchdog_log, f"[{utc_now()}] heartbeat\n{summary}\n")
                last_notice = now

            if stall_seconds and last_activity_seen > 0:
                stalled_for = time.time() - last_activity_seen
                if stalled_for >= stall_seconds and now - last_stall_notice >= stall_seconds:
                    message = (
                        f"No output/log activity for {stalled_for / 60.0:.1f} minutes.\n"
                        f"Process is still running.\n{summarize_run(outdir)}"
                    )
                    notify(ntfy, "Aorta batch watchdog warning", message, priority="4", tags="warning,cta")
                    _append_log(watchdog_log, f"[{utc_now()}] stall warning\n{message}\n")
                    last_stall_notice = now

            time.sleep(poll_seconds)

        pump.join(timeout=5)
        _drain_print_queue(line_queue)
        returncode = int(process.returncode or 0)
        summary = summarize_run(outdir)
        if returncode == 0:
            notify(ntfy, "Aorta batch complete", summary, tags="white_check_mark,cta")
        else:
            notify(
                ntfy,
                "Aorta batch failed",
                f"Return code: {returncode}\n{summary}\nLog: {runner_log}",
                priority="5",
                tags="x,cta",
            )
        _append_log(watchdog_log, f"[{utc_now()}] finished returncode={returncode}\n{summary}\n")
        raise SystemExit(returncode)
    except KeyboardInterrupt:
        process.terminate()
        notify(ntfy, "Aorta batch interrupted", f"Interrupted by user\noutdir: {outdir}", priority="4", tags="warning")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stages",
        default="vista,base,calcium,fat-wall,protrusions,wall-thickness,radiomics",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--staged-script", type=Path, default=DEFAULT_STAGED_SCRIPT)
    parser.add_argument("--nv-python", default=str(DEFAULT_NV_PYTHON))
    parser.add_argument("--nv-device", default="auto")
    parser.add_argument("--vista-workers", type=int, default=1)
    parser.add_argument("--base-workers", type=int, default=2)
    parser.add_argument("--calcium-workers", type=int, default=2)
    parser.add_argument("--fat-wall-workers", type=int, default=1)
    parser.add_argument("--protrusion-workers", type=int, default=1)
    parser.add_argument("--wall-thickness-workers", type=int, default=2)
    parser.add_argument("--radiomics-workers", type=int, default=1)
    parser.add_argument("--radiomics-split-by-region", action="store_true")
    parser.add_argument("--radiomics-region-workers", type=int, default=4)
    parser.add_argument("--metadata-filter", choices=["none", "neuro-cta"], default="neuro-cta")
    parser.add_argument("--metadata-include-keyword", action="append", default=[])
    parser.add_argument("--metadata-exclude-keyword", action="append", default=[])
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-label",
        default="",
        help="Human-readable run label used for auto ntfy topics, e.g. slaobids-aorta-full.",
    )
    parser.add_argument(
        "--ntfy-topic",
        default="",
        help="ntfy topic, full ntfy URL, or 'auto' to derive a readable topic from run metadata.",
    )
    parser.add_argument("--ntfy-topic-prefix", default="aorta-cta")
    parser.add_argument("--ntfy-url", default="")
    parser.add_argument("--ntfy-token", default="")
    parser.add_argument("--ntfy-priority", default="3")
    parser.add_argument("--notify-every-minutes", type=float, default=30.0)
    parser.add_argument("--stall-minutes", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    """Fail before notifications/subprocess launch when required paths are invalid."""
    missing = []
    for label, path in [
        ("manifest", Path(args.manifest)),
        ("config", Path(args.config)),
        ("staged script", Path(args.staged_script)),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if not _executable_exists(str(args.python)):
        missing.append(f"python: {args.python}")
    if missing:
        raise SystemExit("Missing required runner input(s):\n" + "\n".join(missing))


def _executable_exists(command: str) -> bool:
    path = Path(command)
    if path.is_absolute() or "/" in command:
        return path.exists()
    return shutil.which(command) is not None


def build_staged_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(args.staged_script),
        "--manifest",
        str(args.manifest),
        "--outdir",
        str(args.outdir),
        "--stages",
        str(args.stages),
        "--config",
        str(args.config),
        "--metadata-filter",
        str(args.metadata_filter),
        "--vista-workers",
        str(args.vista_workers),
        "--base-workers",
        str(args.base_workers),
        "--calcium-workers",
        str(args.calcium_workers),
        "--fat-wall-workers",
        str(args.fat_wall_workers),
        "--protrusion-workers",
        str(args.protrusion_workers),
        "--wall-thickness-workers",
        str(args.wall_thickness_workers),
        "--radiomics-workers",
        str(args.radiomics_workers),
        "--nv-python",
        str(args.nv_python),
        "--nv-device",
        str(args.nv_device),
    ]
    if not args.no_skip_existing:
        command.append("--skip-existing")
    if args.keep_going:
        command.append("--keep-going")
    if args.dry_run:
        command.append("--dry-run")
    if args.allow_missing_metadata:
        command.append("--allow-missing-metadata")
    if args.radiomics_split_by_region:
        command.extend(["--radiomics-split-by-region", "--radiomics-region-workers", str(args.radiomics_region_workers)])
    for keyword in args.metadata_include_keyword:
        command.extend(["--metadata-include-keyword", str(keyword)])
    for keyword in args.metadata_exclude_keyword:
        command.extend(["--metadata-exclude-keyword", str(keyword)])
    return command


def resolve_ntfy_config(args: argparse.Namespace, outdir: str | Path) -> NtfyConfig:
    """Build ntfy settings and persist generated topic information."""
    topic = args.ntfy_topic or os.environ.get("NTFY_TOPIC", "")
    base_url = args.ntfy_url or os.environ.get("NTFY_URL", "https://ntfy.sh")
    token = args.ntfy_token or os.environ.get("NTFY_TOKEN", "")
    if topic.lower() == "auto":
        topic = generate_ntfy_topic(
            prefix=str(args.ntfy_topic_prefix),
            manifest=Path(args.manifest),
            outdir=Path(outdir),
            run_label=str(getattr(args, "run_label", "")),
        )
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "ntfy_topic.txt").write_text(
            f"{topic}\n{ntfy_endpoint(base_url, topic)}\n",
            encoding="utf-8",
        )
    return NtfyConfig(topic=topic, url=base_url, token=token, priority=str(args.ntfy_priority))


def generate_ntfy_topic(
    prefix: str = "aorta-cta",
    manifest: str | Path | None = None,
    outdir: str | Path | None = None,
    run_label: str = "",
) -> str:
    """Derive a readable ntfy topic from run metadata."""
    tokens = [_slug_token(prefix) or "aorta-cta"]
    if run_label:
        tokens.append(_slug_token(run_label))
    else:
        if manifest is not None:
            manifest_path = Path(manifest)
            if manifest_path.parent.name and manifest_path.stem.lower() in {"manifest", "cases", "subjects"}:
                tokens.append(_slug_token(manifest_path.parent.name))
            tokens.append(_slug_token(manifest_path.stem))
        if outdir is not None:
            tokens.append(_slug_token(Path(outdir).name))
    deduped: list[str] = []
    for token in tokens:
        if token and token not in deduped:
            deduped.append(token)
    topic = "-".join(deduped)
    return topic[:120].strip("-") or "aorta-cta"


def _slug_token(text: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "-" for char in str(text))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def summarize_run(outdir: str | Path) -> str:
    try:
        from aorta_cta_radiomics.batch_progress import compact_ntfy_summary

        return compact_ntfy_summary(outdir)
    except Exception:  # noqa: BLE001
        outdir = Path(outdir)
        lines = [f"outdir: {outdir}"]
        metadata_path = outdir / "metadata_eligibility.csv"
        if metadata_path.exists() and metadata_path.stat().st_size > 0:
            try:
                metadata = pd.read_csv(metadata_path)
                kept = int(metadata["eligible"].astype(bool).sum()) if "eligible" in metadata.columns else 0
                lines.append(f"metadata: kept {kept}/{len(metadata)}")
                if "reason" in metadata.columns:
                    reason_counts = metadata["reason"].fillna("").value_counts().to_dict()
                    lines.append("metadata reasons: " + _format_counts(reason_counts))
            except (OSError, pd.errors.EmptyDataError, KeyError):
                lines.append("metadata: unreadable")
        status_path = outdir / "stage_status.csv"
        if status_path.exists() and status_path.stat().st_size > 0:
            try:
                status = pd.read_csv(status_path)
                lines.append(f"stage rows: {len(status)}")
            except (OSError, pd.errors.EmptyDataError, KeyError):
                lines.append("stage status: unreadable")
        else:
            lines.append("stage status: not written yet")
        latest = latest_activity_time(outdir)
        if latest:
            lines.append(f"latest activity: {datetime.fromtimestamp(latest).isoformat(timespec='seconds')}")
        return "\n".join(lines)


def latest_activity_time(outdir: str | Path) -> float:
    outdir = Path(outdir)
    candidates = [
        outdir / "stage_status.csv",
        outdir / "metadata_eligibility.csv",
        outdir / "features",
        outdir / "qc",
        outdir / "logs",
    ]
    return max((path_mtime(path) for path in candidates), default=0.0)


def path_mtime(path: Path) -> float:
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


def notify(
    config: NtfyConfig,
    title: str,
    message: str,
    priority: str | None = None,
    tags: str = "cta",
) -> None:
    if not config.enabled:
        return
    url = ntfy_endpoint(config.url, config.topic)
    headers = {
        "Title": title,
        "Priority": priority or config.priority,
        "Tags": tags,
    }
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    request = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[watchdog] ntfy notification failed: {exc}", file=sys.stderr)


def ntfy_endpoint(base_url: str, topic: str) -> str:
    if topic.startswith("http://") or topic.startswith("https://"):
        return topic
    return f"{base_url.rstrip('/')}/{topic.strip('/')}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_counts(counts: dict[object, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _pump_process_output(process: subprocess.Popen[str], line_queue: queue.Queue[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        if process.stdout is None:
            return
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            line_queue.put(line)


def _drain_print_queue(line_queue: queue.Queue[str]) -> None:
    while True:
        try:
            line = line_queue.get_nowait()
        except queue.Empty:
            return
        print(line, end="")


def _append_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
