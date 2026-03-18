#!/usr/bin/env python3
"""
Live tqdm monitor for DAYLIGHTBIDS NUDF/LA batch runs.

Reads qc_summary_live.csv and shows progress against expected input cases.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor qc_summary_live.csv with tqdm")
    p.add_argument("--root", required=True, help="DAYLIGHTBIDS root")
    p.add_argument("--use-nondefaced", action="store_true", help="Use root CTA files instead of derivatives/defaced")
    p.add_argument("--input-dir", default=None, help="Override input dir")
    p.add_argument("--input-glob", default=None, help="Override input glob")
    p.add_argument("--out-dir", default=None, help="Output base dir (default: root/derivatives/nudf_la)")
    p.add_argument("--subprocess-log-dir", default=None, help="Per-case log dir (default: <out-dir>/_logs)")
    p.add_argument("--poll-sec", type=float, default=3.0, help="Polling interval in seconds")
    p.add_argument("--stop-when-done", action="store_true", help="Exit when all expected cases are processed")
    p.add_argument(
        "--history-mode",
        choices=("latest-pass", "full-history"),
        default="latest-pass",
        help="How to interpret qc_summary_live history.",
    )
    return p.parse_args()


def _list_inputs(root: Path, args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = root if args.use_nondefaced else root / "derivatives" / "defaced"
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if args.input_glob:
        input_glob = args.input_glob
    else:
        input_glob = "sub-*_acq-CTA_ct.nii.gz" if args.use_nondefaced else "sub-*_acq-CTA_ct_defaced.nii.gz"
    return sorted(input_dir.glob(input_glob))


def _read_summary_rows(summary_live: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not summary_live.exists() or summary_live.stat().st_size == 0:
        return rows
    with summary_live.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _latest_pass_start_row(
    rows: list[dict[str, str]],
    case_to_index: dict[str, int],
) -> int:
    start_row = 0
    last_idx: int | None = None
    for i, row in enumerate(rows):
        cid = row.get("case_id", "")
        idx = case_to_index.get(cid)
        if idx is None:
            continue
        if last_idx is not None and idx < last_idx:
            start_row = i
        last_idx = idx
    return start_row


def _latest_rows_by_case(
    rows: list[dict[str, str]],
    input_case_ids: set[str],
    case_to_index: dict[str, int],
    history_mode: str,
) -> tuple[dict[str, dict[str, str]], int]:
    if history_mode == "latest-pass":
        start_row = _latest_pass_start_row(rows, case_to_index)
        rows = rows[start_row:]
    else:
        start_row = 0

    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        cid = row.get("case_id", "")
        if cid and cid in input_case_ids:
            by_case[cid] = row
    return by_case, start_row


def _latest_case_log(log_dir: Path) -> tuple[str, float] | None:
    if not log_dir.exists():
        return None
    logs = list(log_dir.glob("sub-*_acq-CTA_ct.log"))
    if not logs:
        return None
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    age = time.time() - latest.stat().st_mtime
    return latest.stem, age


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out_dir = Path(args.out_dir) if args.out_dir else root / "derivatives" / "nudf_la"
    summary_live = out_dir / "qc_summary_live.csv"
    log_dir = Path(args.subprocess_log_dir) if args.subprocess_log_dir else (out_dir / "_logs")
    inputs = _list_inputs(root, args)
    total = len(inputs)
    if total == 0:
        raise RuntimeError("No input cases found.")

    input_case_ids = {p.name[:-7] for p in inputs if p.name.endswith(".nii.gz")}
    case_to_index = {case_id: i for i, case_id in enumerate(sorted(input_case_ids))}
    bar = tqdm(total=total, desc="NUDF/LA batch", unit="case", dynamic_ncols=True)
    last_done = -1
    try:
        while True:
            rows = _read_summary_rows(summary_live)
            by_case, start_row = _latest_rows_by_case(
                rows=rows,
                input_case_ids=input_case_ids,
                case_to_index=case_to_index,
                history_mode=args.history_mode,
            )
            done_cases = set(by_case.keys()) & input_case_ids
            done = len(done_cases)
            ok = sum(1 for c in done_cases if by_case[c].get("status") == "ok")
            failed = sum(1 for c in done_cases if by_case[c].get("status") == "failed")
            skipped = sum(1 for c in done_cases if by_case[c].get("status") == "skipped")
            summary_age = (time.time() - summary_live.stat().st_mtime) if summary_live.exists() else float("inf")
            latest = _latest_case_log(log_dir)
            current_case = latest[0] if latest else "-"
            current_case_age = int(latest[1]) if latest else -1

            bar.n = done
            bar.set_postfix(
                ok=ok,
                failed=failed,
                skipped=skipped,
                current=current_case,
                case_age_s=current_case_age,
                summary_age_s=int(summary_age) if summary_age != float("inf") else -1,
                mode=args.history_mode,
                start_row=start_row,
            )
            bar.refresh()

            if done != last_done:
                last_done = done
                ts = time.strftime("%H:%M:%S")
                tqdm.write(f"[{ts}] done={done}/{total} ok={ok} failed={failed} skipped={skipped}")

            if args.stop_when_done and done >= total:
                break
            time.sleep(max(0.2, args.poll_sec))
    except KeyboardInterrupt:
        pass
    finally:
        bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
