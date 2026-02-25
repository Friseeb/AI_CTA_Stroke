#!/usr/bin/env python3
"""Batch LA/LAA relational metrics with optional LA watertight repair."""

from __future__ import annotations

import argparse
from datetime import datetime
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from la_laa_metrics import compute_metrics, load_mesh, load_mesh_vtk_hole_capped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch LA/LAA relational metrics from mesh pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mesh-root",
        default="/mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro",
        help="Root containing per-case mesh folders.",
    )
    p.add_argument("--la-suffix", default="left_atrium_highres", help="LA mesh suffix token.")
    p.add_argument("--laa-suffix", default="laa_nudf", help="LAA mesh suffix token.")
    p.add_argument(
        "--out-csv",
        default="/mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro/la_laa_metrics_batch.csv",
        help="Output CSV path.",
    )
    p.add_argument("--subject", action="append", default=[], help="Optional subject IDs filter (repeatable).")
    p.add_argument("--case-glob", default="sub-*_acq-CTA_ct", help="Case directory glob under --mesh-root.")
    p.add_argument(
        "--repair-la-vtk-holes",
        action="store_true",
        help="Apply VTK hole-capping to LA (.vtk/.vtp) before metric computation.",
    )
    p.add_argument(
        "--la-hole-size",
        type=float,
        default=50.0,
        help="VTK FillHolesFilter size used when --repair-la-vtk-holes is enabled.",
    )
    p.add_argument(
        "--la-repair-mode",
        choices=["fill_holes", "inferior_cap"],
        default="fill_holes",
        help="LA repair strategy used with --repair-la-vtk-holes.",
    )
    p.add_argument(
        "--la-inferior-band-mm",
        type=float,
        default=15.0,
        help="Only for --la-repair-mode inferior_cap: cap loops with centroid Z <= (Zmin + band).",
    )
    p.add_argument("--progress", action="store_true", help="Show tqdm progress if available.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --out-csv (skip cases already marked success or skip_missing_mesh_pair).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write --out-csv every N processed/updated cases (1 = safest for crash recovery).",
    )
    p.add_argument(
        "--isolate-case-process",
        action="store_true",
        help="Run each case in a separate child process to survive native-library crashes (slower, safer).",
    )
    p.add_argument(
        "--case-timeout-sec",
        type=int,
        default=1800,
        help="Per-case timeout used with --isolate-case-process.",
    )
    p.add_argument(
        "--case-memory-limit-gb",
        type=float,
        default=0.0,
        help=(
            "Only with --isolate-case-process. "
            "Set >0 to cap per-case child virtual memory (GB) via RLIMIT_AS."
        ),
    )
    p.add_argument(
        "--troubleshoot-log",
        default="",
        help=(
            "Optional detailed batch log path. "
            "Default: <out-csv stem>_troubleshoot.log in the output directory."
        ),
    )

    p.add_argument("--near-contact-mm", type=float, default=2.0)
    p.add_argument("--closest-quantile", type=float, default=0.02)
    p.add_argument("--max-gap-fail-mm", type=float, default=12.0)
    p.add_argument("--min-ostium-points", type=int, default=200)
    p.add_argument("--proximal-length-mm", type=float, default=10.0)
    p.add_argument("--ostium-abs-cap-mm", type=float, default=5.0)
    return p.parse_args()


def subject_id_from_case(case_id: str) -> str:
    tok = case_id.split("_")[0].replace("sub-", "")
    if tok.isdigit():
        return str(int(tok))
    return tok


def find_first_mesh(case_dir: Path, suffix: str) -> Path | None:
    patterns = [
        f"{case_dir.name}_{suffix}.vtk",
        f"{case_dir.name}_{suffix}.vtp",
        f"{case_dir.name}_{suffix}.ply",
        f"{case_dir.name}_{suffix}.stl",
        f"{case_dir.name}_{suffix}.obj",
    ]
    for name in patterns:
        p = case_dir / name
        if p.exists():
            return p
    return None


def _load_resume_rows(out_csv: Path) -> tuple[list[dict], set[str]]:
    """Load existing output rows and case IDs considered complete."""
    if not out_csv.exists():
        return [], set()

    df = pd.read_csv(out_csv)
    if df.empty or "case_id" not in df.columns:
        return [], set()

    df["case_id"] = df["case_id"].astype(str)
    df = df.drop_duplicates(subset=["case_id"], keep="last")
    rows = df.to_dict(orient="records")

    if "status" in df.columns:
        done_mask = df["status"].astype(str).isin({"success", "skip_missing_mesh_pair"})
    else:
        done_mask = pd.Series([True] * len(df), index=df.index)
    done_case_ids = set(df.loc[done_mask, "case_id"].tolist())
    return rows, done_case_ids


def _upsert_row(rows: list[dict], row_index: dict[str, int], row: dict) -> None:
    case_id = str(row.get("case_id", ""))
    if case_id and case_id in row_index:
        rows[row_index[case_id]] = row
    else:
        row_index[case_id] = len(rows)
        rows.append(row)


def _write_rows(rows: list[dict], out_csv: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {message}\n")


def _base_case_row(case_id: str, sid: str, la_path: str, laa_path: str) -> dict:
    return {
        "case_id": case_id,
        "subject_id": sid,
        "la_path": la_path,
        "laa_path": laa_path,
        "status": "pending",
        "la_boundary_edges_before": "",
        "la_boundary_edges_after": "",
        "la_watertight_before": "",
        "la_watertight_after": "",
        "la_hole_fill_size": "",
    }


def _compute_case_row(
    case_id: str,
    sid: str,
    la_path: str,
    laa_path: str,
    repair_la_vtk_holes: bool,
    la_hole_size: float,
    la_repair_mode: str,
    la_inferior_band_mm: float,
    params: dict[str, float],
) -> dict:
    base = _base_case_row(case_id, sid, la_path, laa_path)
    try:
        la_path_obj = Path(la_path)
        if repair_la_vtk_holes and la_path_obj.suffix.lower() in {".vtk", ".vtp"}:
            mesh_la, repair_qc = load_mesh_vtk_hole_capped(
                la_path,
                hole_size=float(la_hole_size),
                repair_mode=str(la_repair_mode),
                inferior_band_mm=float(la_inferior_band_mm),
            )
            base.update(repair_qc)
        else:
            mesh_la = load_mesh(la_path)
            base["la_watertight_after"] = bool(mesh_la.is_watertight)

        mesh_laa = load_mesh(laa_path)
        metrics = compute_metrics(mesh_laa=mesh_laa, mesh_la=mesh_la, params=params).iloc[0].to_dict()
        row = {**base, **metrics}
        row["status"] = "success" if not bool(row.get("qc_exception", False)) else "failure"
        return row
    except Exception as exc:  # noqa: BLE001
        row = dict(base)
        row["status"] = "failure"
        row["qc_exception"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def _compute_case_row_isolated(
    case_id: str,
    sid: str,
    la_path: str,
    laa_path: str,
    repair_la_vtk_holes: bool,
    la_hole_size: float,
    la_repair_mode: str,
    la_inferior_band_mm: float,
    params: dict[str, float],
    timeout_sec: int,
    case_memory_limit_gb: float,
) -> dict:
    row_base = _base_case_row(case_id, sid, la_path, laa_path)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_f:
        tmp_csv = Path(tmp_f.name)

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("la_laa_metrics.py")),
        "--la",
        la_path,
        "--laa",
        laa_path,
        "--out",
        str(tmp_csv),
        "--near-contact-mm",
        str(float(params["near_contact_mm"])),
        "--closest-quantile",
        str(float(params["closest_quantile"])),
        "--max-gap-fail-mm",
        str(float(params["max_gap_fail_mm"])),
        "--min-ostium-points",
        str(int(params["min_ostium_points"])),
        "--proximal-length-mm",
        str(float(params["proximal_length_mm"])),
        "--ostium-abs-cap-mm",
        str(float(params["ostium_abs_cap_mm"])),
    ]
    if repair_la_vtk_holes:
        cmd.extend(
            [
                "--repair-la-vtk-holes",
                "--la-hole-size",
                str(float(la_hole_size)),
                "--la-repair-mode",
                str(la_repair_mode),
                "--la-inferior-band-mm",
                str(float(la_inferior_band_mm)),
            ]
        )

    preexec = None
    if float(case_memory_limit_gb) > 0:
        mem_bytes = int(float(case_memory_limit_gb) * (1024**3))

        def _set_limits() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        preexec = _set_limits

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        row = _base_case_row(case_id, sid, la_path, laa_path)
        row["status"] = "failure"
        row["qc_exception"] = True
        row["error"] = f"TimeoutError: case exceeded {int(timeout_sec)} sec"
        tmp_csv.unlink(missing_ok=True)
        return row

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        if len(msg) > 600:
            msg = msg[-600:]
        row = dict(row_base)
        row["status"] = "failure"
        row["qc_exception"] = True
        row["error"] = f"RuntimeError: isolated_case_failed rc={proc.returncode}; {msg}"
        tmp_csv.unlink(missing_ok=True)
        return row

    try:
        df = pd.read_csv(tmp_csv)
        if df.empty:
            raise ValueError("empty metrics output")
        metrics = df.iloc[0].to_dict()
        row = {**row_base, **metrics}
        row["status"] = "success" if not bool(row.get("qc_exception", False)) else "failure"
    except Exception:  # noqa: BLE001
        row = dict(row_base)
        row["status"] = "failure"
        row["qc_exception"] = True
        row["error"] = "RuntimeError: isolated case produced unreadable output"
    finally:
        tmp_csv.unlink(missing_ok=True)
    return row


def main() -> int:
    args = parse_args()
    mesh_root = Path(args.mesh_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(1, int(args.checkpoint_every))
    troubleshoot_log = (
        Path(args.troubleshoot_log)
        if str(args.troubleshoot_log).strip()
        else out_csv.with_name(f"{out_csv.stem}_troubleshoot.log")
    )

    _append_log(
        troubleshoot_log,
        (
            "RUN_START "
            f"argv={' '.join(sys.argv)} | mesh_root={args.mesh_root} | out_csv={out_csv} | "
            f"resume={bool(args.resume)} | isolate_case_process={bool(args.isolate_case_process)} | "
            f"repair_la_vtk_holes={bool(args.repair_la_vtk_holes)} | la_repair_mode={args.la_repair_mode} | "
            f"la_inferior_band_mm={args.la_inferior_band_mm} | la_hole_size={args.la_hole_size} | "
            f"checkpoint_every={checkpoint_every} | case_timeout_sec={args.case_timeout_sec} | "
            f"case_memory_limit_gb={args.case_memory_limit_gb}"
        ),
    )

    wanted_subjects = {str(int(s)) for s in args.subject if str(s).isdigit()}
    case_dirs = sorted(mesh_root.glob(args.case_glob))

    rows: list[dict] = []
    row_index: dict[str, int] = {}
    resume_done_case_ids: set[str] = set()
    if args.resume:
        rows, resume_done_case_ids = _load_resume_rows(out_csv)
        row_index = {
            str(r.get("case_id", "")): i
            for i, r in enumerate(rows)
            if str(r.get("case_id", ""))
        }
        print(f"Resume mode: loaded {len(rows)} existing rows from {out_csv}")
        print(f"Resume mode: {len(resume_done_case_ids)} cases marked complete and will be skipped.")
        _append_log(
            troubleshoot_log,
            f"RESUME loaded_rows={len(rows)} done_cases={len(resume_done_case_ids)} from={out_csv}",
        )

    work: list[tuple[str, str, Path, Path]] = []
    resume_skipped = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        sid = subject_id_from_case(case_id)
        if wanted_subjects and sid not in wanted_subjects:
            continue
        if args.resume and case_id in resume_done_case_ids:
            resume_skipped += 1
            continue

        la_path = find_first_mesh(case_dir, args.la_suffix)
        laa_path = find_first_mesh(case_dir, args.laa_suffix)

        if la_path is None or laa_path is None:
            _upsert_row(
                rows,
                row_index,
                {
                    "case_id": case_id,
                    "subject_id": sid,
                    "la_path": str(la_path) if la_path else "",
                    "laa_path": str(laa_path) if laa_path else "",
                    "status": "skip_missing_mesh_pair",
                },
            )
            continue

        work.append((case_id, sid, la_path, laa_path))

    if args.resume and resume_skipped > 0:
        print(f"Resume mode: skipped {resume_skipped} already-complete cases.")
        _append_log(troubleshoot_log, f"RESUME skipped_complete_cases={resume_skipped}")

    work_iter = work
    if args.progress:
        try:
            from tqdm.auto import tqdm

            work_iter = tqdm(work, total=len(work), unit="case", desc="LA/LAA metrics")
        except Exception:
            print("tqdm not available; running without progress bar.")

    params = {
        "near_contact_mm": float(args.near_contact_mm),
        "closest_quantile": float(args.closest_quantile),
        "max_gap_fail_mm": float(args.max_gap_fail_mm),
        "min_ostium_points": int(args.min_ostium_points),
        "proximal_length_mm": float(args.proximal_length_mm),
        "ostium_abs_cap_mm": float(args.ostium_abs_cap_mm),
    }

    dirty_since_write = 0
    for i, (case_id, sid, la_path, laa_path) in enumerate(work_iter, start=1):
        t0 = time.perf_counter()
        _append_log(
            troubleshoot_log,
            (
                f"CASE_START i={i}/{len(work)} case_id={case_id} subject_id={sid} "
                f"la={la_path} laa={laa_path}"
            ),
        )
        if args.isolate_case_process:
            row = _compute_case_row_isolated(
                case_id=case_id,
                sid=sid,
                la_path=str(la_path),
                laa_path=str(laa_path),
                repair_la_vtk_holes=bool(args.repair_la_vtk_holes),
                la_hole_size=float(args.la_hole_size),
                la_repair_mode=str(args.la_repair_mode),
                la_inferior_band_mm=float(args.la_inferior_band_mm),
                params=params,
                timeout_sec=int(args.case_timeout_sec),
                case_memory_limit_gb=float(args.case_memory_limit_gb),
            )
        else:
            row = _compute_case_row(
                case_id=case_id,
                sid=sid,
                la_path=str(la_path),
                laa_path=str(laa_path),
                repair_la_vtk_holes=bool(args.repair_la_vtk_holes),
                la_hole_size=float(args.la_hole_size),
                la_repair_mode=str(args.la_repair_mode),
                la_inferior_band_mm=float(args.la_inferior_band_mm),
                params=params,
            )
        _upsert_row(rows, row_index, row)
        dt = time.perf_counter() - t0
        err = str(row.get("error", "") or "")
        if len(err) > 1000:
            err = err[:1000] + "...(truncated)"
        _append_log(
            troubleshoot_log,
            (
                f"CASE_END i={i}/{len(work)} case_id={case_id} status={row.get('status')} "
                f"qc_exception={row.get('qc_exception', '')} duration_sec={dt:.2f} "
                f"error={err}"
            ),
        )

        dirty_since_write += 1
        if dirty_since_write >= checkpoint_every:
            _write_rows(rows, out_csv)
            dirty_since_write = 0
            _append_log(
                troubleshoot_log,
                f"CHECKPOINT rows_written={len(rows)} out_csv={out_csv}",
            )

    _write_rows(rows, out_csv)
    df = pd.DataFrame(rows)

    ok = int((df["status"] == "success").sum()) if not df.empty else 0
    print(f"Saved batch metrics: {out_csv}")
    print(f"Rows: {len(df)} | success: {ok} | non-success: {len(df) - ok}")
    _append_log(
        troubleshoot_log,
        f"RUN_END rows={len(df)} success={ok} non_success={len(df) - ok} out_csv={out_csv}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
