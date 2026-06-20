#!/usr/bin/env python
"""Build an LAA annotation pilot dashboard (HTML + CSV).

Scans a pilot root (default ``outputs/laa_pilot``) for per-case / per-reader
annotation state and summarizes: which prior sources are staged, whether the
case is finalized, the model/source used, masks saved, timing/confidence, and
the new-vs-old (corrected-vs-candidate) comparison written at Finalize.

The HTML is self-contained (sortable, filterable). The CSV is the same rows for
downstream analysis.

Usage:
  python scripts/build_laa_pilot_dashboard.py \
      --pilot-root outputs/laa_pilot \
      --out outputs/laa_pilot/laa_pilot_dashboard.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse the source->filename map from the tested core (keeps naming DRY).
_MODULE_DIR = Path(__file__).resolve().parents[1] / "slicer_module"
import sys

sys.path.insert(0, str(_MODULE_DIR))
try:
    from laa_annotation_core import CANDIDATE_SOURCE_FILES  # noqa: E402
except Exception:  # pragma: no cover - fallback if import path changes
    CANDIDATE_SOURCE_FILES = {
        "VISTA-3D": ("vista3d_laa", "vista3d_prompt"),
        "NUDF": ("nudf_laa",),
        "TotalSegmentator": ("totalseg_laa", "atrial_appendage_left", "left_atrial_appendage"),
        "Consensus": ("consensus_laa",),
    }

# Columns in display order: (key, header).
COLUMNS = [
    ("case_id", "Case"),
    ("reader_id", "Reader"),
    ("status", "Status"),
    ("staged_sources", "Priors staged"),
    ("candidate_source", "Source used"),
    ("model_used", "Model"),
    ("prompts", "Prompts (+/-)"),
    ("edit_count", "Edits"),
    ("annotation_time_min", "Time (min)"),
    ("seg_conf", "Seg conf"),
    ("type1_conf", "T1 conf"),
    ("image_quality", "IQ"),
    ("type1_present", "T1?"),
    ("whole_saved", "Whole mask"),
    ("type1_saved", "T1 mask"),
    ("cmp_dice", "Dice vs cand"),
    ("cmp_added_ml", "+mL"),
    ("cmp_removed_ml", "-mL"),
    ("cmp_net_ml", "net mL"),
    ("last_modified", "Last modified"),
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _staged_sources(candidate_dir: Path) -> str:
    if not candidate_dir.is_dir():
        return ""
    files = [p.name for p in candidate_dir.glob("*.nii.gz")]
    found = []
    for source, stems in CANDIDATE_SOURCE_FILES.items():
        if any(any(stem in f for stem in stems) for f in files):
            found.append(source)
    return ", ".join(found)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def scan_reader(case_id: str, reader_dir: Path) -> dict[str, Any]:
    reader_id = "(pilot)" if reader_dir.name == "laa_annotation" else reader_dir.name
    logs = reader_dir / "logs"
    session = _load_json(logs / f"{case_id}_session.json")
    pilot = _load_json(reader_dir / "metrics" / f"{case_id}_pilot.json")
    cmp_metrics = _load_json(reader_dir / "metrics" / f"{case_id}_candidate_comparison.json")

    whole = reader_dir / "manual_masks" / f"{case_id}_whole_laa.nii.gz"
    type1 = reader_dir / "type1_masks" / f"{case_id}_type1.nii.gz"

    finalized = bool(session.get("plugin_version")) or whole.exists()
    status = "finalized" if finalized else (session.get("status") or "pending")

    # source used: prefer the explicit field (new finalizations), else infer.
    source_used = session.get("candidate_source") or ""

    time_s = session.get("annotation_time_s") or pilot.get("annotation_time_s")
    time_min = round(time_s / 60.0, 1) if isinstance(time_s, (int, float)) else None

    mtimes = [p.stat().st_mtime for p in reader_dir.rglob("*") if p.is_file()]
    last_mod = (
        datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M") if mtimes else ""
    )

    return {
        "case_id": case_id,
        "reader_id": reader_id,
        "status": status,
        "staged_sources": _staged_sources(reader_dir / "candidate_masks"),
        "candidate_source": source_used,
        "model_used": session.get("model_used") or pilot.get("model_used") or "",
        "prompts": "{}/{}".format(
            session.get("positive_prompt_count", pilot.get("positive_prompt_count", "") or ""),
            session.get("negative_prompt_count", pilot.get("negative_prompt_count", "") or ""),
        ).strip("/") if (session or pilot) else "",
        "edit_count": session.get("edit_count", pilot.get("edit_count", "")),
        "annotation_time_min": time_min,
        "seg_conf": session.get("segmentation_confidence", pilot.get("segmentation_confidence")),
        "type1_conf": session.get("type1_confidence", pilot.get("type1_confidence")),
        "image_quality": session.get("image_quality", pilot.get("image_quality")),
        "type1_present": session.get("type1_present", pilot.get("type1_present")),
        "whole_saved": whole.exists(),
        "type1_saved": type1.exists(),
        "cmp_dice": cmp_metrics.get("dice"),
        "cmp_added_ml": cmp_metrics.get("added_volume_ml"),
        "cmp_removed_ml": cmp_metrics.get("removed_volume_ml"),
        "cmp_net_ml": cmp_metrics.get("volume_change_ml"),
        "last_modified": last_mod,
    }


# Standard per-case output subdirs (NOT reader folders).
_OUTPUT_SUBDIRS = {
    "candidate_masks", "manual_masks", "type1_masks",
    "iterations", "logs", "screenshots", "metrics",
}


def scan_pilot(pilot_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(pilot_root.glob("sub-*"), key=lambda p: p.name):
        if not case_dir.is_dir():
            continue
        ann = case_dir / "laa_annotation"
        if not ann.is_dir():
            rows.append({"case_id": case_dir.name, "reader_id": "", "status": "not staged"})
            continue
        # Reader folders are subdirs whose name is not a standard output subdir.
        reader_dirs = [
            d for d in ann.glob("*") if d.is_dir() and d.name not in _OUTPUT_SUBDIRS
        ]
        # Pilot mode (no reader): outputs live directly under laa_annotation/.
        if (ann / "logs").is_dir() or (ann / "manual_masks").is_dir():
            reader_dirs.append(ann)
        if not reader_dirs:
            rows.append({"case_id": case_dir.name, "reader_id": "", "status": "not staged"})
            continue
        for reader_dir in sorted(reader_dirs, key=lambda p: p.name):
            rows.append(scan_reader(case_dir.name, reader_dir))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finalized = [r for r in rows if r.get("status") == "finalized"]
    by_source: dict[str, int] = {}
    by_model: dict[str, int] = {}
    dices = []
    times = []
    for r in finalized:
        by_source[r.get("candidate_source") or "(unrecorded)"] = (
            by_source.get(r.get("candidate_source") or "(unrecorded)", 0) + 1
        )
        by_model[r.get("model_used") or "(none)"] = by_model.get(r.get("model_used") or "(none)", 0) + 1
        if isinstance(r.get("cmp_dice"), (int, float)):
            dices.append(r["cmp_dice"])
        if isinstance(r.get("annotation_time_min"), (int, float)):
            times.append(r["annotation_time_min"])
    return {
        "n_rows": len(rows),
        "n_finalized": len(finalized),
        "n_pending": sum(1 for r in rows if r.get("status") not in ("finalized", "not staged")),
        "n_not_staged": sum(1 for r in rows if r.get("status") == "not staged"),
        "by_source": by_source,
        "by_model": by_model,
        "mean_dice": round(sum(dices) / len(dices), 3) if dices else None,
        "mean_time_min": round(sum(times) / len(times), 1) if times else None,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = [k for k, _ in COLUMNS]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k), digits=3) for k in keys})


def _status_class(status: str) -> str:
    return {
        "finalized": "ok",
        "pending": "warn",
        "not staged": "muted",
    }.get(status, "warn")


def write_html(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path, pilot_root: Path) -> None:
    def cards() -> str:
        src = " ".join(f"<span class='pill'>{html.escape(k)}: {v}</span>" for k, v in summary["by_source"].items())
        mdl = " ".join(f"<span class='pill'>{html.escape(k)}: {v}</span>" for k, v in summary["by_model"].items())
        return f"""
        <div class='cards'>
          <div class='card'><div class='n'>{summary['n_rows']}</div><div class='l'>case×reader rows</div></div>
          <div class='card ok'><div class='n'>{summary['n_finalized']}</div><div class='l'>finalized</div></div>
          <div class='card warn'><div class='n'>{summary['n_pending']}</div><div class='l'>pending</div></div>
          <div class='card muted'><div class='n'>{summary['n_not_staged']}</div><div class='l'>not staged</div></div>
          <div class='card'><div class='n'>{_fmt(summary['mean_dice'])}</div><div class='l'>mean Dice vs cand</div></div>
          <div class='card'><div class='n'>{_fmt(summary['mean_time_min'],1)}</div><div class='l'>mean time (min)</div></div>
        </div>
        <div class='by'><b>Finalized by source used:</b> {src or '—'}</div>
        <div class='by'><b>Finalized by model:</b> {mdl or '—'}</div>
        """

    head = "".join(f"<th onclick='sortBy({i})'>{html.escape(h)}</th>" for i, (_, h) in enumerate(COLUMNS))
    body_rows = []
    for r in rows:
        tds = []
        for k, _ in COLUMNS:
            val = _fmt(r.get(k))
            cls = ""
            if k == "status":
                cls = f" class='{_status_class(r.get('status',''))}'"
            if k in ("whole_saved", "type1_saved"):
                cls = " class='ok'" if r.get(k) else " class='muted'"
            tds.append(f"<td{cls}>{html.escape(val)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>LAA pilot dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a}}
 h1{{margin:0 0 4px}} .sub{{color:#666;margin-bottom:16px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
 .card{{background:#f4f6f8;border-radius:10px;padding:12px 18px;min-width:110px}}
 .card .n{{font-size:26px;font-weight:700}} .card .l{{color:#666;font-size:12px}}
 .card.ok{{background:#e6f5ea}} .card.warn{{background:#fdf3e0}} .card.muted{{background:#eee}}
 .by{{margin:6px 0;font-size:13px}} .pill{{background:#eef;border-radius:12px;padding:2px 8px;margin-right:4px;font-size:12px;display:inline-block}}
 input{{padding:6px 10px;margin:10px 0;width:280px;border:1px solid #ccc;border-radius:6px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border-bottom:1px solid #e3e3e3;padding:6px 8px;text-align:left;white-space:nowrap}}
 th{{position:sticky;top:0;background:#fafafa;cursor:pointer;user-select:none}}
 td.ok{{color:#137333;font-weight:600}} td.warn{{color:#b06000;font-weight:600}} td.muted{{color:#999}}
 tr:hover{{background:#f7fbff}}
</style></head><body>
<h1>LAA annotation pilot</h1>
<div class='sub'>{html.escape(str(pilot_root))} · generated {generated}</div>
{cards()}
<input id='q' placeholder='filter (case / reader / source / status)…' oninput='filterRows()'>
<table id='t'><thead><tr>{head}</tr></thead><tbody>
{''.join(body_rows)}
</tbody></table>
<script>
 function filterRows(){{
   const q=document.getElementById('q').value.toLowerCase();
   document.querySelectorAll('#t tbody tr').forEach(tr=>{{
     tr.style.display = tr.innerText.toLowerCase().includes(q)?'':'none';
   }});
 }}
 let sortAsc=true,lastCol=-1;
 function sortBy(col){{
   const tb=document.querySelector('#t tbody');
   const rows=[...tb.querySelectorAll('tr')];
   sortAsc = (col===lastCol)? !sortAsc : true; lastCol=col;
   rows.sort((a,b)=>{{
     let x=a.children[col].innerText, y=b.children[col].innerText;
     const nx=parseFloat(x), ny=parseFloat(y);
     if(!isNaN(nx)&&!isNaN(ny)){{x=nx;y=ny;}}
     return (x>y?1:x<y?-1:0)*(sortAsc?1:-1);
   }});
   rows.forEach(r=>tb.appendChild(r));
 }}
</script>
</body></html>"""
    path.write_text(doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[3] / "outputs" / "laa_pilot"
    ap.add_argument("--pilot-root", default=str(repo_default))
    ap.add_argument("--out", default=None, help="HTML output (default <pilot-root>/laa_pilot_dashboard.html)")
    ap.add_argument("--csv", default=None, help="CSV output (default alongside HTML)")
    args = ap.parse_args()

    pilot_root = Path(args.pilot_root).expanduser().resolve()
    if not pilot_root.is_dir():
        raise SystemExit(f"Pilot root not found: {pilot_root}")
    out_html = Path(args.out).expanduser() if args.out else pilot_root / "laa_pilot_dashboard.html"
    out_csv = Path(args.csv).expanduser() if args.csv else out_html.with_suffix(".csv")

    rows = scan_pilot(pilot_root)
    summary = summarize(rows)
    write_csv(rows, out_csv)
    write_html(rows, summary, out_html, pilot_root)

    print(f"Scanned {summary['n_rows']} rows "
          f"({summary['n_finalized']} finalized, {summary['n_pending']} pending, "
          f"{summary['n_not_staged']} not staged)")
    print(f"HTML: {out_html}")
    print(f"CSV : {out_csv}")


if __name__ == "__main__":
    main()
