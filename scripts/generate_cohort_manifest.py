#!/usr/bin/env python3
"""
Generate cohort_manifest.csv + cohort_report.html for a SLAAOBIDS folder.

Scans all sub-* directories and reports, per subject:
  - eCTA, CT_heart, CT_thorax, CT_totalbody, CT_abdomen
  - status (available / error / absent), n_phases, filenames, size MB, slices

Usage:
  python scripts/generate_cohort_manifest.py \
      --root "C:/Users/spost/Desktop/CT_image/SLAAOBIDS - Copy"
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False
    print("[WARN] nibabel not found — slice counts will be 0", file=sys.stderr)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# ── CT type definitions ───────────────────────────────────────────────────────
CT_TYPES = [
    # (key,  display_label,  acq_tag,    multiphase)
    ("ecta",      "eCTA",         "ecta",      False),
    ("ctheart",   "CT_heart",     "ctheart",   True),
    ("ctthorax",  "CT_thorax",    "ctthorax",  False),
    ("ctbody",    "CT_totalbody", "ctbody",    True),
    ("ctabdomen", "CT_abdomen",   "ctabdomen", True),
]
CT_KEYS = [t[0] for t in CT_TYPES]


# ── Scanning ──────────────────────────────────────────────────────────────────

def _nifti_slices(fp: Path) -> tuple[int, str]:
    """Return (n_slices, 'available'|'error')."""
    if not HAS_NIB:
        return 0, "available"
    try:
        shape = nib.load(str(fp)).shape
        return (int(shape[2]) if len(shape) >= 3 else 0), "available"
    except Exception:
        return 0, "error"


def scan_subject(sub_dir: Path) -> dict[str, dict]:
    """Return per-CT-type info dict for one subject directory."""
    result: dict[str, dict] = {}
    for key, _, acq, multiphase in CT_TYPES:
        pattern = (f"*_acq-{acq}_ph*_ct.nii.gz" if multiphase
                   else f"*_acq-{acq}_ct.nii.gz")
        files = sorted(sub_dir.glob(pattern))

        if not files:
            result[key] = {"status": "absent", "n_phases": 0, "files": [],
                           "size_mb": 0.0, "slices": 0, "phases": []}
            continue

        phases = []
        total_mb = 0.0
        total_sl = 0
        worst = "available"

        for fp in files:
            mb = round(fp.stat().st_size / (1024 * 1024), 1)
            total_mb += mb
            sl, st = _nifti_slices(fp)
            total_sl += sl
            if st == "error":
                worst = "error"
            phases.append({"filename": fp.name, "size_mb": mb, "slices": sl})

        result[key] = {
            "status":   worst,
            "n_phases": len(files),
            "files":    [fp.name for fp in files],
            "size_mb":  round(total_mb, 1),
            "slices":   total_sl,
            "phases":   phases,
        }
    return result


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(subjects: list[tuple[str, dict]], out_path: Path) -> None:
    cols = ["subject_id", "n_ct_types"]
    for key, _, _, _ in CT_TYPES:
        cols += [f"{key}_status", f"{key}_n_phases", f"{key}_files",
                 f"{key}_size_mb", f"{key}_slices"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sub_id, ct_data in subjects:
            n_avail = sum(1 for k in CT_KEYS if ct_data[k]["status"] == "available")
            row: dict = {"subject_id": sub_id, "n_ct_types": n_avail}
            for key, _, _, _ in CT_TYPES:
                d = ct_data[key]
                row[f"{key}_status"]   = d["status"]
                row[f"{key}_n_phases"] = d["n_phases"]
                row[f"{key}_files"]    = "; ".join(d["files"])
                row[f"{key}_size_mb"]  = d["size_mb"]
                row[f"{key}_slices"]   = d["slices"]
            w.writerow(row)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _cell(ct_info: dict) -> str:
    """HTML for one CT-type table cell."""
    status = ct_info["status"]
    if status == "absent":
        return "—"

    title = escape("; ".join(ct_info["files"]))

    if status == "error":
        return f'<span class="badge-err" title="{title}">error</span>'

    # available
    html = f'<span class="badge-ok" title="{title}">available</span>'
    phases = ct_info["phases"]
    if phases:
        html += "<br><small style='color:#555'>"
        if len(phases) == 1:
            ph = phases[0]
            detail = []
            if ph["size_mb"]: detail.append(f"{ph['size_mb']} MB")
            if ph["slices"]:  detail.append(f"{ph['slices']} sl")
            html += " · ".join(detail)
        else:
            lines = []
            for ph in phases:
                m = re.search(r"_(ph\d+)_", ph["filename"])
                tag = m.group(1) if m else "?"
                detail = [tag]
                if ph["size_mb"]: detail.append(f"{ph['size_mb']} MB")
                if ph["slices"]:  detail.append(f"{ph['slices']} sl")
                lines.append(" · ".join(detail))
            html += "<br>".join(lines)
        html += "</small>"
    return html


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
  body{font-family:Arial,sans-serif;font-size:13px;margin:24px;color:#222;}
  h1{font-size:18px;margin-bottom:4px;}
  .meta{color:#666;font-size:12px;margin-bottom:16px;}
  .cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;}
  .card{background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:10px 16px;min-width:130px;line-height:1.9;}
  .summary-panel{background:#f0f4ff;border:1px solid #c5cae9;border-radius:8px;padding:14px 18px;margin-bottom:16px;}
  .summary-section{margin-bottom:14px;} .summary-section:last-child{margin-bottom:0;}
  .summary-section h3{font-size:13px;font-weight:bold;color:#283593;margin:0 0 8px 0;border-bottom:1px solid #c5cae9;padding-bottom:4px;}
  .sa-row{display:flex;gap:0;margin-bottom:8px;flex-wrap:wrap;}
  .sa-stat{display:flex;flex-direction:column;align-items:center;background:#fff;border:1px solid #c5cae9;padding:6px 16px;min-width:100px;}
  .sa-stat:first-child{border-radius:6px 0 0 6px;} .sa-stat:last-child{border-radius:0 6px 6px 0;} .sa-stat+.sa-stat{border-left:none;}
  .sa-num{font-size:20px;font-weight:bold;line-height:1.2;} .sa-lbl{font-size:11px;color:#555;text-align:center;white-space:nowrap;}
  .stacked-bar{display:flex;height:18px;border-radius:4px;overflow:hidden;margin-bottom:6px;background:#e0e0e0;}
  .bar-0{background:#e0e0e0;} .bar-1{background:#81c784;} .bar-2{background:#4fc3f7;} .bar-3{background:#7986cb;}
  .bar-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#555;}
  .bar-legend span{display:flex;align-items:center;gap:4px;}
  .leg-dot{width:10px;height:10px;border-radius:2px;display:inline-block;flex-shrink:0;}
  .filter-panel{background:#f8f9ff;border:1px solid #c5cae9;border-radius:8px;padding:12px 16px;margin-bottom:16px;}
  .filter-header{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap;}
  .filter-title{font-weight:bold;font-size:13px;color:#283593;}
  .reset-btn{padding:4px 12px;border:1px solid #bbb;border-radius:12px;background:#fff;font-size:12px;cursor:pointer;}
  .reset-btn:hover{background:#e8eaf6;border-color:#7986cb;}
  #row-count{font-size:12px;color:#666;margin-left:auto;}
  .filter-table{border-collapse:collapse;font-size:12px;}
  .filter-table th,.filter-table td{padding:4px 14px;text-align:center;border:none;}
  .ftype-hdr{text-align:left;color:#444;font-weight:bold;min-width:110px;padding-left:0;}
  .ftype-label{text-align:left;font-weight:bold;color:#333;padding:5px 14px 5px 0;white-space:nowrap;}
  .fstate-hdr{font-weight:bold;}
  .frow-count{text-align:right;font-size:12px;font-weight:bold;color:#283593;padding:4px 8px 4px 14px;white-space:nowrap;}
  .ftotal-row td{background:#eef0fa !important;border-top:2px solid #c5cae9;}
  .ftotal-label{color:#283593;font-weight:bold;} .ftotal-cell{font-size:12px;font-weight:bold;text-align:center;}
  .fc-available{color:#2e7d32;} .fstate-hdr.fc-available{color:#2e7d32;}
  .fc-error{color:#b71c1c;}     .fstate-hdr.fc-error{color:#b71c1c;}
  .fc-absent{color:#757575;}    .fstate-hdr.fc-absent{color:#757575;}
  table{border-collapse:collapse;width:100%;table-layout:fixed;}
  th,td{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top;}
  #report-tbody td,thead th{width:13%;min-width:13%;max-width:13%;white-space:normal;word-break:break-word;vertical-align:top;}
  thead th:nth-child(1){width:10%;} thead th:nth-child(2){width:7%;}
  th{background:#e8eaf6;position:sticky;top:0;z-index:1;}
  tr:nth-child(even){background:#fafafa;}
  .badge-ok {background:#c8e6c9;color:#1b5e20;padding:1px 6px;border-radius:4px;font-size:11px;}
  .badge-err{background:#ffcdd2;color:#b71c1c;padding:1px 6px;border-radius:4px;font-size:11px;}
  small{font-size:11px;} tr.hidden{display:none;}
  .n-filter{display:flex;gap:6px;align-items:center;margin-top:10px;flex-wrap:wrap;font-size:12px;}
  .n-filter strong{color:#283593;}
  .n-filter label{display:flex;align-items:center;gap:3px;cursor:pointer;background:#fff;border:1px solid #c5cae9;border-radius:12px;padding:2px 10px;}
  .n-filter label:hover{background:#e8eaf6;}
"""

_JS = r"""
  var CT_TYPES=%%CT_TYPES%%;
  var CT_KEYS=CT_TYPES.map(function(t){return t.key;});
  var TOTAL=%%TOTAL%%;

  function _getChecked(){
    var c={};CT_KEYS.forEach(function(k){c[k]=[];});
    document.querySelectorAll('.cell-cb:checked').forEach(function(cb){c[cb.dataset.key].push(cb.dataset.state);});
    return c;
  }
  function _getCheckedN(){
    var ns=[];
    document.querySelectorAll('.n-cb:checked').forEach(function(cb){ns.push(parseInt(cb.value));});
    return ns;
  }
  function _updateCounts(){
    var rows=document.querySelectorAll('#report-tbody tr');
    var vis=0,rc={},st={available:0,error:0,absent:0};
    CT_KEYS.forEach(function(k){rc[k]=0;});
    rows.forEach(function(row){
      if(row.classList.contains('hidden'))return;
      vis++;
      CT_KEYS.forEach(function(k){
        var s=row.dataset[k]||'absent';
        if(s!=='absent')rc[k]++;
        if(st[s]!==undefined)st[s]++;
      });
    });
    document.getElementById('row-count').textContent='Showing '+vis+' of '+TOTAL+' subjects';
    CT_KEYS.forEach(function(k){var el=document.getElementById('frc-'+k);if(el)el.textContent=rc[k];});
    ['available','error','absent'].forEach(function(s){var el=document.getElementById('ftotal-'+s);if(el)el.textContent=st[s];});
    var tv=document.getElementById('ftotal-visible');if(tv)tv.textContent=vis;
  }
  function applyFilters(){
    var checked=_getChecked(),checkedN=_getCheckedN();
    document.querySelectorAll('#report-tbody tr').forEach(function(row){
      var hide=false;
      CT_KEYS.forEach(function(key){
        if(hide||checked[key].length===0)return;
        if(checked[key].indexOf(row.dataset[key]||'absent')===-1)hide=true;
      });
      if(!hide&&checkedN.length>0){
        if(checkedN.indexOf(parseInt(row.dataset.n||'0'))===-1)hide=true;
      }
      row.classList.toggle('hidden',hide);
    });
    _updateCounts();_updateEmptyState();
  }
  function _updateEmptyState(){
    var anyChecked=document.querySelector('.cell-cb:checked')!==null;
    var msg=document.getElementById('no-filter-msg');
    var tbl=document.getElementById('report-table');
    var ce=document.getElementById('row-count');
    if(!anyChecked){
      if(msg)msg.style.display='block';
      if(tbl)tbl.style.display='none';
      if(ce)ce.textContent='Showing 0 of '+TOTAL+' subjects';
    }else{
      if(msg)msg.style.display='none';
      if(tbl)tbl.style.display='';
    }
  }
  function toggleSelectAll(cb){
    document.querySelectorAll('.cell-cb').forEach(function(c){c.checked=cb.checked;});
    applyFilters();
  }
  function resetFilters(){
    document.querySelectorAll('.filter-panel input[type=checkbox]').forEach(function(cb){cb.checked=true;});
    document.querySelectorAll('#report-tbody tr').forEach(function(row){row.classList.remove('hidden');});
    _updateCounts();_updateEmptyState();
  }
  function initCounts(){
    _updateCounts();
    var rows=document.querySelectorAll('#report-tbody tr');
    var bins=[0,0,0,0];
    rows.forEach(function(row){
      var n=parseInt(row.dataset.n||'0');
      bins[Math.min(n,3)]++;
    });
    var total=rows.length;
    var bar=document.getElementById('sa-bar');
    var cnt=document.getElementById('sa-counts');
    if(bar&&total>0){
      bar.innerHTML='<div class="stacked-bar">'+
        ['bar-0','bar-1','bar-2','bar-3'].map(function(cls,i){
          return bins[i]?'<div class="'+cls+'" style="width:'+(bins[i]/total*100).toFixed(1)+'%" title="'+bins[i]+' subjects"></div>':'';
        }).join('')+'</div>';
    }
    if(cnt){
      cnt.innerHTML='<div class="sa-row">'+
        ['0 CT types','1 CT type','2 CT types','3+ CT types'].map(function(lbl,i){
          return '<div class="sa-stat"><span class="sa-num">'+bins[i]+'</span><span class="sa-lbl">'+lbl+'</span></div>';
        }).join('')+'</div>';
    }
    _updateEmptyState();
  }
  window.onload=initCounts;
"""


def generate_html(
    subjects: list[tuple[str, dict]],
    root: Path,
    out_path: Path,
    timestamp: str,
) -> None:
    n_total = len(subjects)

    # ── Per-type counts ───────────────────────────────────────────────────────
    counts = {key: {"available": 0, "error": 0, "absent": 0} for key in CT_KEYS}
    for _, ct in subjects:
        for key in CT_KEYS:
            counts[key][ct[key]["status"]] += 1

    # ── Cards ─────────────────────────────────────────────────────────────────
    cards_html = ""
    for key, label, _, _ in CT_TYPES:
        avail  = counts[key]["available"]
        err    = counts[key]["error"]
        absent = counts[key]["absent"]
        cards_html += (
            f'<div class="card"><b>{label}</b><br>'
            f'<span class="badge-ok">{avail} available</span><br>'
        )
        if err:
            cards_html += f'<span class="badge-err">{err} errors</span><br>'
        cards_html += f'<span style="color:#999;font-size:12px">{absent} absent</span></div>'

    # ── Filter table rows ─────────────────────────────────────────────────────
    filter_rows = (
        '<tr class="ftotal-row">'
        '<td class="ftype-label ftotal-label">TOTAL</td>'
        '<td class="fc-available ftotal-cell" id="ftotal-available">—</td>'
        '<td class="fc-error     ftotal-cell" id="ftotal-error">—</td>'
        '<td class="fc-absent    ftotal-cell" id="ftotal-absent">—</td>'
        '<td class="frow-count   ftotal-cell" id="ftotal-visible">—</td>'
        '</tr>\n'
    )
    for key, label, _, _ in CT_TYPES:
        filter_rows += (
            f'<tr><td class="ftype-label">{label}</td>'
            f'<td class="fc-available"><input type="checkbox" class="cell-cb" data-key="{key}" data-state="available" onchange="applyFilters()" checked></td>'
            f'<td class="fc-error"    ><input type="checkbox" class="cell-cb" data-key="{key}" data-state="error"     onchange="applyFilters()" checked></td>'
            f'<td class="fc-absent"   ><input type="checkbox" class="cell-cb" data-key="{key}" data-state="absent"    onchange="applyFilters()" checked></td>'
            f'<td class="frow-count" id="frc-{key}">—</td></tr>\n'
        )

    # ── n_ct_types filter ─────────────────────────────────────────────────────
    n_vals = sorted({
        sum(1 for k in CT_KEYS if ct[k]["status"] == "available")
        for _, ct in subjects
    })
    n_filter_html = '<div class="n-filter"><strong>Filter by # CT types per subject:</strong>'
    for v in n_vals:
        cnt_v = sum(1 for _, ct in subjects
                    if sum(1 for k in CT_KEYS if ct[k]["status"] == "available") == v)
        n_filter_html += (
            f'<label><input type="checkbox" class="n-cb" value="{v}" '
            f'onchange="applyFilters()" checked> {v} &nbsp;<span style="color:#999">({cnt_v})</span></label>'
        )
    n_filter_html += "</div>"

    # ── Table rows ────────────────────────────────────────────────────────────
    tbody = ""
    for sub_id, ct in subjects:
        n_ct  = sum(1 for k in CT_KEYS if ct[k]["status"] == "available")
        attrs = f'data-n="{n_ct}"' + "".join(
            f' data-{key}="{ct[key]["status"]}"' for key in CT_KEYS
        )
        cells = "".join(f"<td>{_cell(ct[key])}</td>" for key, *_ in CT_TYPES)
        tbody += f'<tr {attrs}><td><b>{escape(sub_id)}</b></td><td>{n_ct}</td>{cells}</tr>\n'

    # ── JS data injection ─────────────────────────────────────────────────────
    ct_types_json = (
        "[" + ",".join(f'{{"key":"{k}","label":"{lbl}"}}' for k, lbl, *_ in CT_TYPES) + "]"
    )
    js = _JS.replace("%%CT_TYPES%%", ct_types_json).replace("%%TOTAL%%", str(n_total))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cohort Manifest — SLAAOBIDS</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Cohort Manifest — SLAAOBIDS</h1>
<div class="meta">
  Generated: {timestamp} &nbsp;|&nbsp;
  Subjects: {n_total} &nbsp;|&nbsp;
  Folder: <code>{escape(str(root))}</code>
</div>

<div class="cards">{cards_html}</div>

<div class="summary-panel">
  <div class="summary-section">
    <h3>Subjects by number of available CT types</h3>
    <div id="sa-bar"></div>
    <div id="sa-counts"></div>
    <div class="bar-legend">
      <span><i class="leg-dot" style="background:#e0e0e0"></i>0 CT types</span>
      <span><i class="leg-dot" style="background:#81c784"></i>1 CT type</span>
      <span><i class="leg-dot" style="background:#4fc3f7"></i>2 CT types</span>
      <span><i class="leg-dot" style="background:#7986cb"></i>3+ CT types</span>
    </div>
  </div>
</div>

<div class="filter-panel">
  <div class="filter-header">
    <span class="filter-title">Filter subjects</span>
    <label style="font-size:13px;cursor:pointer;">
      <input type="checkbox" id="select-all-cb" onchange="toggleSelectAll(this)" checked> Select all
    </label>
    <button class="reset-btn" onclick="resetFilters()">Reset all filters</button>
    <span id="row-count">Showing {n_total} of {n_total} subjects</span>
  </div>
  <table class="filter-table">
    <thead><tr>
      <th class="ftype-hdr">CT type</th>
      <th class="fstate-hdr fc-available">available</th>
      <th class="fstate-hdr fc-error">error</th>
      <th class="fstate-hdr fc-absent">absent</th>
      <th class="ftype-hdr" style="text-align:right">Visible</th>
    </tr></thead>
    <tbody>{filter_rows}</tbody>
  </table>
  {n_filter_html}
</div>

<div id="no-filter-msg" style="display:none;padding:20px 0;color:#888;font-style:italic;">
  No filters active — select at least one state to display subjects, or click Reset.
</div>

<table id="report-table">
<thead><tr>
  <th>Subject</th><th># types</th>
  <th>eCTA</th><th>CT_heart</th><th>CT_thorax</th><th>CT_totalbody</th><th>CT_abdomen</th>
</tr></thead>
<tbody id="report-tbody">
{tbody}
</tbody>
</table>

<script>{js}</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate cohort manifest CSV + HTML for a SLAAOBIDS folder"
    )
    p.add_argument(
        "--root",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS - Copy",
        help="SLAAOBIDS root directory to scan",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: same as --root)",
    )
    return p.parse_args()


def main() -> int:
    args    = _parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root

    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sub_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")),
        key=lambda p: p.name,
    )
    print(f"Scanning {len(sub_dirs)} subject folders ...")

    it = tqdm(sub_dirs, desc="Scanning", unit="sub") if tqdm else sub_dirs
    subjects: list[tuple[str, dict]] = []
    for sub_dir in it:
        subjects.append((sub_dir.name, scan_subject(sub_dir)))

    print(f"Done scanning {len(subjects)} subjects.")

    csv_path  = out_dir / "cohort_manifest.csv"
    html_path = out_dir / "cohort_report.html"

    write_csv(subjects, csv_path)
    print(f"CSV  → {csv_path}")

    generate_html(subjects, root, html_path, timestamp)
    print(f"HTML → {html_path}")

    print()
    for key, label, _, _ in CT_TYPES:
        avail = sum(1 for _, d in subjects if d[key]["status"] == "available")
        print(f"  {label:16s}: {avail:4d} available")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
