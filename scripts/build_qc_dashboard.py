#!/usr/bin/env python3
"""
Build QC Dashboard for SLAAO LA/LAA Pipeline.

Scans BIDS root + derivatives, then writes a self-contained HTML dashboard to:
    derivatives/qc_report/dashboard.html

Usage (single-line PowerShell):
    conda run -n cardiac-ct-explorer python scripts/build_qc_dashboard.py

Re-run any time to refresh the dashboard with the latest data.
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BIDS_ROOT   = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
DERIVATIVES = BIDS_ROOT / "derivatives"
DEFACED_DIR = DERIVATIVES / "defaced"
SEG_ECTA    = DERIVATIVES / "nudf_la_eCTA"
SEG_MC      = DERIVATIVES / "nudf_la_multict"
OUT_DIR     = DERIVATIVES / "qc_report"
OUT_HTML    = OUT_DIR / "dashboard.html"

SEG_LOG_CSV     = DERIVATIVES / "seg_summary_full.csv"
RAD_CSV         = DERIVATIVES / "radiomics" / "radiomics_ibsi_all.csv"
CONV_REPORT_DIR = BIDS_ROOT / "conversion Report"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_conv_manifests(conv_dir: Path) -> dict:
    """Merge all conversion_manifest_*.csv, keeping latest timestamp per subject_id.
    Picks up new manifest files automatically — no script changes needed."""
    merged: dict = {}
    manifests = sorted(conv_dir.glob("conversion_manifest_*.csv"))
    if not manifests:
        print(f"  [warn] No conversion_manifest_*.csv found in: {conv_dir}")
        return merged
    print(f"  Found {len(manifests)} manifest file(s)")
    for path in manifests:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("subject_id", "").strip()
                if not sid:
                    continue
                ts = row.get("timestamp", "")
                if sid not in merged or ts > merged[sid].get("timestamp", ""):
                    merged[sid] = row
    return merged


def load_csv_keyed(path: Path, key_col: str) -> dict:
    """Load a CSV into a dict keyed by one column."""
    result = {}
    if not path.exists():
        print(f"  [warn] CSV not found: {path.name}")
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row[key_col]] = row
    return result


def scan_seg_dir(seg_dir: Path) -> dict:
    """Return {case_id: {laa, la, ao}} via a single glob (avoids 800+ iterdir calls)."""
    result = {}
    if not seg_dir.exists():
        return result
    all_files = list(seg_dir.glob("*/*.nii*"))
    for f in tqdm(all_files, desc=f"  {seg_dir.name}", unit="file", dynamic_ncols=True, leave=False):
        case_id = f.parent.name
        if case_id not in result:
            result[case_id] = {"laa": False, "la": False, "ao": False}
        n = f.name
        if "laa_vista3d"         in n: result[case_id]["laa"] = True
        elif "left_atrium_highres" in n: result[case_id]["la"]  = True
        elif "aorta_highres_ts"    in n: result[case_id]["ao"]  = True
    return result


def traffic_light(vals: list) -> str:
    """Green / orange / red / gray from a list of booleans."""
    if not vals:
        return "gray"
    if all(vals):
        return "green"
    if any(vals):
        return "orange"
    return "red"


def safe_round(val, digits=1):
    try:
        return round(float(val), digits)
    except (ValueError, TypeError):
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CSV logs ──────────────────────────────────────────────────────────
    print("Loading seg log CSV ...")
    seg_log = load_csv_keyed(SEG_LOG_CSV, "case_id")
    print(f"  → {len(seg_log)} seg log rows")

    print("Loading radiomics CSV (may take ~10 s for large file) ...")
    rad_data = load_csv_keyed(RAD_CSV, "sub_id") if RAD_CSV.exists() else {}
    print(f"  → {len(rad_data)} radiomics rows")

    # ── Scan derivative directories ────────────────────────────────────────────
    print("Scanning defaced/ ...")
    defaced_set: set[str] = set()
    if DEFACED_DIR.exists():
        for f in DEFACED_DIR.iterdir():
            m = re.match(r"^(sub-\d+_acq-ecta_ct)_defaced\.nii\.gz$", f.name)
            if m:
                defaced_set.add(m.group(1))

    print("Scanning nudf_la_eCTA/ ...")
    seg_ecta = scan_seg_dir(SEG_ECTA)

    print("Scanning nudf_la_multict/ ...")
    seg_mc = scan_seg_dir(SEG_MC)

    # ── Pre-scan ALL BIDS source NIfTI files in one pass ──────────────────────
    print("Pre-scanning BIDS NIfTI files (single pass) ...")
    file_re = re.compile(r"^sub-\d+_acq-(\w+?)(?:_ph(\d+))?_ct\.nii\.gz$")
    bids_nii: dict[str, list] = {}   # sub_id → [Path, ...]
    raw_bids = list(BIDS_ROOT.glob("sub-*/sub-*_acq-*_ct.nii.gz"))
    for f in tqdm(raw_bids, desc="  BIDS files", unit="file", dynamic_ncols=True, leave=False):
        bids_nii.setdefault(f.parent.name, []).append(f)

    # ── Enumerate subjects ─────────────────────────────────────────────────────
    sub_re = re.compile(r"^sub-(\d+)$")
    all_subs = sorted(
        [d.name for d in BIDS_ROOT.iterdir() if d.is_dir() and sub_re.match(d.name)],
        key=lambda s: int(s.split("-")[1]),
    )
    print(f"Found {len(all_subs)} subjects ({len(raw_bids)} NIfTI files)")

    # ── Build per-subject records ──────────────────────────────────────────────
    records = []

    for sub_id in tqdm(all_subs, desc="Processing subjects", unit="sub", dynamic_ncols=True):
        nii_files = bids_nii.get(sub_id, [])

        # Parse source files
        source_by_acq: dict[str, list] = {}
        for f in nii_files:
            m = file_re.match(f.name)
            if not m:
                continue
            acq, ph = m.group(1), m.group(2)
            source_by_acq.setdefault(acq, []).append({"file": f.name, "phase": ph})

        is_ecta   = "ecta" in source_by_acq
        ecta_case = f"{sub_id}_acq-ecta_ct" if is_ecta else None
        defaced   = (ecta_case in defaced_set) if ecta_case else False
        eseg      = seg_ecta.get(ecta_case, {})
        elog      = seg_log.get(ecta_case, {})

        # Multi-CT rows
        mc_rows = []
        for acq, scans in source_by_acq.items():
            if acq == "ecta":
                continue
            for s in scans:
                ph = s["phase"]
                case_id = (
                    f"{sub_id}_acq-{acq}_ph{ph}_ct"
                    if ph is not None
                    else f"{sub_id}_acq-{acq}_ct"
                )
                ms  = seg_mc.get(case_id, {})
                log = seg_log.get(case_id, {})
                mc_rows.append({
                    "case_id":    case_id,
                    "acq":        acq,
                    "phase":      ph or "",
                    "laa":        ms.get("laa", False),
                    "la":         ms.get("la", False),
                    "ao":         ms.get("ao", False),
                    "seg_status": log.get("status", ""),
                })

        # Radiomics
        rad = rad_data.get(sub_id, {})

        # Traffic lights
        laa_all = ([eseg.get("laa", False)] if ecta_case else []) + [r["laa"] for r in mc_rows]
        la_all  = ([eseg.get("la",  False)] if ecta_case else []) + [r["la"]  for r in mc_rows]
        ao_all  = ([eseg.get("ao",  False)] if ecta_case else []) + [r["ao"]  for r in mc_rows]

        # LA FOV skip is expected → orange not red
        la_tl = traffic_light(la_all) if la_all else "gray"
        if la_tl == "red" and elog.get("ts_status") == "skip_la_fov":
            la_tl = "orange"

        records.append({
            "sub_id":       sub_id,
            "ct_types":     sorted(source_by_acq.keys()),
            "source_count": len(nii_files),
            "is_ecta":      is_ecta,
            "defaced":      defaced,
            "ecta_seg":     eseg,
            "mc_rows":      mc_rows,
            "rad_exists":   bool(rad),
            "hu": {
                "laa_med": safe_round(rad.get("laa_original_firstorder_Median")),
                "laa_iqr": safe_round(rad.get("laa_original_firstorder_InterquartileRange")),
                "la_med":  safe_round(rad.get("la_original_firstorder_Median")),
                "la_iqr":  safe_round(rad.get("la_original_firstorder_InterquartileRange")),
                "ao_med":  safe_round(rad.get("aorta_original_firstorder_Median")),
                "ao_iqr":  safe_round(rad.get("aorta_original_firstorder_InterquartileRange")),
            },
            "laa_status": rad.get("laa_status", ""),
            "seg_log": {
                "status":     elog.get("status", ""),
                "ts_status":  elog.get("ts_status", ""),
                "v3d_status": elog.get("v3d_status", ""),
                "message":    elog.get("message", ""),
                "la_voxels":  elog.get("la_voxels", ""),
            },
            "tl": {
                "conv":   "green" if nii_files else "red",
                "deface": "green" if defaced else ("gray" if not is_ecta else "red"),
                "laa":    traffic_light(laa_all) if laa_all else "gray",
                "la":     la_tl,
                "ao":     traffic_light(ao_all)  if ao_all  else "gray",
                "rad":    "green" if bool(rad) else "red",
            },
        })

    # ── Inject subjects from conversion logs that have no folder on disk ───────
    print("Loading conversion manifests ...")
    conv_manifest = load_conv_manifests(CONV_REPORT_DIR)
    print(f"  → {len(conv_manifest)} unique subjects across all manifests")

    folder_sids = {r["sub_id"].replace("sub-", "") for r in records}
    CT_TYPES    = ("ecta", "ctheart", "ctthorax", "ctbody", "ctabdomen")
    n_injected  = 0

    for sid_str, row in conv_manifest.items():
        if sid_str in folder_sids:
            continue  # folder exists on disk — filesystem record takes priority

        ct_statuses = {
            ct: row.get(f"{ct}_status", "").strip()
            for ct in CT_TYPES
            if row.get(f"{ct}_status", "").strip()
        }
        records.append({
            "sub_id":       f"sub-{sid_str}",
            "ct_types":     [],
            "source_count": 0,
            "is_ecta":      False,
            "defaced":      False,
            "ecta_seg":     {},
            "mc_rows":      [],
            "rad_exists":   False,
            "hu":           {"laa_med": None, "laa_iqr": None,
                             "la_med":  None, "la_iqr":  None,
                             "ao_med":  None, "ao_iqr":  None},
            "laa_status":   "",
            "seg_log":      {"status": "", "ts_status": "", "v3d_status": "",
                             "message": "", "la_voxels": ""},
            "no_folder":    True,
            "conv_log": {
                "n_unclassified": row.get("n_unclassified", ""),
                "n_no_dicom":     row.get("n_no_dicom", ""),
                "ct_statuses":    ct_statuses,
                "dicom_dir":      row.get("dicom_dir", ""),
                "timestamp":      row.get("timestamp", ""),
            },
            "tl": {
                "conv":   "red",
                "deface": "gray",
                "laa":    "gray",
                "la":     "gray",
                "ao":     "gray",
                "rad":    "red",
            },
        })
        n_injected += 1

    if n_injected:
        print(f"  → Added {n_injected} subjects from logs with no folder on disk")
        records.sort(key=lambda r: int(r["sub_id"].split("-")[1]))

    # ── Summary counts ─────────────────────────────────────────────────────────
    n_total    = len(records)
    n_nofolder = sum(1 for r in records if r.get("no_folder"))
    n_conv  = sum(1 for r in records if r["tl"]["conv"] == "green")
    n_ecta  = sum(1 for r in records if r["is_ecta"])
    n_def   = sum(1 for r in records if r["defaced"])
    n_rad   = sum(1 for r in records if r["rad_exists"])

    all_seg = {**seg_ecta, **seg_mc}
    n_seg   = len(all_seg)
    n_laa   = sum(1 for v in all_seg.values() if v.get("laa"))
    n_la    = sum(1 for v in all_seg.values() if v.get("la"))
    n_ao    = sum(1 for v in all_seg.values() if v.get("ao"))

    counts = dict(
        total=n_total, converted=n_conv, ecta=n_ecta,
        defaced=n_def, seg_cases=n_seg,
        laa=n_laa, la=n_la, ao=n_ao, radiomics=n_rad,
        nofolder=n_nofolder,
    )
    prog = {
        "conversion": round(100 * n_conv / n_total)   if n_total else 0,
        "defacing":   round(100 * n_def  / n_ecta)    if n_ecta  else 0,
        "seg_laa":    round(100 * n_laa  / n_seg)     if n_seg   else 0,
        "seg_la":     round(100 * n_la   / n_seg)     if n_seg   else 0,
        "seg_ao":     round(100 * n_ao   / n_seg)     if n_seg   else 0,
        "radiomics":  round(100 * n_rad  / n_ecta)    if n_ecta  else 0,
        "shape":      0,
    }

    # ── Archive previous dashboard ─────────────────────────────────────────────
    if OUT_HTML.exists():
        archive_dir = OUT_DIR / "previous dashboard version"
        if not archive_dir.exists():
            archive_dir.mkdir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"dashboard_{ts}.html"
        OUT_HTML.rename(archive_path)
        print(f"Archived previous dashboard → {archive_path.name}")

    # ── Generate HTML ──────────────────────────────────────────────────────────
    print("Generating HTML dashboard...")
    html = build_html(records, counts, prog)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size // 1024
    print(f"\nSaved ({size_kb} KB): {OUT_HTML}")
    print(f"Open:  file:///{OUT_HTML.as_posix()}")


# ── HTML Builder ───────────────────────────────────────────────────────────────

def build_html(records: list, counts: dict, prog: dict) -> str:
    gen_date  = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_js   = json.dumps(records,   ensure_ascii=False, separators=(",", ":"))
    counts_js = json.dumps(counts)
    prog_js   = json.dumps(prog)

    # Use placeholder substitution to avoid Python f-string brace escaping in JS/CSS
    tpl = HTML_TEMPLATE
    tpl = tpl.replace("__DATA__",     data_js)
    tpl = tpl.replace("__COUNTS__",   counts_js)
    tpl = tpl.replace("__PROG__",     prog_js)
    tpl = tpl.replace("__GEN_DATE__", gen_date)
    return tpl


# ── HTML Template ──────────────────────────────────────────────────────────────
# Placeholders: __DATA__  __COUNTS__  __PROG__  __GEN_DATE__

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SLAAO QC Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
<style>
body { font-size: .875rem; }
#rev-bar { background:#f8f9fa; border-bottom:1px solid #dee2e6; padding:5px 14px; font-size:.8rem; position:sticky; top:0; z-index:100; }
.tl { display:inline-block; width:13px; height:13px; border-radius:50%; vertical-align:middle; }
.tl-green  { background:#198754; }
.tl-orange { background:#fd7e14; }
.tl-red    { background:#dc3545; }
.tl-gray   { background:#adb5bd; }
.badge-sm  { font-size:.68rem; padding:2px 5px; }
th { white-space:nowrap; font-size:.8rem; }
td { vertical-align:middle !important; font-size:.8rem; }
.qc-status  { width:105px; font-size:.75rem; padding:1px 4px; }
.qc-comment { width:170px; font-size:.75rem; padding:1px 4px; }
.progress { height:15px; }
.prog-row td { padding:3px 8px; }
</style>
</head>
<body>

<!-- Reviewer bar -->
<div id="rev-bar">
  Reviewer: <strong id="rev-name">—</strong>
  <button class="btn btn-sm btn-outline-secondary py-0 ms-2" onclick="changeReviewer()">Change</button>
  <span class="ms-4 text-muted">Generated: __GEN_DATE__</span>
</div>

<div class="container-fluid px-3 pt-2">
<h5 class="mb-2">SLAAO — LA/LAA Pipeline QC Dashboard</h5>

<!-- Tab nav -->
<ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist">
  <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab1">Overview</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab2">Conversion</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab3">Defacing</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab4">Segmentation</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab5">Radiomics</button></li>
</ul>

<div class="tab-content">

<!-- ══════════════════════════════════════════════════════
     TAB 1 — OVERVIEW
══════════════════════════════════════════════════════ -->
<div id="tab1" class="tab-pane fade show active">

  <!-- Count cards -->
  <div class="d-flex flex-wrap gap-2 mb-3">
    <div class="card text-center px-3 py-2">
      <div class="fs-3 fw-bold" id="ct-total">—</div>
      <div class="text-muted" style="font-size:.72rem">Total subjects</div>
    </div>
    <div class="card text-center px-3 py-2">
      <div class="fs-3 fw-bold text-primary" id="ct-ecta">—</div>
      <div class="text-muted" style="font-size:.72rem">eCTA valid for seg</div>
      <div id="ct-ecta-note" style="display:none;font-size:.65rem;color:#dc3545"></div>
    </div>
    <div class="card text-center px-3 py-2">
      <div class="fs-3 fw-bold text-warning" id="ct-segcases">—</div>
      <div class="text-muted" style="font-size:.72rem">Seg cases total</div>
    </div>
    <div class="card text-center px-3 py-2">
      <div class="fs-3 fw-bold text-success" id="ct-rad">—</div>
      <div class="text-muted" style="font-size:.72rem">Radiomics done</div>
    </div>
    <div class="card text-center px-3 py-2" id="card-nofolder" style="display:none">
      <div class="fs-3 fw-bold text-danger" id="ct-nofolder">—</div>
      <div class="text-muted" style="font-size:.72rem">No folder (log only)</div>
    </div>
  </div>

  <!-- Progress bars -->
  <div class="mb-3" style="max-width:620px">
    <table class="table table-sm table-borderless mb-0">
      <tbody id="prog-body"></tbody>
    </table>
  </div>

  <!-- Legend -->
  <div class="mb-2 small">
    <span class="tl tl-green me-1"></span>Complete &nbsp;
    <span class="tl tl-orange me-1"></span>Partial / expected skip &nbsp;
    <span class="tl tl-red me-1"></span>Missing &nbsp;
    <span class="tl tl-gray me-1"></span>N/A
  </div>

  <!-- Conversion filter buttons -->
  <div class="mb-2 d-flex align-items-center flex-wrap gap-1">
    <span class="small text-muted me-1">Filter by conversion:</span>
    <button id="flt-all"    class="btn btn-sm btn-secondary"       onclick="filterOverview('all')">All</button>
    <button id="flt-green"  class="btn btn-sm btn-outline-success"  onclick="filterOverview('green')">&#x1F7E2; Converted</button>
    <button id="flt-orange" class="btn btn-sm btn-outline-warning"  onclick="filterOverview('orange')">&#x1F7E1; Partial</button>
    <button id="flt-red"    class="btn btn-sm btn-outline-danger"   onclick="filterOverview('red')">&#x1F534; Failed / No folder</button>
  </div>

  <!-- Summary table -->
  <table id="tbl-overview" class="table table-sm table-hover table-bordered" style="width:100%">
    <thead class="table-dark">
      <tr>
        <th>Subject</th><th>CT types</th>
        <th title="Has NIfTI source files">Conv</th>
        <th title="Defaced file exists (eCTA only)">Deface</th>
        <th title="LAA mask exists">LAA</th>
        <th title="LA mask exists">LA</th>
        <th title="Aorta mask exists">Ao</th>
        <th title="Radiomics row present">Rad</th>
      </tr>
    </thead>
    <tbody id="body-overview"></tbody>
  </table>
</div>

<!-- ══════════════════════════════════════════════════════
     TAB 2 — CONVERSION
══════════════════════════════════════════════════════ -->
<div id="tab2" class="tab-pane fade">
  <div class="mb-2">
    <button class="btn btn-sm btn-success" onclick="saveQC('conv')">&#128190; Save QC CSV</button>
    <span class="text-muted ms-2 small" id="conv-msg"></span>
  </div>
  <table id="tbl-conv" class="table table-sm table-hover table-bordered" style="width:100%">
    <thead class="table-dark">
      <tr>
        <th>Subject</th><th>CT types</th><th>NIfTI files</th>
        <th>Status</th>
        <th title="Revised">&#10003;</th><th>QC Status</th><th>Comments</th>
      </tr>
    </thead>
    <tbody id="body-conv"></tbody>
  </table>
</div>

<!-- ══════════════════════════════════════════════════════
     TAB 3 — DEFACING
══════════════════════════════════════════════════════ -->
<div id="tab3" class="tab-pane fade">
  <div class="mb-2">
    <button class="btn btn-sm btn-success" onclick="saveQC('deface')">&#128190; Save QC CSV</button>
    <span class="text-muted ms-2 small" id="deface-msg"></span>
  </div>
  <p class="small text-muted mb-2">Defacing applies to eCTA only. Non-eCTA subjects shown as N/A.</p>
  <table id="tbl-deface" class="table table-sm table-hover table-bordered" style="width:100%">
    <thead class="table-dark">
      <tr>
        <th>Subject</th><th>eCTA</th><th>Defaced file</th>
        <th>QC Status</th><th>Comments</th>
      </tr>
    </thead>
    <tbody id="body-deface"></tbody>
  </table>
</div>

<!-- ══════════════════════════════════════════════════════
     TAB 4 — SEGMENTATION
══════════════════════════════════════════════════════ -->
<div id="tab4" class="tab-pane fade">
  <div class="mb-2">
    <button class="btn btn-sm btn-success" onclick="saveQC('seg')">&#128190; Save QC CSV</button>
    <span class="text-muted ms-2 small" id="seg-msg"></span>
  </div>
  <table id="tbl-seg" class="table table-sm table-hover table-bordered" style="width:100%">
    <thead class="table-dark">
      <tr>
        <th>Subject</th><th>Case ID</th><th>CT type</th><th>Phase</th>
        <th title="Overall QC — green: all segs ok or fixed; orange: pending; red: not fixable or missing segs">Overall</th>
        <th>LAA</th><th>LA</th><th>Ao</th><th>Log status</th>
        <th title="Revised">&#10003;</th><th>QC Status</th><th>Comments</th>
      </tr>
    </thead>
    <tbody id="body-seg"></tbody>
  </table>
</div>

<!-- ══════════════════════════════════════════════════════
     TAB 5 — RADIOMICS
══════════════════════════════════════════════════════ -->
<div id="tab5" class="tab-pane fade">
  <div class="mb-2">
    <button class="btn btn-sm btn-success" onclick="saveQC('rad')">&#128190; Save QC CSV</button>
    <span class="text-muted ms-2 small" id="rad-msg"></span>
  </div>
  <table id="tbl-rad" class="table table-sm table-hover table-bordered" style="width:100%">
    <thead class="table-dark">
      <tr>
        <th>Subject</th><th>Radiomics</th>
        <th title="Segmentation Overall status — red means at least one seg is missing or not fixable">Seg Overall</th>
        <th>LAA median HU (IQR)</th><th>LA median HU (IQR)</th><th>Ao median HU (IQR)</th>
        <th>LAA status</th>
        <th title="Revised">&#10003;</th><th>QC Status</th><th>Comments</th>
      </tr>
    </thead>
    <tbody id="body-rad"></tbody>
  </table>
</div>

</div><!-- tab-content -->
</div><!-- container -->

<!-- ── Scripts ─────────────────────────────────────────────────────────────── -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

<script>
// ════════════════════════════════════════════════════════
// Embedded data
// ════════════════════════════════════════════════════════
const SUBJECTS = __DATA__;
const COUNTS   = __COUNTS__;
const PROG     = __PROG__;

// ════════════════════════════════════════════════════════
// Reviewer
// ════════════════════════════════════════════════════════
let reviewer = '';
function initReviewer() {
  reviewer = localStorage.getItem('qc_reviewer') || '';
  if (!reviewer) {
    reviewer = (prompt('Enter your reviewer name (used for CSV filenames):') || 'unknown').trim();
    localStorage.setItem('qc_reviewer', reviewer);
  }
  document.getElementById('rev-name').textContent = reviewer;
}
function changeReviewer() {
  reviewer = (prompt('Reviewer name:', reviewer) || reviewer).trim();
  localStorage.setItem('qc_reviewer', reviewer);
  document.getElementById('rev-name').textContent = reviewer;
}

// ════════════════════════════════════════════════════════
// DOM helpers
// ════════════════════════════════════════════════════════
function dot(status) {
  return `<span class="tl tl-${status}" title="${status}"></span>`;
}
function yn(val) {
  return val
    ? '<span class="badge bg-success badge-sm">Y</span>'
    : '<span class="badge bg-danger badge-sm">N</span>';
}
function hu(med, iqr) {
  if (med == null) return '<span class="text-muted">—</span>';
  const q = iqr != null ? iqr : '?';
  return `${med} <small class="text-muted">(${q})</small>`;
}

// ════════════════════════════════════════════════════════
// QC localStorage
// ════════════════════════════════════════════════════════
function qcGet(tab, key) {
  const raw = localStorage.getItem(`qc_${tab}_${key}`);
  return raw ? JSON.parse(raw) : { revised: false, status: '', comments: '' };
}
function qcSet(tab, key, field, val) {
  const d = qcGet(tab, key);
  d[field] = val;
  localStorage.setItem(`qc_${tab}_${key}`, JSON.stringify(d));
}
function qcCells(tab, key) {
  const d = qcGet(tab, key);
  const esc = key.replace(/'/g, "\\'");
  return `
    <td class="text-center">
      <input type="checkbox" ${d.revised ? 'checked' : ''}
        onchange="qcSet('${tab}','${esc}','revised',this.checked)">
    </td>
    <td>
      <select class="form-select form-select-sm qc-status"
          onchange="qcSet('${tab}','${esc}','status',this.value)">
        <option value="">— pending —</option>
        <option value="pending"     ${d.status==='pending'    ?'selected':''}>Pending</option>
        <option value="fixed"       ${d.status==='fixed'      ?'selected':''}>Fixed</option>
        <option value="not_fixable" ${d.status==='not_fixable'?'selected':''}>Not fixable</option>
      </select>
    </td>
    <td>
      <input type="text" class="form-control form-control-sm qc-comment"
        value="${(d.comments||'').replace(/"/g,'&quot;')}"
        placeholder="comments…"
        onchange="qcSet('${tab}','${esc}','comments',this.value)">
    </td>`;
}
// ── Defacing helpers ─────────────────────────────────────────────────────────
// Renders the "Defaced file" badge: "Not valid" (orange) overrides Y/N when
// QC status is "not_fixable" (poor imaging quality, not a defacing problem).
function defaceBadge(qcStatus, defaced) {
  if (qcStatus === 'not_fixable')
    return '<span class="badge bg-warning badge-sm text-dark" title="File present but not valid for segmentation (poor imaging quality)">Not valid</span>';
  return yn(defaced);
}
// Called when the QC status dropdown changes in the deface tab — refreshes
// the "Defaced file" cell in the same row without re-rendering the whole table.
function refreshDefCell(sub_id, is_ecta, defaced) {
  const cell = document.getElementById('defcell-' + sub_id.replace('-',''));
  if (!cell) return;
  if (!is_ecta) { cell.innerHTML = '<span class="text-muted">N/A</span>'; return; }
  cell.innerHTML = defaceBadge(qcGet('deface', sub_id).status, defaced);
}
// QC cells for the defacing tab: no revised checkbox, but status change
// immediately refreshes the "Defaced file" badge in the same row.
function defaceQcCells(key, is_ecta, defaced) {
  const d = qcGet('deface', key);
  const esc = key.replace(/'/g, "\\'");
  return `
    <td>
      <select class="form-select form-select-sm qc-status"
          onchange="qcSet('deface','${esc}','status',this.value);refreshDefCell('${esc}',${is_ecta},${defaced})">
        <option value="">— pending —</option>
        <option value="pending"     ${d.status==='pending'    ?'selected':''}>Pending</option>
        <option value="fixed"       ${d.status==='fixed'      ?'selected':''}>Fixed</option>
        <option value="not_fixable" ${d.status==='not_fixable'?'selected':''}>Not fixable</option>
      </select>
    </td>
    <td>
      <input type="text" class="form-control form-control-sm qc-comment"
        value="${(d.comments||'').replace(/"/g,'&quot;')}"
        placeholder="comments…"
        onchange="qcSet('deface','${esc}','comments',this.value)">
    </td>`;
}

// ── Segmentation Overall helper ──────────────────────────────────────────────
// QC status (when set) takes full priority over seg file evaluation.
// dot sort order: 0=red, 1=orange, 2=green — used in data-order for sorting.
function segOverallColor(caseId, laa, la, ao) {
  const st = qcGet('seg', caseId).status;
  if (st === 'fixed')       return { color: 'green',  order: 2 };
  if (st === 'pending')     return { color: 'orange', order: 1 };
  if (st === 'not_fixable') return { color: 'red',    order: 0 };
  // No QC status — fall back to seg file evaluation
  return (laa && la && ao)
    ? { color: 'green', order: 2 }
    : { color: 'red',   order: 0 };
}

// ════════════════════════════════════════════════════════
// CSV save — accumulates history in localStorage, then downloads
// ════════════════════════════════════════════════════════
function saveQC(tab) {
  const now = new Date().toISOString().replace('T',' ').slice(0,19);
  const histKey = `qc_history_${tab}`;
  const existing = JSON.parse(localStorage.getItem(histKey) || '[]');
  const newRows = [];

  if (tab === 'conv') {
    SUBJECTS.forEach(s => {
      const d = qcGet('conv', s.sub_id);
      newRows.push({ subject_id:s.sub_id, reviewer, timestamp:now,
                     revised:d.revised, status:d.status, comments:d.comments });
    });
  } else if (tab === 'deface') {
    SUBJECTS.filter(s => s.is_ecta).forEach(s => {
      const d = qcGet('deface', s.sub_id);
      newRows.push({ subject_id:s.sub_id, defaced_file:s.defaced, reviewer, timestamp:now,
                     status:d.status, comments:d.comments });
    });
  } else if (tab === 'seg') {
    SUBJECTS.forEach(s => {
      const cases = [];
      if (s.is_ecta) cases.push({ case_id:`${s.sub_id}_acq-ecta_ct`, ct:'ecta', phase:'' });
      s.mc_rows.forEach(r => cases.push({ case_id:r.case_id, ct:r.acq, phase:r.phase }));
      cases.forEach(c => {
        const d = qcGet('seg', c.case_id);
        newRows.push({ subject_id:s.sub_id, case_id:c.case_id, ct_type:c.ct, phase:c.phase,
                       reviewer, timestamp:now,
                       revised:d.revised, status:d.status, comments:d.comments });
      });
    });
  } else if (tab === 'rad') {
    SUBJECTS.forEach(s => {
      const d = qcGet('rad', s.sub_id);
      newRows.push({ subject_id:s.sub_id, reviewer, timestamp:now,
                     revised:d.revised, status:d.status, comments:d.comments });
    });
  }

  const all = [...existing, ...newRows];
  localStorage.setItem(histKey, JSON.stringify(all));

  const keys = Object.keys(all[0] || {});
  const csvLines = [
    keys.join(','),
    ...all.map(r => keys.map(k => `"${String(r[k]||'').replace(/"/g,'""')}"`).join(','))
  ];
  const blob = new Blob([csvLines.join('\n')], { type:'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `qc_${reviewer}_${tab}_${now.slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);

  const el = document.getElementById(`${tab}-msg`);
  if (el) el.textContent = `Saved ${all.length} rows · ${now}`;
}

// ════════════════════════════════════════════════════════
// TAB 1 — Overview
// ════════════════════════════════════════════════════════
function buildOverview() {
  // Subjects with eCTA marked not_fixable and no other exam type are excluded
  // from "valid for seg" counts — their eCTA is poor quality, no usable exam.
  const nNotValidEcta = SUBJECTS.filter(s =>
    s.is_ecta && s.mc_rows.length === 0 &&
    qcGet('deface', s.sub_id).status === 'not_fixable'
  ).length;
  const validEcta = COUNTS.ecta - nNotValidEcta;

  document.getElementById('ct-total').textContent    = COUNTS.total;
  document.getElementById('ct-ecta').textContent     = validEcta;
  document.getElementById('ct-segcases').textContent = COUNTS.seg_cases;
  document.getElementById('ct-rad').textContent      = COUNTS.radiomics;
  const noteEl = document.getElementById('ct-ecta-note');
  if (nNotValidEcta > 0) {
    noteEl.textContent = `(${nNotValidEcta} not valid excl.)`;
    noteEl.style.display = '';
  } else {
    noteEl.style.display = 'none';
  }
  if (COUNTS.nofolder > 0) {
    document.getElementById('ct-nofolder').textContent = COUNTS.nofolder;
    document.getElementById('card-nofolder').style.display = '';
  }

  // Defacing numerator: defaced file exists AND QC status ≠ not_fixable
  // Denominator: all eCTA files (COUNTS.ecta — Python-computed, file-based)
  const nDefacedValid = SUBJECTS.filter(s =>
    s.is_ecta && s.defaced && qcGet('deface', s.sub_id).status !== 'not_fixable'
  ).length;
  const defacingPct = COUNTS.ecta ? Math.round(100 * nDefacedValid / COUNTS.ecta) : 0;
  const radiomicsPct = validEcta ? Math.round(100 * COUNTS.radiomics / validEcta) : 0;
  // Segmentation Overall: unique patients with at least one green Overall case.
  // Reads QC status from localStorage — same logic as segOverallColor().
  const nSegOverall = SUBJECTS.filter(s => {
    if (s.no_folder) return false;
    const cases = [];
    if (s.is_ecta) cases.push({ id: `${s.sub_id}_acq-ecta_ct`, laa: s.ecta_seg.laa, la: s.ecta_seg.la, ao: s.ecta_seg.ao });
    s.mc_rows.forEach(r => cases.push({ id: r.case_id, laa: r.laa, la: r.la, ao: r.ao }));
    return cases.some(c => segOverallColor(c.id, c.laa, c.la, c.ao).color === 'green');
  }).length;
  const segOverallPct = COUNTS.converted ? Math.round(100 * nSegOverall / COUNTS.converted) : 0;
  const bars = [
    ['DICOM → NIfTI conversion',  PROG.conversion, COUNTS.converted,  COUNTS.total],
    ['Defacing (eCTA only)',       defacingPct,     nDefacedValid,     COUNTS.ecta],
    ['Segmentation — LAA',        PROG.seg_laa,    COUNTS.laa,        COUNTS.seg_cases],
    ['Segmentation — LA',         PROG.seg_la,     COUNTS.la,         COUNTS.seg_cases],
    ['Segmentation — Ao',         PROG.seg_ao,     COUNTS.ao,         COUNTS.seg_cases],
    ['Segmentation — Overall',    segOverallPct,   nSegOverall,       COUNTS.converted],
    ['Radiomics',                 radiomicsPct,    COUNTS.radiomics,  validEcta],
    ['Shape metrics',             PROG.shape,      0,                 validEcta],
  ];
  let pbHtml = '';
  bars.forEach(([label, pct, done, tot]) => {
    const col = pct >= 90 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger';
    pbHtml += `<tr class="prog-row">
      <td style="width:210px">${label}</td>
      <td style="width:220px">
        <div class="progress">
          <div class="progress-bar ${col}" style="width:${pct}%" role="progressbar">${pct}%</div>
        </div>
      </td>
      <td class="text-muted ps-2" style="font-size:.75rem">${done} / ${tot}</td>
    </tr>`;
  });
  document.getElementById('prog-body').innerHTML = pbHtml;

  let tbody = '';
  SUBJECTS.forEach(s => {
    const badges = s.ct_types.map(t =>
      `<span class="badge bg-secondary badge-sm me-1">${t}</span>`).join('');
    const dQc = qcGet('deface', s.sub_id);
    const defaceDot = (s.is_ecta && dQc.status === 'not_fixable') ? 'orange' : s.tl.deface;
    tbody += `<tr data-conv="${s.tl.conv}">
      <td data-order="${parseInt(s.sub_id.split('-')[1])}">${s.sub_id}</td>
      <td>${badges || '—'}</td>
      <td class="text-center">${dot(s.tl.conv)}</td>
      <td class="text-center">${dot(defaceDot)}</td>
      <td class="text-center">${dot(s.tl.laa)}</td>
      <td class="text-center">${dot(s.tl.la)}</td>
      <td class="text-center">${dot(s.tl.ao)}</td>
      <td class="text-center">${dot(s.tl.rad)}</td>
    </tr>`;
  });
  document.getElementById('body-overview').innerHTML = tbody;
  // Verify data-order on first subject cell
  const firstCell = document.querySelector('#tbl-overview tbody td:first-child');
  if (firstCell) console.log('[QC debug] overview first cell data-order =', firstCell.getAttribute('data-order'), '| text =', firstCell.textContent);
  if (!$.fn.DataTable.isDataTable('#tbl-overview')) {
    $('#tbl-overview').DataTable({ pageLength:25, order:[[0,'asc']],
      columnDefs:[{ orderable:false, targets:[2,3,4,5,6,7] }, { type:'num', targets:0 }] });
  }
}

// ════════════════════════════════════════════════════════
// TAB 2 — Conversion
// ════════════════════════════════════════════════════════
function buildConversion() {
  let tbody = '';
  SUBJECTS.forEach(s => {
    const subNum = parseInt(s.sub_id.split('-')[1]);

    if (s.no_folder) {
      // ── Subject exists in conversion log but has no folder on disk ──
      const cl = s.conv_log;
      const parts = [];
      if (parseInt(cl.n_unclassified) > 0) parts.push(`${cl.n_unclassified} unclassified`);
      if (parseInt(cl.n_no_dicom)     > 0) parts.push(`${cl.n_no_dicom} no DICOM`);
      Object.entries(cl.ct_statuses || {}).forEach(([ct, st]) => parts.push(`${ct}: ${st}`));
      const detail = parts.join(' · ') || 'no status recorded';
      tbody += `<tr class="table-danger">
        <td data-order="${subNum}">${s.sub_id}</td>
        <td><span class="text-muted small">${cl.dicom_dir || '—'}</span></td>
        <td data-order="-1"><span class="badge bg-danger badge-sm">no folder</span></td>
        <td><span class="badge bg-danger badge-sm">check conversion log</span>
            <br><small class="text-muted">${detail}</small></td>
        ${qcCells('conv', s.sub_id)}
      </tr>`;
      return;
    }

    // ── Normal subject with folder on disk ──
    const badges = s.ct_types.map(t =>
      `<span class="badge bg-secondary badge-sm me-1">${t}</span>`).join('') || '—';
    const niiCell = s.source_count > 0
      ? `<span class="badge bg-success badge-sm">${s.source_count} file(s)</span>`
      : `<span class="badge bg-danger badge-sm">None</span>`;
    const statusCell = s.tl.conv === 'green'
      ? '<span class="badge bg-success badge-sm">OK</span>'
      : '<span class="badge bg-danger badge-sm">Empty folder</span>';
    tbody += `<tr>
      <td data-order="${subNum}">${s.sub_id}</td><td>${badges}</td><td data-order="${s.source_count}">${niiCell}</td><td>${statusCell}</td>
      ${qcCells('conv', s.sub_id)}
    </tr>`;
  });
  document.getElementById('body-conv').innerHTML = tbody;
  if (!$.fn.DataTable.isDataTable('#tbl-conv'))
    $('#tbl-conv').DataTable({ pageLength:25, order:[[0,'asc']],
      columnDefs:[{ orderable:false, targets:[4,5,6] }, { type:'num', targets:0 }] });
}

// ════════════════════════════════════════════════════════
// TAB 3 — Defacing
// ════════════════════════════════════════════════════════
function buildDefacing() {
  let tbody = '';
  SUBJECTS.filter(s => !s.no_folder).forEach(s => {
    const ectaCell = s.is_ecta
      ? '<span class="badge bg-primary badge-sm">eCTA</span>'
      : '<span class="text-muted">—</span>';
    const subSafe = s.sub_id.replace('-','');
    const defCell = !s.is_ecta
      ? '<span class="text-muted">N/A</span>'
      : defaceBadge(qcGet('deface', s.sub_id).status, s.defaced);
    tbody += `<tr>
      <td data-order="${parseInt(s.sub_id.split('-')[1])}">${s.sub_id}</td><td>${ectaCell}</td><td class="text-center" id="defcell-${subSafe}">${defCell}</td>
      ${defaceQcCells(s.sub_id, s.is_ecta, s.defaced)}
    </tr>`;
  });
  document.getElementById('body-deface').innerHTML = tbody;
  if (!$.fn.DataTable.isDataTable('#tbl-deface'))
    $('#tbl-deface').DataTable({ pageLength:25, order:[[0,'asc']],
      columnDefs:[{ orderable:false, targets:[3,4] }, { type:'num', targets:0 }] });
}

// ════════════════════════════════════════════════════════
// TAB 4 — Segmentation
// ════════════════════════════════════════════════════════
function buildSegmentation() {
  let tbody = '';
  SUBJECTS.filter(s => !s.no_folder).forEach(s => {
    // eCTA row
    if (s.is_ecta) {
      const cid  = `${s.sub_id}_acq-ecta_ct`;
      const log  = s.seg_log;
      const logCell = log.status
        ? `<small class="text-muted" title="${log.message||''}">${log.status}</small>`
        : '—';
      const ov = segOverallColor(cid, s.ecta_seg.laa, s.ecta_seg.la, s.ecta_seg.ao);
      tbody += `<tr>
        <td data-order="${parseInt(s.sub_id.split('-')[1])}">${s.sub_id}</td>
        <td><small>${cid}</small></td>
        <td><span class="badge bg-primary badge-sm">eCTA</span></td>
        <td data-order="-1">—</td>
        <td class="text-center" data-order="${ov.order}">${dot(ov.color)}</td>
        <td class="text-center">${yn(s.ecta_seg.laa)}</td>
        <td class="text-center">${yn(s.ecta_seg.la)}</td>
        <td class="text-center">${yn(s.ecta_seg.ao)}</td>
        <td>${logCell}</td>
        ${qcCells('seg', cid)}
      </tr>`;
    }
    // Multi-CT rows
    s.mc_rows.forEach(r => {
      const logCell = r.seg_status
        ? `<small class="text-muted">${r.seg_status}</small>`
        : '—';
      const rov = segOverallColor(r.case_id, r.laa, r.la, r.ao);
      tbody += `<tr>
        <td data-order="${parseInt(s.sub_id.split('-')[1])}">${s.sub_id}</td>
        <td><small>${r.case_id}</small></td>
        <td><span class="badge bg-secondary badge-sm">${r.acq}</span></td>
        <td data-order="${r.phase !== '' ? parseInt(r.phase) : -1}">${r.phase || '—'}</td>
        <td class="text-center" data-order="${rov.order}">${dot(rov.color)}</td>
        <td class="text-center">${yn(r.laa)}</td>
        <td class="text-center">${yn(r.la)}</td>
        <td class="text-center">${yn(r.ao)}</td>
        <td>${logCell}</td>
        ${qcCells('seg', r.case_id)}
      </tr>`;
    });
  });
  document.getElementById('body-seg').innerHTML = tbody;
  if (!$.fn.DataTable.isDataTable('#tbl-seg'))
    $('#tbl-seg').DataTable({ pageLength:25, order:[[0,'asc']],
      columnDefs:[{ orderable:false, targets:[9,10,11] }, { type:'num', targets:[0,4] }] });
}

// ════════════════════════════════════════════════════════
// TAB 5 — Radiomics
// ════════════════════════════════════════════════════════
function buildRadiomics() {
  let tbody = '';
  SUBJECTS.filter(s => !s.no_folder).forEach(s => {
    const radBadge = s.rad_exists
      ? '<span class="badge bg-success badge-sm">Y</span>'
      : '<span class="badge bg-danger badge-sm">N</span>';
    const laaStatus = s.laa_status
      ? `<span class="badge bg-warning text-dark badge-sm">${s.laa_status}</span>`
      : (s.rad_exists ? '<span class="text-success" style="font-size:.75rem">ok</span>' : '—');
    const radCases = [];
    if (s.is_ecta) radCases.push({ id: `${s.sub_id}_acq-ecta_ct`, laa: s.ecta_seg.laa, la: s.ecta_seg.la, ao: s.ecta_seg.ao });
    s.mc_rows.forEach(r => radCases.push({ id: r.case_id, laa: r.laa, la: r.la, ao: r.ao }));
    let segOv = { color: 'gray', order: -1 };
    if (radCases.length > 0) {
      const ovResults = radCases.map(c => segOverallColor(c.id, c.laa, c.la, c.ao));
      if      (ovResults.some(r => r.color === 'green'))  segOv = { color: 'green',  order: 2 };
      else if (ovResults.some(r => r.color === 'orange')) segOv = { color: 'orange', order: 1 };
      else                                                segOv = { color: 'red',    order: 0 };
    }
    tbody += `<tr>
      <td data-order="${parseInt(s.sub_id.split('-')[1])}">${s.sub_id}</td>
      <td class="text-center">${radBadge}</td>
      <td class="text-center" data-order="${segOv.order}">${dot(segOv.color)}</td>
      <td data-order="${s.hu.laa_med != null ? s.hu.laa_med : -9999}">${hu(s.hu.laa_med, s.hu.laa_iqr)}</td>
      <td data-order="${s.hu.la_med  != null ? s.hu.la_med  : -9999}">${hu(s.hu.la_med,  s.hu.la_iqr)}</td>
      <td data-order="${s.hu.ao_med  != null ? s.hu.ao_med  : -9999}">${hu(s.hu.ao_med,  s.hu.ao_iqr)}</td>
      <td>${laaStatus}</td>
      ${qcCells('rad', s.sub_id)}
    </tr>`;
  });
  document.getElementById('body-rad').innerHTML = tbody;
  if (!$.fn.DataTable.isDataTable('#tbl-rad'))
    $('#tbl-rad').DataTable({ pageLength:25, order:[[0,'asc']],
      columnDefs:[{ orderable:false, targets:[7,8,9] }, { type:'num', targets:[0,2] }] });
}

// ════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════
// Overview — conversion filter
// ════════════════════════════════════════════════════════
let _ovFilter = 'all';

$.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
  if (settings.nTable.id !== 'tbl-overview') return true;
  if (_ovFilter === 'all') return true;
  const api   = new $.fn.dataTable.Api(settings);
  const node  = api.row(dataIndex).node();
  return node && node.getAttribute('data-conv') === _ovFilter;
});

function filterOverview(val) {
  _ovFilter = val;
  const map = { all:'secondary', green:'success', orange:'warning', red:'danger' };
  Object.keys(map).forEach(k => {
    const btn = document.getElementById('flt-' + k);
    if (!btn) return;
    btn.className = 'btn btn-sm ' + (k === val ? 'btn-' + map[k] : 'btn-outline-' + map[k]);
  });
  $('#tbl-overview').DataTable().draw();
}

document.addEventListener('DOMContentLoaded', () => {
  initReviewer();
  buildOverview();

  document.getElementById('mainTabs').addEventListener('shown.bs.tab', e => {
    const target = e.target.getAttribute('data-bs-target');
    if      (target === '#tab2') buildConversion();
    else if (target === '#tab3') buildDefacing();
    else if (target === '#tab4') buildSegmentation();
    else if (target === '#tab5') buildRadiomics();
  });
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
