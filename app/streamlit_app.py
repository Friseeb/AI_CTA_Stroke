"""DeepStroke — CTA Stroke Pipeline Dashboard (Streamlit)

SOMA-branded front end for the dental / LAA / aortic / sleep-apnea pipelines.

Launch:
    streamlit run app/streamlit_app.py
or:
    bash scripts/run_app.sh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.runner import run_dental, run_laa, run_aortic, run_sleep_apnea  # noqa: E402

_UPLOAD_DIR = REPO_ROOT / "app" / "_uploads"


# ═══════════════════════════════════════════════════════════════
# File selection helpers — native OS dialog + drag-and-drop
# ═══════════════════════════════════════════════════════════════
def _native_pick(mode: str) -> str | None:
    """Open a native OS file/folder dialog and return the chosen path.

    Runs a Tk dialog on the machine hosting Streamlit — works for local
    sessions (the intended desktop use). Returns None if cancelled or if no
    display is available (e.g. a headless/remote server), in which case the
    user falls back to typing a path or drag-and-drop.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        if mode == "dir":
            path = filedialog.askdirectory(parent=root)
        else:
            path = filedialog.askopenfilename(
                parent=root,
                filetypes=[("NIfTI", "*.nii *.nii.gz"), ("DICOM / all files", "*.*")],
            )
        root.update()
        root.destroy()
        return path or None
    except Exception as exc:  # noqa: BLE001 — headless server, no Tk, cancelled, etc.
        st.session_state["_pick_error"] = (
            f"Native file dialog unavailable ({exc}). Type a path or drag-and-drop instead."
        )
        return None


def _default_case_id(path: str) -> str:
    """Best-effort case id from a NIfTI filename (sub-001_acq-CTA_ct → sub-001)."""
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.split("_acq")[0] if "_acq" in name else name


def _stage_upload(uploaded) -> Path:
    """Persist a drag-and-dropped file to a staging dir and return its path."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / uploaded.name
    if not dest.exists() or dest.stat().st_size != uploaded.size:
        dest.write_bytes(uploaded.getbuffer())
    return dest


def _pick_into(state_key: str, mode: str, also_case_id: bool = False) -> None:
    """on_click callback: open native dialog, store result in session_state."""
    picked = _native_pick(mode)
    if picked:
        st.session_state[state_key] = picked
        if also_case_id and not st.session_state.get("case_id"):
            st.session_state["case_id"] = _default_case_id(picked)


# Default output root: BIDS derivatives if the data volume is mounted, else repo.
_BIDS_ROOT = Path("/Volumes/DICOM5/slaobids")
DERIVATIVES_ROOT = str((_BIDS_ROOT / "derivatives") if _BIDS_ROOT.is_dir()
                       else REPO_ROOT / "outputs" / "derivatives")


def _fs_browser(target_key: str, kind: str = "dir") -> None:
    """Reliable in-app filesystem browser (no native dialog dependency).

    Renders navigation controls; on selection writes the chosen path to
    ``st.session_state[target_key]`` and closes the browser. Must be rendered
    *before* the text_input that shares ``target_key`` so the write is legal.
    kind='dir' selects a folder; kind='file' selects a NIfTI.
    """
    cwd_key = f"_cwd_{target_key}"
    seed = st.session_state.get(target_key) or DERIVATIVES_ROOT
    seed_p = Path(seed)
    if seed_p.is_file():
        seed_p = seed_p.parent
    if not seed_p.is_dir():
        seed_p = Path.home()
    st.session_state.setdefault(cwd_key, str(seed_p))
    cwd = Path(st.session_state[cwd_key])
    if not cwd.is_dir():
        cwd = Path.home()
        st.session_state[cwd_key] = str(cwd)

    st.caption(f"📂 `{cwd}`")
    b1, b2, b3 = st.columns(3)
    if b1.button("⬆ Up", key=f"up_{target_key}", use_container_width=True):
        st.session_state[cwd_key] = str(cwd.parent)
        st.rerun()
    if b2.button("🏠 Home", key=f"hm_{target_key}", use_container_width=True):
        st.session_state[cwd_key] = str(Path.home())
        st.rerun()
    if kind == "dir":
        if b3.button("✅ Use this folder", key=f"pick_{target_key}", type="primary",
                     use_container_width=True):
            st.session_state[target_key] = str(cwd)
            st.session_state[f"_open_br_{target_key}"] = False
            st.rerun()

    try:
        subdirs = sorted((p.name for p in cwd.iterdir() if p.is_dir() and not p.name.startswith(".")),
                         key=str.lower)
    except (PermissionError, OSError):
        subdirs = []
    nav = st.selectbox("Open subfolder", ["—"] + subdirs, key=f"nav_{target_key}")
    if nav != "—":
        st.session_state[cwd_key] = str(cwd / nav)
        st.rerun()

    if kind == "file":
        try:
            files = sorted(p.name for p in cwd.iterdir()
                           if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")))
        except (PermissionError, OSError):
            files = []
        picked_file = st.selectbox("Pick a NIfTI here", ["—"] + files, key=f"file_{target_key}")
        if picked_file != "—" and st.button("✅ Use this file", key=f"pickf_{target_key}",
                                             type="primary"):
            st.session_state[target_key] = str(cwd / picked_file)
            if target_key == "nifti_path" and not st.session_state.get("case_id"):
                st.session_state["case_id"] = _default_case_id(picked_file)
            st.session_state[f"_open_br_{target_key}"] = False
            st.rerun()


def _browse_toggle(target_key: str, label: str = "📂 Browse") -> None:
    """A button that toggles the in-app browser expander for target_key."""
    flag = f"_open_br_{target_key}"
    if st.button(label, key=f"tgl_{target_key}", use_container_width=True):
        st.session_state[flag] = not st.session_state.get(flag, False)
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Shared results renderer — defined early so tabs can call it
# ═══════════════════════════════════════════════════════════════
def _show_slice_preview(title: str, nifti_path: Path, is_mask: bool = False) -> None:
    """Show axial/coronal/sagittal mid-slices from a NIfTI file."""
    try:
        import nibabel as nib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj)

        if not is_mask:
            data = np.clip(data, -200, 1500)
            cmap = "gray"
        else:
            cmap = "hot"

        mid = [s // 2 for s in data.shape]
        views = [
            ("Axial", np.rot90(data[:, :, mid[2]])),
            ("Coronal", np.rot90(data[:, mid[1], :])),
            ("Sagittal", np.rot90(data[mid[0], :, :])),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(title, fontsize=10)
        for ax, (name, sl) in zip(axes, views):
            ax.imshow(sl, cmap=cmap, aspect="equal", interpolation="nearest")
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        plt.tight_layout()

        with st.expander(f"📷 {title}"):
            st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.caption(f"Preview unavailable: {exc}")


def _show_results(out_dir_str: str, case_id: str) -> None:
    out_dir = Path(out_dir_str)

    dental_report = out_dir / "dental" / "report.json"
    dental_features = out_dir / "dental" / "candidate_features.json"

    if dental_report.exists():
        st.subheader("🦷 Dental")
        report = json.loads(dental_report.read_text())
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", report.get("status", "—"))
        col2.metric("ROI quality", report.get("roi_quality", "—"))
        col3.metric("Segmentation", report.get("segmentation_status", "—"))
        fov = report.get("fov_completeness") or {}
        col4.metric(
            "FOV",
            "upper+lower" if fov.get("has_upper_dentition") and fov.get("has_lower_dentition") else "partial",
        )

        if dental_features.exists():
            feats = json.loads(dental_features.read_text())
            markers = feats.get("candidate_markers", {})

            col_u, col_l = st.columns(2)
            for jaw, col in [("upper", col_u), ("lower", col_l)]:
                jaw_data = next((x for x in markers.get("teeth_present", []) if x["jaw"] == jaw), None)
                if jaw_data:
                    col.metric(f"{jaw.capitalize()} teeth present", jaw_data["count"])

            peri = markers.get("periapical_candidates", [])
            if peri:
                st.warning(f"⚠ {len(peri)} periapical candidate(s) detected")
                import pandas as pd
                st.dataframe(pd.DataFrame(peri), use_container_width=True)
            else:
                st.success("No periapical candidates detected")

            imp = markers.get("implants", [])
            if imp:
                st.info(f"🔩 {len(imp)} implant(s) detected")

            with st.expander("Full candidate_features.json"):
                st.json(feats)

        preprocessed = out_dir / "dental" / "preprocessed.nii.gz"
        if preprocessed.exists():
            _show_slice_preview("Dental — preprocessed CTA", preprocessed)

    fusion_summary = out_dir / "laa" / "prior_fusion" / case_id / f"{case_id}_prior_fusion_summary.json"
    if fusion_summary.exists():
        st.subheader("🫀 LAA / Prior Fusion")
        summary = json.loads(fusion_summary.read_text())
        cols = st.columns(4)
        for i, (k, v) in enumerate(summary.items()):
            if k not in ("case_id", "sources_used"):
                cols[i % 4].metric(k.replace("_", " "), str(v))

        st.caption(f"Sources used: {', '.join(summary.get('sources_used', []))}")

        with st.expander("Full summary JSON"):
            st.json(summary)

        consensus = out_dir / "laa" / "prior_fusion" / case_id / f"{case_id}_consensus_laa.nii.gz"
        if consensus.exists():
            _show_slice_preview("LAA consensus mask", consensus, is_mask=True)

    aortic_report = out_dir / "aortic" / "report.json"
    if aortic_report.exists():
        st.subheader("🩸 Aortic")
        report = json.loads(aortic_report.read_text())
        cols = st.columns(4)
        for i, (k, v) in enumerate(report.items()):
            if k not in ("case_id", "tasks"):
                cols[i % 4].metric(k.replace("_", " "), str(v))
        tasks_done = report.get("tasks", [])
        if tasks_done:
            st.caption(f"Tasks: {', '.join(tasks_done)}")
        with st.expander("Full report JSON"):
            st.json(report)
        for task in ("calcium", "fat", "wall"):
            seg = out_dir / "aortic" / f"{case_id}_{task}.nii.gz"
            if seg.exists():
                _show_slice_preview(f"Aortic — {task}", seg, is_mask=True)

    sleep_report = out_dir / "sleep_apnea" / "report.json"
    if sleep_report.exists():
        st.subheader("😴 Sleep Apnea")
        report = json.loads(sleep_report.read_text())
        cols = st.columns(4)
        for i, (k, v) in enumerate(report.items()):
            if k not in ("case_id",):
                cols[i % 4].metric(k.replace("_", " "), str(v))
        with st.expander("Full report JSON"):
            st.json(report)


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DeepStroke",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── SOMA brand theme (injected CSS) ───────────────────────────
def _inject_brand() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Cormorant+Garamond:wght@600;700&display=swap');

        :root {
            --soma-plum:#201436; --soma-royal:#4f2683; --soma-violet:#8f55e0;
            --soma-yellow:#fcf05e; --soma-cream:#faf9f5; --soma-lav:#f1ecfa;
            --soma-grey:#6b6862;
        }
        html, body, .stApp, [class*="css"] {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        code, pre, .stCode, [data-testid="stCode"] * {
            font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
        }
        h1, h2, h3, h4 { color: var(--soma-plum); letter-spacing: -0.01em; font-weight: 600; }

        /* Primary buttons — SOMA violet */
        .stButton > button[kind="primary"] {
            background: var(--soma-violet); border: 1px solid var(--soma-royal);
            color: #fff; font-weight: 600; border-radius: 8px;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--soma-royal); border-color: var(--soma-plum);
        }
        .stButton > button[kind="secondary"] {
            border-radius: 8px; border-color: #d9cef0;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: var(--soma-violet); color: var(--soma-royal);
        }
        /* Sidebar accent rail */
        [data-testid="stSidebar"] {
            background: var(--soma-lav);
            border-right: 1px solid #e3d8f6;
        }
        /* Tabs — active tab in violet */
        .stTabs [aria-selected="true"] { color: var(--soma-royal) !important; }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--soma-violet) !important; }

        /* DeepStroke wordmark */
        .ds-brand { display:flex; align-items:center; gap:10px; margin:2px 0 0; }
        .ds-mark {
            width:34px; height:34px; border-radius:9px; flex:none;
            background: radial-gradient(circle at 32% 30%, var(--soma-violet), var(--soma-royal) 70%);
            box-shadow: 0 0 0 3px #e7dbfa;
            position:relative;
        }
        .ds-mark::after{
            content:""; position:absolute; inset:11px; border-radius:50%;
            background: var(--soma-yellow);
        }
        .ds-name { font-size:26px; font-weight:700; color:var(--soma-plum); line-height:1; letter-spacing:-0.02em; }
        .ds-name .accent { color: var(--soma-violet); }
        .ds-tag { font-size:11px; letter-spacing:0.14em; text-transform:uppercase;
                  color:var(--soma-grey); margin-top:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_brand()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="ds-brand">
          <span class="ds-mark"></span>
          <span class="ds-name">Deep<span class="accent">Stroke</span></span>
        </div>
        <div class="ds-tag">CTA stroke pipelines · SOMA Lab</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Research prototype — not for clinical use")
    st.divider()

    pipelines = st.multiselect(
        "Pipelines to run",
        ["Dental", "LAA", "Aortic", "Sleep Apnea"],
        default=["Aortic"],
    )

    # Always visible so the Calcium/Fat/Wall choice is never hidden.
    aortic_tasks = st.multiselect(
        "Aortic tasks (Calcium / Fat / Wall)",
        ["Calcium", "Fat", "Wall"],
        default=["Calcium", "Fat", "Wall"],
        disabled="Aortic" not in pipelines,
        help="Which aortic feature stages to surface. Enable the Aortic pipeline above to use these.",
    )

    st.divider()
    device = st.selectbox(
        "Device",
        ["gpu", "cpu"],
        index=0,
        help="gpu = CUDA (NVIDIA) or MPS (Apple Silicon). Falls back to CPU automatically.",
    )

    st.divider()
    st.caption("v0.1 — DGX Spark / Linux / macOS / Windows")


# ── Tabs ──────────────────────────────────────────────────────
tab_single, tab_batch, tab_results = st.tabs(["Single Patient", "Batch", "Results Viewer"])


# ═══════════════════════════════════════════════════════════════
# SINGLE PATIENT
# ═══════════════════════════════════════════════════════════════
with tab_single:
    st.header("Single Patient")

    # Seed persistent keys once.
    st.session_state.setdefault("nifti_path", "")
    st.session_state.setdefault("case_id", "")
    st.session_state.setdefault("out_dir", DERIVATIVES_ROOT)

    st.markdown("**Input CTA** — drag & drop, browse, or paste a path")

    # 1) Drag-and-drop upload (staged to disk; fires once per new file).
    uploaded = st.file_uploader(
        "Drag & drop a CTA NIfTI here",
        type=["gz", "nii"],
        key="nifti_upload",
        help="For very large or already-on-disk scans, use 📂 Browse instead — no copy.",
    )
    if uploaded is not None:
        _sig = (uploaded.name, uploaded.size)
        if st.session_state.get("_uploaded_sig") != _sig:
            st.session_state["_uploaded_sig"] = _sig
            st.session_state["nifti_path"] = str(_stage_upload(uploaded))
            if not st.session_state["case_id"]:
                st.session_state["case_id"] = _default_case_id(st.session_state["nifti_path"])

    # 2) In-app file browser (rendered BEFORE the text field so it may set the key).
    if st.session_state.get("_open_br_nifti_path"):
        with st.container(border=True):
            _fs_browser("nifti_path", kind="file")
    c_path, c_browse = st.columns([5, 1])
    with c_path:
        st.text_input(
            "NIfTI path",
            key="nifti_path",
            placeholder="/media/friseb/LAAforLAAs/.../sub-001_acq-CTA_ct.nii.gz",
            help="Absolute path on disk — or use Browse / drag-and-drop above.",
        )
    with c_browse:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        _browse_toggle("nifti_path", "📂 Browse")

    st.text_input("Case ID", key="case_id", placeholder="sub-001")

    st.markdown("**Output — a BIDS derivatives folder**")
    if st.session_state.get("_open_br_out_dir"):
        with st.container(border=True):
            _fs_browser("out_dir", kind="dir")
    c_out, c_out_b = st.columns([5, 1])
    with c_out:
        st.text_input(
            "Output directory",
            key="out_dir",
            help="Results are written here (defaults to the derivatives folder).",
        )
    with c_out_b:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        _browse_toggle("out_dir", "📂 Browse")

    nifti_path_str = st.session_state["nifti_path"]
    case_id = st.session_state["case_id"]
    out_dir_str = st.session_state["out_dir"]

    run_btn = st.button("▶  Run pipeline", type="primary", use_container_width=True)

    if run_btn:
        errors = []
        if not nifti_path_str:
            errors.append("NIfTI path is required.")
        elif not Path(nifti_path_str).exists():
            errors.append(f"File not found: {nifti_path_str}")
        if not case_id:
            errors.append("Case ID is required.")
        if not out_dir_str:
            errors.append("Output directory is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            nifti = Path(nifti_path_str)
            out_dir = Path(out_dir_str)
            out_dir.mkdir(parents=True, exist_ok=True)

            log_box = st.empty()
            status_box = st.empty()
            logs: list[str] = []

            def _stream(gen):
                for line in gen:
                    logs.append(line)
                    log_box.code("\n".join(logs[-60:]), language="")

            try:
                if "Dental" in pipelines:
                    status_box.info("Running **Dental** pipeline …")
                    _stream(run_dental(nifti, out_dir / "dental", case_id, device))

                if "LAA" in pipelines:
                    status_box.info("Running **LAA** pipeline …")
                    _stream(run_laa(nifti, out_dir / "laa", case_id, device))

                if "Aortic" in pipelines:
                    status_box.info("Running **Aortic** pipeline …")
                    _stream(run_aortic(nifti, out_dir / "aortic", case_id, aortic_tasks, device))

                if "Sleep Apnea" in pipelines:
                    status_box.info("Running **Sleep Apnea** pipeline …")
                    _stream(run_sleep_apnea(nifti, out_dir / "sleep_apnea", case_id, device))

                status_box.success("Pipeline complete ✓")
                st.session_state["last_out_dir"] = str(out_dir)
                st.session_state["last_case_id"] = case_id

            except Exception as exc:
                status_box.error(f"Pipeline failed: {exc}")
                st.code("\n".join(logs[-30:]))

    # ── Quick preview of most-recent output ───────────────────
    if "last_out_dir" in st.session_state:
        _show_results(st.session_state["last_out_dir"], st.session_state["last_case_id"])


# ═══════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Run")

    st.session_state.setdefault("batch_in", "")
    st.session_state.setdefault("batch_out", DERIVATIVES_ROOT)

    # In-app browsers rendered first so they may set the keys before text_inputs.
    if st.session_state.get("_open_br_batch_in"):
        with st.container(border=True):
            st.caption("Choose input directory")
            _fs_browser("batch_in", kind="dir")
    if st.session_state.get("_open_br_batch_out"):
        with st.container(border=True):
            st.caption("Choose output root")
            _fs_browser("batch_out", kind="dir")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Input directory",
            key="batch_in",
            placeholder="/media/friseb/LAAforLAAs/bids/derivatives/defaced",
            help="Directory containing NIfTI files — Browse or paste a path.",
        )
        _browse_toggle("batch_in", "📂 Browse input folder")
        glob_pattern = st.text_input("Glob pattern", value="*_ct.nii.gz")
    with col2:
        st.text_input("Output root (derivatives)", key="batch_out")
        _browse_toggle("batch_out", "📂 Browse output folder")
        limit = st.number_input("Max cases (0 = all)", min_value=0, value=0, step=1)

    input_dir_str = st.session_state["batch_in"]
    batch_out_str = st.session_state["batch_out"]

    skip_existing = st.checkbox("Skip already-completed cases", value=True)
    batch_btn = st.button("▶  Start batch", type="primary", use_container_width=True)

    if batch_btn:
        if not input_dir_str or not batch_out_str:
            st.error("Input directory and output root are required.")
        else:
            input_dir = Path(input_dir_str)
            batch_out = Path(batch_out_str)
            batch_out.mkdir(parents=True, exist_ok=True)

            cases = sorted(input_dir.glob(glob_pattern))
            if limit:
                cases = cases[: int(limit)]

            if not cases:
                st.warning(f"No files matched {glob_pattern} in {input_dir}")
            else:
                st.info(f"Found {len(cases)} case(s).")
                progress = st.progress(0.0, text="Starting …")
                log_box = st.empty()
                summary_placeholder = st.empty()
                logs: list[str] = []
                results: list[dict] = []

                for i, nifti in enumerate(cases):
                    c_id = nifti.name.replace(".nii.gz", "").replace(".nii", "")
                    case_out = batch_out / c_id

                    if skip_existing:
                        done_marker = case_out / "dental" / "report.json"
                        if done_marker.exists():
                            logs.append(f"[SKIP] {c_id}")
                            log_box.code("\n".join(logs[-40:]), language="")
                            results.append({"case_id": c_id, "status": "skipped"})
                            progress.progress((i + 1) / len(cases), text=f"{i+1}/{len(cases)}")
                            continue

                    logs.append(f"\n=== [{i+1}/{len(cases)}] {c_id} ===")
                    log_box.code("\n".join(logs[-40:]), language="")

                    try:
                        if "Dental" in pipelines:
                            for line in run_dental(nifti, case_out / "dental", c_id, device):
                                logs.append(line)
                                log_box.code("\n".join(logs[-40:]), language="")
                        if "LAA" in pipelines:
                            for line in run_laa(nifti, case_out / "laa", c_id, device):
                                logs.append(line)
                                log_box.code("\n".join(logs[-40:]), language="")
                        if "Aortic" in pipelines:
                            for line in run_aortic(nifti, case_out / "aortic", c_id, aortic_tasks, device):
                                logs.append(line)
                                log_box.code("\n".join(logs[-40:]), language="")
                        if "Sleep Apnea" in pipelines:
                            for line in run_sleep_apnea(nifti, case_out / "sleep_apnea", c_id, device):
                                logs.append(line)
                                log_box.code("\n".join(logs[-40:]), language="")
                        results.append({"case_id": c_id, "status": "ok"})
                    except Exception as exc:
                        logs.append(f"[FAIL] {c_id}: {exc}")
                        results.append({"case_id": c_id, "status": "failed", "error": str(exc)})

                    progress.progress((i + 1) / len(cases), text=f"{i+1}/{len(cases)}")

                    import pandas as pd
                    summary_placeholder.dataframe(
                        pd.DataFrame(results), use_container_width=True
                    )

                progress.progress(1.0, text="Batch complete ✓")


# ═══════════════════════════════════════════════════════════════
# RESULTS VIEWER
# ═══════════════════════════════════════════════════════════════
with tab_results:
    st.header("Results Viewer")

    res_dir_str = st.text_input(
        "Output directory to inspect",
        value=st.session_state.get("last_out_dir", ""),
        placeholder="/tmp/pipeline_out",
    )
    res_case_id = st.text_input(
        "Case ID",
        value=st.session_state.get("last_case_id", ""),
    )

    if res_dir_str and res_case_id:
        _show_results(res_dir_str, res_case_id)

