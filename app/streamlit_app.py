"""AI CTA Stroke — Pipeline Dashboard (Streamlit)

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
    page_title="AI CTA Stroke Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI CTA Stroke")
    st.caption("Research prototype — not for clinical use")
    st.divider()

    pipelines = st.multiselect(
        "Pipelines",
        ["Dental", "LAA", "Aortic", "Sleep Apnea"],
        default=["Dental"],
    )

    aortic_tasks: list[str] = []
    if "Aortic" in pipelines:
        aortic_tasks = st.multiselect(
            "Aortic tasks",
            ["Calcium", "Fat", "Wall"],
            default=["Calcium", "Fat", "Wall"],
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

    col1, col2 = st.columns([2, 1])
    with col1:
        nifti_path_str = st.text_input(
            "NIfTI path",
            placeholder="/media/friseb/LAAforLAAs/.../sub-001_acq-CTA_ct.nii.gz",
            help="Absolute path to the CTA NIfTI file on disk.",
        )
    with col2:
        case_id = st.text_input("Case ID", placeholder="sub-001")
        out_dir_str = st.text_input(
            "Output directory",
            placeholder="/tmp/pipeline_out",
        )

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

    col1, col2 = st.columns(2)
    with col1:
        input_dir_str = st.text_input(
            "Input directory",
            placeholder="/media/friseb/LAAforLAAs/bids/derivatives/defaced",
            help="Directory containing NIfTI files (*_ct.nii.gz).",
        )
        glob_pattern = st.text_input("Glob pattern", value="*_ct.nii.gz")
    with col2:
        batch_out_str = st.text_input(
            "Output root",
            placeholder="/tmp/batch_out",
        )
        limit = st.number_input("Max cases (0 = all)", min_value=0, value=0, step=1)

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

