"""AI CTA Stroke — Pipeline Dashboard (Streamlit)

Launch:
    streamlit run app/streamlit_app.py
or:
    bash scripts/run_app.sh
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.runner import run_dental, run_laa, run_aortic, run_sleep_apnea, run_osa, run_ts_airway_batch  # noqa: E402

_OSA_FEATURE_SETS = [
    "core_osa_backed",
    "core_plus_anatomic_extensions",
    "core_plus_cardiometabolic_ct",
    "all_features_exploratory",
]


# ── NTFY helper ───────────────────────────────────────────────────────────────
def _ntfy(topic: str, title: str, message: str, tags: list[str] | None = None) -> None:
    if not topic.strip():
        return
    try:
        import requests
        requests.post(
            f"https://ntfy.sh/{topic.strip()}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Tags": ",".join(tags or [])},
            timeout=5,
        )
    except Exception:
        pass


# ── Slice preview ─────────────────────────────────────────────────────────────
def _show_slice_preview(title: str, nifti_path: Path, is_mask: bool = False) -> None:
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


# ── Results renderer ──────────────────────────────────────────────────────────
def _show_results(out_dir_str: str, case_id: str) -> None:
    out_dir = Path(out_dir_str)

    # Dental
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

    # OSA
    osa_features = out_dir / "osa" / "features.csv"
    osa_qc = out_dir / "osa" / "qc.csv"

    if osa_features.exists() or osa_qc.exists():
        st.subheader("😴 OSA (stroke-cta-osa)")
        import pandas as pd

        if osa_qc.exists():
            qc_df = pd.read_csv(osa_qc)
            qc_cols = st.columns(4)
            for i, col_name in enumerate(qc_df.columns[:8]):
                qc_cols[i % 4].metric(col_name.replace("_", " "), str(qc_df.iloc[0][col_name]))
            with st.expander("Full QC"):
                st.dataframe(qc_df, use_container_width=True)

        if osa_features.exists():
            with st.expander("Features CSV"):
                st.dataframe(pd.read_csv(osa_features), use_container_width=True)

        for tier in _OSA_FEATURE_SETS:
            tier_csv = out_dir / "osa" / f"{tier}.csv"
            if tier_csv.exists():
                with st.expander(f"Tier: {tier}"):
                    st.dataframe(pd.read_csv(tier_csv), use_container_width=True)

    # LAA / Prior Fusion
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

    # Aortic
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


# ── Hardware detection ────────────────────────────────────────────────────────
@st.cache_resource
def _detect_hardware() -> dict:
    info: dict = {}
    # CPU / RAM
    try:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=False) or 1
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        import os
        info["cpu_cores"] = os.cpu_count() or 1
        info["ram_gb"] = 0.0

    # GPU
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            gpus = []
            for ln in lines:
                parts = ln.split(",")
                name = parts[0].strip()
                mem_str = parts[1].strip() if len(parts) > 1 else "N/A"
                # GB10 reports "N/A" for memory (unified); treat as 128 GB
                mem_mb = None
                if "MiB" in mem_str:
                    mem_mb = int(mem_str.replace("MiB", "").strip())
                elif "N/A" in mem_str or "Not Supported" in mem_str:
                    mem_mb = 128 * 1024  # GB10 unified memory estimate
                gpus.append({"name": name, "vram_mb": mem_mb})
            info["gpus"] = gpus
        else:
            info["gpus"] = []
    except Exception:
        info["gpus"] = []

    # Recommended config
    has_gpu = bool(info.get("gpus"))
    vram_mb = (info["gpus"][0]["vram_mb"] or 0) if has_gpu else 0
    ram_gb = info.get("ram_gb", 0)

    # TS airway workers: each TS run needs ~4–6 GB GPU mem; on GB10 unified we allow more
    if vram_mb >= 80 * 1024:      # ≥80 GB (GB10 unified or H100 80GB)
        rec_ts_workers = 6
    elif vram_mb >= 32 * 1024:
        rec_ts_workers = 4
    elif vram_mb >= 16 * 1024:
        rec_ts_workers = 2
    else:
        rec_ts_workers = 1

    # ETA estimate for 566 slaobids cases at ~1.2 min/case per worker
    min_per_case = 1.2
    ts_eta_h = (566 * min_per_case / rec_ts_workers) / 60

    info["recommended"] = {
        "device": "gpu" if has_gpu else "cpu",
        "ts_workers": rec_ts_workers,
        "ts_eta_h_566": round(ts_eta_h, 1),
    }
    return info


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI CTA Stroke Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI CTA Stroke")
    st.caption("Research prototype — not for clinical use")
    st.divider()

    # Hardware info
    hw = _detect_hardware()
    rec = hw["recommended"]
    gpus = hw.get("gpus", [])
    gpu_label = gpus[0]["name"] if gpus else "None"
    vram_label = f'{gpus[0]["vram_mb"] // 1024} GB' if gpus and gpus[0]["vram_mb"] else "unified"
    st.markdown("**Hardware**")
    hw_cols = st.columns(2)
    hw_cols[0].metric("CPU cores", hw.get("cpu_cores", "?"))
    hw_cols[1].metric("RAM", f'{hw.get("ram_gb", 0):.0f} GB')
    st.caption(f"GPU: {gpu_label} · {vram_label}")
    st.info(
        f"**Recommended config:** device=`{rec['device']}` · "
        f"TS workers=`{rec['ts_workers']}` · "
        f"ETA ~{rec['ts_eta_h_566']} h for 566 cases"
    )
    st.divider()

    pipelines = st.multiselect(
        "Pipelines",
        ["Dental", "OSA", "LAA", "Aortic", "Sleep Apnea"],
        default=["Dental", "OSA"],
    )

    aortic_tasks: list[str] = []
    if "Aortic" in pipelines:
        aortic_tasks = st.multiselect(
            "Aortic tasks",
            ["Calcium", "Fat", "Wall"],
            default=["Calcium", "Fat", "Wall"],
        )

    osa_feature_set: str | None = None
    osa_save_masks = False
    osa_dental_artifacts_dir: str = ""
    if "OSA" in pipelines:
        st.markdown("**OSA options**")
        osa_feature_set = st.selectbox(
            "Feature set",
            ["(auto)"] + _OSA_FEATURE_SETS,
            index=0,
        )
        if osa_feature_set == "(auto)":
            osa_feature_set = None
        osa_save_masks = st.checkbox("Save masks", value=False)
        osa_dental_artifacts_dir = st.text_input(
            "Dental artifacts dir",
            placeholder="/path/to/dental_out (optional)",
            help="Per-case dental output root; OSA pipeline reuses airway/landmarks/features.",
        )

    st.divider()
    device = st.selectbox("Device", ["gpu", "cpu"], index=0)

    st.divider()
    st.markdown("**NTFY notifications**")
    ntfy_topic = st.text_input(
        "ntfy.sh topic",
        placeholder="my-cta-pipeline",
        help="Leave blank to disable. Posts to https://ntfy.sh/<topic>",
    )

    st.divider()
    st.caption("v0.2 — DGX Spark / Linux / macOS / Windows")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_results = st.tabs(["Single Patient", "Batch", "Results Viewer"])


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE PATIENT
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.header("Single Patient")

    col1, col2 = st.columns([2, 1])
    with col1:
        nifti_path_str = st.text_input(
            "NIfTI path",
            placeholder="/media/friseb/LAAforLAAs/.../sub-001_acq-CTA_ct.nii.gz",
        )
    with col2:
        case_id = st.text_input("Case ID", placeholder="sub-001")
        out_dir_str = st.text_input("Output directory", placeholder="/tmp/pipeline_out")

    osa_airway_mask_str = ""
    if "OSA" in pipelines:
        osa_airway_mask_str = st.text_input(
            "External airway mask (optional)",
            placeholder="/path/ts_airway/sub-001/airway.nii.gz",
            help="Pre-computed TS airway mask. Skips the HU fallback.",
        )

    run_btn = st.button("▶  Run pipeline", type="primary", use_container_width=True)

    if run_btn:
        errors = []
        if not nifti_path_str or not Path(nifti_path_str).exists():
            errors.append(f"NIfTI not found: {nifti_path_str or '(empty)'}")
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

                if "OSA" in pipelines:
                    status_box.info("Running **OSA** pipeline …")
                    dental_art = Path(osa_dental_artifacts_dir) if osa_dental_artifacts_dir else None
                    airway_mask = Path(osa_airway_mask_str) if osa_airway_mask_str else None
                    _stream(run_osa(
                        nifti, out_dir / "osa", case_id,
                        feature_set=osa_feature_set,
                        save_masks=osa_save_masks,
                        dental_artifacts_dir=dental_art,
                        external_airway_mask=airway_mask,
                    ))

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
                _ntfy(ntfy_topic, f"CTA — {case_id}", "Pipeline complete ✓", ["white_check_mark"])

            except Exception as exc:
                status_box.error(f"Pipeline failed: {exc}")
                st.code("\n".join(logs[-30:]))
                _ntfy(ntfy_topic, f"CTA — {case_id} FAILED", str(exc), ["x"])

    if "last_out_dir" in st.session_state:
        _show_results(st.session_state["last_out_dir"], st.session_state["last_case_id"])


# ═════════════════════════════════════════════════════════════════════════════
# BATCH
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Run")

    col1, col2 = st.columns(2)
    with col1:
        input_dir_str = st.text_input(
            "Input directory",
            placeholder="/home/friseb/Desktop/slaobids",
            help="Directory containing NIfTI files.",
        )
        glob_pattern = st.text_input("Glob pattern", value="*_acq-CTA_ct.nii.gz", key="batch_glob")
    with col2:
        batch_out_str = st.text_input("Output root", placeholder="/tmp/batch_out")
        limit = st.number_input("Max cases (0 = all)", min_value=0, value=0, step=1)

    # OSA airway options — shown inline when OSA is selected
    airway_source = "none"
    airway_precompute_workers = 4
    osa_airway_mask_dir_str = ""
    if "OSA" in pipelines:
        st.markdown("**Airway masks (OSA)**")
        airway_source = st.radio(
            "Airway source",
            ["None (HU fallback)", "Pre-compute in batch (TS head_glands_cavities)", "Use existing dir"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if airway_source == "Pre-compute in batch (TS head_glands_cavities)":
            airway_precompute_workers = st.number_input(
                "TS workers", min_value=1, max_value=16, value=4, step=1,
                help="Parallel TotalSegmentator processes for airway pre-pass.",
            )
        elif airway_source == "Use existing dir":
            osa_airway_mask_dir_str = st.text_input(
                "Airway mask dir",
                placeholder="/home/friseb/Desktop/ts_airway_slao",
                help="Root dir with <case_id>/airway.nii.gz per case.",
            )

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
                batch_start = time.monotonic()
                n_done = 0

                # ── Airway pre-pass (if requested) ────────────────────────
                airway_mask_dir: Path | None = None
                if "OSA" in pipelines and airway_source == "Pre-compute in batch (TS head_glands_cavities)":
                    airway_mask_dir = batch_out / "_airway_masks"
                    progress.progress(0.0, text="Pre-computing airway masks (TS) …")
                    logs.append("=== Airway pre-pass (TS head_glands_cavities) ===")
                    log_box.code("\n".join(logs[-40:]), language="")
                    for line in run_ts_airway_batch(
                        out_dir=airway_mask_dir,
                        device=device,
                        in_dir=input_dir,
                        fast=False,
                    ):
                        logs.append(line)
                        log_box.code("\n".join(logs[-40:]), language="")
                    logs.append("=== Airway pre-pass complete ===")
                    log_box.code("\n".join(logs[-40:]), language="")

                elif "OSA" in pipelines and airway_source == "Use existing dir" and osa_airway_mask_dir_str:
                    airway_mask_dir = Path(osa_airway_mask_dir_str)

                # ── Per-case loop ─────────────────────────────────────────
                for i, nifti in enumerate(cases):
                    c_id = nifti.name.replace(".nii.gz", "").replace(".nii", "")
                    case_out = batch_out / c_id

                    if skip_existing:
                        dental_done = (case_out / "dental" / "report.json").exists() if "Dental" in pipelines else True
                        osa_done = (case_out / "osa" / "features.csv").exists() if "OSA" in pipelines else True
                        laa_done = (case_out / "laa").exists() if "LAA" in pipelines else True
                        if dental_done and osa_done and laa_done:
                            logs.append(f"[SKIP] {c_id}")
                            log_box.code("\n".join(logs[-40:]), language="")
                            results.append({"case_id": c_id, "status": "skipped"})
                            progress.progress((i + 1) / len(cases), text=f"{i+1}/{len(cases)}")
                            continue

                    logs.append(f"\n=== [{i+1}/{len(cases)}] {c_id} ===")
                    log_box.code("\n".join(logs[-40:]), language="")
                    case_start = time.monotonic()

                    try:
                        if "Dental" in pipelines:
                            for line in run_dental(nifti, case_out / "dental", c_id, device):
                                logs.append(line)
                                log_box.code("\n".join(logs[-40:]), language="")

                        if "OSA" in pipelines:
                            dental_art = Path(osa_dental_artifacts_dir) if osa_dental_artifacts_dir else None
                            airway_mask = (airway_mask_dir / c_id / "airway.nii.gz") if airway_mask_dir else None
                            for line in run_osa(
                                nifti, case_out / "osa", c_id,
                                feature_set=osa_feature_set,
                                save_masks=osa_save_masks,
                                dental_artifacts_dir=dental_art,
                                external_airway_mask=airway_mask,
                            ):
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

                        case_elapsed = time.monotonic() - case_start
                        results.append({"case_id": c_id, "status": "ok", "elapsed_s": f"{case_elapsed:.0f}"})

                    except Exception as exc:
                        logs.append(f"[FAIL] {c_id}: {exc}")
                        results.append({"case_id": c_id, "status": "failed", "error": str(exc)})

                    n_done += 1
                    elapsed = time.monotonic() - batch_start
                    remaining = len(cases) - (i + 1)
                    eta_s = (elapsed / n_done) * remaining if n_done else 0
                    eta_str = f"{eta_s/3600:.1f} h" if eta_s >= 3600 else f"{eta_s/60:.0f} min"
                    progress.progress((i + 1) / len(cases), text=f"{i+1}/{len(cases)} — ETA {eta_str}")

                    import pandas as pd
                    summary_placeholder.dataframe(pd.DataFrame(results), use_container_width=True)

                total_elapsed = time.monotonic() - batch_start
                n_ok = sum(1 for r in results if r["status"] == "ok")
                n_fail = sum(1 for r in results if r["status"] == "failed")
                progress.progress(1.0, text=f"Batch complete ✓  ({n_ok} ok, {n_fail} failed, {total_elapsed/60:.0f} min)")
                _ntfy(
                    ntfy_topic,
                    "CTA Batch complete",
                    f"{n_ok}/{len(cases)} ok · {n_fail} failed · {total_elapsed/60:.0f} min",
                    ["white_check_mark"] if n_fail == 0 else ["warning"],
                )


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS VIEWER
# ═════════════════════════════════════════════════════════════════════════════
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
