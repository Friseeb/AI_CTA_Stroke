#!/usr/bin/env python3
"""Generate a presentation-friendly LA/LAA shape metrics report with 3D viewers.

Example:
  python3 scripts/generate_la_laa_shape_report.py \
    --metrics-csv /path/to/daylightbids/derivatives/shape_meshes_repro/la_laa_metrics_batch.csv \
    --mesh-root /path/to/daylightbids/derivatives/shape_meshes_repro \
    --output-html /path/to/daylightbids/derivatives/shape_meshes_repro/la_laa_shape_report.html \
    --max-3d-cases 6
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

try:
    import seaborn as sns

    HAS_SNS = True
except Exception:  # noqa: BLE001
    HAS_SNS = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    HAS_PLOTLY = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    HAS_SKLEARN = False

from la_laa_metrics import load_mesh


DEFAULT_METRICS_CSV = "./outputs/shape_meshes_repro/la_laa_metrics_batch.csv"
DEFAULT_MESH_ROOT = "./outputs/shape_meshes_repro"
DEFAULT_OUTPUT_HTML = "./outputs/shape_meshes_repro/la_laa_shape_report.html"
DEFAULT_RADIOMICS_DERIVED_CSV = "./outputs/radiomics/radiomics_derived_metrics.csv"

DEFINITIONS: dict[str, str] = {
    "min_distance_mm": "Minimum LA-to-LAA surface distance; large values may indicate mismatch or segmentation separation.",
    "ostium_planarity": "PCA planarity score at pseudo-interface (smaller is more planar).",
    "laa_axis_length_mm": "Distance from ostium center to distal LAA tip along estimated appendage axis.",
    "bend_ostiumNormal_vs_proxLAA_deg": "Angle between ostium plane normal and proximal LAA direction.",
    "bend_LAaxis_vs_LAAaxis_deg": "Angle between LA principal axis and LAA ostium-to-tip axis.",
    "ostium_dist_median_mm": "Median LA distance among LAA points used to fit the ostium plane.",
    "ostium_points_n": "Number of LAA vertices used for ostium plane fitting.",
    "qc_far_apart": "True when minimum LA-LAA gap exceeds failure threshold.",
    "qc_ostium_too_few_points": "True when interface candidate points are insufficient for a stable plane fit.",
    "qc_tip_wrong_side": "True when estimated tip lies on the wrong side of ostium normal.",
    "qc_exception": "True when processing failed with an exception.",
}

FRIENDLY_LABELS: dict[str, str] = {
    "case_id": "Case ID",
    "cluster": "Shape Cluster",
    "integrated_cluster": "Integrated Cluster",
    "n_cases": "Cases",
    "k": "K",
    "silhouette": "Silhouette",
    "inertia": "Inertia",
    "cluster_centroid_distance": "Centroid Distance",
    "centroid_distance": "Centroid Distance",
    "integrated_centroid_distance": "Integrated Centroid Distance",
    "min_distance_mm": "Minimum LA-LAA Distance (mm)",
    "ostium_dist_median_mm": "Median Ostium Distance (mm)",
    "ostium_planarity": "Ostium Planarity",
    "ostium_points_n": "Ostium Points (n)",
    "laa_axis_length_mm": "LAA Axis Length (mm)",
    "bend_ostiumNormal_vs_proxLAA_deg": "Ostium Normal vs Proximal LAA Angle (deg)",
    "bend_LAaxis_vs_LAAaxis_deg": "LA Axis vs LAA Axis Angle (deg)",
    "laa_surface_area_mm2": "LAA Surface Area (mm²)",
    "laa_volume_mm3": "LAA Volume (mm³)",
    "la_surface_area_mm2": "LA Surface Area (mm²)",
    "la_volume_mm3": "LA Volume (mm³)",
    "laa_to_la_volume_ratio": "LAA/LA Volume Ratio",
    "laa_to_la_area_ratio": "LAA/LA Area Ratio",
    "LA_LAA_delta": "LA-LAA HU Delta",
    "Normalized_LAA_defect": "Normalized LAA Defect",
    "LAA_entropy": "LAA Entropy",
    "LAA_uniformity": "LAA Uniformity",
    "LAA_volume_ml": "LAA Volume (mL)",
}

KEY_METRICS = [
    "min_distance_mm",
    "ostium_dist_median_mm",
    "ostium_planarity",
    "laa_axis_length_mm",
    "bend_ostiumNormal_vs_proxLAA_deg",
    "bend_LAaxis_vs_LAAaxis_deg",
    "laa_surface_area_mm2",
    "laa_volume_mm3",
    "la_surface_area_mm2",
    "la_volume_mm3",
]

QC_COLS = ["qc_far_apart", "qc_ostium_too_few_points", "qc_tip_wrong_side", "qc_exception"]

CLUSTER_BASE_FEATURES = [
    "min_distance_mm",
    "ostium_dist_median_mm",
    "ostium_planarity",
    "ostium_points_n",
    "laa_axis_length_mm",
    "bend_ostiumNormal_vs_proxLAA_deg",
    "bend_LAaxis_vs_LAAaxis_deg",
    "laa_surface_area_mm2",
    "laa_volume_mm3",
    "la_surface_area_mm2",
    "la_volume_mm3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate HTML report with LA/LAA shape metrics and interactive 3D scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV, help="Batch metrics CSV from run_la_laa_metrics_batch.py")
    p.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT, help="Mesh root directory (for cases lacking explicit paths)")
    p.add_argument("--output-html", default=DEFAULT_OUTPUT_HTML, help="Output HTML report path")
    p.add_argument("--title", default="LA/LAA Shape Metrics Report", help="Report title")
    p.add_argument("--la-suffix", default="left_atrium_highres", help="LA mesh suffix token")
    p.add_argument("--laa-suffix", default="laa_nudf", help="LAA mesh suffix token")
    p.add_argument("--max-3d-cases", type=int, default=6, help="Maximum number of 3D cases embedded")
    p.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case_id(s) to include in 3D section; repeatable.",
    )
    p.add_argument("--max-faces-per-mesh", type=int, default=120000, help="Downsample meshes for browser rendering")
    p.add_argument("--wireframe-max-edges", type=int, default=42000, help="Maximum mesh edges for wireframe overlay per mesh")
    p.add_argument("--cluster-k", type=int, default=0, help="Fixed k for KMeans (0 = auto by silhouette)")
    p.add_argument("--cluster-k-min", type=int, default=2, help="Minimum k to evaluate when auto-selecting")
    p.add_argument("--cluster-k-max", type=int, default=8, help="Maximum k to evaluate when auto-selecting")
    p.add_argument("--no-clustering", action="store_true", help="Disable clustering section and centroid representative selection")
    p.add_argument(
        "--radiomics-derived-csv",
        default=DEFAULT_RADIOMICS_DERIVED_CSV,
        help="Optional radiomics-derived metrics CSV (patient/case level) for integrated clustering.",
    )
    p.add_argument("--no-integrated-clustering", action="store_true", help="Disable integrated shape+radiomics clustering.")
    p.add_argument("--integrated-k", type=int, default=0, help="Fixed k for integrated KMeans (0 = auto by silhouette)")
    p.add_argument("--integrated-k-min", type=int, default=2, help="Minimum k for integrated auto-selection")
    p.add_argument("--integrated-k-max", type=int, default=8, help="Maximum k for integrated auto-selection")
    p.add_argument(
        "--integrated-min-nonnull-ratio",
        type=float,
        default=0.60,
        help="Minimum non-null ratio per radiomics feature for integrated clustering.",
    )
    p.add_argument(
        "--integrated-max-radiomics-features",
        type=int,
        default=60,
        help="Maximum number of radiomics features to include (top variance among eligible).",
    )
    p.add_argument(
        "--integrated-output-csv",
        default="",
        help="Optional output CSV path for integrated cluster assignments (default beside report).",
    )
    return p.parse_args()


def fmt_float(v: Any, nd: int = 3) -> str:
    try:
        fv = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(fv):
        return ""
    return f"{fv:.{nd}f}"


def _pretty_label(name: Any) -> str:
    if name is None:
        return ""
    s = str(name)
    if s in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[s]
    if s.startswith("qc_"):
        return s.replace("qc_", "QC ").replace("_", " ").strip().title()
    if "_" in s:
        return s.replace("_", " ").strip().title()
    return s


def _subject_label(case_id: str) -> str:
    s = str(case_id or "").strip()
    m = re.search(r"sub-(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Subject {m.group(1)}"
    m2 = re.search(r"(\d+)", s)
    if m2:
        return f"Subject {m2.group(1)}"
    return s if s else "Subject"


def fmt_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if any(tok in col.lower() for tok in ("qc_", "points", "n_")):
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(int(float(x))) if float(x).is_integer() else fmt_float(x, 2))
            elif any(tok in col.lower() for tok in ("deg", "distance", "length", "area", "volume")):
                out[col] = out[col].map(lambda x: fmt_float(x, 2))
            else:
                out[col] = out[col].map(lambda x: fmt_float(x, 4))
    return out


def df_to_html(df: pd.DataFrame, index: bool = True) -> str:
    out = fmt_df_for_html(df)
    out = out.rename(columns={c: _pretty_label(c) for c in out.columns})
    if index:
        out.index = [_pretty_label(i) for i in out.index]
    return out.to_html(index=index, classes="tbl", border=0, escape=True)


def _init_plot_style() -> None:
    if HAS_SNS:
        sns.set_theme(style="white", context="paper")


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=9)


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _img_block(title: str, data_uri: str, alt: str) -> str:
    return (
        "<figure class='panel-card'>"
        f"<figcaption>{escape(title)}</figcaption>"
        f"<img class='plot' src='{data_uri}' alt='{escape(alt)}'/>"
        "</figure>"
    )


def plot_qc_counts(df: pd.DataFrame) -> str:
    if not HAS_MPL:
        return ""
    _init_plot_style()
    counts = []
    for col in QC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            counts.append((col, int(s.astype(bool).sum())))
    if not counts:
        return ""

    qc = pd.DataFrame(counts, columns=["flag", "n_true"]) 
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    if HAS_SNS:
        sns.barplot(data=qc, x="flag", y="n_true", color="#8d1d38", ax=ax)
    else:
        ax.bar(qc["flag"], qc["n_true"], color="#8d1d38")
    ax.set_xlabel("")
    ax.set_ylabel("Cases")
    ax.set_xticks(np.arange(len(qc)))
    ax.set_xticklabels([_pretty_label(v) for v in qc["flag"]], rotation=18, ha="right")
    _style_axis(ax)
    return _img_block("QC Flag Counts", _fig_to_data_uri(fig), "QC counts")


def plot_metric_hist(df: pd.DataFrame, col: str, title: str) -> str:
    if not HAS_MPL or col not in df.columns:
        return ""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return ""
    _init_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    if HAS_SNS:
        sns.histplot(s, bins=28, color="#0d4a73", kde=True, ax=ax)
    else:
        ax.hist(s, bins=28, color="#0d4a73", alpha=0.9)
    ax.set_xlabel(_pretty_label(col))
    ax.set_ylabel("Count")
    _style_axis(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    if not HAS_MPL or x not in df.columns or y not in df.columns:
        return ""
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        return ""
    _init_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.scatter(d[x], d[y], s=11, c="#8d1d38", alpha=0.78, linewidths=0)
    ax.set_xlabel(_pretty_label(x))
    ax.set_ylabel(_pretty_label(y))
    _style_axis(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


def plot_corr_heatmap(df: pd.DataFrame, cols: list[str]) -> str:
    if not HAS_MPL:
        return ""
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        return ""
    d = df[present].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method="spearman", numeric_only=True)
    if corr.isna().all().all():
        return ""

    _init_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if HAS_SNS:
        sns.heatmap(
            corr,
            cmap="vlag",
            center=0.0,
            vmin=-1,
            vmax=1,
            linewidths=0.3,
            linecolor="#efefef",
            cbar=True,
            square=True,
            ax=ax,
        )
        ax.set_xticklabels([_pretty_label(c) for c in corr.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels([_pretty_label(c) for c in corr.index], fontsize=8)
    else:
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels([_pretty_label(c) for c in corr.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_yticklabels([_pretty_label(c) for c in corr.index], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _style_axis(ax)
    return _img_block("Correlation Heatmap (Spearman)", _fig_to_data_uri(fig), "Correlation heatmap")


def plot_integrated_cluster_heatmap(df: pd.DataFrame, rad_cols: list[str]) -> str:
    if not HAS_MPL or "integrated_cluster" not in df.columns:
        return ""
    base_cols = [
        "min_distance_mm",
        "ostium_planarity",
        "laa_axis_length_mm",
        "bend_LAaxis_vs_LAAaxis_deg",
        "laa_to_la_volume_ratio",
        "LA_LAA_delta",
        "Normalized_LAA_defect",
        "LAA_entropy",
        "LAA_uniformity",
        "LAA_volume_ml",
    ]
    present = [c for c in base_cols if c in df.columns]
    if len(present) < 4:
        # fallback to strongest available radiomics columns
        present = [c for c in base_cols if c in df.columns] + [c for c in rad_cols[:8] if c in df.columns]
        present = list(dict.fromkeys(present))
    if len(present) < 4:
        return ""
    work = df[["integrated_cluster"] + present].copy()
    work[present] = work[present].apply(pd.to_numeric, errors="coerce")
    means = work.groupby("integrated_cluster")[present].mean(numeric_only=True)
    if means.empty:
        return ""
    std = means.std(axis=0, ddof=0).replace(0, np.nan)
    z = (means - means.mean(axis=0)) / std
    z = z.fillna(0.0)

    _init_plot_style()
    fig, ax = plt.subplots(figsize=(max(6.6, 0.65 * len(present)), 3.2 + 0.35 * len(z.index)))
    if HAS_SNS:
        sns.heatmap(
            z,
            cmap="RdBu_r",
            center=0.0,
            vmin=-2.5,
            vmax=2.5,
            linewidths=0.35,
            linecolor="#e6eaef",
            cbar_kws={"label": "Cluster Mean Z-Score"},
            ax=ax,
        )
    else:
        im = ax.imshow(z.to_numpy(), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Integrated Cluster")
    ax.set_xticklabels([_pretty_label(c) for c in z.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([f"Cluster {int(i)}" for i in z.index], fontsize=9)
    _style_axis(ax)
    return _img_block(
        "Integrated Cluster Phenotype Heatmap",
        _fig_to_data_uri(fig),
        "Integrated cluster heatmap",
    )


def describe_cluster_profiles(summary: pd.DataFrame, cluster_col: str) -> str:
    if summary.empty or cluster_col not in summary.columns:
        return ""
    num_cols = [
        c for c in summary.columns if c != cluster_col and pd.api.types.is_numeric_dtype(summary[c]) and c != "n_cases"
    ]
    if not num_cols:
        return ""
    center = summary[num_cols].mean(axis=0)
    spread = summary[num_cols].std(axis=0, ddof=0).replace(0, np.nan)
    z = (summary[num_cols] - center) / spread
    z = z.fillna(0.0)

    lines: list[str] = []
    for idx, row in summary.iterrows():
        cid = int(row[cluster_col])
        top = z.loc[idx].abs().sort_values(ascending=False).head(3).index.tolist()
        phrases = []
        for col in top:
            direction = "higher" if float(z.loc[idx, col]) >= 0 else "lower"
            phrases.append(f"{direction} {_pretty_label(col)}")
        if phrases:
            lines.append(f"<li><b>Cluster {cid}</b>: " + ", ".join(phrases) + ".</li>")
    if not lines:
        return ""
    return "<h4>Cluster Interpretation</h4><ul class='cluster-list'>" + "".join(lines) + "</ul>"


def _find_case_mesh_path(case_dir: Path, case_id: str, suffix: str) -> Path | None:
    for ext in ("vtk", "vtp", "ply", "stl", "obj"):
        p = case_dir / f"{case_id}_{suffix}.{ext}"
        if p.exists():
            return p
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return (
        series.astype(str).str.strip().str.lower().map(mapping).fillna(False).astype(bool)
    )


def _cluster_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["laa_to_la_volume_ratio"] = out["laa_volume_mm3"] / out["la_volume_mm3"].replace(0, np.nan)
    out["laa_to_la_area_ratio"] = out["laa_surface_area_mm2"] / out["la_surface_area_mm2"].replace(0, np.nan)
    out["la_compactness_idx"] = (out["la_volume_mm3"] ** (2.0 / 3.0)) / out["la_surface_area_mm2"].replace(0, np.nan)
    out["laa_compactness_idx"] = (out["laa_volume_mm3"] ** (2.0 / 3.0)) / out["laa_surface_area_mm2"].replace(0, np.nan)
    feat_cols = CLUSTER_BASE_FEATURES + [
        "laa_to_la_volume_ratio",
        "laa_to_la_area_ratio",
        "la_compactness_idx",
        "laa_compactness_idx",
    ]
    present = [c for c in feat_cols if c in out.columns]
    return out, present


def _run_clustering(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int] | None:
    if not HAS_SKLEARN or args.no_clustering:
        return None

    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"] == "success"]
    if "qc_exception" in work.columns:
        work = work[~_bool_series(work["qc_exception"])]
    if len(work) < 12:
        return None

    work, feat_cols = _cluster_feature_table(work)
    if len(feat_cols) < 4:
        return None

    mat = work[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x = prep.fit_transform(mat)

    # Evaluate k by silhouette unless fixed k is requested.
    k_scores = []
    k_min = max(2, int(args.cluster_k_min))
    k_max = max(k_min, int(args.cluster_k_max))
    for k in range(k_min, k_max + 1):
        if k >= len(work):
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=40)
        labels = model.fit_predict(x)
        sil = float(silhouette_score(x, labels))
        k_scores.append({"k": k, "silhouette": sil, "inertia": float(model.inertia_)})

    if not k_scores:
        return None

    k_scores_df = pd.DataFrame(k_scores)
    if args.cluster_k and int(args.cluster_k) > 1:
        k_best = min(int(args.cluster_k), len(work) - 1)
    else:
        k_best = int(k_scores_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    model = KMeans(n_clusters=k_best, random_state=42, n_init=40)
    labels = model.fit_predict(x)
    work = work.copy()
    work["cluster"] = labels

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(x)
    work["cluster_pca1"] = emb[:, 0]
    work["cluster_pca2"] = emb[:, 1]

    # Representative = closest case to centroid for each cluster.
    centers = model.cluster_centers_
    dists = np.linalg.norm(x - centers[labels], axis=1)
    work["cluster_centroid_distance"] = dists
    reps = (
        work.sort_values(["cluster", "cluster_centroid_distance", "case_id"])
        .groupby("cluster", as_index=False)
        .first()
    )
    reps = reps.sort_values(["cluster"]).reset_index(drop=True)

    summary = (
        work.groupby("cluster", as_index=False)
        .agg(
            n_cases=("case_id", "size"),
            min_distance_mm=("min_distance_mm", "mean"),
            ostium_planarity=("ostium_planarity", "mean"),
            laa_axis_length_mm=("laa_axis_length_mm", "mean"),
            bend_LAaxis_vs_LAAaxis_deg=("bend_LAaxis_vs_LAAaxis_deg", "mean"),
            laa_volume_mm3=("laa_volume_mm3", "mean"),
            laa_to_la_volume_ratio=("laa_to_la_volume_ratio", "mean"),
        )
        .sort_values("cluster")
    )

    return work, reps, summary, k_scores_df.sort_values("k"), k_best


def _infer_id_column(df: pd.DataFrame) -> str | None:
    for col in ("case_id", "patient_id", "subject_id"):
        if col in df.columns:
            return col
    return None


def _load_radiomics_derived(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rad = pd.read_csv(path)
    id_col = _infer_id_column(rad)
    if id_col is None:
        return None
    out = rad.copy()
    out = out.rename(columns={id_col: "case_id"})
    out["case_id"] = out["case_id"].astype(str).str.strip()
    return out


def _pick_radiomics_features(
    df: pd.DataFrame,
    min_nonnull_ratio: float,
    max_features: int,
) -> list[str]:
    numeric_cols = [
        c
        for c in df.columns
        if c != "case_id" and pd.api.types.is_numeric_dtype(df[c])
    ]
    eligible: list[str] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if float(s.notna().mean()) < float(min_nonnull_ratio):
            continue
        if float(s.std(skipna=True)) <= 1e-12:
            continue
        eligible.append(col)
    if not eligible:
        return []
    variances = df[eligible].apply(pd.to_numeric, errors="coerce").var(skipna=True).sort_values(ascending=False)
    return list(variances.head(max(1, int(max_features))).index)


def _run_integrated_clustering(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, list[str], int] | None:
    if not HAS_SKLEARN or args.no_integrated_clustering:
        return None

    rad_path = Path(str(args.radiomics_derived_csv or "").strip())
    if not str(rad_path):
        return None
    rad = _load_radiomics_derived(rad_path)
    if rad is None or rad.empty:
        return None

    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"] == "success"]
    if "qc_exception" in work.columns:
        work = work[~_bool_series(work["qc_exception"])]
    if len(work) < 12:
        return None

    work, shape_cols = _cluster_feature_table(work)
    if len(shape_cols) < 4:
        return None
    work["case_id"] = work["case_id"].astype(str).str.strip()

    merged = work.merge(rad, on="case_id", how="inner", suffixes=("", "_rad"))
    if len(merged) < 12:
        return None

    rad_cols = _pick_radiomics_features(
        merged,
        min_nonnull_ratio=float(args.integrated_min_nonnull_ratio),
        max_features=int(args.integrated_max_radiomics_features),
    )
    if len(rad_cols) < 3:
        return None

    prep_shape = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    prep_rad = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_shape = prep_shape.fit_transform(merged[shape_cols].apply(pd.to_numeric, errors="coerce").to_numpy())
    x_rad = prep_rad.fit_transform(merged[rad_cols].apply(pd.to_numeric, errors="coerce").to_numpy())

    # Equalize contribution of shape/radiomics blocks despite different dimensionality.
    x_shape = x_shape * np.sqrt(0.5 / max(1, x_shape.shape[1]))
    x_rad = x_rad * np.sqrt(0.5 / max(1, x_rad.shape[1]))
    x = np.hstack([x_shape, x_rad])

    k_scores = []
    k_min = max(2, int(args.integrated_k_min))
    k_max = max(k_min, int(args.integrated_k_max))
    for k in range(k_min, k_max + 1):
        if k >= len(merged):
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=40)
        labels = model.fit_predict(x)
        k_scores.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "inertia": float(model.inertia_),
            }
        )
    if not k_scores:
        return None

    k_scores_df = pd.DataFrame(k_scores).sort_values("k")
    if args.integrated_k and int(args.integrated_k) > 1:
        k_best = min(int(args.integrated_k), len(merged) - 1)
    else:
        k_best = int(k_scores_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    model = KMeans(n_clusters=k_best, random_state=42, n_init=40)
    labels = model.fit_predict(x)
    merged = merged.copy()
    merged["integrated_cluster"] = labels

    centers = model.cluster_centers_
    dists = np.linalg.norm(x - centers[labels], axis=1)
    merged["integrated_centroid_distance"] = dists
    reps = (
        merged.sort_values(["integrated_cluster", "integrated_centroid_distance", "case_id"])
        .groupby("integrated_cluster", as_index=False)
        .first()
        .sort_values("integrated_cluster")
        .reset_index(drop=True)
    )

    summary_aggs: dict[str, tuple[str, str]] = {
        "n_cases": ("case_id", "size"),
        "min_distance_mm": ("min_distance_mm", "mean"),
        "ostium_planarity": ("ostium_planarity", "mean"),
        "laa_axis_length_mm": ("laa_axis_length_mm", "mean"),
    }
    for col in ["LA_LAA_delta", "Normalized_LAA_defect", "LAA_entropy", "LAA_uniformity", "LAA_volume_ml"]:
        if col in merged.columns:
            summary_aggs[col] = (col, "mean")

    summary = merged.groupby("integrated_cluster", as_index=False).agg(**summary_aggs).sort_values("integrated_cluster")
    return merged, reps, summary, k_scores_df, k_best, rad_cols, len(rad)


def _pick_cases(
    df: pd.DataFrame,
    case_ids: list[str],
    max_cases: int,
    cluster_reps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"] == "success"]
    if "qc_exception" in work.columns:
        qx = _bool_series(work["qc_exception"])
        work = work[~qx]

    if case_ids:
        wanted = set(case_ids)
        work = work[work["case_id"].astype(str).isin(wanted)]
        return work.head(max_cases)

    if cluster_reps is not None and not cluster_reps.empty:
        return cluster_reps.head(max_cases)

    sort_cols = [c for c in ["min_distance_mm", "ostium_planarity", "laa_axis_length_mm"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[True, True, False][: len(sort_cols)])
    return work.head(max_cases)


def _downsample_mesh(mesh, max_faces: int):
    n_faces = int(mesh.faces.shape[0])
    if n_faces <= max_faces:
        return mesh
    # Prefer topology-preserving decimation if optional backend is available.
    # Fallback keeps the full mesh to avoid dotted/fragmented random-face artifacts.
    try:
        dec = mesh.simplify_quadric_decimation(int(max_faces))
        if dec is not None and int(dec.faces.shape[0]) > 0:
            dec.process(validate=True)
            return dec
    except Exception:  # noqa: BLE001
        pass
    return mesh


def _mesh_trace(mesh, name: str, color: str, opacity: float):
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={"ambient": 0.06, "diffuse": 0.55, "specular": 0.45, "roughness": 0.88, "fresnel": 0.12},
        lightposition={"x": 260, "y": 190, "z": 260},
    )


def _mesh_wireframe_trace(mesh, name: str, color: str, max_edges: int = 42000):
    try:
        edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    except Exception:  # noqa: BLE001
        return None
    if edges.size == 0:
        return None
    if len(edges) > max_edges:
        rng = np.random.default_rng(0)
        edges = edges[rng.choice(len(edges), size=max_edges, replace=False)]
    v = np.asarray(mesh.vertices, dtype=float)
    seg = np.empty((len(edges) * 3, 3), dtype=float)
    seg[0::3] = v[edges[:, 0]]
    seg[1::3] = v[edges[:, 1]]
    seg[2::3] = np.nan
    return go.Scatter3d(
        x=seg[:, 0],
        y=seg[:, 1],
        z=seg[:, 2],
        mode="lines",
        line={"color": color, "width": 2.2},
        name=f"{name} edges",
        hoverinfo="skip",
    )


def _vector_trace(origin: np.ndarray, direction: np.ndarray, length: float, name: str, color: str):
    end = origin + direction * float(length)
    return go.Scatter3d(
        x=[origin[0], end[0]],
        y=[origin[1], end[1]],
        z=[origin[2], end[2]],
        mode="lines",
        line={"color": color, "width": 8},
        name=name,
    )


def _marker_trace(point: np.ndarray, name: str, color: str):
    return go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode="markers",
        marker={"size": 5, "color": color, "symbol": "circle"},
        name=name,
    )


def _row_vec(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray(
        [
            pd.to_numeric(row.get(f"{prefix}_x"), errors="coerce"),
            pd.to_numeric(row.get(f"{prefix}_y"), errors="coerce"),
            pd.to_numeric(row.get(f"{prefix}_z"), errors="coerce"),
        ],
        dtype=float,
    )


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.full(3, np.nan, dtype=float)
    return v / n


def _make_case_figure(case_title: str, row: pd.Series, mesh_la, mesh_laa, max_faces: int, wireframe_max_edges: int):
    mesh_la = _downsample_mesh(mesh_la, max_faces=max(1000, int(max_faces)))
    mesh_laa = _downsample_mesh(mesh_laa, max_faces=max(1000, int(max_faces)))

    traces = [
        _mesh_trace(mesh_la, "LA", "#8fa6b9", 0.06),
        _mesh_trace(mesh_laa, "LAA", "#bea59f", 0.08),
    ]
    wf_la = _mesh_wireframe_trace(mesh_la, "LA", "#2f6da1", max_edges=int(wireframe_max_edges))
    wf_laa = _mesh_wireframe_trace(mesh_laa, "LAA", "#a2473a", max_edges=int(wireframe_max_edges))
    if wf_la is not None:
        traces.append(wf_la)
    if wf_laa is not None:
        traces.append(wf_laa)

    c = _row_vec(row, "ostium_center")
    n = _norm(_row_vec(row, "ostium_normal"))
    a = _norm(_row_vec(row, "laa_axis"))
    p = _norm(_row_vec(row, "laa_prox_dir"))
    axis_len = pd.to_numeric(row.get("laa_axis_length_mm"), errors="coerce")

    ext = np.asarray(mesh_laa.extents, dtype=float)
    base_len = float(np.nanmax(ext)) if np.isfinite(ext).any() else 15.0
    if not np.isfinite(base_len) or base_len <= 0:
        base_len = 15.0

    if np.isfinite(c).all():
        traces.append(_marker_trace(c, "ostium center", "#1f2b37"))
        if np.isfinite(n).all():
            traces.append(_vector_trace(c, n, 0.45 * base_len, "ostium normal", "#2f6da1"))
        if np.isfinite(a).all():
            vec_len = float(axis_len) if np.isfinite(axis_len) and axis_len > 0 else 0.9 * base_len
            traces.append(_vector_trace(c, a, vec_len, "LAA axis", "#b51234"))
        if np.isfinite(p).all():
            traces.append(_vector_trace(c, p, 0.6 * base_len, "proximal direction", "#217a5a"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{case_title}: LA/LAA geometry and relational vectors",
        template="plotly_white",
        showlegend=False,
        legend={"orientation": "h", "y": -0.11},
        scene={
            "xaxis": {"title": "", "showbackground": True, "backgroundcolor": "#ffffff", "gridcolor": "#e8edf2", "zeroline": False, "showticklabels": False},
            "yaxis": {"title": "", "showbackground": True, "backgroundcolor": "#ffffff", "gridcolor": "#e8edf2", "zeroline": False, "showticklabels": False},
            "zaxis": {"title": "", "showbackground": True, "backgroundcolor": "#ffffff", "gridcolor": "#e8edf2", "zeroline": False, "showticklabels": False},
            "aspectmode": "data",
            "camera": {"projection": {"type": "perspective"}, "eye": {"x": 1.28, "y": 1.16, "z": 0.82}},
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#1e2a35"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        height=760,
    )
    return fig


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        df["case_id"] = np.arange(len(df)).astype(str)
    return df


def _overview(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    success = int((df["status"] == "success").sum()) if "status" in df.columns else n
    fail = n - success
    rows = [
        ("cases_total", n),
        ("cases_success", success),
        ("cases_non_success", fail),
    ]
    for qc in QC_COLS:
        if qc in df.columns:
            s = pd.to_numeric(df[qc], errors="coerce").fillna(0)
            rows.append((f"{qc}_n", int(s.astype(bool).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"]) 


def _metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_METRICS if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["metric", "count", "mean", "std", "p25", "p50", "p75"])

    d = df[cols].apply(pd.to_numeric, errors="coerce")
    out = (
        d.describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "25%", "50%", "75%"]]
        .reset_index()
        .rename(columns={"index": "metric", "25%": "p25", "50%": "p50", "75%": "p75"})
    )
    return out


def _resolve_mesh_paths(row: pd.Series, mesh_root: Path, la_suffix: str, laa_suffix: str) -> tuple[Path | None, Path | None]:
    la_col = str(row.get("la_path", "") or "")
    laa_col = str(row.get("laa_path", "") or "")

    la_path = Path(la_col) if la_col else None
    laa_path = Path(laa_col) if laa_col else None

    if la_path and la_path.exists() and laa_path and laa_path.exists():
        return la_path, laa_path

    case_id = str(row.get("case_id", ""))
    if not case_id:
        return None, None

    case_dir = mesh_root / case_id
    if not case_dir.exists():
        return None, None

    la_found = _find_case_mesh_path(case_dir, case_id, la_suffix)
    laa_found = _find_case_mesh_path(case_dir, case_id, laa_suffix)
    return la_found, laa_found


def _definitions_table(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c in DEFINITIONS:
            rows.append((c, DEFINITIONS[c]))
    return pd.DataFrame(rows, columns=["metric", "definition"])


def build_report(args: argparse.Namespace) -> str:
    metrics_csv = Path(args.metrics_csv)
    output_html = Path(args.output_html)
    mesh_root = Path(args.mesh_root)

    df = _read_metrics(metrics_csv)
    overview = _overview(df)
    metric_summary = _metric_summary(df)
    cluster_section_html = "<p class='muted'>Shape clustering unavailable (requires scikit-learn) or disabled.</p>"
    integrated_section_html = "<p class='muted'>Integrated shape+radiomics clustering skipped (missing radiomics data, insufficient overlap, or disabled).</p>"
    cluster_reps: pd.DataFrame | None = None
    case_reps: pd.DataFrame | None = None
    df_for_cases = df.copy()

    cluster_result = _run_clustering(df, args)
    if cluster_result is not None:
        cluster_df, cluster_reps, cluster_summary, cluster_k_scores, cluster_k = cluster_result
        cluster_map = cluster_df[["case_id", "cluster", "cluster_centroid_distance"]].copy()
        df_for_cases = df_for_cases.merge(cluster_map, on="case_id", how="left")
        reps_small = cluster_reps[["cluster", "case_id", "cluster_centroid_distance"]].copy()
        cluster_expl = describe_cluster_profiles(cluster_summary, "cluster")
        cluster_section_html = (
            f"<p class='muted'>KMeans selected <b>k={cluster_k}</b>. "
            "3D exemplars below are centroid-nearest representatives.</p>"
            f"{df_to_html(cluster_summary.set_index('cluster'))}"
            f"{cluster_expl}"
            "<h4>K Selection (Silhouette)</h4>"
            f"{df_to_html(cluster_k_scores.set_index('k'))}"
            "<h4>Centroid Representative Cases</h4>"
            f"{df_to_html(reps_small, index=False)}"
        )
        case_reps = cluster_reps

    integrated_result = _run_integrated_clustering(df, args)
    if integrated_result is not None:
        (
            integrated_df,
            integrated_reps,
            integrated_summary,
            integrated_k_scores,
            integrated_k,
            integrated_rad_cols,
            integrated_n_rad_total,
        ) = integrated_result
        integrated_map = integrated_df[["case_id", "integrated_cluster", "integrated_centroid_distance"]].copy()
        df_for_cases = df_for_cases.merge(integrated_map, on="case_id", how="left")
        reps_small = integrated_reps[["integrated_cluster", "case_id", "integrated_centroid_distance"]].copy()
        out_csv = Path(args.integrated_output_csv) if str(args.integrated_output_csv).strip() else output_html.with_name(
            f"{output_html.stem}_integrated_clusters.csv"
        )
        save_cols = [
            c
            for c in (
                ["case_id", "integrated_cluster", "integrated_centroid_distance"]
                + CLUSTER_BASE_FEATURES
                + integrated_rad_cols
            )
            if c in integrated_df.columns
        ]
        integrated_df[save_cols].to_csv(out_csv, index=False)
        integrated_expl = describe_cluster_profiles(integrated_summary, "integrated_cluster")
        integrated_heatmap = plot_integrated_cluster_heatmap(integrated_df, integrated_rad_cols)
        integrated_section_html = (
            f"<p class='muted'>Integrated KMeans selected <b>k={integrated_k}</b> on "
            f"<b>{len(integrated_df)}</b> overlapping cases (shape + radiomics available).</p>"
            f"<p class='muted'>Radiomics feature block used: <b>{len(integrated_rad_cols)}</b> descriptors.</p>"
            f"{df_to_html(integrated_summary.set_index('integrated_cluster'))}"
            f"{integrated_expl}"
            f"{integrated_heatmap}"
            "<h4>K Selection (Silhouette)</h4>"
            f"{df_to_html(integrated_k_scores.set_index('k'))}"
            "<h4>Centroid Representative Cases</h4>"
            f"{df_to_html(reps_small, index=False)}"
        )
        case_reps = integrated_reps

    plots: list[str] = []
    plots.append(plot_qc_counts(df))
    plots.append(plot_metric_hist(df, "min_distance_mm", "Distribution: Minimum LA-LAA Distance"))
    plots.append(plot_metric_hist(df, "laa_axis_length_mm", "Distribution: LAA Axis Length"))
    plots.append(plot_metric_hist(df, "bend_LAaxis_vs_LAAaxis_deg", "Distribution: LA-vs-LAA Axis Bend Angle"))
    plots.append(
        plot_scatter(
            df,
            x="laa_axis_length_mm",
            y="bend_LAaxis_vs_LAAaxis_deg",
            title="LAA Axis Length vs Global Bend Angle",
        )
    )
    plots.append(plot_corr_heatmap(df, KEY_METRICS))
    plot_blocks = [p for p in plots if p]
    plots_html = f"<div class='panel-grid'>{''.join(plot_blocks)}</div>" if plot_blocks else ""

    defs = _definitions_table(KEY_METRICS + QC_COLS)

    case_df = _pick_cases(
        df_for_cases,
        case_ids=args.case_id,
        max_cases=int(args.max_3d_cases),
        cluster_reps=case_reps,
    )
    case_blocks: list[str] = []

    if not HAS_PLOTLY:
        case_blocks.append("<p class='muted'>plotly not available; 3D section skipped.</p>")
    elif case_df.empty:
        case_blocks.append("<p class='muted'>No eligible successful cases found for 3D visualization.</p>")
    else:
        include_js: str | bool = "inline"
        for _, row in case_df.iterrows():
            case_id = str(row.get("case_id", ""))
            subject_name = _subject_label(case_id)
            la_path, laa_path = _resolve_mesh_paths(row, mesh_root, args.la_suffix, args.laa_suffix)
            if la_path is None or laa_path is None:
                case_blocks.append(f"<h4>{escape(subject_name)}</h4><p class='muted'>Missing mesh files for this case.</p>")
                continue

            try:
                mesh_la = load_mesh(str(la_path))
                mesh_laa = load_mesh(str(laa_path))
                fig = _make_case_figure(
                    subject_name,
                    row,
                    mesh_la,
                    mesh_laa,
                    max_faces=args.max_faces_per_mesh,
                    wireframe_max_edges=args.wireframe_max_edges,
                )
                fig_html = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs=include_js,
                    config={
                        "displayModeBar": False,
                        "scrollZoom": False,
                        "responsive": True,
                    },
                )
                include_js = False
                cluster_label = row.get("cluster", np.nan)
                centroid_dist = row.get("cluster_centroid_distance", np.nan)
                integrated_label = row.get("integrated_cluster", np.nan)
                integrated_centroid_dist = row.get("integrated_centroid_distance", np.nan)
                meta = pd.DataFrame(
                    {
                        "field": [
                            "case_id",
                            "min_distance_mm",
                            "laa_axis_length_mm",
                            "bend_LAaxis_vs_LAAaxis_deg",
                            "bend_ostiumNormal_vs_proxLAA_deg",
                            "ostium_planarity",
                            "cluster",
                            "centroid_distance",
                            "integrated_cluster",
                            "integrated_centroid_distance",
                        ],
                        "value": [
                            case_id,
                            row.get("min_distance_mm", np.nan),
                            row.get("laa_axis_length_mm", np.nan),
                            row.get("bend_LAaxis_vs_LAAaxis_deg", np.nan),
                            row.get("bend_ostiumNormal_vs_proxLAA_deg", np.nan),
                            row.get("ostium_planarity", np.nan),
                            (int(cluster_label) if pd.notna(cluster_label) else ""),
                            centroid_dist,
                            (int(integrated_label) if pd.notna(integrated_label) else ""),
                            integrated_centroid_dist,
                        ],
                    }
                )
                title = subject_name
                if pd.notna(integrated_label):
                    title = f"{title} (Integrated Cluster {int(integrated_label)})"
                elif pd.notna(cluster_label):
                    title = f"{title} (Shape Cluster {int(cluster_label)})"
                case_blocks.append(f"<h3>{escape(title)}</h3>{fig_html}{df_to_html(meta, index=False)}")
            except Exception as exc:  # noqa: BLE001
                case_blocks.append(
                    f"<h4>{escape(subject_name)}</h4><p class='muted'>3D rendering failed: {escape(type(exc).__name__ + ': ' + str(exc))}</p>"
                )

    css = """
:root {
  --aha-red: #b51234;
  --aha-blue: #0f4c81;
  --ink: #17202a;
  --muted: #4f5b66;
  --line: #d7dde3;
  --paper: #ffffff;
  --panel: #f8fafc;
}
body {
  font-family: "Source Serif Pro", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  margin: 24px 30px;
  color: var(--ink);
  background: var(--paper);
  line-height: 1.45;
}
h1, h2, h3, h4 {
  margin: 0.55em 0 0.34em;
  font-family: "Helvetica Neue", "Avenir Next", "Segoe UI", Arial, sans-serif;
  letter-spacing: 0.02em;
}
h1 {
  color: var(--aha-red);
  font-size: 2.05rem;
  border-bottom: 2px solid var(--aha-red);
  padding-bottom: 0.2rem;
  margin-bottom: 0.35rem;
}
h2 {
  color: var(--aha-blue);
  font-size: 1.25rem;
  border-top: 1px solid var(--line);
  padding-top: 0.45rem;
}
h3, h4 { color: #1f2d3d; }
p { margin: 0.22em 0 0.72em; font-size: 15px; }
.muted { color: var(--muted); font-size: 0.94em; }
.note {
  background: var(--panel);
  border-left: 4px solid var(--aha-red);
  padding: 10px 12px;
  margin: 8px 0 14px;
  font-size: 14px;
}
.tbl {
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0 18px;
  font-size: 13px;
  table-layout: fixed;
}
.tbl th, .tbl td {
  border: 1px solid var(--line);
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}
.tbl th {
  background: #eff3f7;
  color: #223344;
  font-weight: 600;
}
.tbl th:first-child, .tbl td:first-child {
  width: 30%;
}
.plot {
  max-width: 100%;
  height: auto;
  border: 0;
  margin: 0;
}
.panel-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  align-items: start;
}
@media (min-width: 1280px) {
  .panel-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.panel-card {
  margin: 0;
  padding: 8px 10px 8px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}
.panel-card figcaption {
  margin: 0 0 6px 0;
  font-family: "Helvetica Neue", "Avenir Next", "Segoe UI", Arial, sans-serif;
  font-size: 0.92rem;
  font-weight: 600;
  color: #1f2d3d;
}
.panel-card .plot {
  width: 100%;
}
.section { margin-bottom: 20px; }
code {
  background: #edf2f7;
  padding: 1px 5px;
  border-radius: 4px;
  color: #203040;
}
.cluster-list { margin: 0.25rem 0 0.85rem 1.1rem; padding: 0; }
.cluster-list li { margin: 0.16rem 0; }
.section {
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.65s ease, transform 0.65s ease;
}
.section.visible {
  opacity: 1;
  transform: translateY(0);
}
"""

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(args.title)}</title>
  <style>{css}</style>
</head>
<body>
  <h1>{escape(args.title)}</h1>
  <p class=\"muted\">Generated: {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>

  <div class=\"section\">
    <h2>Executive Summary</h2>
    {df_to_html(overview, index=False)}
    <div class=\"note\">
      1) Lower <b>min_distance_mm</b> supports plausible LA/LAA adjacency.<br/>
      2) <b>ostium_planarity</b> near zero indicates a cleaner ostium plane estimate.<br/>
      3) Bend angles summarize appendage orientation relative to ostium and global LA axis.<br/>
      4) QC flags should be reviewed before downstream statistical modeling.
    </div>
  </div>

  <div class=\"section\">
    <h2>Metric Distributions</h2>
    {plots_html if plots_html else '<p class="muted">Plotting libraries unavailable; charts skipped.</p>'}
  </div>

  <div class=\"section\">
    <h2>Metric Summary Table</h2>
    {df_to_html(metric_summary.set_index('metric')) if not metric_summary.empty else '<p class="muted">No numeric metrics available.</p>'}
  </div>

  <div class=\"section\">
    <h2>Cluster Analysis</h2>
    {cluster_section_html}
  </div>

  <div class=\"section\">
    <h2>Integrated Shape + Radiomics Clustering</h2>
    {integrated_section_html}
  </div>

  <div class=\"section\">
    <h2>Definitions</h2>
    {df_to_html(defs, index=False) if not defs.empty else '<p class="muted">No definition rows available.</p>'}
  </div>

  <div class=\"section\">
    <h2>Interactive 3D Cases</h2>
    <p class=\"muted\">Meshes shown as shaded LA/LAA surfaces with explicit wireframe edges, plus overlays: ostium center, ostium normal, LAA axis, proximal direction.</p>
    {''.join(case_blocks)}
  </div>
  <script>
    (function() {{
      const sections = document.querySelectorAll('.section');
      if (!('IntersectionObserver' in window)) {{
        sections.forEach((s) => s.classList.add('visible'));
        return;
      }}
      const obs = new IntersectionObserver((entries) => {{
        entries.forEach((entry) => {{
          if (entry.isIntersecting) {{
            entry.target.classList.add('visible');
            obs.unobserve(entry.target);
          }}
        }});
      }}, {{ threshold: 0.12 }});
      sections.forEach((s) => obs.observe(s));

      // Keep page scroll smooth even when cursor is over Plotly 3D canvases.
      document.querySelectorAll('.js-plotly-plot').forEach((el) => {{
        el.addEventListener('wheel', (e) => {{
          window.scrollBy({{ top: e.deltaY, left: 0, behavior: 'auto' }});
          e.preventDefault();
        }}, {{ passive: false }});
      }});
    }})();
  </script>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return str(output_html)


def main() -> int:
    args = parse_args()
    out = build_report(args)
    print(f"Saved LA/LAA report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
