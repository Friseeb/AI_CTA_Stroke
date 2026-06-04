#!/usr/bin/env python3
"""Generate an HTML report for radiomics exploratory clustering outputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

try:
    import seaborn as sns

    HAS_SNS = True
except Exception:  # noqa: BLE001
    HAS_SNS = False


FEATURE_DEFINITIONS = {
    "Ao_SVC_ratio": "How bright aorta is vs SVC; helps estimate contrast phase.",
    "PA_Ao_ratio": "How bright pulmonary artery is vs aorta; high can indicate earlier right-sided phase.",
    "LV_RV_ratio": "Brightness of LV compared with RV.",
    "Ao_IVC_ratio": "How bright aorta is vs IVC; another phase timing cue.",
    "LA_LAA_delta": "LA minus LAA average brightness (HU); larger gap suggests less LAA opacification.",
    "LA_LAA_ratio": "LAA brightness divided by LA brightness.",
    "Normalized_LAA_defect": "(LA-LAA)/Ao; a normalized filling-defect severity proxy.",
    "LAA_to_Ao_HU_ratio": "LAA brightness relative to aorta.",
    "LAA_to_LA_HU_ratio": "LAA brightness relative to LA.",
    "LAA_p10_to_Ao_HU_ratio": "Lower-end LAA brightness (10th percentile) relative to aorta.",
    "LAA_p90_p10_spread": "Internal LAA spread of intensities (P90-P10): broader = more mixed/heterogeneous.",
    "LAA_entropy": "Texture randomness: high entropy means more mixed/noisy voxel intensities.",
    "LAA_uniformity": "Texture uniformity: high means smoother/more similar intensities.",
    "LAA_variance": "How spread out LAA intensities are around the mean.",
    "LAA_glcm_Imc1": "Neighbor-pattern complexity from GLCM; captures local organization of intensities.",
    "LAA_glszm_SizeZoneNonUniformity": "How variable same-intensity patch sizes are (GLSZM).",
    "LAA_glszm_SizeZoneNonUniformityNormalized": "Size-variability measure normalized for region size.",
    "LAA_glszm_SmallAreaEmphasis": "How much tiny same-intensity patches dominate.",
    "LAA_gldm_DependenceVariance": "Variation in dependence (cluster-support) sizes in GLDM texture.",
    "LAA_volume_ml": "Appendage volume (mL).",
    "Paper2021_mix_vs_thrombus_proxy_zmean": "Combined paper-aligned texture score for mixing-vs-thrombus pattern.",
    "Paper2021_thrombus_vs_no_thrombus_proxy_zmean": "Combined paper-aligned score for thrombus-vs-no-thrombus pattern.",
    "Paper2021_transformed_feature_count": "How many transformed feature families are available for this case (0-5).",
}

RADIOMICS_GLOSSARY_ROWS = [
    {
        "term": "First-order",
        "plain_meaning": "Looks only at voxel intensity distribution (ignores spatial arrangement).",
        "relatable_example": "Like summarizing a class by average score and spread, not who sits next to whom.",
        "common_features": "Mean, variance, entropy, uniformity, percentiles.",
    },
    {
        "term": "GLCM",
        "plain_meaning": "Counts how often intensity pairs occur next to each other.",
        "relatable_example": "Like checking neighbor pairs in a mosaic: dark-next-to-dark vs dark-next-to-bright.",
        "common_features": "Contrast, correlation, IMC1, homogeneity.",
    },
    {
        "term": "GLSZM",
        "plain_meaning": "Measures sizes of connected same-intensity zones.",
        "relatable_example": "Like map regions of similar color: many tiny islands vs few large continents.",
        "common_features": "SZNU, SZNUN, small-area emphasis.",
    },
    {
        "term": "GLDM",
        "plain_meaning": "Quantifies how strongly voxels depend on surrounding similar voxels.",
        "relatable_example": "How stable a local neighborhood is around each voxel.",
        "common_features": "Dependence variance, dependence entropy.",
    },
    {
        "term": "Wavelet transform",
        "plain_meaning": "Separates image texture into coarse and fine frequency components.",
        "relatable_example": "Like splitting music into bass/mid/treble before analysis.",
        "common_features": "W-LLH, W-HLH etc + first-order/texture metrics.",
    },
    {
        "term": "LoG (log-sigma)",
        "plain_meaning": "Edge/spot-enhancing filter at a chosen spatial scale.",
        "relatable_example": "Using different magnifying lenses to emphasize small vs larger structures.",
        "common_features": "LoG1.0, LoG2.0, ... + texture features.",
    },
]

FEATURE_LABELS = {
    "Ao_SVC_ratio": "Ao/SVC",
    "PA_Ao_ratio": "PA/Ao",
    "LV_RV_ratio": "LV/RV",
    "Ao_IVC_ratio": "Ao/IVC",
    "LA_LAA_delta": "LA-LAA dMean",
    "LA_LAA_ratio": "LAA/LA",
    "LA_LAA_p10_delta": "LA-LAA dP10",
    "Ao_LA_delta": "Ao-LA dHU",
    "LV_LA_delta": "LV-LA dHU",
    "RA_RV_delta": "RA-RV dHU",
    "Normalized_LAA_defect": "Norm LAA defect",
    "LAA_to_Ao_HU_ratio": "LAA/Ao",
    "LAA_to_LA_HU_ratio": "LAA/LA HU",
    "LAA_minus_Ao_HU_delta": "LAA-Ao dHU",
    "LAA_minus_LA_HU_delta": "LAA-LA dHU",
    "LAA_p10_to_Ao_HU_ratio": "P10 LAA/Ao",
    "LAA_p90_p10_spread": "LAA P90-P10",
    "LAA_entropy": "LAA entropy",
    "LAA_uniformity": "LAA uniformity",
    "LAA_variance": "LAA variance",
    "LAA_glcm_Imc1": "GLCM IMC1",
    "LAA_glszm_SizeZoneNonUniformity": "GLSZM SZNU",
    "LAA_glszm_SizeZoneNonUniformityNormalized": "GLSZM SZNUN",
    "LAA_glszm_SmallAreaEmphasis": "GLSZM SAE",
    "LAA_gldm_DependenceVariance": "GLDM DepVar",
    "LAA_volume_ml": "LAA volume mL",
    "Paper2021_mix_vs_thrombus_proxy_zmean": "P2021 mix-thrombus z",
    "Paper2021_thrombus_vs_no_thrombus_proxy_zmean": "P2021 thrombus z",
    "Paper2021_transformed_feature_count": "P2021 transformed n",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate HTML report for radiomics clustering results.")
    p.add_argument(
        "--radiomics-dir",
        default="/mnt/cta_ssd/daylightbids/derivatives/radiomics",
        help="Directory containing clustering CSV/JSON outputs.",
    )
    p.add_argument(
        "--derived-csv",
        default=None,
        help="Derived metrics CSV path (default: <radiomics-dir>/radiomics_derived_metrics.csv).",
    )
    p.add_argument(
        "--summary-csv",
        default=None,
        help="Exploratory summary CSV path (default: <radiomics-dir>/radiomics_clusters_exploratory_summary.csv).",
    )
    p.add_argument(
        "--ari-csv",
        default=None,
        help="Pairwise ARI CSV path (default: <radiomics-dir>/radiomics_clusters_pairwise_ari.csv).",
    )
    p.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path (default: <radiomics-dir>/radiomics_clustering_report.html).",
    )
    p.add_argument(
        "--title",
        default="Radiomics Exploratory Clustering Report",
        help="HTML report title.",
    )
    return p.parse_args()


def read_ari(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "run"}).set_index("run")
    else:
        df = df.set_index(df.columns[0])
    return df


def fmt_float(v: float | int | np.floating | np.integer | None, nd: int = 4) -> str:
    if v is None:
        return ""
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(fv):
        return ""
    return f"{fv:.{nd}f}"


def format_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if any(tok in col.lower() for tok in ("silhouette", "ari", "pc", "ratio", "share")):
                out[col] = out[col].map(lambda v: fmt_float(v, nd=4))
            else:
                out[col] = out[col].map(
                    lambda v: fmt_float(v, nd=4) if isinstance(v, (float, np.floating)) else ("" if pd.isna(v) else str(v))
                )
    return out


def df_html(df: pd.DataFrame, classes: str = "tbl") -> str:
    safe = format_df_for_html(df)
    return safe.to_html(index=True, escape=True, classes=classes, border=0)


def radiomics_glossary_html() -> str:
    df = pd.DataFrame(RADIOMICS_GLOSSARY_ROWS)
    return df_html(df.set_index("term"))


def run_label(stem: str) -> str:
    mapping = {
        "radiomics_clusters_all": "All features",
        "radiomics_clusters_hemodynamics": "Hemodynamics",
        "radiomics_clusters_paper_focus": "Paper-focus",
        "radiomics_clusters_texture_only": "Texture-only",
        "radiomics_clusters_transformed_exploratory": "Transformed exploratory",
    }
    if stem in mapping:
        return mapping[stem]
    return stem.replace("radiomics_clusters_", "").replace("_", " ").title()


def _tokenize_feature(rest: str) -> str:
    out = rest
    repl = {
        "firstorder_": "FO ",
        "glcm_": "GLCM ",
        "glszm_": "GLSZM ",
        "gldm_": "GLDM ",
        "shape_": "Shape ",
        "MeshVolume": "MeshVol",
        "SizeZoneNonUniformityNormalized": "SZNUN",
        "SizeZoneNonUniformity": "SZNU",
        "SmallAreaEmphasis": "SAE",
        "DependenceVariance": "DepVar",
    }
    for old, new in repl.items():
        out = out.replace(old, new)
    return out.replace("_", " ").strip()


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    if name.startswith("wavelet-"):
        # Example: wavelet-LLH_firstorder_Entropy
        rest = name[len("wavelet-") :]
        if "_" in rest:
            band, feat = rest.split("_", 1)
            return f"W-{band.upper()} {_tokenize_feature(feat)}"
        return f"W-{rest}"

    if name.startswith("log-sigma-"):
        # Example: log-sigma-3-0-mm-3D_glcm_Contrast
        rest = name[len("log-sigma-") :]
        if "_" in rest:
            sigma_part, feat = rest.split("_", 1)
            sigma_text = sigma_part.replace("-mm-3D", "").replace("-", ".")
            return f"LoG{sigma_text} {_tokenize_feature(feat)}"
        return f"LoG {rest}"

    if name.startswith("square_"):
        return f"Sq {_tokenize_feature(name[len('square_'):])}"

    return _tokenize_feature(name)


def describe_feature(name: str) -> str:
    if name in FEATURE_DEFINITIONS:
        return FEATURE_DEFINITIONS[name]
    if name.startswith("wavelet-"):
        return "Wavelet-transformed radiomic feature (frequency-decomposed texture/intensity descriptor)."
    if name.startswith("log-sigma-"):
        return "LoG (log-sigma) transformed radiomic feature capturing scale-specific edge/texture patterns."
    if name.startswith("square_"):
        return "Square-transformed radiomic feature emphasizing higher intensities."
    return "Radiomic/derived feature used for cluster separation."


def collect_significant_features(radiomics_dir: Path, runs: list[str], top_n_per_run: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stem in runs:
        summary_path = radiomics_dir / f"{stem}_summary_means_zscaled.csv"
        if not summary_path.exists():
            continue
        sm = pd.read_csv(summary_path)
        if "cluster" not in sm.columns:
            continue
        feat_cols = [c for c in sm.columns if c not in {"cluster", "n_patients"}]
        ranges: list[tuple[str, float]] = []
        for c in feat_cols:
            vals = pd.to_numeric(sm[c], errors="coerce")
            if vals.notna().any():
                ranges.append((c, float(vals.max() - vals.min())))
        if not ranges:
            continue
        top = sorted(ranges, key=lambda x: x[1], reverse=True)[:top_n_per_run]
        for feat, ran in top:
            rows.append({"run": stem, "feature": feat, "zmean_range": ran})

    if not rows:
        return pd.DataFrame(
            columns=["feature", "descriptor", "max_zmean_range", "mean_zmean_range", "run_count", "run_max", "definition"]
        )

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("feature", as_index=False)
        .agg(
            max_zmean_range=("zmean_range", "max"),
            mean_zmean_range=("zmean_range", "mean"),
            run_count=("run", "nunique"),
        )
        .sort_values(["max_zmean_range", "run_count"], ascending=[False, False])
    )

    run_max = (
        df.sort_values("zmean_range", ascending=False)
        .drop_duplicates("feature")
        .set_index("feature")["run"]
    )
    agg["descriptor"] = agg["feature"].map(feature_label)
    agg["run_max"] = agg["feature"].map(run_max)
    agg["definition"] = agg["feature"].map(describe_feature)
    return agg


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    raw = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{raw}"


def _style_tufte(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, length=3, width=0.8)
    ax.grid(False)


def _init_plot_style() -> None:
    if HAS_SNS:
        sns.set_theme(style="white", context="paper")


def _img_block(title: str, data_uri: str, alt: str) -> str:
    return (
        f"<h4>{escape(title)}</h4>"
        f"<img class='plot' src='{data_uri}' alt='{escape(alt)}' />"
    )


def plot_run_silhouettes(summary: pd.DataFrame) -> str:
    if not HAS_MPL:
        return "<p class='muted'>matplotlib not available; plots skipped.</p>"

    _init_plot_style()
    x = summary.sort_values("silhouette", ascending=True).copy()
    y = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.hlines(y, xmin=0, xmax=x["silhouette"], color="#9a9a9a", linewidth=1.0)
    ax.plot(x["silhouette"], y, "o", color="#1f1f1f", markersize=4.0)
    ax.set_yticks(y)
    ax.set_yticklabels([run_label(r) for r in x["run"]])
    ax.set_xlabel("Silhouette")
    ax.set_xlim(left=0)
    _style_tufte(ax)
    return _img_block("Run Silhouette Comparison", _fig_to_data_uri(fig), "Silhouette comparison across runs")


def plot_transformed_coverage(derived: pd.DataFrame) -> str:
    if not HAS_MPL or "Paper2021_transformed_feature_count" not in derived.columns:
        return ""
    _init_plot_style()
    vc = derived["Paper2021_transformed_feature_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    if HAS_SNS:
        sns.barplot(x=vc.index.astype(str), y=vc.values, color="#404040", ax=ax)
    else:
        ax.bar(vc.index.astype(str), vc.values, color="#404040", width=0.7)
    ax.set_xlabel("Paper2021 transformed feature count")
    ax.set_ylabel("Patients")
    _style_tufte(ax)
    return _img_block("Transformed Feature Coverage", _fig_to_data_uri(fig), "Transformed feature coverage histogram")


def plot_ari_heatmap(ari: pd.DataFrame) -> str:
    if not HAS_MPL:
        return ""
    _init_plot_style()
    ari_disp = ari.copy()
    ari_disp.index = [run_label(str(i)) for i in ari_disp.index]
    ari_disp.columns = [run_label(str(c)) for c in ari_disp.columns]
    m = ari_disp.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if HAS_SNS:
        sns.heatmap(
            ari_disp.astype(float),
            cmap="Greys",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="#efefef",
            cbar=True,
            ax=ax,
            annot_kws={"fontsize": 7},
        )
    else:
        im = ax.imshow(m, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(ari_disp.columns)))
        ax.set_xticklabels(ari_disp.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(ari_disp.index)))
        ax.set_yticklabels(ari_disp.index, fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#222")
        fig.colorbar(im, ax=ax, fraction=0.048, pad=0.02)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    _style_tufte(ax)
    return _img_block("Pairwise ARI Heatmap", _fig_to_data_uri(fig), "ARI heatmap")


def plot_k_scan(ks: pd.DataFrame, title: str) -> str:
    if not HAS_MPL or ks.empty:
        return ""
    _init_plot_style()
    fig, ax = plt.subplots(figsize=(4.9, 2.5))
    if HAS_SNS:
        sns.lineplot(data=ks, x="k", y="silhouette", marker="o", color="#111111", linewidth=1.0, ax=ax)
    else:
        ax.plot(ks["k"], ks["silhouette"], "-o", color="#111111", linewidth=1.0, markersize=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_xticks(ks["k"])
    _style_tufte(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


def plot_cluster_sizes(size_df: pd.DataFrame, title: str) -> str:
    if not HAS_MPL or size_df.empty:
        return ""
    _init_plot_style()
    fig, ax = plt.subplots(figsize=(4.9, 2.5))
    x = size_df["cluster"].astype(str)
    y = size_df["n_patients"]
    if HAS_SNS:
        sns.barplot(x=x, y=y, color="#404040", ax=ax)
    else:
        ax.bar(x, y, color="#404040", width=0.7)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("N")
    _style_tufte(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


def plot_cluster_profile_heatmap(sm: pd.DataFrame, title: str) -> str:
    if not HAS_MPL:
        return ""
    if "cluster" not in sm.columns:
        return ""
    feat_cols = [c for c in sm.columns if c not in {"cluster", "n_patients"}]
    if not feat_cols:
        return ""

    ranges = []
    for c in feat_cols:
        vals = pd.to_numeric(sm[c], errors="coerce")
        if vals.notna().any():
            ranges.append((c, float(vals.max() - vals.min())))
    if not ranges:
        return ""
    top_features = [f for f, _ in sorted(ranges, key=lambda x: x[1], reverse=True)[:12]]
    mat = sm.set_index("cluster")[top_features].astype(float)
    label_map = {f: feature_label(f) for f in top_features}

    _init_plot_style()
    fig_h = max(2.2, 0.32 * mat.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    v = float(np.nanmax(np.abs(mat.to_numpy()))) if np.isfinite(mat.to_numpy()).any() else 1.0
    if v <= 0:
        v = 1.0
    if HAS_SNS:
        sns.heatmap(
            mat,
            cmap="vlag",
            center=0.0,
            vmin=-v,
            vmax=v,
            linewidths=0.3,
            linecolor="#efefef",
            cbar=True,
            ax=ax,
        )
    else:
        im = ax.imshow(mat.to_numpy(), cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([f"C{int(c)}" for c in mat.index], fontsize=8)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels([label_map[f] for f in top_features], rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    ax.set_yticklabels([f"C{int(c)}" for c in mat.index], fontsize=8)
    ax.set_xticklabels([label_map[f] for f in top_features], rotation=45, ha="right", fontsize=7)
    _style_tufte(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


def _resolve_existing_path(candidate: str | None, radiomics_dir: Path, fallback_name: str | None = None) -> Path | None:
    if candidate:
        p = Path(candidate)
        if p.exists():
            return p
        alt = radiomics_dir / p.name
        if alt.exists():
            return alt
    if fallback_name:
        fb = radiomics_dir / fallback_name
        if fb.exists():
            return fb
    return None


def _infer_shared_id_col(df_input: pd.DataFrame, df_patients: pd.DataFrame, meta_id_col: str | None) -> str | None:
    candidates = []
    if meta_id_col:
        candidates.append(meta_id_col)
    candidates.extend(["patient_id", "subject_id", "case_id"])
    for c in candidates:
        if c in df_input.columns and c in df_patients.columns:
            return c
    return None


def compute_run_pca3_points(radiomics_dir: Path, stem: str, meta: dict, patients_df: pd.DataFrame) -> pd.DataFrame | None:
    if "cluster" not in patients_df.columns:
        return None
    if "used_features" not in meta or not isinstance(meta.get("used_features"), list):
        return None
    used_features = [f for f in meta["used_features"] if isinstance(f, str)]
    if not used_features:
        return None

    input_path = _resolve_existing_path(
        str(meta.get("input_csv")) if meta.get("input_csv") is not None else None,
        radiomics_dir=radiomics_dir,
    )
    if input_path is None:
        return None

    try:
        src = pd.read_csv(input_path)
    except Exception:
        return None

    id_col = _infer_shared_id_col(src, patients_df, str(meta.get("id_column")) if meta.get("id_column") else None)
    if id_col is None:
        return None

    present_features = [c for c in used_features if c in src.columns]
    if len(present_features) < 2:
        return None

    src[id_col] = src[id_col].astype(str)
    pat = patients_df.copy()
    pat[id_col] = pat[id_col].astype(str)
    pat = pat.drop_duplicates(subset=[id_col], keep="first")

    src_idx = src.drop_duplicates(subset=[id_col], keep="first").set_index(id_col)
    wanted_ids = pat[id_col].tolist()
    available_ids = [pid for pid in wanted_ids if pid in src_idx.index]
    if len(available_ids) < 3:
        return None

    x = src_idx.loc[available_ids, present_features].apply(pd.to_numeric, errors="coerce")
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    stds = x.std(axis=0, ddof=0).to_numpy(dtype=float)
    keep = np.isfinite(stds) & (stds > 1e-12)
    if keep.sum() < 2:
        return None
    cols = [c for c, k in zip(x.columns, keep, strict=True) if k]
    x = x[cols]

    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    z = (x - means) / stds
    mat = z.to_numpy(dtype=float)
    mat0 = mat - mat.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(mat0, full_matrices=False)
    except Exception:
        return None

    n_comp = min(3, vt.shape[0], mat0.shape[1])
    comps = vt[:n_comp].T
    pcs = mat0 @ comps
    if n_comp < 3:
        pad = np.zeros((pcs.shape[0], 3 - n_comp), dtype=float)
        pcs = np.hstack([pcs, pad])

    cluster_map = pat.set_index(id_col)["cluster"].to_dict()
    out = pd.DataFrame(
        {
            "id": available_ids,
            "cluster": [cluster_map.get(i, np.nan) for i in available_ids],
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1],
            "pc3": pcs[:, 2],
        }
    )
    out = out.dropna(subset=["cluster"]).copy()
    if out.empty:
        return None
    out["cluster"] = out["cluster"].astype(int)
    return out


def plot_run_pca3d(points_df: pd.DataFrame, title: str) -> str:
    if not HAS_MPL or points_df is None or points_df.empty:
        return ""
    _init_plot_style()
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    clusters = sorted(points_df["cluster"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(clusters):
        sub = points_df[points_df["cluster"] == c]
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            sub["pc3"],
            s=16,
            alpha=0.86,
            color=cmap(i % 10),
            label=f"C{int(c)} (n={len(sub)})",
            edgecolors="none",
        )
    ax.set_xlabel("PC1", labelpad=6)
    ax.set_ylabel("PC2", labelpad=6)
    ax.set_zlabel("PC3", labelpad=6)
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=7, frameon=False, title="Cluster")
    return _img_block(title, _fig_to_data_uri(fig), title)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()


def plot_run_pca3d_interactive(points_df: pd.DataFrame, title: str, stem: str) -> str:
    if points_df is None or points_df.empty:
        return ""

    plot_id = f"pca3d-{_slug(stem)}"
    traces: list[dict[str, object]] = []
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for i, c in enumerate(sorted(points_df["cluster"].unique().tolist())):
        sub = points_df[points_df["cluster"] == c]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"C{int(c)} (n={len(sub)})",
                "x": sub["pc1"].tolist(),
                "y": sub["pc2"].tolist(),
                "z": sub["pc3"].tolist(),
                "text": sub["id"].astype(str).tolist(),
                "hovertemplate": "ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
                "marker": {
                    "size": 4,
                    "opacity": 0.85,
                    "color": palette[i % len(palette)],
                    "line": {"width": 0},
                },
            }
        )

    layout = {
        "title": {"text": title, "font": {"size": 14}},
        "margin": {"l": 0, "r": 0, "t": 36, "b": 0},
        "scene": {
            "xaxis": {"title": "PC1"},
            "yaxis": {"title": "PC2"},
            "zaxis": {"title": "PC3"},
        },
        "legend": {"title": {"text": "Cluster"}},
    }
    config = {"responsive": True, "displaylogo": False}

    traces_json = json.dumps(traces)
    layout_json = json.dumps(layout)
    config_json = json.dumps(config)
    return (
        f"<h4>{escape(title)}</h4>"
        f"<div id='{escape(plot_id)}' class='plotly-3d'></div>"
        "<script>"
        f"(function(){{"
        f"var el=document.getElementById('{escape(plot_id)}');"
        "if(!el){return;}"
        "if(window.Plotly){"
        f"Plotly.newPlot(el,{traces_json},{layout_json},{config_json});"
        "} else {"
        "el.innerHTML='<p class=\"muted\">Interactive Plotly library not loaded.</p>';"
        "}"
        "})();"
        "</script>"
    )


def discover_runs(radiomics_dir: Path) -> list[str]:
    stems: list[str] = []
    for p in sorted(radiomics_dir.glob("radiomics_clusters_*_metadata.json")):
        stem = p.name.replace("_metadata.json", "")
        if stem.endswith(("exploratory_summary", "pairwise_ari")):
            continue
        stems.append(stem)
    return stems


def top_feature_ranges(summary_means: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if "cluster" not in summary_means.columns:
        return pd.DataFrame(columns=["feature", "zmean_range"])
    feat_cols = [c for c in summary_means.columns if c not in {"cluster", "n_patients"}]
    ranges = []
    for c in feat_cols:
        vals = pd.to_numeric(summary_means[c], errors="coerce")
        if vals.notna().any():
            ranges.append((c, float(vals.max() - vals.min())))
    if not ranges:
        return pd.DataFrame(columns=["feature", "zmean_range"])
    top = (
        pd.DataFrame(ranges, columns=["feature", "zmean_range"])
        .sort_values("zmean_range", ascending=False)
        .head(top_n)
    )
    top["descriptor"] = top["feature"].map(feature_label)
    top["definition"] = top["feature"].map(describe_feature)
    return top


def render_cluster_interpretation(sm: pd.DataFrame) -> str:
    if "cluster" not in sm.columns:
        return ""
    feat_cols = [c for c in sm.columns if c not in {"cluster", "n_patients"}]
    if not feat_cols:
        return ""

    rows: list[dict[str, object]] = []
    sm_idx = sm.set_index("cluster")
    for cluster_id, row in sm_idx.iterrows():
        vals = pd.to_numeric(row[feat_cols], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        high = vals.sort_values(ascending=False).head(3)
        low = vals.sort_values(ascending=True).head(3)

        hi_names = [feature_label(c) for c in high.index]
        lo_names = [feature_label(c) for c in low.index]

        interpretation_bits: list[str] = []
        if "Normalized_LAA_defect" in high.index or "LA_LAA_delta" in high.index:
            interpretation_bits.append("relatively lower LAA opacification compared with LA/aorta references")
        if "LAA_to_Ao_HU_ratio" in high.index or "LAA_to_LA_HU_ratio" in high.index:
            interpretation_bits.append("relatively stronger appendage opacification")
        if "LAA_entropy" in high.index or "LAA_variance" in high.index:
            interpretation_bits.append("higher texture heterogeneity")
        if "LAA_uniformity" in high.index:
            interpretation_bits.append("more homogeneous intensities")
        if "Paper2021_mix_vs_thrombus_proxy_zmean" in high.index:
            interpretation_bits.append("higher Paper2021 mixing-vs-thrombus proxy score")
        if "Paper2021_thrombus_vs_no_thrombus_proxy_zmean" in high.index:
            interpretation_bits.append("higher Paper2021 thrombus-vs-no-thrombus proxy score")
        if not interpretation_bits:
            interpretation_bits.append("cluster is defined by a distinct multifeature radiomic profile")

        n_patients = pd.to_numeric(pd.Series([row.get("n_patients", np.nan)]), errors="coerce").iloc[0]
        rows.append(
            {
                "cluster": int(cluster_id),
                "n_patients": int(n_patients) if np.isfinite(n_patients) else np.nan,
                "highest_z_features": "; ".join(hi_names),
                "lowest_z_features": "; ".join(lo_names),
                "clinical_readout": "; ".join(dict.fromkeys(interpretation_bits)),
            }
        )

    if not rows:
        return ""
    out = pd.DataFrame(rows).sort_values("cluster")
    return df_html(out.set_index("cluster"))


def run_section(radiomics_dir: Path, stem: str) -> str:
    meta_path = radiomics_dir / f"{stem}_metadata.json"
    patients_path = radiomics_dir / f"{stem}_patients.csv"
    summary_path = radiomics_dir / f"{stem}_summary_means_zscaled.csv"
    kscan_path = radiomics_dir / f"{stem}_k_scan.csv"

    if not meta_path.exists():
        return f"<section class='run-card'><h3>{escape(run_label(stem))}</h3><p>Missing metadata file.</p></section>"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    stem_label = run_label(stem)
    blocks: list[str] = [f"<h3>{escape(stem_label)}</h3>", f"<p class='muted'><code>{escape(stem)}</code></p>"]
    blocks.append(
        "<p>"
        f"<b>k</b>: {escape(str(meta.get('selected_k', '')))} | "
        f"<b>silhouette</b>: {escape(fmt_float(meta.get('selected_silhouette', np.nan), nd=4))} | "
        f"<b>used features</b>: {escape(str(meta.get('n_used_features', '')))}"
        "</p>"
    )
    preprocess = meta.get("preprocess", {}) if isinstance(meta.get("preprocess", {}), dict) else {}
    pca_info = preprocess.get("pca", {}) if isinstance(preprocess.get("pca", {}), dict) else {}
    stability = (
        preprocess.get("bootstrap_stability", {})
        if isinstance(preprocess.get("bootstrap_stability", {}), dict)
        else {}
    )
    if preprocess:
        line = (
            f"<p class='muted'><b>Preprocess:</b> "
            f"min_nonnull={escape(fmt_float(preprocess.get('min_nonnull_ratio', np.nan), nd=2))}, "
            f"min_variance={escape(fmt_float(preprocess.get('min_variance', np.nan), nd=6))}, "
            f"max_corr={escape(fmt_float(preprocess.get('max_corr', np.nan), nd=3))}"
        )
        if bool(pca_info.get("applied", False)):
            line += (
                ", "
                f"PCA components={escape(str(pca_info.get('n_components', '')))} "
                f"(cum var={escape(fmt_float(pca_info.get('explained_variance_ratio_sum', np.nan), nd=3))})"
            )
        else:
            line += ", PCA not applied"
        line += "</p>"
        blocks.append(line)
    if bool(stability.get("enabled", False)):
        blocks.append(
            "<p class='muted'><b>Stability (bootstrap ARI):</b> "
            f"mean={escape(fmt_float(stability.get('ari_mean', np.nan), nd=4))}, "
            f"std={escape(fmt_float(stability.get('ari_std', np.nan), nd=4))}, "
            f"p05={escape(fmt_float(stability.get('ari_p05', np.nan), nd=4))}, "
            f"p95={escape(fmt_float(stability.get('ari_p95', np.nan), nd=4))}"
            "</p>"
        )

    if patients_path.exists():
        pat = pd.read_csv(patients_path)
        size = pat["cluster"].value_counts().sort_index().rename("n_patients").reset_index()
        size = size.rename(columns={"index": "cluster"})
        blocks.append("<h4>Cluster Sizes</h4>")
        blocks.append(f"<div class='tbl-wrap'>{df_html(size.set_index('cluster'))}</div>")
        blocks.append(plot_cluster_sizes(size, f"{stem_label}: Cluster Sizes"))
        pca3 = compute_run_pca3_points(radiomics_dir=radiomics_dir, stem=stem, meta=meta, patients_df=pat)
        if pca3 is not None:
            blocks.append(
                "<p class='muted'>PCA 3D projection of patient-level run features; points are colored by assigned cluster.</p>"
            )
            blocks.append(plot_run_pca3d_interactive(pca3, f"{stem_label}: PCA 3D Cluster Projection", stem=stem))

        if "LAA_HU_pattern_paper" in pat.columns:
            hu = pd.crosstab(pat["cluster"], pat["LAA_HU_pattern_paper"], normalize="index").round(4)
            blocks.append("<h4>LAA HU Pattern Share by Cluster</h4>")
            blocks.append(f"<div class='tbl-wrap'>{df_html(hu)}</div>")

    if kscan_path.exists():
        ks = pd.read_csv(kscan_path).sort_values("k")
        blocks.append("<h4>K Scan</h4>")
        blocks.append(f"<div class='tbl-wrap'>{df_html(ks.set_index('k'))}</div>")
        blocks.append(plot_k_scan(ks, f"{stem_label}: Silhouette vs k"))

    if summary_path.exists():
        sm = pd.read_csv(summary_path)
        top = top_feature_ranges(sm, top_n=12)
        if not top.empty:
            top3 = ", ".join(top["descriptor"].head(3).tolist())
            blocks.append(
                "<p class='muted'><b>Quick interpretation:</b> "
                f"Main separating features in this run are {escape(top3)}.</p>"
            )
        blocks.append(plot_cluster_profile_heatmap(sm, f"{stem_label}: Cluster Profile Heatmap (top features)"))
        interp_tbl = render_cluster_interpretation(sm)
        if interp_tbl:
            blocks.append("<h4>Cluster Interpretation (Exploratory)</h4>")
            blocks.append(
                "<p class='muted'>Uses cluster-level z-score signatures only; treat as hypothesis-generating and validate against outcomes.</p>"
            )
            blocks.append(f"<div class='tbl-wrap'>{interp_tbl}</div>")
        if not top.empty:
            blocks.append("<h4>Top Separating Features (by z-mean range across clusters)</h4>")
            top_disp = top[["descriptor", "feature", "zmean_range", "definition"]].copy()
            blocks.append(f"<div class='tbl-wrap'>{df_html(top_disp.set_index('descriptor'))}</div>")

    return "<section class='run-card'>" + "\n".join(blocks) + "</section>"


def main() -> int:
    args = parse_args()
    radiomics_dir = Path(args.radiomics_dir)
    if not radiomics_dir.exists():
        raise FileNotFoundError(f"Radiomics directory not found: {radiomics_dir}")

    derived_csv = Path(args.derived_csv) if args.derived_csv else radiomics_dir / "radiomics_derived_metrics.csv"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else radiomics_dir / "radiomics_clusters_exploratory_summary.csv"
    )
    ari_csv = Path(args.ari_csv) if args.ari_csv else radiomics_dir / "radiomics_clusters_pairwise_ari.csv"
    output_html = Path(args.output_html) if args.output_html else radiomics_dir / "radiomics_clustering_report.html"

    if not derived_csv.exists():
        raise FileNotFoundError(f"Derived CSV not found: {derived_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    if not ari_csv.exists():
        raise FileNotFoundError(f"ARI CSV not found: {ari_csv}")

    derived = pd.read_csv(derived_csv)
    summary = pd.read_csv(summary_csv)
    ari = read_ari(ari_csv)
    runs = discover_runs(radiomics_dir)
    transformed_note = ""
    transformed_stem = next((r for r in runs if "transformed" in r.lower()), None)
    if transformed_stem:
        t_meta_path = radiomics_dir / f"{transformed_stem}_metadata.json"
        if t_meta_path.exists():
            t_meta = json.loads(t_meta_path.read_text(encoding="utf-8"))
            t_pre = t_meta.get("preprocess", {}) if isinstance(t_meta.get("preprocess", {}), dict) else {}
            t_pca = t_pre.get("pca", {}) if isinstance(t_pre.get("pca", {}), dict) else {}
            t_stab = (
                t_pre.get("bootstrap_stability", {})
                if isinstance(t_pre.get("bootstrap_stability", {}), dict)
                else {}
            )
            transformed_note = (
                f"4) <code>{escape(transformed_stem)}</code> clusters directly on transformed radiomics "
                f"after non-null/variance/correlation filtering, "
                f"PCA (components={escape(str(t_pca.get('n_components', '')))}; "
                f"cum var={escape(fmt_float(t_pca.get('explained_variance_ratio_sum', np.nan), nd=3))}), "
                f"and bootstrap stability (ARI mean={escape(fmt_float(t_stab.get('ari_mean', np.nan), nd=4))}).<br/>"
            )

    n_patients = len(derived)
    transformed_col = "Paper2021_transformed_feature_count"
    transformed_complete = int((derived[transformed_col] >= 5).sum()) if transformed_col in derived.columns else 0
    transformed_any = int((derived[transformed_col] > 0).sum()) if transformed_col in derived.columns else 0
    transformed_pct = (100.0 * transformed_complete / n_patients) if n_patients else 0.0

    best_row = summary.sort_values("silhouette", ascending=False).iloc[0]

    overview_df = pd.DataFrame(
        [
            ["patients", n_patients],
            ["derived_columns", derived.shape[1]],
            ["best_run", best_row["run"]],
            ["best_k", int(best_row["selected_k"])],
            ["best_silhouette", float(best_row["silhouette"])],
            ["transformed_feature_count>=5", transformed_complete],
            ["transformed_feature_count>0", transformed_any],
            ["transformed_complete_pct", transformed_pct],
        ],
        columns=["metric", "value"],
    )

    sig_features = collect_significant_features(radiomics_dir=radiomics_dir, runs=runs, top_n_per_run=10).head(14)
    if not sig_features.empty:
        sig_features = sig_features[
            ["descriptor", "feature", "max_zmean_range", "mean_zmean_range", "run_count", "run_max", "definition"]
        ]

    summary_disp = summary.copy()
    if "run" in summary_disp.columns:
        summary_disp["run_label"] = summary_disp["run"].astype(str).map(run_label)
        summary_disp = summary_disp.set_index("run_label")
        summary_disp = summary_disp.drop(columns=["run"])
    ari_disp = ari.copy()
    ari_disp.index = [run_label(str(i)) for i in ari_disp.index]
    ari_disp.columns = [run_label(str(c)) for c in ari_disp.columns]

    report_sections = []
    for stem in runs:
        report_sections.append(run_section(radiomics_dir=radiomics_dir, stem=stem))

    css = """
:root {
  --bg: #eef2f5;
  --paper: #ffffff;
  --ink: #15202b;
  --muted: #5d6b78;
  --line: #d5dde5;
  --accent: #1f5a88;
  --accent-soft: #dceaf5;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: radial-gradient(circle at 10% -20%, #ffffff, #e9edf1 70%);
  font-family: "Source Sans 3", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
.container { max-width: 1380px; margin: 20px auto 40px; padding: 0 18px; }
.hero {
  background: linear-gradient(120deg, #163753, #2e6a93);
  color: #fff;
  border-radius: 14px;
  padding: 22px 24px 16px;
  box-shadow: 0 10px 26px rgba(15, 31, 47, 0.18);
}
h1 { margin: 0 0 4px; font-size: 1.58rem; letter-spacing: 0.2px; }
h2 { margin: 1.15rem 0 0.55rem; font-size: 1.14rem; color: #12324d; }
h3 { margin: 0; font-size: 1.05rem; color: #12324d; }
h4 { margin: 0.8rem 0 0.3rem; font-size: 0.95rem; color: #274660; }
p { margin: 0.22rem 0 0.72rem; line-height: 1.42; }
.muted { color: var(--muted); font-size: 0.94rem; }
.note {
  background: var(--accent-soft);
  border: 1px solid #bed4e6;
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 10px 12px;
  margin: 8px 0 14px;
  font-size: 0.92rem;
}
.section-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 14px 6px;
  margin-top: 14px;
  box-shadow: 0 2px 8px rgba(20, 32, 43, 0.04);
}
.tbl-wrap { overflow-x: auto; width: 100%; }
.tbl { border-collapse: separate; border-spacing: 0; width: 100%; margin: 8px 0 18px; font-size: 12.5px; background: #fff; }
.tbl th, .tbl td { border: 1px solid #e0e6ec; padding: 6px 8px; text-align: left; vertical-align: top; }
.tbl th { background: #f6f9fc; font-weight: 600; }
.tbl tr:nth-child(even) td { background: #fbfcfd; }
.run-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(510px, 1fr)); gap: 14px; align-items: start; }
.run-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 14px 6px;
  box-shadow: 0 2px 8px rgba(20, 32, 43, 0.04);
}
code { background: #f0f4f7; border: 1px solid #d9e2eb; padding: 1px 5px; border-radius: 4px; }
.plot { max-width: 100%; height: auto; border: 1px solid #dce3ea; border-radius: 8px; margin: 6px 0 14px; background: #fff; }
.plotly-3d { width: 100%; min-height: 480px; border: 1px solid #dce3ea; border-radius: 8px; margin: 6px 0 14px; background: #fff; }
@media (max-width: 900px) {
  .run-grid { grid-template-columns: 1fr; }
  .plotly-3d { min-height: 420px; }
}
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(args.title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
  <style>{css}</style>
</head>
<body>
  <div class="container">
  <div class="hero">
    <h1>{escape(args.title)}</h1>
    <p class="muted" style="color:#d4e4f2;">Generated: {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
    <p class="muted" style="color:#d4e4f2;">Inputs: <code>{escape(str(derived_csv))}</code>, <code>{escape(str(summary_csv))}</code>, <code>{escape(str(ari_csv))}</code></p>
  </div>

  <section class="section-card">
  <h2>Overview</h2>
  <div class="tbl-wrap">{df_html(overview_df.set_index("metric"))}</div>
  <div class="note">
    <b>How to read this report (succinct):</b><br/>
    1) Higher <b>silhouette</b> means cleaner cluster separation.<br/>
    2) <b>ARI</b> (Adjusted Rand Index) close to 1 means two runs produce very similar patient partitions.<br/>
    3) Heatmaps show cluster-level z-scaled feature means; red/blue indicate relative high/low values within that run.<br/>
    4) "Top separating features" are those with largest between-cluster spread (not inferential p-values).<br/>
    5) Bootstrap ARI (if shown) measures cluster assignment stability under subsampling.
  </div>
  </section>

  <section class="section-card">
  <h2>Run Comparison</h2>
  <div class="tbl-wrap">{df_html(summary_disp)}</div>
  {plot_run_silhouettes(summary)}
  {plot_transformed_coverage(derived)}
  </section>

  <section class="section-card">
  <h2>Radiomics Glossary (Plain Language)</h2>
  <p class="muted">Quick, clinician-friendly interpretation of common radiomics families and transforms used in this report.</p>
  <div class="tbl-wrap">{radiomics_glossary_html()}</div>
  <div class="note">
    <b>How to interpret direction (high vs low):</b><br/>
    1) <b>Higher entropy/variance/SZNU</b> usually means more heterogeneity (more mixed texture).<br/>
    2) <b>Higher uniformity</b> usually means smoother/more homogeneous appearance.<br/>
    3) Effects can vary by acquisition phase and preprocessing, so use cluster context and outcomes for interpretation.
  </div>
  </section>

  <section class="section-card">
  <h2>Pairwise ARI Matrix</h2>
  <div class="tbl-wrap">{df_html(ari_disp)}</div>
  {plot_ari_heatmap(ari)}
  </section>

  <section class="section-card">
  <h2>Significant Features (Exploratory)</h2>
  <p class="muted">Features below are ranked by maximum cluster-separation strength (z-mean range) across runs, with concise definitions.</p>
  <div class="tbl-wrap">{df_html(sig_features.set_index("descriptor")) if not sig_features.empty else "<p class='muted'>No feature-separation table available.</p>"}</div>
  </section>

  <section class="section-card">
  <h2>Paper Transformation Context</h2>
  <div class="note">
    <b>What is represented in this dataset:</b><br/>
    1) This report currently contains paper-aligned composite proxies (<code>Paper2021_*</code>) and per-case transformed-feature coverage (<code>Paper2021_transformed_feature_count</code>).<br/>
    2) A count of 5 means all planned transformed feature families were available for that case in the pipeline; lower counts indicate partial availability.<br/>
    3) The composite proxy scores summarize transformed-feature patterns into robust, lower-dimensional signals for clustering.<br/>
    {transformed_note if transformed_note else ""}
  </div>
  </section>

  <section class="section-card">
  <h2>Per-Run Details</h2>
  <div class="run-grid">
    {"".join(report_sections)}
  </div>
  </section>
  </div>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML report: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
