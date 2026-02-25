#!/usr/bin/env python3
"""Generate an HTML report for radiomics exploratory clustering outputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import datetime
from html import escape
from pathlib import Path

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


FEATURE_DEFINITIONS = {
    "Ao_SVC_ratio": "Aortic-to-SVC attenuation ratio; proxy for arterial contrast dominance.",
    "PA_Ao_ratio": "Pulmonary artery-to-aorta attenuation ratio; high values suggest earlier right-sided phase.",
    "LV_RV_ratio": "Left-to-right ventricular attenuation ratio.",
    "Ao_IVC_ratio": "Aortic-to-IVC attenuation ratio; additional phase-timing proxy.",
    "LA_LAA_delta": "Mean attenuation difference (LA minus LAA); larger values can indicate poorer appendage opacification.",
    "LA_LAA_ratio": "LAA attenuation normalized by LA attenuation.",
    "Normalized_LAA_defect": "(LA - LAA)/Ao; defect severity normalized to arterial enhancement.",
    "LAA_to_Ao_HU_ratio": "LAA attenuation normalized to aortic attenuation.",
    "LAA_to_LA_HU_ratio": "LAA attenuation normalized to LA attenuation.",
    "LAA_p10_to_Ao_HU_ratio": "Low-end LAA attenuation normalized to aorta.",
    "LAA_p90_p10_spread": "Within-LAA intensity spread (90th - 10th percentile).",
    "LAA_entropy": "First-order entropy; higher values indicate broader intensity randomness.",
    "LAA_uniformity": "First-order uniformity; higher values indicate more homogeneous intensities.",
    "LAA_variance": "First-order intensity variance in LAA.",
    "LAA_glcm_Imc1": "GLCM informational measure of correlation (IMC1); texture organization/complexity marker.",
    "LAA_glszm_SizeZoneNonUniformity": "GLSZM size-zone non-uniformity; higher values indicate more variable homogeneous zone sizes.",
    "LAA_glszm_SizeZoneNonUniformityNormalized": "Normalized GLSZM size-zone non-uniformity.",
    "LAA_glszm_SmallAreaEmphasis": "GLSZM small-area emphasis; higher values indicate prominence of small homogeneous zones.",
    "LAA_gldm_DependenceVariance": "GLDM dependence variance; dispersion of dependence sizes.",
    "LAA_volume_ml": "LAA mesh volume in mL.",
    "Paper2021_mix_vs_thrombus_proxy_zmean": "Composite proxy score (z-mean) aligned with mixing-vs-thrombus feature family.",
    "Paper2021_thrombus_vs_no_thrombus_proxy_zmean": "Composite proxy score (z-mean) aligned with thrombus-vs-no-thrombus family.",
    "Paper2021_transformed_feature_count": "Number of transformed-feature components available per case (0-5).",
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
        return pd.DataFrame(columns=["feature", "max_zmean_range", "mean_zmean_range", "run_count", "run_max", "definition"])

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
    ax.set_yticklabels(x["run"])
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
    m = ari.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if HAS_SNS:
        sns.heatmap(
            ari.astype(float),
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
        ax.set_xticks(np.arange(len(ari.columns)))
        ax.set_xticklabels(ari.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(ari.index)))
        ax.set_yticklabels(ari.index, fontsize=8)
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
        ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    ax.set_yticklabels([f"C{int(c)}" for c in mat.index], fontsize=8)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
    _style_tufte(ax)
    return _img_block(title, _fig_to_data_uri(fig), title)


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
    return (
        pd.DataFrame(ranges, columns=["feature", "zmean_range"])
        .sort_values("zmean_range", ascending=False)
        .head(top_n)
    )


def run_section(radiomics_dir: Path, stem: str) -> str:
    meta_path = radiomics_dir / f"{stem}_metadata.json"
    patients_path = radiomics_dir / f"{stem}_patients.csv"
    summary_path = radiomics_dir / f"{stem}_summary_means_zscaled.csv"
    kscan_path = radiomics_dir / f"{stem}_k_scan.csv"

    if not meta_path.exists():
        return f"<h3>{escape(stem)}</h3><p>Missing metadata file.</p>"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    blocks: list[str] = [f"<h3>{escape(stem)}</h3>"]
    blocks.append(
        "<p>"
        f"<b>k</b>: {escape(str(meta.get('selected_k', '')))} | "
        f"<b>silhouette</b>: {escape(fmt_float(meta.get('selected_silhouette', np.nan), nd=4))} | "
        f"<b>used features</b>: {escape(str(meta.get('n_used_features', '')))}"
        "</p>"
    )

    if patients_path.exists():
        pat = pd.read_csv(patients_path)
        size = pat["cluster"].value_counts().sort_index().rename("n_patients").reset_index()
        size = size.rename(columns={"index": "cluster"})
        blocks.append("<h4>Cluster Sizes</h4>")
        blocks.append(df_html(size.set_index("cluster")))
        blocks.append(plot_cluster_sizes(size, f"{stem}: Cluster Sizes"))

        if "LAA_HU_pattern_paper" in pat.columns:
            hu = pd.crosstab(pat["cluster"], pat["LAA_HU_pattern_paper"], normalize="index").round(4)
            blocks.append("<h4>LAA HU Pattern Share by Cluster</h4>")
            blocks.append(df_html(hu))

    if kscan_path.exists():
        ks = pd.read_csv(kscan_path).sort_values("k")
        blocks.append("<h4>K Scan</h4>")
        blocks.append(df_html(ks.set_index("k")))
        blocks.append(plot_k_scan(ks, f"{stem}: Silhouette vs k"))

    if summary_path.exists():
        sm = pd.read_csv(summary_path)
        top = top_feature_ranges(sm, top_n=12)
        if not top.empty:
            top3 = ", ".join(top["feature"].head(3).tolist())
            blocks.append(
                "<p class='muted'><b>Quick interpretation:</b> "
                f"Main separating features in this run are {escape(top3)}.</p>"
            )
        blocks.append(plot_cluster_profile_heatmap(sm, f"{stem}: Cluster Profile Heatmap (top features)"))
        if not top.empty:
            blocks.append("<h4>Top Separating Features (by z-mean range across clusters)</h4>")
            blocks.append(df_html(top.set_index("feature")))

    return "\n".join(blocks)


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

    report_sections = []
    for stem in runs:
        report_sections.append(run_section(radiomics_dir=radiomics_dir, stem=stem))

    css = """
body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
h1, h2, h3, h4 { margin: 0.6em 0 0.35em; }
p { margin: 0.2em 0 0.8em; }
.muted { color: #555; }
.note { background: #fafafa; border-left: 3px solid #d9d9d9; padding: 8px 10px; margin: 6px 0 14px; font-size: 13px; }
.tbl { border-collapse: collapse; width: 100%; margin: 8px 0 20px; font-size: 13px; }
.tbl th, .tbl td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
.tbl th { background: #f5f7fb; }
.grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
code { background: #f3f3f3; padding: 2px 5px; border-radius: 4px; }
.plot { max-width: 100%; height: auto; border: 1px solid #e6e6e6; margin: 6px 0 16px; }
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(args.title)}</title>
  <style>{css}</style>
</head>
<body>
  <h1>{escape(args.title)}</h1>
  <p class="muted">Generated: {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
  <p class="muted">Inputs: <code>{escape(str(derived_csv))}</code>, <code>{escape(str(summary_csv))}</code>, <code>{escape(str(ari_csv))}</code></p>

  <h2>Overview</h2>
  {df_html(overview_df.set_index("metric"))}
  <div class="note">
    <b>How to read this report (succinct):</b><br/>
    1) Higher <b>silhouette</b> means cleaner cluster separation.<br/>
    2) <b>ARI</b> (Adjusted Rand Index) close to 1 means two runs produce very similar patient partitions.<br/>
    3) Heatmaps show cluster-level z-scaled feature means; red/blue indicate relative high/low values within that run.<br/>
    4) "Top separating features" are those with largest between-cluster spread (not inferential p-values).
  </div>

  <h2>Run Comparison</h2>
  {df_html(summary.set_index("run"))}
  {plot_run_silhouettes(summary)}
  {plot_transformed_coverage(derived)}

  <h2>Pairwise ARI Matrix</h2>
  {df_html(ari)}
  {plot_ari_heatmap(ari)}

  <h2>Significant Features (Exploratory)</h2>
  <p class="muted">Features below are ranked by maximum cluster-separation strength (z-mean range) across runs, with concise definitions.</p>
  {df_html(sig_features.set_index("feature")) if not sig_features.empty else "<p class='muted'>No feature-separation table available.</p>"}

  <h2>Per-Run Details</h2>
  <div class="grid">
    {"".join(report_sections)}
  </div>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML report: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
