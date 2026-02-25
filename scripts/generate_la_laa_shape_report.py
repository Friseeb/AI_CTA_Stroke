#!/usr/bin/env python3
"""Generate a presentation-friendly LA/LAA shape metrics report with 3D viewers.

Example:
  python3 scripts/generate_la_laa_shape_report.py \
    --metrics-csv /mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro/la_laa_metrics_batch.csv \
    --mesh-root /mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro \
    --output-html /mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro/la_laa_shape_report.html \
    --max-3d-cases 6
"""

from __future__ import annotations

import argparse
import base64
import io
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

from la_laa_metrics import load_mesh


DEFAULT_METRICS_CSV = "/mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro/la_laa_metrics_batch.csv"
DEFAULT_MESH_ROOT = "/mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro"
DEFAULT_OUTPUT_HTML = "/mnt/cta_ssd/daylightbids/derivatives/shape_meshes_repro/la_laa_shape_report.html"

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
    p.add_argument("--max-faces-per-mesh", type=int, default=45000, help="Downsample meshes for browser rendering")
    return p.parse_args()


def fmt_float(v: Any, nd: int = 3) -> str:
    try:
        fv = float(v)
    except Exception:
        return "" if v is None else str(v)
    if not np.isfinite(fv):
        return ""
    return f"{fv:.{nd}f}"


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
    return fmt_df_for_html(df).to_html(index=index, classes="tbl", border=0, escape=True)


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
    return f"<h4>{escape(title)}</h4><img class='plot' src='{data_uri}' alt='{escape(alt)}'/>"


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
        sns.barplot(data=qc, x="flag", y="n_true", color="#3d3d3d", ax=ax)
    else:
        ax.bar(qc["flag"], qc["n_true"], color="#3d3d3d")
    ax.set_xlabel("QC flag")
    ax.set_ylabel("Cases")
    ax.tick_params(axis="x", labelrotation=20)
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
        sns.histplot(s, bins=28, color="#4b4b4b", kde=True, ax=ax)
    else:
        ax.hist(s, bins=28, color="#4b4b4b", alpha=0.9)
    ax.set_xlabel(col)
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
    ax.scatter(d[x], d[y], s=11, c="#202020", alpha=0.75, linewidths=0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
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
    else:
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    _style_axis(ax)
    return _img_block("Correlation Heatmap (Spearman)", _fig_to_data_uri(fig), "Correlation heatmap")


def _find_case_mesh_path(case_dir: Path, case_id: str, suffix: str) -> Path | None:
    for ext in ("vtk", "vtp", "ply", "stl", "obj"):
        p = case_dir / f"{case_id}_{suffix}.{ext}"
        if p.exists():
            return p
    return None


def _pick_cases(df: pd.DataFrame, case_ids: list[str], max_cases: int) -> pd.DataFrame:
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"] == "success"]
    if "qc_exception" in work.columns:
        qx = pd.to_numeric(work["qc_exception"], errors="coerce").fillna(0).astype(bool)
        work = work[~qx]

    if case_ids:
        wanted = set(case_ids)
        work = work[work["case_id"].astype(str).isin(wanted)]
        return work.head(max_cases)

    sort_cols = [c for c in ["min_distance_mm", "ostium_planarity", "laa_axis_length_mm"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[True, True, False][: len(sort_cols)])
    return work.head(max_cases)


def _downsample_mesh(mesh, max_faces: int):
    n_faces = int(mesh.faces.shape[0])
    if n_faces <= max_faces:
        return mesh
    rng = np.random.default_rng(0)
    keep = rng.choice(n_faces, size=max_faces, replace=False)
    sub = mesh.submesh([keep], append=True, repair=False)
    sub.process(validate=True)
    return sub


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
        flatshading=False,
        lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.1, "roughness": 0.9},
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


def _make_case_figure(case_id: str, row: pd.Series, mesh_la, mesh_laa, max_faces: int):
    mesh_la = _downsample_mesh(mesh_la, max_faces=max(1000, int(max_faces)))
    mesh_laa = _downsample_mesh(mesh_laa, max_faces=max(1000, int(max_faces)))

    traces = [
        _mesh_trace(mesh_la, "LA", "#85c1e9", 0.32),
        _mesh_trace(mesh_laa, "LAA", "#e67e22", 0.62),
    ]

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
        traces.append(_marker_trace(c, "ostium center", "#1f1f1f"))
        if np.isfinite(n).all():
            traces.append(_vector_trace(c, n, 0.45 * base_len, "ostium normal", "#2e4053"))
        if np.isfinite(a).all():
            vec_len = float(axis_len) if np.isfinite(axis_len) and axis_len > 0 else 0.9 * base_len
            traces.append(_vector_trace(c, a, vec_len, "LAA axis", "#c0392b"))
        if np.isfinite(p).all():
            traces.append(_vector_trace(c, p, 0.6 * base_len, "proximal direction", "#117a65"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{case_id}: LA/LAA geometry and relational vectors",
        template="plotly_white",
        showlegend=True,
        legend={"orientation": "h", "y": -0.11},
        scene={
            "xaxis": {"title": "X (mm)", "backgroundcolor": "#fafafa", "gridcolor": "#e5e7e9"},
            "yaxis": {"title": "Y (mm)", "backgroundcolor": "#fafafa", "gridcolor": "#e5e7e9"},
            "zaxis": {"title": "Z (mm)", "backgroundcolor": "#fafafa", "gridcolor": "#e5e7e9"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.95}},
        },
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
    plots_html = "\n".join([p for p in plots if p])

    defs = _definitions_table(KEY_METRICS + QC_COLS)

    case_df = _pick_cases(df, case_ids=args.case_id, max_cases=int(args.max_3d_cases))
    case_blocks: list[str] = []

    if not HAS_PLOTLY:
        case_blocks.append("<p class='muted'>plotly not available; 3D section skipped.</p>")
    elif case_df.empty:
        case_blocks.append("<p class='muted'>No eligible successful cases found for 3D visualization.</p>")
    else:
        include_js: str | bool = "inline"
        for _, row in case_df.iterrows():
            case_id = str(row.get("case_id", ""))
            la_path, laa_path = _resolve_mesh_paths(row, mesh_root, args.la_suffix, args.laa_suffix)
            if la_path is None or laa_path is None:
                case_blocks.append(f"<h4>{escape(case_id)}</h4><p class='muted'>Missing mesh files for this case.</p>")
                continue

            try:
                mesh_la = load_mesh(str(la_path))
                mesh_laa = load_mesh(str(laa_path))
                fig = _make_case_figure(case_id, row, mesh_la, mesh_laa, max_faces=args.max_faces_per_mesh)
                fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
                include_js = False
                meta = pd.DataFrame(
                    {
                        "field": [
                            "case_id",
                            "la_path",
                            "laa_path",
                            "min_distance_mm",
                            "laa_axis_length_mm",
                            "bend_LAaxis_vs_LAAaxis_deg",
                            "bend_ostiumNormal_vs_proxLAA_deg",
                            "ostium_planarity",
                        ],
                        "value": [
                            case_id,
                            str(la_path),
                            str(laa_path),
                            row.get("min_distance_mm", np.nan),
                            row.get("laa_axis_length_mm", np.nan),
                            row.get("bend_LAaxis_vs_LAAaxis_deg", np.nan),
                            row.get("bend_ostiumNormal_vs_proxLAA_deg", np.nan),
                            row.get("ostium_planarity", np.nan),
                        ],
                    }
                )
                case_blocks.append(f"<h3>{escape(case_id)}</h3>{fig_html}{df_to_html(meta, index=False)}")
            except Exception as exc:  # noqa: BLE001
                case_blocks.append(
                    f"<h4>{escape(case_id)}</h4><p class='muted'>3D rendering failed: {escape(type(exc).__name__ + ': ' + str(exc))}</p>"
                )

    css = """
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 22px 24px; color: #111; background: #fff; }
h1, h2, h3, h4 { margin: 0.55em 0 0.32em; }
p { margin: 0.2em 0 0.7em; }
.muted { color: #575757; font-size: 0.96em; }
.note { background: #fafafa; border-left: 3px solid #c8c8c8; padding: 8px 10px; margin: 8px 0 14px; font-size: 13px; }
.tbl { border-collapse: collapse; width: 100%; margin: 8px 0 18px; font-size: 13px; }
.tbl th, .tbl td { border: 1px solid #e1e1e1; padding: 6px 8px; text-align: left; vertical-align: top; }
.tbl th { background: #f5f7f9; }
.plot { max-width: 100%; height: auto; border: 1px solid #ececec; margin: 4px 0 14px; }
.section { margin-bottom: 18px; }
code { background: #f2f2f2; padding: 1px 5px; border-radius: 4px; }
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
  <p class=\"muted\">Input metrics CSV: <code>{escape(str(metrics_csv))}</code></p>

  <div class=\"section\">
    <h2>Executive Summary</h2>
    {df_to_html(overview.set_index('metric'))}
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
    <h2>Definitions</h2>
    {df_to_html(defs, index=False) if not defs.empty else '<p class="muted">No definition rows available.</p>'}
  </div>

  <div class=\"section\">
    <h2>Interactive 3D Cases</h2>
    <p class=\"muted\">Meshes shown as semi-transparent LA (blue) and LAA (orange), with overlays: ostium center, ostium normal, LAA axis, proximal direction.</p>
    {''.join(case_blocks)}
  </div>
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
