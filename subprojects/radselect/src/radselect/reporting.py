"""HTML reporting for radselect runs."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .core import RadselectResult, selected_feature_frequency


def write_html_report(result: RadselectResult, outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "radselect_report.html"
    assets_dir = out / "report_assets"
    figure_rows, figure_notes = write_report_figures(result, assets_dir)
    top_stable = result.stability_selection.copy()
    if not top_stable.empty:
        top_stable = top_stable.sort_values(["modality", "selection_probability"], ascending=[True, False]).head(50)
    frequency = selected_feature_frequency(result.selected_features, result.stability_selection)
    modality_summary = modality_audit_summary(result.modality_audit)
    correlation_summary = correlation_audit_summary(result.correlation_audit)
    flagged_columns = result.column_audit[result.column_audit.get("leakage_risk", False) == True].copy()
    metadata_flags = feature_metadata_flags(result.feature_metadata_audit)
    schema_flags = (
        result.schema_audit[result.schema_audit["issue"].astype(str).ne("")].copy()
        if "issue" in result.schema_audit.columns
        else pd.DataFrame()
    )
    robustness_flags = (
        result.robustness_audit[result.robustness_audit["filter_decision"].astype(str).eq("rejected")].copy()
        if "filter_decision" in result.robustness_audit.columns
        else pd.DataFrame()
    )
    dropped_samples = (
        result.sample_audit[result.sample_audit["status"] == "dropped"].copy()
        if "status" in result.sample_audit.columns
        else pd.DataFrame()
    )
    sections = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>radselect report</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;margin:36px;max-width:1180px;color:#1f2328}",
        "h1,h2{font-weight:500} table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}",
        "th{background:#f6f6f6}.muted{color:#666}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}",
        "code{background:#f6f6f6;padding:1px 4px}",
        "</style></head><body>",
        "<h1>radselect report</h1>",
        "<p class='muted'>Leakage-safe feature selection on already-extracted tabular features.</p>",
        "<h2>Pipeline</h2>",
        "<ol>",
        *[f"<li>{html.escape(step)}</li>" for step in result.manifest.get("selection_pipeline", [])],
        "</ol>",
        "<h2>Analysis method</h2>",
        f"<pre>{html.escape(json.dumps(result.manifest.get('analysis_method', {}), indent=2))}</pre>",
        "<h2>Outcome summary</h2>",
        f"<pre>{html.escape(json.dumps(result.manifest.get('outcome_summary', {}), indent=2))}</pre>",
        "<h2>Quality checks</h2>",
        frame_to_html(result.quality_checks),
        "<h2>Dependency audit</h2>",
        frame_to_html(result.dependency_audit),
        "<h2>Modality audit</h2>",
        frame_to_html(modality_summary)
        if not modality_summary.empty
        else "<p class='muted'>No modality/domain definitions were recorded.</p>",
        frame_to_html(result.modality_audit.head(200)),
        "<h2>Correlation redundancy audit</h2>",
        frame_to_html(correlation_summary)
        if not correlation_summary.empty
        else "<p class='muted'>No above-threshold correlation redundancy decisions were recorded.</p>",
        frame_to_html(result.correlation_audit.head(200)),
        "<h2>Figures</h2>",
        figures_to_html(figure_rows, figure_notes),
        "<div class='grid'>",
        "<section><h2>Performance</h2>",
        frame_to_html(result.performance),
        "</section>",
        "<section><h2>Projection validation</h2>",
        frame_to_html(result.projection_performance),
        "</section>",
        "</div>",
        "<div class='grid'>",
        "<section><h2>Composite scores</h2>",
        frame_to_html(result.composite_scores.head(100)),
        "</section>",
        "<section><h2>Stable features</h2>",
        frame_to_html(top_stable),
        "</section>",
        "</div>",
        "<h2>Final refit signature</h2>",
        frame_to_html(result.final_signature.head(200)),
        "<h2>Final refit parameters</h2>",
        frame_to_html(result.final_signature_parameters.head(200)),
        "<h2>Final refit scores</h2>",
        frame_to_html(result.final_composite_scores.head(100)),
        "<h2>Final projection scores</h2>",
        frame_to_html(result.final_projection_scores.head(100)),
        "<h2>Final projection parameters</h2>",
        frame_to_html(result.final_projection_parameters.head(200)),
        "<h2>Nested tuning</h2>",
        frame_to_html(result.tuning_summary[result.tuning_summary.get("selected", False) == True].head(100))
        if not result.tuning_summary.empty
        else "<p class='muted'>Inner-loop tuning was not run.</p>",
        "<h2>Validation splits</h2>",
        frame_to_html(result.validation_splits.head(200)),
        "<h2>Feature frequency</h2>",
        frame_to_html(frequency.head(200)),
        "<h2>Stability resamples</h2>",
        frame_to_html(result.stability_resamples.head(200)),
        "<h2>Feature metadata</h2>",
        frame_to_html(metadata_flags.head(100))
        if not metadata_flags.empty
        else "<p class='muted'>No feature metadata rows were rejected or flagged as non-compliant.</p>",
        "<h2>Robustness audit</h2>",
        frame_to_html(robustness_flags.head(100))
        if not robustness_flags.empty
        else "<p class='muted'>No features were rejected by robustness screening.</p>",
        "<h2>Sample audit</h2>",
        frame_to_html(dropped_samples.head(100))
        if not dropped_samples.empty
        else "<p class='muted'>No rows were dropped for missing or invalid outcomes/time/event fields.</p>",
        "<h2>Schema audit</h2>",
        frame_to_html(schema_flags.head(100))
        if not schema_flags.empty
        else "<p class='muted'>No external schema issues were detected.</p>",
        "<h2>Column audit</h2>",
        frame_to_html(flagged_columns.head(100))
        if not flagged_columns.empty
        else "<p class='muted'>No candidate columns were flagged as outcome-like leakage risks.</p>",
        "<h2>Selected features</h2>",
        frame_to_html(result.selected_features.head(200)),
        "<h2>Dropped features</h2>",
        frame_to_html(result.dropped_features.head(200)),
        "<h2>Manifest</h2>",
        f"<pre>{html.escape(json.dumps(result.manifest, indent=2))}</pre>",
        "</body></html>",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def modality_audit_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "modality" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for modality, group in frame.groupby("modality", sort=False):
        if "included_in_modality" in group.columns:
            included = group["included_in_modality"].astype(bool)
        else:
            included = pd.Series(False, index=group.index)
        rejected = (
            group["rejected_by_metadata_or_robustness"].astype(bool)
            if "rejected_by_metadata_or_robustness" in group.columns
            else pd.Series(False, index=group.index)
        )
        rows.append(
            {
                "modality": modality,
                "listed_features": int(group["feature"].nunique()) if "feature" in group.columns else int(len(group)),
                "included_features": int(group.loc[included, "feature"].nunique())
                if "feature" in group.columns
                else int(included.sum()),
                "rejected_by_metadata_or_robustness": int(rejected.sum()),
                "not_included": int((~included).sum()),
                "source": ";".join(sorted(group["source"].dropna().astype(str).unique()))
                if "source" in group.columns
                else "",
            }
        )
    return pd.DataFrame(rows)


def correlation_audit_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"modality", "fold", "dropped_feature"}.issubset(frame.columns):
        return pd.DataFrame()
    grouped = frame.groupby(["modality", "fold"], sort=False)
    rows = []
    for (modality, fold), group in grouped:
        rows.append(
            {
                "modality": modality,
                "fold": fold,
                "redundant_features_dropped": int(group["dropped_feature"].nunique()),
                "decisions": int(len(group)),
                "max_abs_correlation": float(group["abs_correlation"].max())
                if "abs_correlation" in group.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def feature_metadata_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    flags = frame.copy()
    rejected = (
        flags["filter_decision"].astype(str).eq("rejected")
        if "filter_decision" in flags.columns
        else pd.Series(False, index=flags.index)
    )
    noncompliant = (
        flags["ibsi_compliant"].map(is_false_like)
        if "ibsi_compliant" in flags.columns
        else pd.Series(False, index=flags.index)
    )
    return flags[rejected | noncompliant]


def is_false_like(value: object) -> bool:
    if value is False:
        return True
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"0", "false", "no", "n", "fail", "failed", "noncompliant"}


def frame_to_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='muted'>No rows.</p>"
    return frame.to_html(index=False, escape=True, max_rows=200)


def write_report_figures(result: RadselectResult, assets_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], [f"Plotting unavailable: {exc}"]

    assets_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []
    notes: list[str] = []
    performance_path = assets_dir / "performance_summary.png"
    if plot_performance(result.performance, performance_path, plt):
        figures.append(("Performance summary", relative_asset(performance_path, assets_dir.parent)))
    else:
        notes.append("No plottable performance metrics were available.")

    stability_path = assets_dir / "stability_selection.png"
    if plot_stability(result.stability_selection, stability_path, plt):
        figures.append(("Stability selection", relative_asset(stability_path, assets_dir.parent)))
    else:
        notes.append("No stability-selection probabilities were available.")

    projection_path = assets_dir / "projection_scores.png"
    if plot_projection(result.projection_scores, projection_path, plt):
        figures.append(("Projection scores", relative_asset(projection_path, assets_dir.parent)))

    return figures, notes


def plot_performance(performance: pd.DataFrame, path: Path, plt) -> bool:
    if performance.empty:
        return False
    metric_columns = [
        column
        for column in [
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "accuracy",
            "roc_auc_ovr",
            "r2",
            "mae",
            "rmse",
            "c_index",
        ]
        if column in performance.columns and pd.to_numeric(performance[column], errors="coerce").notna().any()
    ]
    if not metric_columns:
        return False
    summary = (
        performance.melt(id_vars=["modality"], value_vars=metric_columns, var_name="metric", value_name="value")
        .assign(value=lambda frame: pd.to_numeric(frame["value"], errors="coerce"))
        .dropna(subset=["value"])
        .groupby(["modality", "metric"], as_index=False)["value"]
        .mean()
    )
    if summary.empty:
        return False
    labels = [f"{row.modality}\n{row.metric}" for row in summary.itertuples()]
    fig_width = max(7.5, min(16.0, 0.5 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(range(len(summary)), summary["value"], color="#536b78", width=0.72)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean outer-fold metric")
    ax.set_title("Validation performance")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_stability(stability: pd.DataFrame, path: Path, plt) -> bool:
    if stability.empty or "selection_probability" not in stability.columns:
        return False
    top = stability.copy()
    top["selection_probability"] = pd.to_numeric(top["selection_probability"], errors="coerce")
    top = top.dropna(subset=["selection_probability"]).sort_values("selection_probability", ascending=False).head(25)
    if top.empty:
        return False
    labels = [f"{row.modality}: {row.feature}" for row in top.itertuples()]
    fig_height = max(4.8, 0.28 * len(top))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(range(len(top)), top["selection_probability"], color="#4f7f6f")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection probability")
    ax.set_title("Most stable selected features")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_projection(scores: pd.DataFrame, path: Path, plt) -> bool:
    if scores.empty or not {"component_1", "component_2", "modality"}.issubset(scores.columns):
        return False
    plot_data = scores.copy()
    plot_data["component_1"] = pd.to_numeric(plot_data["component_1"], errors="coerce")
    plot_data["component_2"] = pd.to_numeric(plot_data["component_2"], errors="coerce")
    plot_data = plot_data.dropna(subset=["component_1", "component_2"])
    if plot_data.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for modality, group in plot_data.groupby("modality"):
        ax.scatter(group["component_1"], group["component_2"], s=24, alpha=0.75, label=str(modality))
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Projection scores")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)


def figures_to_html(figures: list[tuple[str, str]], notes: list[str]) -> str:
    parts = []
    if figures:
        for title, src in figures:
            parts.append(
                f"<figure><img src='{html.escape(src)}' alt='{html.escape(title)}' "
                "style='max-width:100%;height:auto;border:1px solid #eee'>"
                f"<figcaption>{html.escape(title)}</figcaption></figure>"
            )
    for note in notes:
        parts.append(f"<p class='muted'>{html.escape(note)}</p>")
    if not parts:
        return "<p class='muted'>No figures were generated.</p>"
    return "\n".join(parts)


def relative_asset(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
