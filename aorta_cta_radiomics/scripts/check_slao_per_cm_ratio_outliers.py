#!/usr/bin/env python
"""Surface outliers among aorta-size-normalized per-cm ratio features."""

from __future__ import annotations

import argparse
import html
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

# The local Anaconda bottleneck extension is compiled against an older NumPy ABI.
# This report does not need bottleneck, so keep pandas on its pure-numpy path.
fake_bottleneck = types.ModuleType("bottleneck")
fake_bottleneck.__version__ = "999.0.0"
sys.modules.setdefault("bottleneck", fake_bottleneck)

import numpy as np
import pandas as pd

pd.set_option("compute.use_bottleneck", False)


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao" / "slao_etiology_aorta_modeling.csv"
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao" / "ratio_outliers"

CASE_ID = "case_id"
SOURCE_LABEL = "source_etiology_label"
PER_CM_SUFFIX = "__per_aortic_length_cm"

DENOMINATOR_PRIORITY = [
    "aorta__calcium_omics__aortic_length_cm__thr_dynamic_lumen_referenced_seed500HU",
    "aorta__calcium_omics__aortic_length_cm__thr_300HU",
]

DOMAIN_LABELS = {
    "calcium": "Calcium",
    "fat": "Periaortic fat",
    "wall_from_fat": "Wall-from-fat",
    "wall_thickness": "Wall thickness",
}

ETIOLOGY_ORDER = ("ESUS", "KAF", "AFDAS", "New_ECG_AF")
ETIOLOGY_LABELS = {
    "ESUS": "ESUS",
    "KAF": "KAF",
    "AFDAS": "AFDAS",
    "New_ECG_AF": "ECG-AF",
}
ETIOLOGY_COLORS = {
    "ESUS": "#4c78a8",
    "KAF": "#b45f06",
    "AFDAS": "#6a994e",
    "New_ECG_AF": "#7b2cbf",
}


def main() -> None:
    args = build_parser().parse_args()
    analysis_path = args.analysis.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(analysis_path, dtype=str)
    ratio_sources, length_source = add_per_cm_ratio_features(frame)
    ratio_columns = [
        column for column in frame.columns if column.startswith("ratio_aorta_size__") and column.endswith(PER_CM_SUFFIX)
    ]
    if not ratio_columns:
        raise ValueError(f"No per-cm ratio columns could be generated from {analysis_path}")

    feature_summary, review_rows = summarize_ratio_outliers(
        frame,
        ratio_columns,
        ratio_sources,
        length_source=length_source,
        top_n=args.top_n,
        robust_z_threshold=args.robust_z_threshold,
        iqr_multiplier=args.iqr_multiplier,
    )
    case_summary = summarize_cases(review_rows)
    etiology_summary = summarize_etiologies(frame, case_summary)
    distribution_summary = summarize_feature_etiology_distributions(frame, feature_summary)

    feature_path = outdir / "per_cm_ratio_feature_summary.csv"
    rows_path = outdir / "per_cm_ratio_outlier_rows.csv"
    case_path = outdir / "per_cm_ratio_case_summary.csv"
    etiology_path = outdir / "per_cm_ratio_etiology_outlier_summary.csv"
    distribution_path = outdir / "per_cm_ratio_distribution_by_etiology.csv"
    report_path = outdir / "per_cm_ratio_outliers.html"

    feature_summary.to_csv(feature_path, index=False)
    review_rows.to_csv(rows_path, index=False)
    case_summary.to_csv(case_path, index=False)
    etiology_summary.to_csv(etiology_path, index=False)
    distribution_summary.to_csv(distribution_path, index=False)
    report_path.write_text(
        build_report(
            frame,
            feature_summary,
            review_rows,
            case_summary,
            etiology_summary,
            distribution_summary,
            analysis_path=analysis_path,
            length_source=length_source,
            top_n=args.top_n,
            robust_z_threshold=args.robust_z_threshold,
            iqr_multiplier=args.iqr_multiplier,
        ),
        encoding="utf-8",
    )

    print(f"Feature summary: {feature_path}")
    print(f"Outlier rows: {rows_path}")
    print(f"Case summary: {case_path}")
    print(f"Etiology summary: {etiology_path}")
    print(f"Distribution summary: {distribution_path}")
    print(f"Report: {report_path}")


def add_per_cm_ratio_features(frame: pd.DataFrame) -> tuple[dict[str, str], str | None]:
    aortic_length = first_numeric_column(frame, DENOMINATOR_PRIORITY)
    length_source = first_existing_column(frame, DENOMINATOR_PRIORITY)
    if aortic_length is None:
        return {}, None

    ratio_sources: dict[str, str] = {}
    for column in list(frame.columns):
        domain = feature_domain(column)
        if not domain:
            continue
        lower = column.lower()
        if "volume" not in lower:
            continue
        if "aorta_volume" in lower or "aortic_volume" in lower:
            continue
        numerator = pd.to_numeric(frame[column], errors="coerce")
        safe_name = sanitize_feature_name(column)
        ratio_column = f"ratio_aorta_size__{safe_name}{PER_CM_SUFFIX}"
        frame[ratio_column] = safe_divide(numerator, aortic_length)
        ratio_sources[ratio_column] = column

    return ratio_sources, length_source


def summarize_ratio_outliers(
    frame: pd.DataFrame,
    ratio_columns: list[str],
    ratio_sources: dict[str, str],
    *,
    length_source: str | None,
    top_n: int,
    robust_z_threshold: float,
    iqr_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    review_rows = []
    aortic_length = pd.to_numeric(frame[length_source], errors="coerce") if length_source else pd.Series(np.nan, index=frame.index)

    for column in ratio_columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.dropna()
        numerator_column = ratio_sources.get(column, "")
        numerator = (
            pd.to_numeric(frame[numerator_column], errors="coerce")
            if numerator_column in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )

        stats = distribution_stats(valid)
        if valid.empty:
            continue
        robust_z = robust_z_scores(values, stats["median"], stats["mad"])
        high_iqr_flag = pd.Series(False, index=frame.index)
        low_iqr_flag = pd.Series(False, index=frame.index)
        if is_finite_positive(stats["iqr"]):
            high_iqr_flag = values.gt(stats["q75"] + iqr_multiplier * stats["iqr"])
            low_iqr_flag = values.lt(stats["q25"] - iqr_multiplier * stats["iqr"])
        robust_flag = robust_z.abs().ge(robust_z_threshold) if stats["mad"] > 0 else pd.Series(False, index=frame.index)
        extreme_flag = robust_flag | high_iqr_flag | low_iqr_flag

        top_indices = valid.sort_values(ascending=False).head(top_n).index
        review_mask = extreme_flag | values.index.isin(top_indices)
        reviewed = frame.loc[review_mask.fillna(False)].copy()

        top_case_id = ""
        top_label = ""
        if not valid.empty:
            top_idx = valid.idxmax()
            top_case_id = str(frame.at[top_idx, CASE_ID]) if CASE_ID in frame.columns else ""
            top_label = str(frame.at[top_idx, SOURCE_LABEL]) if SOURCE_LABEL in frame.columns else ""

        feature_rows.append(
            {
                "feature": column,
                "feature_label": ratio_feature_label(column),
                "domain": feature_domain(numerator_column or column),
                "domain_label": DOMAIN_LABELS.get(feature_domain(numerator_column or column), ""),
                "numerator_feature": numerator_column,
                "numerator_feature_label": base_feature_label(numerator_column),
                "n": int(valid.size),
                "missing_n": int(values.isna().sum()),
                "zero_n": int(values.eq(0).sum()),
                "median": stats["median"],
                "q25": stats["q25"],
                "q75": stats["q75"],
                "q95": stats["q95"],
                "q99": stats["q99"],
                "iqr": stats["iqr"],
                "mad": stats["mad"],
                "max": stats["max"],
                "max_abs_robust_z": safe_float(robust_z.abs().max()),
                "iqr_high_fence": safe_float(stats["q75"] + iqr_multiplier * stats["iqr"])
                if is_finite_positive(stats["iqr"])
                else "",
                "n_extreme_any": int(extreme_flag.fillna(False).sum()),
                "n_extreme_robust_z": int(robust_flag.fillna(False).sum()),
                "n_extreme_iqr_high": int(high_iqr_flag.fillna(False).sum()),
                "n_review_rows": int(review_mask.fillna(False).sum()),
                "top_case_id": top_case_id,
                "top_source_etiology_label": top_label,
            }
        )

        for idx, source in reviewed.iterrows():
            rz = robust_z.loc[idx]
            reason = flag_reason(
                is_top=idx in top_indices,
                robust=bool(robust_flag.loc[idx]),
                high_iqr=bool(high_iqr_flag.loc[idx]),
                low_iqr=bool(low_iqr_flag.loc[idx]),
                top_n=top_n,
                robust_z_threshold=robust_z_threshold,
                iqr_multiplier=iqr_multiplier,
            )
            review_rows.append(
                {
                    **context_values(source),
                    "feature": column,
                    "feature_label": ratio_feature_label(column),
                    "domain": feature_domain(numerator_column or column),
                    "domain_label": DOMAIN_LABELS.get(feature_domain(numerator_column or column), ""),
                    "numerator_feature": numerator_column,
                    "numerator_feature_label": base_feature_label(numerator_column),
                    "numerator_value": safe_float(numerator.loc[idx]),
                    "aortic_length_source": length_source or "",
                    "aortic_length_cm": safe_float(aortic_length.loc[idx]),
                    "ratio_value": safe_float(values.loc[idx]),
                    "robust_z": safe_float(rz),
                    "abs_robust_z": safe_float(abs(rz)) if pd.notna(rz) else "",
                    "feature_median": stats["median"],
                    "feature_q75": stats["q75"],
                    "feature_iqr": stats["iqr"],
                    "feature_iqr_high_fence": safe_float(stats["q75"] + iqr_multiplier * stats["iqr"])
                    if is_finite_positive(stats["iqr"])
                    else "",
                    "is_extreme": bool(extreme_flag.loc[idx]),
                    "is_top_n": bool(idx in top_indices),
                    "flag_reason": reason,
                }
            )

    feature_summary = pd.DataFrame(feature_rows).sort_values(
        ["n_extreme_any", "max_abs_robust_z", "max"],
        ascending=[False, False, False],
        na_position="last",
    )
    review_table = pd.DataFrame(review_rows)
    if not review_table.empty:
        review_table["_sort_abs_robust_z"] = pd.to_numeric(review_table["abs_robust_z"], errors="coerce").fillna(-1)
        review_table["_sort_ratio_value"] = pd.to_numeric(review_table["ratio_value"], errors="coerce").fillna(-1)
        review_table = review_table.sort_values(
            ["is_extreme", "_sort_abs_robust_z", "_sort_ratio_value"],
            ascending=[False, False, False],
        ).drop(columns=["_sort_abs_robust_z", "_sort_ratio_value"])
    return feature_summary, review_table


def summarize_cases(review_rows: pd.DataFrame) -> pd.DataFrame:
    if review_rows.empty:
        return pd.DataFrame()
    rows = []
    for case_id, group in review_rows.groupby(CASE_ID, dropna=False):
        extreme = group[group["is_extreme"].astype(bool)]
        abs_z = pd.to_numeric(group["abs_robust_z"], errors="coerce")
        ratio_value = pd.to_numeric(group["ratio_value"], errors="coerce")
        max_z_idx = abs_z.idxmax() if abs_z.notna().any() else ratio_value.idxmax()
        max_row = group.loc[max_z_idx]
        rows.append(
            {
                CASE_ID: case_id,
                "record_id": first_value(group, "record_id"),
                SOURCE_LABEL: first_value(group, SOURCE_LABEL),
                "stroke_mechanism": first_value(group, "stroke_mechanism"),
                "source_cohort": first_value(group, "source_cohort"),
                "age": first_value(group, "age"),
                "sex": first_value(group, "sex"),
                "n_extreme_ratio_features": int(len(extreme)),
                "n_reviewed_ratio_features": int(len(group)),
                "n_reviewed_domains": int(group["domain"].nunique(dropna=True)),
                "max_abs_robust_z": safe_float(abs_z.max()),
                "max_ratio_value": safe_float(ratio_value.max()),
                "max_outlier_feature": str(max_row.get("feature", "")),
                "max_outlier_feature_label": str(max_row.get("feature_label", "")),
                "max_outlier_flag_reason": str(max_row.get("flag_reason", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_extreme_ratio_features", "max_abs_robust_z", "n_reviewed_ratio_features"],
        ascending=[False, False, False],
        na_position="last",
    )


def summarize_etiologies(frame: pd.DataFrame, case_summary: pd.DataFrame) -> pd.DataFrame:
    if SOURCE_LABEL not in frame.columns:
        return pd.DataFrame()
    base = frame[[CASE_ID, SOURCE_LABEL]].drop_duplicates()
    if case_summary.empty:
        base["n_extreme_ratio_features"] = 0
        base["n_reviewed_ratio_features"] = 0
    else:
        base = base.merge(
            case_summary[[CASE_ID, "n_extreme_ratio_features", "n_reviewed_ratio_features"]],
            on=CASE_ID,
            how="left",
        )
        base[["n_extreme_ratio_features", "n_reviewed_ratio_features"]] = base[
            ["n_extreme_ratio_features", "n_reviewed_ratio_features"]
        ].fillna(0)
    rows = []
    for label, group in base.groupby(SOURCE_LABEL, dropna=False):
        rows.append(
            {
                SOURCE_LABEL: label,
                "n_cases": int(len(group)),
                "n_cases_with_extreme_ratio": int(group["n_extreme_ratio_features"].gt(0).sum()),
                "pct_cases_with_extreme_ratio": safe_float(100.0 * group["n_extreme_ratio_features"].gt(0).mean()),
                "median_extreme_ratio_features": safe_float(group["n_extreme_ratio_features"].median()),
                "max_extreme_ratio_features": safe_float(group["n_extreme_ratio_features"].max()),
                "n_cases_reviewed_as_top_n": int(group["n_reviewed_ratio_features"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_cases_with_extreme_ratio", ascending=False)


def summarize_feature_etiology_distributions(frame: pd.DataFrame, feature_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return pd.DataFrame()
    rows = []
    labels = [label for label in ETIOLOGY_ORDER if SOURCE_LABEL in frame.columns and frame[SOURCE_LABEL].eq(label).any()]
    labels.extend(
        sorted(
            {
                str(value)
                for value in frame.get(SOURCE_LABEL, pd.Series(dtype=object)).dropna().unique()
                if str(value) not in ETIOLOGY_ORDER
            }
        )
    )
    for _, feature_row in feature_summary.iterrows():
        feature = str(feature_row["feature"])
        if feature not in frame.columns:
            continue
        values = numeric_ratio_values(frame[feature])
        groups: list[tuple[str, pd.Series]] = [("Overall", pd.Series(True, index=frame.index))]
        groups.extend((label, frame[SOURCE_LABEL].eq(label)) for label in labels)
        for label, mask in groups:
            group_values = values.loc[mask].dropna()
            stats = distribution_stats(group_values)
            rows.append(
                {
                    "feature": feature,
                    "feature_label": feature_row.get("feature_label", ratio_feature_label(feature)),
                    "domain": feature_row.get("domain", feature_domain(feature)),
                    "domain_label": feature_row.get("domain_label", DOMAIN_LABELS.get(feature_domain(feature), "")),
                    SOURCE_LABEL: ETIOLOGY_LABELS.get(label, label),
                    "n": int(group_values.size),
                    "missing_n": int(values.loc[mask].isna().sum()),
                    "median": stats["median"],
                    "q25": stats["q25"],
                    "q75": stats["q75"],
                    "q95": stats["q95"],
                    "q99": stats["q99"],
                    "max": stats["max"],
                }
            )
    return pd.DataFrame(rows)


def distribution_plot_grid(frame: pd.DataFrame, feature_summary: pd.DataFrame) -> str:
    if feature_summary.empty:
        return "<p>No valid per-cm ratio variables were available for plotting.</p>"
    cards = []
    for _, row in feature_summary.iterrows():
        feature = str(row["feature"])
        if feature not in frame.columns:
            continue
        values = numeric_ratio_values(frame[feature]).dropna()
        values = values[values >= 0]
        if values.empty:
            continue
        label = str(row.get("feature_label", ratio_feature_label(feature)))
        domain_label = str(row.get("domain_label", DOMAIN_LABELS.get(feature_domain(feature), "")))
        stats = distribution_stats(values)
        numerator = str(row.get("numerator_feature", ""))
        meta = (
            f"{html.escape(domain_label)} | n={int(stats_n(values))} | "
            f"median {format_axis_number(stats['median'])} | "
            f"q95 {format_axis_number(stats['q95'])} | "
            f"max {format_axis_number(stats['max'])} | "
            f"extreme rows {format_integer(row.get('n_extreme_any', 0))}"
        )
        cards.append(
            f"""
    <section class="plot-card">
      <h3>{html.escape(label)}</h3>
      <p class="plot-meta">{meta}</p>
      <div class="plot-pair">
        {histogram_svg(values)}
        {boxplot_svg_by_etiology(frame, feature)}
      </div>
      <p class="plot-source">Source numerator: <code>{html.escape(numerator)}</code></p>
    </section>"""
        )
    if not cards:
        return "<p>No valid per-cm ratio variables were available for plotting.</p>"
    return '<div class="plot-grid">' + "\n".join(cards) + "\n  </div>"


def histogram_svg(values: pd.Series) -> str:
    raw = values.dropna()
    raw = raw[raw >= 0]
    if raw.empty:
        return empty_plot_svg("Histogram", "No nonnegative values")

    width = 500
    height = 210
    left = 44
    right = 18
    top = 24
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_raw = safe_axis_max(raw)
    max_log = math.log1p(max_raw)
    bins = min(24, max(8, int(math.sqrt(len(raw)))))
    edges = np.linspace(0.0, max(max_log, 1.0), bins + 1)
    counts, edges = np.histogram(np.log1p(raw.to_numpy(dtype=float)), bins=edges)
    max_count = max(int(counts.max()), 1)
    axis_y = top + plot_height

    parts = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="Histogram">',
        svg_text(left, 14, "Histogram", size=12, weight="600"),
        svg_line(left, top, left, axis_y, "#8a8a8a"),
        svg_line(left, axis_y, left + plot_width, axis_y, "#8a8a8a"),
    ]
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        bar_x0 = left + (x0 / max(max_log, 1.0)) * plot_width
        bar_x1 = left + (x1 / max(max_log, 1.0)) * plot_width
        bar_height = (count / max_count) * plot_height
        y = axis_y - bar_height
        parts.append(
            f'<rect x="{bar_x0:.2f}" y="{y:.2f}" width="{max(bar_x1 - bar_x0 - 1, 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#4c78a8" opacity="0.78" />'
        )

    median = float(raw.median())
    q95 = float(raw.quantile(0.95))
    parts.append(marker_line(median, max_raw, left, top, plot_width, plot_height, "#111", "median"))
    parts.append(marker_line(q95, max_raw, left, top, plot_width, plot_height, "#c2410c", "q95"))
    parts.extend(log_axis_ticks_svg(max_raw, left, axis_y, plot_width))
    parts.append(svg_text(left + plot_width / 2, height - 5, "ratio value", size=10, anchor="middle", color="#555"))
    parts.append(svg_text(6, top + 4, f"max count {max_count}", size=10, color="#555"))
    parts.append("</svg>")
    return "\n".join(parts)


def boxplot_svg_by_etiology(frame: pd.DataFrame, feature: str) -> str:
    if SOURCE_LABEL not in frame.columns:
        return empty_plot_svg("Box plot by etiology", "No etiology labels")
    values = numeric_ratio_values(frame[feature])
    raw_all = values.dropna()
    raw_all = raw_all[raw_all >= 0]
    if raw_all.empty:
        return empty_plot_svg("Box plot by etiology", "No nonnegative values")

    groups = [label for label in ETIOLOGY_ORDER if frame[SOURCE_LABEL].eq(label).any()]
    groups.extend(sorted({str(value) for value in frame[SOURCE_LABEL].dropna().unique() if str(value) not in ETIOLOGY_ORDER}))
    width = 500
    left = 92
    right = 18
    top = 28
    bottom = 36
    row_gap = 34
    height = top + bottom + row_gap * max(len(groups), 1)
    plot_width = width - left - right
    axis_y = height - bottom + 4
    max_raw = safe_axis_max(raw_all)

    parts = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="Box plot by etiology">',
        svg_text(left, 16, "Box plot by etiology", size=12, weight="600"),
        svg_line(left, axis_y, left + plot_width, axis_y, "#8a8a8a"),
    ]
    for idx, label in enumerate(groups):
        group_values = values.loc[frame[SOURCE_LABEL].eq(label)].dropna()
        group_values = group_values[group_values >= 0]
        y = top + idx * row_gap + 13
        label_text = ETIOLOGY_LABELS.get(label, label)
        color = ETIOLOGY_COLORS.get(label, "#555")
        parts.append(svg_text(4, y + 4, f"{label_text} n={len(group_values)}", size=10, color="#333"))
        if group_values.empty:
            parts.append(svg_text(left, y + 4, "no data", size=10, color="#777"))
            continue
        q05 = float(group_values.quantile(0.05))
        q25 = float(group_values.quantile(0.25))
        median = float(group_values.median())
        q75 = float(group_values.quantile(0.75))
        q95 = float(group_values.quantile(0.95))
        x05 = scale_log_raw(q05, max_raw, left, plot_width)
        x25 = scale_log_raw(q25, max_raw, left, plot_width)
        x50 = scale_log_raw(median, max_raw, left, plot_width)
        x75 = scale_log_raw(q75, max_raw, left, plot_width)
        x95 = scale_log_raw(q95, max_raw, left, plot_width)
        parts.append(svg_line(x05, y, x95, y, color, width=1.5))
        parts.append(svg_line(x05, y - 5, x05, y + 5, color, width=1.5))
        parts.append(svg_line(x95, y - 5, x95, y + 5, color, width=1.5))
        box_width = max(x75 - x25, 2)
        parts.append(
            f'<rect x="{x25:.2f}" y="{y - 8:.2f}" width="{box_width:.2f}" height="16" '
            f'fill="{color}" opacity="0.25" stroke="{color}" stroke-width="1.5" />'
        )
        parts.append(svg_line(x50, y - 9, x50, y + 9, "#111", width=1.7))
        outliers = group_values[group_values.gt(q95)].sort_values()
        if len(outliers) > 16:
            sample_idx = np.unique(np.linspace(0, len(outliers) - 1, 16).astype(int))
            outliers = outliers.iloc[sample_idx]
        for value in outliers:
            x = scale_log_raw(float(value), max_raw, left, plot_width)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="{color}" opacity="0.75" />')
    parts.extend(log_axis_ticks_svg(max_raw, left, axis_y, plot_width))
    parts.append(svg_text(left + plot_width / 2, height - 5, "ratio value", size=10, anchor="middle", color="#555"))
    parts.append(svg_text(left + plot_width - 4, 16, "log axis", size=10, anchor="end", color="#777"))
    parts.append("</svg>")
    return "\n".join(parts)


def empty_plot_svg(title: str, message: str) -> str:
    width = 500
    height = 120
    return "\n".join(
        [
            f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            svg_text(18, 24, title, size=12, weight="600"),
            svg_text(18, 58, message, size=11, color="#777"),
            "</svg>",
        ]
    )


def distribution_stats(valid: pd.Series) -> dict[str, float]:
    if valid.empty:
        return {key: math.nan for key in ["median", "q25", "q75", "q95", "q99", "iqr", "mad", "max"]}
    median = safe_float(valid.median())
    q25 = safe_float(valid.quantile(0.25))
    q75 = safe_float(valid.quantile(0.75))
    mad = safe_float((valid - median).abs().median())
    return {
        "median": median,
        "q25": q25,
        "q75": q75,
        "q95": safe_float(valid.quantile(0.95)),
        "q99": safe_float(valid.quantile(0.99)),
        "iqr": safe_float(q75 - q25),
        "mad": mad,
        "max": safe_float(valid.max()),
    }


def robust_z_scores(values: pd.Series, median: float, mad: float) -> pd.Series:
    if not is_finite_positive(mad):
        return pd.Series(np.nan, index=values.index)
    return 0.6745 * (values - median) / mad


def flag_reason(
    *,
    is_top: bool,
    robust: bool,
    high_iqr: bool,
    low_iqr: bool,
    top_n: int,
    robust_z_threshold: float,
    iqr_multiplier: float,
) -> str:
    reasons = []
    if robust:
        reasons.append(f"abs robust z >= {robust_z_threshold:g}")
    if high_iqr:
        reasons.append(f"> q75 + {iqr_multiplier:g} IQR")
    if low_iqr:
        reasons.append(f"< q25 - {iqr_multiplier:g} IQR")
    if is_top:
        reasons.append(f"top {top_n}")
    return "; ".join(reasons)


def context_values(row: pd.Series) -> dict[str, object]:
    columns = [
        CASE_ID,
        "record_id",
        SOURCE_LABEL,
        "stroke_mechanism",
        "source_cohort",
        "age",
        "sex",
    ]
    return {column: row.get(column, "") for column in columns}


def add_per_cm_sort_key(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(format_number)
    return display


def build_report(
    frame: pd.DataFrame,
    feature_summary: pd.DataFrame,
    review_rows: pd.DataFrame,
    case_summary: pd.DataFrame,
    etiology_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    *,
    analysis_path: Path,
    length_source: str | None,
    top_n: int,
    robust_z_threshold: float,
    iqr_multiplier: float,
) -> str:
    n_features = int(len(feature_summary))
    n_review_rows = int(len(review_rows))
    n_extreme_rows = int(review_rows["is_extreme"].astype(bool).sum()) if not review_rows.empty else 0
    n_extreme_cases = int(case_summary["n_extreme_ratio_features"].gt(0).sum()) if not case_summary.empty else 0
    distribution_columns = [
        "feature_label",
        SOURCE_LABEL,
        "n",
        "median",
        "q25",
        "q75",
        "q95",
        "max",
    ]
    top_feature_columns = [
        "feature_label",
        "domain_label",
        "numerator_feature",
        "n",
        "median",
        "q75",
        "q95",
        "q99",
        "max",
        "max_abs_robust_z",
        "n_extreme_any",
        "top_case_id",
        "top_source_etiology_label",
    ]
    top_row_columns = [
        CASE_ID,
        "record_id",
        SOURCE_LABEL,
        "feature_label",
        "domain_label",
        "numerator_feature",
        "numerator_value",
        "aortic_length_cm",
        "ratio_value",
        "robust_z",
        "flag_reason",
    ]
    case_columns = [
        CASE_ID,
        "record_id",
        SOURCE_LABEL,
        "n_extreme_ratio_features",
        "n_reviewed_ratio_features",
        "max_abs_robust_z",
        "max_outlier_feature_label",
        "max_outlier_flag_reason",
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SLAO Per-cm Ratio Outliers</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #222; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    h3 {{ font-size: 15px; margin: 0 0 6px; }}
    p {{ max-width: 1040px; line-height: 1.45; }}
    table {{ border-collapse: collapse; font-size: 12px; margin: 12px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, td:nth-child(2), td:nth-child(3), td:nth-child(4) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f5f5f5; padding: 1px 4px; }}
    .note {{ color: #555; }}
    .metric {{ display: inline-block; margin: 10px 18px 8px 0; }}
    .metric strong {{ display: block; font-size: 20px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 14px; margin-top: 12px; }}
    .plot-card {{ border: 1px solid #d8d8d8; padding: 12px; break-inside: avoid; background: #fff; }}
    .plot-meta {{ font-size: 12px; color: #555; margin: 0 0 8px; }}
    .plot-source {{ font-size: 11px; color: #666; margin: 6px 0 0; overflow-wrap: anywhere; }}
    .plot-pair {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    svg.plot {{ max-width: 100%; height: auto; display: block; background: #fff; }}
  </style>
</head>
<body>
  <h1>SLAO Per-cm Ratio Outliers</h1>
  <p class="note">Analysis source: <code>{html.escape(str(analysis_path))}</code><br>
  Aortic-length denominator: <code>{html.escape(length_source or "not found")}</code></p>
  <p>Per-cm ratios were recomputed from the current etiology modeling table using the same aortic-length priority and volume-feature rules as the radiomics score builder. Extreme flags use absolute robust z >= {robust_z_threshold:g} when MAD is nonzero, or values outside q25/q75 +/- {iqr_multiplier:g} IQR when IQR is nonzero. The top {top_n} values per feature are also included for visual review, even when not formally extreme.</p>
  <div class="metric"><strong>{n_features}</strong> per-cm ratio features</div>
  <div class="metric"><strong>{n_review_rows}</strong> case-feature rows surfaced</div>
  <div class="metric"><strong>{n_extreme_rows}</strong> case-feature rows formally extreme</div>
  <div class="metric"><strong>{n_extreme_cases}</strong> cases with at least one formal extreme</div>

  <h2>Distribution Graphs</h2>
  <p class="note">Histograms and box plots use a log-scaled ratio axis so the long right tails remain visible. Box plots are stratified by source etiology; boxes show the IQR, center lines show medians, whiskers show the 5th to 95th percentile range, and dots show values above the 95th percentile.</p>
  {distribution_plot_grid(frame, feature_summary)}

  <h2>Etiology Summary</h2>
  {format_table(etiology_summary)}

  <h2>Distribution Summary by Etiology</h2>
  {format_table(distribution_summary[distribution_columns] if not distribution_summary.empty else distribution_summary)}

  <h2>Worst Case-Feature Rows</h2>
  {format_table(review_rows.head(80)[top_row_columns] if not review_rows.empty else review_rows)}

  <h2>Worst Features</h2>
  {format_table(feature_summary.head(60)[top_feature_columns] if not feature_summary.empty else feature_summary)}

  <h2>Worst Cases</h2>
  {format_table(case_summary.head(60)[case_columns] if not case_summary.empty else case_summary)}
</body>
</html>
"""


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    display = frame.copy()
    for column in display.columns:
        if column == "record_id" or column.endswith("_id"):
            continue
        numeric = pd.to_numeric(display[column], errors="coerce")
        if numeric.notna().any() and numeric.notna().sum() >= max(1, int(0.5 * len(display))):
            if is_count_column(column) and numeric.dropna().map(lambda value: float(value).is_integer()).all():
                display[column] = numeric.map(format_integer)
            else:
                display[column] = numeric.map(format_number)
    return display.to_html(index=False, escape=True)


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def first_numeric_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return None


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator > 0)
    return numerator / denominator


def sanitize_feature_name(column: str) -> str:
    safe = []
    for char in column:
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    text = "".join(safe)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def feature_domain(name: str) -> str:
    lower = name.lower()
    if "wall_thickness" in lower or "wall_gt4mm" in lower:
        return "wall_thickness"
    if "wall_from_fat" in lower or "experimental_wall_from_fat" in lower:
        return "wall_from_fat"
    if "calcium" in lower or "calcification" in lower or "agatston" in lower:
        return "calcium"
    if "fat" in lower or "periaortic" in lower:
        return "fat"
    return ""


def ratio_feature_label(feature: str) -> str:
    base = feature.removeprefix("ratio_aorta_size__")
    if base.endswith(PER_CM_SUFFIX):
        base = base[: -len(PER_CM_SUFFIX)]
    return base_feature_label(base) + " / aortic cm"


def base_feature_label(feature: str) -> str:
    lower = feature.lower()
    threshold = threshold_suffix(feature)
    if "hu_gt_1000_volume" in lower:
        return "Very dense calcium volume" + threshold
    if "calcium_volume" in lower:
        return "Calcium volume" + threshold
    if "agatston" in lower:
        return "Agatston-like calcium" + threshold
    if "surfacevolumeratio" in lower:
        return "Periaortic fat surface-to-volume ratio"
    if "voxelvolume" in lower:
        return "Periaortic fat voxel volume"
    if "meshvolume" in lower:
        return "Periaortic fat mesh volume"
    if "hu_refined_aorta_added_volume" in lower:
        return "HU-refined added aorta volume"
    if "whole_aorta_fat_omics_periaortic_fat_volume" in lower:
        return "Whole-aorta periaortic fat volume"
    if "periaortic_fat_volume_0_2mm" in lower:
        return "Periaortic fat volume 0-2 mm"
    if "periaortic_fat_volume_2_5mm" in lower:
        return "Periaortic fat volume 2-5 mm"
    if "volume_per_cm" in lower:
        return "Periaortic fat volume per original cm"
    if "periaortic_fat_volume" in lower:
        return "Periaortic fat volume"
    if "fat_support_0_5mm_volume" in lower:
        return "Fat-supported wall volume"
    if "wall_candidate_volume" in lower:
        return "Candidate wall volume"
    if "closed_outer_envelope_volume" in lower:
        return "Outer envelope volume"
    if "contrast_lumen_volume" in lower:
        return "Contrast lumen volume"
    if "gt4mm_volume" in lower or "wall_thickness_gt4mm_volume" in lower:
        return "Wall >4 mm volume"
    if "wall_volume" in lower:
        return "Wall volume"
    return fallback_feature_label(feature)


def threshold_suffix(feature: str) -> str:
    lower = feature.lower()
    if "dynamic_lumen_referenced_seed500hu" in lower:
        return " dynamic threshold"
    marker = "__thr_"
    if marker in feature:
        value = feature.split(marker, 1)[1].replace("_", " ")
        return f" {threshold_label(value)}"
    marker = "_thr_"
    if marker in feature:
        value = feature.split(marker, 1)[1].replace("_", " ")
        return f" {threshold_label(value)}"
    return ""


def threshold_label(value: str) -> str:
    text = value.strip()
    return f"{text} HU" if text.isdigit() else text


def fallback_feature_label(feature: str) -> str:
    text = feature
    prefixes = [
        "aorta__calcium_omics__",
        "aorta_wall_band__calcification__",
        "aorta_wall_dynamic__calcification_dynamic_threshold__",
        "aorta_wall_dynamic__calcification__",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__",
        "aortic_wall__wall_thickness_threshold__",
        "aortic_wall__wall_thickness__",
        "aorta_segment_whole_aorta_fat_omics_",
        "aorta_segment:whole_aorta__fat_omics__",
        "periaortic_fat__fat_omics__",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("__thr_dynamic_lumen_referenced_seed500hu", "")
    text = text.replace("__thr_dynamic_lumen_referenced_seed500HU", "")
    text = text.replace("__thr_", " ")
    text = text.replace("_", " ")
    return " ".join(word.capitalize() if word.islower() else word for word in text.split())


def numeric_ratio_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def stats_n(values: pd.Series) -> int:
    return int(values.dropna().size)


def safe_axis_max(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean >= 0]
    if clean.empty:
        return 1.0
    maximum = float(clean.max())
    return maximum if np.isfinite(maximum) and maximum > 0 else 1.0


def scale_log_raw(value: float, max_raw: float, left: float, plot_width: float) -> float:
    try:
        raw = max(float(value), 0.0)
    except (TypeError, ValueError):
        raw = 0.0
    max_log = max(math.log1p(max(max_raw, 0.0)), 1e-9)
    return left + (math.log1p(min(raw, max_raw)) / max_log) * plot_width


def marker_line(
    value: float,
    max_raw: float,
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    color: str,
    label: str,
) -> str:
    x = scale_log_raw(value, max_raw, left, plot_width)
    return "\n".join(
        [
            svg_line(x, top, x, top + plot_height, color, width=1.2, dash="3 3"),
            svg_text(x + 4, top + 11, f"{label} {format_axis_number(value)}", size=9, color=color),
        ]
    )


def log_axis_ticks_svg(max_raw: float, left: float, axis_y: float, plot_width: float) -> list[str]:
    parts = []
    for tick in log_axis_ticks(max_raw):
        x = scale_log_raw(tick, max_raw, left, plot_width)
        parts.append(svg_line(x, axis_y, x, axis_y + 4, "#8a8a8a"))
        parts.append(svg_text(x, axis_y + 17, format_axis_number(tick), size=9, anchor="middle", color="#555"))
    return parts


def log_axis_ticks(max_raw: float) -> list[float]:
    if not np.isfinite(max_raw) or max_raw <= 0:
        return [0.0, 1.0]
    if max_raw <= 1:
        ticks = list(np.linspace(0, max_raw, 5))
    elif max_raw <= 10:
        ticks = [0, 1, 2, 5, 10]
    elif max_raw <= 100:
        ticks = [0, 1, 5, 10, 25, 50, 100]
    else:
        ticks = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
    ticks = [float(tick) for tick in ticks if tick <= max_raw * 1.001]
    if len(ticks) < 3:
        ticks = list(np.linspace(0, max_raw, 4))
    if max_raw > 0 and abs(ticks[-1] - max_raw) / max_raw > 0.25:
        ticks.append(float(max_raw))
    cleaned: list[float] = []
    for tick in ticks:
        if not cleaned or abs(tick - cleaned[-1]) > max(max_raw * 0.002, 1e-9):
            cleaned.append(float(tick))
    return cleaned[-7:]


def svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 10,
    anchor: str = "start",
    color: str = "#222",
    weight: str | None = None,
) -> str:
    weight_attr = f' font-weight="{html.escape(weight)}"' if weight else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{html.escape(anchor)}" '
        f'fill="{html.escape(color)}"{weight_attr}>{html.escape(str(text))}</text>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    *,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{html.escape(color)}" stroke-width="{width:.2f}"{dash_attr} />'
    )


def format_axis_number(value: object) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if absolute >= 10_000:
        return f"{number / 1000:.0f}k"
    if absolute >= 1000:
        return f"{number / 1000:.1f}k"
    if absolute >= 100:
        return f"{number:.0f}"
    if absolute >= 10:
        return f"{number:.1f}"
    if absolute >= 1:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{number:.2g}"


def first_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.empty:
        return ""
    value = frame[column].iloc[0]
    return "" if pd.isna(value) else value


def is_finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number) and number > 0


def safe_float(value: object) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if np.isfinite(number) else ""


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


def format_integer(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{int(float(value))}"
    except (TypeError, ValueError):
        return str(value)


def is_count_column(column: str) -> bool:
    return (
        column in {"n", "missing_n", "zero_n"}
        or column.startswith("n_")
        or column.endswith("_n")
        or column.startswith("max_extreme")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--robust-z-threshold", type=float, default=5.0)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    return parser


if __name__ == "__main__":
    main()
