#!/usr/bin/env python
"""Build a Tufte-like HTML report for SLAO MACE vs aorta CTA features."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This report uses seaborn/matplotlib. In this repo, run it with "
        "`./.venv_dt/bin/python aorta_cta_radiomics/scripts/build_slao_mace_aorta_report.py`."
    ) from exc


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "mace_slao"
DEFAULT_ANALYSIS = DEFAULT_OUTDIR / "slao_mace_aorta_modeling.csv"
DEFAULT_UNIVARIATE = DEFAULT_OUTDIR / "univariate_mace_primary.csv"
DEFAULT_SUMMARY = DEFAULT_OUTDIR / "slao_mace_aorta_summary.json"
DEFAULT_REPORT = DEFAULT_OUTDIR / "slao_mace_aorta_report.html"
DEFAULT_ASSETS = DEFAULT_OUTDIR / "report_assets"

DOMAIN_LABELS = {
    "calcium": "Calcium",
    "fat": "Periaortic fat",
    "wall_from_fat": "Wall-from-fat",
    "wall_thickness": "Wall thickness",
}

DOMAIN_COLORS = {
    "calcium": "#8a5a2b",
    "fat": "#4c7f73",
    "wall_from_fat": "#6b5876",
    "wall_thickness": "#335f86",
}

RISK_FACTORS = [
    ("hypertension", "Hypertension"),
    ("diabetes", "Diabetes"),
    ("dyslipidemia", "Dyslipidemia"),
    ("coronary_artery_disease", "Coronary artery disease"),
    ("heart_failure", "Heart failure history"),
    ("atrial_fibrillation", "Atrial fibrillation"),
    ("prior_stroke_tia", "Prior stroke/TIA"),
    ("cta_extended", "Extended CTA"),
    ("stroke_confirmed", "Stroke confirmed"),
    ("lvo", "LVO"),
    ("iv_thrombolysis", "IV thrombolysis"),
    ("evt", "EVT"),
]


def main() -> None:
    args = build_parser().parse_args()
    analysis_path = args.analysis.expanduser().resolve()
    univariate_path = args.univariate.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis_rows, analysis_header = read_csv(analysis_path)
    univariate_rows, _ = read_csv(univariate_path)

    html_text = build_report(
        summary=summary,
        analysis_rows=analysis_rows,
        analysis_header=analysis_header,
        univariate_rows=univariate_rows,
        analysis_path=analysis_path,
        univariate_path=univariate_path,
        summary_path=summary_path,
        assets_dir=assets_dir,
        top_n=args.top_n,
        violin_top_n=args.violin_top_n,
    )
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--univariate", type=Path, default=DEFAULT_UNIVARIATE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--top-n", type=int, default=28)
    parser.add_argument("--violin-top-n", type=int, default=12)
    return parser


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def build_report(
    *,
    summary: dict[str, object],
    analysis_rows: list[dict[str, str]],
    analysis_header: list[str],
    univariate_rows: list[dict[str, str]],
    analysis_path: Path,
    univariate_path: Path,
    summary_path: Path,
    assets_dir: Path,
    top_n: int,
    violin_top_n: int,
) -> str:
    event_rows = [row for row in analysis_rows if row.get("mace_primary") == "1"]
    non_event_rows = [row for row in analysis_rows if row.get("mace_primary") == "0"]
    top_features = univariate_rows[:top_n]
    feature_columns = [
        column
        for column in analysis_header
        if feature_domain(column)
    ]
    domain_counts = Counter(feature_domain(column) for column in feature_columns)
    missingness_rows = feature_missingness(analysis_rows, feature_columns)
    covariate_rows = covariate_summary(event_rows, non_event_rows)
    study_arm_counts = two_group_counts(analysis_rows, "study_arm")
    sex_counts = two_group_counts(analysis_rows, "sex")
    component_counts = component_rows(summary)
    seaborn_figures = build_seaborn_figures(
        analysis_rows=analysis_rows,
        univariate_rows=univariate_rows,
        violin_top_n=violin_top_n,
        assets_dir=assets_dir,
    )

    cards = [
        ("Matched SLAO cases", fmt_int(summary.get("merged_rows", 0))),
        ("MACE events", fmt_int(summary.get("mace_primary_events", 0))),
        ("Non-events", fmt_int(summary.get("mace_primary_nonevents", 0))),
        ("Feature columns", fmt_int(summary.get("selected_feature_columns", 0))),
    ]
    event_rate = event_fraction(summary)

    body = f"""
    <header class="page-header">
      <p class="eyebrow">SLAO BIDS aorta CTA radiomics</p>
      <h1>MACE and aorta phenotype report</h1>
      <p class="subtitle">Outcome source: <code>{esc(summary.get("redcap_csv", ""))}</code><br>
      Cohort filter: <strong>{esc(summary.get("source_cohort", ""))}</strong>. Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}.</p>
    </header>

    <section class="stat-strip">
      {''.join(stat_card(label, value) for label, value in cards)}
    </section>

    <section class="narrative">
      <p><span class="newthought">The corrected Desktop dataset changes the endpoint burden.</span>
      In the current joined table, {fmt_int(summary.get("mace_primary_events", 0))} of
      {fmt_int(summary.get("merged_rows", 0))} matched SLAO cases have <code>mace_composite=1</code>
      ({format_percent(event_rate)}). The aorta feature table contributes {fmt_int(summary.get("selected_feature_columns", 0))}
      calcium, periaortic fat, wall-from-fat, and wall-thickness variables.</p>
      <p class="sidenote">This is an exploratory, unadjusted screen. The feature ranking requires at least
      {fmt_int(summary.get("univariate_min_n", 0))} non-missing feature/outcome pairs; associations are not adjusted
      for age, stroke status, or vascular risk factors.</p>
    </section>

    <section class="grid two">
      <figure>
        <h2>MACE burden</h2>
        {event_bar(summary)}
      </figure>
      <figure>
        <h2>Outcome components</h2>
        {horizontal_bar_svg(component_counts, width=560, bar_color="#575757")}
      </figure>
    </section>

    <section>
      <h2>Seaborn violin plots</h2>
      <p class="caption">Top univariate aorta features, standardized within feature. Violin widths show
      distribution shape, embedded points show individual cases, and inner lines mark quartiles.</p>
      {embedded_image(seaborn_figures.get("top_feature_violins", ""), "Seaborn violin plots for top aorta features by MACE")}
    </section>

    <section>
      <h2>Top univariate feature separation</h2>
      <p class="caption">Directional AUROC ranks each feature by unadjusted separation of MACE events vs non-events.
      The vertical rule is chance performance at 0.50.</p>
      {auc_lollipop_svg(top_features)}
    </section>

    <section>
      <h2>Effect direction</h2>
      <p class="caption">Standardized mean difference is event mean minus non-event mean. Positive values mean higher
      values in MACE cases.</p>
      {smd_dot_svg(top_features)}
    </section>

    <section class="grid two">
      <figure>
        <h2>Feature domains</h2>
        {horizontal_bar_svg(
            [(DOMAIN_LABELS.get(domain, domain), count, domain) for domain, count in sorted(domain_counts.items())],
            width=560,
            bar_color=None,
        )}
      </figure>
      <figure>
        <h2>Study arm</h2>
        {grouped_bar_svg(study_arm_counts)}
      </figure>
    </section>

    <section class="grid two">
      <figure>
        <h2>Sex distribution</h2>
        {grouped_bar_svg(sex_counts)}
      </figure>
      <figure>
        <h2>Feature completeness</h2>
        {missingness_svg(missingness_rows[:16])}
      </figure>
    </section>

    <section>
      <h2>Clinical balance</h2>
      {table_html(
          ["Variable", "MACE", "No MACE", "Difference"],
          covariate_rows,
          row_class=lambda row: "emph" if row[0] == "Age, years" else "",
      )}
    </section>

    <section>
      <h2>Ranked feature table</h2>
      {top_feature_table(top_features)}
    </section>

    <section class="foot">
      <h2>Files</h2>
      <p>Analysis CSV: <code>{esc(str(analysis_path))}</code><br>
      Univariate CSV: <code>{esc(str(univariate_path))}</code><br>
      Summary JSON: <code>{esc(str(summary_path))}</code></p>
      <p>Join notes: {fmt_int(summary.get("outcome_records", 0))} SLAO outcome records,
      {fmt_int(summary.get("feature_rows", 0))} aorta feature rows, {fmt_int(summary.get("merged_rows", 0))}
      matched rows. Unmatched aorta cases: <code>{esc(', '.join(summary.get("unmatched_feature_cases", [])))}</code>.</p>
    </section>
    """
    return page_template(body)


def page_template(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLAO MACE Aorta Report</title>
<style>
{css()}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def css() -> str:
    return """
:root {
  --paper: #fffff8;
  --ink: #151515;
  --muted: #666;
  --rule: #d8d0bc;
  --faint: #efeadc;
  --accent: #335f86;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.48;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 42px 34px 70px;
}
.page-header {
  border-bottom: 1px solid var(--rule);
  padding-bottom: 18px;
  margin-bottom: 26px;
}
.eyebrow {
  margin: 0 0 10px;
  color: var(--muted);
  font-size: 13px;
  letter-spacing: .08em;
  text-transform: uppercase;
}
h1 {
  font-size: clamp(36px, 6vw, 68px);
  line-height: .98;
  font-weight: 400;
  margin: 0 0 14px;
  max-width: 900px;
}
h2 {
  font-size: 24px;
  line-height: 1.1;
  font-weight: 400;
  margin: 0 0 12px;
}
.subtitle, .caption, .sidenote, .foot {
  color: var(--muted);
}
.subtitle { font-size: 16px; margin: 0; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .9em;
  background: #f5f1e4;
  padding: 1px 4px;
}
.stat-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  margin: 22px 0 28px;
}
.stat {
  padding: 18px 18px 15px 0;
  border-right: 1px solid var(--rule);
}
.stat:last-child { border-right: 0; }
.stat .value {
  font-size: 42px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.stat .label {
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-top: 7px;
}
.narrative {
  max-width: 780px;
  font-size: 19px;
  margin: 0 0 32px;
}
.newthought {
  font-variant: small-caps;
  letter-spacing: .03em;
}
.sidenote {
  border-left: 1px solid var(--rule);
  padding-left: 14px;
  font-size: 15px;
}
section { margin: 34px 0 44px; }
.grid {
  display: grid;
  gap: 34px;
  align-items: start;
}
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
figure { margin: 0; }
svg {
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
}
.figure-img {
  width: 100%;
  display: block;
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  padding: 10px 0;
}
.axis { stroke: var(--rule); stroke-width: 1; }
.guide { stroke: #c7bea9; stroke-width: 1; stroke-dasharray: 3 5; }
.tick text, .small, .label {
  fill: var(--muted);
  color: var(--muted);
  font-size: 12px;
}
.chart-label {
  font-size: 13px;
  fill: var(--ink);
}
.chart-value {
  font-size: 12px;
  fill: var(--muted);
  font-variant-numeric: tabular-nums;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th {
  text-align: left;
  font-weight: 400;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding: 7px 8px 7px 0;
}
td {
  border-bottom: 1px solid var(--faint);
  padding: 7px 8px 7px 0;
  vertical-align: top;
}
td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
tr.emph td { background: rgba(216, 208, 188, .18); }
.domain-pill {
  display: inline-block;
  border-left: 8px solid currentColor;
  padding-left: 6px;
}
.foot {
  border-top: 1px solid var(--rule);
  padding-top: 18px;
  font-size: 14px;
}
@media (max-width: 820px) {
  main { padding: 28px 18px 54px; }
  .stat-strip, .grid.two { grid-template-columns: 1fr; }
  .stat { border-right: 0; border-bottom: 1px solid var(--rule); }
  .stat:last-child { border-bottom: 0; }
}
"""


def build_seaborn_figures(
    *,
    analysis_rows: list[dict[str, str]],
    univariate_rows: list[dict[str, str]],
    violin_top_n: int,
    assets_dir: Path,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    violin_data = top_feature_violin_frame(analysis_rows, univariate_rows[:violin_top_n])
    if not violin_data.empty:
        figures["top_feature_violins"] = seaborn_violin_data_uri(
            violin_data,
            output_path=assets_dir / "top_feature_violins.png",
        )
    return figures


def top_feature_violin_frame(
    analysis_rows: list[dict[str, str]],
    top_rows: list[dict[str, str]],
) -> "pd.DataFrame":
    source = pd.DataFrame(analysis_rows)
    records: list[dict[str, object]] = []
    if source.empty or "mace_primary" not in source.columns:
        return pd.DataFrame()
    for rank, row in enumerate(top_rows, start=1):
        feature = row.get("feature_name", "")
        if feature not in source.columns:
            continue
        values = pd.to_numeric(source[feature], errors="coerce")
        outcome = source["mace_primary"].map({"0": "No MACE", "1": "MACE"})
        valid = values.notna() & outcome.notna()
        valid_values = values.loc[valid]
        if valid_values.empty:
            continue
        std = valid_values.std(ddof=0)
        if not std or math.isnan(float(std)):
            continue
        z_values = (values.loc[valid] - valid_values.mean()) / std
        label = feature_label(feature)
        if len(label) > 36:
            label = label[:33].rstrip() + "..."
        auc = to_float(row.get("auc_directional", "")) or math.nan
        smd = to_float(row.get("standardized_mean_difference", "")) or math.nan
        facet_label = f"{rank}. {label}\nAUC {auc:.3f} | SMD {smd:+.2f}"
        for case_id, mace_label, z_value in zip(source.loc[valid, "case_id"], outcome.loc[valid], z_values):
            records.append(
                {
                    "case_id": case_id,
                    "outcome": mace_label,
                    "z": float(z_value),
                    "feature": facet_label,
                    "domain": row.get("domain", ""),
                }
            )
    return pd.DataFrame.from_records(records)


def seaborn_violin_data_uri(frame: "pd.DataFrame", *, output_path: Path | None = None) -> str:
    sns.set_theme(
        context="paper",
        style="white",
        font="serif",
        rc={
            "axes.edgecolor": "#d8d0bc",
            "axes.labelcolor": "#151515",
            "axes.titlecolor": "#151515",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "figure.facecolor": "#fffff8",
            "axes.facecolor": "#fffff8",
        },
    )
    order = ["No MACE", "MACE"]
    palette = {"No MACE": "#b9ae98", "MACE": DOMAIN_COLORS["wall_thickness"]}

    def draw_violin(data: "pd.DataFrame", **_: object) -> None:
        ax = plt.gca()
        sns.violinplot(
            data=data,
            x="outcome",
            y="z",
            hue="outcome",
            order=order,
            hue_order=order,
            palette=palette,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            saturation=0.85,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="outcome",
            y="z",
            order=order,
            color="#111111",
            alpha=0.20,
            size=1.8,
            jitter=0.22,
            ax=ax,
        )
        ax.axhline(0, color="#c7bea9", linewidth=0.8, linestyle=(0, (2, 4)), zorder=0)
        ax.set_xlabel("")
        ax.set_ylabel("z" if ax.get_subplotspec().is_first_col() else "")
        ax.tick_params(axis="x", labelrotation=0)

    grid = sns.FacetGrid(
        frame,
        col="feature",
        col_wrap=3,
        sharey=True,
        height=2.45,
        aspect=1.18,
        despine=True,
    )
    grid.map_dataframe(draw_violin)
    grid.set_titles("{col_name}", size=8.8)
    grid.set_ylabels("Standardized feature value")
    for ax in grid.axes.flat:
        ax.grid(False)
        ax.spines["left"].set_color("#d8d0bc")
        ax.spines["bottom"].set_color("#d8d0bc")
    grid.fig.subplots_adjust(top=0.96, hspace=0.46, wspace=0.20)

    buffer = io.BytesIO()
    grid.fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="#fffff8")
    plt.close(grid.fig)
    image_bytes = buffer.getvalue()
    if output_path is not None:
        output_path.write_bytes(image_bytes)
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def embedded_image(data_uri: str, alt: str) -> str:
    if not data_uri:
        return '<p class="caption">No Seaborn figure could be generated.</p>'
    return f'<img class="figure-img" src="{esc(data_uri)}" alt="{esc(alt)}">'


def stat_card(label: str, value: str) -> str:
    return f'<div class="stat"><div class="value">{esc(value)}</div><div class="label">{esc(label)}</div></div>'


def event_fraction(summary: dict[str, object]) -> float:
    events = to_float(summary.get("mace_primary_events", 0)) or 0.0
    total = to_float(summary.get("merged_rows", 0)) or 0.0
    return events / total if total else 0.0


def event_bar(summary: dict[str, object]) -> str:
    events = int(to_float(summary.get("mace_primary_events", 0)) or 0)
    total = int(to_float(summary.get("merged_rows", 0)) or 0)
    nonevents = max(total - events, 0)
    frac = events / total if total else 0.0
    width = 560
    height = 145
    bar_x = 18
    bar_y = 66
    bar_w = width - 36
    event_w = bar_w * frac
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="MACE event burden">
  <text x="18" y="30" class="chart-label">MACE events</text>
  <text x="{width - 18}" y="30" text-anchor="end" class="chart-value">{events}/{total} ({format_percent(frac)})</text>
  <rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="18" fill="#eee8d8"></rect>
  <rect x="{bar_x}" y="{bar_y}" width="{event_w:.1f}" height="18" fill="{DOMAIN_COLORS['wall_thickness']}"></rect>
  <line x1="{bar_x}" x2="{bar_x}" y1="{bar_y - 6}" y2="{bar_y + 30}" class="axis"></line>
  <line x1="{bar_x + bar_w}" x2="{bar_x + bar_w}" y1="{bar_y - 6}" y2="{bar_y + 30}" class="axis"></line>
  <text x="{bar_x}" y="{bar_y + 52}" class="chart-value">0</text>
  <text x="{bar_x + event_w:.1f}" y="{bar_y - 12}" text-anchor="middle" class="chart-value">{events} events</text>
  <text x="{bar_x + bar_w}" y="{bar_y + 52}" text-anchor="end" class="chart-value">{nonevents} non-events</text>
</svg>
"""


def component_rows(summary: dict[str, object]) -> list[tuple[str, int, str]]:
    labels = {
        "all_cause_death": "All-cause death",
        "cardiovascular_hospitalization": "CV hospitalization",
        "recurrent_ischemic_stroke": "Recurrent ischemic stroke",
        "recurrent_tia": "Recurrent TIA",
        "myocardial_infarction": "Myocardial infarction",
        "acute_coronary_infarction": "Acute coronary infarction",
    }
    counts = summary.get("component_counts", {})
    rows: list[tuple[str, int, str]] = []
    if isinstance(counts, dict):
        for key, label in labels.items():
            value = int(to_float(counts.get(key, 0)) or 0)
            if value > 0:
                rows.append((label, value, "component"))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def horizontal_bar_svg(
    rows: list[tuple[str, int, str]],
    *,
    width: int,
    bar_color: str | None,
) -> str:
    if not rows:
        return '<p class="caption">No rows available.</p>'
    row_h = 30
    left = 170
    right = 52
    top = 14
    height = top * 2 + row_h * len(rows)
    max_value = max(value for _, value, _ in rows) or 1
    plot_w = width - left - right
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i, (label, value, domain) in enumerate(rows):
        y = top + i * row_h + 14
        bar_w = plot_w * value / max_value
        color = bar_color or DOMAIN_COLORS.get(domain, "#555")
        pieces.append(f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" class="chart-label">{esc(label)}</text>')
        pieces.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y}" y2="{y}" class="axis"></line>')
        pieces.append(f'<rect x="{left}" y="{y - 6}" width="{bar_w:.1f}" height="12" fill="{color}"></rect>')
        pieces.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 4}" class="chart-value">{value}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def auc_lollipop_svg(rows: list[dict[str, str]]) -> str:
    if not rows:
        return '<p class="caption">No univariate rows available.</p>'
    width = 1120
    row_h = 26
    top = 26
    bottom = 42
    left = 365
    right = 112
    height = top + bottom + row_h * len(rows)
    x_min = 0.50
    x_max = max(0.65, max(to_float(row.get("auc_directional", "")) or 0.5 for row in rows) + 0.015)
    plot_w = width - left - right
    chance_x = scale(0.5, x_min, x_max, left, left + plot_w)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Top univariate AUROC features">']
    pieces.append(f'<line x1="{chance_x:.1f}" x2="{chance_x:.1f}" y1="8" y2="{height - bottom + 10}" class="guide"></line>')
    pieces.append(f'<text x="{chance_x:.1f}" y="{height - 12}" text-anchor="middle" class="chart-value">0.50</text>')
    pieces.append(f'<text x="{left + plot_w}" y="{height - 12}" text-anchor="middle" class="chart-value">{x_max:.2f}</text>')
    for i, row in enumerate(rows):
        y = top + i * row_h
        auc = to_float(row.get("auc_directional", "")) or 0.5
        x = scale(auc, x_min, x_max, left, left + plot_w)
        domain = row.get("domain", "")
        color = DOMAIN_COLORS.get(domain, "#555")
        label = feature_label(row.get("feature_name", ""))
        pieces.append(f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" class="chart-label"><title>{esc(row.get("feature_name", ""))}</title>{esc(label)}</text>')
        pieces.append(f'<line x1="{chance_x:.1f}" x2="{x:.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="1.5"></line>')
        pieces.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.2" fill="{color}"></circle>')
        pieces.append(f'<text x="{left + plot_w + 12}" y="{y + 4}" class="chart-value">{auc:.3f}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def smd_dot_svg(rows: list[dict[str, str]]) -> str:
    if not rows:
        return '<p class="caption">No univariate rows available.</p>'
    width = 1120
    row_h = 26
    top = 24
    bottom = 42
    left = 365
    right = 112
    height = top + bottom + row_h * len(rows)
    max_abs = max(abs(to_float(row.get("standardized_mean_difference", "")) or 0.0) for row in rows)
    x_min = -max(0.45, max_abs + 0.05)
    x_max = max(0.45, max_abs + 0.05)
    plot_w = width - left - right
    zero_x = scale(0.0, x_min, x_max, left, left + plot_w)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="SMD feature effects">']
    pieces.append(f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="8" y2="{height - bottom + 10}" class="guide"></line>')
    pieces.append(f'<text x="{zero_x:.1f}" y="{height - 12}" text-anchor="middle" class="chart-value">0</text>')
    pieces.append(f'<text x="{left}" y="{height - 12}" text-anchor="middle" class="chart-value">{x_min:.1f}</text>')
    pieces.append(f'<text x="{left + plot_w}" y="{height - 12}" text-anchor="middle" class="chart-value">+{x_max:.1f}</text>')
    for i, row in enumerate(rows):
        y = top + i * row_h
        smd = to_float(row.get("standardized_mean_difference", "")) or 0.0
        x = scale(smd, x_min, x_max, left, left + plot_w)
        domain = row.get("domain", "")
        color = DOMAIN_COLORS.get(domain, "#555")
        label = feature_label(row.get("feature_name", ""))
        pieces.append(f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" class="chart-label"><title>{esc(row.get("feature_name", ""))}</title>{esc(label)}</text>')
        pieces.append(f'<line x1="{zero_x:.1f}" x2="{x:.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="1.5"></line>')
        pieces.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.2" fill="{color}"></circle>')
        pieces.append(f'<text x="{left + plot_w + 12}" y="{y + 4}" class="chart-value">{smd:+.2f}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def grouped_bar_svg(counts: dict[str, dict[str, int]]) -> str:
    labels = sorted(counts)
    if not labels:
        return '<p class="caption">No rows available.</p>'
    width = 560
    row_h = 34
    top = 20
    left = 170
    right = 52
    height = top * 2 + row_h * len(labels)
    plot_w = width - left - right
    max_value = max(max(group.values()) for group in counts.values()) or 1
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i, label in enumerate(labels):
        y = top + i * row_h + 10
        event = counts[label].get("event", 0)
        nonevent = counts[label].get("nonevent", 0)
        event_w = plot_w * event / max_value
        nonevent_w = plot_w * nonevent / max_value
        pieces.append(f'<text x="{left - 12}" y="{y + 7}" text-anchor="end" class="chart-label">{esc(label or "(blank)")}</text>')
        pieces.append(f'<rect x="{left}" y="{y - 8}" width="{event_w:.1f}" height="8" fill="{DOMAIN_COLORS["wall_thickness"]}"></rect>')
        pieces.append(f'<rect x="{left}" y="{y + 3}" width="{nonevent_w:.1f}" height="8" fill="#b9ae98"></rect>')
        pieces.append(f'<text x="{left + max(event_w, nonevent_w) + 8:.1f}" y="{y + 7}" class="chart-value">{event}/{nonevent}</text>')
    pieces.append(f'<text x="{left}" y="{height - 4}" class="chart-value">blue = MACE, tan = no MACE</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def missingness_svg(rows: list[tuple[str, int, str]]) -> str:
    if not rows:
        return '<p class="caption">No missing feature values.</p>'
    return horizontal_bar_svg([(feature_label(name), count, domain) for name, count, domain in rows], width=560, bar_color="#8f8a7d")


def top_feature_table(rows: list[dict[str, str]]) -> str:
    table_rows = []
    for row in rows:
        domain = row.get("domain", "")
        color = DOMAIN_COLORS.get(domain, "#555")
        table_rows.append(
            [
                f'<span class="domain-pill" style="color:{color}">{esc(DOMAIN_LABELS.get(domain, domain))}</span>',
                f'<span title="{esc(row.get("feature_name", ""))}">{esc(feature_label(row.get("feature_name", "")))}</span>',
                esc(row.get("n", "")),
                esc(row.get("n_events", "")),
                esc(row.get("event_mean", "")),
                esc(row.get("nonevent_mean", "")),
                esc(row.get("standardized_mean_difference", "")),
                esc(row.get("auc_directional", "")),
            ]
        )
    return table_html(
        ["Domain", "Feature", "N", "Events", "Event mean", "No-event mean", "SMD", "Directional AUROC"],
        table_rows,
        raw=True,
        numeric_columns={2, 3, 4, 5, 6, 7},
    )


def table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    raw: bool = False,
    numeric_columns: set[int] | None = None,
    row_class=None,
) -> str:
    numeric_columns = numeric_columns or set()
    parts = ["<table>", "<thead><tr>"]
    for i, header in enumerate(headers):
        cls = ' class="num"' if i in numeric_columns else ""
        parts.append(f"<th{cls}>{esc(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        cls = f' class="{row_class(row)}"' if row_class else ""
        parts.append(f"<tr{cls}>")
        for i, value in enumerate(row):
            cell = str(value) if raw else esc(str(value))
            cell_cls = ' class="num"' if i in numeric_columns else ""
            parts.append(f"<td{cell_cls}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def covariate_summary(event_rows: list[dict[str, str]], non_event_rows: list[dict[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    event_age = numeric_values(event_rows, "age")
    non_event_age = numeric_values(non_event_rows, "age")
    rows.append(
        [
            "Age, years",
            mean_sd(event_age),
            mean_sd(non_event_age),
            diff_text(mean(event_age) - mean(non_event_age) if event_age and non_event_age else math.nan),
        ]
    )
    for column, label in RISK_FACTORS:
        rows.append(
            [
                label,
                binary_fraction(event_rows, column),
                binary_fraction(non_event_rows, column),
                binary_difference(event_rows, non_event_rows, column),
            ]
        )
    return rows


def feature_missingness(rows: list[dict[str, str]], feature_columns: list[str]) -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    for column in feature_columns:
        missing = sum(1 for row in rows if not row.get(column, "").strip())
        if missing:
            result.append((column, missing, feature_domain(column)))
    return sorted(result, key=lambda item: item[1], reverse=True)


def two_group_counts(rows: list[dict[str, str]], column: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"event": 0, "nonevent": 0})
    for row in rows:
        group = "event" if row.get("mace_primary") == "1" else "nonevent"
        value = row.get(column, "") or "(blank)"
        counts[value][group] += 1
    return dict(counts)


def numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = to_float(row.get(column, ""))
        if value is not None:
            values.append(value)
    return values


def mean_sd(values: list[float]) -> str:
    if not values:
        return ""
    mu = mean(values)
    if len(values) < 2:
        return f"{mu:.1f}"
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return f"{mu:.1f} ± {math.sqrt(var):.1f}"


def binary_fraction(rows: list[dict[str, str]], column: str) -> str:
    valid = [row for row in rows if row.get(column, "") in {"0", "1"}]
    if not valid:
        return ""
    yes = sum(1 for row in valid if row.get(column) == "1")
    return f"{yes}/{len(valid)} ({100 * yes / len(valid):.1f}%)"


def binary_difference(event_rows: list[dict[str, str]], non_event_rows: list[dict[str, str]], column: str) -> str:
    event_valid = [row for row in event_rows if row.get(column, "") in {"0", "1"}]
    non_event_valid = [row for row in non_event_rows if row.get(column, "") in {"0", "1"}]
    if not event_valid or not non_event_valid:
        return ""
    event_p = sum(1 for row in event_valid if row.get(column) == "1") / len(event_valid)
    non_event_p = sum(1 for row in non_event_valid if row.get(column) == "1") / len(non_event_valid)
    return f"{100 * (event_p - non_event_p):+.1f} pp"


def diff_text(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:+.1f}"


def feature_domain(name: str) -> str:
    lower = name.lower()
    if "wall_thickness" in lower:
        return "wall_thickness"
    if "wall_from_fat" in lower or "experimental_wall_from_fat" in lower:
        return "wall_from_fat"
    if "calcium" in lower or "calcification" in lower:
        return "calcium"
    if "fat" in lower or "periaortic" in lower:
        return "fat"
    return ""


def feature_label(name: str) -> str:
    label = name
    replacements = [
        ("aortic_wall__wall_thickness_threshold__", ""),
        ("aortic_wall__wall_thickness__", ""),
        ("aorta_wall_dynamic__calcification_dynamic_threshold__", "dynamic threshold "),
        ("aorta_wall_dynamic__calcification__", "dynamic calcium "),
        ("aorta_wall_band__calcification__", "wall band calcium "),
        ("aorta_wall_from_fat__experimental_wall_from_fat_lumen__", "wall-from-fat "),
        ("periaortic_fat__fat_omics__", "fat "),
        ("periaortic_fat__radiomics_", "fat radiomics "),
        ("aorta__calcium_omics__", "calcium "),
        ("aorta_segment:whole_aorta__calcium_omics__", "whole-aorta calcium "),
        ("__thr_dynamic_lumen_referenced_seed500HU", " (dynamic)"),
        ("__thr_dynamic_lumen_referenced", " (dynamic)"),
        ("__thr_> 4 mm", " (>4 mm)"),
        ("__thr_", " (thr "),
        ("original_", ""),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.replace("__", " / ").replace("_", " ")
    label = label.replace("hu", "HU").replace("mm3", "mm3")
    if len(label) > 58:
        label = label[:55].rstrip() + "..."
    return label


def scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + (value - source_min) * (target_max - target_min) / (source_max - source_min)


def to_float(value: object) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def fmt_int(value: object) -> str:
    number = to_float(value)
    if number is None:
        return str(value)
    return f"{int(number):,}"


def format_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
