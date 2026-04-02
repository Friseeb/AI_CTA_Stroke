#!/usr/bin/env python
"""
plot_05_hu_ratio.py — LA/Aorta median HU ratio.
Produces: raincloud_hu_ratio.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Colour palette ────────────────────────────────────────────────────────────
_CT_COLOURS = {
    'eCTA':         '#e64c2e',
    'CT_abdomen':   '#3182bd',
    'CT_totalbody': '#31a354',
    'CT_thorax':    '#756bb1',
    'CT_heart':     '#fd8d3c',
}
_FALLBACK_PALETTE = ['#8c6d31', '#637939', '#393b79', '#843c39', '#5254a3']

def _ct_colour(ct_type, all_ct_types):
    if ct_type in _CT_COLOURS:
        return _CT_COLOURS[ct_type]
    return _FALLBACK_PALETTE[list(all_ct_types).index(ct_type) % len(_FALLBACK_PALETTE)]


# ── Core drawing function (identical across all plot_0N scripts) ──────────────
def _draw_raincloud_row(ax, data, y_center, color, *,
                        log_scale=False,
                        cloud_height=0.32, cloud_gap=0.03,
                        box_height=0.09,
                        dot_gap=0.08, dot_spread=0.14,
                        alpha_cloud=0.55, alpha_dots=0.45, dot_size=8,
                        rng=None):
    """
    Horizontal raincloud row on an INVERTED y-axis:
      - half-violin KDE cloud  ABOVE y_center  (lower y numerically → higher visually)
      - boxplot                AT    y_center
      - jittered scatter dots  BELOW y_center  (higher y numerically → lower visually)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if log_scale:
        data = data[data > 0]
    n = len(data)
    if n < 2:
        return

    # ── KDE cloud (above center: lower y numerically) ─────────────────────
    kde_input = np.log10(data) if log_scale else data
    try:
        kde = gaussian_kde(kde_input, bw_method='scott')
    except np.linalg.LinAlgError:
        kde = gaussian_kde(kde_input, bw_method=0.4)
    xs = np.linspace(kde_input.min(), kde_input.max(), 300)
    ys_norm = kde(xs)
    ys_norm = ys_norm / ys_norm.max() * cloud_height
    x_plot = 10 ** xs if log_scale else xs
    base = y_center - cloud_gap                          # inner edge, just above center
    ax.fill_between(x_plot, base - ys_norm, base,
                    color=color, alpha=alpha_cloud, linewidth=0, zorder=3)
    ax.plot(x_plot, base - ys_norm, color=color, lw=0.9, alpha=0.85, zorder=3)

    # ── Boxplot (at center) ────────────────────────────────────────────────
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    wlo = max(data.min(), q25 - 1.5 * iqr)
    whi = min(data.max(), q75 + 1.5 * iqr)
    outliers = data[(data < wlo) | (data > whi)]
    blo, bhi = y_center - box_height / 2, y_center + box_height / 2
    ax.add_patch(plt.Rectangle((q25, blo), q75 - q25, box_height,
                                facecolor=color, alpha=0.78, edgecolor='#333', lw=0.8, zorder=4))
    ax.plot([q50] * 2, [blo, bhi], color='#111', lw=1.6, zorder=5)
    ax.plot([wlo, q25], [y_center] * 2, color='#444', lw=0.9, zorder=4)
    ax.plot([q75, whi], [y_center] * 2, color='#444', lw=0.9, zorder=4)
    cap = box_height * 0.35
    for tip in (wlo, whi):
        ax.plot([tip] * 2, [y_center - cap, y_center + cap], color='#444', lw=0.9, zorder=4)
    if len(outliers):
        ax.scatter(outliers, np.full(len(outliers), y_center),
                   c=color, s=dot_size * 1.8, alpha=0.6, zorder=6, edgecolors='none')

    # ── Jittered dots (below center: higher y numerically) ────────────────
    dot_center = y_center + dot_gap + dot_spread / 2
    jitter = rng.uniform(-dot_spread / 2, dot_spread / 2, size=n)
    ax.scatter(data, dot_center + jitter,
               c=color, s=dot_size, alpha=alpha_dots, zorder=3, edgecolors='none')


# ── Axis setup helper ─────────────────────────────────────────────────────────
def _setup_ax(ax, ct_types, data_dict, all_ct_types, *,
              log_scale=False, row_spacing=1.25):
    rng = np.random.default_rng(42)
    y_positions, y_labels = [], []
    for i, ct in enumerate(ct_types):
        y = i * row_spacing
        y_positions.append(y)
        arr = np.asarray(data_dict.get(ct, []), dtype=float)
        valid = arr[np.isfinite(arr)]
        if log_scale:
            valid = valid[valid > 0]
        n = len(valid)
        y_labels.append(f"{ct}  n={n}")
        if n >= 2:
            _draw_raincloud_row(ax, valid, y, color=_ct_colour(ct, all_ct_types),
                                log_scale=log_scale, rng=rng)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ypad = row_spacing * 0.65
    ax.set_ylim(-ypad, (len(ct_types) - 1) * row_spacing + row_spacing)
    ax.invert_yaxis()
    if log_scale:
        ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return y_positions


def _fig_height(n_rows, row_spacing=1.25):
    return max(3.5, n_rows * row_spacing + 2.2)


# ── Plot ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='LA/Aorta median HU ratio raincloud')
    parser.add_argument('--csv',    required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    ct_types = sorted(df['ct_type'].dropna().unique().tolist())
    print(f"CT types: {ct_types}  |  rows: {len(df)}")

    col = 'ratio_la_aorta_median'
    fh = _fig_height(len(ct_types))
    fig, ax = plt.subplots(figsize=(7.5, fh))

    data_dict = {ct: df.loc[df['ct_type'] == ct, col].dropna().values for ct in ct_types}
    _setup_ax(ax, ct_types, data_dict, ct_types, log_scale=False)
    ax.axvline(0.8, color='#cc0000', lw=1.3, ls='--', label='0.8', zorder=2)
    ax.axvline(1.2, color='#cc0000', lw=1.3, ls='--', label='1.2', zorder=2)
    ax.legend(fontsize=9, framealpha=0.75)
    ax.set_xlabel('Ratio (LA Median HU / Aorta Median HU)', fontsize=9)
    ax.set_title('LA / Aorta Median HU Ratio', fontsize=12, fontweight='bold')

    fig.tight_layout()
    out = outdir / 'raincloud_hu_ratio.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
