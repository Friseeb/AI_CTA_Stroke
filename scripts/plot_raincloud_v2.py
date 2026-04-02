#!/usr/bin/env python
"""
plot_raincloud_v2.py — Horizontal raincloud plots for SLAAO radiomics QC.
Uses only matplotlib, numpy, pandas, scipy — no external raincloud library.

Usage:
    conda run -n cardiac-ct-explorer python scripts/plot_raincloud_v2.py \
        --csv <path/to/radiomics_ibsi_all.csv> \
        --outdir <path/to/plots/> \
        [--plot4-only]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde


# ── Colour palette (consistent per CT type across all plots) ─────────────────
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
    idx = list(all_ct_types).index(ct_type) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx]


# ── Core raincloud row ───────────────────────────────────────────────────────
def _draw_raincloud_row(ax, data, y_center, color, *,
                        log_scale=False,
                        cloud_height=0.32, cloud_gap=0.03,
                        box_height=0.09,
                        dot_gap=0.08, dot_spread=0.14,
                        alpha_cloud=0.55, alpha_dots=0.45, dot_size=8,
                        rng=None):
    """
    Draw a single horizontal raincloud row centred at y_center:
      - half-violin KDE (above y_center)
      - boxplot         (at    y_center)
      - jittered dots   (below y_center)
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

    # ── KDE cloud ─────────────────────────────────────────────────────────
    kde_input = np.log10(data) if log_scale else data
    try:
        kde = gaussian_kde(kde_input, bw_method='scott')
    except np.linalg.LinAlgError:
        kde = gaussian_kde(kde_input, bw_method=0.4)

    x_lo, x_hi = kde_input.min(), kde_input.max()
    xs = np.linspace(x_lo, x_hi, 300)
    ys = kde(xs)
    ys_norm = ys / ys.max() * cloud_height

    x_plot = 10 ** xs if log_scale else xs
    base = y_center + cloud_gap
    ax.fill_between(x_plot, base, base + ys_norm,
                    color=color, alpha=alpha_cloud, linewidth=0, zorder=3)
    ax.plot(x_plot, base + ys_norm, color=color, lw=0.9, alpha=0.85, zorder=3)

    # ── Boxplot ───────────────────────────────────────────────────────────
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    wlo = max(data.min(), q25 - 1.5 * iqr)
    whi = min(data.max(), q75 + 1.5 * iqr)
    outliers = data[(data < wlo) | (data > whi)]

    blo = y_center - box_height / 2
    bhi = y_center + box_height / 2
    ax.add_patch(plt.Rectangle(
        (q25, blo), q75 - q25, box_height,
        facecolor=color, alpha=0.78, edgecolor='#333', lw=0.8, zorder=4))
    ax.plot([q50, q50], [blo, bhi], color='#111', lw=1.6, zorder=5)
    ax.plot([wlo, q25], [y_center, y_center], color='#444', lw=0.9, zorder=4)
    ax.plot([q75, whi], [y_center, y_center], color='#444', lw=0.9, zorder=4)
    cap = box_height * 0.35
    for tip in (wlo, whi):
        ax.plot([tip, tip], [y_center - cap, y_center + cap], color='#444', lw=0.9, zorder=4)
    if len(outliers):
        ax.scatter(outliers, np.full(len(outliers), y_center),
                   c=color, s=dot_size * 1.8, alpha=0.6, zorder=6, edgecolors='none')

    # ── Jittered scatter dots ─────────────────────────────────────────────
    dot_y = y_center - dot_gap - dot_spread / 2
    jitter = rng.uniform(-dot_spread / 2, dot_spread / 2, size=n)
    ax.scatter(data, dot_y + jitter,
               c=color, s=dot_size, alpha=alpha_dots, zorder=3, edgecolors='none')


def _setup_ax(ax, ct_types, data_dict, all_ct_types, *,
              log_scale=False, row_spacing=1.25):
    """
    Populate ax with one raincloud row per ct_type.
    data_dict: {ct_type: np.ndarray}
    Returns list of y positions.
    """
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
            _draw_raincloud_row(ax, valid, y,
                                color=_ct_colour(ct, all_ct_types),
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4 — Median HU by structure  (built first for preview)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_hu_by_structure(df, ct_types, all_ct_types, outdir):
    structures = [
        ('LAA',   'laa_original_firstorder_Median'),
        ('LA',    'la_original_firstorder_Median'),
        ('Aorta', 'aorta_original_firstorder_Median'),
    ]
    fh = _fig_height(len(ct_types))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fh), sharey=False)

    for ax, (struct, col) in zip(axes, structures):
        data_dict = {
            ct: df.loc[df['ct_type'] == ct, col].dropna().values
            for ct in ct_types
        }
        _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=False)
        ax.set_xlabel('Median HU', fontsize=9)
        ax.set_title(struct, fontsize=12, fontweight='bold')
        ax.axvline(0, color='#aaa', lw=0.7, ls='--', zorder=1)

    fig.suptitle('Median HU by Structure and CT Type', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_hu_by_structure.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — ROI voxels (pre-resampling), log scale
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_roi_voxels(df, ct_types, all_ct_types, outdir):
    structures = [
        ('LAA',   'laa_roi_voxels'),
        ('LA',    'la_roi_voxels'),
        ('Aorta', 'aorta_roi_voxels'),
    ]
    fh = _fig_height(len(ct_types))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fh))

    for ax, (struct, col) in zip(axes, structures):
        data_dict = {
            ct: df.loc[df['ct_type'] == ct, col].dropna().values
            for ct in ct_types
        }
        _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=True)
        ax.set_xlabel('Voxel Count (pre-resampling)', fontsize=9)
        ax.set_title(struct, fontsize=12, fontweight='bold')

    fig.suptitle('ROI Voxel Count — Pre-Resampling (log scale)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_roi_voxels.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — Before vs after resampling
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_before_after(df, ct_types, all_ct_types, outdir):
    structures = [
        ('LAA',   'laa_roi_voxels',   'laa_diagnostics_Mask-interpolated_VoxelNum'),
        ('LA',    'la_roi_voxels',    'la_diagnostics_Mask-interpolated_VoxelNum'),
        ('Aorta', 'aorta_roi_voxels', 'aorta_diagnostics_Mask-interpolated_VoxelNum'),
    ]
    PRE_CLR  = '#e6812e'
    POST_CLR = '#2e86e6'
    row_spacing = 1.5
    rng = np.random.default_rng(42)

    fh = _fig_height(len(ct_types), row_spacing=row_spacing) + 0.8
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fh))

    for ax, (struct, pre_col, post_col) in zip(axes, structures):
        y_positions, y_labels = [], []
        for i, ct in enumerate(ct_types):
            y = i * row_spacing
            y_positions.append(y)

            sub = df[df['ct_type'] == ct]
            pre  = sub[pre_col].dropna().values.astype(float)
            post = sub[post_col].dropna().values.astype(float)
            pre  = pre[pre > 0];  n_pre  = len(pre)
            post = post[post > 0]; n_post = len(post)
            y_labels.append(f"{ct}  pre={n_pre} / post={n_post}")

            offset = 0.27
            kw = dict(log_scale=True, cloud_height=0.20, box_height=0.07,
                      dot_spread=0.10, dot_gap=0.06, rng=rng)
            if n_pre  >= 2:
                _draw_raincloud_row(ax, pre,  y - offset, PRE_CLR,  **kw)
            if n_post >= 2:
                _draw_raincloud_row(ax, post, y + offset, POST_CLR, **kw)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_ylim(-(row_spacing * 0.65),
                    (len(ct_types) - 1) * row_spacing + row_spacing)
        ax.set_xlabel('Voxel Count', fontsize=9)
        ax.set_title(struct, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = [
        mpatches.Patch(color=PRE_CLR,  label='Pre-resampling (original)'),
        mpatches.Patch(color=POST_CLR, label='Post-resampling (interpolated)'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               framealpha=0.7, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle('Voxel Count Before vs After Resampling (log scale)',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_before_after.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3 — Interpolated voxel count with QC thresholds
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_interpolated_volumes(df, ct_types, all_ct_types, outdir):
    structures = [
        ('LAA',   'laa_diagnostics_Mask-interpolated_VoxelNum',
         dict(hard=1000,  soft=5000)),
        ('LA',    'la_diagnostics_Mask-interpolated_VoxelNum',
         dict(hard=10000, soft=50000)),
        ('Aorta', 'aorta_diagnostics_Mask-interpolated_VoxelNum',
         None),
    ]
    fh = _fig_height(len(ct_types))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, fh))

    for ax, (struct, col, thresh) in zip(axes, structures):
        data_dict = {
            ct: df.loc[df['ct_type'] == ct, col].dropna().values
            for ct in ct_types
        }
        _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=True)
        ax.set_xlabel('Interpolated VoxelNum', fontsize=9)
        ax.set_title(struct, fontsize=12, fontweight='bold')
        if thresh:
            ax.axvline(thresh['hard'], color='#cc0000', lw=1.3, ls='--',
                       label=f"Hard = {thresh['hard']:,}", zorder=2)
            ax.axvline(thresh['soft'], color='#e68a00', lw=1.3, ls=':',
                       label=f"Soft = {thresh['soft']:,}", zorder=2)
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.75)

    fig.suptitle('Interpolated Voxel Count by Structure and CT Type (log scale)',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_interpolated_volumes.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 5 — LA / Aorta HU ratio
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_hu_ratio(df, ct_types, all_ct_types, outdir):
    col = 'ratio_la_aorta_median'
    fh = _fig_height(len(ct_types))
    fig, ax = plt.subplots(figsize=(7.5, fh))
    data_dict = {
        ct: df.loc[df['ct_type'] == ct, col].dropna().values
        for ct in ct_types
    }
    _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=False)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 6 — LAA / LA HU ratio  (LAA-valid rows only)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_laa_la_ratio(df, ct_types, all_ct_types, outdir):
    col_ratio = 'ratio_laa_la_median'
    col_laa   = 'laa_original_firstorder_Median'
    df_v = df[df[col_laa].notna()]
    fh = _fig_height(len(ct_types))
    fig, ax = plt.subplots(figsize=(7.5, fh))
    data_dict = {
        ct: df_v.loc[df_v['ct_type'] == ct, col_ratio].dropna().values
        for ct in ct_types
    }
    _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=False)
    ax.set_xlabel('Ratio (LAA Median HU / LA Median HU)', fontsize=9)
    ax.set_title('LAA / LA Median HU Ratio\n(subjects with valid LAA features only)',
                  fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_laa_la_ratio.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 7 — LAA / Aorta HU ratio  (LAA-valid rows only)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_laa_aorta_ratio(df, ct_types, all_ct_types, outdir):
    col_ratio = 'ratio_laa_aorta_median'
    col_laa   = 'laa_original_firstorder_Median'
    df_v = df[df[col_laa].notna()]
    fh = _fig_height(len(ct_types))
    fig, ax = plt.subplots(figsize=(7.5, fh))
    data_dict = {
        ct: df_v.loc[df_v['ct_type'] == ct, col_ratio].dropna().values
        for ct in ct_types
    }
    _setup_ax(ax, ct_types, data_dict, all_ct_types, log_scale=False)
    ax.set_xlabel('Ratio (LAA Median HU / Aorta Median HU)', fontsize=9)
    ax.set_title('LAA / Aorta Median HU Ratio\n(subjects with valid LAA features only)',
                  fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = outdir / 'raincloud_laa_aorta_ratio.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Horizontal raincloud plots for SLAAO radiomics QC'
    )
    parser.add_argument('--csv',        required=True,
                        help='Path to radiomics CSV')
    parser.add_argument('--outdir',     required=True,
                        help='Output folder for PNGs')
    parser.add_argument('--plot4-only', action='store_true',
                        help='Generate only raincloud_hu_by_structure.png (preview test)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    all_ct_types = sorted(df['ct_type'].dropna().unique().tolist())
    ct_types = all_ct_types  # same; explicit for clarity

    print(f"CT types found : {ct_types}")
    print(f"Total rows     : {len(df)}")
    print(f"Output folder  : {outdir}")
    print()

    # Plot 4 always runs first (preview / validation)
    _plot_hu_by_structure(df, ct_types, all_ct_types, outdir)

    if not args.plot4_only:
        _plot_roi_voxels(df, ct_types, all_ct_types, outdir)
        _plot_before_after(df, ct_types, all_ct_types, outdir)
        _plot_interpolated_volumes(df, ct_types, all_ct_types, outdir)
        _plot_hu_ratio(df, ct_types, all_ct_types, outdir)
        _plot_laa_la_ratio(df, ct_types, all_ct_types, outdir)
        _plot_laa_aorta_ratio(df, ct_types, all_ct_types, outdir)

    print("\nDone.")


if __name__ == '__main__':
    main()
