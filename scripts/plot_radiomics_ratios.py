"""
Radiomics ratio plots — 4 figures saved as PNG.
Input:  radiomics_ts_laa.csv  (SLAAOBIDS/derivatives/)
Output: 4 PNG files in the same folder.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
CSV     = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS\derivatives\radiomics_ts_laa.csv")
OUT_DIR = CSV.parent

# ── load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df = df.dropna(subset=["ratio_laa_la_median"]).copy()
print(f"Rows loaded: {len(df)}")
print(df["ct_type"].value_counts().to_string())

CT_TYPES = ["eCTA", "CT_thorax", "CT_abdomen"]
PALETTE  = {"eCTA": "#4878CF", "CT_thorax": "#6ACC65", "CT_abdomen": "#D65F5F"}

sns.set_theme(style="whitegrid", context="notebook")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Boxplot: absolute HU medians, grouped by structure × ct_type
# ─────────────────────────────────────────────────────────────────────────────
long1 = df.melt(
    id_vars=["sub_id", "ct_type"],
    value_vars=[
        "laa_original_firstorder_Median",
        "la_original_firstorder_Median",
        "aorta_original_firstorder_Median",
    ],
    var_name="structure", value_name="HU_median",
)
long1["structure"] = long1["structure"].map({
    "laa_original_firstorder_Median":   "LAA",
    "la_original_firstorder_Median":    "LA",
    "aorta_original_firstorder_Median": "Aorta",
})

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=long1, x="structure", y="HU_median", hue="ct_type",
    palette=PALETTE, order=["LAA", "LA", "Aorta"], hue_order=CT_TYPES,
    ax=ax, flierprops=dict(marker="o", markersize=3, alpha=0.5),
    linewidth=1.2,
)
ax.set_title("Absolute HU Median Values by Structure and CT Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Structure", fontsize=12)
ax.set_ylabel("HU Median", fontsize=12)
ax.legend(title="CT Type", fontsize=10)
sns.despine()
plt.tight_layout()
out1 = OUT_DIR / "plot1_hu_median_boxplot.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Boxplot: three median ratios, grouped by ratio × ct_type
# ─────────────────────────────────────────────────────────────────────────────
long2 = df.melt(
    id_vars=["sub_id", "ct_type"],
    value_vars=["ratio_laa_la_median", "ratio_laa_aorta_median", "ratio_la_aorta_median"],
    var_name="ratio", value_name="value",
)
long2["ratio"] = long2["ratio"].map({
    "ratio_laa_la_median":    "LAA / LA",
    "ratio_laa_aorta_median": "LAA / Aorta",
    "ratio_la_aorta_median":  "LA / Aorta",
})

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=long2, x="ratio", y="value", hue="ct_type",
    palette=PALETTE, order=["LAA / LA", "LAA / Aorta", "LA / Aorta"],
    hue_order=CT_TYPES, ax=ax,
    flierprops=dict(marker="o", markersize=3, alpha=0.5),
    linewidth=1.2,
)
ax.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.0, label="Ratio = 1")
ax.set_title("HU Median Ratios by CT Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Ratio", fontsize=12)
ax.set_ylabel("Ratio Value", fontsize=12)
ax.legend(title="CT Type", fontsize=10)
sns.despine()
plt.tight_layout()
out2 = OUT_DIR / "plot2_ratio_boxplot.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Histogram: ratio distributions — eCTA and CT_thorax only
# ─────────────────────────────────────────────────────────────────────────────
TYPES_HIST   = ["eCTA", "CT_thorax"]
RATIO_COLS   = ["ratio_laa_la_median", "ratio_laa_aorta_median"]
RATIO_LABELS = {"ratio_laa_la_median": "LAA / LA", "ratio_laa_aorta_median": "LAA / Aorta"}
RATIO_COLORS = {"ratio_laa_la_median": "#4878CF",  "ratio_laa_aorta_median": "#D65F5F"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, ct in zip(axes, TYPES_HIST):
    sub = df[df["ct_type"] == ct]
    for col in RATIO_COLS:
        ax.hist(
            sub[col].dropna(), bins=30,
            color=RATIO_COLORS[col], alpha=0.55, label=RATIO_LABELS[col],
            edgecolor="none",
        )
        median_val = sub[col].median()
        ax.axvline(median_val, color=RATIO_COLORS[col], linestyle="--",
                   linewidth=1.2, alpha=0.9)
    ax.axvline(1.0, color="dimgray", linestyle=":", linewidth=1.0, label="Ratio = 1")
    ax.set_title(f"{ct}  (n = {len(sub)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ratio Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)

fig.suptitle(
    "Distribution of HU Median Ratios (LAA/LA and LAA/Aorta)\n"
    "Dashed lines = group medians   |   CT_abdomen excluded (n = 9)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
out3 = OUT_DIR / "plot3_ratio_histogram.png"
plt.savefig(out3, dpi=150)
plt.close()
print(f"Saved: {out3}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Scatter: ratio_laa_la_median (X) vs ratio_laa_aorta_median (Y)
#           color = ct_type   |   point size = ratio_la_aorta_median
# ─────────────────────────────────────────────────────────────────────────────
size_raw = df["ratio_la_aorta_median"].values.astype(float)
vmin, vmax = np.nanpercentile(size_raw, 2), np.nanpercentile(size_raw, 98)
size_norm = 20 + 200 * np.clip((size_raw - vmin) / (vmax - vmin + 1e-9), 0, 1)

CT_MARKERS = {"eCTA": "o", "CT_thorax": "s", "CT_abdomen": "^"}

fig, ax = plt.subplots(figsize=(9, 7))
for ct in CT_TYPES:
    mask = (df["ct_type"] == ct).values
    ax.scatter(
        df.loc[mask, "ratio_laa_la_median"],
        df.loc[mask, "ratio_laa_aorta_median"],
        c=PALETTE[ct],
        s=size_norm[mask],
        marker=CT_MARKERS[ct],
        alpha=0.72,
        label=ct,
        edgecolors="white",
        linewidths=0.4,
    )

ax.axhline(1.0, color="dimgray", linestyle="--", linewidth=0.8)
ax.axvline(1.0, color="dimgray", linestyle="--", linewidth=0.8)

# Legend 1: ct_type
leg1 = ax.legend(title="CT Type", loc="upper left", framealpha=0.85, fontsize=10)

# Legend 2: size proxy for ratio_la_aorta_median
q_pcts  = [0.10, 0.50, 0.90]
q_vals  = [float(np.nanpercentile(size_raw, p * 100)) for p in q_pcts]
q_sizes = [float(20 + 200 * np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1)) for v in q_vals]
size_handles = [
    plt.scatter([], [], s=sz, color="gray", alpha=0.7,
                label=f"p{int(p*100)}: {v:.2f}")
    for p, v, sz in zip(q_pcts, q_vals, q_sizes)
]
ax.legend(handles=size_handles, title="LA / Aorta\n(point size)", loc="lower right",
          framealpha=0.85, fontsize=10)
ax.add_artist(leg1)

ax.set_xlabel("Ratio  LAA / LA  (median HU)", fontsize=12)
ax.set_ylabel("Ratio  LAA / Aorta  (median HU)", fontsize=12)
ax.set_title(
    "LAA/LA vs LAA/Aorta Median Ratios\n"
    "Point size encodes LA/Aorta ratio   |   color = CT type",
    fontsize=13, fontweight="bold",
)
sns.despine()
plt.tight_layout()
out4 = OUT_DIR / "plot4_scatter_ratios.png"
plt.savefig(out4, dpi=150)
plt.close()
print(f"Saved: {out4}")
print("\nAll 4 plots saved.")
