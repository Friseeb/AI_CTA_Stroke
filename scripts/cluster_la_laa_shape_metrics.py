#!/usr/bin/env python3
"""Cluster LA/LAA shape metrics from batch outputs.

Example:
  /opt/anaconda3/envs/laa-shape/bin/python scripts/cluster_la_laa_shape_metrics.py \
    --input-csv /path/to/daylightbids/derivatives/shape_meshes_repro/la_laa_metrics_batch_dedup.csv \
    --out-dir /path/to/daylightbids/derivatives/shape_meshes_repro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = "./outputs/shape_meshes_repro/la_laa_metrics_batch_dedup.csv"
DEFAULT_OUT_DIR = "./outputs/shape_meshes_repro"

# Focus on morphology/relational geometry; avoid raw coordinate frame fields.
BASE_FEATURES = [
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
        description="Cluster LA/LAA shape metrics using KMeans with silhouette-based k selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-csv", default=DEFAULT_INPUT, help="Input metrics CSV.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--k-min", type=int, default=2, help="Minimum k to evaluate.")
    p.add_argument("--k-max", type=int, default=8, help="Maximum k to evaluate.")
    p.add_argument("--k", type=int, default=0, help="Fixed k (0 = auto-select by silhouette).")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    p.add_argument("--n-init", type=int, default=50, help="KMeans n_init.")
    return p.parse_args()


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["laa_to_la_volume_ratio"] = out["laa_volume_mm3"] / out["la_volume_mm3"].replace(0, np.nan)
    out["laa_to_la_area_ratio"] = out["laa_surface_area_mm2"] / out["la_surface_area_mm2"].replace(0, np.nan)
    out["la_compactness_idx"] = (out["la_volume_mm3"] ** (2.0 / 3.0)) / out["la_surface_area_mm2"].replace(0, np.nan)
    out["laa_compactness_idx"] = (out["laa_volume_mm3"] ** (2.0 / 3.0)) / out["laa_surface_area_mm2"].replace(0, np.nan)
    feat_cols = BASE_FEATURES + [
        "laa_to_la_volume_ratio",
        "laa_to_la_area_ratio",
        "la_compactness_idx",
        "laa_compactness_idx",
    ]
    return out, feat_cols


def _prep_matrix(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, Pipeline]:
    mat = df[feat_cols].astype(float).to_numpy()
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x = prep.fit_transform(mat)
    return x, prep


def evaluate_k(x: np.ndarray, k_min: int, k_max: int, seed: int, n_init: int) -> pd.DataFrame:
    rows: list[dict] = []
    for k in range(k_min, k_max + 1):
        if k >= len(x):
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(x)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if "qc_exception" in df.columns:
        df = df[~_to_bool(df["qc_exception"])].copy()
    if len(df) < 10:
        raise RuntimeError(f"Too few rows after filtering: {len(df)}")

    df, feat_cols = build_feature_table(df)
    x, prep = _prep_matrix(df, feat_cols)

    k_scores = evaluate_k(
        x=x,
        k_min=max(2, args.k_min),
        k_max=max(args.k_min, args.k_max),
        seed=args.random_seed,
        n_init=args.n_init,
    )
    if k_scores.empty:
        raise RuntimeError("No valid k values to evaluate.")

    if args.k and args.k > 1:
        k_best = args.k
    else:
        k_best = int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"])

    model = KMeans(n_clusters=k_best, random_state=args.random_seed, n_init=args.n_init)
    labels = model.fit_predict(x)
    pca = PCA(n_components=2, random_state=args.random_seed)
    emb = pca.fit_transform(x)

    result = df.copy()
    result["cluster"] = labels
    result["pca1"] = emb[:, 0]
    result["pca2"] = emb[:, 1]

    cluster_counts = result.groupby("cluster", as_index=False).size().rename(columns={"size": "n_cases"})
    cluster_means = result.groupby("cluster")[feat_cols].mean(numeric_only=True).reset_index()
    cluster_summary = cluster_counts.merge(cluster_means, on="cluster", how="left")

    assign_csv = out_dir / "la_laa_shape_cluster_assignments.csv"
    summary_csv = out_dir / "la_laa_shape_cluster_summary.csv"
    scores_csv = out_dir / "la_laa_shape_cluster_k_scores.csv"
    features_csv = out_dir / "la_laa_shape_cluster_features_used.csv"

    result[
        [
            "case_id",
            "subject_id",
            "cluster",
            "pca1",
            "pca2",
            "min_distance_mm",
            "ostium_planarity",
            "laa_axis_length_mm",
            "bend_LAaxis_vs_LAAaxis_deg",
            "laa_surface_area_mm2",
            "laa_volume_mm3",
            "la_surface_area_mm2",
            "la_volume_mm3",
            "laa_to_la_volume_ratio",
            "laa_to_la_area_ratio",
        ]
    ].to_csv(assign_csv, index=False)
    cluster_summary.to_csv(summary_csv, index=False)
    k_scores.sort_values("k").to_csv(scores_csv, index=False)
    pd.DataFrame({"feature": feat_cols}).to_csv(features_csv, index=False)

    best_row = k_scores[k_scores["k"] == k_best].iloc[0]
    print(f"Input rows (post-filter): {len(result)}")
    print(f"Chosen k: {k_best}")
    print(
        "k metrics: "
        f"silhouette={best_row['silhouette']:.4f}, "
        f"calinski_harabasz={best_row['calinski_harabasz']:.2f}, "
        f"davies_bouldin={best_row['davies_bouldin']:.4f}"
    )
    print("Cluster sizes:")
    for _, r in cluster_counts.sort_values("cluster").iterrows():
        print(f"  cluster {int(r['cluster'])}: {int(r['n_cases'])}")
    print(f"Wrote: {assign_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {scores_csv}")
    print(f"Wrote: {features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
