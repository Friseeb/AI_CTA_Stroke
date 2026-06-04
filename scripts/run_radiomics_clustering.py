#!/usr/bin/env python3
"""
Cluster analysis + PCA visualizations for radiomics_features.csv.

Typical usage:
  python -u scripts/run_radiomics_clustering.py \
    --input /path/to/daylightbids/derivatives/nudf_la/radiomics_features.csv \
    --output-dir /path/to/daylightbids/derivatives/nudf_la/cluster \
    --segment laa \
    --algo kmeans \
    --k 3 \
    --pca-variance 0.95
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Iterable

# Force non-interactive backend for safe plotting in scripts
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd


META_COLS = {"case_id", "segment", "cta_path", "mask_path"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster analysis for radiomics features")
    p.add_argument("--input", required=True, help="radiomics_features.csv")
    p.add_argument("--output-dir", required=True, help="Output directory for CSV/plots")
    p.add_argument("--segment", default="all", help="Filter by segment: all|aorta|la|laa")
    p.add_argument(
        "--segments",
        default=None,
        help="Comma-separated segments for batch mode (e.g., laa,la,aorta).",
    )
    p.add_argument("--batch-segments", action="store_true", help="Run all segments in batch")
    p.add_argument(
        "--combine-segments",
        action="store_true",
        help="Combine multiple segments per case into one feature vector",
    )
    p.add_argument(
        "--combine-list",
        default="laa,la,aorta",
        help="Segments to combine (comma-separated). Default: laa,la,aorta",
    )
    p.add_argument(
        "--allow-missing-segments",
        action="store_true",
        help="Allow missing segments when combining (outer join + impute)",
    )
    p.add_argument("--algo", default="kmeans", choices=["kmeans", "agglomerative", "gmm"])
    p.add_argument("--k", type=int, default=3, help="Number of clusters")
    p.add_argument("--k-range", default="2,10", help="Range for k-scan (kmeans only), e.g. 2,10")
    p.add_argument("--auto-k", action="store_true", help="Auto-select k by max silhouette (kmeans only)")
    p.add_argument(
        "--k-criterion",
        default="silhouette",
        choices=["silhouette", "min_inertia"],
        help="Auto-k criterion for kmeans: silhouette (max) or min_inertia (min within range)",
    )
    p.add_argument("--pca-variance", type=float, default=0.95, help="PCA variance to keep (0-1)")
    p.add_argument("--no-pca", action="store_true", help="Disable PCA for clustering")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--top-features", type=int, default=30, help="Top features for ANOVA report")
    p.add_argument("--umap", action="store_true", help="Generate UMAP 2D visualization")
    p.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--umap-3d", action="store_true", help="Generate UMAP 3D visualization")
    p.add_argument("--pca-3d", action="store_true", help="Generate PCA 3D visualization")
    p.add_argument("--tsne", action="store_true", help="Generate t-SNE 2D visualization")
    p.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--tsne-iter", type=int, default=1000, help="t-SNE iterations")
    p.add_argument("--tsne-pca-dims", type=int, default=50, help="t-SNE PCA pre-reduction dims")
    p.add_argument("--heatmap", action="store_true", help="Cluster centroid heatmap")
    p.add_argument("--heatmap-top", type=int, default=25, help="Top features for heatmap")
    p.add_argument(
        "--primary-top",
        type=int,
        default=20,
        help="Top features (by |centroid z|) used to define primary radiomics per cluster",
    )
    p.add_argument("--parallel", action="store_true", help="Enable parallel processing where possible")
    p.add_argument("--jobs", type=int, default=0, help="Parallel jobs (0=auto, -1=all)")
    p.add_argument(
        "--combined-and-batch",
        action="store_true",
        help="Run combined (multi-segment) + per-segment batch in one call",
    )
    return p.parse_args()


def _split_range(value: str) -> Tuple[int, int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--k-range must be like 2,10")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 2 or hi < lo:
        raise ValueError("Invalid k-range")
    return lo, hi


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    feat_df = df.drop(columns=[c for c in df.columns if c in META_COLS], errors="ignore")
    # Coerce to numeric and drop all-NaN columns
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.dropna(axis=1, how="all")
    # Drop constant columns
    nunique = feat_df.nunique(dropna=True)
    feat_df = feat_df.loc[:, nunique > 1]
    return feat_df


def _prepare_matrix(feat_df: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(feat_df.values)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs


def _apply_pca(Xs: np.ndarray, variance: float, random_state: int):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance, random_state=random_state)
    Xp = pca.fit_transform(Xs)
    return pca, Xp


def _cluster(X: np.ndarray, algo: str, k: int, random_state: int):
    if algo == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        return model, labels
    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        return model, labels
    if algo == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=k, random_state=random_state)
        labels = model.fit_predict(X)
        return model, labels
    raise ValueError(f"Unknown algo: {algo}")


def _scan_kmeans(X: np.ndarray, k_range: Tuple[int, int], random_state: int, n_jobs: int | None) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    def _eval_k(k: int):
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        return {"k": k, "inertia": model.inertia_, "silhouette": sil}

    if n_jobs is not None and n_jobs != 1:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(delayed(_eval_k)(k) for k in range(k_range[0], k_range[1] + 1))
    else:
        rows = [_eval_k(k) for k in range(k_range[0], k_range[1] + 1)]
    return pd.DataFrame(rows)


def _plot_elbow_silhouette(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["k"], df["inertia"], marker="o", color="#1f77b4")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (KMeans)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["silhouette"], marker="o", color="#ff7f0e")
    ax2.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cluster_sizes(labels: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    uniq, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(u) for u in uniq], counts, color="#2ca02c")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster sizes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_pca_scatter(Xs: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, out_path: Path) -> None:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca2 = PCA(n_components=2, random_state=0)
    coords = pca2.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D) colored by cluster")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if "segment" in meta.columns:
        # By segment
        fig, ax = plt.subplots(figsize=(6, 5))
        for seg in sorted(meta["segment"].unique()):
            idx = meta["segment"].values == seg
            ax.scatter(coords[idx, 0], coords[idx, 1], s=16, alpha=0.7, label=str(seg))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA (2D) colored by segment")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path.with_name(out_path.stem + "_by_segment.png"), dpi=200)
        plt.close(fig)


def _plot_pca_3d(Xs: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pca3 = PCA(n_components=3, random_state=0)
    coords = pca3.fit_transform(Xs)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap="tab10", s=14, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA (3D) colored by cluster")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_umap(
    Xs: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    out_path: Path,
    n_neighbors: int,
    min_dist: float,
) -> None:
    import os
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    try:
        import umap  # type: ignore
    except Exception:
        print("UMAP not available. Install with: pip install umap-learn")
        return

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    coords = reducer.fit_transform(Xs)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP (2D) colored by cluster")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if "segment" in meta.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        for seg in sorted(meta["segment"].unique()):
            idx = meta["segment"].values == seg
            ax.scatter(coords[idx, 0], coords[idx, 1], s=16, alpha=0.7, label=str(seg))
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("UMAP (2D) colored by segment")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path.with_name(out_path.stem + "_by_segment.png"), dpi=200)
        plt.close(fig)


def _plot_umap_3d(
    Xs: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_neighbors: int,
    min_dist: float,
) -> None:
    import os
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    try:
        import umap  # type: ignore
    except Exception:
        print("UMAP not available. Install with: pip install umap-learn")
        return

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=0)
    coords = reducer.fit_transform(Xs)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap="tab10", s=14, alpha=0.8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_title("UMAP (3D) colored by cluster")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_tsne(
    Xs: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    perplexity: float,
    n_iter: int,
    pca_dims: int,
) -> None:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    X_in = Xs
    if Xs.shape[1] > pca_dims:
        X_in = PCA(n_components=pca_dims, random_state=0).fit_transform(Xs)

    # scikit-learn >=1.2 uses max_iter instead of n_iter
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=0,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=0,
        )
    coords = tsne.fit_transform(X_in)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=16, alpha=0.8)
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    ax.set_title("t-SNE (2D) colored by cluster")
    ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _anova_top_features(Xs: np.ndarray, labels: np.ndarray, feature_names: List[str], top_n: int) -> pd.DataFrame:
    from sklearn.feature_selection import f_classif

    f_vals, p_vals = f_classif(Xs, labels)
    df = pd.DataFrame({"feature": feature_names, "f": f_vals, "p": p_vals})
    df = df.sort_values("f", ascending=False).head(top_n)
    return df


def _parse_feature_name(name: str) -> dict:
    # Handles combined features like "laa__original_glcm_Contrast"
    segment = None
    base = name
    if "__" in name:
        segment, base = name.split("__", 1)

    parts = base.split("_", 2)
    image_type = parts[0] if len(parts) >= 1 else ""
    feature_class = parts[1] if len(parts) >= 2 else ""
    feature_name = parts[2] if len(parts) >= 3 else ""
    return {
        "segment": segment,
        "image_type": image_type,
        "feature_class": feature_class,
        "feature_name": feature_name,
    }


def _feature_meaning(feature_class: str, feature_name: str) -> str:
    # Generic percentile parsing
    if feature_name.endswith("Percentile"):
        return "Intensity value below which given percent of voxel values fall"
    if feature_name.endswith("Percentile") is False and "Percentile" in feature_name:
        return "Percentile of voxel intensities"

    firstorder = {
        "Mean": "Average intensity",
        "Median": "Median intensity",
        "Minimum": "Minimum intensity",
        "Maximum": "Maximum intensity",
        "Range": "Maximum minus minimum intensity",
        "Variance": "Intensity variance",
        "Skewness": "Asymmetry of intensity distribution",
        "Kurtosis": "Tailedness of intensity distribution",
        "Energy": "Sum of squared intensities",
        "TotalEnergy": "Energy scaled by voxel volume",
        "Entropy": "Randomness of intensity distribution",
        "Uniformity": "Sum of squared probabilities (uniformity)",
        "RootMeanSquared": "Root mean square of intensities",
        "MeanAbsoluteDeviation": "Mean absolute deviation from mean",
        "RobustMeanAbsoluteDeviation": "Mean absolute deviation from mean within robust range",
        "InterquartileRange": "Q3 minus Q1 of intensities",
        "10Percentile": "10th percentile of intensities",
        "90Percentile": "90th percentile of intensities",
        "25Percentile": "25th percentile of intensities",
        "75Percentile": "75th percentile of intensities",
    }
    shape = {
        "Volume": "Physical volume of the mask",
        "MeshVolume": "Volume from surface mesh",
        "VoxelVolume": "Volume from voxel count",
        "SurfaceArea": "Surface area of the mask",
        "SurfaceVolumeRatio": "Surface area divided by volume",
        "Sphericity": "Similarity to a sphere (1=sphere)",
        "Compactness1": "Compactness based on surface/volume",
        "Compactness2": "Compactness based on surface/volume",
        "Maximum3DDiameter": "Maximum 3D diameter",
        "Maximum2DDiameterSlice": "Maximum 2D diameter within slices",
        "Maximum2DDiameterRow": "Maximum 2D diameter in row direction",
        "Maximum2DDiameterColumn": "Maximum 2D diameter in column direction",
        "MajorAxisLength": "Length of major principal axis",
        "MinorAxisLength": "Length of minor principal axis",
        "LeastAxisLength": "Length of least principal axis",
        "Elongation": "Minor/Major axis length ratio",
        "Flatness": "Least/Major axis length ratio",
    }
    glcm = {
        "Autocorrelation": "Linear dependence of gray levels",
        "ClusterProminence": "Asymmetry and peakedness of clusters",
        "ClusterShade": "Asymmetry of clusters",
        "ClusterTendency": "Grouping of similar gray levels",
        "Contrast": "Local intensity variation",
        "Correlation": "Linear dependency of gray levels",
        "DifferenceAverage": "Average difference of gray levels",
        "DifferenceEntropy": "Entropy of difference distribution",
        "DifferenceVariance": "Variance of gray-level differences",
        "Id": "Inverse difference",
        "Idm": "Inverse difference moment",
        "Idmn": "Normalized inverse difference moment",
        "Idn": "Normalized inverse difference",
        "Imc1": "Information measure of correlation 1",
        "Imc2": "Information measure of correlation 2",
        "InverseVariance": "Inverse variance",
        "JointAverage": "Mean of joint probability",
        "JointEntropy": "Entropy of joint probability",
        "JointEnergy": "Energy of joint probability",
        "MaximumProbability": "Highest joint probability",
        "SumAverage": "Average of sum distribution",
        "SumEntropy": "Entropy of sum distribution",
        "SumSquares": "Variance of sum distribution",
        "MCC": "Maximal correlation coefficient",
    }
    glrlm = {
        "ShortRunEmphasis": "Emphasis on short runs",
        "LongRunEmphasis": "Emphasis on long runs",
        "GrayLevelNonUniformity": "Gray-level non-uniformity",
        "GrayLevelNonUniformityNormalized": "Normalized gray-level non-uniformity",
        "RunLengthNonUniformity": "Run-length non-uniformity",
        "RunLengthNonUniformityNormalized": "Normalized run-length non-uniformity",
        "RunPercentage": "Homogeneity of runs",
        "RunEntropy": "Entropy of run-length distribution",
        "RunVariance": "Variance of run lengths",
        "GrayLevelVariance": "Variance of gray levels in runs",
        "LowGrayLevelRunEmphasis": "Emphasis on low gray-level runs",
        "HighGrayLevelRunEmphasis": "Emphasis on high gray-level runs",
        "ShortRunLowGrayLevelEmphasis": "Short runs with low gray levels",
        "ShortRunHighGrayLevelEmphasis": "Short runs with high gray levels",
        "LongRunLowGrayLevelEmphasis": "Long runs with low gray levels",
        "LongRunHighGrayLevelEmphasis": "Long runs with high gray levels",
    }
    glszm = {
        "SmallAreaEmphasis": "Emphasis on small zones",
        "LargeAreaEmphasis": "Emphasis on large zones",
        "GrayLevelNonUniformity": "Gray-level non-uniformity",
        "GrayLevelNonUniformityNormalized": "Normalized gray-level non-uniformity",
        "SizeZoneNonUniformity": "Zone size non-uniformity",
        "SizeZoneNonUniformityNormalized": "Normalized zone size non-uniformity",
        "ZonePercentage": "Homogeneity of zones",
        "ZoneEntropy": "Entropy of zone-size distribution",
        "ZoneVariance": "Variance of zone sizes",
        "GrayLevelVariance": "Variance of gray levels in zones",
        "LowGrayLevelZoneEmphasis": "Emphasis on low gray-level zones",
        "HighGrayLevelZoneEmphasis": "Emphasis on high gray-level zones",
        "SmallAreaLowGrayLevelEmphasis": "Small zones with low gray levels",
        "SmallAreaHighGrayLevelEmphasis": "Small zones with high gray levels",
        "LargeAreaLowGrayLevelEmphasis": "Large zones with low gray levels",
        "LargeAreaHighGrayLevelEmphasis": "Large zones with high gray levels",
    }
    gldm = {
        "SmallDependenceEmphasis": "Emphasis on small dependencies",
        "LargeDependenceEmphasis": "Emphasis on large dependencies",
        "GrayLevelNonUniformity": "Gray-level non-uniformity",
        "DependenceNonUniformity": "Dependence non-uniformity",
        "DependenceNonUniformityNormalized": "Normalized dependence non-uniformity",
        "DependenceEntropy": "Entropy of dependence distribution",
        "DependenceVariance": "Variance of dependence sizes",
        "GrayLevelVariance": "Variance of gray levels",
        "LowGrayLevelEmphasis": "Emphasis on low gray levels",
        "HighGrayLevelEmphasis": "Emphasis on high gray levels",
        "SmallDependenceLowGrayLevelEmphasis": "Small dependencies with low gray levels",
        "SmallDependenceHighGrayLevelEmphasis": "Small dependencies with high gray levels",
        "LargeDependenceLowGrayLevelEmphasis": "Large dependencies with low gray levels",
        "LargeDependenceHighGrayLevelEmphasis": "Large dependencies with high gray levels",
    }
    ngtdm = {
        "Coarseness": "Local uniformity (coarse textures)",
        "Contrast": "Intensity contrast between a voxel and neighbors",
        "Busyness": "Rate of gray-level change",
        "Complexity": "Intensity and spatial variation",
        "Strength": "Primitives and contrast of texture",
    }

    lookup = {
        "firstorder": firstorder,
        "shape": shape,
        "glcm": glcm,
        "glrlm": glrlm,
        "glszm": glszm,
        "gldm": gldm,
        "ngtdm": ngtdm,
    }

    if feature_class in lookup and feature_name in lookup[feature_class]:
        return lookup[feature_class][feature_name]
    return "See PyRadiomics documentation for definition"


def _combine_segments(
    df: pd.DataFrame, segments: List[str], allow_missing: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "segment" not in df.columns:
        raise RuntimeError("Segment column missing; cannot combine.")

    meta_cols = [c for c in df.columns if c in META_COLS]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    frames = []
    for seg in segments:
        sub = df[df["segment"] == seg].copy()
        if sub.empty:
            print(f"⚠ No rows for segment: {seg}")
        sub = sub.drop_duplicates(subset=["case_id"], keep="first")
        seg_feats = sub[["case_id"] + feat_cols].copy()
        seg_feats = seg_feats.set_index("case_id")
        seg_feats = seg_feats.rename(columns={c: f"{seg}__{c}" for c in feat_cols})
        frames.append(seg_feats)

    join = "outer" if allow_missing else "inner"
    combined = pd.concat(frames, axis=1, join=join)
    combined = combined.reset_index()

    # Minimal meta (case_id only)
    meta = combined[["case_id"]].copy()
    feat_df = combined.drop(columns=["case_id"])
    return meta, feat_df


def _resolve_jobs(jobs: int) -> int | None:
    if jobs == 0:
        return None
    return jobs


def _run_one_segment(df: pd.DataFrame, segment_label: str, args: argparse.Namespace, out_dir: Path) -> None:
    if args.combine_segments:
        segments = _parse_segments_arg(args.combine_list)
        meta, feat_df = _combine_segments(df, segments, args.allow_missing_segments)
    else:
        if segment_label != "all" and "segment" in df.columns:
            df = df[df["segment"] == segment_label].copy()

        if df.empty:
            print(f"⚠ No rows for segment: {segment_label}")
            return

        meta = df[[c for c in df.columns if c in META_COLS]].copy()
        feat_df = _select_features(df)

    feature_names = list(feat_df.columns)
    Xs = _prepare_matrix(feat_df)

    # PCA for clustering
    if args.no_pca:
        X_cluster = Xs
    else:
        pca, X_cluster = _apply_pca(Xs, args.pca_variance, args.random_state)
        ev = pd.DataFrame({"explained_variance_ratio": pca.explained_variance_ratio_})
        ev.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    # Auto-k (kmeans)
    chosen_k = args.k
    if args.algo == "kmeans":
        k_lo, k_hi = _split_range(args.k_range)
        kscan = _scan_kmeans(X_cluster, (k_lo, k_hi), args.random_state, _resolve_jobs(args.jobs) if args.parallel else None)
        kscan.to_csv(out_dir / "kmeans_k_scan.csv", index=False)
        _plot_elbow_silhouette(kscan, out_dir / "kmeans_elbow_silhouette.png")
        if args.auto_k:
            if args.k_criterion == "min_inertia":
                chosen_k = int(kscan.sort_values("inertia", ascending=True).iloc[0]["k"])
            else:
                chosen_k = int(kscan.sort_values("silhouette", ascending=False).iloc[0]["k"])
            print(f"Auto-k ({args.k_criterion}) selected k={chosen_k} for segment {segment_label}")
    elif args.auto_k:
        print("⚠ --auto-k ignored for non-kmeans algorithms.")

    model, labels = _cluster(X_cluster, args.algo, chosen_k, args.random_state)

    # Save assignments
    out_assign = meta.copy()
    out_assign["cluster"] = labels
    out_assign.to_csv(out_dir / "cluster_assignments.csv", index=False)

    # Centroids + representative subjects
    uniq = np.unique(labels)
    centroids = []
    reps = []
    for c in uniq:
        idx = labels == c
        centroid = Xs[idx].mean(axis=0)
        centroids.append(centroid)
        # representative = closest to centroid in clustering space
        dist = np.linalg.norm(X_cluster[idx] - X_cluster[idx].mean(axis=0), axis=1)
        rep_idx = np.where(idx)[0][np.argmin(dist)]
        reps.append(
            {
                "cluster": int(c),
                "case_id": str(meta.iloc[rep_idx]["case_id"]),
                "distance": float(dist.min()),
            }
        )

    centroids_df = pd.DataFrame(centroids, columns=feature_names)
    centroids_df.insert(0, "cluster", uniq.astype(int))
    centroids_df.to_csv(out_dir / "cluster_centroids_z.csv", index=False)
    pd.DataFrame(reps).to_csv(out_dir / "cluster_representatives.csv", index=False)

    # Primary radiomics summary per cluster (most common feature class/image type)
    primary_rows = []
    top_n = max(1, args.primary_top)
    for _, row in centroids_df.iterrows():
        c = int(row["cluster"])
        vals = row.drop(labels=["cluster"]).astype(float)
        # Top features by absolute centroid magnitude
        top_feats = vals.abs().sort_values(ascending=False).head(top_n).index.tolist()
        parsed = [_parse_feature_name(f) for f in top_feats]

        def _mode(items):
            items = [i for i in items if i]
            if not items:
                return ""
            return max(set(items), key=items.count)

        primary_rows.append(
            {
                "cluster": c,
                "primary_segment": _mode([p.get("segment") for p in parsed]),
                "primary_image_type": _mode([p.get("image_type") for p in parsed]),
                "primary_feature_class": _mode([p.get("feature_class") for p in parsed]),
                "top_features": ";".join(top_feats),
            }
        )

    pd.DataFrame(primary_rows).to_csv(out_dir / "cluster_primary_radiomics.csv", index=False)

    # Plots (optionally parallel)
    plot_tasks = []
    plot_tasks.append(lambda: _plot_cluster_sizes(labels, out_dir / "cluster_sizes.png"))
    plot_tasks.append(lambda: _plot_pca_scatter(Xs, labels, meta, out_dir / "pca_scatter.png"))
    if args.umap:
        plot_tasks.append(
            lambda: _plot_umap(
                Xs,
                labels,
                meta,
                out_dir / "umap_scatter.png",
                args.umap_n_neighbors,
                args.umap_min_dist,
            )
        )
    if args.pca_3d:
        plot_tasks.append(lambda: _plot_pca_3d(Xs, labels, out_dir / "pca_scatter_3d.png"))
    if args.umap_3d:
        plot_tasks.append(
            lambda: _plot_umap_3d(
                Xs,
                labels,
                out_dir / "umap_scatter_3d.png",
                args.umap_n_neighbors,
                args.umap_min_dist,
            )
        )
    if args.tsne:
        plot_tasks.append(
            lambda: _plot_tsne(
                Xs,
                labels,
                out_dir / "tsne_scatter.png",
                args.tsne_perplexity,
                args.tsne_iter,
                args.tsne_pca_dims,
            )
        )

    # Plot sequentially to avoid threading issues with numba/umap on macOS
    for fn in plot_tasks:
        fn()

    # Top features by ANOVA
    top_feats = _anova_top_features(Xs, labels, feature_names, args.top_features)
    # Add feature meanings
    meta_rows = []
    for feat in top_feats["feature"].tolist():
        info = _parse_feature_name(feat)
        info["meaning"] = _feature_meaning(info.get("feature_class", ""), info.get("feature_name", ""))
        meta_rows.append(info)
    meta_df = pd.DataFrame(meta_rows)
    top_feats = pd.concat([top_feats.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    top_feats.to_csv(out_dir / "top_features_anova.csv", index=False)

    # Heatmap of cluster centroids (top features)
    if args.heatmap:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except Exception:
            print("Seaborn not available. Install with: pip install seaborn")
        else:
            top_n = min(args.heatmap_top, len(top_feats))
            top_names = top_feats["feature"].head(top_n).tolist()
            heat = centroids_df.set_index("cluster")[top_names].T
            fig_w = max(6, heat.shape[1] * 1.2)
            fig_h = max(8, heat.shape[0] * 0.35)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(heat, cmap="vlag", center=0, ax=ax, cbar_kws={"shrink": 0.6})
            ax.set_title("Cluster centroid heatmap (z-score features)")
            ax.tick_params(axis="x", labelrotation=0, labelsize=10)
            ax.tick_params(axis="y", labelrotation=0, labelsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "cluster_centroid_heatmap.png", dpi=200)
            plt.close(fig)

    print(f"Saved outputs in {out_dir}")


def _parse_segments_arg(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.combined_and_batch:
        # Run combined first
        combined_out = base_out / "combined"
        combined_out.mkdir(parents=True, exist_ok=True)
        args.combine_segments = True
        _run_one_segment(df, "combined", args, combined_out)
        # Then run per-segment batch
        args.combine_segments = False
        args.batch_segments = True

    if args.combine_segments and args.batch_segments:
        raise RuntimeError("Use either --combine-segments or --batch-segments, not both (unless --combined-and-batch).")

    if args.batch_segments:
        segs = _parse_segments_arg(args.segments)
        if not segs:
            if "segment" not in df.columns:
                raise RuntimeError("No segment column found; cannot batch by segment.")
            segs = sorted(df["segment"].dropna().unique().tolist())
        for seg in segs:
            seg_out = base_out / f"segment_{seg}"
            seg_out.mkdir(parents=True, exist_ok=True)
            _run_one_segment(df, seg, args, seg_out)
        return 0

    # Single segment
    _run_one_segment(df, args.segment, args, base_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
