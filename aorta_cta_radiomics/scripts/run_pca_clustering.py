#!/usr/bin/env python
"""Run PCA and unsupervised clustering on aggregated aorta CTA wide features."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = AORTA_ROOT / "outputs" / "aorta_batch_run" / "features" / "modeling_wide_features.csv"
DOMAIN_DEFINITIONS = {
    "calcium": (
        "calcium",
        "calcification",
        "agatston",
    ),
    "wall": (
        "wall_thickness",
        "aortic_wall",
        "aorta_wall_band",
        "aorta_wall_dynamic",
        "aorta_wall_from_fat",
        "wall_from_fat",
        "lumen_protrusions",
        "protrusion",
        "ulcer",
        "lumen",
    ),
    "peri_fat": (
        "periaortic_fat",
        "fat_omics",
        "shell_0_2mm",
        "shell_2_5mm",
        "shell_5_10mm",
        "adipose",
    ),
}
INTERPRETABLE_FEATURES = [
    {
        "domain": "calcium",
        "label": "Dynamic calcium volume",
        "column": "aorta__calcium_omics__aortic_volume_mm3__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "ml",
        "scale": 0.001,
    },
    {
        "domain": "calcium",
        "label": "Dynamic Agatston-like burden",
        "column": "aorta__calcium_omics__aortic_agatston_modified__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "arb.",
        "scale": 1.0,
    },
    {
        "domain": "calcium",
        "label": "Dynamic calcium lesions",
        "column": "aorta__calcium_omics__num_lesions__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "count",
        "scale": 1.0,
    },
    {
        "domain": "calcium",
        "label": "Dynamic calcium per cm",
        "column": "aorta__calcium_omics__calcium_per_cm__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "mm3/cm",
        "scale": 1.0,
    },
    {
        "domain": "calcium",
        "label": "Dynamic calcium mass per cm",
        "column": "aorta__calcium_omics__calcium_mass_proxy_per_cm__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "HU*mm3/cm",
        "scale": 1.0,
    },
    {
        "domain": "calcium",
        "label": "Max circumferential calcium arc",
        "column": "aorta__calcium_omics__circumferential_arc_max__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "degrees",
        "scale": 1.0,
    },
    {
        "domain": "calcium",
        "label": "Anterior calcium fraction proxy",
        "column": "aorta__calcium_omics__anterior_proxy_fraction__thr_dynamic_lumen_referenced_seed500HU",
        "unit": "fraction",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Mean wall thickness",
        "column": "aortic_wall__wall_thickness__wall_mean_mm",
        "unit": "mm",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "95th percentile wall thickness",
        "column": "aortic_wall__wall_thickness__wall_p95_mm",
        "unit": "mm",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Max wall thickness",
        "column": "aortic_wall__wall_thickness__wall_max_mm",
        "unit": "mm",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Wall fraction above 4 mm",
        "column": "aortic_wall__wall_thickness_threshold__wall_thickness_gt4mm_wall_fraction__thr_> 4 mm",
        "unit": "fraction",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Wall volume above 4 mm",
        "column": "aortic_wall__wall_thickness_threshold__wall_thickness_gt4mm_volume_mm3__thr_> 4 mm",
        "unit": "ml",
        "scale": 0.001,
    },
    {
        "domain": "wall",
        "label": "Protrusion candidates",
        "column": "aorta_lumen__lumen_protrusions__candidate_count",
        "unit": "count",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Protrusions at least 4 mm",
        "column": "aorta_lumen__lumen_protrusions__candidate_count_depth_ge_4mm",
        "unit": "count",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Max inward protrusion depth",
        "column": "aorta_lumen__lumen_protrusions__max_protrusion_depth_mm",
        "unit": "mm",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Max outward ulcer-like depth",
        "column": "aorta_lumen__lumen_protrusions__max_outward_ulcer_like_depth_mm",
        "unit": "mm",
        "scale": 1.0,
    },
    {
        "domain": "wall",
        "label": "Max lumen compromise",
        "column": "aorta_lumen__lumen_protrusions__max_percent_lumen_compromise",
        "unit": "percent",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Periaortic fat volume",
        "column": "periaortic_fat__fat_omics__periaortic_fat_volume_mm3",
        "unit": "ml",
        "scale": 0.001,
    },
    {
        "domain": "peri_fat",
        "label": "Periaortic fat volume per cm",
        "column": "periaortic_fat__fat_omics__periaortic_fat_volume_per_cm",
        "unit": "mm3/cm",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Inner 0-2 mm fat volume",
        "column": "periaortic_fat__fat_omics__periaortic_fat_volume_0_2mm",
        "unit": "ml",
        "scale": 0.001,
    },
    {
        "domain": "peri_fat",
        "label": "Outer 2-5 mm fat volume",
        "column": "periaortic_fat__fat_omics__periaortic_fat_volume_2_5mm",
        "unit": "ml",
        "scale": 0.001,
    },
    {
        "domain": "peri_fat",
        "label": "Periaortic fat mean HU",
        "column": "periaortic_fat__fat_omics__periaortic_mean_HU",
        "unit": "HU",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Inner 0-2 mm fat mean HU",
        "column": "periaortic_fat__fat_omics__periaortic_mean_HU_0_2mm",
        "unit": "HU",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Outer 2-5 mm fat mean HU",
        "column": "periaortic_fat__fat_omics__periaortic_mean_HU_2_5mm",
        "unit": "HU",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Higher-HU fat fraction (-50 to -30)",
        "column": "periaortic_fat__fat_omics__periaortic_high_HU_fraction_m50_m30",
        "unit": "fraction",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Higher-HU fat fraction (-70 to -30)",
        "column": "periaortic_fat__fat_omics__periaortic_high_HU_fraction_m70_m30",
        "unit": "fraction",
        "scale": 1.0,
    },
    {
        "domain": "peri_fat",
        "label": "Periaortic fat radial gradient",
        "column": "periaortic_fat__fat_omics__periaortic_radial_gradient",
        "unit": "HU/mm",
        "scale": 1.0,
    },
]


def main() -> None:
    args = build_parser().parse_args()
    features_path = Path(args.features).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found: {features_path}")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else features_path.parents[1] / "pca_clustering"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(features_path)
    prepared = prepare_feature_matrix(
        raw,
        case_id_column=args.case_id_column,
        max_missing=args.max_missing,
        min_variance=args.min_variance,
        drop_regex=args.drop_regex,
    )
    if prepared.matrix.shape[1] < 2:
        raise ValueError("Need at least two usable numeric features after filtering.")
    if prepared.matrix.shape[0] < 3:
        raise ValueError("Need at least three cases for PCA and clustering.")

    domain_result = None
    if args.analysis_mode == "domain-balanced":
        domain_result = run_domain_pca(
            prepared=prepared,
            domain_components=args.domain_components,
            min_domain_features=args.min_domain_features,
        )
        domain_result.scores.to_csv(outdir / "domain_pca_scores.csv", index=False)
        domain_result.variance.to_csv(outdir / "domain_pca_explained_variance.csv", index=False)
        domain_result.loadings.to_csv(outdir / "domain_pca_loadings.csv", index=False)
        domain_result.top_loadings.to_csv(outdir / "domain_pca_top_loadings.csv", index=False)
        domain_result.summary.to_csv(outdir / "domain_feature_summary.csv", index=False)
        pca_matrix = domain_result.scaled_scores
        pca_feature_names = domain_result.score_feature_names
        analysis_feature_count = domain_result.original_feature_count
        cluster_source_matrix = pca_matrix
        cluster_basis = "scaled_domain_pcs"
    else:
        pca_matrix = prepared.matrix
        pca_feature_names = prepared.feature_names
        analysis_feature_count = len(prepared.feature_names)
        cluster_source_matrix = None
        cluster_basis = "integrated_pcs"

    n_components = min(args.n_components, prepared.matrix.shape[0] - 1, pca_matrix.shape[1])
    pca_result = run_pca(pca_matrix, prepared.case_ids, pca_feature_names, n_components)
    if args.cluster_components:
        cluster_components = min(args.cluster_components, n_components)
        cluster_matrix = pca_result.scores.iloc[:, 1 : cluster_components + 1].to_numpy()
        cluster_basis = "integrated_pcs"
    else:
        cluster_components = pca_matrix.shape[1] if cluster_source_matrix is not None else n_components
        cluster_matrix = (
            cluster_source_matrix
            if cluster_source_matrix is not None
            else pca_result.scores.iloc[:, 1 : n_components + 1].to_numpy()
        )
    cluster_result = run_kmeans_grid(
        cluster_matrix,
        case_ids=prepared.case_ids,
        k_values=parse_k_values(args.k_values),
        random_state=args.random_state,
    )

    scores = pca_result.scores.copy()
    if domain_result is not None:
        scores = scores.merge(domain_result.scores, on="case_id", how="left")
    for k, labels in cluster_result.labels_by_k.items():
        scores[f"cluster_k{k}"] = labels
    if cluster_result.best_k is not None:
        scores["cluster_best"] = cluster_result.labels_by_k[cluster_result.best_k]
    scores.to_csv(outdir / "pca_scores_clusters.csv", index=False)
    pca_result.variance.to_csv(outdir / "pca_explained_variance.csv", index=False)
    pca_result.loadings.to_csv(outdir / "pca_loadings.csv", index=False)
    top_loadings = top_component_loadings(pca_result.loadings, args.top_loadings)
    top_loadings.to_csv(outdir / "pca_top_loadings.csv", index=False)
    prepared.feature_report.to_csv(outdir / "feature_filter_report.csv", index=False)
    cluster_result.selection.to_csv(outdir / "cluster_model_selection.csv", index=False)

    if cluster_result.best_k is not None:
        write_cluster_profiles(
            outdir=outdir,
            case_ids=prepared.case_ids,
            feature_names=prepared.feature_names,
            scaled_matrix=prepared.matrix,
            original_imputed=prepared.original_imputed,
            scores=scores,
            best_cluster_column="cluster_best",
            top_n=args.profile_top_features,
        )
        write_interpretability_outputs(
            outdir=outdir,
            raw=raw,
            scores=scores,
            cluster_matrix=cluster_matrix,
            cluster_column="cluster_best",
            case_id_column=args.case_id_column,
            top_features_per_domain=args.interpretability_top_features,
            representative_cases=args.representative_cases,
        )

    write_plots(
        outdir=outdir,
        scores=scores,
        variance=pca_result.variance,
        top_loadings=top_loadings,
        cluster_profiles_path=outdir / "cluster_profile_top_features.csv",
        best_k=cluster_result.best_k,
    )

    output_names = [
        "pca_clustering_explorer.html",
        "pca_scores_clusters.csv",
        "pca_explained_variance.csv",
        "pca_loadings.csv",
        "pca_top_loadings.csv",
        "feature_filter_report.csv",
        "cluster_model_selection.csv",
        "cluster_summary.csv",
        "cluster_profile_top_features.csv",
        "interpretable_feature_dictionary.csv",
        "interpretable_case_features.csv",
        "interpretable_case_features_long.csv",
        "cluster_interpretable_profiles.csv",
        "cluster_interpretation_summary.csv",
        "cluster_representative_cases.csv",
        "domain_state_case_features.csv",
        "domain_state_summary.csv",
        "domain_state_by_cluster.csv",
        "domain_feature_summary.csv",
        "domain_pca_scores.csv",
        "domain_pca_explained_variance.csv",
        "domain_pca_loadings.csv",
        "domain_pca_top_loadings.csv",
        "figures/pca_explained_variance.png",
        "figures/pca_clusters_best.png",
        "figures/pca_top_loadings.png",
        "figures/cluster_profile_heatmap.png",
    ]
    features_excluded = int(
        (
            (prepared.feature_report["status"] == "excluded")
            & (prepared.feature_report["reason"] != "case_id")
        ).sum()
    )
    summary = {
        "features_path": str(features_path),
        "outdir": str(outdir),
        "analysis_mode": args.analysis_mode,
        "domains": list(DOMAIN_DEFINITIONS),
        "cases": int(prepared.matrix.shape[0]),
        "input_columns": int(raw.shape[1]),
        "usable_numeric_features": int(prepared.matrix.shape[1]),
        "features_used": int(analysis_feature_count),
        "features_excluded": features_excluded,
        "pca_components": int(n_components),
        "cluster_components": int(cluster_components),
        "cluster_basis": cluster_basis,
        "k_values": list(cluster_result.labels_by_k.keys()),
        "best_k": cluster_result.best_k,
        "best_silhouette": cluster_result.best_silhouette,
        "outputs": output_names,
    }
    (outdir / "pca_clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_explorer_html(outdir, summary)

    print(f"Cases: {prepared.matrix.shape[0]}")
    print(f"Analysis mode: {args.analysis_mode}")
    print(f"Features used: {analysis_feature_count} / {raw.shape[1] - 1}")
    print(f"PCA components: {n_components}")
    print(f"Cluster basis: {cluster_basis} ({cluster_components} variables)")
    if cluster_result.best_k is not None:
        print(f"Best k: {cluster_result.best_k} (silhouette={cluster_result.best_silhouette:.3f})")
    print(f"Explorer: {outdir / 'pca_clustering_explorer.html'}")
    print(f"Wrote PCA/clustering outputs to: {outdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        type=Path,
        help="Wide feature CSV. Defaults to the completed aorta_batch_run modeling table.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=Path,
        help="Output directory. Defaults to <batch_outdir>/pca_clustering.",
    )
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument(
        "--max-missing",
        type=float,
        default=0.25,
        help="Exclude features with a missing fraction above this value before imputation.",
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=1e-12,
        help="Exclude numeric features with variance at or below this threshold.",
    )
    parser.add_argument(
        "--drop-regex",
        default=r"(software_version|segmentation_method|status)$",
        help="Regex for feature names to exclude before numeric filtering. Use '' to disable.",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["domain-balanced", "global"],
        default="domain-balanced",
        help="domain-balanced computes calcium/wall/peri-fat PCs first; global uses all features directly.",
    )
    parser.add_argument(
        "--domain-components",
        type=int,
        default=3,
        help="Number of within-domain PCs retained for each domain in --analysis-mode domain-balanced.",
    )
    parser.add_argument(
        "--min-domain-features",
        type=int,
        default=2,
        help="Minimum usable features required for each domain.",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument(
        "--cluster-components",
        type=int,
        default=0,
        help=(
            "Number of leading integrated PCs used for KMeans. "
            "Use 0 for all domain PCs or all computed PCs."
        ),
    )
    parser.add_argument("--k-values", default="2,3,4,5,6", help="Comma-separated KMeans k values.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-loadings", type=int, default=25)
    parser.add_argument("--profile-top-features", type=int, default=40)
    parser.add_argument("--interpretability-top-features", type=int, default=3)
    parser.add_argument("--representative-cases", type=int, default=8)
    return parser


class PreparedMatrix:
    def __init__(
        self,
        case_ids: list[str],
        feature_names: list[str],
        matrix: np.ndarray,
        original_imputed: pd.DataFrame,
        feature_report: pd.DataFrame,
    ) -> None:
        self.case_ids = case_ids
        self.feature_names = feature_names
        self.matrix = matrix
        self.original_imputed = original_imputed
        self.feature_report = feature_report


class PcaResult:
    def __init__(self, scores: pd.DataFrame, variance: pd.DataFrame, loadings: pd.DataFrame) -> None:
        self.scores = scores
        self.variance = variance
        self.loadings = loadings


class ClusterResult:
    def __init__(
        self,
        labels_by_k: dict[int, np.ndarray],
        selection: pd.DataFrame,
        best_k: int | None,
        best_silhouette: float | None,
    ) -> None:
        self.labels_by_k = labels_by_k
        self.selection = selection
        self.best_k = best_k
        self.best_silhouette = best_silhouette


class DomainPcaResult:
    def __init__(
        self,
        scores: pd.DataFrame,
        scaled_scores: np.ndarray,
        score_feature_names: list[str],
        variance: pd.DataFrame,
        loadings: pd.DataFrame,
        top_loadings: pd.DataFrame,
        summary: pd.DataFrame,
        original_feature_count: int,
    ) -> None:
        self.scores = scores
        self.scaled_scores = scaled_scores
        self.score_feature_names = score_feature_names
        self.variance = variance
        self.loadings = loadings
        self.top_loadings = top_loadings
        self.summary = summary
        self.original_feature_count = original_feature_count


def prepare_feature_matrix(
    frame: pd.DataFrame,
    case_id_column: str,
    max_missing: float,
    min_variance: float,
    drop_regex: str,
) -> PreparedMatrix:
    if case_id_column not in frame.columns:
        raise ValueError(f"Missing case id column: {case_id_column}")
    if not 0 <= max_missing <= 1:
        raise ValueError("--max-missing must be between 0 and 1.")

    drop_pattern = re.compile(drop_regex) if drop_regex else None
    case_ids = frame[case_id_column].astype(str).tolist()
    numeric_columns: dict[str, pd.Series] = {}
    report_rows: list[dict[str, object]] = []

    for column in frame.columns:
        if column == case_id_column:
            report_rows.append(_feature_report_row(column, "excluded", "case_id", frame[column]))
            continue
        if drop_pattern and drop_pattern.search(column):
            report_rows.append(_feature_report_row(column, "excluded", "drop_regex", frame[column]))
            continue

        numeric = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_missing = numeric.dropna()
        missing_fraction = float(numeric.isna().mean())
        variance = float(non_missing.var(ddof=0)) if len(non_missing) else math.nan
        if non_missing.empty:
            status, reason = "excluded", "non_numeric_or_all_missing"
        elif missing_fraction > max_missing:
            status, reason = "excluded", "too_missing"
        elif not math.isfinite(variance) or variance <= min_variance:
            status, reason = "excluded", "low_variance"
        else:
            status, reason = "used", ""
            numeric_columns[column] = numeric
        report_rows.append(
            {
                "feature": column,
                "status": status,
                "reason": reason,
                "missing_fraction": missing_fraction,
                "non_missing_count": int(non_missing.shape[0]),
                "variance": variance,
                "unique_values": int(non_missing.nunique()),
            }
        )

    numeric_frame = pd.DataFrame(numeric_columns)
    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(numeric_frame)
    original_imputed = pd.DataFrame(imputed_values, columns=numeric_frame.columns)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed_values)
    return PreparedMatrix(
        case_ids=case_ids,
        feature_names=list(numeric_frame.columns),
        matrix=scaled,
        original_imputed=original_imputed,
        feature_report=pd.DataFrame(report_rows),
    )


def _feature_report_row(feature: str, status: str, reason: str, series: pd.Series) -> dict[str, object]:
    return {
        "feature": feature,
        "status": status,
        "reason": reason,
        "missing_fraction": float(series.isna().mean()),
        "non_missing_count": int(series.notna().sum()),
        "variance": "",
        "unique_values": int(series.nunique(dropna=True)),
    }


def run_pca(
    matrix: np.ndarray,
    case_ids: list[str],
    feature_names: list[str],
    n_components: int,
) -> PcaResult:
    pca = PCA(n_components=n_components, random_state=0)
    scores_matrix = pca.fit_transform(matrix)
    pc_names = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(scores_matrix, columns=pc_names)
    scores.insert(0, "case_id", case_ids)

    variance = pd.DataFrame(
        {
            "component": pc_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings = pd.DataFrame(pca.components_.T, columns=pc_names)
    loadings.insert(0, "feature", feature_names)
    return PcaResult(scores=scores, variance=variance, loadings=loadings)


def run_domain_pca(
    prepared: PreparedMatrix,
    domain_components: int,
    min_domain_features: int,
) -> DomainPcaResult:
    if domain_components < 1:
        raise ValueError("--domain-components must be >= 1.")
    if min_domain_features < 1:
        raise ValueError("--min-domain-features must be >= 1.")

    feature_to_index = {feature: index for index, feature in enumerate(prepared.feature_names)}
    domain_score_frames: list[pd.DataFrame] = [pd.DataFrame({"case_id": prepared.case_ids})]
    variance_frames: list[pd.DataFrame] = []
    loading_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    score_feature_names: list[str] = []

    for domain in DOMAIN_DEFINITIONS:
        features = [feature for feature in prepared.feature_names if assign_feature_domain(feature) == domain]
        if len(features) < min_domain_features:
            raise ValueError(
                f"Domain '{domain}' has only {len(features)} usable feature(s); "
                f"need at least {min_domain_features}."
            )

        indices = [feature_to_index[feature] for feature in features]
        domain_matrix = prepared.matrix[:, indices]
        n_components = min(domain_components, domain_matrix.shape[0] - 1, domain_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        score_matrix = pca.fit_transform(domain_matrix)
        domain_pc_names = [f"{domain}_PC{i}" for i in range(1, n_components + 1)]
        score_feature_names.extend(domain_pc_names)

        scores = pd.DataFrame(score_matrix, columns=domain_pc_names)
        scores.insert(0, "case_id", prepared.case_ids)
        domain_score_frames.append(scores)

        variance = pd.DataFrame(
            {
                "domain": domain,
                "component": domain_pc_names,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        variance_frames.append(variance)

        component_loadings = pd.DataFrame(pca.components_.T, columns=domain_pc_names)
        component_loadings.insert(0, "feature", features)
        long_loadings = component_loadings.melt(
            id_vars="feature",
            var_name="component",
            value_name="loading",
        )
        long_loadings.insert(0, "domain", domain)
        long_loadings["abs_loading"] = long_loadings["loading"].abs()
        loading_frames.append(long_loadings)

        summary_rows.append(
            {
                "domain": domain,
                "usable_features": len(features),
                "pca_components": n_components,
                "cumulative_explained_variance_ratio": float(
                    variance["cumulative_explained_variance_ratio"].iloc[-1]
                ),
            }
        )

    scores = domain_score_frames[0]
    for frame in domain_score_frames[1:]:
        scores = scores.merge(frame, on="case_id", how="left")
    scaled_scores = StandardScaler().fit_transform(scores[score_feature_names].to_numpy())
    variance = pd.concat(variance_frames, ignore_index=True)
    loadings = pd.concat(loading_frames, ignore_index=True)
    return DomainPcaResult(
        scores=scores,
        scaled_scores=scaled_scores,
        score_feature_names=score_feature_names,
        variance=variance,
        loadings=loadings,
        top_loadings=top_domain_loadings(loadings, top_n=25),
        summary=pd.DataFrame(summary_rows),
        original_feature_count=sum(int(row["usable_features"]) for row in summary_rows),
    )


def assign_feature_domain(feature: str) -> str | None:
    name = feature.lower()
    for domain, tokens in DOMAIN_DEFINITIONS.items():
        if any(token in name for token in tokens):
            return domain
    return None


def top_domain_loadings(loadings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (_domain, _component), frame in loadings.groupby(["domain", "component"], sort=False):
        frames.append(frame.sort_values("abs_loading", ascending=False).head(top_n))
    if not frames:
        return pd.DataFrame(columns=list(loadings.columns))
    return pd.concat(frames, ignore_index=True)


def parse_k_values(k_values: str) -> list[int]:
    values = sorted({int(value.strip()) for value in k_values.split(",") if value.strip()})
    if any(value < 2 for value in values):
        raise ValueError("All k values must be >= 2.")
    return values


def run_kmeans_grid(
    cluster_matrix: np.ndarray,
    case_ids: list[str],
    k_values: list[int],
    random_state: int,
) -> ClusterResult:
    labels_by_k: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    best_k: int | None = None
    best_silhouette = -math.inf
    n_cases = len(case_ids)

    for k in k_values:
        if k >= n_cases:
            rows.append({"k": k, "status": "skipped", "reason": "k_not_less_than_case_count"})
            continue
        model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = model.fit_predict(cluster_matrix)
        labels_by_k[k] = labels
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(cluster_matrix, labels))
        else:
            silhouette = math.nan
        rows.append(
            {
                "k": k,
                "status": "fit",
                "reason": "",
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
                "cluster_sizes": ";".join(str(int((labels == label).sum())) for label in sorted(set(labels))),
            }
        )
        if math.isfinite(silhouette) and silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    return ClusterResult(
        labels_by_k=labels_by_k,
        selection=pd.DataFrame(rows),
        best_k=best_k,
        best_silhouette=float(best_silhouette) if best_k is not None else None,
    )


def top_component_loadings(loadings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for component in [col for col in loadings.columns if col.startswith("PC")]:
        part = loadings[["feature", component]].copy()
        part["component"] = component
        part["loading"] = part[component]
        part["abs_loading"] = part["loading"].abs()
        part = part.sort_values("abs_loading", ascending=False).head(top_n)
        frames.append(part[["component", "feature", "loading", "abs_loading"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_cluster_profiles(
    outdir: Path,
    case_ids: list[str],
    feature_names: list[str],
    scaled_matrix: np.ndarray,
    original_imputed: pd.DataFrame,
    scores: pd.DataFrame,
    best_cluster_column: str,
    top_n: int,
) -> None:
    labels = scores[best_cluster_column].to_numpy()
    pc_columns = [column for column in scores.columns if re.fullmatch(r"PC\d+", column)][:5]
    cluster_scores = scores[["case_id", best_cluster_column, *pc_columns]].copy()
    aggregations = {"n": ("case_id", "count")}
    aggregations.update({f"{column}_mean": (column, "mean") for column in pc_columns})
    summary = (
        cluster_scores.groupby(best_cluster_column)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={best_cluster_column: "cluster"})
    )
    summary.to_csv(outdir / "cluster_summary.csv", index=False)

    scaled = pd.DataFrame(scaled_matrix, columns=feature_names)
    scaled.insert(0, "cluster", labels)
    profile_z = scaled.groupby("cluster").mean().reset_index()
    profile_z.to_csv(outdir / "cluster_profile_zscores.csv", index=False)

    original = original_imputed.copy()
    original.insert(0, "cluster", labels)
    original_profile = original.groupby("cluster").mean().reset_index()
    original_profile.to_csv(outdir / "cluster_profile_feature_means.csv", index=False)

    z_only = profile_z.drop(columns=["cluster"])
    top_features = z_only.abs().max(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    top = profile_z[["cluster", *top_features]].melt(
        id_vars="cluster",
        var_name="feature",
        value_name="cluster_mean_z",
    )
    top["max_abs_cluster_mean_z"] = top["feature"].map(z_only.abs().max(axis=0))
    top = top.sort_values(["max_abs_cluster_mean_z", "feature", "cluster"], ascending=[False, True, True])
    top.to_csv(outdir / "cluster_profile_top_features.csv", index=False)

    assignments = pd.DataFrame({"case_id": case_ids, "cluster": labels})
    assignments.to_csv(outdir / "cluster_assignments_best.csv", index=False)


def write_interpretability_outputs(
    outdir: Path,
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    cluster_matrix: np.ndarray,
    cluster_column: str,
    case_id_column: str,
    top_features_per_domain: int,
    representative_cases: int,
) -> None:
    available_specs = [spec for spec in INTERPRETABLE_FEATURES if str(spec["column"]) in raw.columns]
    dictionary = pd.DataFrame(
        [
            {
                "slug": _slugify(str(spec["label"])),
                "domain": spec["domain"],
                "label": spec["label"],
                "source_column": spec["column"],
                "unit": spec["unit"],
                "scale": spec["scale"],
            }
            for spec in available_specs
        ]
    )
    dictionary.to_csv(outdir / "interpretable_feature_dictionary.csv", index=False)
    if not available_specs or cluster_column not in scores.columns:
        return

    cluster_lookup = scores.set_index("case_id")[cluster_column]
    cluster_values = raw[case_id_column].astype(str).map(cluster_lookup)
    wide = pd.DataFrame({"case_id": raw[case_id_column].astype(str), cluster_column: cluster_values})
    long_rows: list[dict[str, object]] = []
    for spec in available_specs:
        slug = _slugify(str(spec["label"]))
        values = pd.to_numeric(raw[str(spec["column"])], errors="coerce") * float(spec["scale"])
        wide[slug] = values
        for case_id, cluster, value in zip(wide["case_id"], wide[cluster_column], values, strict=True):
            long_rows.append(
                {
                    "case_id": case_id,
                    "cluster": cluster,
                    "domain": spec["domain"],
                    "feature": slug,
                    "label": spec["label"],
                    "value": value,
                    "unit": spec["unit"],
                }
            )
    wide.to_csv(outdir / "interpretable_case_features.csv", index=False)
    pd.DataFrame(long_rows).to_csv(outdir / "interpretable_case_features_long.csv", index=False)

    profile = _cluster_interpretable_profiles(wide, available_specs, cluster_column)
    profile.to_csv(outdir / "cluster_interpretable_profiles.csv", index=False)
    summary = _cluster_interpretation_summary(profile, top_features_per_domain)
    summary.to_csv(outdir / "cluster_interpretation_summary.csv", index=False)
    domain_states, domain_state_summary, domain_state_by_cluster = _domain_state_outputs(
        wide=wide,
        feature_specs=available_specs,
        cluster_column=cluster_column,
    )
    domain_states.to_csv(outdir / "domain_state_case_features.csv", index=False)
    domain_state_summary.to_csv(outdir / "domain_state_summary.csv", index=False)
    domain_state_by_cluster.to_csv(outdir / "domain_state_by_cluster.csv", index=False)
    representatives = _representative_cases(
        case_ids=scores["case_id"].astype(str).tolist(),
        labels=scores[cluster_column].to_numpy(),
        cluster_matrix=cluster_matrix,
        top_n=representative_cases,
    )
    representatives.to_csv(outdir / "cluster_representative_cases.csv", index=False)


def _cluster_interpretable_profiles(
    wide: pd.DataFrame,
    feature_specs: list[dict[str, object]],
    cluster_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in feature_specs:
        slug = _slugify(str(spec["label"]))
        values = pd.to_numeric(wide[slug], errors="coerce")
        cohort_mean = float(values.mean(skipna=True)) if values.notna().any() else math.nan
        cohort_median = float(values.median(skipna=True)) if values.notna().any() else math.nan
        cohort_std = float(values.std(skipna=True, ddof=0)) if values.notna().sum() > 1 else math.nan
        for cluster, group in wide.groupby(cluster_column, dropna=False):
            cluster_values = pd.to_numeric(group[slug], errors="coerce")
            mean = float(cluster_values.mean(skipna=True)) if cluster_values.notna().any() else math.nan
            median = float(cluster_values.median(skipna=True)) if cluster_values.notna().any() else math.nan
            q25 = float(cluster_values.quantile(0.25)) if cluster_values.notna().any() else math.nan
            q75 = float(cluster_values.quantile(0.75)) if cluster_values.notna().any() else math.nan
            delta = mean - cohort_mean if math.isfinite(mean) and math.isfinite(cohort_mean) else math.nan
            delta_z = (
                delta / cohort_std
                if math.isfinite(delta) and cohort_std and cohort_std > 0
                else math.nan
            )
            rows.append(
                {
                    "cluster": cluster,
                    "domain": spec["domain"],
                    "feature": slug,
                    "label": spec["label"],
                    "unit": spec["unit"],
                    "n_cases": int(group.shape[0]),
                    "n_nonmissing": int(cluster_values.notna().sum()),
                    "cluster_mean": mean,
                    "cluster_median": median,
                    "cluster_q25": q25,
                    "cluster_q75": q75,
                    "cohort_mean": cohort_mean,
                    "cohort_median": cohort_median,
                    "delta_vs_cohort_mean": delta,
                    "delta_z": delta_z,
                    "direction": _direction(delta_z),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["abs_delta_z"] = frame["delta_z"].abs()
    return frame.sort_values(["cluster", "domain", "abs_delta_z"], ascending=[True, True, False])


def _cluster_interpretation_summary(
    profile: pd.DataFrame,
    top_features_per_domain: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if profile.empty:
        return pd.DataFrame(rows)
    for cluster, cluster_frame in profile.groupby("cluster", dropna=False):
        row: dict[str, object] = {"cluster": cluster, "n_cases": int(cluster_frame["n_cases"].max())}
        plain_parts: list[str] = []
        for domain in DOMAIN_DEFINITIONS:
            domain_frame = cluster_frame[
                (cluster_frame["domain"] == domain)
                & (cluster_frame["direction"] != "similar")
                & (cluster_frame["delta_z"].notna())
            ].sort_values("abs_delta_z", ascending=False)
            if domain_frame.empty:
                fallback = (
                    cluster_frame[
                        (cluster_frame["domain"] == domain)
                        & (cluster_frame["delta_z"].notna())
                    ]
                    .sort_values("abs_delta_z", ascending=False)
                    .head(1)
                )
                summaries = [
                    f"no prominent shift; largest deviation: {_format_driver(feature_row)}"
                    for _, feature_row in fallback.iterrows()
                ]
            else:
                summaries = [
                    _format_driver(feature_row)
                    for _, feature_row in domain_frame.head(top_features_per_domain).iterrows()
                ]
            row[f"{domain}_drivers"] = "; ".join(summaries)
            plain_parts.extend(f"{domain}: {summary}" for summary in summaries[:2])
        row["plain_language_summary"] = " | ".join(plain_parts)
        rows.append(row)
    return pd.DataFrame(rows)


def _domain_state_outputs(
    wide: pd.DataFrame,
    feature_specs: list[dict[str, object]],
    cluster_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create forced three-domain phenotypes from interpretable feature composites."""
    states = wide[["case_id", cluster_column]].copy()
    domains = list(DOMAIN_DEFINITIONS)
    for domain in domains:
        slugs = [
            _slugify(str(spec["label"]))
            for spec in feature_specs
            if spec["domain"] == domain and _slugify(str(spec["label"])) in wide.columns
        ]
        score_column = f"{domain}_score"
        state_column = f"{domain}_state"
        feature_count_column = f"{domain}_feature_count"
        if not slugs:
            states[score_column] = np.nan
            states[state_column] = "missing"
            states[feature_count_column] = 0
            continue
        values = wide[slugs].apply(pd.to_numeric, errors="coerce")
        z_values = []
        for column in values.columns:
            series = values[column]
            std = series.std(skipna=True, ddof=0)
            if not math.isfinite(float(std)) or float(std) <= 0:
                continue
            z_values.append((series - series.mean(skipna=True)) / float(std))
        if not z_values:
            states[score_column] = np.nan
            states[state_column] = "missing"
            states[feature_count_column] = values.notna().sum(axis=1)
            continue
        z_frame = pd.concat(z_values, axis=1)
        score = z_frame.mean(axis=1, skipna=True)
        states[score_column] = score
        states[state_column] = _tertile_states(score)
        states[feature_count_column] = z_frame.notna().sum(axis=1)

    states["forced_domain_phenotype"] = [
        "__".join(f"{domain}_{row[f'{domain}_state']}" for domain in domains)
        for _, row in states.iterrows()
    ]
    summary = _domain_state_summary(states, cluster_column)
    by_cluster = _domain_state_by_cluster(states, cluster_column)
    return states, summary, by_cluster


def _tertile_states(score: pd.Series) -> pd.Series:
    valid = score.dropna()
    if valid.empty:
        return pd.Series(["missing"] * len(score), index=score.index, dtype="object")
    low_cut = float(valid.quantile(1 / 3))
    high_cut = float(valid.quantile(2 / 3))
    if low_cut == high_cut:
        median = float(valid.median())
        low_cut = median
        high_cut = median

    def classify(value: object) -> str:
        numeric = float(value) if pd.notna(value) else math.nan
        if not math.isfinite(numeric):
            return "missing"
        if numeric <= low_cut:
            return "low"
        if numeric >= high_cut:
            return "high"
        return "mid"

    return score.map(classify)


def _domain_state_summary(states: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if states.empty:
        return pd.DataFrame(rows)
    domains = list(DOMAIN_DEFINITIONS)
    for phenotype, group in states.groupby("forced_domain_phenotype", dropna=False):
        row: dict[str, object] = {
            "forced_domain_phenotype": phenotype,
            "n_cases": int(group.shape[0]),
            "clusters": _cluster_mix(group, cluster_column),
        }
        for domain in domains:
            score = pd.to_numeric(group[f"{domain}_score"], errors="coerce")
            row[f"{domain}_state"] = group[f"{domain}_state"].mode(dropna=False).iloc[0]
            row[f"{domain}_score_mean"] = float(score.mean(skipna=True)) if score.notna().any() else math.nan
            row[f"{domain}_score_median"] = (
                float(score.median(skipna=True)) if score.notna().any() else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_cases", "forced_domain_phenotype"], ascending=[False, True])


def _domain_state_by_cluster(states: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if states.empty or cluster_column not in states.columns:
        return pd.DataFrame(rows)
    domains = list(DOMAIN_DEFINITIONS)
    for cluster, cluster_frame in states.groupby(cluster_column, dropna=False):
        cluster_n = int(cluster_frame.shape[0])
        for domain in domains:
            counts = cluster_frame[f"{domain}_state"].value_counts(dropna=False)
            row: dict[str, object] = {
                "cluster": cluster,
                "domain": domain,
                "n_cases": cluster_n,
                "low": int(counts.get("low", 0)),
                "mid": int(counts.get("mid", 0)),
                "high": int(counts.get("high", 0)),
                "missing": int(counts.get("missing", 0)),
                "top_state": str(counts.index[0]) if not counts.empty else "missing",
                "top_state_fraction": float(counts.iloc[0] / cluster_n) if cluster_n else math.nan,
                "top_phenotypes": _top_phenotypes(cluster_frame),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["cluster", "domain"])


def _cluster_mix(group: pd.DataFrame, cluster_column: str) -> str:
    if cluster_column not in group.columns:
        return ""
    counts = group[cluster_column].value_counts(dropna=False).sort_index()
    return "; ".join(f"{cluster}:{int(count)}" for cluster, count in counts.items())


def _top_phenotypes(group: pd.DataFrame, limit: int = 3) -> str:
    counts = group["forced_domain_phenotype"].value_counts(dropna=False).head(limit)
    return "; ".join(f"{phenotype}:{int(count)}" for phenotype, count in counts.items())


def _representative_cases(
    case_ids: list[str],
    labels: np.ndarray,
    cluster_matrix: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster)
        if len(indices) == 0:
            continue
        center = cluster_matrix[indices].mean(axis=0)
        distances = np.linalg.norm(cluster_matrix[indices] - center, axis=1)
        order = np.argsort(distances)[:top_n]
        for rank, local_index in enumerate(order, start=1):
            index = int(indices[int(local_index)])
            rows.append(
                {
                    "cluster": cluster,
                    "rank": rank,
                    "case_id": case_ids[index],
                    "distance_to_cluster_center": float(distances[int(local_index)]),
                }
            )
    return pd.DataFrame(rows)


def _format_driver(row: pd.Series) -> str:
    if row["direction"] == "high":
        direction = "higher"
    elif row["direction"] == "low":
        direction = "lower"
    else:
        direction = "similar"
    value = _format_value(row["cluster_mean"], row["unit"])
    cohort = _format_value(row["cohort_mean"], row["unit"])
    return f"{direction} {row['label']} ({value} vs cohort {cohort})"


def _format_value(value: object, unit: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    digits = 3 if abs(numeric) < 1 else 2
    return f"{numeric:.{digits}f} {unit}".strip()


def _direction(delta_z: float) -> str:
    if not math.isfinite(delta_z) or abs(delta_z) < 0.35:
        return "similar"
    return "high" if delta_z > 0 else "low"


def _slugify(label: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", label.lower())).strip("_")


def write_plots(
    outdir: Path,
    scores: pd.DataFrame,
    variance: pd.DataFrame,
    top_loadings: pd.DataFrame,
    cluster_profiles_path: Path,
    best_k: int | None,
) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    _plot_variance(variance, figdir / "pca_explained_variance.png")
    if best_k is not None:
        _plot_clusters(scores, best_k, figdir / "pca_clusters_best.png")
    _plot_top_loadings(top_loadings, figdir / "pca_top_loadings.png")
    if cluster_profiles_path.exists():
        _plot_cluster_heatmap(cluster_profiles_path, figdir / "cluster_profile_heatmap.png")


def write_explorer_html(outdir: Path, summary: dict[str, object]) -> Path:
    """Write a self-contained static HTML explorer for PCA and clustering outputs."""
    payload = {
        "summary": summary,
        "scores": _records_from_csv(outdir / "pca_scores_clusters.csv"),
        "variance": _records_from_csv(outdir / "pca_explained_variance.csv"),
        "modelSelection": _records_from_csv(outdir / "cluster_model_selection.csv"),
        "clusterSummary": _records_from_csv(outdir / "cluster_summary.csv"),
        "clusterProfiles": _records_from_csv(outdir / "cluster_profile_top_features.csv"),
        "topLoadings": _records_from_csv(outdir / "pca_top_loadings.csv"),
        "domainSummary": _records_from_csv(outdir / "domain_feature_summary.csv"),
        "domainVariance": _records_from_csv(outdir / "domain_pca_explained_variance.csv"),
        "domainTopLoadings": _records_from_csv(outdir / "domain_pca_top_loadings.csv"),
        "interpretableFeatures": _records_from_csv(outdir / "interpretable_feature_dictionary.csv"),
        "interpretableCases": _records_from_csv(outdir / "interpretable_case_features_long.csv"),
        "interpretableProfiles": _records_from_csv(outdir / "cluster_interpretable_profiles.csv"),
        "interpretationSummary": _records_from_csv(outdir / "cluster_interpretation_summary.csv"),
        "representativeCases": _records_from_csv(outdir / "cluster_representative_cases.csv"),
        "domainStateCases": _records_from_csv(outdir / "domain_state_case_features.csv"),
        "domainStateSummary": _records_from_csv(outdir / "domain_state_summary.csv"),
        "domainStateByCluster": _records_from_csv(outdir / "domain_state_by_cluster.csv"),
        "featureReport": _feature_report_summary(outdir / "feature_filter_report.csv"),
        "recursiveSelection": _recursive_selection_payload(outdir),
        "centroidCases": _centroid_case_payload(outdir),
    }
    html_text = _explorer_html(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    path = outdir / "pca_clustering_explorer.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def _records_from_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    return json.loads(frame.to_json(orient="records"))


def _recursive_selection_payload(outdir: Path) -> dict[str, object]:
    selection_dir = outdir.parent / "domain_balanced_recursive_selection"
    selected_pca_dir = selection_dir / "pca_clustering"
    if not selection_dir.exists():
        return {"available": False}
    return {
        "available": True,
        "summary": _read_json(selection_dir / "selection_summary.json"),
        "selectedPcaSummary": _read_json(selected_pca_dir / "pca_clustering_summary.json"),
        "selectedFeatures": _records_from_csv(selection_dir / "selected_features.csv"),
        "trace": _records_from_csv(selection_dir / "selection_trace.csv"),
        "modelSelection": _records_from_csv(selection_dir / "final_model_selection.csv"),
        "stability": _records_from_csv(selection_dir / "stability_summary.csv"),
        "links": {
            "report": "../domain_balanced_recursive_selection/selection_report.html",
            "explorer": "../domain_balanced_recursive_selection/pca_clustering/pca_clustering_explorer.html",
            "selectedFeatures": "../domain_balanced_recursive_selection/selected_features.csv",
            "trace": "../domain_balanced_recursive_selection/selection_trace.csv",
        },
    }


def _centroid_case_payload(outdir: Path) -> list[dict[str, object]]:
    representatives_path = outdir / "cluster_representative_cases.csv"
    if not representatives_path.exists() or representatives_path.stat().st_size == 0:
        return []
    representatives = pd.read_csv(representatives_path)
    if representatives.empty or "rank" not in representatives.columns:
        return []
    representatives = representatives[representatives["rank"] == 1].copy()
    if representatives.empty:
        return []

    scores = _frame_by_case(outdir / "pca_scores_clusters.csv")
    domain_states = _frame_by_case(outdir / "domain_state_case_features.csv")
    interpretations = _frame_by_cluster(outdir / "cluster_interpretation_summary.csv")
    batch_dir = outdir.parent
    rows: list[dict[str, object]] = []
    for _, row in representatives.sort_values("cluster").iterrows():
        case_id = str(row["case_id"])
        cluster = row["cluster"]
        image_path = batch_dir / "cases" / case_id / "figures" / case_id / f"{case_id}_aorta_qc_overlay.png"
        features_path = batch_dir / "cases" / case_id / "features" / "case_level_features.csv"
        masks_path = batch_dir / "cases" / case_id / "masks" / case_id
        nifti_payload = _centroid_nifti_payload(batch_dir, outdir, case_id)
        score_row = scores.get(case_id, {})
        state_row = domain_states.get(case_id, {})
        interpretation_row = interpretations.get(str(cluster), {})
        row_payload = {
            "cluster": _json_scalar(cluster),
            "case_id": case_id,
            "distance_to_cluster_center": _json_scalar(row["distance_to_cluster_center"]),
            "qc_overlay": _relative_path(image_path, outdir),
            "qc_overlay_exists": image_path.exists(),
            "features_csv": _relative_path(features_path, outdir),
            "features_csv_exists": features_path.exists(),
            "masks_dir": _relative_path(masks_path, outdir),
            "masks_dir_exists": masks_path.exists(),
            "plain_language_summary": interpretation_row.get("plain_language_summary", ""),
            "domain_phenotype": state_row.get("forced_domain_phenotype", ""),
            "calcium_state": state_row.get("calcium_state", ""),
            "wall_state": state_row.get("wall_state", ""),
            "peri_fat_state": state_row.get("peri_fat_state", ""),
            "PC1": score_row.get("PC1", ""),
            "PC2": score_row.get("PC2", ""),
            "PC3": score_row.get("PC3", ""),
            "calcium_PC1": score_row.get("calcium_PC1", ""),
            "wall_PC1": score_row.get("wall_PC1", ""),
            "peri_fat_PC1": score_row.get("peri_fat_PC1", ""),
        }
        row_payload.update(nifti_payload)
        rows.append(row_payload)
    return rows


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _nifti_data_url(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/gzip;base64,{encoded}"


def _nifti_nonzero_center_frac(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    try:
        import nibabel as nib

        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        coords = np.argwhere(data > 0)
        if coords.size == 0:
            return []
        dims = np.asarray(data.shape[:3], dtype=float)
        center = coords.mean(axis=0)
        frac = (center + 0.5) / dims
        return [float(value) for value in frac[:3]]
    except Exception:
        return []


def _nifti_crop_bounds(path: Path | None, margin: int = 24) -> tuple[slice, slice, slice] | None:
    if path is None or not path.exists():
        return None
    try:
        import nibabel as nib

        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        coords = np.argwhere(data > 0)
        if coords.size == 0:
            return None
        lower = np.maximum(coords.min(axis=0)[:3] - margin, 0)
        upper = np.minimum(coords.max(axis=0)[:3] + margin + 1, np.asarray(data.shape[:3]))
        return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))  # type: ignore[return-value]
    except Exception:
        return None


def _write_cropped_nifti(
    source: Path | None,
    target_dir: Path,
    bounds: tuple[slice, slice, slice] | None,
    *,
    require_nonzero: bool = False,
) -> Path | None:
    if source is None or not source.exists() or bounds is None:
        return source if source is not None and source.exists() else None
    try:
        import nibabel as nib

        image = nib.load(str(source))
        if any((item.stop or 0) > image.shape[index] for index, item in enumerate(bounds)):
            return source
        data = np.asanyarray(image.dataobj)
        cropped = np.asarray(data[bounds]).copy()
        if require_nonzero and not np.any(cropped):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        translation = np.eye(4)
        translation[:3, 3] = [item.start or 0 for item in bounds]
        affine = image.affine @ translation
        header = image.header.copy()
        output = nib.Nifti1Image(cropped, affine, header=header)
        output.set_data_dtype(image.get_data_dtype())
        qform_code = int(image.header["qform_code"]) or 1
        sform_code = int(image.header["sform_code"]) or 1
        output.set_qform(affine, code=qform_code)
        output.set_sform(affine, code=sform_code)
        nib.save(output, str(target))
        return target
    except Exception:
        return source


def _nifti_record(
    label: str,
    path: Path | None,
    outdir: Path,
    *,
    colormap: str,
    opacity: float,
) -> dict[str, object]:
    return {
        "label": label,
        "path": _relative_path(path, outdir) if path is not None else "",
        "data_url": _nifti_data_url(path),
        "exists": bool(path is not None and path.exists()),
        "file_name": path.name if path is not None else "",
        "colormap": colormap,
        "opacity": opacity,
    }


def _centroid_nifti_payload(batch_dir: Path, outdir: Path, case_id: str) -> dict[str, object]:
    masks_dir = batch_dir / "cases" / case_id / "masks" / case_id
    vista_dir = batch_dir / "vista_aorta" / case_id
    source_base = _first_existing(
        [
            masks_dir / f"{case_id}_aorta_mask_cleaned.nii.gz",
            vista_dir / f"{case_id}_aorta6.nii.gz",
        ]
    )
    crop_bounds = _nifti_crop_bounds(source_base)
    viewer_dir = outdir / "centroid_nifti" / case_id
    base = _write_cropped_nifti(source_base, viewer_dir, crop_bounds)
    base_label = "Aorta mask cleaned"
    if source_base is not None and source_base.name.endswith("_aorta6.nii.gz"):
        base_label = "VISTA aorta segmentation"

    overlay_specs = [
        (
            "VISTA aorta segmentation",
            vista_dir / f"{case_id}_aorta6.nii.gz",
            "gray",
            0.35,
        ),
        (
            "Aorta wall band",
            masks_dir / f"{case_id}_aorta_wall_band.nii.gz",
            "green",
            0.45,
        ),
        (
            "Dynamic wall calcium",
            masks_dir / f"{case_id}_calcification_aorta_wall_dynamic_seed500HU.nii.gz",
            "hot",
            0.75,
        ),
        (
            "Periaortic fat",
            masks_dir / f"{case_id}_periaortic_fat.nii.gz",
            "blue",
            0.35,
        ),
        (
            "Wall thickness >4 mm labels",
            masks_dir / f"{case_id}_wall_thickness_gt_4mm_TEE_analogue_labels.nii.gz",
            "redyell",
            0.55,
        ),
        (
            "Outward protrusions >=4 mm",
            _first_existing(
                [
                    masks_dir
                    / f"{case_id}_wall_lumen_protrusion_outward_ulcer_like_aorta_surface_core_depth_ge_4mm_labels_3d.nii.gz",
                    masks_dir
                    / f"{case_id}_wall_lumen_protrusion_outward_ulcer_like_aorta_surface_native_depth_ge_4mm_labels_3d.nii.gz",
                ]
            ),
            "blue2magenta",
            0.75,
        ),
    ]
    overlays = []
    for label, path, colormap, opacity in overlay_specs:
        cropped_path = _write_cropped_nifti(path, viewer_dir, crop_bounds, require_nonzero=True)
        if cropped_path is not None and cropped_path.exists() and cropped_path != base:
            overlays.append(_nifti_record(label, cropped_path, outdir, colormap=colormap, opacity=opacity))
    return {
        "nifti_base": _relative_path(base, outdir) if base is not None else "",
        "nifti_base_data_url": _nifti_data_url(base),
        "nifti_base_exists": bool(base is not None and base.exists()),
        "nifti_base_label": base_label if base is not None else "",
        "nifti_base_file": base.name if base is not None else "",
        "nifti_center_frac": _nifti_nonzero_center_frac(base),
        "nifti_overlays": overlays,
        "nifti_note": (
            "The original CTA volume was not copied into this batch output. This viewer embeds "
            "cropped aorta-domain NIfTI masks directly in the HTML and centers on the aorta mask."
        ),
    }


def _frame_by_case(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "case_id" not in frame.columns:
        return {}
    return {
        str(row["case_id"]): {key: _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    }


def _frame_by_cluster(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "cluster" not in frame.columns:
        return {}
    return {
        str(row["cluster"]): {key: _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    }


def _relative_path(path: Path, outdir: Path) -> str:
    return os.path.relpath(path, outdir).replace(os.sep, "/")


def _json_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_report_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path)
    grouped = (
        frame.groupby(["status", "reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status", "reason"])
    )
    grouped["reason"] = grouped["reason"].fillna("")
    return json.loads(grouped.to_json(orient="records"))


def _explorer_html(payload_json: str) -> str:
    css = r"""
:root {
  color-scheme: light;
  --ink: #172120;
  --muted: #64716f;
  --line: #d9e2df;
  --panel: #ffffff;
  --surface: #f7faf9;
  --accent: #0f766e;
  --accent-2: #6d28d9;
  --warn: #b45309;
  --bad: #b91c1c;
  --good: #047857;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #edf3f1;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: #ffffff;
  border-bottom: 1px solid var(--line);
  padding: 16px 22px 12px;
}
.header-row { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; }
h1 { margin: 0; font-size: 24px; line-height: 1.15; letter-spacing: 0; }
.subhead { margin-top: 5px; color: var(--muted); font-size: 12px; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 32px;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  color: var(--ink);
  font-size: 12px;
  font-weight: 700;
}
main { max-width: 1800px; margin: 0 auto; padding: 16px 22px 26px; }
.tabbar {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--line);
}
.tab-button {
  border: 1px solid var(--line);
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  background: #f7faf9;
  color: var(--ink);
  padding: 9px 13px;
  font: inherit;
  font-size: 13px;
  font-weight: 760;
  cursor: pointer;
}
.tab-button.active {
  background: #ffffff;
  color: var(--accent);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.stats {
  display: grid;
  grid-template-columns: repeat(6, minmax(126px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px;
}
.stat .label { color: var(--muted); font-size: 11px; }
.stat .value { margin-top: 3px; font-size: 23px; font-weight: 780; line-height: 1; }
.stat .note {
  margin-top: 5px;
  color: var(--muted);
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 390px;
  gap: 14px;
  align-items: start;
}
.section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 14px;
}
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 11px 13px;
  border-bottom: 1px solid var(--line);
}
.section-title { font-size: 14px; font-weight: 760; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
select,
input[type="search"] {
  min-height: 32px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 6px 8px;
  font: inherit;
  font-size: 12px;
}
.plot-wrap { padding: 12px; }
#scatter {
  position: relative;
  width: 100%;
  height: min(68vh, 660px);
  min-height: 540px;
  background: #fbfcfc;
  border: 1px solid var(--line);
  overflow: hidden;
  touch-action: none;
}
#scatter canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.plot-status {
  position: absolute;
  left: 12px;
  bottom: 10px;
  color: var(--muted);
  font-size: 12px;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 5px 7px;
  pointer-events: none;
}
.plot-tooltip {
  position: fixed;
  z-index: 50;
  display: none;
  max-width: 260px;
  padding: 7px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.96);
  color: var(--ink);
  box-shadow: 0 8px 28px rgba(23, 33, 32, 0.16);
  font-size: 12px;
  pointer-events: none;
}
.axis { stroke: #9aa7a4; stroke-width: 1; }
.grid { stroke: #dce4e1; stroke-width: 1; }
.point { cursor: pointer; opacity: 0.84; stroke: #ffffff; stroke-width: 1; }
.point:hover { opacity: 1; stroke: #172120; stroke-width: 1.3; }
.point.selected { opacity: 1; stroke: #172120; stroke-width: 2.4; }
.axis-label { fill: #34413f; font-size: 12px; }
.tick-label { fill: #667370; font-size: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: var(--muted); font-size: 12px; }
.legend-item { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 11px; height: 11px; border-radius: 99px; display: inline-block; }
.panel { position: sticky; top: 88px; }
.selected-panel { padding: 13px; }
.case-title { margin: 0 0 6px; font-size: 20px; line-height: 1.1; }
.domain-phenotype {
  margin-top: 8px;
  padding: 7px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f0fdfa;
  color: #115e59;
  font-size: 12px;
  font-weight: 760;
  overflow-wrap: anywhere;
}
.case-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
.metric {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: var(--surface);
  min-width: 0;
}
.metric .name { color: var(--muted); font-size: 11px; }
.metric .num { margin-top: 2px; font-size: 16px; font-weight: 760; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th,
td { border-bottom: 1px solid #e7ecea; padding: 7px 8px; text-align: right; white-space: nowrap; }
th:first-child,
td:first-child { text-align: left; }
th { background: #f6f8f7; color: #44504f; font-size: 11px; text-transform: uppercase; letter-spacing: 0; }
.table-wrap { overflow: auto; max-height: 360px; }
tbody tr:hover td { background: #f2faf8; }
tbody tr.selected td { background: #e5f3ef; }
.badge { border-radius: 999px; padding: 2px 7px; font-size: 11px; font-weight: 760; color: #fff; }
.grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.figure-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 12px; }
.figure-tile { border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: var(--surface); }
.figure-tile img {
  width: 100%;
  aspect-ratio: 1.25;
  object-fit: contain;
  background: #fff;
  border: 1px solid var(--line);
}
.figure-tile div { margin-top: 6px; font-size: 12px; color: var(--muted); }
.bars { padding: 10px 13px 13px; display: grid; gap: 6px; }
.bar-row {
  display: grid;
  grid-template-columns: minmax(210px, 1fr) minmax(100px, 220px) 64px;
  gap: 8px;
  align-items: center;
}
.bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
.bar-track { height: 9px; border-radius: 99px; background: #e7ecea; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); }
.bar-fill.neg { background: var(--warn); }
.empty { color: var(--muted); padding: 14px; }
.verdict {
  padding: 12px 13px;
  background: #f0fdfa;
  border-bottom: 1px solid var(--line);
  color: #115e59;
  font-size: 13px;
  font-weight: 720;
}
.verdict.warn {
  background: #fffbeb;
  color: #92400e;
}
.wide-text {
  max-width: 860px;
  white-space: normal;
  text-align: left;
}
.centroid-layout {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 14px;
  padding: 12px;
}
.centroid-list {
  display: grid;
  gap: 8px;
  align-content: start;
}
.centroid-card {
  display: block;
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  color: var(--ink);
  padding: 10px;
  text-align: left;
  cursor: pointer;
  font: inherit;
}
.centroid-card.active {
  border-color: var(--accent);
  background: #e8f5f2;
}
.centroid-case { font-size: 17px; font-weight: 780; margin-top: 3px; }
.centroid-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
.nifti-frame {
  position: relative;
  margin-top: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #05070a;
}
.nifti-viewer {
  display: block;
  width: 100%;
  height: min(72vh, 760px);
  min-height: 560px;
  background: #05070a;
}
.nifti-status {
  position: absolute;
  left: 12px;
  bottom: 10px;
  color: #dbe8e5;
  font-size: 12px;
  background: rgba(5, 7, 10, 0.76);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 6px;
  padding: 5px 7px;
  pointer-events: none;
}
.nifti-status.error { color: #fecaca; }
.viewer-note {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
}
.overlay-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.overlay-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
  color: var(--muted);
  background: var(--surface);
  font-size: 12px;
}
.viewer-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.viewer-details {
  display: grid;
  grid-template-columns: repeat(3, minmax(150px, 1fr));
  gap: 8px;
  margin: 10px 0;
}
@media (max-width: 1180px) {
  .stats { grid-template-columns: repeat(3, minmax(126px, 1fr)); }
  .workspace, .grid-two, .centroid-layout { grid-template-columns: 1fr; }
  .panel { position: static; }
  .figure-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 680px) {
  header, main { padding-left: 13px; padding-right: 13px; }
  .header-row, .section-head { flex-direction: column; align-items: stretch; }
  .stats, .figure-grid { grid-template-columns: 1fr; }
  #scatter { height: 58vh; min-height: 420px; }
}
"""
    script = r"""
const payload = JSON.parse(document.getElementById("payload").textContent);
const scores = payload.scores || [];
const pcs = Object.keys(scores[0] || {}).filter(key => /^PC\d+$/.test(key));
const domainPcs = Object.keys(scores[0] || {}).filter(key => /^[a-z_]+_PC\d+$/.test(key));
const axisVariables = [...pcs, ...domainPcs];
const clusterColumns = Object.keys(scores[0] || {}).filter(key => key.startsWith("cluster_"));
const palette = [
  "#2c7fb8", "#24b9c9", "#7c3aed", "#f59e0b", "#059669",
  "#dc2626", "#64748b", "#d946ef", "#0f766e", "#9333ea"
];
const state = {
  x: pcs[0] || "PC1",
  y: pcs[1] || pcs[0] || "PC1",
  z: pcs[2] || pcs[1] || pcs[0] || "PC1",
  color: clusterColumns.includes("cluster_best") ? "cluster_best" : clusterColumns[0],
  selected: scores[0] ? scores[0].case_id : "",
  centroid: (payload.centroidCases || [])[0] ? (payload.centroidCases || [])[0].case_id : "",
  activeClusters: new Set()
};
let THREE = null;
let OrbitControls = null;
let Niivue = null;
const scatter3d = {
  initialized: false,
  container: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  pointsGroup: null,
  axesGroup: null,
  raycaster: null,
  pointer: null,
  pointMeshes: [],
  hovered: null,
  animationId: null,
  resizeObserver: null
};
const niftiViewer = {
  nv: null,
  canvas: null,
  loadedCase: "",
  token: 0
};

async function loadThree() {
  if (THREE && OrbitControls) return true;
  try {
    const threeModule = await import("three");
    const controlsModule = await import("three/addons/controls/OrbitControls.js");
    THREE = threeModule;
    OrbitControls = controlsModule.OrbitControls;
    return true;
  } catch (error) {
    console.warn("Unable to load Three.js for the 3D PCA plot:", error);
    return false;
  }
}

async function loadNiiVue() {
  if (Niivue) return true;
  try {
    const niivueModule = await import("niivue");
    Niivue = niivueModule.Niivue;
    return Boolean(Niivue);
  } catch (error) {
    console.warn("Unable to load NiiVue for the centroid NIfTI viewer:", error);
    return false;
  }
}

function byId(id) {
  return document.getElementById(id);
}

function num(value) {
  const out = Number(value);
  return Number.isFinite(out) ? out : null;
}

function fmt(value, digits = 2) {
  const out = num(value);
  if (out === null) return "";
  return out.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function shortFmt(value, digits = 2) {
  const out = num(value);
  if (out === null) return "NA";
  return out.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function clusterValues() {
  return [...new Set(scores.map(row => String(row[state.color] ?? "NA")))].sort((a, b) => {
    return Number(a) - Number(b) || a.localeCompare(b);
  });
}

function colorFor(value) {
  const key = String(value ?? "NA");
  if (key === "NA" || key === "null") return "#a3aaa7";
  const numeric = Number(key);
  const idx = Number.isFinite(numeric) ? numeric : Math.abs(hashString(key));
  return palette[Math.abs(idx) % palette.length];
}

function swatchColor(colormap) {
  const lookup = {
    gray: "#94a3b8",
    green: "#16a34a",
    hot: "#f97316",
    blue: "#2563eb",
    redyell: "#eab308",
    blue2magenta: "#d946ef",
    red: "#dc2626"
  };
  return lookup[colormap] || "#64748b";
}

function hashString(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) hash = ((hash << 5) - hash) + text.charCodeAt(i);
  return hash;
}

function populateControls() {
  byId("xSelect").innerHTML = axisVariables.map(pc => `<option value="${pc}">${pc}</option>`).join("");
  byId("ySelect").innerHTML = axisVariables.map(pc => `<option value="${pc}">${pc}</option>`).join("");
  byId("zSelect").innerHTML = axisVariables.map(pc => `<option value="${pc}">${pc}</option>`).join("");
  byId("colorSelect").innerHTML = clusterColumns.map(col => {
    return `<option value="${col}">${col.replace("cluster_", "cluster ")}</option>`;
  }).join("");
  byId("loadingPc").innerHTML = pcs.map(pc => `<option value="${pc}">${pc}</option>`).join("");
  byId("xSelect").value = state.x;
  byId("ySelect").value = state.y;
  byId("zSelect").value = state.z;
  byId("colorSelect").value = state.color;
  byId("loadingPc").value = pcs[0] || "";
  resetClusterFilter();
}

function resetClusterFilter() {
  state.activeClusters = new Set(clusterValues());
  renderClusterFilter();
}

function renderClusterFilter() {
  byId("clusterFilter").innerHTML = clusterValues().map(value => {
    const checked = state.activeClusters.has(value) ? "checked" : "";
    return `<label class="legend-item"><input type="checkbox" data-cluster="${value}" ${checked}>
      <span class="swatch" style="background:${colorFor(value)}"></span>${value}</label>`;
  }).join("");
}

function filteredScores() {
  const q = byId("caseSearch").value.trim().toLowerCase();
  return scores.filter(row => {
    const cluster = String(row[state.color] ?? "NA");
    if (!state.activeClusters.has(cluster)) return false;
    if (!q) return true;
    return String(row.case_id).toLowerCase().includes(q);
  });
}

function renderStats() {
  const s = payload.summary || {};
  byId("stats").innerHTML = [
    ["Cases", s.cases, "rows in PCA scores"],
    ["Domain features", s.features_used, `${s.usable_numeric_features} usable numeric`],
    ["Domain PCs", domainPcs.length, s.cluster_basis || ""],
    ["PCs", s.pca_components, `${shortFmt(firstVariance(), 1)}% PC1 variance`],
    ["Best k", s.best_k, `silhouette ${shortFmt(s.best_silhouette, 3)}`],
    ["Cluster PCs", s.cluster_components, "used for KMeans"],
    ["Shown", filteredScores().length, "after filters"]
  ].map(([label, value, note]) => {
    return `<div class="stat"><div class="label">${label}</div><div class="value">${value ?? ""}</div>
      <div class="note">${note}</div></div>`;
  }).join("");
}

function firstVariance() {
  const first = (payload.variance || [])[0];
  return first ? Number(first.explained_variance_ratio) * 100 : null;
}

function renderRecursiveSelectionTab() {
  const recursive = payload.recursiveSelection || {};
  if (!recursive.available) {
    byId("recursiveVerdict").className = "verdict warn";
    byId("recursiveVerdict").textContent = "Recursive-selection outputs were not found next to this PCA run.";
    return;
  }
  const main = payload.summary || {};
  const selection = recursive.summary || {};
  const selectedPca = recursive.selectedPcaSummary || {};
  const fullSilhouette = num(main.best_silhouette);
  const selectedSilhouette = num(selectedPca.best_silhouette ?? selection.best_silhouette);
  const fullFeatures = num(main.features_used);
  const selectedFeatures = num(selection.selected_features);
  const stability = num(selection.stability_adjusted_rand_mean);
  const separationWinner = selectedSilhouette !== null && fullSilhouette !== null &&
    selectedSilhouette > fullSilhouette ? "recursive selection" : "full PCA";
  const featureReduction = fullFeatures && selectedFeatures ?
    (100 * (1 - selectedFeatures / fullFeatures)) : null;
  byId("recursiveVerdict").className = separationWinner === "full PCA" ? "verdict warn" : "verdict";
  byId("recursiveVerdict").textContent = separationWinner === "full PCA"
    ? `Full PCA separates clusters better by silhouette (${shortFmt(fullSilhouette, 3)} vs ` +
      `${shortFmt(selectedSilhouette, 3)}). Recursive selection is better for interpretability: ` +
      `${selectedFeatures} variables, ${shortFmt(featureReduction, 1)}% fewer features, stability ARI ` +
      `${shortFmt(stability, 3)}.`
    : `Recursive selection separates clusters better by silhouette (${shortFmt(selectedSilhouette, 3)} vs ` +
      `${shortFmt(fullSilhouette, 3)}) while using ${selectedFeatures} variables.`;

  byId("recursiveComparisonRows").innerHTML = [
    ["Cases", main.cases, selection.cases, "same cohort"],
    ["Features used", fullFeatures, selectedFeatures, `${shortFmt(featureReduction, 1)}% fewer selected features`],
    ["Best k", main.best_k, selectedPca.best_k ?? selection.best_k, "selected panel prefers fewer clusters"],
    ["Best silhouette", shortFmt(fullSilhouette, 3), shortFmt(selectedSilhouette, 3), separationWinner],
    ["Cluster basis", main.cluster_basis, selectedPca.cluster_basis || "scaled_domain_pcs", "both domain balanced"],
    ["Candidate mode", "all usable domain features", selection.candidate_mode || "", "recursive tab uses readable features"],
    ["Stability ARI mean", "", shortFmt(stability, 3), "subsample reproducibility for recursive panel"],
  ].map(([metric, fullValue, recursiveValue, interpretation]) => {
    return `<tr><td>${metric}</td><td>${fullValue ?? ""}</td><td>${recursiveValue ?? ""}</td>` +
      `<td class="wide-text">${interpretation || ""}</td></tr>`;
  }).join("");

  byId("recursiveSelectedRows").innerHTML = (recursive.selectedFeatures || []).map(row => {
    return `<tr><td>${row.domain}</td><td>${row.domain_selection_order}</td>` +
      `<td class="wide-text">${row.feature}</td><td>${row.selection_step}</td>` +
      `<td>${row.domain_rank}</td><td>${fmt(row.domain_pca_representativeness, 3)}</td></tr>`;
  }).join("");
  byId("recursiveTraceRows").innerHTML = (recursive.trace || []).map(row => {
    return `<tr><td>${row.selection_step}</td><td>${row.domain}</td>` +
      `<td class="wide-text">${row.feature}</td><td>${row.reason}</td>` +
      `<td>${fmt(row.objective, 3)}</td><td>${fmt(row.silhouette, 3)}</td>` +
      `<td>${row.best_k ?? ""}</td><td>${fmt(row.redundancy_max_abs_corr, 3)}</td></tr>`;
  }).join("");
  byId("recursiveModelRows").innerHTML = (recursive.modelSelection || []).map(row => {
    return `<tr><td>${row.k}</td><td>${row.status}</td><td>${fmt(row.silhouette, 3)}</td>` +
      `<td>${fmt(row.inertia, 0)}</td><td>${row.cluster_sizes || ""}</td></tr>`;
  }).join("");
  byId("recursiveStabilityRows").innerHTML = (recursive.stability || []).map(row => {
    return `<tr><td>${row.repeat}</td><td>${row.sample_n}</td><td>${row.best_k}</td>` +
      `<td>${fmt(row.adjusted_rand_index, 3)}</td><td>${row.subset_cluster_sizes || ""}</td></tr>`;
  }).join("");
  const links = recursive.links || {};
  if (links.report) byId("recursiveReportLink").href = links.report;
  if (links.explorer) byId("recursiveExplorerLink").href = links.explorer;
  if (links.selectedFeatures) byId("recursiveFeaturesLink").href = links.selectedFeatures;
  if (links.trace) byId("recursiveTraceLink").href = links.trace;
}

function renderCentroidViewer() {
  const centroids = payload.centroidCases || [];
  if (!centroids.length) {
    byId("centroidList").innerHTML = `<div class="empty">No centroid patients were found.</div>`;
    byId("centroidViewerBody").innerHTML = `<div class="empty">Run clustering first.</div>`;
    return;
  }
  if (!state.centroid || !centroids.some(row => row.case_id === state.centroid)) {
    state.centroid = centroids[0].case_id;
  }
  const current = centroids.find(row => row.case_id === state.centroid) || centroids[0];
  byId("centroidList").innerHTML = centroids.map(row => {
    const active = row.case_id === current.case_id ? " active" : "";
    return `<button class="centroid-card${active}" type="button" data-centroid-case="${row.case_id}">
      <div><span class="badge" style="background:${colorFor(row.cluster)}">cluster ${row.cluster}</span></div>
      <div class="centroid-case">${row.case_id}</div>
      <div class="centroid-meta">distance ${fmt(row.distance_to_cluster_center, 3)}</div>
      <div class="centroid-meta">${row.domain_phenotype || ""}</div>
    </button>`;
  }).join("");

  const overlays = (current.nifti_overlays || []).filter(item => item.exists && item.path);
  const overlayHtml = overlays.length
    ? `<div class="overlay-list">${overlays.map(item => `
        <a class="overlay-pill" href="${item.path}" title="${item.file_name || item.label}">
          <span class="swatch" style="background:${swatchColor(item.colormap)}"></span>${item.label}
        </a>`).join("")}</div>`
    : `<div class="viewer-note">No overlay NIfTI masks found for this centroid case.</div>`;
  const baseButton = current.nifti_base_exists
    ? `<a class="button" href="${current.nifti_base}">Open Base NIfTI</a>`
    : "";
  byId("centroidViewerBody").innerHTML = `
    <div>
      <h2 class="case-title">Cluster ${current.cluster} centroid: ${current.case_id}</h2>
      <div class="subhead">Closest observed patient to the KMeans cluster center. Viewer loads .nii.gz volumes directly.</div>
      <div class="domain-phenotype">${current.domain_phenotype || "domain phenotype unavailable"}</div>
      <div class="viewer-details">
        <div class="metric"><div class="name">Distance to center</div><div class="num">${fmt(current.distance_to_cluster_center, 3)}</div></div>
        <div class="metric"><div class="name">Calcium</div><div class="num">${current.calcium_state || "NA"}</div></div>
        <div class="metric"><div class="name">Wall</div><div class="num">${current.wall_state || "NA"}</div></div>
        <div class="metric"><div class="name">Peri-fat</div><div class="num">${current.peri_fat_state || "NA"}</div></div>
        <div class="metric"><div class="name">PC1 / PC2 / PC3</div><div class="num">${fmt(current.PC1, 2)} / ${fmt(current.PC2, 2)} / ${fmt(current.PC3, 2)}</div></div>
        <div class="metric"><div class="name">Domain PC1s</div><div class="num">${fmt(current.calcium_PC1, 2)} / ${fmt(current.wall_PC1, 2)} / ${fmt(current.peri_fat_PC1, 2)}</div></div>
      </div>
      <div class="subhead wide-text">${current.plain_language_summary || ""}</div>
      <div class="viewer-actions">
        <button class="button" type="button" id="centroidScatterButton">View in 3D Scatter</button>
        ${baseButton}
        <a class="button" href="${current.features_csv}">Case Features</a>
        <a class="button" href="${current.masks_dir}">Masks Folder</a>
      </div>
      <div class="viewer-note">Base: ${current.nifti_base_label || "NIfTI unavailable"} ${current.nifti_base_file ? `(${current.nifti_base_file})` : ""}</div>
      <div class="viewer-note">${current.nifti_note || ""}</div>
      ${overlayHtml}
    </div>
    <div class="nifti-frame">
      <canvas id="centroidNiftiCanvas" class="nifti-viewer"></canvas>
      <div class="nifti-status" id="centroidNiftiStatus">Loading NIfTI volumes...</div>
    </div>
  `;
  byId("centroidFeatureRows").innerHTML = (payload.interpretableCases || [])
    .filter(row => row.case_id === current.case_id)
    .map(row => `<tr><td>${row.domain}</td><td>${row.label}</td><td>${fmt(row.value, 3)}</td><td>${row.unit || ""}</td></tr>`)
    .join("");
  const button = byId("centroidScatterButton");
  if (button) {
    button.addEventListener("click", () => {
      selectCase(current.case_id);
      activateTab("tabFullPca");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
  loadCentroidNifti(current);
}

function setNiftiStatus(text, isError = false) {
  const status = byId("centroidNiftiStatus");
  if (!status) return;
  status.textContent = text;
  status.className = `nifti-status${isError ? " error" : ""}`;
}

async function loadCentroidNifti(current) {
  const canvas = byId("centroidNiftiCanvas");
  if (!canvas) return;
  const token = ++niftiViewer.token;
  const baseUrl = current.nifti_base_data_url || current.nifti_base;
  if (!current.nifti_base_exists || !baseUrl) {
    setNiftiStatus(`No NIfTI base volume found for ${current.case_id}.`, true);
    return;
  }
  setNiftiStatus(`Loading ${current.case_id} .nii.gz volumes...`);
  if (!Niivue) {
    const loaded = await loadNiiVue();
    if (!loaded) {
      setNiftiStatus("NiiVue could not be loaded. Check network access for the CDN import.", true);
      return;
    }
  }
  if (token !== niftiViewer.token) return;
  try {
    if (!niftiViewer.nv || niftiViewer.canvas !== canvas) {
      niftiViewer.nv = new Niivue({
        backColor: [0, 0, 0, 1],
        dragAndDropEnabled: false,
        show3Dcrosshair: true,
        textHeight: 0.03
      });
      niftiViewer.canvas = canvas;
      niftiViewer.nv.attachToCanvas(canvas);
    }
    const overlays = (current.nifti_overlays || []).filter(item => item.exists && item.path);
    const volumes = [
      {
        url: baseUrl,
        name: current.nifti_base_file || current.nifti_base,
        colormap: "gray",
        opacity: 1
      },
      ...overlays.map(item => ({
        url: item.data_url || item.path,
        name: item.file_name || item.path,
        colormap: item.colormap || "red",
        opacity: num(item.opacity) ?? 0.45
      }))
    ];
    await niftiViewer.nv.loadVolumes(volumes);
    if (typeof niftiViewer.nv.setSliceType === "function" && niftiViewer.nv.sliceTypeMultiplanar !== undefined) {
      niftiViewer.nv.setSliceType(niftiViewer.nv.sliceTypeMultiplanar);
    }
    if (Array.isArray(current.nifti_center_frac) && current.nifti_center_frac.length === 3) {
      niftiViewer.nv.scene.crosshairPos = current.nifti_center_frac.map(value => Number(value));
      if (typeof niftiViewer.nv.drawScene === "function") niftiViewer.nv.drawScene();
    }
    niftiViewer.loadedCase = current.case_id;
    setNiftiStatus(`Loaded ${volumes.length} NIfTI volume${volumes.length === 1 ? "" : "s"} for ${current.case_id}.`);
  } catch (error) {
    console.error("Unable to load centroid NIfTI volumes:", error);
    const detail = error && error.message ? ` ${error.message}` : "";
    setNiftiStatus(`Could not load ${current.case_id} NIfTI volumes.${detail}`, true);
  }
}

function renderScatter() {
  const rows = filteredScores().filter(row => {
    return num(row[state.x]) !== null && num(row[state.y]) !== null && num(row[state.z]) !== null;
  });
  if (!THREE || !OrbitControls) {
    byId("scatter").innerHTML = `<div class="plot-status">3D library unavailable. Tables are still usable.</div>`;
    renderLegend();
    return;
  }
  initScatter3d();
  clearGroup(scatter3d.pointsGroup);
  clearGroup(scatter3d.axesGroup);
  scatter3d.pointMeshes = [];
  if (!rows.length) {
    updatePlotStatus("No cases for the selected axes and filters.");
    renderLegend();
    renderThree();
    return;
  }
  updatePlotStatus(`${rows.length} cases | drag to rotate | scroll to zoom | click a point to select`);
  const xExtent = paddedExtent(rows.map(row => num(row[state.x])).filter(value => value !== null));
  const yExtent = paddedExtent(rows.map(row => num(row[state.y])).filter(value => value !== null));
  const zExtent = paddedExtent(rows.map(row => num(row[state.z])).filter(value => value !== null));
  buildScatterAxes(xExtent, yExtent, zExtent);
  for (const row of rows) {
    const cluster = String(row[state.color] ?? "NA");
    const isSelected = row.case_id === state.selected;
    const geometry = new THREE.SphereGeometry(isSelected ? 0.145 : 0.095, 18, 14);
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(colorFor(cluster)),
      roughness: 0.5,
      metalness: 0.05,
      emissive: isSelected ? new THREE.Color("#172120") : new THREE.Color("#000000"),
      emissiveIntensity: isSelected ? 0.18 : 0
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      scaleToCube(num(row[state.x]), xExtent),
      scaleToCube(num(row[state.y]), yExtent),
      scaleToCube(num(row[state.z]), zExtent)
    );
    mesh.userData = { row, cluster };
    scatter3d.pointsGroup.add(mesh);
    scatter3d.pointMeshes.push(mesh);
  }
  renderLegend();
  renderThree();
}

function paddedExtent(values) {
  if (!values.length) return [-1, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min - 1, max + 1];
  const pad = Math.max((max - min) * 0.08, 1e-9);
  return [min - pad, max + pad];
}

function scaleToCube(value, extent) {
  if (value === null || extent[0] === extent[1]) return 0;
  return ((value - extent[0]) / (extent[1] - extent[0]) - 0.5) * 10;
}

function initScatter3d() {
  if (scatter3d.initialized) {
    resizeScatter();
    return;
  }
  const container = byId("scatter");
  container.innerHTML = "";
  scatter3d.container = container;
  scatter3d.scene = new THREE.Scene();
  scatter3d.scene.background = new THREE.Color("#fbfcfc");
  scatter3d.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  scatter3d.camera.position.set(9, 7, 11);
  scatter3d.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  scatter3d.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(scatter3d.renderer.domElement);
  const status = document.createElement("div");
  status.id = "plotStatus";
  status.className = "plot-status";
  container.appendChild(status);
  scatter3d.pointsGroup = new THREE.Group();
  scatter3d.axesGroup = new THREE.Group();
  scatter3d.scene.add(scatter3d.axesGroup);
  scatter3d.scene.add(scatter3d.pointsGroup);
  scatter3d.scene.add(new THREE.AmbientLight("#ffffff", 0.72));
  const light = new THREE.DirectionalLight("#ffffff", 1.15);
  light.position.set(7, 10, 8);
  scatter3d.scene.add(light);
  scatter3d.controls = new OrbitControls(scatter3d.camera, scatter3d.renderer.domElement);
  scatter3d.controls.enableDamping = true;
  scatter3d.controls.dampingFactor = 0.08;
  scatter3d.controls.target.set(0, 0, 0);
  scatter3d.controls.update();
  scatter3d.raycaster = new THREE.Raycaster();
  scatter3d.pointer = new THREE.Vector2();
  scatter3d.renderer.domElement.addEventListener("pointermove", onScatterPointerMove);
  scatter3d.renderer.domElement.addEventListener("pointerleave", hideScatterTooltip);
  scatter3d.renderer.domElement.addEventListener("click", onScatterClick);
  scatter3d.resizeObserver = new ResizeObserver(resizeScatter);
  scatter3d.resizeObserver.observe(container);
  scatter3d.initialized = true;
  resizeScatter();
  animateScatter();
}

function resizeScatter() {
  if (!scatter3d.renderer || !scatter3d.container) return;
  const rect = scatter3d.container.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  scatter3d.camera.aspect = width / height;
  scatter3d.camera.updateProjectionMatrix();
  scatter3d.renderer.setSize(width, height, false);
  renderThree();
}

function animateScatter() {
  scatter3d.animationId = requestAnimationFrame(animateScatter);
  if (scatter3d.controls) scatter3d.controls.update();
  renderThree();
}

function renderThree() {
  if (!scatter3d.renderer || !scatter3d.scene || !scatter3d.camera) return;
  scatter3d.renderer.render(scatter3d.scene, scatter3d.camera);
}

function resetCamera() {
  if (!scatter3d.camera || !scatter3d.controls) return;
  scatter3d.camera.position.set(9, 7, 11);
  scatter3d.controls.target.set(0, 0, 0);
  scatter3d.controls.update();
  renderThree();
}

function updatePlotStatus(text) {
  const node = byId("plotStatus");
  if (node) node.textContent = text;
}

function clearGroup(group) {
  if (!group) return;
  while (group.children.length) {
    const object = group.children.pop();
    disposeObject(object);
  }
}

function disposeObject(object) {
  if (!object) return;
  if (object.children) object.children.forEach(disposeObject);
  if (object.geometry) object.geometry.dispose();
  if (object.material) {
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (material.map) material.map.dispose();
      material.dispose();
    }
  }
}

function buildScatterAxes(xExtent, yExtent, zExtent) {
  const grid = new THREE.GridHelper(10, 10, "#cfd8d5", "#e7eeeb");
  grid.position.y = -5;
  scatter3d.axesGroup.add(grid);
  addAxisLine(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(5, -5, -5), "#0f766e");
  addAxisLine(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(-5, 5, -5), "#6d28d9");
  addAxisLine(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(-5, -5, 5), "#b45309");
  addAxisRangeLabel(`${shortFmt(xExtent[0], 1)} to ${shortFmt(xExtent[1], 1)}`, new THREE.Vector3(1.8, -5.62, -5), "#0f766e");
  addAxisRangeLabel(`${shortFmt(yExtent[0], 1)} to ${shortFmt(yExtent[1], 1)}`, new THREE.Vector3(-5.65, 1.8, -5), "#6d28d9");
  addAxisRangeLabel(`${shortFmt(zExtent[0], 1)} to ${shortFmt(zExtent[1], 1)}`, new THREE.Vector3(-5.65, -5, 1.8), "#b45309");
}

function addAxisLine(from, to, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
  const material = new THREE.LineBasicMaterial({ color });
  scatter3d.axesGroup.add(new THREE.Line(geometry, material));
  const labelPosition = to.clone();
  if (to.x > from.x) labelPosition.x += 0.72;
  if (to.y > from.y) labelPosition.y += 0.72;
  if (to.z > from.z) labelPosition.z += 0.72;
  const axisName = to.x > from.x ? state.x : to.y > from.y ? state.y : state.z;
  const sprite = makeTextSprite(axisName, color, 0.72);
  sprite.position.copy(labelPosition);
  scatter3d.axesGroup.add(sprite);
}

function addAxisRangeLabel(text, position, color) {
  const sprite = makeTextSprite(text, color, 0.44);
  sprite.position.copy(position);
  scatter3d.axesGroup.add(sprite);
}

function makeTextSprite(text, color, height = 0.56) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  context.font = "600 42px Inter, Arial, sans-serif";
  const metrics = context.measureText(text);
  canvas.width = Math.ceil(metrics.width + 36);
  canvas.height = 72;
  context.font = "600 42px Inter, Arial, sans-serif";
  context.fillStyle = "rgba(255,255,255,0.88)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = color;
  context.fillText(text, 18, 50);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set((canvas.width / canvas.height) * height, height, 1);
  return sprite;
}

function pickScatterObject(event) {
  if (!scatter3d.renderer || !scatter3d.camera || !scatter3d.raycaster) return null;
  const rect = scatter3d.renderer.domElement.getBoundingClientRect();
  scatter3d.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  scatter3d.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  scatter3d.raycaster.setFromCamera(scatter3d.pointer, scatter3d.camera);
  const hits = scatter3d.raycaster.intersectObjects(scatter3d.pointMeshes, false);
  return hits.length ? hits[0].object : null;
}

function onScatterPointerMove(event) {
  const object = pickScatterObject(event);
  scatter3d.renderer.domElement.style.cursor = object ? "pointer" : "grab";
  if (!object) {
    hideScatterTooltip();
    return;
  }
  showScatterTooltip(event, object.userData.row);
}

function onScatterClick(event) {
  const object = pickScatterObject(event);
  if (object && object.userData.row) selectCase(object.userData.row.case_id);
}

function showScatterTooltip(event, row) {
  const node = tooltipNode();
  node.style.display = "block";
  node.style.left = `${event.clientX + 12}px`;
  node.style.top = `${event.clientY + 12}px`;
  node.innerHTML = `<strong>${escapeHtml(String(row.case_id))}</strong><br>` +
    `${escapeHtml(state.color)}: ${escapeHtml(String(row[state.color] ?? "NA"))}<br>` +
    `${escapeHtml(state.x)}: ${fmt(row[state.x], 2)}<br>` +
    `${escapeHtml(state.y)}: ${fmt(row[state.y], 2)}<br>` +
    `${escapeHtml(state.z)}: ${fmt(row[state.z], 2)}`;
}

function hideScatterTooltip() {
  const node = byId("plotTooltip");
  if (node) node.style.display = "none";
}

function tooltipNode() {
  let node = byId("plotTooltip");
  if (!node) {
    node = document.createElement("div");
    node.id = "plotTooltip";
    node.className = "plot-tooltip";
    document.body.appendChild(node);
  }
  return node;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function renderLegend() {
  byId("legend").innerHTML = clusterValues().map(value => {
    const count = scores.filter(row => String(row[state.color] ?? "NA") === value).length;
    return `<span class="legend-item"><span class="swatch" style="background:${colorFor(value)}"></span>
      ${value} (${count})</span>`;
  }).join("");
}

function selectCase(caseId) {
  state.selected = caseId;
  const row = scores.find(item => item.case_id === caseId) || scores[0];
  if (!row) return;
  byId("caseTitle").textContent = row.case_id;
  byId("caseSubhead").textContent = clusterColumns.map(col => {
    return `${col.replace("cluster_", "cluster ")}: ${row[col] ?? "NA"}`;
  }).join(" | ");
  byId("caseMetrics").innerHTML = pcs.slice(0, 10).map(pc => {
    return `<div class="metric"><div class="name">${pc}</div><div class="num">${fmt(row[pc], 2)}</div></div>`;
  }).join("") + domainPcs.slice(0, 6).map(pc => {
    return `<div class="metric"><div class="name">${pc}</div><div class="num">${fmt(row[pc], 2)}</div></div>`;
  }).join("");
  renderSelectedDomainState(row.case_id);
  renderSelectedInterpretability(row.case_id);
  if (byId("tabFullPca").classList.contains("active")) {
    renderScatter();
  }
  renderCaseTable();
}

function renderCaseTable() {
  const rows = filteredScores().slice().sort((a, b) => String(a.case_id).localeCompare(String(b.case_id)));
  byId("caseCount").textContent = `${rows.length} shown`;
  byId("caseRows").innerHTML = rows.map(row => {
    const selected = row.case_id === state.selected ? " class=\"selected\"" : "";
    return `<tr${selected} data-case="${row.case_id}">
      <td><button class="button" data-case-button="${row.case_id}">${row.case_id}</button></td>
      <td>${row[state.color] ?? ""}</td>
      <td>${fmt(row.PC1, 2)}</td>
      <td>${fmt(row.PC2, 2)}</td>
      <td>${fmt(row.PC3, 2)}</td>
      <td>${fmt(row.PC4, 2)}</td>
      <td>${fmt(row.PC5, 2)}</td>
    </tr>`;
  }).join("");
}

function renderModelTables() {
  byId("modelRows").innerHTML = (payload.modelSelection || []).map(row => {
    return `<tr><td>${row.k}</td><td>${row.status}</td><td>${fmt(row.silhouette, 3)}</td>
      <td>${fmt(row.inertia, 0)}</td><td>${row.cluster_sizes || ""}</td></tr>`;
  }).join("");
  byId("clusterRows").innerHTML = (payload.clusterSummary || []).map(row => {
    return `<tr><td><span class="badge" style="background:${colorFor(row.cluster)}">${row.cluster}</span></td>
      <td>${row.n}</td><td>${fmt(row.PC1_mean, 2)}</td><td>${fmt(row.PC2_mean, 2)}</td>
      <td>${fmt(row.PC3_mean, 2)}</td><td>${fmt(row.PC4_mean, 2)}</td><td>${fmt(row.PC5_mean, 2)}</td></tr>`;
  }).join("");
  byId("featureReportRows").innerHTML = (payload.featureReport || []).map(row => {
    return `<tr><td>${row.status}</td><td>${row.reason || ""}</td><td>${row.count}</td></tr>`;
  }).join("");
}

function renderLoadings() {
  const component = byId("loadingPc").value || pcs[0];
  const rows = (payload.topLoadings || [])
    .filter(row => row.component === component)
    .sort((a, b) => Math.abs(Number(b.loading)) - Math.abs(Number(a.loading)))
    .slice(0, 25);
  const maxAbs = Math.max(...rows.map(row => Math.abs(Number(row.loading))), 1e-9);
  byId("loadingBars").innerHTML = rows.map(row => {
    const value = Number(row.loading);
    const width = Math.abs(value) / maxAbs * 100;
    return `<div class="bar-row">
      <div class="bar-label" title="${row.feature}">${row.feature}</div>
      <div class="bar-track">
        <div class="bar-fill ${value < 0 ? "neg" : ""}" style="width:${width}%"></div>
      </div>
      <div>${fmt(value, 3)}</div>
    </div>`;
  }).join("");
}

function renderProfiles() {
  const rows = payload.clusterProfiles || [];
  byId("profileRows").innerHTML = rows.map(row => {
    return `<tr><td>${row.cluster}</td><td>${row.feature}</td>
      <td>${fmt(row.cluster_mean_z, 3)}</td><td>${fmt(row.max_abs_cluster_mean_z, 3)}</td></tr>`;
  }).join("");
}

function renderSelectedInterpretability(caseId) {
  const rows = (payload.interpretableCases || []).filter(row => row.case_id === caseId);
  byId("selectedInterpretRows").innerHTML = rows.map(row => {
    return `<tr><td>${row.domain}</td><td>${row.label}</td>
      <td>${fmt(row.value, 3)}</td><td>${row.unit || ""}</td></tr>`;
  }).join("");
}

function renderSelectedDomainState(caseId) {
  const row = (payload.domainStateCases || []).find(item => item.case_id === caseId);
  if (!row) {
    byId("caseDomainPhenotype").textContent = "";
    byId("caseDomainStates").innerHTML = "";
    return;
  }
  const domains = [
    ["calcium", "Calcium"],
    ["wall", "Wall"],
    ["peri_fat", "Peri-fat"]
  ];
  byId("caseDomainPhenotype").textContent = row.forced_domain_phenotype || "";
  byId("caseDomainStates").innerHTML = domains.map(([key, label]) => {
    return `<div class="metric"><div class="name">${label}</div>
      <div class="num">${row[`${key}_state`] || "NA"}</div>
      <div class="name">score ${fmt(row[`${key}_score`], 2)} | features ${row[`${key}_feature_count`] ?? ""}</div>
    </div>`;
  }).join("");
}

function renderInterpretabilityTables() {
  byId("interpretationRows").innerHTML = (payload.interpretationSummary || []).map(row => {
    return `<tr><td><span class="badge" style="background:${colorFor(row.cluster)}">${row.cluster}</span></td>
      <td>${row.n_cases}</td><td>${row.calcium_drivers || ""}</td>
      <td>${row.wall_drivers || ""}</td><td>${row.peri_fat_drivers || ""}</td></tr>`;
  }).join("");
  byId("representativeRows").innerHTML = (payload.representativeCases || []).map(row => {
    return `<tr><td>${row.cluster}</td><td>${row.rank}</td><td>${row.case_id}</td>
      <td>${fmt(row.distance_to_cluster_center, 3)}</td></tr>`;
  }).join("");
  byId("readableProfileRows").innerHTML = (payload.interpretableProfiles || [])
    .filter(row => row.direction !== "similar")
    .sort((a, b) => {
      return Number(a.cluster) - Number(b.cluster) ||
        String(a.domain).localeCompare(String(b.domain)) ||
        Math.abs(Number(b.delta_z)) - Math.abs(Number(a.delta_z));
    })
    .map(row => {
      return `<tr><td>${row.cluster}</td><td>${row.domain}</td><td>${row.direction}</td>
        <td>${row.label}</td><td>${fmt(row.cluster_mean, 3)}</td>
        <td>${fmt(row.cohort_mean, 3)}</td><td>${fmt(row.delta_z, 2)}</td>
        <td>${row.unit || ""}</td></tr>`;
    }).join("");
}

function renderDomainStateTables() {
  byId("domainStateRows").innerHTML = (payload.domainStateSummary || []).map(row => {
    return `<tr><td>${row.forced_domain_phenotype}</td><td>${row.n_cases}</td>
      <td>${row.calcium_state}</td><td>${row.wall_state}</td><td>${row.peri_fat_state}</td>
      <td>${fmt(row.calcium_score_mean, 2)}</td><td>${fmt(row.wall_score_mean, 2)}</td>
      <td>${fmt(row.peri_fat_score_mean, 2)}</td><td>${row.clusters || ""}</td></tr>`;
  }).join("");
  byId("domainStateClusterRows").innerHTML = (payload.domainStateByCluster || []).map(row => {
    return `<tr><td>${row.cluster}</td><td>${row.domain}</td><td>${row.n_cases}</td>
      <td>${row.low}</td><td>${row.mid}</td><td>${row.high}</td><td>${row.missing}</td>
      <td>${row.top_state}</td><td>${fmt(row.top_state_fraction, 2)}</td>
      <td>${row.top_phenotypes || ""}</td></tr>`;
  }).join("");
}

function renderDomainTables() {
  byId("domainRows").innerHTML = (payload.domainSummary || []).map(row => {
    return `<tr><td>${row.domain}</td><td>${row.usable_features}</td><td>${row.pca_components}</td>
      <td>${fmt(row.cumulative_explained_variance_ratio, 3)}</td></tr>`;
  }).join("");
  byId("domainVarianceRows").innerHTML = (payload.domainVariance || []).map(row => {
    return `<tr><td>${row.domain}</td><td>${row.component}</td>
      <td>${fmt(row.explained_variance_ratio, 3)}</td>
      <td>${fmt(row.cumulative_explained_variance_ratio, 3)}</td></tr>`;
  }).join("");
  renderDomainLoadings();
}

function renderDomainLoadings() {
  const rows = payload.domainTopLoadings || [];
  const selectedDomain = byId("domainLoadingDomain").value || "all";
  const selectedComponent = byId("domainLoadingComponent").value || "all";
  byId("domainLoadingRows").innerHTML = rows
    .filter(row => selectedDomain === "all" || row.domain === selectedDomain)
    .filter(row => selectedComponent === "all" || row.component === selectedComponent)
    .slice(0, 150)
    .map(row => {
      return `<tr><td>${row.domain}</td><td>${row.component}</td><td>${row.feature}</td>
        <td>${fmt(row.loading, 4)}</td><td>${fmt(row.abs_loading, 4)}</td></tr>`;
    }).join("");
}

function populateDomainLoadingControls() {
  const rows = payload.domainTopLoadings || [];
  const domains = [...new Set(rows.map(row => row.domain))].sort();
  const components = [...new Set(rows.map(row => row.component))].sort((a, b) => {
    return a.localeCompare(b, undefined, { numeric: true });
  });
  byId("domainLoadingDomain").innerHTML = `<option value="all">all domains</option>` +
    domains.map(domain => `<option value="${domain}">${domain}</option>`).join("");
  byId("domainLoadingComponent").innerHTML = `<option value="all">all components</option>` +
    components.map(component => `<option value="${component}">${component}</option>`).join("");
}

function renderFigures() {
  const figures = [
    ["PCA clusters", "figures/pca_clusters_best.png"],
    ["Explained variance", "figures/pca_explained_variance.png"],
    ["Top loadings", "figures/pca_top_loadings.png"],
    ["Cluster profiles", "figures/cluster_profile_heatmap.png"]
  ];
  byId("figures").innerHTML = figures.map(([label, path]) => {
    return `<a class="figure-tile" href="${path}"><img src="${path}" alt="${label}"><div>${label}</div></a>`;
  }).join("");
}

function tabTargetFromHash() {
  const hash = window.location.hash.replace("#", "");
  if (hash === "centroid") return "tabCentroidViewer";
  if (hash === "recursive") return "tabRecursiveSelection";
  return "tabFullPca";
}

function hashForTab(target) {
  if (target === "tabCentroidViewer") return "#centroid";
  if (target === "tabRecursiveSelection") return "#recursive";
  return "#pca";
}

function activateTab(target, updateHash = true) {
  document.querySelectorAll("[data-tab-target]").forEach(item => {
    item.classList.toggle("active", item.dataset.tabTarget === target);
  });
  document.querySelectorAll(".tab-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === target);
  });
  if (updateHash) {
    history.replaceState(null, "", hashForTab(target));
  }
  if (target === "tabFullPca") {
    setTimeout(() => {
      resizeScatter();
      renderScatter();
    }, 0);
  }
  if (target === "tabCentroidViewer") {
    setTimeout(() => renderCentroidViewer(), 0);
  }
}

function wireEvents() {
  document.querySelectorAll("[data-tab-target]").forEach(button => {
    button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
  });
  byId("xSelect").addEventListener("change", event => {
    state.x = event.target.value;
    renderScatter();
  });
  byId("ySelect").addEventListener("change", event => {
    state.y = event.target.value;
    renderScatter();
  });
  byId("zSelect").addEventListener("change", event => {
    state.z = event.target.value;
    renderScatter();
  });
  byId("colorSelect").addEventListener("change", event => {
    state.color = event.target.value;
    resetClusterFilter();
    renderAll();
  });
  byId("resetCamera").addEventListener("click", resetCamera);
  byId("caseSearch").addEventListener("input", renderAll);
  byId("loadingPc").addEventListener("change", renderLoadings);
  byId("domainLoadingDomain").addEventListener("change", renderDomainLoadings);
  byId("domainLoadingComponent").addEventListener("change", renderDomainLoadings);
  byId("clusterFilter").addEventListener("change", event => {
    const target = event.target;
    if (!target.dataset.cluster) return;
    if (target.checked) state.activeClusters.add(target.dataset.cluster);
    else state.activeClusters.delete(target.dataset.cluster);
    renderAll();
  });
  byId("caseRows").addEventListener("click", event => {
    const button = event.target.closest("[data-case-button]");
    if (button) selectCase(button.dataset.caseButton);
  });
  byId("centroidList").addEventListener("click", event => {
    const button = event.target.closest("[data-centroid-case]");
    if (!button) return;
    state.centroid = button.dataset.centroidCase;
    renderCentroidViewer();
  });
}

function renderAll() {
  renderStats();
  if (byId("tabFullPca").classList.contains("active")) {
    renderScatter();
  }
  renderCaseTable();
}

async function startExplorer() {
  await loadThree();
  populateControls();
  populateDomainLoadingControls();
  wireEvents();
  activateTab(tabTargetFromHash(), false);
  renderModelTables();
  renderInterpretabilityTables();
  renderDomainStateTables();
  renderRecursiveSelectionTab();
  renderLoadings();
  renderProfiles();
  renderDomainTables();
  renderFigures();
  renderAll();
  selectCase(state.selected);
}

startExplorer();
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aorta PCA Clustering Explorer</title>
<style>{css}</style>
</head>
<body>
<header>
  <div class="header-row">
    <div>
      <h1>Aorta PCA Clustering Explorer</h1>
      <div class="subhead">
        Static explorer for PCA scores, KMeans clusters, loadings, and cluster profiles.
      </div>
    </div>
    <nav class="actions" aria-label="Output links">
      <a class="button" href="pca_scores_clusters.csv">Scores</a>
      <a class="button" href="cluster_assignments_best.csv">Assignments</a>
      <a class="button" href="pca_top_loadings.csv">Loadings</a>
      <a class="button" href="cluster_profile_top_features.csv">Profiles</a>
      <a class="button" href="domain_state_case_features.csv">Domain States</a>
      <a class="button" href="../aorta_batch_review.html">Batch Review</a>
    </nav>
  </div>
</header>
<main>
  <div class="tabbar" role="tablist" aria-label="Analysis views">
    <button class="tab-button active" type="button" data-tab-target="tabFullPca">Full PCA</button>
    <button class="tab-button" type="button" data-tab-target="tabCentroidViewer">Centroid Viewer</button>
    <button class="tab-button" type="button" data-tab-target="tabRecursiveSelection">
      Recursive Selection
    </button>
  </div>

  <section class="tab-panel active" id="tabFullPca">
  <section class="stats" id="stats" aria-label="Analysis summary"></section>
  <div class="workspace">
    <div>
      <section class="section">
        <div class="section-head">
          <div>
            <div class="section-title">3D PCA Score Space</div>
            <div class="subhead" id="caseCount"></div>
          </div>
          <div class="controls">
            <select id="xSelect" aria-label="X component"></select>
            <select id="ySelect" aria-label="Y component"></select>
            <select id="zSelect" aria-label="Z component"></select>
            <select id="colorSelect" aria-label="Cluster column"></select>
            <button class="button" id="resetCamera" type="button">Reset View</button>
            <input id="caseSearch" type="search" placeholder="Search case" aria-label="Search case">
          </div>
        </div>
        <div class="plot-wrap">
          <div id="scatter" role="img" aria-label="3D PCA scatter plot">
            <div class="plot-status">Loading 3D PCA plot...</div>
          </div>
          <div class="legend" id="legend"></div>
          <div class="legend" id="clusterFilter"></div>
        </div>
      </section>

      <section class="section">
        <div class="section-head"><div class="section-title">Case Scores</div></div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Case</th><th>Cluster</th><th>PC1</th><th>PC2</th><th>PC3</th><th>PC4</th><th>PC5</th>
              </tr>
            </thead>
            <tbody id="caseRows"></tbody>
          </table>
        </div>
      </section>
    </div>

    <aside class="section panel">
      <div class="section-head"><div class="section-title">Selected Case</div></div>
      <div class="selected-panel">
        <h2 class="case-title" id="caseTitle"></h2>
        <div class="subhead" id="caseSubhead"></div>
        <div class="domain-phenotype" id="caseDomainPhenotype"></div>
        <div class="case-grid" id="caseDomainStates"></div>
        <div class="case-grid" id="caseMetrics"></div>
        <div class="section-head" style="padding-left:0;padding-right:0;margin-top:10px;">
          <div class="section-title">Readable Features</div>
        </div>
        <div class="table-wrap" style="max-height:320px;">
          <table>
            <thead><tr><th>Domain</th><th>Feature</th><th>Value</th><th>Unit</th></tr></thead>
            <tbody id="selectedInterpretRows"></tbody>
          </table>
        </div>
      </div>
    </aside>
  </div>

  <div class="grid-two">
    <section class="section">
      <div class="section-head"><div class="section-title">Cluster Model Selection</div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>k</th><th>Status</th><th>Silhouette</th><th>Inertia</th><th>Sizes</th></tr></thead>
          <tbody id="modelRows"></tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">Cluster Summary</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Cluster</th><th>n</th><th>PC1</th><th>PC2</th><th>PC3</th><th>PC4</th><th>PC5</th></tr>
          </thead>
          <tbody id="clusterRows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <section class="section">
    <div class="section-head"><div class="section-title">Readable Cluster Interpretation</div></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Cluster</th><th>n</th><th>Calcium drivers</th><th>Wall drivers</th><th>Peri-fat drivers</th>
          </tr>
        </thead>
        <tbody id="interpretationRows"></tbody>
      </table>
    </div>
  </section>

  <div class="grid-two">
    <section class="section">
      <div class="section-head"><div class="section-title">Forced Domain Phenotypes</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Phenotype</th><th>n</th><th>Calcium</th><th>Wall</th><th>Peri-fat</th>
              <th>Calcium Score</th><th>Wall Score</th><th>Peri-fat Score</th><th>Clusters</th>
            </tr>
          </thead>
          <tbody id="domainStateRows"></tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">Domain States By KMeans Cluster</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Cluster</th><th>Domain</th><th>n</th><th>Low</th><th>Mid</th><th>High</th>
              <th>Missing</th><th>Top State</th><th>Top Fraction</th><th>Top Phenotypes</th>
            </tr>
          </thead>
          <tbody id="domainStateClusterRows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <div class="grid-two">
    <section class="section">
      <div class="section-head"><div class="section-title">Representative Cases</div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Cluster</th><th>Rank</th><th>Case</th><th>Distance</th></tr></thead>
          <tbody id="representativeRows"></tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">Readable Cluster Profile</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Cluster</th><th>Domain</th><th>Direction</th><th>Feature</th>
              <th>Cluster Mean</th><th>Cohort Mean</th><th>Delta z</th><th>Unit</th>
            </tr>
          </thead>
          <tbody id="readableProfileRows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <div class="grid-two">
    <section class="section">
      <div class="section-head"><div class="section-title">Domain Coverage</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Domain</th><th>Usable Features</th><th>Domain PCs</th><th>Cumulative Variance</th></tr>
          </thead>
          <tbody id="domainRows"></tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">Domain PCA Variance</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Domain</th><th>Component</th><th>Variance</th><th>Cumulative</th></tr>
          </thead>
          <tbody id="domainVarianceRows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <section class="section">
    <div class="section-head">
      <div class="section-title">Top PCA Loadings</div>
      <select id="loadingPc" aria-label="Loading component"></select>
    </div>
    <div class="bars" id="loadingBars"></div>
  </section>

  <section class="section">
    <div class="section-head">
      <div class="section-title">Top Original Feature Loadings By Domain</div>
      <div class="controls">
        <select id="domainLoadingDomain" aria-label="Domain loading domain"></select>
        <select id="domainLoadingComponent" aria-label="Domain loading component"></select>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Domain</th><th>Component</th><th>Feature</th><th>Loading</th><th>Abs Loading</th></tr>
        </thead>
        <tbody id="domainLoadingRows"></tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <div class="section-head"><div class="section-title">Top Cluster-Separating Features</div></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Cluster</th><th>Feature</th><th>Mean z</th><th>Max abs z</th></tr></thead>
        <tbody id="profileRows"></tbody>
      </table>
    </div>
  </section>

  <div class="grid-two">
    <section class="section">
      <div class="section-head"><div class="section-title">Feature Filter Summary</div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Status</th><th>Reason</th><th>Count</th></tr></thead>
          <tbody id="featureReportRows"></tbody>
        </table>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><div class="section-title">Generated Figures</div></div>
      <div class="figure-grid" id="figures"></div>
    </section>
  </div>
  </section>

  <section class="tab-panel" id="tabCentroidViewer">
    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-title">Cluster Centroid Patients</div>
          <div class="subhead">
            The centroid patient is the observed case closest to each KMeans cluster center.
          </div>
        </div>
      </div>
      <div class="centroid-layout">
        <div class="centroid-list" id="centroidList"></div>
        <div>
          <div id="centroidViewerBody"></div>
          <section class="section" style="margin-top:14px;">
            <div class="section-head"><div class="section-title">Centroid Case Readable Features</div></div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Domain</th><th>Feature</th><th>Value</th><th>Unit</th></tr></thead>
                <tbody id="centroidFeatureRows"></tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </section>
  </section>

  <section class="tab-panel" id="tabRecursiveSelection">
    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-title">Domain-Balanced Recursive Selection</div>
          <div class="subhead">Comparison against the full domain-balanced PCA run.</div>
        </div>
        <nav class="actions" aria-label="Recursive selection links">
          <a class="button" id="recursiveReportLink" href="../domain_balanced_recursive_selection/selection_report.html">
            Selection Report
          </a>
          <a class="button" id="recursiveExplorerLink" href="../domain_balanced_recursive_selection/pca_clustering/pca_clustering_explorer.html">
            Selected PCA Explorer
          </a>
          <a class="button" id="recursiveFeaturesLink" href="../domain_balanced_recursive_selection/selected_features.csv">
            Selected Features
          </a>
          <a class="button" id="recursiveTraceLink" href="../domain_balanced_recursive_selection/selection_trace.csv">
            Trace
          </a>
        </nav>
      </div>
      <div class="verdict" id="recursiveVerdict"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Metric</th><th>Full PCA</th><th>Recursive Selection</th><th>Interpretation</th></tr>
          </thead>
          <tbody id="recursiveComparisonRows"></tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><div class="section-title">Selected Variables</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Domain</th><th>Order</th><th>Feature</th><th>Step</th><th>Domain Rank</th>
              <th>Representativeness</th>
            </tr>
          </thead>
          <tbody id="recursiveSelectedRows"></tbody>
        </table>
      </div>
    </section>

    <div class="grid-two">
      <section class="section">
        <div class="section-head"><div class="section-title">Recursive Trace</div></div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Step</th><th>Domain</th><th>Feature</th><th>Reason</th><th>Objective</th>
                <th>Silhouette</th><th>Best k</th><th>Redundancy</th>
              </tr>
            </thead>
            <tbody id="recursiveTraceRows"></tbody>
          </table>
        </div>
      </section>
      <section class="section">
        <div class="section-head"><div class="section-title">Selected-Panel KMeans</div></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>k</th><th>Status</th><th>Silhouette</th><th>Inertia</th><th>Sizes</th></tr></thead>
            <tbody id="recursiveModelRows"></tbody>
          </table>
        </div>
      </section>
    </div>

    <section class="section">
      <div class="section-head"><div class="section-title">Subsample Stability</div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Repeat</th><th>Sample n</th><th>Best k</th><th>ARI</th><th>Subset Sizes</th></tr></thead>
          <tbody id="recursiveStabilityRows"></tbody>
        </table>
      </div>
    </section>
  </section>
</main>
<script type="application/json" id="payload">{payload_json}</script>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/",
    "niivue": "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/index.js"
  }}
}}
</script>
<script type="module">{script}</script>
</body>
</html>
"""


def _plot_variance(variance: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(variance)) + 1
    ax.bar(x, variance["explained_variance_ratio"], color="#0f766e", label="Component")
    ax.plot(
        x,
        variance["cumulative_explained_variance_ratio"],
        color="#7c3aed",
        marker="o",
        label="Cumulative",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(x)
    ax.set_ylim(0, max(1.0, float(variance["cumulative_explained_variance_ratio"].max()) * 1.05))
    ax.legend(frameon=False)
    ax.set_title("PCA explained variance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_clusters(scores: pd.DataFrame, best_k: int, path: Path) -> None:
    labels = scores["cluster_best"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        scores["PC1"],
        scores["PC2"],
        c=labels,
        cmap="tab10",
        s=34,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="#d0d7d4", linewidth=0.8)
    ax.axvline(0, color="#d0d7d4", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA scores with KMeans clusters (k={best_k})")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", frameon=False, loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_top_loadings(top_loadings: pd.DataFrame, path: Path) -> None:
    if top_loadings.empty:
        return
    plot_components = ["PC1", "PC2"]
    part = top_loadings[top_loadings["component"].isin(plot_components)].copy()
    part = part.groupby("component", group_keys=False).head(15)
    nrows = len(plot_components)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 8), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, component in zip(axes, plot_components, strict=False):
        comp = part[part["component"] == component].sort_values("loading")
        colors = np.where(comp["loading"] >= 0, "#0f766e", "#b45309")
        ax.barh(comp["feature"], comp["loading"], color=colors)
        ax.set_title(f"Top {component} loadings")
        ax.axvline(0, color="#3f4746", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cluster_heatmap(cluster_profiles_path: Path, path: Path) -> None:
    top = pd.read_csv(cluster_profiles_path)
    if top.empty:
        return
    matrix = top.pivot(index="feature", columns="cluster", values="cluster_mean_z")
    feature_order = (
        top.groupby("feature")["max_abs_cluster_mean_z"]
        .first()
        .sort_values(ascending=True)
        .index
    )
    matrix = matrix.loc[feature_order]
    height = max(7, min(18, 0.23 * len(matrix)))
    fig, ax = plt.subplots(figsize=(8, height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(matrix.shape[1]), labels=[str(col) for col in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Cluster")
    ax.set_title("Top cluster-separating features")
    fig.colorbar(image, ax=ax, label="Cluster mean z-score", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
