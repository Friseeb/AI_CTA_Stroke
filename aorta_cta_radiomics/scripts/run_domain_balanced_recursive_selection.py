#!/usr/bin/env python
"""Domain-balanced recursive feature selection for aorta CTA radiomics.

This is an unsupervised selector. It recursively adds variables while enforcing
the same target count for calcium, wall, and peri-fat features. Candidate
features are scored by the KMeans silhouette of a domain-balanced PCA embedding,
with a penalty for redundancy within the same domain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from run_pca_clustering import (
    DEFAULT_FEATURES,
    DOMAIN_DEFINITIONS,
    INTERPRETABLE_FEATURES,
    assign_feature_domain,
    parse_k_values,
    prepare_feature_matrix,
)


def main() -> None:
    args = build_parser().parse_args()
    features_path = Path(args.features).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found: {features_path}")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else features_path.parents[1] / "domain_balanced_recursive_selection"
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
    all_domain_features = features_by_domain(prepared.feature_names)
    domain_features = candidate_features_by_domain(
        all_domain_features=all_domain_features,
        prepared_feature_names=prepared.feature_names,
        candidate_mode=args.candidate_mode,
    )
    for domain in DOMAIN_DEFINITIONS:
        if len(domain_features[domain]) < args.features_per_domain:
            raise ValueError(
                f"Domain '{domain}' has only {len(domain_features[domain])} usable candidate features "
                f"for --candidate-mode {args.candidate_mode}; need {args.features_per_domain}."
            )

    scaled = pd.DataFrame(prepared.matrix, columns=prepared.feature_names)
    ranking = rank_domain_features(
        prepared=prepared,
        domain_features=domain_features,
        domain_components=args.domain_components,
    )
    ranking.to_csv(outdir / "domain_feature_ranking.csv", index=False)

    selected_by_domain, trace, candidate_evaluations = recursive_select(
        scaled=scaled,
        prepared=prepared,
        ranking=ranking,
        domain_features=domain_features,
        features_per_domain=args.features_per_domain,
        candidate_pool_per_domain=args.candidate_pool_per_domain,
        lookahead_per_step=args.lookahead_per_step,
        domain_components=args.domain_components,
        k_values=parse_k_values(args.k_values),
        random_state=args.random_state,
        redundancy_penalty=args.redundancy_penalty,
        max_correlation=args.max_correlation,
    )
    selected = selected_feature_table(selected_by_domain, ranking, trace)
    selected.to_csv(outdir / "selected_features.csv", index=False)
    trace.to_csv(outdir / "selection_trace.csv", index=False)
    candidate_evaluations.to_csv(outdir / "candidate_evaluations.csv", index=False)

    final_model = evaluate_selected_set(
        scaled=scaled,
        selected_by_domain=selected_by_domain,
        domain_components=args.domain_components,
        k_values=parse_k_values(args.k_values),
        random_state=args.random_state,
    )
    final_model.selection.to_csv(outdir / "final_model_selection.csv", index=False)
    final_scores = final_model.scores.copy()
    final_scores.insert(0, "case_id", prepared.case_ids)
    final_scores["cluster_best"] = final_model.labels
    final_scores.to_csv(outdir / "selected_scores_clusters.csv", index=False)

    stability = stability_summary(
        matrix=final_model.matrix,
        labels=final_model.labels,
        best_k=final_model.best_k,
        repeats=args.stability_repeats,
        fraction=args.stability_fraction,
        random_state=args.random_state,
    )
    stability.to_csv(outdir / "stability_summary.csv", index=False)

    selected_columns = [feature for domain in DOMAIN_DEFINITIONS for feature in selected_by_domain[domain]]
    selected_raw = raw[[args.case_id_column, *selected_columns]].copy()
    selected_raw.to_csv(outdir / "selected_wide_features.csv", index=False)

    summary = {
        "features_path": str(features_path),
        "outdir": str(outdir),
        "cases": len(prepared.case_ids),
        "usable_numeric_features": len(prepared.feature_names),
        "features_per_domain": args.features_per_domain,
        "candidate_mode": args.candidate_mode,
        "selected_features": len(selected_columns),
        "domains": {
            domain: {
                "usable_features": len(all_domain_features[domain]),
                "candidate_features": len(domain_features[domain]),
                "selected_features": len(selected_by_domain[domain]),
            }
            for domain in DOMAIN_DEFINITIONS
        },
        "domain_components": args.domain_components,
        "k_values": parse_k_values(args.k_values),
        "best_k": final_model.best_k,
        "best_silhouette": final_model.best_silhouette,
        "stability_adjusted_rand_mean": float(stability["adjusted_rand_index"].mean())
        if not stability.empty
        else math.nan,
        "stability_adjusted_rand_median": float(stability["adjusted_rand_index"].median())
        if not stability.empty
        else math.nan,
        "outputs": [
            "selection_report.html",
            "selected_features.csv",
            "selection_trace.csv",
            "candidate_evaluations.csv",
            "domain_feature_ranking.csv",
            "final_model_selection.csv",
            "selected_scores_clusters.csv",
            "stability_summary.csv",
            "selected_wide_features.csv",
            "pca_clustering/pca_clustering_explorer.html",
        ],
    }
    (outdir / "selection_summary.json").write_text(json.dumps(clean_json(summary), indent=2), encoding="utf-8")
    write_selection_report(outdir, summary)

    if not args.skip_pca_rerun:
        rerun_selected_pca(
            selected_features_path=outdir / "selected_wide_features.csv",
            outdir=outdir / "pca_clustering",
            args=args,
        )

    print(f"Cases: {len(prepared.case_ids)}")
    print(f"Selected features: {len(selected_columns)} ({args.features_per_domain} per domain)")
    print(f"Best k: {final_model.best_k} (silhouette={final_model.best_silhouette:.3f})")
    if not stability.empty:
        print(
            "Stability ARI: "
            f"mean={stability['adjusted_rand_index'].mean():.3f}, "
            f"median={stability['adjusted_rand_index'].median():.3f}"
        )
    print(f"Selection report: {outdir / 'selection_report.html'}")
    if not args.skip_pca_rerun:
        print(f"Selected-feature explorer: {outdir / 'pca_clustering' / 'pca_clustering_explorer.html'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=DEFAULT_FEATURES, type=Path)
    parser.add_argument("--outdir", default=None, type=Path)
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument("--max-missing", type=float, default=0.25)
    parser.add_argument("--min-variance", type=float, default=1e-12)
    parser.add_argument("--drop-regex", default=r"(software_version|segmentation_method|status)$")
    parser.add_argument(
        "--candidate-mode",
        choices=["interpretable", "all"],
        default="interpretable",
        help="Use the curated readable feature set by default; choose all for fully data-driven selection.",
    )
    parser.add_argument("--features-per-domain", type=int, default=6)
    parser.add_argument("--domain-components", type=int, default=3)
    parser.add_argument("--candidate-pool-per-domain", type=int, default=80)
    parser.add_argument("--lookahead-per-step", type=int, default=16)
    parser.add_argument("--k-values", default="2,3,4,5,6")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--redundancy-penalty",
        type=float,
        default=0.04,
        help="Penalty applied to candidate objective for max same-domain absolute correlation.",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.98,
        help="Skip candidates above this same-domain correlation when alternatives exist.",
    )
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--stability-fraction", type=float, default=0.8)
    parser.add_argument("--skip-pca-rerun", action="store_true")
    return parser


class SelectedModel:
    def __init__(
        self,
        matrix: np.ndarray,
        scores: pd.DataFrame,
        selection: pd.DataFrame,
        labels: np.ndarray,
        best_k: int,
        best_silhouette: float,
    ) -> None:
        self.matrix = matrix
        self.scores = scores
        self.selection = selection
        self.labels = labels
        self.best_k = best_k
        self.best_silhouette = best_silhouette


def features_by_domain(feature_names: list[str]) -> dict[str, list[str]]:
    grouped = {domain: [] for domain in DOMAIN_DEFINITIONS}
    for feature in feature_names:
        domain = assign_feature_domain(feature)
        if domain in grouped:
            grouped[domain].append(feature)
    return grouped


def candidate_features_by_domain(
    all_domain_features: dict[str, list[str]],
    prepared_feature_names: list[str],
    candidate_mode: str,
) -> dict[str, list[str]]:
    if candidate_mode == "all":
        return {domain: list(features) for domain, features in all_domain_features.items()}
    prepared_set = set(prepared_feature_names)
    curated = {domain: [] for domain in DOMAIN_DEFINITIONS}
    for spec in INTERPRETABLE_FEATURES:
        column = str(spec["column"])
        domain = str(spec["domain"])
        if domain in curated and column in prepared_set:
            curated[domain].append(column)
    return curated


def rank_domain_features(
    prepared,
    domain_features: dict[str, list[str]],
    domain_components: int,
) -> pd.DataFrame:
    feature_to_index = {feature: index for index, feature in enumerate(prepared.feature_names)}
    rows: list[dict[str, object]] = []
    for domain, features in domain_features.items():
        indices = [feature_to_index[feature] for feature in features]
        matrix = prepared.matrix[:, indices]
        n_components = min(domain_components, matrix.shape[0] - 1, matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(matrix)
        weights = pca.explained_variance_ratio_
        if weights.sum() > 0:
            weights = weights / weights.sum()
        scores = np.abs(pca.components_.T) @ weights
        for feature, score in zip(features, scores, strict=True):
            rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "domain_pca_representativeness": float(score),
                    "usable_domain_features": len(features),
                }
            )
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["domain", "domain_pca_representativeness", "feature"],
        ascending=[True, False, True],
    )
    ranking["domain_rank"] = ranking.groupby("domain").cumcount() + 1
    return ranking


def recursive_select(
    scaled: pd.DataFrame,
    prepared,
    ranking: pd.DataFrame,
    domain_features: dict[str, list[str]],
    features_per_domain: int,
    candidate_pool_per_domain: int,
    lookahead_per_step: int,
    domain_components: int,
    k_values: list[int],
    random_state: int,
    redundancy_penalty: float,
    max_correlation: float,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    selected_by_domain = {domain: [] for domain in DOMAIN_DEFINITIONS}
    trace_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    ranking_lookup = ranking.set_index("feature")
    global_step = 0

    for domain in DOMAIN_DEFINITIONS:
        feature = str(ranking[ranking["domain"] == domain].iloc[0]["feature"])
        selected_by_domain[domain].append(feature)
        global_step += 1
        trace_rows.append(
            trace_row(
                step=global_step,
                domain=domain,
                feature=feature,
                reason="seed_top_domain_pca_feature",
                objective=math.nan,
                silhouette=math.nan,
                best_k=math.nan,
                redundancy=0.0,
                ranking_lookup=ranking_lookup,
                selected_by_domain=selected_by_domain,
            )
        )

    for selection_round in range(2, features_per_domain + 1):
        for domain in DOMAIN_DEFINITIONS:
            candidates = candidate_list(
                domain=domain,
                ranking=ranking,
                selected=selected_by_domain[domain],
                candidate_pool_per_domain=candidate_pool_per_domain,
            )
            candidates = prune_redundant_candidates(
                scaled=scaled,
                domain=domain,
                candidates=candidates,
                selected=selected_by_domain[domain],
                max_correlation=max_correlation,
            )[:lookahead_per_step]
            best: dict[str, object] | None = None
            for candidate in candidates:
                trial = {key: list(value) for key, value in selected_by_domain.items()}
                trial[domain].append(candidate)
                redundancy = max_abs_corr(scaled, candidate, selected_by_domain[domain])
                model = evaluate_selected_set(
                    scaled=scaled,
                    selected_by_domain=trial,
                    domain_components=domain_components,
                    k_values=k_values,
                    random_state=random_state,
                )
                objective = model.best_silhouette - redundancy_penalty * redundancy
                row = {
                    "round": selection_round,
                    "domain": domain,
                    "candidate": candidate,
                    "objective": objective,
                    "silhouette": model.best_silhouette,
                    "best_k": model.best_k,
                    "redundancy_max_abs_corr": redundancy,
                    "candidate_domain_rank": int(ranking_lookup.loc[candidate, "domain_rank"]),
                    "candidate_representativeness": float(
                        ranking_lookup.loc[candidate, "domain_pca_representativeness"]
                    ),
                }
                evaluation_rows.append(row)
                if best is None or float(row["objective"]) > float(best["objective"]):
                    best = row
            if best is None:
                raise RuntimeError(f"No candidate available for domain '{domain}' at round {selection_round}.")

            selected_by_domain[domain].append(str(best["candidate"]))
            global_step += 1
            trace_rows.append(
                trace_row(
                    step=global_step,
                    domain=domain,
                    feature=str(best["candidate"]),
                    reason=f"recursive_round_{selection_round}",
                    objective=float(best["objective"]),
                    silhouette=float(best["silhouette"]),
                    best_k=int(best["best_k"]),
                    redundancy=float(best["redundancy_max_abs_corr"]),
                    ranking_lookup=ranking_lookup,
                    selected_by_domain=selected_by_domain,
                )
            )

    trace = pd.DataFrame(trace_rows)
    evaluations = pd.DataFrame(evaluation_rows)
    return selected_by_domain, trace, evaluations


def candidate_list(
    domain: str,
    ranking: pd.DataFrame,
    selected: list[str],
    candidate_pool_per_domain: int,
) -> list[str]:
    pool = ranking[ranking["domain"] == domain].head(candidate_pool_per_domain)
    selected_set = set(selected)
    return [str(feature) for feature in pool["feature"] if str(feature) not in selected_set]


def prune_redundant_candidates(
    scaled: pd.DataFrame,
    domain: str,
    candidates: list[str],
    selected: list[str],
    max_correlation: float,
) -> list[str]:
    if not selected:
        return candidates
    kept = [candidate for candidate in candidates if max_abs_corr(scaled, candidate, selected) <= max_correlation]
    return kept if kept else candidates


def max_abs_corr(scaled: pd.DataFrame, candidate: str, selected: list[str]) -> float:
    if not selected:
        return 0.0
    values = []
    candidate_series = scaled[candidate]
    for feature in selected:
        corr = candidate_series.corr(scaled[feature])
        if math.isfinite(float(corr)):
            values.append(abs(float(corr)))
    return max(values) if values else 0.0


def trace_row(
    step: int,
    domain: str,
    feature: str,
    reason: str,
    objective: float,
    silhouette: float,
    best_k: float,
    redundancy: float,
    ranking_lookup: pd.DataFrame,
    selected_by_domain: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "selection_step": step,
        "domain": domain,
        "feature": feature,
        "reason": reason,
        "objective": objective,
        "silhouette": silhouette,
        "best_k": best_k,
        "redundancy_max_abs_corr": redundancy,
        "domain_rank": int(ranking_lookup.loc[feature, "domain_rank"]),
        "domain_pca_representativeness": float(
            ranking_lookup.loc[feature, "domain_pca_representativeness"]
        ),
        "selected_calcium": len(selected_by_domain["calcium"]),
        "selected_wall": len(selected_by_domain["wall"]),
        "selected_peri_fat": len(selected_by_domain["peri_fat"]),
        "selected_total": sum(len(value) for value in selected_by_domain.values()),
    }


def evaluate_selected_set(
    scaled: pd.DataFrame,
    selected_by_domain: dict[str, list[str]],
    domain_components: int,
    k_values: list[int],
    random_state: int,
) -> SelectedModel:
    matrix, scores = domain_balanced_embedding(scaled, selected_by_domain, domain_components)
    rows: list[dict[str, object]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    best_k: int | None = None
    best_silhouette = -math.inf
    for k in k_values:
        if k >= matrix.shape[0]:
            rows.append({"k": k, "status": "skipped", "reason": "k_not_less_than_case_count"})
            continue
        model = KMeans(n_clusters=k, n_init=25, random_state=random_state)
        labels = model.fit_predict(matrix)
        labels_by_k[k] = labels
        silhouette = float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else math.nan
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
    if best_k is None:
        raise RuntimeError("Could not fit any KMeans model for selected feature set.")
    return SelectedModel(
        matrix=matrix,
        scores=scores,
        selection=pd.DataFrame(rows),
        labels=labels_by_k[best_k],
        best_k=best_k,
        best_silhouette=float(best_silhouette),
    )


def domain_balanced_embedding(
    scaled: pd.DataFrame,
    selected_by_domain: dict[str, list[str]],
    domain_components: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for domain in DOMAIN_DEFINITIONS:
        features = selected_by_domain[domain]
        if not features:
            raise ValueError(f"No selected features for domain '{domain}'.")
        matrix = scaled[features].to_numpy()
        n_components = min(domain_components, matrix.shape[0] - 1, matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        domain_scores = pca.fit_transform(matrix)
        columns = [f"{domain}_PC{i}" for i in range(1, n_components + 1)]
        parts.append(pd.DataFrame(domain_scores, columns=columns))
    scores = pd.concat(parts, axis=1)
    matrix = StandardScaler().fit_transform(scores.to_numpy())
    return matrix, scores


def selected_feature_table(
    selected_by_domain: dict[str, list[str]],
    ranking: pd.DataFrame,
    trace: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ranking_lookup = ranking.set_index("feature")
    trace_lookup = trace.set_index("feature")
    for domain in DOMAIN_DEFINITIONS:
        for index, feature in enumerate(selected_by_domain[domain], start=1):
            rows.append(
                {
                    "domain": domain,
                    "domain_selection_order": index,
                    "feature": feature,
                    "selection_step": int(trace_lookup.loc[feature, "selection_step"]),
                    "reason": trace_lookup.loc[feature, "reason"],
                    "domain_rank": int(ranking_lookup.loc[feature, "domain_rank"]),
                    "domain_pca_representativeness": float(
                        ranking_lookup.loc[feature, "domain_pca_representativeness"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def stability_summary(
    matrix: np.ndarray,
    labels: np.ndarray,
    best_k: int,
    repeats: int,
    fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if repeats < 1:
        return pd.DataFrame()
    if not 0 < fraction <= 1:
        raise ValueError("--stability-fraction must be in (0, 1].")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    n_cases = matrix.shape[0]
    sample_n = max(best_k + 2, int(round(n_cases * fraction)))
    sample_n = min(sample_n, n_cases)
    for repeat in range(1, repeats + 1):
        indices = np.sort(rng.choice(n_cases, size=sample_n, replace=False))
        model = KMeans(n_clusters=best_k, n_init=25, random_state=random_state + repeat)
        subset_labels = model.fit_predict(matrix[indices])
        ari = adjusted_rand_score(labels[indices], subset_labels)
        rows.append(
            {
                "repeat": repeat,
                "sample_fraction": fraction,
                "sample_n": sample_n,
                "best_k": best_k,
                "adjusted_rand_index": float(ari),
                "subset_cluster_sizes": ";".join(
                    str(int((subset_labels == label).sum())) for label in sorted(set(subset_labels))
                ),
            }
        )
    return pd.DataFrame(rows)


def rerun_selected_pca(selected_features_path: Path, outdir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_pca_clustering.py")),
        "--features",
        str(selected_features_path),
        "--outdir",
        str(outdir),
        "--case-id-column",
        args.case_id_column,
        "--max-missing",
        str(args.max_missing),
        "--min-variance",
        str(args.min_variance),
        "--drop-regex",
        args.drop_regex,
        "--analysis-mode",
        "domain-balanced",
        "--domain-components",
        str(args.domain_components),
        "--k-values",
        args.k_values,
        "--random-state",
        str(args.random_state),
    ]
    subprocess.run(command, check=True)


def write_selection_report(outdir: Path, summary: dict[str, object]) -> None:
    selected = records_from_csv(outdir / "selected_features.csv")
    trace = records_from_csv(outdir / "selection_trace.csv")
    model = records_from_csv(outdir / "final_model_selection.csv")
    stability = records_from_csv(outdir / "stability_summary.csv")
    payload = json.dumps(
        clean_json(
            {
                "summary": summary,
                "selected": selected,
                "trace": trace,
                "model": model,
                "stability": stability,
            }
        ),
        separators=(",", ":"),
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Domain-Balanced Recursive Selection</title>
<style>
body {{ margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172120; background: #edf3f1; }}
header {{ padding: 18px 22px; background: #fff; border-bottom: 1px solid #d9e2df; display: flex; gap: 16px; justify-content: space-between; align-items: flex-start; }}
h1 {{ margin: 0; font-size: 23px; line-height: 1.15; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 16px 22px 28px; }}
a {{ color: #0f766e; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.subhead {{ color: #64716f; font-size: 12px; margin-top: 5px; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.button {{ border: 1px solid #d9e2df; border-radius: 6px; background: #f7faf9; padding: 7px 10px; font-weight: 760; font-size: 12px; color: #172120; }}
.stats {{ display: grid; grid-template-columns: repeat(5, minmax(135px, 1fr)); gap: 10px; margin-bottom: 14px; }}
.stat, .section {{ background: #fff; border: 1px solid #d9e2df; border-radius: 8px; overflow: hidden; }}
.stat {{ padding: 11px; }}
.label {{ color: #64716f; font-size: 11px; }}
.value {{ margin-top: 3px; font-size: 23px; font-weight: 780; }}
.section {{ margin-bottom: 14px; }}
.section-head {{ padding: 11px 13px; border-bottom: 1px solid #d9e2df; font-size: 14px; font-weight: 760; }}
.table-wrap {{ overflow: auto; max-height: 520px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ border-bottom: 1px solid #e7ecea; padding: 7px 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f6f8f7; color: #44504f; text-transform: uppercase; font-size: 11px; letter-spacing: 0; }}
.badge {{ display: inline-block; min-width: 64px; border-radius: 999px; color: #fff; padding: 2px 8px; font-weight: 760; text-align: center; }}
.calcium {{ background: #2563eb; }}
.wall {{ background: #7c3aed; }}
.peri_fat {{ background: #0f766e; }}
@media (max-width: 760px) {{ header {{ display: block; }} .actions {{ margin-top: 12px; }} .stats {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 13px; padding-right: 13px; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Domain-Balanced Recursive Selection</h1>
    <div class="subhead">Equal-count recursive selection across calcium, wall, and peri-fat variables.</div>
  </div>
  <nav class="actions">
    <a class="button" href="selected_features.csv">Selected Features</a>
    <a class="button" href="selection_trace.csv">Trace</a>
    <a class="button" href="candidate_evaluations.csv">Candidate Evaluations</a>
    <a class="button" href="pca_clustering/pca_clustering_explorer.html">Selected PCA Explorer</a>
  </nav>
</header>
<main>
  <section class="stats" id="stats"></section>
  <section class="section">
    <div class="section-head">Selected Features</div>
    <div class="table-wrap"><table><thead><tr><th>Domain</th><th>Order</th><th>Feature</th><th>Step</th><th>Domain Rank</th><th>Representativeness</th></tr></thead><tbody id="selectedRows"></tbody></table></div>
  </section>
  <section class="section">
    <div class="section-head">Selection Trace</div>
    <div class="table-wrap"><table><thead><tr><th>Step</th><th>Domain</th><th>Feature</th><th>Reason</th><th>Objective</th><th>Silhouette</th><th>Best k</th><th>Redundancy</th></tr></thead><tbody id="traceRows"></tbody></table></div>
  </section>
  <section class="section">
    <div class="section-head">Final KMeans Model Selection</div>
    <div class="table-wrap"><table><thead><tr><th>k</th><th>Status</th><th>Silhouette</th><th>Inertia</th><th>Sizes</th></tr></thead><tbody id="modelRows"></tbody></table></div>
  </section>
  <section class="section">
    <div class="section-head">Subsample Stability</div>
    <div class="table-wrap"><table><thead><tr><th>Repeat</th><th>Sample n</th><th>k</th><th>ARI</th><th>Sizes</th></tr></thead><tbody id="stabilityRows"></tbody></table></div>
  </section>
</main>
<script type="application/json" id="payload">{payload}</script>
<script>
const payload = JSON.parse(document.getElementById("payload").textContent);
const domains = ["calcium", "wall", "peri_fat"];
function byId(id) {{ return document.getElementById(id); }}
function fmt(value, digits = 3) {{
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return number.toLocaleString(undefined, {{ maximumFractionDigits: digits }});
}}
function domainBadge(domain) {{ return `<span class="badge ${{domain}}">${{domain}}</span>`; }}
function renderStats() {{
  const s = payload.summary || {{}};
  const stability = payload.stability || [];
  const ari = stability.length ? stability.reduce((sum, row) => sum + Number(row.adjusted_rand_index || 0), 0) / stability.length : null;
  byId("stats").innerHTML = [
    ["Cases", s.cases, "input rows"],
    ["Selected", s.selected_features, `${{s.features_per_domain}} per domain`],
    ["Usable", s.usable_numeric_features, "numeric features"],
    ["Best k", s.best_k, `silhouette ${{fmt(s.best_silhouette)}}`],
    ["Stability", fmt(ari), "mean ARI"]
  ].map(([label, value, note]) => `<div class="stat"><div class="label">${{label}}</div><div class="value">${{value ?? ""}}</div><div class="label">${{note}}</div></div>`).join("");
}}
function renderTables() {{
  byId("selectedRows").innerHTML = (payload.selected || []).map(row => `<tr><td>${{domainBadge(row.domain)}}</td><td>${{row.domain_selection_order}}</td><td>${{row.feature}}</td><td>${{row.selection_step}}</td><td>${{row.domain_rank}}</td><td>${{fmt(row.domain_pca_representativeness)}}</td></tr>`).join("");
  byId("traceRows").innerHTML = (payload.trace || []).map(row => `<tr><td>${{row.selection_step}}</td><td>${{domainBadge(row.domain)}}</td><td>${{row.feature}}</td><td>${{row.reason}}</td><td>${{fmt(row.objective)}}</td><td>${{fmt(row.silhouette)}}</td><td>${{row.best_k ?? ""}}</td><td>${{fmt(row.redundancy_max_abs_corr)}}</td></tr>`).join("");
  byId("modelRows").innerHTML = (payload.model || []).map(row => `<tr><td>${{row.k}}</td><td>${{row.status}}</td><td>${{fmt(row.silhouette)}}</td><td>${{fmt(row.inertia, 0)}}</td><td>${{row.cluster_sizes || ""}}</td></tr>`).join("");
  byId("stabilityRows").innerHTML = (payload.stability || []).map(row => `<tr><td>${{row.repeat}}</td><td>${{row.sample_n}}</td><td>${{row.best_k}}</td><td>${{fmt(row.adjusted_rand_index)}}</td><td>${{row.subset_cluster_sizes || ""}}</td></tr>`).join("");
}}
renderStats();
renderTables();
</script>
</body>
</html>
"""
    (outdir / "selection_report.html").write_text(html, encoding="utf-8")


def records_from_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    return json.loads(frame.to_json(orient="records"))


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
