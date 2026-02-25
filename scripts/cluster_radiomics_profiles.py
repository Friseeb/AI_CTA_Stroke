#!/usr/bin/env python3
"""Cluster radiomics-derived patient profiles using numpy+pandas only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster patient-level radiomics profiles (k-means, optional auto-k)."
    )
    p.add_argument("--input-csv", required=True, help="Patient-level derived radiomics CSV")
    p.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix path (default: <input_dir>/radiomics_clusters)",
    )
    p.add_argument(
        "--id-column",
        default=None,
        help="Patient identifier column (default: auto from patient_id/case_id/subject_id)",
    )
    p.add_argument(
        "--include-regex",
        action="append",
        default=[],
        help="Keep only feature columns matching regex (repeatable).",
    )
    p.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help="Drop feature columns matching regex (repeatable).",
    )
    p.add_argument(
        "--min-nonnull-ratio",
        type=float,
        default=0.5,
        help="Drop features with lower non-null fraction.",
    )
    p.add_argument("--k", type=int, default=None, help="Fixed k for k-means. If unset, auto-select.")
    p.add_argument("--k-min", type=int, default=2, help="Auto-k minimum.")
    p.add_argument("--k-max", type=int, default=8, help="Auto-k maximum.")
    p.add_argument("--n-init", type=int, default=20, help="K-means random starts per k.")
    p.add_argument("--max-iter", type=int, default=200, help="K-means max iterations.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    return p.parse_args()


def find_id_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--id-column not found: {requested}")
        return requested
    for c in ("patient_id", "case_id", "subject_id"):
        if c in df.columns:
            return c
    raise ValueError("Could not infer ID column. Pass --id-column.")


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    out: list[re.Pattern[str]] = []
    for p in patterns:
        out.append(re.compile(p))
    return out


def pick_feature_columns(
    df: pd.DataFrame,
    id_col: str,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
    min_nonnull_ratio: float,
) -> tuple[list[str], dict[str, str]]:
    reasons: dict[str, str] = {}

    numeric_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    kept: list[str] = []
    for col in numeric_cols:
        if include_patterns and not any(p.search(col) for p in include_patterns):
            reasons[col] = "excluded_by_include_regex"
            continue
        if exclude_patterns and any(p.search(col) for p in exclude_patterns):
            reasons[col] = "excluded_by_exclude_regex"
            continue
        nonnull_ratio = float(df[col].notna().mean())
        if nonnull_ratio < float(min_nonnull_ratio):
            reasons[col] = f"nonnull_ratio<{min_nonnull_ratio}"
            continue
        kept.append(col)
    return kept, reasons


def prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, dict[str, list[str] | int]]:
    if not feature_cols:
        raise ValueError("No numeric feature columns selected for clustering.")

    x = df[feature_cols].copy()
    medians = x.median(numeric_only=True)
    x = x.fillna(medians)

    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    variable_mask = (stds.to_numpy(dtype=float) > 1e-12) & np.isfinite(stds.to_numpy(dtype=float))
    if not variable_mask.any():
        raise ValueError("All selected features are constant after imputation.")

    variable_cols = [c for c, keep in zip(feature_cols, variable_mask, strict=True) if keep]
    dropped_constant = [c for c, keep in zip(feature_cols, variable_mask, strict=True) if not keep]
    x_var = x[variable_cols]
    means = x_var.mean(axis=0)
    stds = x_var.std(axis=0, ddof=0)
    z = (x_var - means) / stds
    return z.to_numpy(dtype=float), {
        "used_features": variable_cols,
        "dropped_constant_features": dropped_constant,
        "n_used_features": len(variable_cols),
    }


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def silhouette_score_from_distance(dmat: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= n:
        return float("nan")

    sil = np.zeros(n, dtype=float)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_count = int(own_mask.sum())
        if own_count <= 1:
            sil[i] = 0.0
            continue
        a = dmat[i, own_mask].sum() / (own_count - 1)

        b = float("inf")
        for c in unique:
            if c == own:
                continue
            mask = labels == c
            if not mask.any():
                continue
            b = min(b, float(dmat[i, mask].mean()))
        if not np.isfinite(b) or max(a, b) == 0:
            sil[i] = 0.0
        else:
            sil[i] = (b - a) / max(a, b)
    return float(np.mean(sil))


def run_kmeans_once(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n, p = x.shape
    if k <= 1 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    idx = rng.choice(n, size=k, replace=False)
    centers = x[idx].copy()
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:
                centers[j] = x[rng.integers(0, n)]

    dist2 = np.sum((x - centers[labels]) ** 2, axis=1)
    inertia = float(dist2.sum())
    return labels, centers, inertia


def run_kmeans_best(
    x: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(n_init):
        labels, centers, inertia = run_kmeans_once(x, k=k, rng=rng, max_iter=max_iter)
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels, centers, inertia
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def pca2_scores(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comps = vt[:2].T
    if comps.shape[1] < 2:
        return np.column_stack([x0 @ comps[:, 0], np.zeros(x.shape[0])])
    return x0 @ comps


def main() -> int:
    args = parse_args()
    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_prefix = Path(args.output_prefix) if args.output_prefix else (in_csv.parent / "radiomics_clusters")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    id_col = find_id_column(df, args.id_column)

    include_patterns = compile_patterns(args.include_regex)
    exclude_patterns = compile_patterns(args.exclude_regex)
    candidate_cols, excluded = pick_feature_columns(
        df=df,
        id_col=id_col,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        min_nonnull_ratio=float(args.min_nonnull_ratio),
    )
    x, matrix_info = prepare_matrix(df, candidate_cols)
    used_features: list[str] = list(matrix_info["used_features"])  # type: ignore[assignment]

    n = x.shape[0]
    dmat = pairwise_distances(x)

    if args.k is not None:
        k_values = [int(args.k)]
    else:
        k_lo = max(2, int(args.k_min))
        k_hi = min(int(args.k_max), n - 1)
        if k_lo > k_hi:
            raise ValueError(f"Invalid auto-k range after bounds: [{k_lo}, {k_hi}] with n={n}")
        k_values = list(range(k_lo, k_hi + 1))

    results: list[dict[str, float | int]] = []
    best: dict[str, object] | None = None
    for k in k_values:
        labels, centers, inertia = run_kmeans_best(
            x=x,
            k=k,
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
            seed=int(args.seed) + k,
        )
        sil = silhouette_score_from_distance(dmat, labels)
        results.append({"k": int(k), "silhouette": float(sil), "inertia": float(inertia)})
        score = sil if np.isfinite(sil) else -np.inf
        if best is None or score > float(best["score"]) or (
            score == float(best["score"]) and k < int(best["k"])
        ):
            best = {
                "k": int(k),
                "labels": labels,
                "centers": centers,
                "inertia": float(inertia),
                "silhouette": float(sil),
                "score": float(score),
            }
    assert best is not None

    labels = np.asarray(best["labels"], dtype=int)
    k_best = int(best["k"])
    pcs = pca2_scores(x)

    patient_cols = [id_col]
    for c in ("phase", "LAA_HU_pattern_paper"):
        if c in df.columns:
            patient_cols.append(c)
    out_patients = df[patient_cols].copy()
    out_patients["cluster"] = labels
    out_patients["cluster_label"] = out_patients["cluster"].map(lambda c: f"C{int(c)}")
    out_patients["pc1"] = pcs[:, 0]
    out_patients["pc2"] = pcs[:, 1]
    out_patients_path = Path(f"{out_prefix}_patients.csv")
    out_patients.to_csv(out_patients_path, index=False)

    summary = out_patients.groupby("cluster", dropna=False).size().rename("n_patients").reset_index()
    feat_means = pd.DataFrame(x, columns=used_features).groupby(labels).mean().reset_index().rename(columns={"index": "cluster"})
    feat_means = feat_means.rename(columns={feat_means.columns[0]: "cluster"})
    out_summary = summary.merge(feat_means, on="cluster", how="left")
    out_summary_path = Path(f"{out_prefix}_summary_means_zscaled.csv")
    out_summary.to_csv(out_summary_path, index=False)

    out_scan = pd.DataFrame(results).sort_values("k")
    out_scan_path = Path(f"{out_prefix}_k_scan.csv")
    out_scan.to_csv(out_scan_path, index=False)

    meta = {
        "input_csv": str(in_csv),
        "id_column": id_col,
        "n_samples": int(n),
        "selected_k": int(k_best),
        "selected_silhouette": float(best["silhouette"]),
        "selected_inertia": float(best["inertia"]),
        "k_candidates": k_values,
        "n_candidate_features": len(candidate_cols),
        "n_used_features": int(matrix_info["n_used_features"]),
        "used_features": used_features,
        "dropped_constant_features": matrix_info["dropped_constant_features"],
        "excluded_features": excluded,
        "outputs": {
            "patients_csv": str(out_patients_path),
            "summary_csv": str(out_summary_path),
            "k_scan_csv": str(out_scan_path),
        },
    }
    out_meta_path = Path(f"{out_prefix}_metadata.json")
    out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Input rows: {n}")
    print(f"Candidate features: {len(candidate_cols)} | used features: {matrix_info['n_used_features']}")
    print(f"Selected k: {k_best} | silhouette: {float(best['silhouette']):.4f} | inertia: {float(best['inertia']):.4f}")
    print(f"Saved: {out_patients_path}")
    print(f"Saved: {out_summary_path}")
    print(f"Saved: {out_scan_path}")
    print(f"Saved: {out_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
