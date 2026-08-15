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
        "--row-filter",
        action="append",
        default=[],
        help="Row filter in the form column=value or column!=value (repeatable).",
    )
    p.add_argument(
        "--drop-duplicate-id",
        action="store_true",
        help="If multiple rows remain per ID after filtering, keep first and drop the rest.",
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
    p.add_argument(
        "--min-variance",
        type=float,
        default=0.0,
        help="Drop features with variance <= this threshold after imputation.",
    )
    p.add_argument(
        "--max-corr",
        type=float,
        default=1.0,
        help="If <1, greedily drop features with abs(correlation) > threshold.",
    )
    p.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="If >0, run k-means on first N PCA components.",
    )
    p.add_argument(
        "--pca-var-ratio",
        type=float,
        default=0.0,
        help="If in (0,1], choose number of PCA components to reach this cumulative variance.",
    )
    p.add_argument("--k", type=int, default=None, help="Fixed k for k-means. If unset, auto-select.")
    p.add_argument("--k-min", type=int, default=2, help="Auto-k minimum.")
    p.add_argument("--k-max", type=int, default=8, help="Auto-k maximum.")
    p.add_argument("--n-init", type=int, default=20, help="K-means random starts per k.")
    p.add_argument("--max-iter", type=int, default=200, help="K-means max iterations.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument(
        "--bootstrap-ari-iters",
        type=int,
        default=0,
        help="If >0, run bootstrap/subsample stability with ARI against full-data labels.",
    )
    p.add_argument(
        "--bootstrap-frac",
        type=float,
        default=0.8,
        help="Subsample fraction for bootstrap ARI stability.",
    )
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


def apply_row_filters(df: pd.DataFrame, filters: list[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = df.copy()
    audit: list[dict[str, object]] = []
    for expr in filters:
        m = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*(!=|=)\s*(.+?)\s*", expr)
        if m is None:
            raise ValueError(f"Invalid --row-filter '{expr}'. Use column=value or column!=value.")
        col, op, rhs = m.group(1), m.group(2), m.group(3)
        if col not in out.columns:
            raise ValueError(f"--row-filter column not found: {col}")
        before = int(len(out))
        series = out[col].astype(str)
        if op == "=":
            out = out[series.eq(rhs)].copy()
        else:
            out = out[series.ne(rhs)].copy()
        audit.append({"filter": expr, "rows_before": before, "rows_after": int(len(out))})
    return out, audit


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


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_variance: float = 0.0,
    max_corr: float = 1.0,
) -> tuple[np.ndarray, dict[str, list[str] | int]]:
    if not feature_cols:
        raise ValueError("No numeric feature columns selected for clustering.")

    x = df[feature_cols].copy()
    medians = x.median(numeric_only=True)
    x = x.fillna(medians)

    variances = x.var(axis=0, ddof=0).to_numpy(dtype=float)
    variable_mask = np.isfinite(variances) & (variances > max(float(min_variance), 1e-12))
    if not variable_mask.any():
        raise ValueError("All selected features are constant/low-variance after imputation.")

    variable_cols = [c for c, keep in zip(feature_cols, variable_mask, strict=True) if keep]
    dropped_low_variance = [c for c, keep in zip(feature_cols, variable_mask, strict=True) if not keep]
    x_var = x[variable_cols]

    dropped_correlated: list[str] = []
    corr_kept_cols = variable_cols
    if 0.0 < float(max_corr) < 1.0 and len(variable_cols) > 1:
        corr = x_var.corr().to_numpy(dtype=float)
        keep_mask = np.ones(len(variable_cols), dtype=bool)
        for i in range(len(variable_cols)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(variable_cols)):
                if not keep_mask[j]:
                    continue
                cij = corr[i, j]
                if np.isfinite(cij) and abs(cij) > float(max_corr):
                    keep_mask[j] = False
        corr_kept_cols = [c for c, keep in zip(variable_cols, keep_mask, strict=True) if keep]
        dropped_correlated = [c for c, keep in zip(variable_cols, keep_mask, strict=True) if not keep]
        x_var = x_var[corr_kept_cols]

    means = x_var.mean(axis=0)
    stds = x_var.std(axis=0, ddof=0)
    z = (x_var - means) / stds
    return z.to_numpy(dtype=float), {
        "used_features": corr_kept_cols,
        "dropped_low_variance_features": dropped_low_variance,
        "dropped_correlated_features": dropped_correlated,
        "n_used_features": len(corr_kept_cols),
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


def pca_project(
    x: np.ndarray,
    pca_components: int = 0,
    pca_var_ratio: float = 0.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if pca_components <= 0 and not (0.0 < pca_var_ratio <= 1.0):
        return x, {"applied": False, "n_components": int(x.shape[1]), "explained_variance_ratio_sum": 1.0}

    x0 = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    if s.size == 0:
        raise ValueError("PCA failed: empty singular value set.")
    eig = (s ** 2) / max(1, x.shape[0] - 1)
    total = float(eig.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PCA failed: non-positive explained variance.")
    ratio = eig / total
    cum = np.cumsum(ratio)

    if pca_components > 0:
        n_comp = min(int(pca_components), x.shape[1], vt.shape[0])
    else:
        n_comp = int(np.searchsorted(cum, float(pca_var_ratio), side="left") + 1)
        n_comp = min(max(1, n_comp), x.shape[1], vt.shape[0])

    comps = vt[:n_comp].T
    x_pca = x0 @ comps
    return x_pca, {
        "applied": True,
        "n_components": int(n_comp),
        "explained_variance_ratio_sum": float(cum[n_comp - 1]),
    }


def _comb2(n: int) -> float:
    return float(n * (n - 1) / 2.0)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    la = np.asarray(labels_a, dtype=int)
    lb = np.asarray(labels_b, dtype=int)
    if la.shape[0] != lb.shape[0]:
        raise ValueError("ARI requires same-length label vectors.")
    n = la.shape[0]
    if n < 2:
        return float("nan")

    _, ia = np.unique(la, return_inverse=True)
    _, ib = np.unique(lb, return_inverse=True)
    na = int(ia.max() + 1)
    nb = int(ib.max() + 1)

    contingency = np.zeros((na, nb), dtype=np.int64)
    np.add.at(contingency, (ia, ib), 1)

    sum_comb_c = float(np.sum(contingency * (contingency - 1) / 2.0))
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    sum_comb_a = float(np.sum(a * (a - 1) / 2.0))
    sum_comb_b = float(np.sum(b * (b - 1) / 2.0))
    comb_n = _comb2(n)
    if comb_n <= 0:
        return float("nan")

    expected = (sum_comb_a * sum_comb_b) / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_comb_c - expected) / denom)


def bootstrap_stability_ari(
    x_model: np.ndarray,
    labels_full: np.ndarray,
    k: int,
    n_iter: int,
    frac: float,
    n_init: int,
    max_iter: int,
    seed: int,
) -> dict[str, object]:
    n = x_model.shape[0]
    if n_iter <= 0:
        return {"enabled": False}
    if not (0.0 < frac <= 1.0):
        raise ValueError("--bootstrap-frac must be in (0, 1].")
    m = int(round(frac * n))
    m = max(m, 2 * int(k), 10)
    m = min(m, n)

    rng = np.random.default_rng(seed)
    aris: list[float] = []
    for i in range(int(n_iter)):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        x_sub = x_model[idx]
        ref = labels_full[idx]
        pred, _, _ = run_kmeans_best(
            x=x_sub,
            k=int(k),
            n_init=int(n_init),
            max_iter=int(max_iter),
            seed=int(seed) + 10000 + i,
        )
        aris.append(adjusted_rand_index(ref, pred))

    arr = np.asarray(aris, dtype=float)
    return {
        "enabled": True,
        "n_iter": int(n_iter),
        "subsample_fraction": float(frac),
        "subsample_n": int(m),
        "ari_mean": float(np.nanmean(arr)),
        "ari_std": float(np.nanstd(arr, ddof=0)),
        "ari_min": float(np.nanmin(arr)),
        "ari_p05": float(np.nanpercentile(arr, 5)),
        "ari_median": float(np.nanpercentile(arr, 50)),
        "ari_p95": float(np.nanpercentile(arr, 95)),
        "ari_max": float(np.nanmax(arr)),
        "ari_values": [float(v) for v in arr],
    }


def main() -> int:
    args = parse_args()
    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_prefix = Path(args.output_prefix) if args.output_prefix else (in_csv.parent / "radiomics_clusters")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_csv)
    n_rows_raw = int(len(df_raw))
    df, row_filter_audit = apply_row_filters(df_raw, args.row_filter)
    if df.empty:
        raise ValueError("No rows remain after --row-filter.")

    id_col = find_id_column(df, args.id_column)
    duplicated_ids = int(df[id_col].duplicated(keep=False).sum())
    if duplicated_ids > 0:
        if args.drop_duplicate_id:
            df = df.drop_duplicates(subset=[id_col], keep="first").copy()
        else:
            raise ValueError(
                f"Found {duplicated_ids} rows with duplicated IDs in '{id_col}'. "
                "Use --row-filter to isolate one row per case or pass --drop-duplicate-id."
            )

    include_patterns = compile_patterns(args.include_regex)
    exclude_patterns = compile_patterns(args.exclude_regex)
    candidate_cols, excluded = pick_feature_columns(
        df=df,
        id_col=id_col,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        min_nonnull_ratio=float(args.min_nonnull_ratio),
    )
    x_report, matrix_info = prepare_matrix(
        df,
        candidate_cols,
        min_variance=float(args.min_variance),
        max_corr=float(args.max_corr),
    )
    used_features: list[str] = list(matrix_info["used_features"])  # type: ignore[assignment]
    x_model, pca_info = pca_project(
        x_report,
        pca_components=int(args.pca_components),
        pca_var_ratio=float(args.pca_var_ratio),
    )

    n = x_model.shape[0]
    dmat = pairwise_distances(x_model)

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
            x=x_model,
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
    pcs = pca2_scores(x_model)

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
    feat_means = (
        pd.DataFrame(x_report, columns=used_features).groupby(labels).mean().reset_index().rename(columns={"index": "cluster"})
    )
    feat_means = feat_means.rename(columns={feat_means.columns[0]: "cluster"})
    out_summary = summary.merge(feat_means, on="cluster", how="left")
    out_summary_path = Path(f"{out_prefix}_summary_means_zscaled.csv")
    out_summary.to_csv(out_summary_path, index=False)

    out_scan = pd.DataFrame(results).sort_values("k")
    out_scan_path = Path(f"{out_prefix}_k_scan.csv")
    out_scan.to_csv(out_scan_path, index=False)

    stability = bootstrap_stability_ari(
        x_model=x_model,
        labels_full=labels,
        k=k_best,
        n_iter=int(args.bootstrap_ari_iters),
        frac=float(args.bootstrap_frac),
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        seed=int(args.seed),
    )

    meta = {
        "input_csv": str(in_csv),
        "id_column": id_col,
        "n_rows_raw": n_rows_raw,
        "n_rows_after_filters": int(len(df)),
        "row_filters": row_filter_audit,
        "drop_duplicate_id": bool(args.drop_duplicate_id),
        "duplicated_id_rows_detected": duplicated_ids,
        "n_samples": int(n),
        "selected_k": int(k_best),
        "selected_silhouette": float(best["silhouette"]),
        "selected_inertia": float(best["inertia"]),
        "k_candidates": k_values,
        "n_candidate_features": len(candidate_cols),
        "n_used_features": int(matrix_info["n_used_features"]),
        "used_features": used_features,
        "dropped_low_variance_features": matrix_info["dropped_low_variance_features"],
        "dropped_correlated_features": matrix_info["dropped_correlated_features"],
        "excluded_features": excluded,
        "preprocess": {
            "min_nonnull_ratio": float(args.min_nonnull_ratio),
            "min_variance": float(args.min_variance),
            "max_corr": float(args.max_corr),
            "pca_components_arg": int(args.pca_components),
            "pca_var_ratio_arg": float(args.pca_var_ratio),
            "pca": pca_info,
            "bootstrap_stability": stability,
        },
        "outputs": {
            "patients_csv": str(out_patients_path),
            "summary_csv": str(out_summary_path),
            "k_scan_csv": str(out_scan_path),
        },
    }
    out_meta_path = Path(f"{out_prefix}_metadata.json")
    out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Input rows (raw): {n_rows_raw} | after filters/dedup: {len(df)}")
    print(f"Candidate features: {len(candidate_cols)} | used features: {matrix_info['n_used_features']}")
    if bool(pca_info.get("applied", False)):
        print(
            "PCA applied: "
            f"n_components={pca_info['n_components']} | "
            f"cum_explained_var={float(pca_info['explained_variance_ratio_sum']):.4f}"
        )
    print(f"Selected k: {k_best} | silhouette: {float(best['silhouette']):.4f} | inertia: {float(best['inertia']):.4f}")
    if bool(stability.get("enabled", False)):
        print(
            "Bootstrap ARI: "
            f"mean={float(stability['ari_mean']):.4f} | "
            f"std={float(stability['ari_std']):.4f} | "
            f"p05={float(stability['ari_p05']):.4f} | "
            f"p95={float(stability['ari_p95']):.4f}"
        )
    print(f"Saved: {out_patients_path}")
    print(f"Saved: {out_summary_path}")
    print(f"Saved: {out_scan_path}")
    print(f"Saved: {out_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
