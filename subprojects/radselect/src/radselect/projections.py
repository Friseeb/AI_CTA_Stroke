"""Projection and dimensionality-reduction helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_projection(frame, y, modalities, config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build optional full-dataset projection outputs for visualization.

    Model validation remains foldwise in ``core.py``. These outputs are intended
    for transparent exploration and reporting.
    """

    if config.projection == "none":
        return projection_scores_frame([]), projection_loadings_frame([])
    score_rows = []
    loading_rows = []
    for modality, columns in modalities.items():
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.loc[:, numeric.notna().any(axis=0)]
        if numeric.empty:
            continue
        n_components = min(config.projection_components, numeric.shape[0], numeric.shape[1])
        if n_components < 1:
            continue
        if config.projection == "pca":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n_components))
            scores = model.fit_transform(numeric)
            pca = model.named_steps["pca"]
            loadings = pca.components_.T
            explained = pca.explained_variance_ratio_
        elif config.projection == "pls":
            target = projection_target(y, config)
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                PLSRegression(n_components=n_components, scale=False),
            )
            scores = model.fit_transform(numeric, target)[0]
            pls = model.named_steps["plsregression"]
            loadings = pls.x_loadings_
            explained = np.full(n_components, math.nan)
        else:
            continue
        ids = frame[config.id_column].tolist() if config.id_column and config.id_column in frame.columns else list(frame.index)
        for row_id, values in zip(ids, scores, strict=False):
            row = {"id": row_id, "modality": modality, "projection": config.projection}
            for idx, value in enumerate(values[:n_components], start=1):
                row[f"component_{idx}"] = float(value)
            score_rows.append(row)
        for feature, values in zip(numeric.columns, loadings, strict=False):
            for idx, value in enumerate(values[:n_components], start=1):
                loading_rows.append(
                    {
                        "modality": modality,
                        "projection": config.projection,
                        "feature": feature,
                        "component": idx,
                        "loading": float(value),
                        "explained_variance_ratio": float(explained[idx - 1]),
                    }
                )
    return projection_scores_frame(score_rows), projection_loadings_frame(loading_rows)


def fit_projection_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train,
    columns: list[str],
    config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit PCA/PLS on training data only and transform train/test rows."""

    if config.projection == "none":
        return pd.DataFrame(), pd.DataFrame(), {"status": "disabled", "n_input_features": 0, "n_components": 0}

    available = [column for column in columns if column in train.columns and column in test.columns]
    train_numeric = train[available].apply(pd.to_numeric, errors="coerce")
    test_numeric = test[available].apply(pd.to_numeric, errors="coerce")
    if train_numeric.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_numeric_features", "n_input_features": 0, "n_components": 0}

    missing = train_numeric.isna().mean()
    variances = train_numeric.var(skipna=True)
    uniques = train_numeric.nunique(dropna=True)
    kept = [
        column
        for column in train_numeric.columns
        if missing.get(column, 1.0) <= config.max_missing
        and variances.get(column, 0.0) > config.min_variance
        and uniques.get(column, 0) >= config.min_unique
    ]
    if not kept:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_features_after_filtering", "n_input_features": 0, "n_components": 0}

    n_components = min(config.projection_components, len(kept), len(train_numeric))
    if n_components < 1:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_components", "n_input_features": len(kept), "n_components": 0}

    if config.projection == "pca":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n_components))
        train_scores = model.fit_transform(train_numeric[kept])
        test_scores = model.transform(test_numeric[kept])
    elif config.projection == "pls":
        target = projection_target(y_train, config)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PLSRegression(n_components=n_components, scale=False),
        )
        train_scores = model.fit_transform(train_numeric[kept], target)[0]
        test_scores = model.transform(test_numeric[kept])
    else:
        return pd.DataFrame(), pd.DataFrame(), {"status": f"unsupported_projection:{config.projection}", "n_input_features": len(kept), "n_components": 0}

    feature_names = [f"{config.projection}_component_{idx}" for idx in range(1, n_components + 1)]
    metadata = {"status": "ok", "n_input_features": len(kept), "n_components": n_components}
    return (
        pd.DataFrame(train_scores[:, :n_components], columns=feature_names, index=train.index),
        pd.DataFrame(test_scores[:, :n_components], columns=feature_names, index=test.index),
        metadata,
    )


def build_final_projection(
    frame: pd.DataFrame,
    y,
    modalities,
    config,
    external_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit final PCA/PLS projections on development rows and save transform parameters."""

    if config.projection == "none":
        return final_projection_scores_frame([]), final_projection_parameters_frame([])
    score_rows = []
    parameter_rows = []
    for modality, columns in modalities.items():
        available = [column for column in columns if column in frame.columns]
        train_numeric = frame[available].apply(pd.to_numeric, errors="coerce")
        if train_numeric.empty:
            continue
        kept = projection_feature_filter(train_numeric, config)
        if not kept:
            continue
        n_components = min(config.projection_components, len(kept), len(train_numeric))
        if n_components < 1:
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_imputed = imputer.fit_transform(train_numeric[kept])
        train_scaled = scaler.fit_transform(train_imputed)
        if config.projection == "pca":
            model = PCA(n_components=n_components)
            train_scores = model.fit_transform(train_scaled)
            weights = model.components_.T
            explained = model.explained_variance_ratio_
        elif config.projection == "pls":
            target = projection_target(y, config)
            model = PLSRegression(n_components=n_components, scale=False)
            train_scores = model.fit_transform(train_scaled, target)[0]
            weights = model.x_rotations_
            explained = np.full(n_components, math.nan)
        else:
            continue

        add_projection_score_rows(
            score_rows,
            frame,
            train_scores,
            modality,
            config,
            "final_projection_development",
            n_components,
        )
        if external_data is not None:
            external_numeric = external_data[kept].apply(pd.to_numeric, errors="coerce")
            external_scaled = scaler.transform(imputer.transform(external_numeric))
            external_scores = model.transform(external_scaled)
            add_projection_score_rows(
                score_rows,
                external_data,
                external_scores,
                modality,
                config,
                "final_projection_external",
                n_components,
            )

        medians = imputer.statistics_
        means = scaler.mean_
        scales = scaler.scale_
        for feature_index, feature in enumerate(kept):
            for component_index in range(n_components):
                parameter_rows.append(
                    {
                        "fold": "final_projection",
                        "modality": modality,
                        "projection": config.projection,
                        "feature": feature,
                        "component": component_index + 1,
                        "impute_median": float(medians[feature_index]),
                        "scale_mean": float(means[feature_index]),
                        "scale_std": float(scales[feature_index]) if scales[feature_index] else 1.0,
                        "weight": float(weights[feature_index, component_index]),
                        "explained_variance_ratio": float(explained[component_index]),
                        "n_components": n_components,
                    }
                )
    return final_projection_scores_frame(score_rows), final_projection_parameters_frame(parameter_rows)


def projection_scores_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = ["id", "modality", "projection"]
    return frame.reindex(columns=[*base_columns, *component_columns(frame)])


def projection_loadings_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(
        columns=[
            "modality",
            "projection",
            "feature",
            "component",
            "loading",
            "explained_variance_ratio",
        ]
    )


def final_projection_scores_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = ["id", "fold", "modality", "projection", "n_components"]
    return frame.reindex(columns=[*base_columns, *component_columns(frame)])


def final_projection_parameters_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(
        columns=[
            "fold",
            "modality",
            "projection",
            "feature",
            "component",
            "impute_median",
            "scale_mean",
            "scale_std",
            "weight",
            "explained_variance_ratio",
            "n_components",
        ]
    )


def component_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("component_")]
    return sorted(columns, key=component_column_sort_key) or ["component_1"]


def component_column_sort_key(column: str) -> tuple[int, str]:
    suffix = str(column).removeprefix("component_")
    try:
        return int(suffix), str(column)
    except ValueError:
        return 10**9, str(column)


def projection_feature_filter(frame: pd.DataFrame, config) -> list[str]:
    missing = frame.isna().mean()
    variances = frame.var(skipna=True)
    uniques = frame.nunique(dropna=True)
    return [
        column
        for column in frame.columns
        if missing.get(column, 1.0) <= config.max_missing
        and variances.get(column, 0.0) > config.min_variance
        and uniques.get(column, 0) >= config.min_unique
    ]


def add_projection_score_rows(
    rows: list[dict],
    frame: pd.DataFrame,
    scores: np.ndarray,
    modality: str,
    config,
    fold: str,
    n_components: int,
) -> None:
    ids = frame[config.id_column].tolist() if config.id_column and config.id_column in frame.columns else list(frame.index)
    for row_id, values in zip(ids, scores, strict=False):
        row = {
            "id": row_id,
            "fold": fold,
            "modality": modality,
            "projection": config.projection,
            "n_components": n_components,
        }
        for idx, value in enumerate(values[:n_components], start=1):
            row[f"component_{idx}"] = float(value)
        rows.append(row)


def projection_target(y, config) -> np.ndarray:
    if config.task == "regression":
        return pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).to_numpy(dtype=float)
    if config.task in {"binary", "multiclass"}:
        return LabelEncoder().fit_transform(pd.Series(y).astype(str))
    target = pd.DataFrame(y)
    event = pd.to_numeric(target["event"], errors="coerce").fillna(0).to_numpy(dtype=float)
    time = pd.to_numeric(target["time"], errors="coerce").fillna(target["time"].median()).to_numpy(dtype=float)
    return event / np.maximum(time, 1.0)
