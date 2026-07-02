"""Configuration objects for radselect."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


TASKS = ("binary", "multiclass", "regression", "survival", "competing_risk")
PROJECTIONS = ("none", "pca", "pls")
SCREENING_METHODS = ("univariate", "mutual_info")


@dataclass(slots=True)
class RunConfig:
    """Configuration for one radselect run.

    All outcome-driven filtering and model-based selection is fit inside the
    training partition supplied by the outer validation split.
    """

    task: str
    target_column: str | None = None
    time_column: str | None = None
    event_column: str | None = None
    competing_event_code: int | str = 1
    id_column: str | None = None
    group_column: str | None = None
    holdout_groups: list[str] = field(default_factory=list)
    feature_columns: list[str] | None = None
    radiomics_columns: list[str] = field(default_factory=list)
    clinical_columns: list[str] = field(default_factory=list)
    domains: dict[str, list[str]] = field(default_factory=dict)
    max_missing: float = 0.20
    min_variance: float = 1e-8
    min_unique: int = 2
    top_k: int = 30
    screening_method: str = "univariate"
    mutual_info_neighbors: int = 3
    correlation_threshold: float = 0.85
    correlation_method: str = "spearman"
    elastic_net_c: float = 0.2
    elastic_net_alpha: float = 0.01
    elastic_net_l1_ratio: float = 0.5
    tune_elastic_net: bool = True
    inner_splits: int = 3
    elastic_net_c_grid: list[float] = field(default_factory=lambda: [0.05, 0.2, 1.0])
    elastic_net_alpha_grid: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    elastic_net_l1_ratio_grid: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    outer_splits: int = 5
    stability_resamples: int = 100
    stability_train_fraction: float = 0.75
    stability_threshold: float = 0.50
    random_state: int = 13
    feature_metadata_csv: Path | None = None
    require_ibsi_compliant: bool = False
    ibsi_require_listed: bool = False
    robustness_csv: Path | None = None
    robustness_min_icc: float = 0.75
    robustness_require_listed: bool = False
    projection: str = "none"
    projection_components: int = 5

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unsupported task '{self.task}'. Expected one of: {', '.join(TASKS)}.")
        if self.projection not in PROJECTIONS:
            raise ValueError(
                f"Unsupported projection '{self.projection}'. Expected one of: {', '.join(PROJECTIONS)}."
            )
        if self.screening_method not in SCREENING_METHODS:
            raise ValueError(
                f"Unsupported screening_method '{self.screening_method}'. "
                f"Expected one of: {', '.join(SCREENING_METHODS)}."
            )
        if self.task in {"binary", "multiclass", "regression"} and not self.target_column:
            raise ValueError(f"task={self.task} requires target_column.")
        if self.task in {"survival", "competing_risk"} and not (self.time_column and self.event_column):
            raise ValueError(f"task={self.task} requires time_column and event_column.")
        if self.holdout_groups and not self.group_column:
            raise ValueError("holdout_groups requires group_column.")
        if not 0 <= self.max_missing <= 1:
            raise ValueError("max_missing must be between 0 and 1.")
        if self.min_variance < 0:
            raise ValueError("min_variance must be non-negative.")
        if self.min_unique < 1:
            raise ValueError("min_unique must be at least 1.")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1.")
        if self.stability_resamples < 0:
            raise ValueError("stability_resamples must be non-negative.")
        if not 0 < self.stability_train_fraction < 1:
            raise ValueError("stability_train_fraction must be between 0 and 1.")
        if not 0 <= self.stability_threshold <= 1:
            raise ValueError("stability_threshold must be between 0 and 1.")
        if not 0 <= self.robustness_min_icc <= 1:
            raise ValueError("robustness_min_icc must be between 0 and 1.")
        if self.projection_components < 1:
            raise ValueError("projection_components must be at least 1.")
        if self.outer_splits < 2:
            raise ValueError("outer_splits must be at least 2.")
        if self.inner_splits < 2:
            raise ValueError("inner_splits must be at least 2.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.mutual_info_neighbors < 1:
            raise ValueError("mutual_info_neighbors must be at least 1.")
        if self.elastic_net_c <= 0:
            raise ValueError("elastic_net_c must be positive.")
        if self.elastic_net_alpha <= 0:
            raise ValueError("elastic_net_alpha must be positive.")
        if not 0 <= self.elastic_net_l1_ratio <= 1:
            raise ValueError("elastic_net_l1_ratio must be between 0 and 1.")
        if not all(value > 0 for value in self.elastic_net_c_grid):
            raise ValueError("elastic_net_c_grid values must be positive.")
        if not all(value > 0 for value in self.elastic_net_alpha_grid):
            raise ValueError("elastic_net_alpha_grid values must be positive.")
        if not all(0 <= value <= 1 for value in self.elastic_net_l1_ratio_grid):
            raise ValueError("elastic_net_l1_ratio_grid values must be between 0 and 1.")
        if self.require_ibsi_compliant and self.feature_metadata_csv is None:
            raise ValueError("require_ibsi_compliant requires feature_metadata_csv.")
        if self.ibsi_require_listed and not self.require_ibsi_compliant:
            raise ValueError("ibsi_require_listed requires require_ibsi_compliant.")
