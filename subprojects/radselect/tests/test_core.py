from __future__ import annotations

import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from radselect import RunConfig, apply_composite_score_parameters, apply_projection_parameters, run_selection
from radselect.cli import build_parser, effective_run_settings, main as cli_main
from radselect.core import parse_requirement_record


def test_binary_selection_is_foldwise_and_outputs_stability():
    rng = np.random.default_rng(7)
    n = 90
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.8, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(n)],
            "mace": y,
            "center": np.repeat(["a", "b", "c"], n // 3),
            "rad_signal": signal,
            "rad_signal_copy": signal + rng.normal(scale=0.001, size=n),
            "rad_noise": rng.normal(size=n),
            "clinical_age": 65 + rng.normal(size=n),
            "constant": 1.0,
        }
    )
    config = RunConfig(
        task="binary",
        target_column="mace",
        id_column="case_id",
        group_column="center",
        feature_columns=["rad_signal", "rad_signal_copy", "rad_noise", "clinical_age", "constant"],
        radiomics_columns=["rad_signal", "rad_signal_copy", "rad_noise"],
        clinical_columns=["clinical_age", "constant"],
        outer_splits=3,
        stability_resamples=5,
        top_k=4,
        tune_elastic_net=False,
        random_state=11,
    )
    result = run_selection(frame, config)

    assert not result.selected_features.empty
    assert not result.modality_audit.empty
    included_memberships = result.modality_audit[result.modality_audit["included_in_modality"]]
    assert {("radiomics", "rad_signal"), ("clinical", "clinical_age"), ("combined", "rad_signal")}.issubset(
        set(zip(included_memberships["modality"], included_memberships["feature"], strict=False))
    )
    assert result.manifest["modality_audit"]["included_memberships"] >= 1
    assert not result.correlation_audit.empty
    redundant_pairs = {
        frozenset({row.kept_feature, row.dropped_feature})
        for row in result.correlation_audit.itertuples(index=False)
    }
    assert frozenset({"rad_signal", "rad_signal_copy"}) in redundant_pairs
    assert {"kept_feature", "dropped_feature", "abs_correlation", "threshold", "method"}.issubset(
        result.correlation_audit.columns
    )
    assert not result.performance.empty
    assert set(result.performance["fold"]) == {"group_fold_1", "group_fold_2", "group_fold_3"}
    assert "rad_signal" in set(result.stability_selection["feature"])
    assert not result.stability_resamples.empty
    assert {
        "sampling_unit",
        "n_train_groups",
        "train_groups",
        "train_row_indices",
        "train_ids",
        "selected_features",
        "model_status",
    }.issubset(result.stability_resamples.columns)
    assert result.stability_resamples["resample"].nunique() == 5
    assert set(result.stability_resamples["sampling_unit"]) == {"group"}
    assert set(result.stability_resamples["n_train_groups"]) == {2}
    for row in result.stability_resamples.itertuples(index=False):
        selected_centers = set(str(row.train_groups).split(";"))
        selected_ids = set(str(row.train_ids).split(";"))
        expected_ids = set(frame.loc[frame["center"].isin(selected_centers), "case_id"])
        assert selected_ids == expected_ids
    assert result.manifest["stability_analysis"]["sampling_unit"] == "group"
    assert result.manifest["stability_analysis"]["group_column"] == "center"
    assert "constant" in set(result.dropped_features["feature"])
    assert not result.composite_scores.empty
    assert set(result.composite_scores["fold"]) == {"group_fold_1", "group_fold_2", "group_fold_3"}
    assert np.isfinite(result.composite_scores["composite_score"]).all()
    assert not result.final_signature.empty
    assert set(result.final_signature["fold"]) == {"final_refit"}
    assert not result.final_signature_parameters.empty
    assert {"median", "mean", "std", "weight"}.issubset(result.final_signature_parameters.columns)
    assert not result.final_composite_scores.empty
    assert set(result.final_composite_scores["fold"]) == {"final_refit_development"}


def test_cli_parser_exposes_run_score_and_project_commands():
    parser = build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--input",
            "features.csv",
            "--outdir",
            "out",
            "--task",
            "binary",
            "--target",
            "mace",
        ]
    )
    assert run_args.command == "run"
    assert run_args.input.name == "features.csv"

    score_args = parser.parse_args(
        [
            "score",
            "--input",
            "features.csv",
            "--parameters",
            "final_signature_parameters.csv",
            "--output",
            "scores.csv",
        ]
    )
    assert score_args.command == "score"
    assert score_args.parameters.name == "final_signature_parameters.csv"

    project_args = parser.parse_args(
        [
            "project",
            "--input",
            "features.csv",
            "--parameters",
            "final_projection_parameters.csv",
            "--output",
            "projection_scores.csv",
        ]
    )
    assert project_args.command == "project"
    assert project_args.parameters.name == "final_projection_parameters.csv"


def test_dependency_requirement_parser_keeps_extras_out_of_runtime_scope():
    runtime = parse_requirement_record("scikit-learn>=1.3")
    reports = parse_requirement_record('matplotlib>=3.7; extra == "reports"')
    survival = parse_requirement_record("lifelines>=0.28; extra == 'survival'")
    dev = parse_requirement_record('pytest>=8.0; extra == "dev"')

    assert runtime == {"package": "scikit-learn", "scope": "runtime_required", "extra": ""}
    assert reports == {"package": "matplotlib", "scope": "optional_extra", "extra": "reports"}
    assert survival == {"package": "lifelines", "scope": "optional_extra", "extra": "survival"}
    assert dev == {"package": "pytest", "scope": "dev_extra", "extra": "dev"}


def test_json_config_values_are_not_overridden_by_unprovided_cli_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "input": "features.csv",
                "outdir": "out",
                "task": "competing_risk",
                "time_column": "time",
                "event_column": "event",
                "competing_event_code": "2",
                "require_ibsi_compliant": True,
                "ibsi_require_listed": True,
                "feature_metadata_csv": "feature_metadata.csv",
                "robustness_require_listed": True,
                "projection": "pls",
                "tune_elastic_net": False,
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["run", "--config", str(config_path)])
    settings = effective_run_settings(args)

    assert settings["competing_event_code"] == "2"
    assert settings["require_ibsi_compliant"] is True
    assert settings["ibsi_require_listed"] is True
    assert settings["robustness_require_listed"] is True
    assert settings["projection"] == "pls"
    assert settings["tune_elastic_net"] is False

    enabled_args = build_parser().parse_args(["run", "--config", str(config_path), "--tune-elastic-net"])
    enabled_settings = effective_run_settings(enabled_args)
    assert enabled_settings["tune_elastic_net"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_variance", -0.1, "min_variance must be non-negative"),
        ("min_unique", 0, "min_unique must be at least 1"),
        ("correlation_threshold", 1.5, "correlation_threshold must be between 0 and 1"),
        ("stability_resamples", -1, "stability_resamples must be non-negative"),
        ("stability_threshold", 1.2, "stability_threshold must be between 0 and 1"),
        ("robustness_min_icc", -0.1, "robustness_min_icc must be between 0 and 1"),
        ("projection_components", 0, "projection_components must be at least 1"),
    ],
)
def test_run_config_rejects_invalid_numeric_controls(field, value, message):
    config = RunConfig(
        task="binary",
        target_column="target",
        feature_columns=["rad_x"],
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_regression_selection_runs():
    rng = np.random.default_rng(9)
    n = 80
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": 2.5 * x + rng.normal(scale=0.5, size=n),
            "rad_x": x,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="regression",
        target_column="target",
        id_column="id",
        feature_columns=["rad_x", "rad_noise"],
        outer_splits=4,
        stability_resamples=3,
        tune_elastic_net=False,
        random_state=3,
    )
    result = run_selection(frame, config)

    assert not result.performance.empty
    assert "r2" in result.performance.columns
    assert not result.selected_features.empty


def test_empty_final_signature_artifacts_keep_readable_headers(tmp_path):
    n = 48
    frame = pd.DataFrame(
        {
            "id": [f"case_{idx}" for idx in range(n)],
            "target": np.tile([0, 1], n // 2),
            "constant": np.nan,
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["constant"],
        outer_splits=3,
        stability_resamples=0,
        tune_elastic_net=False,
        projection="pca",
        projection_components=2,
        random_state=30,
    )

    result = run_selection(frame, config)
    result.write(tmp_path)

    final_parameters = pd.read_csv(tmp_path / "final_signature_parameters.csv")
    final_scores = pd.read_csv(tmp_path / "final_composite_scores.csv")
    composite_scores = pd.read_csv(tmp_path / "composite_scores.csv")
    selected_features = pd.read_csv(tmp_path / "selected_features.csv")
    projection_predictions = pd.read_csv(tmp_path / "projection_predictions.csv")
    projection_scores = pd.read_csv(tmp_path / "projection_scores.csv")
    projection_loadings = pd.read_csv(tmp_path / "projection_loadings.csv")
    final_projection_scores = pd.read_csv(tmp_path / "final_projection_scores.csv")
    final_projection_parameters = pd.read_csv(tmp_path / "final_projection_parameters.csv")

    assert final_parameters.empty
    assert {"fold", "modality", "task", "feature", "median", "mean", "std", "weight"}.issubset(
        final_parameters.columns
    )
    assert final_scores.empty
    assert {"id", "fold", "modality", "task", "composite_score", "features"}.issubset(final_scores.columns)
    assert composite_scores.empty
    assert {"id", "fold", "modality", "task", "composite_score", "features"}.issubset(composite_scores.columns)
    assert selected_features.empty
    assert {"modality", "fold", "feature", "rank", "relevance", "stage"}.issubset(selected_features.columns)
    assert projection_predictions.empty
    assert {"id", "fold", "modality", "projection", "y_true", "prediction"}.issubset(
        projection_predictions.columns
    )
    assert projection_scores.empty
    assert {"id", "modality", "projection", "component_1"}.issubset(projection_scores.columns)
    assert projection_loadings.empty
    assert {"modality", "projection", "feature", "component", "loading"}.issubset(projection_loadings.columns)
    assert final_projection_scores.empty
    assert {"id", "fold", "modality", "projection", "n_components", "component_1"}.issubset(
        final_projection_scores.columns
    )
    assert final_projection_parameters.empty
    assert {"feature", "component", "impute_median", "scale_mean", "scale_std", "weight"}.issubset(
        final_projection_parameters.columns
    )


def test_invalid_outcome_rows_are_dropped_and_audited():
    rng = np.random.default_rng(22)
    n = 72
    signal = rng.normal(size=n)
    target = (signal + rng.normal(scale=0.6, size=n) > 0).astype(float)
    target[[3, 11]] = np.nan
    frame = pd.DataFrame(
        {
            "id": [f"case_{i}" for i in range(n)],
            "target": target,
            "rad_signal": signal,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        random_state=22,
    )

    result = run_selection(frame, config)

    dropped = result.sample_audit[result.sample_audit["status"] == "dropped"]
    assert len(dropped) == 2
    assert set(dropped["reason"]) == {"missing_outcome"}
    assert result.manifest["n_rows_raw"] == n
    assert result.manifest["n_rows"] == n - 2
    assert result.manifest["sample_audit"]["development_rows_dropped"] == 2
    assert "nan" not in set(result.predictions["y_true"].astype(str))


def test_binary_outer_validation_requires_two_rows_per_class():
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "target": [0, 0, 1],
            "rad_signal": [0.1, 0.2, 0.9],
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal"],
        outer_splits=3,
        stability_resamples=0,
        tune_elastic_net=False,
    )

    with pytest.raises(ValueError, match="Each outcome class needs at least 2 rows"):
        run_selection(frame, config)


def test_classification_outer_validation_requires_two_classes():
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "target": [1, 1],
            "rad_signal": [0.1, 0.2],
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal"],
        outer_splits=2,
        stability_resamples=0,
        tune_elastic_net=False,
    )

    with pytest.raises(ValueError, match="Classification tasks require at least 2 outcome classes"):
        run_selection(frame, config)


def test_regression_outer_validation_requires_two_development_rows():
    frame = pd.DataFrame(
        {
            "id": ["a"],
            "target": [1.5],
            "rad_signal": [0.1],
        }
    )
    config = RunConfig(
        task="regression",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal"],
        outer_splits=2,
        stability_resamples=0,
        tune_elastic_net=False,
    )

    with pytest.raises(ValueError, match="At least 2 development rows are required"):
        run_selection(frame, config)


def test_survival_outer_validation_requires_two_development_rows():
    frame = pd.DataFrame(
        {
            "id": ["a"],
            "time": [12.0],
            "event": [1],
            "rad_signal": [0.1],
        }
    )
    config = RunConfig(
        task="survival",
        time_column="time",
        event_column="event",
        id_column="id",
        feature_columns=["rad_signal"],
        outer_splits=2,
        stability_resamples=0,
        tune_elastic_net=False,
    )

    with pytest.raises(ValueError, match="At least 2 development rows are required"):
        run_selection(frame, config)


def test_robustness_filter_removes_failed_features(tmp_path):
    rng = np.random.default_rng(10)
    n = 70
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": y,
            "rad_good": rng.normal(size=n),
            "rad_failed": x,
        }
    )
    robustness = pd.DataFrame(
        {
            "feature": ["rad_failed"],
            "robust": [False],
        }
    )
    robustness_path = tmp_path / "robustness.csv"
    robustness.to_csv(robustness_path, index=False)
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_good", "rad_failed"],
        outer_splits=3,
        stability_resamples=2,
        tune_elastic_net=False,
        robustness_csv=robustness_path,
        random_state=2,
    )

    result = run_selection(frame, config)

    assert "rad_failed" not in set(result.selected_features["feature"])
    assert "rad_failed" not in set(result.stability_selection["feature"])
    assert result.manifest["robustness"]["rejected_features"] == 1
    audit = result.robustness_audit.set_index("feature")
    assert audit.loc["rad_failed", "filter_decision"] == "rejected"
    assert audit.loc["rad_failed", "bool_column"] == "robust"


def test_robustness_filter_audits_multiple_robustness_axes(tmp_path):
    rng = np.random.default_rng(25)
    n = 72
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": (signal + rng.normal(scale=0.6, size=n) > 0).astype(int),
            "rad_stable": signal,
            "rad_segmentation_unstable": rng.normal(size=n),
            "rad_acquisition_unstable": rng.normal(size=n),
        }
    )
    robustness = pd.DataFrame(
        {
            "feature": ["rad_stable", "rad_segmentation_unstable", "rad_acquisition_unstable"],
            "test_retest_icc": [0.91, 0.94, 0.93],
            "segmentation_icc": [0.88, 0.61, 0.86],
            "acquisition_icc": [0.89, 0.86, 0.52],
        }
    )
    robustness_path = tmp_path / "robustness_axes.csv"
    robustness.to_csv(robustness_path, index=False)
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_stable", "rad_segmentation_unstable", "rad_acquisition_unstable"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        robustness_csv=robustness_path,
        robustness_min_icc=0.75,
        random_state=25,
    )

    result = run_selection(frame, config)

    assert "rad_stable" in set(result.stability_selection["feature"])
    assert "rad_segmentation_unstable" not in set(result.stability_selection["feature"])
    assert "rad_acquisition_unstable" not in set(result.stability_selection["feature"])
    audit = result.robustness_audit.set_index("feature")
    assert audit.loc["rad_stable", "filter_decision"] == "retained"
    assert audit.loc["rad_segmentation_unstable", "filter_decision"] == "rejected"
    assert audit.loc["rad_segmentation_unstable", "min_robustness_score"] == pytest.approx(0.61)
    assert audit.loc["rad_acquisition_unstable", "min_robustness_score"] == pytest.approx(0.52)
    assert audit.loc["rad_stable", "robustness_columns"] == "test_retest_icc;segmentation_icc;acquisition_icc"


def test_feature_metadata_can_enforce_ibsi_compliance(tmp_path):
    rng = np.random.default_rng(21)
    n = 78
    good = rng.normal(size=n)
    bad = good + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": (good + rng.normal(scale=0.6, size=n) > 0).astype(int),
            "rad_ibsi": good,
            "rad_non_ibsi": bad,
            "clinical_age": 68 + rng.normal(size=n),
        }
    )
    metadata = pd.DataFrame(
        {
            "feature": ["rad_ibsi", "rad_non_ibsi"],
            "ibsi_compliant": [True, False],
        }
    )
    metadata_path = tmp_path / "feature_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_ibsi", "rad_non_ibsi", "clinical_age"],
        radiomics_columns=["rad_ibsi", "rad_non_ibsi"],
        clinical_columns=["clinical_age"],
        feature_metadata_csv=metadata_path,
        require_ibsi_compliant=True,
        outer_splits=3,
        stability_resamples=2,
        tune_elastic_net=False,
        random_state=21,
    )

    result = run_selection(frame, config)

    assert "rad_non_ibsi" not in set(result.selected_features["feature"])
    assert "rad_non_ibsi" not in set(result.stability_selection["feature"])
    assert result.manifest["feature_metadata"]["status"] == "applied"
    assert result.manifest["feature_metadata"]["rejected_features"] == 1
    audit = result.feature_metadata_audit.set_index("feature")
    assert audit.loc["rad_non_ibsi", "filter_decision"] == "rejected"


def test_apply_composite_score_parameters_requires_signature_features():
    frame = pd.DataFrame({"id": ["case_1"], "rad_present": [1.2]})
    parameters = pd.DataFrame(
        {
            "fold": ["final_refit"],
            "modality": ["radiomics"],
            "task": ["binary"],
            "feature": ["rad_missing"],
            "median": [0.0],
            "mean": [0.0],
            "std": [1.0],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing features required by the signature: rad_missing"):
        apply_composite_score_parameters(frame, parameters, id_column="id")


def test_apply_projection_parameters_requires_signature_features():
    frame = pd.DataFrame({"id": ["case_1"], "rad_present": [1.2]})
    parameters = pd.DataFrame(
        {
            "fold": ["final_projection"],
            "modality": ["radiomics"],
            "projection": ["pca"],
            "feature": ["rad_missing"],
            "component": [1],
            "impute_median": [0.0],
            "scale_mean": [0.0],
            "scale_std": [1.0],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing features required by the projection signature: rad_missing"):
        apply_projection_parameters(frame, parameters, id_column="id")


def test_pls_final_projection_parameters_reproduce_scores():
    rng = np.random.default_rng(27)
    n = 72
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": [f"case_{idx}" for idx in range(n)],
            "target": (signal + rng.normal(scale=0.7, size=n) > 0).astype(int),
            "rad_signal": signal,
            "rad_signal_shifted": 0.6 * signal + rng.normal(scale=0.3, size=n),
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_signal_shifted", "rad_noise"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        projection="pls",
        projection_components=2,
        random_state=27,
    )

    result = run_selection(frame, config)
    applied = apply_projection_parameters(frame, result.final_projection_parameters, id_column="id")

    component_columns = [column for column in result.final_projection_scores.columns if column.startswith("component_")]
    expected = result.final_projection_scores[["id", "modality", "projection", *component_columns]].sort_values(
        ["id", "modality"]
    )
    observed = applied[["id", "modality", "projection", *component_columns]].sort_values(["id", "modality"])
    assert set(result.final_projection_parameters["projection"]) == {"pls"}
    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cli_writes_expected_artifacts(tmp_path):
    rng = np.random.default_rng(12)
    n = 60
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.7, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(n)],
            "center": np.repeat(["site_a", "site_b", "site_c"], n // 3),
            "mace": y,
            "rad_texture": x,
            "rad_noise": rng.normal(size=n),
            "clinical_age": 70 + rng.normal(size=n),
        }
    )
    input_path = tmp_path / "input.csv"
    outdir = tmp_path / "out"
    frame.to_csv(input_path, index=False)

    cli_main(
        [
            "run",
            "--input",
            str(input_path),
            "--outdir",
            str(outdir),
            "--task",
            "binary",
            "--target",
            "mace",
            "--id-column",
            "case_id",
            "--group-column",
            "center",
            "--holdout-group",
            "site_c",
            "--feature-regex",
            "^(rad_|clinical_)",
            "--radiomics-regex",
            "^rad_",
            "--clinical-regex",
            "^clinical_",
            "--outer-splits",
            "3",
            "--stability-resamples",
            "2",
            "--no-tune-elastic-net",
            "--projection",
            "pca",
            "--projection-components",
            "2",
        ]
    )

    expected = {
        "selected_features.csv",
        "column_audit.csv",
        "modality_audit.csv",
        "correlation_audit.csv",
        "schema_audit.csv",
        "robustness_audit.csv",
        "feature_metadata_audit.csv",
        "sample_audit.csv",
        "quality_checks.csv",
        "dependency_audit.csv",
        "dropped_features.csv",
        "selected_feature_frequency.csv",
        "validation_splits.csv",
        "stability_selection.csv",
        "stability_resamples.csv",
        "performance.csv",
        "predictions.csv",
        "composite_scores.csv",
        "final_signature.csv",
        "final_signature_parameters.csv",
        "final_composite_scores.csv",
        "projection_performance.csv",
        "projection_predictions.csv",
        "projection_scores.csv",
        "projection_loadings.csv",
        "final_projection_scores.csv",
        "final_projection_parameters.csv",
        "tuning_summary.csv",
        "effective_config.json",
        "manifest.json",
        "provenance.json",
        "output_manifest.json",
        "run_invocation.json",
        "radselect_report.html",
    }
    assert expected.issubset({path.name for path in outdir.iterdir()})
    provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["runtime_note"].endswith("has no LLM/OpenAI runtime dependency.")
    assert provenance["dependency_audit"]["blocked_runtime_dependencies"] == 0
    report = (outdir / "radselect_report.html").read_text(encoding="utf-8")
    assert "Feature frequency" in report
    assert "Figures" in report
    assert "Column audit" in report
    assert "Modality audit" in report
    assert "Correlation redundancy audit" in report
    assert "Feature metadata" in report
    assert "Robustness audit" in report
    assert "Sample audit" in report
    assert "Schema audit" in report
    assert "Analysis method" in report
    assert "Outcome summary" in report
    assert "Quality checks" in report
    assert "Dependency audit" in report
    assert "Projection validation" in report
    assert "Composite scores" in report
    assert "Final refit signature" in report
    assert "Final refit parameters" in report
    assert "Final refit scores" in report
    assert "Final projection scores" in report
    assert "Final projection parameters" in report
    assert "Validation splits" in report
    assert "Stability resamples" in report
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout_validation"]["enabled"]
    assert manifest["holdout_validation"]["groups"] == ["site_c"]
    assert manifest["schema_audit"]["external_issues"] == 0
    assert manifest["sample_audit"]["external_rows_retained"] == n // 3
    assert manifest["composite_scores"]["rows"] > 0
    assert manifest["final_signature"]["rows"] > 0
    assert manifest["final_signature"]["parameter_rows"] > 0
    assert manifest["final_signature"]["score_rows"] > 0
    assert manifest["projection_validation"]["enabled"]
    assert manifest["projection_validation"]["rows"] > 0
    assert manifest["final_projection"]["enabled"]
    assert manifest["final_projection"]["score_rows"] > 0
    assert manifest["final_projection"]["parameter_rows"] > 0
    assert manifest["modality_audit"]["included_memberships"] > 0
    assert manifest["correlation_audit"]["threshold"] == pytest.approx(0.85)
    assert manifest["dependency_audit"]["blocked_runtime_dependencies"] == 0
    expected_config_path = str((outdir / "effective_config.json").resolve())
    assert manifest["rerun"]["effective_config_path"] == expected_config_path
    assert "radselect run --config" in manifest["rerun"]["recommended_command"]
    assert expected_config_path in manifest["rerun"]["recommended_command"]
    assert manifest["nested_tuning"]["enabled"] is False
    assert manifest["quality_checks"]["status_counts"]["pass"] >= 6
    assert manifest["stability_analysis"]["resample_rows"] > 0
    assert manifest["stability_analysis"]["sampling_unit"] == "group"
    assert manifest["stability_analysis"]["group_column"] == "center"
    assert manifest["validation_splits"]["external_rows"] == n // 3
    output_manifest = json.loads((outdir / "output_manifest.json").read_text(encoding="utf-8"))
    artifacts = {artifact["path"]: artifact for artifact in output_manifest["artifacts"]}
    assert "output_manifest.json" not in artifacts
    assert "radselect_report.html" in artifacts
    assert "effective_config.json" in artifacts
    assert "run_invocation.json" in artifacts
    assert len(artifacts["manifest.json"]["sha256"]) == 64
    assert artifacts["selected_features.csv"]["rows"] >= 0
    assert artifacts["selected_features.csv"]["columns"] > 0
    dropped_features = pd.read_csv(outdir / "dropped_features.csv")
    assert {"modality", "fold", "feature", "reason", "value", "compared_with"}.issubset(
        dropped_features.columns
    )
    invocation = json.loads((outdir / "run_invocation.json").read_text(encoding="utf-8"))
    assert invocation["recommended_rerun_command"] == manifest["rerun"]["recommended_command"]
    assert invocation["effective_config"] == "effective_config.json"
    assert invocation["effective_config_path"] == expected_config_path
    assert invocation["argv"][0] == "run"
    assert invocation["effective_settings"]["task"] == "binary"
    validation_splits = pd.read_csv(outdir / "validation_splits.csv")
    assert {"fold", "dataset", "role", "id", "group"}.issubset(validation_splits.columns)
    assert set(validation_splits.loc[validation_splits["dataset"].eq("external"), "group"]) == {"site_c"}
    assert "site_c" not in set(validation_splits.loc[validation_splits["dataset"].eq("development"), "group"])
    for fold, group in validation_splits[validation_splits["dataset"].eq("development")].groupby("fold"):
        train_ids = set(group.loc[group["role"].eq("train"), "id"])
        test_ids = set(group.loc[group["role"].eq("test"), "id"])
        assert train_ids
        assert test_ids
        assert train_ids.isdisjoint(test_ids), fold
    projection_performance = pd.read_csv(outdir / "projection_performance.csv")
    assert {"projection", "n_input_features", "n_components"}.issubset(projection_performance.columns)
    assert set(projection_performance["projection"]) == {"pca"}
    final_projection_scores = pd.read_csv(outdir / "final_projection_scores.csv")
    assert {"final_projection_development", "final_projection_external"}.issubset(
        set(final_projection_scores["fold"])
    )
    assert {"id", "modality", "projection", "component_1"}.issubset(final_projection_scores.columns)
    final_projection_parameters = pd.read_csv(outdir / "final_projection_parameters.csv")
    assert {
        "feature",
        "component",
        "impute_median",
        "scale_mean",
        "scale_std",
        "weight",
        "explained_variance_ratio",
    }.issubset(final_projection_parameters.columns)
    assert set(final_projection_parameters["projection"]) == {"pca"}
    applied_projection_path = tmp_path / "applied_projection.csv"
    cli_main(
        [
            "project",
            "--input",
            str(input_path),
            "--parameters",
            str(outdir / "final_projection_parameters.csv"),
            "--output",
            str(applied_projection_path),
            "--id-column",
            "case_id",
        ]
    )
    applied_projection = pd.read_csv(applied_projection_path)
    assert (tmp_path / "applied_projection_manifest.json").exists()
    component_columns = [column for column in final_projection_scores.columns if column.startswith("component_")]
    projection_compare_columns = ["id", "modality", "projection", *component_columns]
    expected_projection = final_projection_scores[projection_compare_columns].sort_values(["id", "modality"])
    observed_projection = applied_projection[projection_compare_columns].sort_values(["id", "modality"])
    pd.testing.assert_frame_equal(
        observed_projection.reset_index(drop=True),
        expected_projection.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    composite_scores = pd.read_csv(outdir / "composite_scores.csv")
    assert {"composite_score", "raw_composite_score", "features", "weight_abs_sum"}.issubset(
        composite_scores.columns
    )
    assert "external" in set(composite_scores["fold"])
    final_signature = pd.read_csv(outdir / "final_signature.csv")
    assert {"selection_probability", "stable", "model_status"}.issubset(final_signature.columns)
    final_parameters = pd.read_csv(outdir / "final_signature_parameters.csv")
    assert {"feature", "median", "mean", "std", "weight"}.issubset(final_parameters.columns)
    assert set(final_signature["feature"]).issubset(set(final_parameters["feature"]))
    final_scores = pd.read_csv(outdir / "final_composite_scores.csv")
    assert {"final_refit_development", "final_refit_external"}.issubset(set(final_scores["fold"]))
    applied_scores_path = tmp_path / "applied_scores.csv"
    cli_main(
        [
            "score",
            "--input",
            str(input_path),
            "--parameters",
            str(outdir / "final_signature_parameters.csv"),
            "--output",
            str(applied_scores_path),
            "--id-column",
            "case_id",
        ]
    )
    applied_scores = pd.read_csv(applied_scores_path)
    assert (tmp_path / "applied_scores_manifest.json").exists()
    assert {"signature_fold", "modality", "composite_score", "raw_composite_score"}.issubset(
        applied_scores.columns
    )
    expected_scores = final_scores[["id", "modality", "composite_score", "raw_composite_score"]].sort_values(
        ["id", "modality"]
    )
    observed_scores = applied_scores[["id", "modality", "composite_score", "raw_composite_score"]].sort_values(
        ["id", "modality"]
    )
    pd.testing.assert_frame_equal(
        observed_scores.reset_index(drop=True),
        expected_scores.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    schema_audit = pd.read_csv(outdir / "schema_audit.csv")
    assert {"development", "external"}.issubset(set(schema_audit["dataset"]))
    assert schema_audit["issue"].fillna("").eq("").all()
    quality_checks = pd.read_csv(outdir / "quality_checks.csv")
    assert {"check", "status", "details"}.issubset(quality_checks.columns)
    check_status = quality_checks.set_index("check")["status"].to_dict()
    assert check_status["runtime_has_no_llm_dependency"] == "pass"
    assert check_status["validation_splits_recorded"] == "pass"
    assert check_status["external_schema_validated"] == "pass"
    assert check_status["metadata_columns_protected"] == "pass"
    assert check_status["modality_definitions_recorded"] == "pass"
    assert check_status["correlation_redundancy_audited"] == "pass"
    assert check_status["nested_elastic_net_tuning_recorded"] == "not_applicable"
    assert check_status["final_signature_reproducible"] == "pass"
    assert check_status["final_projection_reproducible"] == "pass"
    tuning_summary = pd.read_csv(outdir / "tuning_summary.csv")
    assert {
        "modality",
        "outer_fold",
        "candidate",
        "mean_inner_score",
        "selected",
        "metric",
    }.issubset(tuning_summary.columns)
    dependency_audit = pd.read_csv(outdir / "dependency_audit.csv")
    assert {"scope", "extra", "normalized_package", "blocked_llm_or_openai_dependency"}.issubset(
        dependency_audit.columns
    )
    runtime_dependencies = dependency_audit[dependency_audit["scope"].eq("runtime_required")]
    assert runtime_dependencies["extra"].fillna("").eq("").all()
    assert not {"matplotlib", "seaborn", "lifelines", "pytest", "ruff"} & set(
        runtime_dependencies["normalized_package"]
    )
    assert not runtime_dependencies["blocked_llm_or_openai_dependency"].astype(bool).any()
    frequency = pd.read_csv(outdir / "selected_feature_frequency.csv")
    assert {"modality", "feature", "selected_folds", "selection_probability"}.issubset(frequency.columns)
    modality_audit = pd.read_csv(outdir / "modality_audit.csv")
    assert {"modality", "feature", "source", "included_in_modality", "reason"}.issubset(modality_audit.columns)
    included_modalities = modality_audit.loc[modality_audit["included_in_modality"], "modality"]
    assert {"radiomics", "clinical", "combined"}.issubset(set(included_modalities))
    correlation_audit = pd.read_csv(outdir / "correlation_audit.csv")
    assert {"kept_feature", "dropped_feature", "abs_correlation", "threshold", "method", "decision"}.issubset(
        correlation_audit.columns
    )
    stability_resamples = pd.read_csv(outdir / "stability_resamples.csv")
    assert {
        "modality",
        "resample",
        "sampling_unit",
        "n_train_groups",
        "train_groups",
        "train_ids",
        "selected_features",
        "model_status",
    }.issubset(stability_resamples.columns)
    assert stability_resamples["resample"].nunique() == 2
    assert set(stability_resamples["sampling_unit"]) == {"group"}
    assert check_status["stability_resamples_recorded"] == "pass"
    if importlib.util.find_spec("matplotlib"):
        assert (outdir / "report_assets" / "performance_summary.png").exists()
        assert (outdir / "report_assets" / "stability_selection.png").exists()
        assert (outdir / "report_assets" / "projection_scores.png").exists()


def test_cli_init_config_and_run_from_json_config(tmp_path):
    config_path = tmp_path / "radselect_config.json"
    cli_main(["init-config", "--output", str(config_path)])
    template = json.loads(config_path.read_text(encoding="utf-8"))
    assert template["task"] == "binary"
    assert "feature_regex" in template
    expected_template_keys = {
        "competing_event_code",
        "exclude_regex",
        "min_variance",
        "min_unique",
        "top_k",
        "mutual_info_neighbors",
        "correlation_method",
        "elastic_net_c",
        "elastic_net_alpha",
        "elastic_net_l1_ratio",
        "elastic_net_c_grid",
        "elastic_net_alpha_grid",
        "elastic_net_l1_ratio_grid",
        "stability_train_fraction",
        "stability_threshold",
        "robustness_csv",
        "robustness_min_icc",
        "robustness_require_listed",
        "random_state",
    }
    assert expected_template_keys.issubset(template.keys())

    rng = np.random.default_rng(18)
    n = 66
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(n)],
            "mace": (x + rng.normal(scale=0.6, size=n) > 0).astype(int),
            "rad_texture": x,
            "rad_noise": rng.normal(size=n),
            "clinical_age": 70 + rng.normal(size=n),
        }
    )
    input_path = tmp_path / "features.csv"
    outdir = tmp_path / "out_config"
    frame.to_csv(input_path, index=False)
    run_config = {
        "input": str(input_path),
        "outdir": str(outdir),
        "task": "binary",
        "target": "mace",
        "id_column": "case_id",
        "feature_regex": ["^(rad_|clinical_)"],
        "radiomics_regex": ["^rad_"],
        "clinical_regex": ["^clinical_"],
        "outer_splits": 3,
        "stability_resamples": 2,
        "tune_elastic_net": False,
        "projection": "none",
    }
    config_path.write_text(json.dumps(run_config), encoding="utf-8")

    cli_main(["run", "--config", str(config_path)])

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    effective = json.loads((outdir / "effective_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((outdir / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["input"]["rows"] == n
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["run_config"]["path"].endswith("radselect_config.json")
    assert effective["input"] == str(input_path)
    assert provenance["manifest"]["input"]["sha256"] == manifest["input"]["sha256"]


def test_metadata_columns_are_protected_from_direct_api_feature_list():
    rng = np.random.default_rng(19)
    n = 72
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(n)],
            "target": (signal + rng.normal(scale=0.7, size=n) > 0).astype(int),
            "rad_signal": signal,
            "target_copy_like_name": signal + rng.normal(scale=0.1, size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="case_id",
        feature_columns=["case_id", "target", "rad_signal", "target_copy_like_name"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        random_state=9,
    )

    result = run_selection(frame, config)

    assert "target" not in set(result.selected_features["feature"])
    assert "case_id" not in set(result.stability_selection["feature"])
    audit = result.column_audit.set_index("column")
    assert audit.loc["target", "role"] == "outcome"
    assert not bool(audit.loc["target", "included_as_candidate"])
    assert bool(audit.loc["target_copy_like_name", "leakage_risk"])


def test_mutual_info_screening_records_stage():
    rng = np.random.default_rng(20)
    n = 78
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": (signal + rng.normal(scale=0.6, size=n) > 0).astype(int),
            "rad_signal": signal,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        screening_method="mutual_info",
        mutual_info_neighbors=3,
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        random_state=10,
    )

    result = run_selection(frame, config)

    assert not result.selected_features.empty
    assert set(result.selected_features["stage"]) == {"elastic_net_after_mutual_info"}
    assert "foldwise mutual_info conventional feature screening" in result.manifest["selection_pipeline"]


def test_nested_tuning_records_selected_candidate():
    rng = np.random.default_rng(13)
    n = 72
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "target": (signal + rng.normal(scale=0.7, size=n) > 0).astype(int),
            "rad_signal": signal,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=3,
        inner_splits=2,
        stability_resamples=0,
        elastic_net_c_grid=[0.1, 1.0],
        elastic_net_l1_ratio_grid=[0.5],
        random_state=4,
    )

    result = run_selection(frame, config)

    assert not result.tuning_summary.empty
    selected = result.tuning_summary[result.tuning_summary["selected"]]
    assert len(selected) == 3
    assert set(selected["outer_fold"]) == {"fold_1", "fold_2", "fold_3"}
    assert result.manifest["nested_tuning"]["enabled"]
    assert result.manifest["nested_tuning"]["selected_candidate_rows"] == 3
    check_status = result.quality_checks.set_index("check")["status"].to_dict()
    assert check_status["nested_elastic_net_tuning_recorded"] == "pass"


def test_multiclass_selection_runs():
    rng = np.random.default_rng(14)
    n = 90
    y = np.repeat([0, 1, 2], n // 3)
    x = y + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame(
        {
            "id": range(n),
            "class": y,
            "rad_signal": x,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="multiclass",
        target_column="class",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=3,
        stability_resamples=2,
        tune_elastic_net=False,
        random_state=5,
    )

    result = run_selection(frame, config)

    assert not result.selected_features.empty
    assert "balanced_accuracy" in result.performance.columns
    assert set(result.performance["task"]) == {"multiclass"}


def test_survival_selection_outputs_cindex_and_predictions():
    rng = np.random.default_rng(15)
    n = 80
    risk = rng.normal(size=n)
    event = rng.binomial(1, 0.7, size=n)
    time = np.maximum(1, 60 - 12 * risk + rng.normal(scale=4, size=n))
    frame = pd.DataFrame(
        {
            "id": range(n),
            "time": time,
            "event": event,
            "rad_risk": risk,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="survival",
        time_column="time",
        event_column="event",
        id_column="id",
        feature_columns=["rad_risk", "rad_noise"],
        outer_splits=4,
        stability_resamples=2,
        tune_elastic_net=False,
        random_state=6,
    )

    result = run_selection(frame, config)

    assert not result.performance.empty
    assert "c_index" in result.performance.columns
    assert not result.predictions.empty
    assert set(result.performance["task"]) == {"survival"}


def test_survival_no_features_keeps_predictions_csv_readable(tmp_path):
    n = 60
    frame = pd.DataFrame(
        {
            "id": [f"case_{idx}" for idx in range(n)],
            "time": np.arange(1, n + 1),
            "event": np.tile([0, 1], n // 2),
            "all_missing": np.nan,
        }
    )
    config = RunConfig(
        task="survival",
        time_column="time",
        event_column="event",
        id_column="id",
        feature_columns=["all_missing"],
        outer_splits=3,
        stability_resamples=0,
        tune_elastic_net=False,
        random_state=31,
    )

    result = run_selection(frame, config)
    result.write(tmp_path)

    predictions = pd.read_csv(tmp_path / "predictions.csv")
    performance = pd.read_csv(tmp_path / "performance.csv")

    assert predictions.empty
    assert {"id", "fold", "modality", "y_true", "prediction", "risk", "time", "event"}.issubset(
        predictions.columns
    )
    assert not performance.empty
    assert set(performance["status"]) == {"no_features"}


def test_survival_external_validation_uses_training_imputation_for_external_missingness(monkeypatch):
    class FakeCoxPHFitter:
        def __init__(self, **kwargs):
            self.params_ = pd.Series(dtype=float)

        def fit(self, frame, duration_col, event_col, **kwargs):
            feature_columns = [column for column in frame.columns if column not in {duration_col, event_col}]
            assert np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all()
            self.params_ = pd.Series({column: 1.0 for column in feature_columns})
            return self

        def predict_partial_hazard(self, frame):
            assert np.isfinite(frame.to_numpy(dtype=float)).all()
            return pd.Series(np.exp(frame.to_numpy(dtype=float).sum(axis=1)))

    monkeypatch.setitem(sys.modules, "lifelines", types.SimpleNamespace(CoxPHFitter=FakeCoxPHFitter))
    rng = np.random.default_rng(26)
    n_train = 84
    n_external = 24
    train_risk = rng.normal(size=n_train)
    external_risk = rng.normal(size=n_external)
    train = pd.DataFrame(
        {
            "id": [f"train_{idx}" for idx in range(n_train)],
            "time": np.maximum(1, 80 - 16 * train_risk + rng.normal(scale=3, size=n_train)),
            "event": np.ones(n_train, dtype=int),
            "rad_risk": train_risk,
            "rad_noise": rng.normal(size=n_train),
        }
    )
    external = pd.DataFrame(
        {
            "id": [f"external_{idx}" for idx in range(n_external)],
            "time": np.maximum(1, 80 - 16 * external_risk + rng.normal(scale=3, size=n_external)),
            "event": np.ones(n_external, dtype=int),
            "rad_risk": np.nan,
            "rad_noise": rng.normal(size=n_external),
        }
    )
    config = RunConfig(
        task="survival",
        time_column="time",
        event_column="event",
        id_column="id",
        feature_columns=["rad_risk", "rad_noise"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        top_k=1,
        random_state=26,
    )

    result = run_selection(train, config, external_data=external)

    external_predictions = result.predictions[result.predictions["fold"].eq("external")]
    assert len(external_predictions) == n_external
    assert np.isfinite(external_predictions["risk"]).all()
    external_performance = result.performance[result.performance["fold"].eq("external")]
    assert set(external_performance["status"]) == {"ok"}


def test_competing_risk_treats_event_code_of_interest_as_event():
    rng = np.random.default_rng(16)
    n = 75
    risk = rng.normal(size=n)
    event = np.where(risk > 0.5, 1, np.where(risk < -0.8, 2, 0))
    time = np.maximum(1, 50 - 10 * risk + rng.normal(scale=5, size=n))
    frame = pd.DataFrame(
        {
            "id": range(n),
            "time": time,
            "event": event,
            "rad_risk": risk,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="competing_risk",
        time_column="time",
        event_column="event",
        competing_event_code=1,
        id_column="id",
        feature_columns=["rad_risk", "rad_noise"],
        outer_splits=3,
        stability_resamples=2,
        tune_elastic_net=False,
        random_state=7,
    )

    result = run_selection(frame, config)

    assert not result.performance.empty
    assert "c_index" in result.performance.columns
    assert set(result.performance["task"]) == {"competing_risk"}
    assert result.manifest["analysis_method"]["method"] == "cause_specific_cox_or_signed_score_fallback"
    assert result.manifest["analysis_method"]["event_of_interest"] == "1"
    assert result.manifest["analysis_method"]["competing_events_handling"] == (
        "treated_as_censored_for_selection_and_c_index"
    )
    assert result.manifest["outcome_summary"]["event_of_interest_count"] == int((event == 1).sum())
    assert result.manifest["outcome_summary"]["competing_event_count"] == int((event == 2).sum())
    assert result.manifest["outcome_summary"]["censored_count"] == int((event == 0).sum())


def test_external_validation_rows_are_reported():
    rng = np.random.default_rng(17)
    n_train = 70
    n_external = 30
    train_signal = rng.normal(size=n_train)
    external_signal = rng.normal(size=n_external)
    train = pd.DataFrame(
        {
            "id": [f"train_{i}" for i in range(n_train)],
            "target": (train_signal + rng.normal(scale=0.8, size=n_train) > 0).astype(int),
            "rad_signal": train_signal,
            "rad_noise": rng.normal(size=n_train),
        }
    )
    external = pd.DataFrame(
        {
            "id": [f"external_{i}" for i in range(n_external)],
            "target": (external_signal + rng.normal(scale=0.8, size=n_external) > 0).astype(int),
            "rad_signal": external_signal,
            "rad_noise": rng.normal(size=n_external),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        projection="pca",
        projection_components=2,
        random_state=8,
    )

    result = run_selection(train, config, external_data=external)

    assert "external" in set(result.performance["fold"])
    assert "external_fit" in set(result.selected_features["fold"])
    assert result.predictions[result.predictions["fold"] == "external"]["id"].str.startswith("external_").all()
    assert "external" in set(result.projection_performance["fold"])
    assert result.projection_predictions[
        result.projection_predictions["fold"] == "external"
    ]["id"].str.startswith("external_").all()


def test_external_validation_requires_matching_feature_schema():
    rng = np.random.default_rng(24)
    n_train = 72
    n_external = 24
    signal = rng.normal(size=n_train)
    external_signal = rng.normal(size=n_external)
    train = pd.DataFrame(
        {
            "id": [f"train_{i}" for i in range(n_train)],
            "target": (signal + rng.normal(scale=0.7, size=n_train) > 0).astype(int),
            "rad_signal": signal,
            "rad_noise": rng.normal(size=n_train),
        }
    )
    external = pd.DataFrame(
        {
            "id": [f"external_{i}" for i in range(n_external)],
            "target": (external_signal + rng.normal(scale=0.7, size=n_external) > 0).astype(int),
            "rad_signal": external_signal,
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=3,
        stability_resamples=1,
        tune_elastic_net=False,
        random_state=24,
    )

    with pytest.raises(ValueError, match="External data is missing required feature columns: rad_noise"):
        run_selection(train, config, external_data=external)


def test_holdout_group_is_removed_from_development_and_evaluated_as_external():
    rng = np.random.default_rng(23)
    n = 90
    signal = rng.normal(size=n)
    center = np.repeat(["site_a", "site_b", "site_c"], n // 3)
    frame = pd.DataFrame(
        {
            "id": [f"case_{i}" for i in range(n)],
            "center": center,
            "target": (signal + rng.normal(scale=0.7, size=n) > 0).astype(int),
            "rad_signal": signal,
            "rad_noise": rng.normal(size=n),
        }
    )
    config = RunConfig(
        task="binary",
        target_column="target",
        id_column="id",
        group_column="center",
        holdout_groups=["site_c"],
        feature_columns=["rad_signal", "rad_noise"],
        outer_splits=2,
        stability_resamples=1,
        tune_elastic_net=False,
        random_state=23,
    )

    result = run_selection(frame, config)

    assert result.manifest["holdout_validation"]["enabled"]
    assert result.manifest["holdout_validation"]["groups"] == ["site_c"]
    assert result.manifest["n_rows"] == 60
    assert result.manifest["sample_audit"]["external_rows_retained"] == 30
    assert "external" in set(result.performance["fold"])
    external_ids = set(result.predictions[result.predictions["fold"] == "external"]["id"])
    assert external_ids == set(frame.loc[frame["center"].eq("site_c"), "id"])
