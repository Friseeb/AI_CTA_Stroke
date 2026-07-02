"""Command-line interface for radselect."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

import pandas as pd

from .config import TASKS, RunConfig
from .core import apply_composite_score_parameters, apply_projection_parameters, run_selection, write_output_manifest
from .reporting import write_html_report


DEFAULT_RUN_OPTIONS = {
    "external_input": None,
    "target": None,
    "time_column": None,
    "event_column": None,
    "competing_event_code": "1",
    "id_column": None,
    "group_column": None,
    "holdout_group": [],
    "feature_regex": None,
    "exclude_regex": [],
    "radiomics_regex": [],
    "clinical_regex": [],
    "domain": [],
    "max_missing": 0.20,
    "min_variance": 1e-8,
    "min_unique": 2,
    "top_k": 30,
    "screening_method": "univariate",
    "mutual_info_neighbors": 3,
    "correlation_threshold": 0.85,
    "correlation_method": "spearman",
    "elastic_net_c": 0.2,
    "elastic_net_alpha": 0.01,
    "elastic_net_l1_ratio": 0.5,
    "tune_elastic_net": True,
    "no_tune_elastic_net": False,
    "inner_splits": 3,
    "elastic_net_c_grid": [0.05, 0.2, 1.0],
    "elastic_net_alpha_grid": [0.001, 0.01, 0.1],
    "elastic_net_l1_ratio_grid": [0.1, 0.5, 0.9],
    "outer_splits": 5,
    "stability_resamples": 100,
    "stability_train_fraction": 0.75,
    "stability_threshold": 0.50,
    "robustness_csv": None,
    "robustness_min_icc": 0.75,
    "robustness_require_listed": False,
    "projection": "none",
    "projection_components": 5,
    "random_state": 13,
    "feature_metadata_csv": None,
    "require_ibsi_compliant": False,
    "ibsi_require_listed": False,
}


CONFIG_TEMPLATE = {
    "input": "features.csv",
    "external_input": None,
    "outdir": "radselect_out",
    "task": "binary",
    "target": "outcome",
    "time_column": None,
    "event_column": None,
    "competing_event_code": "1",
    "id_column": "case_id",
    "group_column": None,
    "holdout_group": [],
    "feature_regex": ["^rad_", "^clinical_"],
    "exclude_regex": [],
    "radiomics_regex": ["^rad_"],
    "clinical_regex": ["^clinical_"],
    "domain": ["texture:^rad_.*texture", "shape:^rad_.*shape"],
    "max_missing": 0.20,
    "min_variance": 1e-8,
    "min_unique": 2,
    "top_k": 30,
    "screening_method": "univariate",
    "mutual_info_neighbors": 3,
    "correlation_threshold": 0.85,
    "correlation_method": "spearman",
    "elastic_net_c": 0.2,
    "elastic_net_alpha": 0.01,
    "elastic_net_l1_ratio": 0.5,
    "tune_elastic_net": True,
    "inner_splits": 3,
    "elastic_net_c_grid": [0.05, 0.2, 1.0],
    "elastic_net_alpha_grid": [0.001, 0.01, 0.1],
    "elastic_net_l1_ratio_grid": [0.1, 0.5, 0.9],
    "outer_splits": 5,
    "stability_resamples": 100,
    "stability_train_fraction": 0.75,
    "stability_threshold": 0.50,
    "robustness_csv": None,
    "robustness_min_icc": 0.75,
    "robustness_require_listed": False,
    "projection": "pca",
    "projection_components": 5,
    "random_state": 13,
    "feature_metadata_csv": None,
    "require_ibsi_compliant": False,
    "ibsi_require_listed": False,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args._argv = list(argv) if argv is not None else sys.argv[1:]
    if args.command == "run":
        run_command(args)
    elif args.command == "score":
        score_command(args)
    elif args.command == "project":
        project_command(args)
    elif args.command == "init-config":
        init_config_command(args)
    else:
        parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radselect", description="Feature selection for radiomic/clinical tables.")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser(
        "run",
        help="Run selection, stability analysis, validation, and reporting.",
        argument_default=argparse.SUPPRESS,
    )
    run.add_argument("--config", type=Path, help="Optional JSON run configuration.")
    run.add_argument("--input", type=Path, help="Input CSV containing one row per subject/case.")
    run.add_argument("--external-input", type=Path, help="Optional external validation CSV.")
    run.add_argument("--outdir", type=Path, help="Output directory.")
    run.add_argument("--task", choices=TASKS)
    run.add_argument("--target", help="Outcome column for classification/regression.")
    run.add_argument("--time-column", help="Time column for survival/competing-risk tasks.")
    run.add_argument("--event-column", help="Event column for survival/competing-risk tasks.")
    run.add_argument("--competing-event-code", help="Event code of interest for competing-risk mode.")
    run.add_argument("--id-column")
    run.add_argument("--group-column", help="Center/site/scanner/group column for held-out group validation.")
    run.add_argument(
        "--holdout-group",
        action="append",
        help="Group value to hold out from development and evaluate as center/site-held-out validation. Can repeat.",
    )
    run.add_argument("--feature-regex", action="append", help="Regex for candidate feature columns. Can repeat.")
    run.add_argument("--exclude-regex", action="append", help="Regex for columns to exclude. Can repeat.")
    run.add_argument("--radiomics-regex", action="append", help="Regex assigning columns to radiomics.")
    run.add_argument("--clinical-regex", action="append", help="Regex assigning columns to clinical.")
    run.add_argument("--domain", action="append", help="Named domain as name:regex. Can repeat.")
    run.add_argument("--max-missing", type=float)
    run.add_argument("--min-variance", type=float)
    run.add_argument("--min-unique", type=int)
    run.add_argument("--top-k", type=int)
    run.add_argument("--screening-method", choices=["univariate", "mutual_info"])
    run.add_argument("--mutual-info-neighbors", type=int)
    run.add_argument("--correlation-threshold", type=float)
    run.add_argument("--correlation-method", choices=["spearman", "pearson", "kendall"])
    run.add_argument("--elastic-net-c", type=float)
    run.add_argument("--elastic-net-alpha", type=float)
    run.add_argument("--elastic-net-l1-ratio", type=float)
    run.add_argument(
        "--tune-elastic-net",
        dest="tune_elastic_net",
        action="store_true",
        help="Enable inner-loop elastic-net tuning.",
    )
    run.add_argument("--no-tune-elastic-net", action="store_true", help="Disable inner-loop elastic-net tuning.")
    run.add_argument("--inner-splits", type=int)
    run.add_argument("--elastic-net-c-grid")
    run.add_argument("--elastic-net-alpha-grid")
    run.add_argument("--elastic-net-l1-ratio-grid")
    run.add_argument("--outer-splits", type=int)
    run.add_argument("--stability-resamples", type=int)
    run.add_argument("--stability-train-fraction", type=float)
    run.add_argument("--stability-threshold", type=float)
    run.add_argument("--robustness-csv", type=Path)
    run.add_argument("--robustness-min-icc", type=float)
    run.add_argument("--robustness-require-listed", action="store_true")
    run.add_argument("--projection", choices=["none", "pca", "pls"])
    run.add_argument("--projection-components", type=int)
    run.add_argument("--random-state", type=int)
    run.add_argument("--feature-metadata-csv", type=Path, help="Optional feature metadata/provenance CSV.")
    run.add_argument(
        "--require-ibsi-compliant",
        action="store_true",
        help="Reject features listed in feature metadata as non-IBSI-compliant.",
    )
    run.add_argument(
        "--ibsi-require-listed",
        action="store_true",
        help="When requiring IBSI compliance, also reject features absent from the metadata CSV.",
    )

    init_config = subparsers.add_parser("init-config", help="Write a template JSON run configuration.")
    init_config.add_argument("--output", type=Path, help="Output JSON path. Prints to stdout if omitted.")

    score = subparsers.add_parser(
        "score",
        help="Apply final_signature_parameters.csv to a new CSV without rerunning feature selection.",
    )
    score.add_argument("--input", type=Path, required=True, help="Input CSV containing feature columns to score.")
    score.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="CSV written by radselect run as final_signature_parameters.csv.",
    )
    score.add_argument("--output", type=Path, required=True, help="Output CSV for applied composite scores.")
    score.add_argument("--id-column", help="Optional ID column to carry into the score output.")

    project = subparsers.add_parser(
        "project",
        help="Apply final_projection_parameters.csv to a new CSV without refitting PCA/PLS.",
    )
    project.add_argument("--input", type=Path, required=True, help="Input CSV containing feature columns to project.")
    project.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="CSV written by radselect run as final_projection_parameters.csv.",
    )
    project.add_argument("--output", type=Path, required=True, help="Output CSV for applied projection scores.")
    project.add_argument("--id-column", help="Optional ID column to carry into the projection output.")
    return parser


def run_command(args: argparse.Namespace) -> None:
    settings = effective_run_settings(args)
    input_path = require_path(settings, "input")
    outdir = require_path(settings, "outdir")
    if not settings.get("task"):
        raise ValueError("Missing required option: task. Provide --task or set task in --config.")
    frame = pd.read_csv(input_path)
    external_path = optional_path(settings.get("external_input"))
    external = pd.read_csv(external_path) if external_path else None
    feature_columns = select_columns(frame, settings.get("feature_regex"), settings.get("exclude_regex"), settings)
    radiomics_columns = select_columns(frame, settings.get("radiomics_regex"), settings.get("exclude_regex"), settings, base=feature_columns)
    clinical_columns = select_columns(frame, settings.get("clinical_regex"), settings.get("exclude_regex"), settings, base=feature_columns)
    domains = parse_domains(frame, settings.get("domain") or [], feature_columns)
    config = RunConfig(
        task=settings["task"],
        target_column=settings.get("target"),
        time_column=settings.get("time_column"),
        event_column=settings.get("event_column"),
        competing_event_code=settings.get("competing_event_code", "1"),
        id_column=settings.get("id_column"),
        group_column=settings.get("group_column"),
        holdout_groups=settings.get("holdout_group") or [],
        feature_columns=feature_columns,
        radiomics_columns=radiomics_columns,
        clinical_columns=clinical_columns,
        domains=domains,
        max_missing=settings["max_missing"],
        min_variance=settings["min_variance"],
        min_unique=settings["min_unique"],
        top_k=settings["top_k"],
        screening_method=settings["screening_method"],
        mutual_info_neighbors=settings["mutual_info_neighbors"],
        correlation_threshold=settings["correlation_threshold"],
        correlation_method=settings["correlation_method"],
        elastic_net_c=settings["elastic_net_c"],
        elastic_net_alpha=settings["elastic_net_alpha"],
        elastic_net_l1_ratio=settings["elastic_net_l1_ratio"],
        tune_elastic_net=bool(settings.get("tune_elastic_net", not settings.get("no_tune_elastic_net", False))),
        inner_splits=settings["inner_splits"],
        elastic_net_c_grid=parse_float_list(settings["elastic_net_c_grid"]),
        elastic_net_alpha_grid=parse_float_list(settings["elastic_net_alpha_grid"]),
        elastic_net_l1_ratio_grid=parse_float_list(settings["elastic_net_l1_ratio_grid"]),
        outer_splits=settings["outer_splits"],
        stability_resamples=settings["stability_resamples"],
        stability_train_fraction=settings["stability_train_fraction"],
        stability_threshold=settings["stability_threshold"],
        random_state=settings["random_state"],
        feature_metadata_csv=optional_path(settings.get("feature_metadata_csv")),
        require_ibsi_compliant=settings["require_ibsi_compliant"],
        ibsi_require_listed=settings["ibsi_require_listed"],
        robustness_csv=optional_path(settings.get("robustness_csv")),
        robustness_min_icc=settings["robustness_min_icc"],
        robustness_require_listed=settings["robustness_require_listed"],
        projection=settings["projection"],
        projection_components=settings["projection_components"],
    )
    result = run_selection(frame, config, external_data=external)
    result.manifest["input"] = input_fingerprint(input_path, frame)
    if external_path is not None and external is not None:
        result.manifest["external_input"] = input_fingerprint(external_path, external)
    if hasattr(args, "config") and args.config:
        result.manifest["run_config"] = input_fingerprint(args.config, None)
    effective_settings = json_ready_settings(settings)
    result.manifest["effective_settings"] = effective_settings
    recommended_rerun = recommended_rerun_command(outdir)
    result.manifest["rerun"] = {
        "effective_config": "effective_config.json",
        "effective_config_path": str((outdir / "effective_config.json").expanduser().resolve()),
        "run_invocation": "run_invocation.json",
        "working_directory": str(Path.cwd().resolve()),
        "recommended_command": recommended_rerun,
    }
    result.write(outdir)
    (outdir / "effective_config.json").write_text(
        json.dumps(effective_settings, indent=2) + "\n",
        encoding="utf-8",
    )
    report = write_html_report(result, outdir)
    write_run_invocation(outdir, args, effective_settings, report, recommended_rerun)
    write_output_manifest(outdir)
    print(f"Wrote radselect outputs to {outdir}")
    print(f"Report: {report}")


def score_command(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser()
    parameter_path = args.parameters.expanduser()
    output_path = args.output.expanduser()
    frame = pd.read_csv(input_path)
    parameters = pd.read_csv(parameter_path)
    scores = apply_composite_score_parameters(frame, parameters, id_column=args.id_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    manifest = {
        "package": "radselect",
        "command": "score",
        "input": input_fingerprint(input_path, frame),
        "parameters": input_fingerprint(parameter_path, parameters),
        "output": input_fingerprint(output_path, scores),
        "id_column": args.id_column,
        "rows": int(len(scores)),
        "modalities": sorted(scores["modality"].astype(str).unique().tolist()) if not scores.empty else [],
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote radselect scores to {output_path}")
    print(f"Manifest: {manifest_path}")


def project_command(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser()
    parameter_path = args.parameters.expanduser()
    output_path = args.output.expanduser()
    frame = pd.read_csv(input_path)
    parameters = pd.read_csv(parameter_path)
    projections = apply_projection_parameters(frame, parameters, id_column=args.id_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(output_path, index=False)
    manifest = {
        "package": "radselect",
        "command": "project",
        "input": input_fingerprint(input_path, frame),
        "parameters": input_fingerprint(parameter_path, parameters),
        "output": input_fingerprint(output_path, projections),
        "id_column": args.id_column,
        "rows": int(len(projections)),
        "modalities": sorted(projections["modality"].astype(str).unique().tolist()) if not projections.empty else [],
        "projections": sorted(projections["projection"].astype(str).unique().tolist()) if not projections.empty else [],
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote radselect projections to {output_path}")
    print(f"Manifest: {manifest_path}")


def select_columns(
    frame: pd.DataFrame,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    settings: dict,
    *,
    base: list[str] | None = None,
) -> list[str]:
    candidates = base if base is not None else list(frame.columns)
    if include_patterns:
        included = [
            column
            for column in candidates
            if any(re.search(pattern, column) for pattern in include_patterns)
        ]
    else:
        excluded = {
            value
            for value in [
                settings.get("target"),
                settings.get("time_column"),
                settings.get("event_column"),
                settings.get("id_column"),
                settings.get("group_column"),
            ]
            if value
        }
        included = [
            column
            for column in candidates
            if column not in excluded and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
    if exclude_patterns:
        included = [
            column
            for column in included
            if not any(re.search(pattern, column) for pattern in exclude_patterns)
        ]
    return list(dict.fromkeys(included))


def parse_domains(frame: pd.DataFrame, specs: list[str], feature_columns: list[str]) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Domain must be formatted as name:regex, got {spec!r}.")
        name, pattern = spec.split(":", 1)
        domains[name] = [
            column
            for column in feature_columns
            if column in frame.columns and re.search(pattern, column)
        ]
    return domains


def parse_float_list(text: str) -> list[float]:
    if isinstance(text, (list, tuple)):
        return [float(value) for value in text]
    values = []
    for item in str(text).split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("Expected at least one comma-separated numeric value.")
    return values


def effective_run_settings(args: argparse.Namespace) -> dict:
    settings = dict(DEFAULT_RUN_OPTIONS)
    config_path = getattr(args, "config", None)
    if config_path:
        settings.update(load_json_config(config_path))
    cli_values = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "config"} and value is not None
    }
    settings.update(cli_values)
    if settings.get("no_tune_elastic_net"):
        settings["tune_elastic_net"] = False
    settings["feature_regex"] = optional_list(settings.get("feature_regex"))
    settings["exclude_regex"] = optional_list(settings.get("exclude_regex")) or []
    settings["radiomics_regex"] = optional_list(settings.get("radiomics_regex")) or []
    settings["clinical_regex"] = optional_list(settings.get("clinical_regex")) or []
    settings["domain"] = optional_list(settings.get("domain")) or []
    settings["holdout_group"] = optional_list(settings.get("holdout_group")) or []
    return settings


def load_json_config(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def init_config_command(args: argparse.Namespace) -> None:
    text = json.dumps(CONFIG_TEMPLATE, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote template config to {args.output}")
    else:
        print(text, end="")


def write_run_invocation(
    outdir: Path,
    args: argparse.Namespace,
    effective_settings: dict,
    report: Path,
    recommended_rerun: str,
) -> Path:
    record = {
        "schema_version": 1,
        "package": "radselect",
        "command": "run",
        "working_directory": str(Path.cwd().resolve()),
        "argv": list(getattr(args, "_argv", [])),
        "effective_config": "effective_config.json",
        "effective_config_path": str((outdir / "effective_config.json").expanduser().resolve()),
        "recommended_rerun_command": recommended_rerun,
        "report": Path(report).name,
        "effective_settings": effective_settings,
    }
    path = outdir / "run_invocation.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def recommended_rerun_command(outdir: Path) -> str:
    working_directory = Path.cwd().resolve()
    config_path = (outdir / "effective_config.json").expanduser().resolve()
    return f"cd {shlex.quote(str(working_directory))} && radselect run --config {shlex.quote(str(config_path))}"


def optional_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def require_path(settings: dict, key: str) -> Path:
    value = settings.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required option: {key}. Provide --{key.replace('_', '-')} or set {key} in --config.")
    return Path(value).expanduser()


def optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def input_fingerprint(path: Path, frame: pd.DataFrame | None) -> dict:
    resolved = path.expanduser().resolve()
    record = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if frame is not None:
        record["rows"] = int(len(frame))
        record["columns"] = int(len(frame.columns))
        record["column_names"] = [str(column) for column in frame.columns]
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready_settings(settings: dict) -> dict:
    ready = {}
    for key, value in settings.items():
        if isinstance(value, Path):
            ready[key] = str(value)
        else:
            ready[key] = value
    return ready


if __name__ == "__main__":
    main()
