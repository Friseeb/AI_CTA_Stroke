"""Build 3D Slicer QC review sets from pipeline outputs."""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


ANATOMY_ALIASES: dict[str, list[str]] = {
    "aorta": ["aorta", "aortic", "periaortic", "lumen_protrusion", "lumen_hu"],
    "carotid": ["carotid", "carotids", "cca", "ica"],
    "vertebral": ["vertebral", "vertebrals", "vert"],
    "artery": ["aorta", "aortic", "periaortic", "carotid", "vertebral", "artery", "arteries", "vessel"],
}

TASK_ALIASES: dict[str, list[str]] = {
    "segmentation": ["segmentation", "segment", "segments", "mask", "cleaned"],
    "calcification": ["calcification", "calcium", "calcified", "bone"],
    "adipose_tissue": ["adipose", "fat", "tissue"],
    "wall_from_fat": ["aortic_wall", "wall_from_fat", "fat_lumen", "contrast_lumen", "wall_lumen", "lumen_hu"],
    "lumen_protrusion": ["lumen_protrusion", "protrusion", "indentation"],
    "perivessel_wall_radiomics": ["wall", "perivessel", "peri", "shell", "radiomics"],
    "flow_dynamics": ["flow", "dynamics", "cfd", "wss"],
    "shape": ["shape", "geometry", "segments", "mask"],
}

CATEGORY_COLORS: dict[str, tuple[float, float, float]] = {
    "artery": (0.9, 0.05, 0.05),
    "bone": (1.0, 0.95, 0.78),
    "tissue": (0.1, 0.7, 0.35),
    "fat": (1.0, 0.95, 0.0),
    "flow": (0.0, 0.75, 1.0),
    "lumen": (0.0, 0.85, 1.0),
    "wall": (0.0, 0.75, 0.45),
    "ulcer": (0.55, 0.0, 1.0),
    "protrusion": (0.78, 0.0, 0.2),
    "shape": (0.65, 0.35, 1.0),
    "other": (0.6, 0.6, 0.6),
}


@dataclass(frozen=True)
class MaskRecord:
    case_id: str
    image_path: str
    mask_path: str
    anatomy: str
    task: str
    label: str
    category: str


def load_case_table(
    manifest_path: str | Path | None,
    clinical_table_path: str | Path | None = None,
    outputs_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load case rows from a manifest, optionally joined to clinical variables."""
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path)
    else:
        if outputs_root is None:
            raise ValueError("Either manifest_path or outputs_root is required.")
        manifest = _discover_cases_from_outputs(Path(outputs_root))

    if "case_id" not in manifest.columns:
        raise ValueError("Manifest must contain a case_id column.")

    if clinical_table_path is not None:
        clinical = pd.read_csv(clinical_table_path)
        if "case_id" not in clinical.columns:
            raise ValueError("Clinical table must contain a case_id column.")
        manifest = manifest.merge(clinical, on="case_id", how="left", suffixes=("", "_clinical"))
    manifest["case_id"] = manifest["case_id"].astype(str)
    return manifest


def select_cases(
    cases: pd.DataFrame,
    case_ids: Iterable[str] | None = None,
    filters: Iterable[str] | None = None,
    feature_tables: Iterable[str | Path] | None = None,
    outlier_features: Iterable[str] | None = None,
    outlier_method: str = "quantile",
    outlier_direction: str = "both",
    outlier_quantile: float = 0.95,
    outlier_z: float = 3.0,
    outlier_top_n: int = 10,
) -> pd.DataFrame:
    """Select cases by explicit IDs, manifest/clinical filters, and feature outliers."""
    selected = cases.copy()
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        selected = selected[selected["case_id"].isin(wanted)]

    for expression in filters or []:
        selected = _apply_filter_expression(selected, expression)

    if outlier_features:
        feature_frame = load_feature_tables(feature_tables or [])
        feature_frame = feature_frame[feature_frame["case_id"].isin(selected["case_id"].astype(str))]
        outlier_case_ids = select_outlier_case_ids(
            feature_frame,
            selectors=list(outlier_features),
            method=outlier_method,
            direction=outlier_direction,
            quantile=outlier_quantile,
            z_threshold=outlier_z,
            top_n=outlier_top_n,
        )
        selected = selected[selected["case_id"].isin(outlier_case_ids)]

    return selected.reset_index(drop=True)


def load_feature_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load long-format or wide-format feature CSVs into normalized rows."""
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty or "case_id" not in frame.columns:
            continue
        if {"feature_name", "feature_value"}.issubset(frame.columns):
            out = frame.copy()
            for column in ["region", "feature_group", "threshold_if_applicable", "mask_name"]:
                if column not in out.columns:
                    out[column] = ""
            out["source_feature_file"] = str(path)
            frames.append(out)
        else:
            value_columns = [column for column in frame.columns if column != "case_id"]
            melted = frame.melt(id_vars=["case_id"], value_vars=value_columns, var_name="feature_name")
            melted = melted.rename(columns={"value": "feature_value"})
            melted["region"] = ""
            melted["feature_group"] = "wide_or_qc"
            melted["threshold_if_applicable"] = ""
            melted["mask_name"] = ""
            melted["source_feature_file"] = str(path)
            frames.append(melted)
    if not frames:
        return pd.DataFrame(
            columns=[
                "case_id",
                "region",
                "feature_group",
                "feature_name",
                "feature_value",
                "threshold_if_applicable",
                "mask_name",
                "source_feature_file",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["case_id"] = out["case_id"].astype(str)
    out["feature_value"] = pd.to_numeric(out["feature_value"], errors="coerce")
    return out


def select_outlier_case_ids(
    features: pd.DataFrame,
    selectors: list[str],
    method: str = "quantile",
    direction: str = "both",
    quantile: float = 0.95,
    z_threshold: float = 3.0,
    top_n: int = 10,
) -> set[str]:
    """Return case IDs that are outliers for one or more feature selectors."""
    selected: set[str] = set()
    for selector in selectors:
        values = _feature_rows_for_selector(features, selector)
        values = values.dropna(subset=["feature_value"])
        if values.empty:
            continue
        values = values.sort_values("feature_value")
        if method == "top-n":
            if direction in {"low", "both"}:
                selected.update(values.head(top_n)["case_id"].astype(str))
            if direction in {"high", "both"}:
                selected.update(values.tail(top_n)["case_id"].astype(str))
        elif method == "zscore":
            mean = float(values["feature_value"].mean())
            std = float(values["feature_value"].std(ddof=0))
            if std == 0:
                continue
            z = (values["feature_value"] - mean) / std
            if direction == "high":
                selected.update(values.loc[z >= z_threshold, "case_id"].astype(str))
            elif direction == "low":
                selected.update(values.loc[z <= -z_threshold, "case_id"].astype(str))
            else:
                selected.update(values.loc[z.abs() >= z_threshold, "case_id"].astype(str))
        elif method == "quantile":
            high_cutoff = float(values["feature_value"].quantile(quantile))
            low_cutoff = float(values["feature_value"].quantile(1.0 - quantile))
            if direction in {"high", "both"}:
                selected.update(values.loc[values["feature_value"] >= high_cutoff, "case_id"].astype(str))
            if direction in {"low", "both"}:
                selected.update(values.loc[values["feature_value"] <= low_cutoff, "case_id"].astype(str))
        else:
            raise ValueError(f"Unsupported outlier method: {method}")
    return selected


def discover_mask_records(
    selected_cases: pd.DataFrame,
    anatomies: Iterable[str],
    tasks: Iterable[str],
    outputs_root: str | Path,
    project_root: str | Path | None = None,
    manifest_base: str | Path | None = None,
) -> list[MaskRecord]:
    """Find masks for the selected cases and classify them for Slicer display."""
    outputs = Path(outputs_root)
    root = Path(project_root) if project_root is not None else Path.cwd()
    base = Path(manifest_base) if manifest_base is not None else root
    anatomy_list = [item.lower() for item in anatomies] or ["all"]
    task_list = [item.lower() for item in tasks] or ["all"]
    records: list[MaskRecord] = []
    for _, row in selected_cases.iterrows():
        case_id = str(row["case_id"])
        image_path = _image_path_for_row(row, root, base)
        mask_paths = _manifest_mask_paths(row, root, base)
        mask_paths.extend(_output_mask_paths(outputs, case_id))
        unique_paths = sorted({path for path in mask_paths if _looks_like_nifti(path)})
        for path in unique_paths:
            anatomy = infer_anatomy(path, anatomy_list)
            if anatomy is None:
                continue
            task = infer_task(path, task_list)
            if task is None:
                continue
            category = infer_category(path, anatomy=anatomy, task=task)
            records.append(
                MaskRecord(
                    case_id=case_id,
                    image_path=str(image_path),
                    mask_path=str(path),
                    anatomy=anatomy,
                    task=task,
                    label=_label_for_mask(path, case_id),
                    category=category,
                )
            )
    return _with_context_trace_records(records, selected_cases, outputs, root, base, task_list)


def _with_context_trace_records(
    records: list[MaskRecord],
    selected_cases: pd.DataFrame,
    outputs_root: Path,
    project_root: Path,
    manifest_base: Path,
    task_list: list[str],
) -> list[MaskRecord]:
    """Add anatomy context masks needed to interpret task-specific overlays."""
    if "all" not in task_list and not any(
        task in task_list
        for task in [
            "calcification",
            "adipose_tissue",
            "wall_from_fat",
            "lumen_protrusion",
            "perivessel_wall_radiomics",
            "shape",
        ]
    ):
        return records

    existing_keys = {(record.case_id, Path(record.mask_path).resolve()) for record in records}
    additions: list[MaskRecord] = []
    for _, row in selected_cases.iterrows():
        case_id = str(row["case_id"])
        image_path = _image_path_for_row(row, project_root, manifest_base)
        if not str(image_path):
            matched = [record for record in records if record.case_id == case_id]
            image_path = Path(matched[0].image_path) if matched else Path("")
        for label, trace_path in _context_trace_paths(outputs_root, project_root, case_id):
            key = (case_id, trace_path.resolve())
            if key in existing_keys:
                continue
            additions.append(
                MaskRecord(
                    case_id=case_id,
                    image_path=str(image_path),
                    mask_path=str(trace_path),
                    anatomy="aorta",
                    task="context_trace",
                    label=_short_qc_label(label),
                    category="artery",
                )
            )
            existing_keys.add(key)
    return additions + records


def _context_trace_paths(outputs_root: Path, project_root: Path, case_id: str) -> list[tuple[str, Path]]:
    """Return one preferred aorta trace for context display."""
    paths: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in _vista_aorta_candidates(project_root, case_id):
        if path.exists() and path.resolve() not in seen:
            paths.append((label, path))
            seen.add(path.resolve())
            break
    if paths:
        return paths

    for label, path in _highres_aorta_candidates(project_root, case_id):
        if path.exists() and path.resolve() not in seen:
            paths.append((label, path))
            seen.add(path.resolve())
            break
    if paths:
        return paths

    trace_path = _find_case_mask(outputs_root, case_id, ["aorta_mask_cleaned", "aorta.nii.gz"])
    if trace_path is not None and trace_path.resolve() not in seen:
        paths.append(("aorta_trace", trace_path))
    return paths


def _vista_aorta_candidates(project_root: Path, case_id: str) -> list[tuple[str, Path]]:
    bases = [project_root, project_root.parent]
    candidates: list[tuple[str, Path]] = []
    for base in bases:
        candidates.extend(
            [
                (
                    "aorta_vista_trace",
                    base / "outputs" / "test" / f"{case_id}_full" / "nv_segment_ct_aorta" / f"{case_id}_aorta6.nii.gz",
                ),
                (
                    "aorta_vista_trace",
                    base / "outputs" / "test" / f"{case_id}_full" / "aorta_candidates" / "aorta_nv_segment_ct.nii.gz",
                ),
            ]
        )
    return candidates


def _highres_aorta_candidates(project_root: Path, case_id: str) -> list[tuple[str, Path]]:
    bases = [project_root, project_root.parent]
    candidates: list[tuple[str, Path]] = []
    for base in bases:
        candidates.extend(
            [
                (
                    "aorta_highres_trace",
                    base / "outputs" / "test" / f"{case_id}_full" / "totalseg_heartchambers_highres" / "aorta.nii.gz",
                ),
                (
                    "aorta_highres_trace",
                    base
                    / "outputs"
                    / "test"
                    / f"{case_id}_full"
                    / "aorta_candidates"
                    / "aorta_totalseg_heartchambers_highres.nii.gz",
                ),
            ]
        )
    return candidates


def _find_case_mask(outputs_root: Path, case_id: str, name_tokens: list[str]) -> Path | None:
    paths = _all_output_mask_paths(outputs_root, case_id)
    for token in name_tokens:
        token_lower = token.lower()
        for path in paths:
            lower = path.name.lower()
            if token_lower in lower or lower == token_lower:
                return path
    return None


def write_selection_table(records: list[MaskRecord], output_path: str | Path) -> Path:
    """Write selected case/mask rows for auditability."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(path, index=False)
    return path


def write_slicer_scripts(records: list[MaskRecord], output_dir: str | Path) -> list[Path]:
    """Write one Slicer Python loader script per selected case."""
    script_dir = Path(output_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    for case_id, case_records in _records_by_case(records).items():
        script_path = script_dir / f"{case_id}_load_qc_in_slicer.py"
        script_path.write_text(_slicer_script(case_id, case_records), encoding="utf-8")
        scripts.append(script_path)
    return scripts


def write_slicer_launcher(scripts: list[Path], output_dir: str | Path, case_index: int = 0) -> Path | None:
    """Write a macOS shell launcher for the selected Slicer script."""
    if not scripts:
        return None
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    index = min(max(case_index, 0), len(scripts) - 1)
    script_path = scripts[index].resolve()
    launcher = outdir / "open_in_slicer.sh"
    launcher.write_text(
        f'''#!/usr/bin/env bash
set -euo pipefail

SLICER_SCRIPT={str(script_path)!r}
LOG_FILE="$(dirname "$SLICER_SCRIPT")/$(basename "$SLICER_SCRIPT" .py)_launch.log"

if [[ -n "${{SLICER_APP:-}}" && -d "$SLICER_APP" ]]; then
  SLICER_APP_PATH="$SLICER_APP"
elif [[ -d /Applications/Slicer.app ]]; then
  SLICER_APP_PATH="/Applications/Slicer.app"
else
  SLICER_APP_PATH="$(find /Applications -maxdepth 1 -name 'Slicer*.app' -type d | sort | tail -n 1)"
  if [[ -z "$SLICER_APP_PATH" ]]; then
    echo "Could not find 3D Slicer.app. Set SLICER_APP=/path/to/Slicer.app and retry." >&2
    exit 1
  fi
fi

echo "Launching: open -n -a $SLICER_APP_PATH --args --ignore-slicerrc --python-script $SLICER_SCRIPT"
echo "Log: $LOG_FILE"
open -n -a "$SLICER_APP_PATH" --args --ignore-slicerrc --python-script "$SLICER_SCRIPT" >"$LOG_FILE" 2>&1
echo "Requested Slicer load for: $SLICER_SCRIPT"
''',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def write_review_outputs(
    selected_cases: pd.DataFrame,
    records: list[MaskRecord],
    scripts: list[Path],
    output_dir: str | Path,
    reviewer: str,
    anatomies: Iterable[str],
    tasks: Iterable[str],
    comments: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Write structured QC task, comment template, and reviewer log outputs."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    script_by_case = {script.name.removesuffix("_load_qc_in_slicer.py"): script for script in scripts}
    records_by_case = _records_by_case(records)
    anatomy_text = ",".join(anatomies)
    task_text = ",".join(tasks)
    comment_text = " | ".join(comments or [])

    task_rows: list[dict[str, object]] = []
    comment_rows: list[dict[str, object]] = []
    for _, row in selected_cases.iterrows():
        case_id = str(row["case_id"])
        case_records = records_by_case.get(case_id, [])
        task_rows.append(
            {
                "created_at_utc": created_at,
                "reviewer": reviewer,
                "case_id": case_id,
                "anatomy_selection": anatomy_text,
                "task_selection": task_text,
                "status": "pending_review",
                "image_path": case_records[0].image_path if case_records else row.get("image_path", ""),
                "selected_mask_count": len(case_records),
                "slicer_script": str(script_by_case.get(case_id, "")),
                "task_comments": comment_text,
            }
        )
        comment_rows.append(
            {
                "created_at_utc": created_at,
                "reviewer": reviewer,
                "case_id": case_id,
                "anatomy_selection": anatomy_text,
                "task_selection": task_text,
                "qc_pass": "",
                "finding_category": "",
                "severity": "",
                "mask_or_segment": "",
                "slice_or_location": "",
                "comments": "",
                "recommended_action": "",
            }
        )

    structured = {
        "created_at_utc": created_at,
        "reviewer": reviewer,
        "anatomy_selection": list(anatomies),
        "task_selection": list(tasks),
        "task_comments": list(comments or []),
        "case_count": int(selected_cases["case_id"].nunique()) if not selected_cases.empty else 0,
        "mask_count": len(records),
        "cases": task_rows,
        "masks": [asdict(record) for record in records],
    }

    paths = {
        "tasks_csv": outdir / "qc_review_tasks.csv",
        "comments_csv": outdir / "qc_review_comments_template.csv",
        "structured_json": outdir / "qc_review_selection.json",
        "run_log_csv": outdir / "qc_slicer_run_log.csv",
    }
    pd.DataFrame(task_rows).to_csv(paths["tasks_csv"], index=False)
    pd.DataFrame(comment_rows).to_csv(paths["comments_csv"], index=False)
    paths["structured_json"].write_text(json.dumps(structured, indent=2), encoding="utf-8")
    run_log = pd.DataFrame(
        [
            {
                "created_at_utc": created_at,
                "reviewer": reviewer,
                "case_count": structured["case_count"],
                "mask_count": len(records),
                "anatomy_selection": anatomy_text,
                "task_selection": task_text,
                "task_comments": comment_text,
            }
        ]
    )
    if paths["run_log_csv"].exists():
        prior = pd.read_csv(paths["run_log_csv"])
        run_log = pd.concat([prior, run_log], ignore_index=True)
    run_log.to_csv(paths["run_log_csv"], index=False)
    return paths


def launch_slicer(script_path: str | Path, slicer_executable: str | Path | None = None) -> subprocess.Popen:
    """Launch 3D Slicer with a generated Python script."""
    executable = find_slicer_executable(slicer_executable)
    if executable is None:
        raise FileNotFoundError(
            "Could not find 3D Slicer. Pass --slicer-executable or set SLICER_EXECUTABLE."
        )
    app_path = _slicer_app_from_executable(executable)
    if app_path is not None:
        return subprocess.Popen(
            ["open", "-n", "-a", str(app_path), "--args", "--ignore-slicerrc", "--python-script", str(script_path)]
        )
    return subprocess.Popen([str(executable), "--ignore-slicerrc", "--python-script", str(script_path)])


def find_slicer_executable(slicer_executable: str | Path | None = None) -> Path | None:
    """Find a Slicer executable on macOS or PATH."""
    if slicer_executable:
        path = Path(slicer_executable)
        return path if path.exists() else None
    env_path = os.environ.get("SLICER_EXECUTABLE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    path_from_shell = shutil.which("Slicer")
    if path_from_shell:
        return Path(path_from_shell)
    candidates = sorted(glob.glob("/Applications/Slicer*.app/Contents/MacOS/Slicer"))
    if candidates:
        return Path(candidates[-1])
    default = Path("/Applications/Slicer.app/Contents/MacOS/Slicer")
    return default if default.exists() else None


def _slicer_app_from_executable(executable: Path) -> Path | None:
    parts = executable.parts
    for index, part in enumerate(parts):
        if part.endswith(".app"):
            return Path(*parts[: index + 1])
    return None


def default_feature_paths(outputs_root: str | Path) -> list[Path]:
    """Find likely feature/QC tables under an outputs root."""
    root = Path(outputs_root)
    patterns = [
        root / "features" / "*.csv",
        root / "qc" / "*.csv",
        root / "*" / "features" / "*.csv",
        root / "*" / "qc" / "*.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(glob_paths(pattern))
    return sorted({path for path in paths if path.exists()})


def glob_paths(pattern: Path) -> list[Path]:
    return [Path(path) for path in glob.glob(str(pattern))]


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    manifest_base = manifest_path.parent if manifest_path else project_root
    anatomies = args.anatomy or ["aorta"]
    tasks = args.task or ["segmentation"]

    cases = load_case_table(manifest_path, args.clinical_table, outputs_root)
    feature_paths = [Path(path) for path in args.feature_table] or default_feature_paths(outputs_root)
    selected_cases = select_cases(
        cases,
        case_ids=args.case_id,
        filters=args.filter,
        feature_tables=feature_paths,
        outlier_features=args.outlier_feature,
        outlier_method=args.outlier_method,
        outlier_direction=args.outlier_direction,
        outlier_quantile=args.outlier_quantile,
        outlier_z=args.outlier_z,
        outlier_top_n=args.outlier_top_n,
    )
    if args.max_cases:
        selected_cases = selected_cases.head(args.max_cases)

    records = discover_mask_records(
        selected_cases,
        anatomies=anatomies,
        tasks=tasks,
        outputs_root=outputs_root,
        project_root=project_root,
        manifest_base=manifest_base,
    )
    output_dir = Path(args.outdir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    selection_path = write_selection_table(records, output_dir / "qc_slicer_selection.csv")
    scripts = write_slicer_scripts(records, output_dir / "slicer_scripts")
    review_paths = write_review_outputs(
        selected_cases=selected_cases,
        records=records,
        scripts=scripts,
        output_dir=output_dir,
        reviewer=args.reviewer,
        anatomies=anatomies,
        tasks=tasks,
        comments=args.task_comment,
    )
    print(f"Selected cases: {selected_cases['case_id'].nunique()}")
    print(f"Selected masks: {len(records)}")
    print(f"Selection table: {selection_path}")
    print(f"Review tasks: {review_paths['tasks_csv']}")
    print(f"Comments template: {review_paths['comments_csv']}")
    print(f"Structured selection: {review_paths['structured_json']}")
    print(f"Reviewer log: {review_paths['run_log_csv']}")
    if scripts:
        print(f"Slicer scripts: {output_dir / 'slicer_scripts'}")
        launcher = write_slicer_launcher(scripts, output_dir, args.case_index)
        if launcher:
            print(f"Slicer launcher: {launcher}")
    else:
        print("No Slicer scripts were written because no matching masks were found.")

    if args.open_slicer and scripts:
        index = min(max(args.case_index, 0), len(scripts) - 1)
        launch_slicer(scripts[index], args.slicer_executable)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Case manifest with case_id and image_path columns.")
    parser.add_argument("--clinical-table", help="Optional clinical variables table keyed by case_id.")
    parser.add_argument("--outputs-root", default="outputs", help="Pipeline output root or one run output directory.")
    parser.add_argument("--project-root", default=".", help="Project root for resolving relative paths.")
    parser.add_argument("--outdir", default="outputs/qc_slicer", help="Directory for selection CSV and scripts.")
    parser.add_argument("--reviewer", default=getpass.getuser(), help="Reviewer name/ID written to QC logs.")
    parser.add_argument("--task-comment", action="append", default=[], help="Structured task note/comment. Repeatable.")
    parser.add_argument("--anatomy", action="append", default=[], help="Anatomy to review. Repeatable.")
    parser.add_argument("--task", action="append", default=[], help="Task/mask family to review. Repeatable.")
    parser.add_argument("--case-id", action="append", default=[], help="Specific case ID to include. Repeatable.")
    parser.add_argument("--filter", action="append", default=[], help="Case/clinical filter, e.g. SLAO=1 or age>=70.")
    parser.add_argument("--feature-table", action="append", default=[], help="Feature/QC CSV for outlier selection.")
    parser.add_argument("--outlier-feature", action="append", default=[], help="Feature selector, optionally region:feature.")
    parser.add_argument("--outlier-method", choices=["quantile", "zscore", "top-n"], default="quantile")
    parser.add_argument("--outlier-direction", choices=["high", "low", "both"], default="both")
    parser.add_argument("--outlier-quantile", type=float, default=0.95)
    parser.add_argument("--outlier-z", type=float, default=3.0)
    parser.add_argument("--outlier-top-n", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--open-slicer", action="store_true", help="Open the selected case in 3D Slicer.")
    parser.add_argument("--case-index", type=int, default=0, help="Which selected case script to open.")
    parser.add_argument("--slicer-executable", help="Path to 3D Slicer executable.")
    return parser


def _discover_cases_from_outputs(outputs_root: Path) -> pd.DataFrame:
    case_ids: set[str] = set()
    for mask_dir in list(outputs_root.glob("masks/*")) + list(outputs_root.glob("*/masks/*")):
        if mask_dir.is_dir():
            case_ids.add(mask_dir.name)
    return pd.DataFrame({"case_id": sorted(case_ids), "image_path": ""})


def _apply_filter_expression(frame: pd.DataFrame, expression: str) -> pd.DataFrame:
    operators = [">=", "<=", "!=", "=", ">", "<"]
    operator = next((candidate for candidate in operators if candidate in expression), None)
    if operator is None:
        raise ValueError(f"Unsupported filter expression: {expression}")
    column, raw_value = [part.strip() for part in expression.split(operator, 1)]
    if column not in frame.columns:
        raise ValueError(f"Filter column not found: {column}")
    series = frame[column]
    if operator == "=":
        values = [value.strip() for value in raw_value.split(",")]
        return frame[series.astype(str).isin(values)]
    if operator == "!=":
        values = [value.strip() for value in raw_value.split(",")]
        return frame[~series.astype(str).isin(values)]
    numeric = pd.to_numeric(series, errors="coerce")
    value = float(raw_value)
    if operator == ">":
        return frame[numeric > value]
    if operator == "<":
        return frame[numeric < value]
    if operator == ">=":
        return frame[numeric >= value]
    if operator == "<=":
        return frame[numeric <= value]
    raise ValueError(f"Unsupported filter operator: {operator}")


def _feature_rows_for_selector(features: pd.DataFrame, selector: str) -> pd.DataFrame:
    if features.empty:
        return features
    parts = [part.strip() for part in selector.split(":")]
    frame = features.copy()
    if len(parts) == 1:
        exact = frame[frame["feature_name"].astype(str) == parts[0]]
        if not exact.empty:
            return exact
        return frame[frame["feature_name"].astype(str).str.contains(parts[0], case=False, regex=False)]
    if len(parts) >= 2:
        frame = frame[frame["region"].astype(str) == parts[0]]
        frame = frame[frame["feature_name"].astype(str) == parts[1]]
    if len(parts) >= 3:
        frame = frame[frame["threshold_if_applicable"].astype(str) == parts[2]]
    return frame


def _image_path_for_row(row: pd.Series, project_root: Path, manifest_base: Path) -> Path:
    for column in ["image_path", "cta_path", "volume_path", "ct_path"]:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return _resolve_path(value, [manifest_base, project_root])
    return Path("")


def _manifest_mask_paths(row: pd.Series, project_root: Path, manifest_base: Path) -> list[Path]:
    paths: list[Path] = []
    for column, value in row.items():
        column_lower = str(column).lower()
        if not any(token in column_lower for token in ["mask_path", "segmentation_path", "label_path"]):
            continue
        if isinstance(value, str) and value.strip():
            paths.append(_resolve_path(value, [manifest_base, project_root]))
    return paths


def _output_mask_paths(outputs_root: Path, case_id: str) -> list[Path]:
    return [path for path in _all_output_mask_paths(outputs_root, case_id) if _include_output_mask_in_qc(path)]


def _all_output_mask_paths(outputs_root: Path, case_id: str) -> list[Path]:
    case_dirs = [outputs_root / "masks" / case_id]
    case_dirs.extend(outputs_root.glob(f"*/masks/{case_id}"))
    paths: list[Path] = []
    for case_dir in case_dirs:
        if case_dir.exists():
            paths.extend(case_dir.glob("*.nii"))
            paths.extend(case_dir.glob("*.nii.gz"))
    return paths


def _include_output_mask_in_qc(path: Path) -> bool:
    lower = path.name.lower()
    if "boundary" in lower:
        return False
    if "calcification_aorta_wall_dynamic_seed" in lower and lower.endswith("_candidate.nii.gz"):
        return True
    if lower.endswith("_aortic_wall_hu_refined_aorta_trace.nii.gz"):
        return True
    if lower.endswith("_aortic_wall_candidate_from_fat_lumen.nii.gz"):
        return True
    if lower.endswith("_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz"):
        return True
    if "_lumen_hu_label_" in lower and lower.endswith(".nii.gz"):
        return True
    if lower.endswith("_periaortic_fat_0_2mm.nii.gz") or lower.endswith("_periaortic_fat_2_5mm.nii.gz"):
        return True
    if "lumen_protrusion" in lower and "_depth_ge_" in lower and lower.endswith("_labels_3d.nii.gz"):
        return True
    return False


def _resolve_path(path_text: str, bases: list[Path]) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def _looks_like_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def infer_anatomy(path: Path, allowed: list[str]) -> str | None:
    lower = path.name.lower()
    if "all" in allowed:
        return _first_matching_alias(lower, ANATOMY_ALIASES) or "unknown"
    for anatomy in allowed:
        tokens = ANATOMY_ALIASES.get(anatomy, [anatomy])
        if any(token in lower for token in tokens):
            return anatomy
    return None


def infer_task(path: Path, allowed: list[str]) -> str | None:
    lower = path.name.lower()
    if lower.endswith("_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz"):
        if "lumen_protrusion" in allowed:
            return "lumen_protrusion"
        if "all" in allowed:
            return "segmentation"
    if "all" in allowed:
        return _first_matching_alias(lower, TASK_ALIASES) or "segmentation"
    for task in allowed:
        tokens = TASK_ALIASES.get(task, [task])
        if task == "segmentation" and any(token in lower for token in ["calcification", "calcium", "shell"]):
            continue
        if any(token in lower for token in tokens):
            return task
    return None


def infer_category(path: Path, anatomy: str, task: str) -> str:
    lower = path.name.lower()
    if "lumen_core" in lower or lower.endswith("aorta_mask_cleaned.nii.gz"):
        return "artery"
    if "aortic_wall_hu_refined_aorta_trace" in lower:
        return "artery"
    if "search_band" in lower or ("wall_band" in lower and "calcification" not in lower):
        return "tissue"
    if "calcification" in lower or ("calcium" in lower and "search_band" not in lower):
        return "bone"
    if "aortic_wall_contrast_lumen" in lower:
        return "lumen"
    if "_lumen_hu_label_" in lower:
        return "lumen"
    if "aortic_wall_candidate_from_fat_lumen" in lower:
        return "wall"
    if any(token in lower for token in ["fat", "adipose"]):
        return "fat"
    if "outward_ulcer_like" in lower:
        return "ulcer"
    if "inward" in lower and ("lumen_protrusion" in lower or "protrusion" in lower):
        return "protrusion"
    if "lumen_protrusion" in lower or "protrusion" in lower:
        return "shape"
    if "bone" in lower:
        return "bone"
    if any(token in lower for token in ["wall", "shell", "tissue", "peri"]):
        return "tissue"
    if task == "flow_dynamics":
        return "flow"
    if task == "shape":
        return "shape"
    if anatomy in {"aorta", "carotid", "vertebral", "artery"}:
        return "artery"
    return "other"


def _first_matching_alias(text: str, aliases: dict[str, list[str]]) -> str | None:
    for name, tokens in aliases.items():
        if any(token in text for token in tokens):
            return name
    return None


def _label_for_mask(path: Path, case_id: str) -> str:
    stem = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    prefix = f"{case_id}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return _short_qc_label(stem)


def _short_qc_label(stem: str) -> str:
    if stem in {"aorta_vista_trace", "aorta_highres_trace", "aorta_trace"}:
        return "Aorta"
    if stem == "calcification_aorta_wall_dynamic_seed500HU_candidate":
        return "Bone"
    if stem == "aortic_wall_contrast_lumen_from_centerline_hu":
        return "Lumen"
    if stem.startswith("lumen_hu_label_") or "_lumen_hu_label_" in stem:
        return _short_lumen_hu_label(stem)
    if stem == "aortic_wall_hu_refined_aorta_trace":
        return "Aorta HU"
    if stem == "aortic_wall_candidate_from_fat_lumen":
        return "Wall"
    if stem == "periaortic_fat_0_2mm":
        return "Fat 0-2"
    if stem == "periaortic_fat_2_5mm":
        return "Fat 2-5"
    if stem.startswith("wall_lumen_protrusion_inward_") and "_depth_ge_" in stem:
        return _short_protrusion_label(stem, prefix="P")
    if stem.startswith("wall_lumen_protrusion_outward_ulcer_like_") and "_depth_ge_" in stem:
        return _short_protrusion_label(stem, prefix="U")
    if stem.startswith("lumen_protrusion_inward_") and "_depth_ge_" in stem:
        return _short_protrusion_label(stem, prefix="P")
    if stem.startswith("lumen_protrusion_outward_ulcer_like_") and "_depth_ge_" in stem:
        return _short_protrusion_label(stem, prefix="U")
    return stem


def _short_protrusion_label(stem: str, prefix: str) -> str:
    if "_aorta_surface_native_" in stem:
        source = "surf"
    elif "_aorta_surface_core_" in stem:
        source = "core"
    else:
        source = "proj"
    threshold = stem.split("_depth_ge_", maxsplit=1)[1].split("mm", maxsplit=1)[0].replace("p", ".")
    return f"{prefix}{threshold} {source}"


def _short_lumen_hu_label(stem: str) -> str:
    method = stem.removeprefix("lumen_hu_")
    if "_lumen_hu_" in stem:
        method = stem.split("_lumen_hu_", maxsplit=1)[-1]
    labels = {
        "label_lumen_p25_500": "L-P25",
        "label_lumen_p50_500": "L-P50",
        "label_lumen_p75_500": "L-P75",
        "label_lumen_p90_500": "L-P90",
        "label_youden_500": "L-Youden",
        "label_wall_p99_500": "L-W99",
    }
    return labels.get(method, method)


def _qc_opacity(category: str) -> float:
    if category == "artery":
        return 0.35
    if category in {"protrusion", "ulcer", "bone", "fat", "lumen", "wall"}:
        return 0.95
    if category == "tissue":
        return 0.55
    return 0.8


def _qc_fill_opacity(category: str) -> float:
    if category == "artery":
        return 0.05
    if category == "tissue":
        return 0.45
    return 0.95


def _records_by_case(records: list[MaskRecord]) -> dict[str, list[MaskRecord]]:
    grouped: dict[str, list[MaskRecord]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(record)
    return grouped


def _slicer_script(case_id: str, records: list[MaskRecord]) -> str:
    if not records:
        raise ValueError("At least one record is required for a Slicer script.")
    image_path = records[0].image_path
    masks = []
    for record in records:
        color = CATEGORY_COLORS.get(record.category, CATEGORY_COLORS["other"])
        masks.append(
            {
                "path": record.mask_path,
                "label": record.label,
                "category": record.category,
                "color": color,
                "opacity": _qc_opacity(record.category),
                "fill_opacity": _qc_fill_opacity(record.category),
                "outline_opacity": 1.0,
            }
        )
    return f'''# Auto-generated by aorta_cta_radiomics.qc_slicer
from pathlib import Path

import slicer

slicer.mrmlScene.Clear(0)

CASE_ID = {case_id!r}
IMAGE_PATH = {image_path!r}
MASKS = {json.dumps(masks, indent=2)}
SCRIPT_PATH = Path(globals().get("__file__", ".")).resolve()
STATUS_DIR = SCRIPT_PATH.parent if SCRIPT_PATH.name != "." else Path.cwd()
STATUS_PATH = str(STATUS_DIR / (CASE_ID + "_slicer_loader_status.txt"))


def log_status(*parts):
    text = " ".join(str(part) for part in parts)
    print(text)
    try:
        with open(STATUS_PATH, "a", encoding="utf-8") as handle:
            handle.write(text + "\\n")
    except Exception:
        pass


def segment_ids(segmentation):
    return [segmentation.GetNthSegmentID(index) for index in range(segmentation.GetNumberOfSegments())]


try:
    Path(STATUS_PATH).write_text("", encoding="utf-8")
except Exception:
    pass

volume_node = slicer.util.loadVolume(IMAGE_PATH, {{"name": CASE_ID + "_CTA"}})
if volume_node:
    display = volume_node.GetDisplayNode()
    if display:
        display.AutoWindowLevelOff()
        display.SetWindow(900)
        display.SetLevel(250)
    try:
        slicer.util.setSliceViewerLayers(background=volume_node, fit=True)
    except TypeError:
        slicer.util.setSliceViewerLayers(background=volume_node)
    log_status("Loaded CTA:", IMAGE_PATH)
else:
    log_status("Could not load CTA:", IMAGE_PATH)

segmentation_logic = slicer.modules.segmentations.logic()
loaded_segments = 0
for spec in MASKS:
    label_node = slicer.util.loadLabelVolume(spec["path"], {{"name": spec["label"] + "_label"}})
    if not label_node:
        log_status("Could not load mask:", spec["path"])
        continue
    label_node.SetDisplayVisibility(False)
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", spec["label"])
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetDisplayVisibility(True)
    if volume_node:
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    segmentation_logic.ImportLabelmapToSegmentationNode(label_node, segmentation_node)
    display_node = segmentation_node.GetDisplayNode()
    if display_node:
        display_node.SetVisibility(True)
        display_node.SetOpacity3D(float(spec["opacity"]))
        display_node.SetVisibility2DFill(True)
        display_node.SetVisibility2DOutline(True)
        display_node.SetOpacity2DFill(float(spec["fill_opacity"]))
        display_node.SetOpacity2DOutline(float(spec["outline_opacity"]))
    segmentation = segmentation_node.GetSegmentation()
    segment_count = segmentation.GetNumberOfSegments()
    for index in range(segment_count):
        segment = segmentation.GetNthSegment(index)
        segment_id = segmentation.GetNthSegmentID(index)
        segment.SetName(spec["label"] if segment_count == 1 else str(index + 1).zfill(3))
        segment.SetColor(float(spec["color"][0]), float(spec["color"][1]), float(spec["color"][2]))
        if display_node:
            display_node.SetSegmentVisibility(segment_id, True)
            display_node.SetSegmentOpacity3D(segment_id, float(spec["opacity"]))
            display_node.SetSegmentOpacity2DFill(segment_id, float(spec["fill_opacity"]))
            display_node.SetSegmentOpacity2DOutline(segment_id, float(spec["outline_opacity"]))
    segmentation.Modified()
    segmentation_node.Modified()
    if display_node:
        display_node.Modified()
    loaded_segments += segment_count
    log_status("Loaded mask:", spec["label"], "segments:", segment_count, "path:", spec["path"])
    slicer.mrmlScene.RemoveNode(label_node)

try:
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
except Exception:
    pass
try:
    slicer.util.setSliceViewerLayers(background=volume_node, fit=True)
except Exception:
    pass
try:
    slicer.util.resetSliceViews()
except Exception:
    pass
try:
    slicer.util.selectModule("Segmentations")
except Exception:
    pass
slicer.app.processEvents()
log_status("Loaded QC scene for", CASE_ID, "with", len(MASKS), "masks and", loaded_segments, "segments")
'''


if __name__ == "__main__":
    main()
