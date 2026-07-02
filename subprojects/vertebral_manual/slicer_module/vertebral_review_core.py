"""Pure-Python helpers for the CTA vertebral review Slicer modules."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PLUGIN_VERSION = "0.4.0"

REVIEW_STATUS_OPTIONS = (
    "in_progress",
    "accepted",
    "needs_second_review",
    "rejected_bad_image",
    "rejected_bad_segmentation",
    "deferred",
)

LEFT_LABEL_VALUE = 1
RIGHT_LABEL_VALUE = 2
LEFT_SEGMENT_NAME = "Vert L"
RIGHT_SEGMENT_NAME = "Vert R"
LEFT_SEGMENT_COLOR = (0.0, 0.75, 0.35)
RIGHT_SEGMENT_COLOR = (0.85, 0.9, 0.0)
LEFT_CURVE_NAME = "vertebral_centerline_L"
RIGHT_CURVE_NAME = "vertebral_centerline_R"
FORAMEN_PRIOR_NAME = "vertebral_foramen_negative_prior"
FORAMEN_PRIOR_COLOR = (0.05, 0.35, 1.0)

REVIEW_CSV_FIELDS = (
    "timestamp",
    "case_id",
    "reviewer_id",
    "review_status",
    "cta_path",
    "label_path",
    "output_label",
    "output_log",
    "centerlines_path",
    "negative_prior_nodes",
    "negative_prior_paths",
    "negative_prior_overlap_voxels",
    "scene_path",
    "notes",
)

MANIFEST_FIELDS = (
    "case_id",
    "cta_path",
    "label_path",
    "foramen_prior_path",
    "reviewer_id",
    "review_status",
    "notes",
)

QUEUE_STATUS_FIELDS = (
    "timestamp",
    "case_id",
    "queue_status",
    "reviewer_id",
    "review_status",
    "cta_path",
    "label_path",
    "foramen_prior_path",
    "output_label",
    "output_log",
    "notes",
)


@dataclass(frozen=True)
class VertebralOutputPaths:
    """Standard output paths for a finalized vertebral review."""

    output_dir: Path
    clean_label: Path
    log_json: Path
    review_csv: Path
    centerlines: Path
    scene: Path


def validate_review_status(status: str) -> str:
    """Return a valid review status or raise a clear error."""
    value = (status or "").strip()
    if value not in REVIEW_STATUS_OPTIONS:
        allowed = ", ".join(REVIEW_STATUS_OPTIONS)
        raise ValueError(f"Invalid review status {status!r}. Allowed values: {allowed}.")
    return value


def normalize_reviewer_id(reviewer_id: str | None, required: bool = True) -> str:
    """Normalize reviewer id for logs."""
    value = (reviewer_id or "").strip()
    if required and not value:
        raise ValueError("Reviewer ID is required before finalizing the case.")
    return value or "unspecified"


def infer_case_id(*names: object) -> str:
    """Infer a case id from node names or file paths."""
    for item in names:
        if item is None:
            continue
        text = str(item)
        if not text:
            continue
        stem = Path(text).name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        else:
            stem = Path(stem).stem
        match = re.search(r"(sub-[A-Za-z0-9]+)", stem)
        if match:
            return match.group(1)
        cleaned = re.sub(r"(_vert(_clean)?|_label(map)?|_seg(mentation)?|_cta|_ct)$", "", stem, flags=re.I)
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip("_")
        if cleaned:
            return cleaned
    return "vertebral_case"


def output_paths(output_dir: str | Path, case_id: str) -> VertebralOutputPaths:
    """Build the standard output paths for one finalized case."""
    outdir = Path(output_dir)
    return VertebralOutputPaths(
        output_dir=outdir,
        clean_label=outdir / f"{case_id}_vert_clean.nii.gz",
        log_json=outdir / f"{case_id}_vert_clean_log.json",
        review_csv=outdir / f"{case_id}_vertebral_review.csv",
        centerlines=outdir / f"{case_id}_vertebral_centerlines.mrk.json",
        scene=outdir / f"{case_id}_vertebral_review.mrml",
    )


def label_contract_metadata() -> dict[str, Any]:
    """Return the bilateral vertebral label contract."""
    return {
        "label_contract": "bilateral_vertebral_v1",
        "labels": {
            str(LEFT_LABEL_VALUE): {
                "name": LEFT_SEGMENT_NAME,
                "side": "left",
                "color": list(LEFT_SEGMENT_COLOR),
            },
            str(RIGHT_LABEL_VALUE): {
                "name": RIGHT_SEGMENT_NAME,
                "side": "right",
                "color": list(RIGHT_SEGMENT_COLOR),
            },
        },
    }


def build_review_log(
    *,
    case_id: str,
    reviewer_id: str,
    review_status: str,
    cta_node: str,
    label_node: str,
    cta_path: str | None,
    label_path: str | None,
    output_paths: VertebralOutputPaths,
    params: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    negative_priors: list[dict[str, Any]] | None = None,
    centerlines_saved: bool = False,
    scene_saved: bool = False,
    plugin_version: str = PLUGIN_VERSION,
) -> dict[str, Any]:
    """Build the canonical JSON review log."""
    status = validate_review_status(review_status)
    reviewer = normalize_reviewer_id(reviewer_id, required=False)
    return {
        "timestamp": datetime.now().isoformat(),
        "plugin_version": plugin_version,
        "case_id": case_id,
        "reviewer_id": reviewer,
        "review_status": status,
        "cta_node": cta_node,
        "label_node": label_node,
        "cta_path": cta_path,
        "label_path": label_path,
        "output_label": str(output_paths.clean_label),
        "output_log": str(output_paths.log_json),
        "review_csv": str(output_paths.review_csv),
        "centerlines_path": str(output_paths.centerlines) if centerlines_saved else "",
        "scene_path": str(output_paths.scene) if scene_saved else "",
        "negative_priors": negative_priors or [],
        "params": params or {},
        "warnings": warnings or [],
        **label_contract_metadata(),
    }


def review_csv_row(log: dict[str, Any]) -> dict[str, str]:
    """Flatten a JSON review log into one CSV row."""
    params = log.get("params") if isinstance(log.get("params"), dict) else {}
    negative_priors = log.get("negative_priors") if isinstance(log.get("negative_priors"), list) else []
    prior_nodes = [str(p.get("node", "")) for p in negative_priors if isinstance(p, dict)]
    prior_paths = [str(p.get("path", "")) for p in negative_priors if isinstance(p, dict)]
    overlap_voxels = sum(
        int(p.get("overlap_voxels", 0))
        for p in negative_priors
        if isinstance(p, dict) and str(p.get("overlap_voxels", "0")).isdigit()
    )
    return {
        "timestamp": str(log.get("timestamp", "")),
        "case_id": str(log.get("case_id", "")),
        "reviewer_id": str(log.get("reviewer_id", "")),
        "review_status": str(log.get("review_status", "")),
        "cta_path": str(log.get("cta_path") or ""),
        "label_path": str(log.get("label_path") or ""),
        "output_label": str(log.get("output_label", "")),
        "output_log": str(log.get("output_log", "")),
        "centerlines_path": str(log.get("centerlines_path", "")),
        "negative_prior_nodes": ";".join(prior_nodes),
        "negative_prior_paths": ";".join(prior_paths),
        "negative_prior_overlap_voxels": str(overlap_voxels),
        "scene_path": str(log.get("scene_path", "")),
        "notes": str(params.get("notes", "")),
    }


def append_review_csv(path: str | Path, log: dict[str, Any]) -> None:
    """Append a review row, writing the header when the file is new."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REVIEW_CSV_FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow(review_csv_row(log))


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a single-case queue manifest."""
    manifest_path = Path(path)
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        missing = {"case_id", "cta_path"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for raw in reader:
            row = {field: (raw.get(field) or "").strip() for field in MANIFEST_FIELDS}
            if not row["case_id"]:
                row["case_id"] = infer_case_id(row["cta_path"], row["label_path"])
            if row["case_id"] and row["cta_path"]:
                rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no usable rows: {manifest_path}")
    return rows


def queue_status_path(manifest_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Return the append-only queue status CSV path."""
    manifest = Path(manifest_path)
    base_dir = Path(output_dir) if output_dir else manifest.parent
    return base_dir / f"{manifest.stem}_queue_status.csv"


def queue_status_row(
    manifest_row: dict[str, str],
    *,
    queue_status: str,
    reviewer_id: str = "",
    review_status: str = "",
    output_label: str = "",
    output_log: str = "",
    notes: str = "",
) -> dict[str, str]:
    """Build one append-only queue status row."""
    return {
        "timestamp": datetime.now().isoformat(),
        "case_id": str(manifest_row.get("case_id", "")),
        "queue_status": queue_status,
        "reviewer_id": reviewer_id or str(manifest_row.get("reviewer_id", "")),
        "review_status": review_status or str(manifest_row.get("review_status", "")),
        "cta_path": str(manifest_row.get("cta_path", "")),
        "label_path": str(manifest_row.get("label_path", "")),
        "foramen_prior_path": str(manifest_row.get("foramen_prior_path", "")),
        "output_label": output_label,
        "output_log": output_log,
        "notes": notes or str(manifest_row.get("notes", "")),
    }


def append_queue_status(path: str | Path, row: dict[str, str]) -> None:
    """Append a queue status event."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(QUEUE_STATUS_FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in QUEUE_STATUS_FIELDS})


def latest_queue_status_by_case(path: str | Path) -> dict[str, dict[str, str]]:
    """Read the latest queue status event per case."""
    csv_path = Path(path)
    latest: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        return latest
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            if case_id:
                latest[case_id] = {k: (v or "") for k, v in row.items()}
    return latest


def first_pending_index(rows: list[dict[str, str]], latest_status: dict[str, dict[str, str]]) -> int:
    """Return the first row not marked completed in the status log."""
    for index, row in enumerate(rows):
        status = latest_status.get(row.get("case_id", ""), {}).get("queue_status", "")
        if status != "completed":
            return index
    return max(len(rows) - 1, 0)


def validate_label_values(values: set[int]) -> list[str]:
    """Return warnings for a candidate bilateral vertebral labelmap."""
    warnings: list[str] = []
    nonzero = {int(v) for v in values if int(v) != 0}
    expected = {LEFT_LABEL_VALUE, RIGHT_LABEL_VALUE}
    missing = expected - nonzero
    extra = nonzero - expected
    if missing:
        warnings.append(f"Missing expected vertebral label values: {sorted(missing)}.")
    if extra:
        warnings.append(f"Unexpected nonzero label values found: {sorted(extra)}.")
    return warnings
