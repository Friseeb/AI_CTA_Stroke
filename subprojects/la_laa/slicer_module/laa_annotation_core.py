"""Pure-Python core for the LAA Completion & SLAAO Annotation Assistant.

This module holds everything that does NOT need 3D Slicer: the segmentation
label contract, the prompt schema, pilot / reproducibility metrics, session
logging, the on-disk output layout, and the MONAILabel inference-request
builder. It must stay importable in a plain Python interpreter (no `slicer`,
`vtk`, or `qt` imports) so it can be unit-tested in CI.

The Slicer scripted module (`LAACompletionAssistant.py`) imports these helpers
and is responsible only for the GUI / scene interaction.

Design philosophy: pilot-first. The first goal is a reproducible annotation SOP
with feasibility, timing, and interobserver-reproducibility metrics, NOT model
training. See `../docs/SOP.md`.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PLUGIN_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Segmentation label contract
# ---------------------------------------------------------------------------
#
# Label 1 (Whole LAA) is the primary annotation target: ostial boundary to the
# most distal tip, including dominant + secondary lobes and distal
# hypoattenuated regions. Label 2 (SLAAO Type 1 region) is a GEOMETRIC subregion
# nested inside Label 1; HU analysis is downstream, so the reader never applies
# an HU threshold. Labels 3-7 are optional during the pilot and primarily
# support future MONAI training / error analysis.

WHOLE_LAA_LABEL = 1
TYPE1_LABEL = 2

LAA_LABEL_CONTRACT: dict[int, dict[str, Any]] = {
    1: {"name": "Whole LAA", "color": (0.90, 0.20, 0.20), "primary": True},
    2: {"name": "SLAAO Type 1 region", "color": (1.00, 0.80, 0.10), "nested_in": 1},
    3: {"name": "LA body", "color": (0.20, 0.55, 0.90), "primary": False},
    4: {"name": "Pulmonary veins", "color": (0.20, 0.80, 0.80), "primary": False},
    5: {"name": "Coronary artery", "color": (0.85, 0.40, 0.90), "primary": False},
    6: {"name": "Aorta / pulmonary artery", "color": (0.95, 0.55, 0.20), "primary": False},
    7: {"name": "Other hard-negative", "color": (0.55, 0.55, 0.55), "primary": False},
}

PRIMARY_LABELS = (WHOLE_LAA_LABEL, TYPE1_LABEL)


def label_contract_metadata() -> dict[str, Any]:
    """Return the LAA label contract as serializable metadata."""
    return {
        "label_contract": "laa_completion_v1",
        "primary_target": WHOLE_LAA_LABEL,
        "labels": {
            str(value): {
                "name": spec["name"],
                "color": list(spec["color"]),
                "primary": value in PRIMARY_LABELS,
            }
            for value, spec in LAA_LABEL_CONTRACT.items()
        },
    }


def validate_label_values(values: Sequence[int]) -> list[str]:
    """Return warnings for a candidate LAA labelmap's set of voxel values.

    The whole-LAA label (1) must be present; Type 1 (2) is optional and may be
    empty. Values outside the contract are flagged.
    """
    warnings: list[str] = []
    nonzero = {int(v) for v in values if int(v) != 0}
    if WHOLE_LAA_LABEL not in nonzero:
        warnings.append(f"Whole-LAA label {WHOLE_LAA_LABEL} is absent (no LAA segmented).")
    extra = nonzero - set(LAA_LABEL_CONTRACT)
    if extra:
        warnings.append(f"Label values outside the contract: {sorted(extra)}.")
    return warnings


# ---------------------------------------------------------------------------
# Prompt schema (positive / negative anatomical prompts)
# ---------------------------------------------------------------------------

PROMPT_TYPES = ("positive", "negative")

POSITIVE_CATEGORIES = (
    "distal_tip",
    "distal_lobe",
    "missed_appendage",
    "type1_region",
)

NEGATIVE_CATEGORIES = (
    "la_body",
    "pulmonary_vein",
    "coronary_artery",
    "aorta",
    "pulmonary_artery",
    "myocardium",
    "artifact",
)

PROMPT_CATEGORIES = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES


def validate_prompt(prompt_type: str, category: str) -> None:
    """Raise ValueError if a prompt type / category pair is invalid."""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(
            f"Invalid prompt type {prompt_type!r}. Allowed: {', '.join(PROMPT_TYPES)}."
        )
    allowed = POSITIVE_CATEGORIES if prompt_type == "positive" else NEGATIVE_CATEGORIES
    if category not in allowed:
        raise ValueError(
            f"Category {category!r} is not valid for {prompt_type} prompts. "
            f"Allowed: {', '.join(allowed)}."
        )


@dataclass
class Prompt:
    """A single positive or negative annotation prompt.

    `coordinate` is an (R, A, S) point in patient space (mm). `timestamp` is set
    automatically when omitted.
    """

    prompt_type: str
    category: str
    coordinate: tuple[float, float, float]
    model_used: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        validate_prompt(self.prompt_type, self.category)
        self.coordinate = tuple(float(c) for c in self.coordinate)  # type: ignore[assignment]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_type": self.prompt_type,
            "category": self.category,
            "coordinate": list(self.coordinate),
            "model_used": self.model_used,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prompt":
        return cls(
            prompt_type=d["prompt_type"],
            category=d["category"],
            coordinate=tuple(d["coordinate"]),
            model_used=d.get("model_used", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class PromptLog:
    """Ordered collection of prompts placed during a case."""

    case_id: str = ""
    reader_id: str = ""
    prompts: list[Prompt] = field(default_factory=list)

    def add(
        self,
        prompt_type: str,
        category: str,
        coordinate: tuple[float, float, float],
        model_used: str = "",
    ) -> Prompt:
        prompt = Prompt(prompt_type, category, coordinate, model_used=model_used)
        self.prompts.append(prompt)
        return prompt

    @property
    def positive_count(self) -> int:
        return sum(1 for p in self.prompts if p.prompt_type == "positive")

    @property
    def negative_count(self) -> int:
        return sum(1 for p in self.prompts if p.prompt_type == "negative")

    @property
    def count(self) -> int:
        return len(self.prompts)

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.prompts:
            counts[p.category] = counts.get(p.category, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reader_id": self.reader_id,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "by_category": self.by_category(),
            "prompts": [p.to_dict() for p in self.prompts],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptLog":
        return cls(
            case_id=d.get("case_id", ""),
            reader_id=d.get("reader_id", ""),
            prompts=[Prompt.from_dict(p) for p in d.get("prompts", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "PromptLog":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Pilot metrics
# ---------------------------------------------------------------------------

IMAGE_QUALITY_SCALE = (1, 2, 3, 4, 5)  # 1 = non-diagnostic, 5 = excellent


@dataclass
class PilotMetrics:
    """Per-case feasibility / timing / confidence metrics for the pilot study."""

    case_id: str = ""
    reader_id: str = ""
    model_used: str = ""

    annotation_time_s: Optional[float] = None
    correction_time_s: Optional[float] = None

    prompt_count: int = 0
    positive_prompt_count: int = 0
    negative_prompt_count: int = 0
    edit_count: int = 0

    segmentation_confidence: Optional[float] = None  # 0-1
    type1_confidence: Optional[float] = None  # 0-1
    image_quality: Optional[int] = None  # 1-5
    type1_present: Optional[bool] = None

    annotation_date: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.annotation_date:
            self.annotation_date = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PilotMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "PilotMetrics":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> list[str]:
        """Return validation warnings (empty list = valid)."""
        warnings: list[str] = []
        if not self.case_id:
            warnings.append("case_id is empty")
        for name in ("segmentation_confidence", "type1_confidence"):
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                warnings.append(f"{name} {val} out of range [0, 1]")
        if self.image_quality is not None and self.image_quality not in IMAGE_QUALITY_SCALE:
            warnings.append(
                f"image_quality {self.image_quality} not in {list(IMAGE_QUALITY_SCALE)}"
            )
        expected_total = self.positive_prompt_count + self.negative_prompt_count
        if self.prompt_count != expected_total:
            warnings.append(
                f"prompt_count {self.prompt_count} != positive+negative {expected_total}"
            )
        return warnings


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
#
#   <case_dir>/laa_annotation/[<reader_id>/]
#     candidate_masks/   manual_masks/   type1_masks/
#     iterations/        logs/           screenshots/   metrics/

# Candidate prior sources -> filename stems to look for (in candidate_masks/ or a
# prior-fusion dir). TotalSegmentator's `total` task emits the LAA as
# `atrial_appendage_left`; prior fusion restages it as `*_totalseg_laa`.
CANDIDATE_SOURCE_FILES: dict[str, tuple[str, ...]] = {
    "VISTA-3D": ("vista3d_laa", "vista3d_prompt"),
    "NUDF": ("nudf_laa",),
    "TotalSegmentator": ("totalseg_laa", "atrial_appendage_left", "left_atrial_appendage"),
    "Consensus": ("consensus_laa",),
}


def resolve_candidate_file(search_dirs: Sequence[str | Path], stems: Sequence[str]) -> Path | None:
    """First existing ``<stem>.nii.gz`` (or ``*_<stem>.nii.gz``) across search dirs."""
    for directory in search_dirs:
        d = Path(directory)
        if not d.is_dir():
            continue
        for stem in stems:
            exact = d / f"{stem}.nii.gz"
            if exact.exists():
                return exact
            hits = sorted(d.glob(f"*_{stem}.nii.gz")) or sorted(d.glob(f"*{stem}*.nii.gz"))
            if hits:
                return hits[0]
    return None


LAA_ANNOTATION_SUBDIR = "laa_annotation"
_OUTPUT_SUBDIRS = (
    "candidate_masks",
    "manual_masks",
    "type1_masks",
    "iterations",
    "logs",
    "screenshots",
    "metrics",
)


@dataclass(frozen=True)
class LaaAnnotationPaths:
    """Standard output paths for one case (and optional reader)."""

    root: Path
    candidate_masks: Path
    manual_masks: Path
    type1_masks: Path
    iterations: Path
    logs: Path
    screenshots: Path
    metrics: Path
    case_id: str
    reader_id: str

    def session_csv(self) -> Path:
        return self.logs / f"{self.case_id}_session.csv"

    def session_json(self) -> Path:
        return self.logs / f"{self.case_id}_session.json"

    def prompt_log(self) -> Path:
        return self.logs / f"{self.case_id}_prompts.json"

    def pilot_metrics(self) -> Path:
        return self.metrics / f"{self.case_id}_pilot.json"

    def whole_laa_mask(self) -> Path:
        return self.manual_masks / f"{self.case_id}_whole_laa.nii.gz"

    def type1_mask(self) -> Path:
        return self.type1_masks / f"{self.case_id}_type1.nii.gz"

    def comparison_mask(self) -> Path:
        """New (corrected) vs old (candidate) labelmap: 1 kept / 2 added / 3 removed."""
        return self.manual_masks / f"{self.case_id}_whole_laa_vs_candidate.nii.gz"

    def comparison_metrics_path(self) -> Path:
        return self.metrics / f"{self.case_id}_candidate_comparison.json"

    def mkdirs(self) -> "LaaAnnotationPaths":
        for sub in (
            self.candidate_masks,
            self.manual_masks,
            self.type1_masks,
            self.iterations,
            self.logs,
            self.screenshots,
            self.metrics,
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return self


def output_paths(
    case_dir: str | Path, case_id: str, reader_id: str = ""
) -> LaaAnnotationPaths:
    """Build the standard output tree for a case.

    When `reader_id` is given (reproducibility mode), outputs are nested under a
    per-reader subfolder so Reader A/B/C annotations stay separate.
    """
    root = Path(case_dir) / LAA_ANNOTATION_SUBDIR
    if reader_id:
        root = root / reader_id
    return LaaAnnotationPaths(
        root=root,
        candidate_masks=root / "candidate_masks",
        manual_masks=root / "manual_masks",
        type1_masks=root / "type1_masks",
        iterations=root / "iterations",
        logs=root / "logs",
        screenshots=root / "screenshots",
        metrics=root / "metrics",
        case_id=case_id,
        reader_id=reader_id,
    )


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------

SESSION_CSV_FIELDS = (
    "timestamp",
    "case_id",
    "reader_id",
    "model_used",
    "annotation_time_s",
    "correction_time_s",
    "prompt_count",
    "positive_prompt_count",
    "negative_prompt_count",
    "edit_count",
    "segmentation_confidence",
    "type1_confidence",
    "image_quality",
    "type1_present",
    "whole_laa_mask",
    "type1_mask",
    "output_dir",
    "notes",
)


def build_session_log(
    *,
    case_id: str,
    reader_id: str,
    pilot: PilotMetrics,
    prompt_log: PromptLog,
    output_dir: str | Path,
    whole_laa_mask: str | Path | None = None,
    type1_mask: str | Path | None = None,
    warnings: list[str] | None = None,
    plugin_version: str = PLUGIN_VERSION,
) -> dict[str, Any]:
    """Build the canonical JSON session log for one finalized annotation.

    Captures every item the spec requires under "Logging": case id, user,
    timestamp, model, prompt coordinates / types / categories, edit count,
    durations, image-quality and confidence scores, final accepted mask, and the
    output folder.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "plugin_version": plugin_version,
        "case_id": case_id,
        "reader_id": reader_id,
        "model_used": pilot.model_used,
        "annotation_time_s": pilot.annotation_time_s,
        "correction_time_s": pilot.correction_time_s,
        "prompt_count": prompt_log.count,
        "positive_prompt_count": prompt_log.positive_count,
        "negative_prompt_count": prompt_log.negative_count,
        "prompts_by_category": prompt_log.by_category(),
        "prompts": [p.to_dict() for p in prompt_log.prompts],
        "edit_count": pilot.edit_count,
        "segmentation_confidence": pilot.segmentation_confidence,
        "type1_confidence": pilot.type1_confidence,
        "image_quality": pilot.image_quality,
        "type1_present": pilot.type1_present,
        "whole_laa_mask": str(whole_laa_mask) if whole_laa_mask else "",
        "type1_mask": str(type1_mask) if type1_mask else "",
        "output_dir": str(output_dir),
        "notes": pilot.notes,
        "warnings": warnings or [],
        **label_contract_metadata(),
    }


def session_csv_row(log: dict[str, Any]) -> dict[str, str]:
    """Flatten a session log into a single CSV row."""
    return {field_name: _csv_value(log.get(field_name)) for field_name in SESSION_CSV_FIELDS}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def append_session_csv(path: str | Path, log: dict[str, Any]) -> None:
    """Append a session row, writing the header when the file is new."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SESSION_CSV_FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow(session_csv_row(log))


# ---------------------------------------------------------------------------
# MONAILabel inference request
# ---------------------------------------------------------------------------


def build_monai_inference_request(
    *,
    image: str,
    model: str,
    prompt_log: PromptLog,
    current_label: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a MONAILabel inference request payload (no network I/O).

    Positive and negative prompt coordinates are split into the `foreground` /
    `background` point lists MONAILabel interactive models expect. The Slicer
    module performs the actual HTTP POST; keeping this pure makes it testable.
    """
    foreground = [list(p.coordinate) for p in prompt_log.prompts if p.prompt_type == "positive"]
    background = [list(p.coordinate) for p in prompt_log.prompts if p.prompt_type == "negative"]
    request: dict[str, Any] = {
        "model": model,
        "image": image,
        "foreground": foreground,
        "background": background,
        "params": dict(params or {}),
    }
    if current_label is not None:
        request["label"] = current_label
    return request


# ---------------------------------------------------------------------------
# Reproducibility metrics
# ---------------------------------------------------------------------------


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    Two empty masks are defined as a perfect match (1.0).
    """
    a_m = _as_bool_mask(a)
    b_m = _as_bool_mask(b)
    denom = a_m.sum() + b_m.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a_m, b_m).sum() / denom)


# New-mask-vs-old-candidate comparison labelmap encoding.
COMPARISON_LABELS = {"unchanged": 1, "added": 2, "removed": 3}


def comparison_labelmap(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Encode the new (corrected) mask against the old (candidate) mask.

    Returns a uint8 labelmap on the shared voxel grid:
    ``0`` background, ``1`` unchanged (kept), ``2`` added by the reader,
    ``3`` removed by the reader. ``old`` and ``new`` must share a shape.
    """
    old_m = _as_bool_mask(old)
    new_m = _as_bool_mask(new)
    if old_m.shape != new_m.shape:
        raise ValueError(
            f"comparison_labelmap shape mismatch: old {old_m.shape} vs new {new_m.shape}"
        )
    out = np.zeros(old_m.shape, dtype=np.uint8)
    out[old_m & new_m] = COMPARISON_LABELS["unchanged"]
    out[new_m & ~old_m] = COMPARISON_LABELS["added"]
    out[old_m & ~new_m] = COMPARISON_LABELS["removed"]
    return out


def comparison_metrics(
    old: np.ndarray, new: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)
) -> dict[str, Any]:
    """Volume/overlap metrics for the new (corrected) mask vs the old candidate."""
    old_m = _as_bool_mask(old)
    new_m = _as_bool_mask(new)
    voxel_ml = float(np.prod([float(s) for s in spacing])) / 1000.0
    inter = int(np.logical_and(old_m, new_m).sum())
    union = int(np.logical_or(old_m, new_m).sum())
    o = int(old_m.sum())
    n = int(new_m.sum())
    added = int((new_m & ~old_m).sum())
    removed = int((old_m & ~new_m).sum())
    return {
        "dice": dice(old_m, new_m),
        "jaccard": (inter / union) if union else 1.0,
        "old_voxels": o,
        "new_voxels": n,
        "added_voxels": added,
        "removed_voxels": removed,
        "old_volume_ml": o * voxel_ml,
        "new_volume_ml": n * voxel_ml,
        "added_volume_ml": added * voxel_ml,
        "removed_volume_ml": removed * voxel_ml,
        "volume_change_ml": (n - o) * voxel_ml,
        "volume_change_pct": ((n - o) / o * 100.0) if o else None,
    }


def _surface_distances(
    a: np.ndarray, b: np.ndarray, spacing: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dist a-surface -> b-surface, dist b-surface -> a-surface) in mm."""
    from scipy import ndimage

    a_m = _as_bool_mask(a)
    b_m = _as_bool_mask(b)
    spacing = tuple(float(s) for s in spacing)

    def _border(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask
        eroded = ndimage.binary_erosion(mask)
        return mask & ~eroded

    a_border = _border(a_m)
    b_border = _border(b_m)
    # distance transform of the complement gives distance-to-nearest-surface
    dt_b = ndimage.distance_transform_edt(~b_border, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~a_border, sampling=spacing)
    return dt_b[a_border], dt_a[b_border]


def hd95(a: np.ndarray, b: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile (robust) Hausdorff distance in mm.

    Returns 0.0 when both masks are empty, and inf when exactly one is empty.
    """
    a_m = _as_bool_mask(a)
    b_m = _as_bool_mask(b)
    if not a_m.any() and not b_m.any():
        return 0.0
    if not a_m.any() or not b_m.any():
        return float("inf")
    d_ab, d_ba = _surface_distances(a_m, b_m, spacing)
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def surface_dice(
    a: np.ndarray,
    b: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0,
) -> float:
    """Surface Dice at a tolerance: fraction of surface within `tolerance_mm`.

    Two empty masks return 1.0; exactly one empty returns 0.0.
    """
    a_m = _as_bool_mask(a)
    b_m = _as_bool_mask(b)
    if not a_m.any() and not b_m.any():
        return 1.0
    if not a_m.any() or not b_m.any():
        return 0.0
    d_ab, d_ba = _surface_distances(a_m, b_m, spacing)
    n = d_ab.size + d_ba.size
    if n == 0:
        return 1.0
    within = (d_ab <= tolerance_mm).sum() + (d_ba <= tolerance_mm).sum()
    return float(within / n)


def pairwise_metrics(
    a: np.ndarray,
    b: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    """Dice / Surface Dice / HD95 between two masks."""
    return {
        "dice": dice(a, b),
        "surface_dice": surface_dice(a, b, spacing, tolerance_mm),
        "hd95_mm": hd95(a, b, spacing),
    }


def interrater_report(
    masks_by_reader: dict[str, np.ndarray],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0,
) -> dict[str, Any]:
    """Compute pairwise reproducibility metrics across readers for one case.

    `masks_by_reader` maps reader id -> binary mask (same grid). Returns per-pair
    metrics plus the mean of each metric across all pairs.
    """
    readers = sorted(masks_by_reader)
    pairs: list[dict[str, Any]] = []
    for i, ra in enumerate(readers):
        for rb in readers[i + 1 :]:
            m = pairwise_metrics(
                masks_by_reader[ra], masks_by_reader[rb], spacing, tolerance_mm
            )
            pairs.append({"reader_a": ra, "reader_b": rb, **m})

    def _mean(key: str) -> Optional[float]:
        vals = [p[key] for p in pairs if np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else None

    return {
        "readers": readers,
        "n_pairs": len(pairs),
        "tolerance_mm": tolerance_mm,
        "spacing": list(spacing),
        "pairs": pairs,
        "mean_dice": _mean("dice"),
        "mean_surface_dice": _mean("surface_dice"),
        "mean_hd95_mm": _mean("hd95_mm"),
    }


# ---------------------------------------------------------------------------
# Manifest / case-id helpers (shared conventions with vertebral_review_core)
# ---------------------------------------------------------------------------

REPRO_MANIFEST_FIELDS = ("case_id", "reader_id", "mask_path")


def infer_case_id(*names: object) -> str:
    """Infer a BIDS-style case id from node names or file paths."""
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
        cleaned = re.sub(
            r"(_whole_laa|_type1|_laa|_mask|_seg(mentation)?|_cta|_ct)$", "", stem, flags=re.I
        )
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip("_")
        if cleaned:
            return cleaned
    return "laa_case"


def read_repro_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a reproducibility manifest of per-reader finalized masks.

    Required columns: `case_id`, `reader_id`, `mask_path`. `case_id` is inferred
    from `mask_path` when blank.
    """
    manifest_path = Path(path)
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        missing = {"reader_id", "mask_path"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for raw in reader:
            row = {f: (raw.get(f) or "").strip() for f in REPRO_MANIFEST_FIELDS}
            if not row["case_id"]:
                row["case_id"] = infer_case_id(row["mask_path"])
            if row["reader_id"] and row["mask_path"]:
                rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no usable rows: {manifest_path}")
    return rows
