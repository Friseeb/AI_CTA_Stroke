"""Canonical, evidence-gated feature sets.

Four named feature sets select columns by :class:`evidence_registry.EvidenceTier`.
They are the contract behind the ``--feature-set`` CLI flag and the tiered
subset CSVs written by :mod:`stroke_cta_osa.output`.

    core_osa_backed                 -> Tier 1 only
    core_plus_anatomic_extensions   -> Tier 1 + Tier 2
    core_plus_cardiometabolic_ct    -> Tier 1 + Tier 3
    all_features_exploratory        -> Tier 1 + Tier 2 + Tier 3 + Tier 4

Identifier and QC columns are *support* columns: they are appended to every
subset regardless of tier so each subset CSV is self-describing. Crucially,
support columns are NOT evidence features, so adding them never violates the
"core file contains no Tier 2/3/4 features" rule.
"""

from __future__ import annotations

from typing import Iterable

from . import evidence_registry as er
from . import metric_registry as mr
from .evidence_registry import (
    FS_ALL, FS_CORE, FS_CORE_PLUS_ANATOMIC, FS_CORE_PLUS_CARDIOMETABOLIC,
    EvidenceTier,
)

# Ordered list of allowed feature-set names.
ALLOWED_FEATURE_SETS: tuple[str, ...] = (
    FS_CORE, FS_CORE_PLUS_ANATOMIC, FS_CORE_PLUS_CARDIOMETABOLIC, FS_ALL,
)
DEFAULT_MODELING_FEATURE_SET = FS_CORE

# feature-set name -> tiers it admits.
_SET_TIERS: dict[str, tuple[EvidenceTier, ...]] = {
    FS_CORE: (EvidenceTier.TIER_1_CORE_OSA_BACKED,),
    FS_CORE_PLUS_ANATOMIC: (
        EvidenceTier.TIER_1_CORE_OSA_BACKED,
        EvidenceTier.TIER_2_OSA_PLAUSIBLE_CT_ANATOMIC,
    ),
    FS_CORE_PLUS_CARDIOMETABOLIC: (
        EvidenceTier.TIER_1_CORE_OSA_BACKED,
        EvidenceTier.TIER_3_CT_CARDIOMETABOLIC_OR_VASCULAR,
    ),
    FS_ALL: tuple(EvidenceTier),
}

_SET_DESCRIPTIONS: dict[str, str] = {
    FS_CORE: "Primary analysis set — only features with prior adult OSA imaging support.",
    FS_CORE_PLUS_ANATOMIC: "Tier 1 + anatomically-grounded CT extensions (mechanistic/secondary).",
    FS_CORE_PLUS_CARDIOMETABOLIC: "Tier 1 + cardiometabolic/vascular CT adiposity (stroke/MACE/AF/AFDAS risk).",
    FS_ALL: "Everything implemented — Tier 1-4, radiomics, engineered ratios, untrained composites.",
}


# --- Support (non-evidence) columns ----------------------------------------

# Identifiers + provenance always carried with every subset.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "pipeline", "pipeline_version", "config_hash", "processing_timestamp",
    "patient_id", "study_id", "scan_id", "input_path_hash", "input_kind",
    "airway_source", "airway_provider_notes",
)


def _qc_columns() -> list[str]:
    """All registry columns in the qc family (coverage, flags, reliability)."""
    return [m.feature_name for m in mr.all_metrics() if m.family == "qc"]


def _method_and_confidence_columns() -> list[str]:
    """Per-family method/confidence/availability strings worth carrying along.

    These describe *how* a feature was produced; they are not themselves
    evidence-tiered measurements, so they are safe to include in any subset.
    """
    out: list[str] = []
    for m in mr.all_metrics():
        n = m.feature_name
        if (n.endswith("_method") or n.endswith("_confidence")
                or n.endswith("_mask_available") or n.endswith("_roi_method")
                or n.endswith("_contrast_sensitive")):
            out.append(n)
    return out


def support_columns() -> list[str]:
    """Identifier + QC + method/confidence columns, de-duplicated, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for col in (*IDENTIFIER_COLUMNS, *_qc_columns(), *_method_and_confidence_columns()):
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


# --- Public API ------------------------------------------------------------

def list_feature_sets() -> list[str]:
    return list(ALLOWED_FEATURE_SETS)


def describe(feature_set: str) -> str:
    return _SET_DESCRIPTIONS.get(feature_set, "")


def tiers_for(feature_set: str) -> tuple[EvidenceTier, ...]:
    if feature_set not in _SET_TIERS:
        raise ValueError(
            f"unknown feature set {feature_set!r}; "
            f"choose one of {ALLOWED_FEATURE_SETS}"
        )
    return _SET_TIERS[feature_set]


def evidence_features(feature_set: str, *, implemented_only: bool = False
                      ) -> list[str]:
    """Evidence feature names admitted by a feature set (tier membership)."""
    tiers = set(tiers_for(feature_set))
    out: list[str] = []
    for e in er.all_evidence():
        if e.evidence_tier in tiers and (e.implemented or not implemented_only):
            out.append(e.feature_name)
    return out


def member_columns(feature_set: str, available: Iterable[str]) -> list[str]:
    """Resolve a feature set to *actual column names* present in ``available``.

    For each admitted evidence feature, picks the canonical name or a present
    alias (see :func:`evidence_registry.resolve_to_columns`). Evidence features
    with no present column are dropped here; the output writer re-adds them as
    NA columns so the schema stays stable.
    """
    avail = list(available)
    avail_set = set(avail)
    cols: list[str] = []
    seen: set[str] = set()
    for name in evidence_features(feature_set):
        resolved = er.resolve_to_columns(name, avail_set)
        if resolved and resolved not in seen:
            seen.add(resolved)
            cols.append(resolved)
    return cols


def subset_columns(feature_set: str, available: Iterable[str]) -> list[str]:
    """Full ordered column list for a subset CSV: support + evidence members.

    Every admitted evidence feature is represented even when the pipeline did
    not emit a value — those names are appended so the writer can fill NA,
    honouring "missing optional features appear as NA, not absent columns".
    """
    avail = list(available)
    avail_set = set(avail)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(col: str) -> None:
        if col not in seen:
            seen.add(col)
            ordered.append(col)

    for col in support_columns():
        if col in avail_set:
            add(col)
    # Evidence features in canonical registry order, then any present alias.
    for name in evidence_features(feature_set):
        resolved = er.resolve_to_columns(name, avail_set)
        add(resolved if resolved else name)
    return ordered


def membership_table() -> list[dict[str, object]]:
    """One row per evidence feature with boolean membership in each set."""
    rows: list[dict[str, object]] = []
    for e in er.all_evidence():
        row: dict[str, object] = {
            "feature_name": e.feature_name,
            "evidence_tier": e.evidence_tier.value,
            "implemented": e.implemented,
        }
        for fs in ALLOWED_FEATURE_SETS:
            row[fs] = e.evidence_tier in set(tiers_for(fs))
        rows.append(row)
    return rows
