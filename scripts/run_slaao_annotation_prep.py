#!/usr/bin/env python3
"""Prepare per-case annotation packages for 3D Slicer / MONAILabel.

For each case:
  1. Copies CT volume + consensus/prior masks into a structured annotation folder
  2. Writes a blank SLAAO_labels.json template
  3. Writes a session.json manifest
  4. Optionally generates a MONAILabel-compatible dataset.json

The resulting folder layout is:
  <store-root>/<case_id>/annotation/
    <case_id>_ct.nii.gz
    consensus_laa.nii.gz
    positive_prior.nii.gz
    negative_prior.nii.gz
    SLAAO_labels.json          <- blank template, fill in 3D Slicer
    session.json               <- session metadata

Example (single case from existing prior fusion output):
  python scripts/run_slaao_annotation_prep.py \\
    --case-id sub-001_acq-CTA_ct \\
    --ct-path derivatives/defaced/sub-001_defaced.nii.gz \\
    --prior-fusion-dir derivatives/prior_fusion/sub-001 \\
    --store-root derivatives/annotation_store

Example (batch - reads case list from CSV):
  python scripts/run_slaao_annotation_prep.py \\
    --batch-csv derivatives/prior_fusion/cases.csv \\
    --ct-root derivatives/defaced \\
    --prior-fusion-root derivatives/prior_fusion \\
    --store-root derivatives/annotation_store \\
    --monailabel-dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.laa_slaao.annotation_store import AnnotationStore
from python.laa_slaao.peri_laa_fat import (
    DEFAULT_SHELLS_MM, run_peri_laa_fat_from_paths,
)
from python.laa_slaao.slaao_schema import SLAAOLabels


def _parse_shells(spec: str) -> list[tuple[float, float]]:
    """Parse ``"0-2,2-5,5-10"`` → ``[(0,2),(2,5),(5,10)]``."""
    out: list[tuple[float, float]] = []
    for band in spec.split(","):
        band = band.strip()
        if not band:
            continue
        lo, hi = band.split("-")
        out.append((float(lo), float(hi)))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare LAA/SLAAO annotation packages for 3D Slicer / MONAILabel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--case-id", default=None, help="Single case ID")
    g.add_argument("--batch-csv", default=None, help="CSV with 'case_id' column for batch mode")

    # Single-case paths
    p.add_argument("--ct-path", default=None, help="CT NIfTI path (single-case mode)")
    p.add_argument("--prior-fusion-dir", default=None, help="Prior fusion output dir (single-case)")

    # Batch-mode path patterns
    p.add_argument("--ct-root", default=None, help="Root dir for CTs; expects <root>/<case_id>_defaced.nii.gz")
    p.add_argument("--prior-fusion-root", default=None, help="Root dir for prior fusion; expects <root>/<case_id>/")

    # Output
    p.add_argument("--store-root", required=True, help="Annotation store root directory")

    # MONAILabel
    p.add_argument(
        "--monailabel-dataset",
        action="store_true",
        help="Write a MONAILabel-compatible dataset.json in <store-root>",
    )

    # Peri-LAA fat
    p.add_argument(
        "--peri-laa-fat", action="store_true",
        help="Also compute peri-LAA fat shells from the consensus LAA mask "
             "and stage them in the annotation directory.",
    )
    p.add_argument(
        "--peri-laa-shells-mm", default="0-2,2-5,5-10",
        help="Comma-separated radial bands in mm, e.g. '0-2,2-5,5-10'.",
    )
    p.add_argument(
        "--peri-laa-fat-hu-min", type=float, default=-190.0,
        help="Lower fat HU threshold for adipose voxel inclusion.",
    )
    p.add_argument(
        "--peri-laa-fat-hu-max", type=float, default=-30.0,
        help="Upper fat HU threshold for adipose voxel inclusion.",
    )
    p.add_argument(
        "--peri-laa-fat-laa-source", default="consensus_laa",
        choices=["consensus_laa", "intersection_laa", "union_laa",
                 "nudf_laa", "vista3d_laa", "totalseg_laa"],
        help="Which LAA mask (in --prior-fusion-dir) drives the shell.",
    )
    p.add_argument(
        "--peri-laa-fat-no-exclusion", action="store_true",
        help="Do not subtract the negative prior from the fat shells "
             "(use only if the negative prior is unreliable).",
    )

    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _prior_path(fusion_dir: Path, case_id: str, suffix: str) -> Path:
    return fusion_dir / f"{case_id}_{suffix}.nii.gz"


def _resolve_ct(ct_root: Path, case_id: str) -> Path:
    for pattern in [
        f"{case_id}_defaced.nii.gz",
        f"{case_id}.nii.gz",
        f"{case_id}_ct.nii.gz",
    ]:
        p = ct_root / pattern
        if p.exists():
            return p
    raise FileNotFoundError(f"CT not found for {case_id} under {ct_root}")


def prep_case(
    store: AnnotationStore,
    case_id: str,
    ct_path: Path,
    fusion_dir: Path,
    quiet: bool = False,
    peri_laa_fat_opts: Optional[dict] = None,
) -> Path:
    """Prepare one case. If `peri_laa_fat_opts` is given, also computes the
    peri-LAA fat shells and stages them in the annotation dir."""
    def log(msg: str):
        if not quiet:
            print(msg, flush=True)

    consensus = _prior_path(fusion_dir, case_id, "consensus_laa")
    positive  = _prior_path(fusion_dir, case_id, "positive_prior")
    negative  = _prior_path(fusion_dir, case_id, "negative_prior")

    fat_labels_path: Optional[Path] = None
    fat_metrics_path: Optional[Path] = None
    if peri_laa_fat_opts:
        laa_source = peri_laa_fat_opts["laa_source"]
        laa_path = _prior_path(fusion_dir, case_id, laa_source)
        if not laa_path.exists():
            log(f"  [warn] {case_id}: no {laa_source} mask at {laa_path} — "
                f"skipping peri-LAA fat.")
        else:
            neg_path = negative if (
                negative.exists() and not peri_laa_fat_opts["no_exclusion"]
            ) else None
            log(f"  [peri-LAA fat] {case_id}: LAA={laa_source}"
                f"{' + neg-prior exclusion' if neg_path else ''}, "
                f"shells={peri_laa_fat_opts['shells_mm']}")
            try:
                _, paths = run_peri_laa_fat_from_paths(
                    ct_path=ct_path,
                    laa_mask_path=laa_path,
                    out_dir=fusion_dir,  # write next to fusion outputs
                    case_id=case_id,
                    fat_hu_min=peri_laa_fat_opts["hu_min"],
                    fat_hu_max=peri_laa_fat_opts["hu_max"],
                    shells_mm=peri_laa_fat_opts["shells_mm"],
                    negative_prior_path=neg_path,
                    write_per_shell_masks=False,
                )
                fat_labels_path = paths["labels"]
                fat_metrics_path = paths["metrics"]
            except Exception as exc:
                log(f"  [warn] {case_id}: peri-LAA fat failed: {exc}")

    ann_dir = store.init_annotation_package(
        case_id=case_id,
        ct_path=ct_path,
        consensus_laa_path=consensus if consensus.exists() else None,
        positive_prior_path=positive if positive.exists() else None,
        negative_prior_path=negative if negative.exists() else None,
        peri_laa_fat_labels_path=fat_labels_path,
        peri_laa_fat_metrics_path=fat_metrics_path,
    )
    log(f"  [ok] {case_id} -> {ann_dir}")
    return ann_dir


def write_monailabel_dataset(store: AnnotationStore) -> Path:
    """Write a minimal MONAILabel dataset.json for all cases in the store."""
    dataset = {"name": "LAA_SLAAO", "description": "Thrombus-inclusive LAA annotation dataset", "cases": []}
    for case_id in store.list_cases():
        ann_dir = store.annotation_dir(case_id)
        ct = ann_dir / f"{case_id}_ct.nii.gz"
        label = ann_dir / "corrected_LAA_mask.nii.gz"
        if not ct.exists():
            ct = ann_dir / "consensus_laa.nii.gz"
        dataset["cases"].append({
            "id": case_id,
            "image": str(ct.relative_to(store.root)) if ct.exists() else "",
            "label": str(label.relative_to(store.root)) if label.exists() else "",
        })
    out = store.root / "dataset.json"
    out.write_text(json.dumps(dataset, indent=2))
    return out


def main() -> None:
    args = _parse_args()
    store = AnnotationStore(Path(args.store_root))

    def log(msg: str):
        if not args.quiet:
            print(msg, flush=True)

    fat_opts: Optional[dict] = None
    if args.peri_laa_fat:
        fat_opts = {
            "shells_mm": _parse_shells(args.peri_laa_shells_mm),
            "hu_min": args.peri_laa_fat_hu_min,
            "hu_max": args.peri_laa_fat_hu_max,
            "laa_source": args.peri_laa_fat_laa_source,
            "no_exclusion": args.peri_laa_fat_no_exclusion,
        }

    if args.case_id:
        # Single-case mode
        ct_path = Path(args.ct_path) if args.ct_path else None
        fusion_dir = Path(args.prior_fusion_dir) if args.prior_fusion_dir else None

        if ct_path is None:
            raise SystemExit("--ct-path is required in single-case mode")
        if fusion_dir is None:
            raise SystemExit("--prior-fusion-dir is required in single-case mode")

        prep_case(store, args.case_id, ct_path, fusion_dir,
                  quiet=args.quiet, peri_laa_fat_opts=fat_opts)

    else:
        # Batch mode
        ct_root = Path(args.ct_root) if args.ct_root else None
        fusion_root = Path(args.prior_fusion_root) if args.prior_fusion_root else None

        if ct_root is None or fusion_root is None:
            raise SystemExit("--ct-root and --prior-fusion-root are required in batch mode")

        df = pd.read_csv(args.batch_csv)
        if "case_id" not in df.columns:
            raise SystemExit("CSV must have a 'case_id' column")

        log(f"[annotation_prep] batch: {len(df)} cases")
        ok, failed = 0, []
        for case_id in df["case_id"]:
            try:
                ct_path = _resolve_ct(ct_root, case_id)
                fusion_dir = fusion_root / case_id
                prep_case(store, case_id, ct_path, fusion_dir,
                          quiet=args.quiet, peri_laa_fat_opts=fat_opts)
                ok += 1
            except Exception as exc:
                log(f"  [fail] {case_id}: {exc}")
                failed.append(case_id)

        log(f"[annotation_prep] done: {ok} ok, {len(failed)} failed")
        if failed:
            log(f"  failed: {failed}")

    if args.monailabel_dataset:
        out = write_monailabel_dataset(store)
        log(f"[annotation_prep] MONAILabel dataset.json -> {out}")


if __name__ == "__main__":
    main()
