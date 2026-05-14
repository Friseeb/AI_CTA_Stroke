#!/usr/bin/env python3
"""Batch orchestrator for the LAA SLAAO thrombus-inclusive analysis pipeline.

Runs the following steps per case (all steps are individually skippable):

  1. Prior fusion        -- fuse_laa_priors.py
  2. Filling-defect map  -- generate_filling_defect_map.py
  3. SLAAO labels        -- generate_slaao_labels.py

Expects a BIDS-like derivatives layout. Outputs land in:
  <output-root>/<CASE_ID>/prior_fusion/
  <output-root>/<CASE_ID>/<CASE_ID>_filling_defect_map.nii.gz
  <output-root>/<CASE_ID>/<CASE_ID>_SLAAO_labels.json

A batch summary CSV is written to <output-root>/laa_slaao_batch_summary.csv.

Example:
  python scripts/run_laa_slaao_batch.py \\
    --nudf-root <BIDS_ROOT>/derivatives/nudf_la \\
    --totalseg-root <BIDS_ROOT>/derivatives/totalseg \\
    --ct-root <BIDS_ROOT>/derivatives/defaced \\
    --output-root <BIDS_ROOT>/derivatives/laa_slaao \\
    --nudf-suffix _laa_nudf.nii.gz \\
    --ct-suffix _defaced.nii.gz
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class CasePaths:
    case_id: str
    nudf_mask: Path | None = None
    vista3d_mask: Path | None = None
    totalseg_laa_mask: Path | None = None
    totalseg_dir: Path | None = None
    ct_volume: Path | None = None
    laa_corrected_mask: Path | None = None  # overrides nudf_mask for defect map if present


@dataclass
class CaseResult:
    case_id: str
    step_prior_fusion: str = "skipped"
    step_filling_defect: str = "skipped"
    step_slaao_labels: str = "skipped"
    slaao_dark_thrombus_component: str = ""
    slaao_contrast_stagnation: str = ""
    slaao_rim_pattern: str = ""
    slaao_whole_LAA_involvement: str = ""
    slaao_regional_pooling: str = ""
    slaao_distal_tip_involvement: str = ""
    slaao_mixed_pattern: str = ""
    slaao_uncertain_artifact: str = ""
    error: str = ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch LAA SLAAO pipeline: prior fusion → filling-defect → SLAAO labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nudf-root", default=None, help="Root dir containing <CASE_ID>/<CASE_ID><nudf-suffix>")
    p.add_argument("--nudf-suffix", default="_laa_nudf.nii.gz", help="NUDF mask filename suffix")
    p.add_argument("--vista3d-root", default=None, help="Root dir for VISTA3D masks (optional)")
    p.add_argument("--vista3d-suffix", default="_laa_vista3d.nii.gz", help="VISTA3D mask filename suffix")
    p.add_argument("--totalseg-root", default=None, help="Root dir for TotalSegmentator per-structure dirs")
    p.add_argument("--totalseg-laa-suffix", default="_laa_totalseg.nii.gz", help="TotalSegmentator LAA mask suffix")
    p.add_argument("--ct-root", default=None, help="Root dir containing defaced CT volumes")
    p.add_argument("--ct-suffix", default="_defaced.nii.gz", help="CT volume filename suffix")
    p.add_argument(
        "--corrected-mask-root",
        default=None,
        help=(
            "Root dir for expert-corrected LAA masks. If present, used for filling-defect step "
            "instead of NUDF mask. Suffix: <CASE_ID>_laa_corrected.nii.gz"
        ),
    )
    p.add_argument("--output-root", required=True, help="Root output directory")
    p.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Explicit list of case IDs to process. Auto-discovered from nudf-root if omitted.",
    )

    p.add_argument("--skip-prior-fusion", action="store_true", help="Skip step 1 (prior fusion)")
    p.add_argument("--skip-filling-defect", action="store_true", help="Skip step 2 (filling-defect map)")
    p.add_argument("--skip-slaao-labels", action="store_true", help="Skip step 3 (SLAAO labels)")

    # Forwarded to fuse_laa_priors.py
    p.add_argument("--majority-threshold", type=float, default=0.5)
    p.add_argument("--neg-distance-mm", type=float, default=5.0)

    # Forwarded to generate_filling_defect_map.py
    p.add_argument("--lumen-min", type=float, default=200.0)
    p.add_argument("--stagnation-min", type=float, default=50.0)
    p.add_argument("--smooth-sigma", type=float, default=0.5)
    p.add_argument("--mixed-band-hu", type=float, default=30.0)

    # Forwarded to generate_slaao_labels.py
    p.add_argument("--thrombus-frac", type=float, default=0.05)
    p.add_argument("--stagnation-frac", type=float, default=0.10)
    p.add_argument("--mixed-frac", type=float, default=0.05)
    p.add_argument("--whole-laa-frac", type=float, default=0.60)
    p.add_argument("--rim-shell-voxels", type=int, default=3)
    p.add_argument("--rim-frac", type=float, default=0.30)
    p.add_argument("--distal-frac", type=float, default=0.20)
    p.add_argument("--uncertain-disagree-frac", type=float, default=0.20)

    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")
    return p.parse_args()


def _check_env() -> None:
    import importlib
    details: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("nibabel", "numpy", "scipy"):
        try:
            m = importlib.import_module(pkg)
            details[pkg] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            details[f"{pkg}_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _discover_case_ids(nudf_root: Path, nudf_suffix: str) -> list[str]:
    case_ids = []
    for mask_path in sorted(nudf_root.glob(f"*/*{nudf_suffix}")):
        case_id = mask_path.name.replace(nudf_suffix, "")
        case_ids.append(case_id)
    return case_ids


def _resolve_case(case_id: str, args: argparse.Namespace) -> CasePaths:
    cp = CasePaths(case_id=case_id)

    if args.nudf_root:
        candidate = Path(args.nudf_root) / case_id / f"{case_id}{args.nudf_suffix}"
        if candidate.exists():
            cp.nudf_mask = candidate

    if args.vista3d_root:
        candidate = Path(args.vista3d_root) / case_id / f"{case_id}{args.vista3d_suffix}"
        if candidate.exists():
            cp.vista3d_mask = candidate

    if args.totalseg_root:
        ts_dir = Path(args.totalseg_root) / case_id
        if ts_dir.is_dir():
            cp.totalseg_dir = ts_dir
        ts_laa = ts_dir / f"{case_id}{args.totalseg_laa_suffix}"
        if ts_laa.exists():
            cp.totalseg_laa_mask = ts_laa

    if args.ct_root:
        candidate = Path(args.ct_root) / f"{case_id}{args.ct_suffix}"
        if candidate.exists():
            cp.ct_volume = candidate

    if args.corrected_mask_root:
        candidate = Path(args.corrected_mask_root) / case_id / f"{case_id}_laa_corrected.nii.gz"
        if candidate.exists():
            cp.laa_corrected_mask = candidate

    return cp


def _run(cmd: list[str], dry_run: bool) -> str:
    """Run a subprocess command; return 'ok', 'dry_run', or 'error: <msg>'."""
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return "dry_run"
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        short = (result.stderr or result.stdout or "").strip().splitlines()
        msg = short[-1] if short else "non-zero exit"
        return f"error: {msg}"
    return "ok"


def _step_prior_fusion(cp: CasePaths, out_dir: Path, args: argparse.Namespace) -> str:
    fusion_dir = out_dir / "prior_fusion"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "fuse_laa_priors.py"),
        "--output-dir", str(fusion_dir),
        "--case-id", cp.case_id,
        "--majority-threshold", str(args.majority_threshold),
        "--neg-distance-mm", str(args.neg_distance_mm),
    ]
    if cp.nudf_mask:
        cmd += ["--nudf", str(cp.nudf_mask)]
    if cp.vista3d_mask:
        cmd += ["--vista3d", str(cp.vista3d_mask)]
    if cp.totalseg_laa_mask:
        cmd += ["--totalseg-laa", str(cp.totalseg_laa_mask)]
    if cp.totalseg_dir:
        cmd += ["--totalseg-dir", str(cp.totalseg_dir)]

    if not cp.nudf_mask and not cp.vista3d_mask and not cp.totalseg_laa_mask:
        return "skip_no_priors"
    return _run(cmd, args.dry_run)


def _step_filling_defect(cp: CasePaths, out_dir: Path, args: argparse.Namespace) -> str:
    laa_mask = cp.laa_corrected_mask or cp.nudf_mask
    if laa_mask is None:
        return "skip_no_mask"
    if cp.ct_volume is None:
        return "skip_no_ct"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_filling_defect_map.py"),
        "--ct", str(cp.ct_volume),
        "--laa-mask", str(laa_mask),
        "--output-dir", str(out_dir),
        "--case-id", cp.case_id,
        "--lumen-min", str(args.lumen_min),
        "--stagnation-min", str(args.stagnation_min),
        "--smooth-sigma", str(args.smooth_sigma),
        "--mixed-band-hu", str(args.mixed_band_hu),
    ]
    return _run(cmd, args.dry_run)


def _step_slaao_labels(cp: CasePaths, out_dir: Path, args: argparse.Namespace) -> str:
    defect_map = out_dir / f"{cp.case_id}_filling_defect_map.nii.gz"
    laa_mask = cp.laa_corrected_mask or cp.nudf_mask
    if not defect_map.exists() and not args.dry_run:
        return "skip_no_defect_map"
    if laa_mask is None:
        return "skip_no_mask"

    disagree_map = out_dir / "prior_fusion" / f"{cp.case_id}_laa_prior_disagreement.nii.gz"

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_slaao_labels.py"),
        "--filling-defect-map", str(defect_map),
        "--laa-mask", str(laa_mask),
        "--output-dir", str(out_dir),
        "--case-id", cp.case_id,
        "--thrombus-frac", str(args.thrombus_frac),
        "--stagnation-frac", str(args.stagnation_frac),
        "--mixed-frac", str(args.mixed_frac),
        "--whole-laa-frac", str(args.whole_laa_frac),
        "--rim-shell-voxels", str(args.rim_shell_voxels),
        "--rim-frac", str(args.rim_frac),
        "--distal-frac", str(args.distal_frac),
        "--uncertain-disagree-frac", str(args.uncertain_disagree_frac),
    ]
    if disagree_map.exists():
        cmd += ["--disagreement-map", str(disagree_map)]

    return _run(cmd, args.dry_run)


def _read_slaao_json(out_dir: Path, case_id: str) -> dict[str, str]:
    path = out_dir / f"{case_id}_SLAAO_labels.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {k: str(v) for k, v in data.get("slaao_labels", {}).items()}
    except Exception:
        return {}


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.case_ids:
        case_ids = args.case_ids
    elif args.nudf_root:
        case_ids = _discover_case_ids(Path(args.nudf_root), args.nudf_suffix)
        print(f"Discovered {len(case_ids)} cases from {args.nudf_root}")
    else:
        raise RuntimeError("Provide --case-ids or --nudf-root to discover cases.")

    results: list[CaseResult] = []

    for case_id in case_ids:
        print(f"\n--- {case_id} ---")
        cp = _resolve_case(case_id, args)
        out_dir = output_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        result = CaseResult(case_id=case_id)
        try:
            if not args.skip_prior_fusion:
                status = _step_prior_fusion(cp, out_dir, args)
                result.step_prior_fusion = status
                print(f"  prior_fusion: {status}")

            if not args.skip_filling_defect:
                status = _step_filling_defect(cp, out_dir, args)
                result.step_filling_defect = status
                print(f"  filling_defect: {status}")

            if not args.skip_slaao_labels:
                status = _step_slaao_labels(cp, out_dir, args)
                result.step_slaao_labels = status
                print(f"  slaao_labels: {status}")

            slaao = _read_slaao_json(out_dir, case_id)
            for feat, val in slaao.items():
                key = f"slaao_{feat}"
                if hasattr(result, key):
                    setattr(result, key, val)

        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {result.error}")

        results.append(result)

    # --- Write batch summary CSV ---
    summary_path = output_root / "laa_slaao_batch_summary.csv"
    fieldnames = [f.name for f in fields(CaseResult)]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    ok = sum(1 for r in results if r.error == "")
    print(f"\nBatch complete: {ok}/{len(results)} cases without errors")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
