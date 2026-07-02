"""CLI entry point.

Subcommands:
  qc              run QC on one input and emit a single-row qc.csv
  extract         run full feature extraction on one input
  batch           run extract on every input in a directory or text file list
  summarize       summarise a features.csv (counts, missingness, QC-pass rate)
  compare-dental  join a CTA features.csv with a dental features.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import DISCLAIMER
from .clinical import merge_clinical
from .compare_dental import compare_with_dental
from .config import PipelineConfig, apply_overrides, load_config
from .features import extract_case
from .logging_utils import configure_logging, get_logger
from .output import append_processing_log, write_outputs
from .qc_slicer import open_in_slicer

app = typer.Typer(
    name="stroke-cta-osa",
    help=f"CTA-derived airway/adiposity features for OSA phenotyping in stroke cohorts.\n\n{DISCLAIMER}",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console(stderr=True)
log = get_logger("cli")


# --- shared options --------------------------------------------------------

_VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG logging.")]
_ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c",
                                                    help="YAML config override.")]
_OutOpt    = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]


def _setup(verbose: bool, config: Optional[Path]) -> PipelineConfig:
    configure_logging(verbose=verbose)
    cfg = load_config(config)
    console.print(f"[bold yellow]{DISCLAIMER}[/bold yellow]", highlight=False)
    return cfg


# --- qc ---------------------------------------------------------------------

@app.command()
def qc(
    input: Annotated[Path, typer.Argument(help="DICOM dir, NIfTI, or DICOM zip.")],
    out: _OutOpt,
    patient_id: Annotated[Optional[str], typer.Option("--patient-id")] = None,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Run QC only and emit a single-row qc.csv."""
    cfg = _setup(verbose, config)
    result = extract_case(input_path=input, out_dir=out, cfg=cfg,
                          patient_id=patient_id)
    write_outputs([result], out_dir=out)
    console.print(f"[green]QC complete.[/green] qc.csv → {out / 'qc.csv'}")
    if not result.qc["qc_pass"]:
        console.print(f"[red]QC failed:[/red] {result.qc['qc_failure_reasons']}")
        raise typer.Exit(1)


# --- extract ----------------------------------------------------------------

@app.command()
def extract(
    input: Annotated[Path, typer.Argument(help="DICOM dir, NIfTI, or DICOM zip.")],
    out: _OutOpt,
    patient_id: Annotated[Optional[str], typer.Option("--patient-id")] = None,
    dental_mask: Annotated[Optional[Path], typer.Option(
        "--dental-mask", help="Reuse dental airway mask NIfTI.")] = None,
    dental_landmarks: Annotated[Optional[Path], typer.Option(
        "--dental-landmarks", help="Reuse dental landmarks JSON.")] = None,
    dental_features: Annotated[Optional[Path], typer.Option(
        "--dental-features", help="Reuse dental airway features JSON.")] = None,
    dental_mandible_mask: Annotated[Optional[Path], typer.Option(
        "--dental-mandible-mask",
        help="Reuse dental mandible/lower-jawbone mask NIfTI.",
    )] = None,
    dental_artifacts_dir: Annotated[Optional[Path], typer.Option(
        "--dental-artifacts-dir",
        help="Dental pipeline output dir for this case; discovers airway, "
             "landmarks, features, and lower_jawbone/mandible masks.",
    )] = None,
    external_mask: Annotated[Optional[Path], typer.Option(
        "--external-mask", help="External airway mask NIfTI.")] = None,
    fallback: Annotated[Optional[str], typer.Option(
        "--fallback",
        help="Airway fallback: threshold_connected_component | external_mask_only | none.",
    )] = None,
    save_masks: Annotated[bool, typer.Option("--save-masks")] = False,
    no_qc_images: Annotated[bool, typer.Option("--no-qc-images")] = False,
    radiomics: Annotated[bool, typer.Option("--radiomics")] = False,
    # ---- v2 mask + landmark options ----
    external_airway_mask: Annotated[Optional[Path], typer.Option(
        "--external-airway-mask",
        help="External airway mask NIfTI (preferred over fallback / dental).",
    )] = None,
    external_tongue_mask: Annotated[Optional[Path], typer.Option(
        "--external-tongue-mask", help="External tongue mask NIfTI.")] = None,
    external_mandible_mask: Annotated[Optional[Path], typer.Option(
        "--external-mandible-mask", help="External mandible mask NIfTI.")] = None,
    external_soft_palate_mask: Annotated[Optional[Path], typer.Option(
        "--external-soft-palate-mask",
        help="External soft-palate mask NIfTI.")] = None,
    external_oral_cavity_mask: Annotated[Optional[Path], typer.Option(
        "--external-oral-cavity-mask",
        help="External oral-cavity mask NIfTI.")] = None,
    landmarks: Annotated[Optional[Path], typer.Option(
        "--landmarks", help="Explicit landmarks JSON.")] = None,
    save_qc_images: Annotated[bool, typer.Option(
        "--save-qc-images",
        help="Enable matplotlib QC overlay generation (overrides --no-qc-images).",
    )] = False,
    feature_set: Annotated[Optional[str], typer.Option(
        "--feature-set",
        help="Default modelling feature set recorded for this run and reported "
             "after extraction. All tiered subset CSVs are written regardless. "
             "One of: core_osa_backed | core_plus_anatomic_extensions | "
             "core_plus_cardiometabolic_ct | all_features_exploratory.",
    )] = None,
    open_slicer: Annotated[bool, typer.Option(
        "--open-slicer",
        help="After extraction, launch 3D Slicer with the QC scene "
             "(implies --save-masks).",
    )] = False,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Run full feature extraction for one CTA."""
    if open_slicer:
        save_masks = True  # the loader needs files on disk
    cfg = _setup(verbose, config)
    feature_set = _validate_feature_set(feature_set)
    if dental_artifacts_dir is not None:
        discovered = _discover_dental_artifacts(dental_artifacts_dir)
        dental_mask = dental_mask or discovered["airway"]
        dental_landmarks = dental_landmarks or discovered["landmarks"]
        dental_features = dental_features or discovered["features"]
        dental_mandible_mask = dental_mandible_mask or discovered["mandible"]
    overrides = {
        "airway.dental_airway_mask_path": str(dental_mask) if dental_mask else None,
        "airway.dental_landmarks_path": str(dental_landmarks) if dental_landmarks else None,
        "airway.dental_features_path": str(dental_features) if dental_features else None,
        "airway.use_existing_dental_airway_outputs":
            any([dental_mask, dental_landmarks, dental_features]) or None,
        "mandible.dental_mandible_mask_path":
            str(dental_mandible_mask) if dental_mandible_mask else None,
        "airway.external_mask_path": str(external_mask) if external_mask else None,
        "airway.fallback_method": fallback,
        "output.save_masks": save_masks or None,
        "output.save_qc_images": False if no_qc_images else None,
        "radiomics.enabled": True if radiomics else None,
        "feature_selection.default_modeling_feature_set": feature_set,
    }
    cfg = apply_overrides(cfg, {k: v for k, v in overrides.items() if v is not None})

    result = extract_case(
        input_path=input, out_dir=out, cfg=cfg,
        patient_id=patient_id,
        external_airway_mask_path=external_airway_mask,
        external_tongue_mask_path=external_tongue_mask,
        external_mandible_mask_path=external_mandible_mask,
        external_soft_palate_mask_path=external_soft_palate_mask,
        external_oral_cavity_mask_path=external_oral_cavity_mask,
        external_landmarks_path=landmarks,
    )
    paths = write_outputs([result], out_dir=out)
    append_processing_log(out / "case_processing_log.jsonl", result,
                          {"input_path": str(input)})

    console.rule("[bold green]Extraction complete[/bold green]")
    console.print(f"features.csv:  [cyan]{paths['features']}[/cyan]")
    console.print(f"qc.csv:        [cyan]{paths['qc']}[/cyan]")
    chosen = cfg.feature_selection.default_modeling_feature_set
    if chosen in paths:
        console.print(f"modelling set: [cyan]{paths[chosen]}[/cyan] "
                      f"[dim]({chosen})[/dim]")
    slicer_script = result.identifiers.get("slicer_loader_script") or ""
    if slicer_script:
        console.print(f"Slicer loader: [cyan]{slicer_script}[/cyan]")
        if open_slicer:
            open_in_slicer(Path(slicer_script))
    if not result.qc["qc_pass"]:
        console.print(f"[yellow]QC NOT PASSED:[/yellow] {result.qc['qc_failure_reasons']}")
    if result.errors:
        console.print(f"[red]Errors:[/red] {result.errors}")
        raise typer.Exit(1)


# --- batch ------------------------------------------------------------------

@app.command()
def batch(
    inputs: Annotated[Path, typer.Argument(
        help="Directory of cases OR a text file listing one input path per line.")],
    out: _OutOpt,
    glob: Annotated[str, typer.Option("--glob",
        help="Glob for case directories or NIfTI files (only when `inputs` is a dir).",
    )] = "*",
    dental_artifacts_dir: Annotated[Optional[Path], typer.Option(
        "--dental-artifacts-dir",
        help="Root with per-case dental outputs; discovers airway, landmarks, "
             "features, and lower_jawbone/mandible masks.",
    )] = None,
    airway_mask_dir: Annotated[Optional[Path], typer.Option(
        "--airway-mask-dir",
        help="Root with per-case real airway masks at <dir>/<case_id>/airway.nii.gz "
             "(e.g. from scripts/run_ts_airway_batch.py). Used as the external "
             "airway mask, overriding the HU fallback.",
    )] = None,
    save_masks: Annotated[bool, typer.Option("--save-masks")] = False,
    save_qc_images: Annotated[bool, typer.Option("--save-qc-images")] = False,
    feature_set: Annotated[Optional[str], typer.Option(
        "--feature-set",
        help="Default modelling feature set for this batch. All tiered subset "
             "CSVs are written regardless of this choice.",
    )] = None,
    workers: Annotated[str, typer.Option(
        "--workers", "-j",
        help="Parallel worker processes: 'auto' (default) picks a safe count "
             "from available RAM and the largest input's estimated peak, then "
             "caps by CPU and case count. '1' forces sequential. An integer N "
             "is still clamped by memory headroom.",
    )] = "auto",
    precompute_airway: Annotated[bool, typer.Option(
        "--precompute-airway",
        help="Two-pass mode: first compute & cache every airway mask "
             "(out/_airway_cache/), then run feature extraction reusing them. "
             "Avoids recomputing the airway segmentation on feature re-runs / "
             "config sweeps.",
    )] = False,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Run extract on multiple inputs and write a single aggregated features.csv."""
    cfg = _setup(verbose, config)
    feature_set = _validate_feature_set(feature_set)
    requested = _parse_workers(workers)
    overrides = {
        "output.save_masks": True if save_masks else None,
        "output.save_qc_images": True if save_qc_images else None,
        "feature_selection.default_modeling_feature_set": feature_set,
    }
    cfg = apply_overrides(cfg, {k: v for k, v in overrides.items() if v is not None})

    case_paths = _collect_case_paths(inputs, glob)
    if not case_paths:
        console.print(f"[red]No input cases found at {inputs}[/red]")
        raise typer.Exit(1)
    console.print(f"Found {len(case_paths)} input(s).")

    from . import parallel
    plan = parallel.auto_worker_count(case_paths, requested=requested)
    console.print(f"[dim]workers: {plan.workers} × {plan.threads_per_worker} "
                  f"thread(s)  ·  {plan.reason}[/dim]")

    # Optional pass 1: compute & cache airway masks, reused by the feature pass.
    airway_cache: dict[str, str] = {}
    if precompute_airway:
        airway_cache = _precompute_airway_pass(case_paths, out, cfg, plan, verbose)

    # Resolve real airway masks from --airway-mask-dir (<dir>/<case_id>/airway.nii.gz).
    def _airway_from_dir(p: Path) -> Optional[str]:
        if airway_mask_dir is None:
            return None
        import re
        m = re.match(r"(sub-[0-9A-Za-z]+)", p.name)
        cid = m.group(1) if m else p.stem
        cand = Path(airway_mask_dir) / cid / "airway.nii.gz"
        return str(cand) if cand.is_file() else None

    # Build per-case configs once (dental discovery happens in the parent so the
    # worker only needs a finished PipelineConfig).
    jobs = [
        _build_case_job(p, out, cfg, dental_artifacts_dir, verbose,
                        external_airway_mask=(airway_cache.get(p.name)
                                              or _airway_from_dir(p)))
        for p in case_paths
    ]

    results = _run_feature_pass(jobs, out, plan)

    paths = write_outputs(results, out_dir=out, long_format=True)
    console.rule(f"[bold green]Batch complete — {len(results)} cases[/bold green]")
    console.print(f"features.csv: [cyan]{paths['features']}[/cyan]")
    console.print(f"qc.csv:       [cyan]{paths['qc']}[/cyan]")
    # Report missingness by evidence tier so analysts see Tier-1 completeness.
    miss_path = paths.get("feature_missingness_by_tier")
    if miss_path is not None and Path(miss_path).is_file():
        import pandas as pd
        console.print("[bold]Missingness by evidence tier:[/bold]")
        for _, r in pd.read_csv(miss_path).iterrows():
            console.print(
                f"  {r['evidence_tier']:42s} "
                f"{int(r['n_available'])}/{int(r['n_features'])} available "
                f"({r['percent_missing']}% missing)"
            )


# --- summarize --------------------------------------------------------------

@app.command()
def summarize(
    features_csv: Annotated[Path, typer.Argument()],
    by_evidence_tier: Annotated[bool, typer.Option(
        "--by-evidence-tier",
        help="Report per-evidence-tier feature availability/missingness.",
    )] = False,
) -> None:
    """Summarise an existing features.csv."""
    import pandas as pd
    df = pd.read_csv(features_csv)
    console.print(f"Rows: [bold]{len(df)}[/bold]")
    console.print(f"Columns: [bold]{len(df.columns)}[/bold]")

    if by_evidence_tier:
        from . import evidence_registry as er
        available = set(df.columns)
        console.print("\n[bold]By evidence tier:[/bold]")
        for tier in er.EvidenceTier:
            specs = er.by_tier(tier)
            n_feat = len(specs)
            n_avail = sum(
                1 for s in specs
                if (c := er.resolve_to_columns(s.feature_name, available))
                and c in df.columns and df[c].notna().any()
            )
            pct = round(100.0 * (n_feat - n_avail) / n_feat, 1) if n_feat else 0.0
            console.print(
                f"  {tier.value:42s} {n_avail}/{n_feat} available ({pct}% missing)"
            )
        return
    if "qc_pass" in df.columns:
        passed = int(df["qc_pass"].astype(bool).sum())
        console.print(f"qc_pass: [green]{passed}[/green] / {len(df)}")
    if "airway_mask_available" in df.columns:
        avail = int(df["airway_mask_available"].astype(bool).sum())
        console.print(f"airway_mask_available: [green]{avail}[/green] / {len(df)}")
    if "airway_source" in df.columns:
        console.print("Airway sources:")
        for src, n in df["airway_source"].value_counts().items():
            console.print(f"  {src}: {n}")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(10)
    if not miss.empty:
        console.print("Top missing columns:")
        for col, n in miss.items():
            console.print(f"  {col}: {n}")


# --- compare-dental ---------------------------------------------------------

@app.command(name="compare-dental")
def compare_dental_cmd(
    cta_features: Annotated[Path, typer.Argument()],
    dental_features: Annotated[Path, typer.Argument()],
    out: _OutOpt,
    patient_id_column: Annotated[str, typer.Option("--patient-id-column")] = "patient_id",
    scan_id_column: Annotated[str, typer.Option("--scan-id-column")] = "scan_id",
) -> None:
    summary = compare_with_dental(
        cta_features_csv=cta_features,
        dental_features_csv=dental_features,
        out_dir=out,
        patient_id_column=patient_id_column,
        scan_id_column=scan_id_column,
    )
    console.print_json(json.dumps(summary, default=str))


# --- merge-clinical (bonus subcommand for completeness) ---------------------

@app.command(name="merge-clinical")
def merge_clinical_cmd(
    features_csv: Annotated[Path, typer.Argument()],
    clinical_csv: Annotated[Path, typer.Argument()],
    out: _OutOpt,
    patient_id_column: Annotated[str, typer.Option("--patient-id-column")] = "patient_id",
    scan_id_column: Annotated[str, typer.Option("--scan-id-column")] = "scan_id",
) -> None:
    summary = merge_clinical(
        features_csv=features_csv, clinical_csv=clinical_csv,
        out_path=out / "merged_features.csv",
        patient_id_column=patient_id_column, scan_id_column=scan_id_column,
    )
    console.print_json(json.dumps(summary, default=str))


# --- list-features ----------------------------------------------------------

@app.command(name="list-features")
def list_features_cmd(
    out: Annotated[Optional[Path], typer.Option(
        "--out", "-o",
        help="Output file. Format inferred from extension: .csv | .json. "
             "Defaults to printing the table to stdout.",
    )] = None,
    fmt: Annotated[str, typer.Option(
        "--format", help="Force output format: csv | json | table.",
    )] = "auto",
    feature_set: Annotated[Optional[str], typer.Option(
        "--feature-set",
        help="Filter to one evidence-gated feature set: core_osa_backed | "
             "core_plus_anatomic_extensions | core_plus_cardiometabolic_ct | "
             "all_features_exploratory.",
    )] = None,
    evidence_tier: Annotated[Optional[str], typer.Option(
        "--evidence-tier",
        help="Filter to one evidence tier, e.g. TIER_1_CORE_OSA_BACKED.",
    )] = None,
) -> None:
    """Export the feature dictionary with evidence-tier metadata.

    Without filters this exports every registry column annotated with its
    evidence tier/class/analysis-role. With ``--feature-set`` or
    ``--evidence-tier`` it exports only the evidence features in that group —
    handy for building a Tier-1-only feature dictionary for primary analysis.
    """
    from . import evidence_registry as er
    from . import feature_sets as fs

    if feature_set is not None and feature_set not in fs.ALLOWED_FEATURE_SETS:
        raise typer.BadParameter(
            f"unknown feature set {feature_set!r}; choose one of "
            f"{', '.join(fs.ALLOWED_FEATURE_SETS)}"
        )
    tier_enum = None
    if evidence_tier is not None:
        try:
            tier_enum = er.EvidenceTier(evidence_tier)
        except ValueError:
            raise typer.BadParameter(
                f"unknown evidence tier {evidence_tier!r}; choose one of "
                f"{', '.join(t.value for t in er.EvidenceTier)}"
            )

    if feature_set is not None or tier_enum is not None:
        names: Optional[set[str]] = None
        if feature_set is not None:
            names = set(fs.evidence_features(feature_set))
        if tier_enum is not None:
            tier_names = {e.feature_name for e in er.by_tier(tier_enum)}
            names = tier_names if names is None else (names & tier_names)
        recs = [r for r in er.to_records() if r["feature_name"] in names]
    else:
        recs = er.merged_records()

    if out is not None:
        ext = out.suffix.lower().lstrip(".")
        fmt = ext if fmt == "auto" else fmt
        if fmt == "csv":
            import csv as _csv
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(recs[0].keys()) if recs else [])
                w.writeheader()
                w.writerows(recs)
        elif fmt == "json":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(recs, indent=2, default=str))
        else:
            raise typer.BadParameter(
                f"unknown format {fmt!r}; use csv or json (or omit --out for table)"
            )
        console.print(f"[green]Wrote {fmt} ({len(recs)} rows) → {out}[/green]")
    else:
        for r in recs:
            ev = r.get("evidence_tier", "") or "-"
            console.print(
                f"  [bold]{r['feature_name']:55s}[/bold] "
                f"[{r.get('family', r.get('feature_family','')):14s}] "
                f"[{r['unit']:8s}] [{ev}]"
            )
        console.print(f"\n[dim]{len(recs)} features. Evidence tiers: "
                      f"{', '.join(t.value for t in er.EvidenceTier)}[/dim]")


# --- validate-landmarks -----------------------------------------------------

@app.command(name="validate-landmarks")
def validate_landmarks_cmd(
    image: Annotated[Path, typer.Argument(help="CTA NIfTI for shape/affine check.")],
    landmarks_path: Annotated[Path, typer.Argument(
        help="Landmarks JSON to validate.")],
) -> None:
    """Validate that a landmarks JSON is well-formed against an image."""
    from .io import load_input
    from .landmarks import load_landmarks, validate_landmarks
    cta, _ = load_input(image)
    bundle = load_landmarks(landmarks_path)
    warnings = validate_landmarks(bundle, image=cta)
    if not warnings:
        console.print(f"[green]OK — {landmarks_path.name} validates against "
                      f"image {image.name}[/green]")
        return
    console.print(f"[yellow]{len(warnings)} warning(s):[/yellow]")
    for w in warnings:
        console.print(f"  - {w}")


# --- helpers ----------------------------------------------------------------

def _validate_feature_set(feature_set: Optional[str]) -> Optional[str]:
    """Validate a --feature-set value against the canonical list."""
    if feature_set is None:
        return None
    from .feature_sets import ALLOWED_FEATURE_SETS
    if feature_set not in ALLOWED_FEATURE_SETS:
        raise typer.BadParameter(
            f"unknown feature set {feature_set!r}; choose one of "
            f"{', '.join(ALLOWED_FEATURE_SETS)}"
        )
    return feature_set


def _parse_workers(workers: str) -> Optional[int]:
    """Parse the --workers value: 'auto' -> None, else a positive int."""
    if workers is None or str(workers).strip().lower() == "auto":
        return None
    try:
        n = int(workers)
    except ValueError:
        raise typer.BadParameter(
            f"--workers must be 'auto' or a positive integer, got {workers!r}")
    if n < 1:
        raise typer.BadParameter("--workers must be >= 1")
    return n


def _build_case_job(p: Path, out: Path, cfg, dental_artifacts_dir: Optional[Path],
                    verbose: bool, external_airway_mask: Optional[str] = None):
    """Resolve per-case dental overrides and package a picklable CaseJob."""
    from .parallel import CaseJob
    pid = p.name
    per_case_cfg = cfg
    if dental_artifacts_dir is not None:
        d = _discover_dental_artifacts(dental_artifacts_dir, case_id=pid)
        per_case_cfg = apply_overrides(cfg, {
            "airway.use_existing_dental_airway_outputs": True,
            "airway.dental_airway_mask_path": str(d["airway"]) if d["airway"] else None,
            "airway.dental_landmarks_path": str(d["landmarks"]) if d["landmarks"] else None,
            "airway.dental_features_path": str(d["features"]) if d["features"] else None,
            "mandible.dental_mandible_mask_path":
                str(d["mandible"]) if d["mandible"] else None,
        })
    return CaseJob(input_path=str(p), out_dir=str(out), pid=pid,
                   cfg=per_case_cfg, verbose=verbose,
                   external_airway_mask=external_airway_mask)


def _execute(worker_fn, jobs, plan):
    """Run jobs sequentially or in a memory-bounded process pool, yielding
    outcomes as they complete.

    ``worker_fn`` must be a module-level (picklable) callable. The pool uses the
    ``spawn`` start method to avoid fork + threaded-ITK deadlocks, and per-worker
    BLAS/ITK thread caps are exported before it starts.
    """
    if plan.workers <= 1:
        for job in jobs:
            yield worker_fn(job)
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from . import parallel
    parallel.apply_thread_limits(plan.threads_per_worker)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=ctx) as ex:
        futures = [ex.submit(worker_fn, job) for job in jobs]
        for fut in as_completed(futures):
            yield fut.result()


def _run_feature_pass(jobs, out: Path, plan) -> list:
    """Run the feature-extraction pass, collecting CaseResults + processing log."""
    from . import parallel
    results = []
    n = len(jobs)
    done = 0
    for outcome in _execute(parallel.run_case, jobs, plan):
        done += 1
        if outcome.error is not None:
            console.print(f"[{done}/{n}] {outcome.pid}  [red]crashed:[/red] {outcome.error}")
            continue
        console.print(f"[{done}/{n}] {outcome.pid}  [green]ok[/green]")
        results.append(outcome.result)
        append_processing_log(out / "case_processing_log.jsonl", outcome.result,
                              {"input_path": outcome.input_path})
    return results


def _precompute_airway_pass(case_paths, out: Path, cfg, plan, verbose: bool
                            ) -> dict[str, str]:
    """Pass 1: compute and cache airway masks; return {pid: mask_path}."""
    from . import parallel
    cache_dir = out / "_airway_cache"
    jobs = [
        parallel.AirwayJob(
            input_path=str(p), cache_path=str(cache_dir / f"{p.name}.airway.nii.gz"),
            pid=p.name, cfg=cfg, verbose=verbose,
        )
        for p in case_paths
    ]
    console.rule(f"[bold]Pass 1/2 — airway precompute → {cache_dir}[/bold]")
    cache: dict[str, str] = {}
    n = len(jobs)
    done = 0
    for outcome in _execute(parallel.run_airway_precompute, jobs, plan):
        done += 1
        if outcome.error is not None:
            console.print(f"[{done}/{n}] {outcome.pid}  [red]airway failed:[/red] "
                          f"{outcome.error}")
            continue
        if outcome.mask_path:
            cache[outcome.pid] = outcome.mask_path
        console.print(f"[{done}/{n}] {outcome.pid}  [green]airway:[/green] "
                      f"{outcome.source}")
    console.rule("[bold]Pass 2/2 — feature extraction[/bold]")
    return cache


def _collect_case_paths(inputs: Path, glob: str) -> list[Path]:
    if inputs.is_file():
        # text file with one path per line
        if inputs.suffix.lower() in (".txt", ".lst"):
            lines = [ln.strip() for ln in inputs.read_text().splitlines() if ln.strip()]
            return [Path(x) for x in lines if Path(x).exists()]
        # single NIfTI / zip
        if inputs.suffix.lower() in (".gz", ".nii", ".zip") or inputs.name.endswith(".nii.gz"):
            return [inputs]
    if inputs.is_dir():
        matches = sorted(inputs.glob(glob))
        # accept either case directories (containing DICOM) or NIfTI files
        return [m for m in matches if m.is_dir() or m.suffix.lower() in (".gz", ".nii", ".zip")
                or m.name.endswith(".nii.gz")]
    return []


def _discover_dental_artifacts(
    root: Path,
    *,
    case_id: Optional[str] = None,
) -> dict[str, Optional[Path]]:
    """Find reusable outputs from the sibling dental pipeline."""
    roots = [Path(root)]
    if case_id:
        roots = [Path(root) / case_id, Path(root) / Path(case_id).stem, Path(root)]

    def first_existing(candidates: list[str]) -> Optional[Path]:
        for r in roots:
            for rel in candidates:
                p = r / rel
                if p.is_file():
                    return p
        return None

    return {
        "airway": first_existing(["airway.nii.gz", "mask_airway.nii.gz"]),
        "landmarks": first_existing(["landmarks.json", "airway_landmarks.json"]),
        "features": first_existing(["airway_features.json"]),
        "mandible": first_existing([
            "mandible.nii.gz",
            "lower_jawbone.nii.gz",
            "roi/_tseg_teeth/lower_jawbone.nii.gz",
            "segmentations/totalsegmentator_teeth/lower_jawbone.nii.gz",
            "roi/_tseg_teeth/mandible.nii.gz",
            "segmentations/totalsegmentator_teeth/mandible.nii.gz",
        ]),
    }


if __name__ == "__main__":
    app()
