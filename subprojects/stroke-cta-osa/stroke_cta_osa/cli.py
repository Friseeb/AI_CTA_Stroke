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
    external_mask: Annotated[Optional[Path], typer.Option(
        "--external-mask", help="External airway mask NIfTI.")] = None,
    fallback: Annotated[Optional[str], typer.Option(
        "--fallback",
        help="Airway fallback: threshold_connected_component | external_mask_only | none.",
    )] = None,
    save_masks: Annotated[bool, typer.Option("--save-masks")] = False,
    no_qc_images: Annotated[bool, typer.Option("--no-qc-images")] = False,
    radiomics: Annotated[bool, typer.Option("--radiomics")] = False,
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
    overrides = {
        "airway.dental_airway_mask_path": str(dental_mask) if dental_mask else None,
        "airway.dental_landmarks_path": str(dental_landmarks) if dental_landmarks else None,
        "airway.dental_features_path": str(dental_features) if dental_features else None,
        "airway.use_existing_dental_airway_outputs":
            any([dental_mask, dental_landmarks, dental_features]) or None,
        "airway.external_mask_path": str(external_mask) if external_mask else None,
        "airway.fallback_method": fallback,
        "output.save_masks": save_masks or None,
        "output.save_qc_images": False if no_qc_images else None,
        "radiomics.enabled": True if radiomics else None,
    }
    cfg = apply_overrides(cfg, {k: v for k, v in overrides.items() if v is not None})

    result = extract_case(input_path=input, out_dir=out, cfg=cfg,
                          patient_id=patient_id)
    paths = write_outputs([result], out_dir=out)
    append_processing_log(out / "case_processing_log.jsonl", result,
                          {"input_path": str(input)})

    console.rule("[bold green]Extraction complete[/bold green]")
    console.print(f"features.csv:  [cyan]{paths['features']}[/cyan]")
    console.print(f"qc.csv:        [cyan]{paths['qc']}[/cyan]")
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
        help="Root with per-case <case_id>/{airway.nii.gz,landmarks.json,airway_features.json}",
    )] = None,
    save_masks: Annotated[bool, typer.Option("--save-masks")] = False,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Run extract on multiple inputs and write a single aggregated features.csv."""
    cfg = _setup(verbose, config)
    if save_masks:
        cfg = apply_overrides(cfg, {"output.save_masks": True})

    case_paths = _collect_case_paths(inputs, glob)
    if not case_paths:
        console.print(f"[red]No input cases found at {inputs}[/red]")
        raise typer.Exit(1)
    console.print(f"Found {len(case_paths)} input(s).")

    results = []
    for i, p in enumerate(case_paths, 1):
        pid = p.name
        # Auto-locate dental artefacts when given
        per_case_cfg = cfg
        if dental_artifacts_dir is not None:
            d = Path(dental_artifacts_dir) / pid
            airway = d / "airway.nii.gz"
            lm = d / "landmarks.json"
            feats = d / "airway_features.json"
            per_case_cfg = apply_overrides(cfg, {
                "airway.use_existing_dental_airway_outputs": True,
                "airway.dental_airway_mask_path": str(airway) if airway.is_file() else None,
                "airway.dental_landmarks_path": str(lm) if lm.is_file() else None,
                "airway.dental_features_path": str(feats) if feats.is_file() else None,
            })

        console.print(f"[{i}/{len(case_paths)}] {pid}")
        try:
            r = extract_case(input_path=p, out_dir=out, cfg=per_case_cfg,
                             patient_id=pid)
        except Exception as exc:
            log.exception("Case %s crashed:", pid)
            console.print(f"  [red]crashed:[/red] {exc}")
            continue
        results.append(r)
        append_processing_log(out / "case_processing_log.jsonl", r,
                              {"input_path": str(p)})

    paths = write_outputs(results, out_dir=out, long_format=True)
    console.rule(f"[bold green]Batch complete — {len(results)} cases[/bold green]")
    console.print(f"features.csv: [cyan]{paths['features']}[/cyan]")
    console.print(f"qc.csv:       [cyan]{paths['qc']}[/cyan]")


# --- summarize --------------------------------------------------------------

@app.command()
def summarize(
    features_csv: Annotated[Path, typer.Argument()],
) -> None:
    """Summarise an existing features.csv."""
    import pandas as pd
    df = pd.read_csv(features_csv)
    console.print(f"Rows: [bold]{len(df)}[/bold]")
    console.print(f"Columns: [bold]{len(df.columns)}[/bold]")
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


# --- helpers ----------------------------------------------------------------

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


if __name__ == "__main__":
    app()
