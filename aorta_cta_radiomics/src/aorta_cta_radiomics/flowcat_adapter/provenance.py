"""FLOWCAT repository, model-file, dependency, and command provenance."""

from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .discovery import sha256_file
from .schemas import ProcessingProvenance


DEFAULT_FLOWCAT_REPOSITORY_URL = "https://github.com/FLOWCAT-CV/arterial"


def collect_processing_provenance(
    repository_path: str | Path,
    *,
    model_files: Mapping[str, str | Path] | None = None,
    dependencies: Sequence[str] = ("numpy", "nibabel", "networkx"),
    command_line_parameters: Mapping[str, Any] | None = None,
    processing_status: str = "not_started",
    started_at: str | None = None,
    ended_at: str | None = None,
) -> ProcessingProvenance:
    """Collect reproducible local provenance without importing FLOWCAT itself."""

    repo = Path(repository_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"FLOWCAT repository directory not found: {repo}")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git(repo, "rev-parse", "HEAD")
    remote = _git(repo, "remote", "get-url", "origin", required=False) or DEFAULT_FLOWCAT_REPOSITORY_URL

    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, raw_path in (model_files or {}).items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FLOWCAT model file {name!r} not found: {path}")
        paths[str(name)] = str(path)
        hashes[str(name)] = sha256_file(path)

    return ProcessingProvenance(
        repository_url=remote,
        branch=branch,
        commit=commit,
        model_files=paths,
        model_file_sha256=hashes,
        dependency_versions=collect_dependency_versions(dependencies),
        command_line_parameters=dict(command_line_parameters or {}),
        processing_status=processing_status,
        started_at=started_at,
        ended_at=ended_at,
    )


def collect_dependency_versions(packages: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[str(package)] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[str(package)] = "not-installed"
    return versions


def _git(repo: Path, *arguments: str, required: bool = True) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if required:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"Could not collect FLOWCAT Git provenance: {detail}")
    return None
