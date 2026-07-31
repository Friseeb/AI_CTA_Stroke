#!/usr/bin/env python
"""Read-only preflight audit for a FLOWCAT-backed carotid pipeline.

The command prints one JSON object to stdout and never writes into the case
directory. Pickle and object-NPY artifacts are inventoried and hashed but are
not opened. A non-runnable audit exits with status 2.

Example:
  python scripts/audit_carotid_pipeline.py \
      --case-dir /path/to/deidentified-flowcat-case \
      --case-id sub-001 --pretty
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import platform
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence


AORTA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AORTA_ROOT / "src"))

from aorta_cta_radiomics.flowcat_adapter import (  # noqa: E402
    DEFAULT_REQUIRED_OUTPUTS,
    build_output_manifest,
)


REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("nibabel", "nibabel"),
    ("networkx", "networkx"),
    ("aorta_cta_radiomics", "aorta-cta-radiomics"),
)

OPTIONAL_IMPORTS: tuple[tuple[str, str], ...] = (
    ("SimpleITK", "SimpleITK"),
    ("vtk", "vtk"),
    ("pyvista", "pyvista"),
    ("torch", "torch"),
    ("arterial", "arterial"),
    ("psutil", "psutil"),
)


def audit_case(
    case_directory: str | os.PathLike[str],
    *,
    case_id: str,
    mode: str = "extracranial_vessels",
    required_outputs: Sequence[str] = DEFAULT_REQUIRED_OUTPUTS,
    required_imports: Sequence[tuple[str, str]] = REQUIRED_IMPORTS,
    optional_imports: Sequence[tuple[str, str]] = OPTIONAL_IMPORTS,
) -> dict[str, Any]:
    """Return a PHI-minimal, JSON-compatible audit without changing files."""

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("An explicit non-empty case_id is required")
    root = Path(case_directory).expanduser().resolve(strict=False)
    required = tuple(dict.fromkeys(str(name) for name in required_outputs))
    import_inventory = {
        "required": _probe_imports(required_imports),
        "optional": _probe_imports(optional_imports),
    }
    missing_required_imports = sorted(
        name for name, result in import_inventory["required"].items() if not result["available"]
    )

    native_outputs: list[dict[str, Any]] = []
    missing_required_outputs: list[str] = list(required)
    invalid_required_outputs: list[str] = []
    adapter_error: str | None = None
    sha_status: dict[str, Any] = {
        "algorithm": "sha256",
        "status": "unavailable",
        "present_artifact_count": 0,
        "hashed_artifact_count": 0,
        "all_present_artifacts_hashed": False,
        "persistence": "not_written_read_only",
    }

    if root.is_dir():
        try:
            manifest = build_output_manifest(
                root,
                case_id=case_id,
                mode=mode,
                required=required,
                hash_files=True,
            )
            missing_required_outputs = list(manifest.missing_required_outputs)
            for artifact in sorted(manifest.artifacts, key=lambda item: item.name):
                # CTA is an input rather than a FLOWCAT-native output.
                if artifact.name == "cta":
                    continue
                relative_path = None
                if artifact.path is not None:
                    relative_path = str(Path(artifact.path).resolve().relative_to(root))
                native_outputs.append(
                    {
                        "name": artifact.name,
                        "relative_path": relative_path,
                        "format": artifact.format,
                        "present": artifact.present,
                        "required": artifact.required,
                        "size_bytes": artifact.size_bytes,
                        "sha256": artifact.sha256,
                        "validation_status": artifact.validation_status,
                        "coordinate_reference": artifact.coordinate_reference.value,
                        "units": artifact.units,
                    }
                )
                if artifact.required and artifact.validation_status == "invalid":
                    invalid_required_outputs.append(artifact.name)

            present = [artifact for artifact in native_outputs if artifact["present"]]
            hashed = [artifact for artifact in present if _is_sha256(artifact["sha256"])]
            hashes_complete = (
                bool(present)
                and len(hashed) == len(present)
                and not missing_required_outputs
                and not invalid_required_outputs
            )
            sha_status = {
                "algorithm": "sha256",
                "status": "complete" if hashes_complete else "incomplete",
                "present_artifact_count": len(present),
                "hashed_artifact_count": len(hashed),
                "all_present_artifacts_hashed": hashes_complete,
                "persistence": "not_written_read_only",
            }
        except Exception as exc:  # fail closed; do not emit path-bearing exception text
            adapter_error = type(exc).__name__
    else:
        adapter_error = "CaseDirectoryUnavailable"

    blocking_reasons: list[str] = []
    if adapter_error is not None:
        blocking_reasons.append("flowcat_inventory_error")
    if missing_required_outputs:
        blocking_reasons.append("required_native_artifacts_missing")
    if invalid_required_outputs:
        blocking_reasons.append("required_native_artifacts_invalid")
    if missing_required_imports:
        blocking_reasons.append("required_imports_unavailable")
    if sha_status["status"] != "complete" and root.is_dir():
        blocking_reasons.append("sha256_manifest_incomplete")

    audit_status = "pass" if not blocking_reasons else "fail"
    return {
        "schema_version": "1.0.0",
        "audit_status": audit_status,
        "case": {
            "case_id": case_id,
            "case_directory": str(root),
            "mode": mode,
        },
        "runtime": {
            "python": _python_inventory(),
            "platform": _platform_inventory(),
            "hardware": _hardware_inventory(),
        },
        "imports": {
            **import_inventory,
            "missing_required": missing_required_imports,
            "required_imports_ok": not missing_required_imports,
        },
        "flowcat": {
            "required_native_outputs": list(required),
            "missing_required_native_outputs": sorted(missing_required_outputs),
            "invalid_required_native_outputs": sorted(invalid_required_outputs),
            "native_outputs": native_outputs,
            "sha256_manifest": sha_status,
            "unsafe_artifacts_opened": False,
            "adapter_error_type": adapter_error,
        },
        "blocking_reasons": sorted(set(blocking_reasons)),
        "exit_code": 0 if audit_status == "pass" else 2,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="FLOWCAT case directory to audit")
    parser.add_argument("--case-id", required=True, help="Explicit de-identified case ID for JSON output")
    parser.add_argument(
        "--mode",
        choices=("extracranial_vessels", "intracranial_vessels"),
        default="extracranial_vessels",
    )
    parser.add_argument("--pretty", action="store_true", help="Indent the JSON output")
    args = parser.parse_args(argv)

    try:
        report = audit_case(args.case_dir, case_id=args.case_id, mode=args.mode)
    except Exception as exc:
        # Argument values are already explicit. Keep unexpected details out of
        # the report so paths from dependencies or image headers cannot leak.
        report = {
            "schema_version": "1.0.0",
            "audit_status": "fail",
            "case": {
                "case_id": args.case_id,
                "case_directory": str(Path(args.case_dir).expanduser().resolve(strict=False)),
                "mode": args.mode,
            },
            "error_type": type(exc).__name__,
            "blocking_reasons": ["audit_error"],
            "exit_code": 2,
        }
    json.dump(
        report,
        sys.stdout,
        indent=2 if args.pretty else None,
        sort_keys=True,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return int(report["exit_code"])


def _probe_imports(packages: Sequence[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for module_name, distribution_name in packages:
        try:
            # Some optional scientific packages warn or print while importing.
            # Keep stdout a single valid JSON document and do not echo local
            # installation paths from those messages.
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                warnings.simplefilter("ignore")
                module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = metadata.version(distribution_name)
                except metadata.PackageNotFoundError:
                    version = None
            results[module_name] = {
                "available": True,
                "version": str(version) if version is not None else None,
                "error_type": None,
            }
        except Exception as exc:
            results[module_name] = {
                "available": False,
                "version": None,
                "error_type": type(exc).__name__,
            }
    return results


def _python_inventory() -> dict[str, Any]:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "bitness": 64 if sys.maxsize > 2**32 else 32,
    }


def _platform_inventory() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


def _hardware_inventory() -> dict[str, Any]:
    return {
        "logical_cpu_count": os.cpu_count(),
        "total_ram_bytes": _total_ram_bytes(),
        "gpu_probe": "not_performed",
    }


def _total_ram_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = page_size * page_count
    return total if total > 0 else None


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
