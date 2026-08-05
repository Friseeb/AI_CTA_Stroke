from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_carotid_pipeline.py"
SPEC = importlib.util.spec_from_file_location("audit_carotid_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


MODE = "extracranial_vessels"


def _write(path: Path, payload: bytes = b"native-output") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _complete_native_case(root: Path) -> dict[str, Path]:
    return {
        "segmentation": _write(root / MODE / "segmentation.nii.gz", b"segmentation"),
        "branch_model": _write(root / MODE / "branch_model.vtk", b"branch-model"),
        "centerline_segments_array": _write(
            root / MODE / "centerline_segments_array.npy", b"centerline-array"
        ),
        "segments_graph_pred": _write(
            root / MODE / "segments_graph_pred.pickle", b"predicted-graph"
        ),
        "local_graph": _write(root / MODE / "local_graph.pickle", b"local-graph"),
    }


def _audit(root: Path, **kwargs):
    return audit_module.audit_case(
        root,
        case_id="anon-case",
        required_imports=kwargs.pop("required_imports", ()),
        optional_imports=kwargs.pop("optional_imports", ()),
        **kwargs,
    )


def test_missing_native_outputs_fail_closed_with_exit_two(tmp_path):
    _write(tmp_path / MODE / "segmentation.nii.gz")

    report = _audit(tmp_path)

    assert report["audit_status"] == "fail"
    assert report["exit_code"] == 2
    assert report["flowcat"]["missing_required_native_outputs"] == [
        "branch_model",
        "centerline_segments_array",
        "local_graph",
        "segments_graph_pred",
    ]
    assert "required_native_artifacts_missing" in report["blocking_reasons"]
    assert report["flowcat"]["unsafe_artifacts_opened"] is False


def test_complete_native_outputs_pass_and_every_file_has_sha256(tmp_path):
    files = _complete_native_case(tmp_path)

    report = _audit(tmp_path)

    assert report["audit_status"] == "pass"
    assert report["exit_code"] == 0
    assert report["flowcat"]["missing_required_native_outputs"] == []
    assert report["flowcat"]["sha256_manifest"] == {
        "algorithm": "sha256",
        "status": "complete",
        "present_artifact_count": 5,
        "hashed_artifact_count": 5,
        "all_present_artifacts_hashed": True,
        "persistence": "not_written_read_only",
    }
    by_name = {item["name"]: item for item in report["flowcat"]["native_outputs"]}
    for name, path in files.items():
        assert by_name[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert by_name[name]["relative_path"] == str(path.relative_to(tmp_path))
        assert not Path(by_name[name]["relative_path"]).is_absolute()


def test_zero_byte_required_artifact_is_invalid_and_fails_closed(tmp_path):
    _complete_native_case(tmp_path)
    (tmp_path / MODE / "branch_model.vtk").write_bytes(b"")

    report = _audit(tmp_path)

    assert report["exit_code"] == 2
    assert report["flowcat"]["invalid_required_native_outputs"] == ["branch_model"]
    assert "required_native_artifacts_invalid" in report["blocking_reasons"]


def test_audit_does_not_write_or_open_unsafe_artifacts(tmp_path):
    _complete_native_case(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    report = _audit(tmp_path)

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert report["flowcat"]["sha256_manifest"]["persistence"] == "not_written_read_only"
    assert report["flowcat"]["unsafe_artifacts_opened"] is False


def test_report_does_not_inventory_unrelated_phi_named_files(tmp_path):
    _complete_native_case(tmp_path)
    _write(tmp_path / "MRN-123456_patient-name.txt", b"must-not-appear")

    report = _audit(tmp_path)
    serialized = json.dumps(report)

    assert report["case"]["case_id"] == "anon-case"
    assert report["case"]["case_directory"] == str(tmp_path.resolve())
    assert "MRN-123456" not in serialized
    assert "patient-name.txt" not in serialized


def test_required_import_failure_blocks_but_optional_failure_only_inventories(tmp_path):
    _complete_native_case(tmp_path)

    required_failure = _audit(
        tmp_path,
        required_imports=(("module_that_does_not_exist_xyz", "module_that_does_not_exist_xyz"),),
    )
    optional_failure = _audit(
        tmp_path,
        optional_imports=(("module_that_does_not_exist_xyz", "module_that_does_not_exist_xyz"),),
    )

    assert required_failure["exit_code"] == 2
    assert required_failure["imports"]["missing_required"] == ["module_that_does_not_exist_xyz"]
    assert required_failure["imports"]["required"]["module_that_does_not_exist_xyz"]["error_type"]
    assert optional_failure["exit_code"] == 0
    assert optional_failure["imports"]["optional"]["module_that_does_not_exist_xyz"]["available"] is False


def test_runtime_inventory_excludes_hostname_and_python_executable(tmp_path):
    _complete_native_case(tmp_path)

    runtime = _audit(tmp_path)["runtime"]

    assert runtime["python"]["version"]
    assert runtime["platform"]["system"]
    assert runtime["hardware"]["logical_cpu_count"]
    assert "hostname" not in runtime["platform"]
    assert "node" not in runtime["platform"]
    assert "executable" not in runtime["python"]


def test_missing_case_directory_returns_json_and_exit_two(tmp_path, capsys):
    missing = tmp_path / "missing-case"

    exit_code = audit_module.main(
        ["--case-dir", str(missing), "--case-id", "anon-case"]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 2
    assert report["exit_code"] == 2
    assert report["audit_status"] == "fail"
    assert report["flowcat"]["adapter_error_type"] == "CaseDirectoryUnavailable"
    assert report["flowcat"]["missing_required_native_outputs"] == sorted(
        audit_module.DEFAULT_REQUIRED_OUTPUTS
    )
