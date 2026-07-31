from __future__ import annotations

from argparse import Namespace
import json

import numpy as np

from aorta_cta_radiomics import carotid_cli
from aorta_cta_radiomics.flowcat_adapter import ProcessingProvenance


def _incomplete_case(tmp_path):
    case = tmp_path / "case"
    segmentation = case / "extracranial_vessels" / "segmentation.nii.gz"
    segmentation.parent.mkdir(parents=True)
    segmentation.write_bytes(b"nonempty-inventory-fixture")
    return case


def test_manifest_writes_inventory_and_returns_two_when_native_outputs_are_missing(
    tmp_path, monkeypatch
):
    case = _incomplete_case(tmp_path)
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        carotid_cli,
        "_provenance",
        lambda *args, **kwargs: ProcessingProvenance(
            branch="models/2026-03", commit="a" * 40
        ),
    )
    args = Namespace(
        case_dir=str(case),
        case_id="anon-case",
        flowcat_repo=str(tmp_path),
        output=str(output),
        mode="extracranial_vessels",
    )

    exit_code = carotid_cli.command_manifest(args)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert manifest["case"]["case_id"] == "anon-case"
    assert manifest["missing_required_outputs"] == [
        "branch_model",
        "centerline_segments_array",
        "local_graph",
        "segments_graph_pred",
    ]


def test_run_refuses_unsafe_artifacts_before_opening_them(tmp_path):
    outdir = tmp_path / "output"
    args = Namespace(outdir=str(outdir), trusted_flowcat_artifacts=False)

    exit_code = carotid_cli.command_run(args)
    failure = json.loads((outdir / "failure.json").read_text(encoding="utf-8"))

    assert exit_code == 2
    assert failure["stage"] == "preflight"
    assert failure["error_type"] == "ValueError"
    assert "--trusted-flowcat-artifacts" in failure["error_message"]


def test_yaml_config_uses_spacing_based_resolution_floor():
    payload = {
        "flowcat": {"target_labels": ["RCCA"]},
        "coordinate_system": {"sample_spacing_mm": 0.8, "bifurcation_exclusion_mm": 2.5},
        "label_propagation": {"minimum_confidence": 0.1},
        "radial_shells": {
            "bands_mm": [{"name": "near", "inner_mm": 0.0, "outer_mm": 2.0}],
            "partial_volume_exclusion_mm": 0.2,
        },
        "adaptive_envelope": {
            "fat": {"alpha": 1.0, "minimum_mm": "resolution_floor", "maximum_mm": 5.0}
        },
        "tissue_composition": {
            "fat_windows_hu": {
                "primary": [-190, -30],
                "narrow_sensitivity": [-150, -50],
                "wide_sensitivity": [-200, -20],
            },
            "high_density_hu": 300,
        },
        "sectors": {"primary_count": 16},
        "surface_contact": {"distances_mm": [0.5, 1.0, 2.0], "primary_distance_mm": 1.0},
    }

    config = carotid_cli._analysis_config(payload, (0.5, 0.5, 1.2))

    assert config.required_flowcat_codes == ("RCCA",)
    assert np.isclose(config.adaptive_law.minimum_mm, 2.4)
    assert config.sector_count == 16
    assert config.tissue_thresholds.high_density_min_hu == 300


def test_model_payload_discovery_is_limited_to_model_pth_files(tmp_path):
    included = tmp_path / "arterial" / "segmentation" / "models" / "task" / "weights.pth"
    included.parent.mkdir(parents=True)
    included.write_bytes(b"weights")
    ignored = tmp_path / "arterial" / "segmentation" / "models" / "task" / "plans.json"
    ignored.write_text("{}", encoding="utf-8")
    outside = tmp_path / "unrelated" / "weights.pth"
    outside.parent.mkdir()
    outside.write_bytes(b"other")

    payloads = carotid_cli._model_payloads(tmp_path)

    assert payloads == {
        "arterial/segmentation/models/task/weights.pth": included.resolve()
    }
