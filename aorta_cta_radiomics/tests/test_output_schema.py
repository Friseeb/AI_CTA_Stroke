from __future__ import annotations

from dataclasses import fields
import json

from aorta_cta_radiomics.output_schema import (
    CaseLevelOutput,
    SamplePointLevelOutput,
    VesselLevelOutput,
    configuration_sha256,
    write_output_bundle,
)


def _case():
    return CaseLevelOutput(
        case_id="sub-test",
        cta_path="cta.nii.gz",
        cta_metadata={},
        original_spacing_xyz_mm=(0.5, 0.5, 1.0),
        original_dimensions_ijk=(20, 20, 20),
        orientation=("R", "A", "S"),
        affine=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        available_vessel_segments=["RCCA"],
        software_versions={"package": "test"},
        model_versions={"flowcat": "test"},
        flowcat_repository="https://github.com/FLOWCAT-CV/arterial",
        flowcat_branch="models/2026-03",
        flowcat_commit="abc",
        configuration_sha256=configuration_sha256({"a": 1}),
        processing_status="synthetic_test",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        processing_duration_seconds=1.0,
        cpu_information="test",
        gpu_information=None,
        peak_ram_bytes=None,
        peak_gpu_memory_bytes=None,
    )


def test_configuration_hash_is_order_independent():
    assert configuration_sha256({"a": 1, "b": [2, 3]}) == configuration_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_three_level_bundle_writes_explicit_formats(tmp_path):
    vessel = VesselLevelOutput(
        case_id="sub-test",
        vessel_name="RCCA",
        laterality="right",
        branch_id="1",
        vessel_label_confidence=1.0,
        total_length_mm=10.0,
        assessable_length_mm=9.0,
        nonassessable_length_mm=1.0,
        mean_radius_mm=3.0,
        minimum_radius_mm=2.5,
        maximum_radius_mm=3.5,
        radius_method="MISR",
        diameter_method="2*MISR",
        tortuosity=1.01,
        mean_curvature_per_mm=0.01,
        calcium_burden_mm3=0.0,
        total_shell_volume_mm3=100.0,
        valid_shell_volume_mm3=90.0,
        pvat_volume_mm3=20.0,
        pvat_fraction=2 / 9,
        mean_pvat_attenuation_hu=-80.0,
        perivascular_soft_tissue_volume_mm3=60.0,
        muscle_contact_fraction=0.25,
        vein_contact_fraction=0.25,
        bone_contact_fraction=0.0,
        thyroid_contact_fraction=0.0,
        other_soft_tissue_fraction=0.5,
        uncertain_fraction=0.0,
    )
    sample = SamplePointLevelOutput(
        case_id="sub-test",
        vessel_name="RCCA",
        laterality="right",
        branch_id="1",
        path_distance_mm=0.0,
        physical_ras_x_mm=1.0,
        physical_ras_y_mm=2.0,
        physical_ras_z_mm=3.0,
        tangent_ras=(0.0, 0.0, 1.0),
        first_normal_ras=(1.0, 0.0, 0.0),
        second_normal_ras=(0.0, 1.0, 0.0),
        local_radius_mm=3.0,
        radius_method="MISR",
        local_diameter_mm=6.0,
        diameter_method="2*MISR",
        curvature_per_mm=0.0,
        torsion_per_mm=0.0,
        distance_to_bifurcation_mm=5.0,
    )
    result = write_output_bundle(
        tmp_path, _case(), [vessel], [sample], prefer_parquet=False
    )
    assert result.case_json.is_file()
    assert result.vessel_csv.is_file()
    assert result.sample_point_table.name.endswith(".csv")
    manifest = json.loads(result.manifest_json.read_text())
    assert manifest["sample_point_format"] == "csv_requested"
    assert manifest["files"]["sample_point_level"]["rows"] == 1
    assert manifest["files"]["case_summary"]["sha256"]


def test_required_schema_fields_remain_present():
    case_names = {item.name for item in fields(CaseLevelOutput)}
    vessel_names = {item.name for item in fields(VesselLevelOutput)}
    sample_names = {item.name for item in fields(SamplePointLevelOutput)}
    assert {"flowcat_commit", "configuration_sha256", "missing_outputs"} <= case_names
    assert {"pvat_fraction", "uncertain_fraction", "manual_review_required"} <= vessel_names
    assert {"path_distance_mm", "sector_measurements", "qc_flags"} <= sample_names
