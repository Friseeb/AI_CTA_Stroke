from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aorta_cta_radiomics.carotid_analysis import (
    CarotidAnalysisConfig,
    CarotidAnalysisError,
    CoarseAnatomyMasksIJK,
    SupportedBulbInterval,
    analyze_carotid_case,
    flowcat_branch_to_centerline_seed,
    nibabel_ijk_to_perivascular_zyx,
    perivascular_geometry_from_nifti,
    perivascular_zyx_to_nibabel_ijk,
)
from aorta_cta_radiomics.flowcat_adapter import (
    ArteryIdentity,
    BinaryNifti,
    CaseMetadata,
    CenterlineBranch,
    CenterlinePoint,
    CoordinateReference,
    FlowcatOutputManifest,
    ImageGeometry,
    ProcessingProvenance,
    ValidatedFlowcatCase,
)
from aorta_cta_radiomics.perivascular._spatial import indices_zyx_to_physical_xyz
from aorta_cta_radiomics.perivascular.composition import TissueLabel


def _geometry(shape=(31, 31, 41), affine=None):
    transform = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    spacing = tuple(float(value) for value in np.linalg.norm(transform[:3, :3], axis=0))
    return ImageGeometry(
        shape_ijk=shape,
        spacing_mm=spacing,
        affine_ras_mm=tuple(tuple(float(value) for value in row) for row in transform),
        orientation=("R", "A", "S"),
    )


def _branch(
    geometry: ImageGeometry,
    *,
    code="RCCA",
    class_id=3,
    branch_id="cell-0",
    artery=True,
    reference=CoordinateReference.NIFTI_RAS_PHYSICAL_MM,
):
    affine = np.asarray(geometry.affine_ras_mm)
    ijk = np.column_stack(
        (np.full(31, 15.0), np.full(31, 15.0), np.arange(5.0, 36.0))
    )
    ras = (np.column_stack((ijk, np.ones(len(ijk)))) @ affine.T)[:, :3]
    path = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(ras, axis=0), axis=1))))
    identity = ArteryIdentity(class_id, code, "common carotid artery", "right", 0.95)
    points = tuple(
        CenterlinePoint(
            coordinates_mm=tuple(float(value) for value in ras[index]),
            path_distance_mm=float(path[index]),
            local_radius_mm=3.0,
            branch_id=branch_id,
            coordinate_reference=reference,
            artery=identity if artery else None,
            vessel_label_confidence=0.95 if artery else None,
        )
        for index in range(len(ras))
    )
    return CenterlineBranch(
        branch_id=branch_id,
        cell_id=int(branch_id.split("-")[-1]),
        points=points,
        coordinate_reference=reference,
        artery=identity if artery else None,
    )


def _validated_case(*, branches=None):
    geometry = _geometry()
    ii, jj, kk = np.indices(geometry.shape_ijk)
    lumen = ((ii - 15) ** 2 + (jj - 15) ** 2 <= 3**2) & (kk >= 5) & (kk <= 35)
    manifest = FlowcatOutputManifest(
        schema_version="1.0.0",
        case=CaseMetadata(
            case_id="synthetic-rcca",
            case_directory="/synthetic",
            cta_path="/synthetic/cta.nii.gz",
        ),
        artifacts=(),
        provenance=ProcessingProvenance(
            branch="models/2026-03",
            commit="synthetic-commit",
            processing_status="complete",
        ),
    )
    supplied = tuple(branches) if branches is not None else (_branch(geometry),)
    return ValidatedFlowcatCase(
        manifest=manifest,
        cta_geometry=geometry,
        segmentation=BinaryNifti(lumen, geometry, "/synthetic/segmentation.nii.gz"),
        branches=supplied,
        segments_graph_pred=object(),
    )


def _cta_and_anatomy(case):
    shape = case.cta_geometry.shape_ijk
    cta = np.full(shape, -100.0, dtype=np.float32)
    cta[case.segmentation.data] = 300.0
    muscle = np.zeros(shape, dtype=bool)
    vein = np.zeros(shape, dtype=bool)
    muscle[19:23, 8:23, 5:36] = True
    vein[8:23, 19:23, 5:36] = True
    return cta, CoarseAnatomyMasksIJK(vein=vein, skeletal_muscle=muscle)


def test_ijk_zyx_contract_round_trip_and_physical_affine_mapping():
    volume = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    zyx = nibabel_ijk_to_perivascular_zyx(volume)
    assert zyx.shape == (4, 3, 2)
    assert zyx[3, 2, 1] == volume[1, 2, 3]
    np.testing.assert_array_equal(perivascular_zyx_to_nibabel_ijk(zyx), volume)

    affine = np.asarray(
        [[0.0, -2.0, 0.0, 10.0], [1.0, 0.0, 0.0, 20.0], [0.0, 0.0, 3.0, -5.0], [0, 0, 0, 1]]
    )
    geometry = _geometry((2, 3, 4), affine)
    mapped = perivascular_geometry_from_nifti(geometry)
    physical = indices_zyx_to_physical_xyz(
        np.asarray([[3.0, 2.0, 1.0]]),
        mapped.spacing_xyz_mm,
        mapped.origin_ras_xyz_mm,
        mapped.direction_ijk_to_ras,
    )
    expected = (affine @ np.asarray([1.0, 2.0, 3.0, 1.0]))[:3]
    np.testing.assert_allclose(physical[0], expected)


def test_sheared_affine_fails_before_physical_distance_analysis():
    affine = np.eye(4)
    affine[0, 1] = 0.2
    with pytest.raises(CarotidAnalysisError, match="shear"):
        perivascular_geometry_from_nifti(_geometry((3, 3, 3), affine))


def test_synthetic_flowcat_carotid_runs_all_first_milestone_components():
    case = _validated_case()
    cta, anatomy = _cta_and_anatomy(case)
    result = analyze_carotid_case(
        case,
        cta,
        config=CarotidAnalysisConfig(required_flowcat_codes=("RCCA",)),
        coarse_anatomy_ijk=anatomy,
    )

    assert result.case_id == "synthetic-rcca"
    assert set(np.unique(result.anatomical_artery_volume_ijk.labels)) <= {0, 3}
    assert set(result.branches) == {"RCCA"}
    branch = result.branches["RCCA"]
    assert branch.target_lumen_zyx.shape == tuple(reversed(case.cta_geometry.shape_ijk))
    assert branch.derived_volume_label == 3
    assert len(branch.fixed_shells.bands) == 5
    assert "adaptive_shell" in branch.adaptive_shell.bands
    assert len(branch.radial_profiles) == 5
    assert branch.longitudinal_profiles
    assert len(branch.sector_summaries) == 5 * 8
    assert branch.sector_assignment.labels.max() <= 8
    assert branch.composition.mask(TissueLabel.ADIPOSE).any()
    assert branch.composition.mask(TissueLabel.SKELETAL_MUSCLE).any()
    assert branch.composition.mask(TissueLabel.VEIN).any()
    assert branch.contacts.at(0.5).classes["skeletal_muscle"].contact_fraction > 0
    assert branch.contacts.at(0.5).classes["vein"].contact_fraction > 0

    assert len(result.vessel_outputs) == 1
    assert len(result.sample_outputs) == len(branch.coordinate_system.path_mm)
    vessel = result.vessel_outputs[0]
    assert vessel.vessel_name == "RCCA"
    assert vessel.pvat_volume_mm3 > 0
    assert vessel.valid_shell_volume_mm3 > 0
    assert vessel.manual_review_required is True
    assert "manual_visual_review_pending" in vessel.qc_flags
    assert "coarse_anatomy_bone_unavailable" in vessel.qc_flags
    assert "coarse_anatomy_airway_unavailable" in vessel.qc_flags
    assert "coarse_anatomy_thyroid_gland_unavailable" in vessel.qc_flags
    assert vessel.bone_contact_fraction is None
    assert vessel.thyroid_contact_fraction is None
    assert vessel.vein_contact_fraction is not None
    assert vessel.muscle_contact_fraction is not None
    assert vessel.longitudinal_summaries["configured_intervals"] == []
    availability = vessel.longitudinal_summaries["coarse_anatomy_availability"]
    assert availability["vein"] is True
    assert availability["skeletal_muscle"] is True
    assert availability["bone"] is False
    assert all(
        sample.sector_measurements["longitudinal_intervals"] == []
        for sample in result.sample_outputs
    )
    # The production implementation compacts finite vessel-coordinate voxels
    # once.  Verify its sample summaries against the direct full-grid formula.
    sample_index = len(result.sample_outputs) // 2
    sample = result.sample_outputs[sample_index]
    path_map = branch.sector_assignment.path_mm
    slab = np.isfinite(path_map) & (
        np.abs(path_map - sample.path_distance_mm) <= 0.5
    )
    for name, mask in branch.fixed_shells.filtered_masks.items():
        selected = slab & mask
        valid = selected & branch.composition.valid_denominator_mask
        fat = valid & branch.composition.mask(TissueLabel.ADIPOSE)
        record = sample.radial_band_measurements[name]
        assert record["voxels"] == int(selected.sum())
        assert record["valid_voxels"] == int(valid.sum())
        expected_fraction = float(fat.sum() / valid.sum()) if valid.any() else None
        assert record["fat_fraction"] == expected_fraction
    for sector in range(1, branch.sector_assignment.number_of_sectors + 1):
        selected = (
            slab
            & (branch.sector_assignment.labels == sector)
            & branch.composition.valid_denominator_mask
        )
        fat = selected & branch.composition.mask(TissueLabel.ADIPOSE)
        record = sample.sector_measurements["sectors"][str(sector)]
        assert record["valid_voxels"] == int(selected.sum())
        expected_fraction = float(fat.sum() / selected.sum()) if selected.any() else None
        assert record["fat_fraction"] == expected_fraction
    assert result.case_output.cta_metadata["array_order_input"] == "nibabel_ijk"
    assert result.case_output.cta_metadata["array_order_analysis"] == "perivascular_kji_zyx"
    assert result.case_output.processing_status == "partial_missing_optional_context"
    assert "manual_visual_review_pending" in result.case_output.global_qc_flags
    assert "coarse_anatomy_bone_unavailable" in result.case_output.global_qc_flags
    assert result.visual_review_required is True


def test_bulb_is_only_an_evidence_backed_configured_interval_never_a_volume_label():
    case = _validated_case()
    cta, _ = _cta_and_anatomy(case)
    interval = SupportedBulbInterval(
        flowcat_code="RCCA",
        start_path_mm=10.0,
        end_path_mm=15.0,
        support_source="manual_landmark_review",
        support_reference="reviewer-1:landmarks-v1",
    )
    result = analyze_carotid_case(
        case,
        cta,
        config=CarotidAnalysisConfig(
            required_flowcat_codes=("RCCA",),
            bulb_intervals=(interval,),
        ),
    )
    configured = result.vessel_outputs[0].longitudinal_summaries["configured_intervals"]
    assert configured == [interval.to_record()]
    assert configured[0]["is_inferred"] is False
    inside = [
        sample
        for sample in result.sample_outputs
        if interval.start_path_mm <= sample.path_distance_mm <= interval.end_path_mm
    ]
    assert inside
    assert all(
        sample.sector_measurements["longitudinal_intervals"]
        == ["configured_carotid_bulb_interval"]
        for sample in inside
    )
    assert set(np.unique(result.anatomical_artery_volume_ijk.labels)) <= {0, 3}


def test_missing_cta_label_or_physical_coordinates_fail_closed():
    case = _validated_case()
    config = CarotidAnalysisConfig(required_flowcat_codes=("RCCA",))
    with pytest.raises(CarotidAnalysisError, match="CTA voxel data"):
        analyze_carotid_case(case, None, config=config)
    cta, _ = _cta_and_anatomy(case)
    with pytest.raises(CarotidAnalysisError, match="LCCA is unavailable"):
        analyze_carotid_case(
            case,
            cta,
            config=CarotidAnalysisConfig(required_flowcat_codes=("LCCA",)),
        )

    unlabeled = _branch(case.cta_geometry, artery=False)
    with pytest.raises(CarotidAnalysisError, match="no anatomical FLOWCAT label"):
        flowcat_branch_to_centerline_seed(unlabeled)
    relative = _branch(
        case.cta_geometry,
        reference=CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM,
    )
    with pytest.raises(CarotidAnalysisError, match="not in NIfTI RAS"):
        flowcat_branch_to_centerline_seed(relative)


def test_class_zero_other_and_class_thirteen_ba_share_derived_label_but_keep_source_codes():
    case = _validated_case()
    other_identity = ArteryIdentity(0, "other", "other artery", None, 0.8)
    other = replace(
        case.branches[0],
        artery=other_identity,
        points=tuple(replace(point, artery=other_identity) for point in case.branches[0].points),
    )
    ba_identity = ArteryIdentity(13, "BA", "basilar artery", None, 0.9)
    ba = replace(
        case.branches[0],
        branch_id="cell-13",
        cell_id=13,
        artery=ba_identity,
        points=tuple(
            replace(point, branch_id="cell-13", artery=ba_identity)
            for point in case.branches[0].points
        ),
    )
    other_seed = flowcat_branch_to_centerline_seed(other)
    ba_seed = flowcat_branch_to_centerline_seed(ba)
    assert other_seed.label_id == 13
    assert ba_seed.label_id == 13
    assert other_seed.label_name == "other"
    assert ba_seed.label_name == "BA"


def test_unsubstantiated_bulb_fails_closed():
    with pytest.raises(CarotidAnalysisError, match="support_source"):
        SupportedBulbInterval("RCCA", 1.0, 2.0, "", "")


def test_supplied_empty_coarse_masks_mean_available_not_anatomy_absent_by_assumption():
    case = _validated_case()
    cta, partial = _cta_and_anatomy(case)
    empty = np.zeros(case.cta_geometry.shape_ijk, dtype=bool)
    complete_context = CoarseAnatomyMasksIJK(
        vein=partial.vein,
        bone=empty,
        airway=empty,
        thyroid_gland=empty,
        skeletal_muscle=partial.skeletal_muscle,
    )
    result = analyze_carotid_case(
        case,
        cta,
        config=CarotidAnalysisConfig(required_flowcat_codes=("RCCA",)),
        coarse_anatomy_ijk=complete_context,
    )
    vessel = result.vessel_outputs[0]
    assert result.case_output.processing_status == "automated_pass_visual_pending"
    assert all(result.coarse_anatomy_availability[name] for name in (
        "vein",
        "bone",
        "airway",
        "thyroid_gland",
        "skeletal_muscle",
    ))
    assert not any("coarse_anatomy_" in flag and "unavailable" in flag for flag in vessel.qc_flags)
    assert vessel.bone_contact_fraction == 0.0
    assert vessel.thyroid_contact_fraction == 0.0
    assert vessel.manual_review_required is True
    assert "manual_visual_review_pending" in vessel.qc_flags


def test_coarse_anatomy_must_use_exact_cta_ijk_shape():
    case = _validated_case()
    cta, _ = _cta_and_anatomy(case)
    bad = CoarseAnatomyMasksIJK(vein=np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(CarotidAnalysisError, match="IJK shape"):
        analyze_carotid_case(
            case,
            cta,
            config=CarotidAnalysisConfig(required_flowcat_codes=("RCCA",)),
            coarse_anatomy_ijk=bad,
        )
