import json
from pathlib import Path

import pandas as pd

from aorta_cta_radiomics.qc_slicer import (
    discover_mask_records,
    load_feature_tables,
    select_cases,
    select_outlier_case_ids,
    write_review_outputs,
    write_selection_table,
    write_slicer_scripts,
)


def test_select_cases_by_clinical_filter_and_feature_outlier(tmp_path: Path):
    cases = pd.DataFrame(
        {
            "case_id": ["CASE1", "CASE2", "CASE3"],
            "image_path": ["case1.nii.gz", "case2.nii.gz", "case3.nii.gz"],
            "SLAO": [1, 1, 0],
        }
    )
    feature_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "case_id": ["CASE1", "CASE2", "CASE3"],
            "region": ["aorta", "aorta", "aorta"],
            "feature_name": ["volume", "volume", "volume"],
            "feature_value": [10.0, 50.0, 100.0],
        }
    ).to_csv(feature_path, index=False)

    selected = select_cases(
        cases,
        filters=["SLAO=1"],
        feature_tables=[feature_path],
        outlier_features=["aorta:volume"],
        outlier_method="top-n",
        outlier_direction="high",
        outlier_top_n=1,
    )

    assert selected["case_id"].tolist() == ["CASE2"]


def test_outlier_selection_from_wide_feature_table(tmp_path: Path):
    feature_path = tmp_path / "wide.csv"
    pd.DataFrame({"case_id": ["A", "B"], "mask_volume_mm3": [100.0, 900.0]}).to_csv(feature_path, index=False)

    features = load_feature_tables([feature_path])
    selected = select_outlier_case_ids(
        features,
        selectors=["mask_volume_mm3"],
        method="top-n",
        direction="high",
        top_n=1,
    )

    assert selected == {"B"}


def test_discover_masks_and_write_slicer_review_outputs(tmp_path: Path):
    project = tmp_path
    outputs = project / "outputs" / "run"
    mask_dir = outputs / "masks" / "CASE1"
    mask_dir.mkdir(parents=True)
    image_path = project / "CASE1_cta.nii.gz"
    image_path.touch()
    calcium_path = mask_dir / "CASE1_calcification_aorta_wall_dynamic_seed500HU_candidate.nii.gz"
    artery_path = mask_dir / "CASE1_aorta_mask_cleaned.nii.gz"
    calcium_path.touch()
    artery_path.touch()
    cases = pd.DataFrame({"case_id": ["CASE1"], "image_path": [str(image_path)]})

    records = discover_mask_records(
        cases,
        anatomies=["aorta"],
        tasks=["calcification"],
        outputs_root=outputs,
        project_root=project,
        manifest_base=project,
    )

    assert len(records) == 2
    assert records[0].label == "Aorta"
    assert records[0].category == "artery"
    assert records[1].label == "Bone"
    assert records[1].category == "bone"

    selection_path = write_selection_table(records, tmp_path / "qc" / "selection.csv")
    scripts = write_slicer_scripts(records, tmp_path / "qc" / "slicer_scripts")
    review_paths = write_review_outputs(
        selected_cases=cases,
        records=records,
        scripts=scripts,
        output_dir=tmp_path / "qc",
        reviewer="tester",
        anatomies=["aorta"],
        tasks=["calcification"],
        comments=["inspect dynamic calcium tails"],
    )

    assert selection_path.exists()
    assert scripts and "loadVolume" in scripts[0].read_text(encoding="utf-8")
    tasks = pd.read_csv(review_paths["tasks_csv"])
    assert tasks.loc[0, "reviewer"] == "tester"
    assert tasks.loc[0, "task_comments"] == "inspect dynamic calcium tails"
    structured = json.loads(review_paths["structured_json"].read_text(encoding="utf-8"))
    assert structured["reviewer"] == "tester"
    assert structured["mask_count"] == 2
