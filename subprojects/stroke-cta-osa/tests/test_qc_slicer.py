"""Slicer QC loader generator.

We don't run Slicer here — we just (1) ensure the generated script compiles
as valid Python and (2) ensure the auto-discovery walks the case dir and
picks up exactly the mask files that exist.
"""

import sys
import types
from pathlib import Path

import pytest


def _stub_slicer(monkeypatch):
    """Provide a stub `slicer` module so the generated script can be exec'd."""
    slicer = types.ModuleType("slicer")
    slicer.mrmlScene = types.SimpleNamespace(
        Clear=lambda *a, **kw: None,
        AddNewNodeByClass=lambda *a, **kw: types.SimpleNamespace(
            CreateDefaultDisplayNodes=lambda: None,
            SetDisplayVisibility=lambda *a: None,
            SetReferenceImageGeometryParameterFromVolumeNode=lambda *a: None,
            GetDisplayNode=lambda: None,
            GetSegmentation=lambda: types.SimpleNamespace(
                GetNumberOfSegments=lambda: 0,
                Modified=lambda: None,
                GetNthSegment=lambda i: None,
                GetNthSegmentID=lambda i: "",
            ),
            CreateClosedSurfaceRepresentation=lambda: None,
            Modified=lambda: None,
        ),
        RemoveNode=lambda *a: None,
    )
    util = types.SimpleNamespace(
        loadVolume=lambda *a, **kw: None,
        loadLabelVolume=lambda *a, **kw: None,
        setSliceViewerLayers=lambda *a, **kw: None,
        resetSliceViews=lambda: None,
        selectModule=lambda *a: None,
    )
    slicer.util = util
    slicer.modules = types.SimpleNamespace(
        segmentations=types.SimpleNamespace(
            logic=lambda: types.SimpleNamespace(
                ImportLabelmapToSegmentationNode=lambda *a, **kw: None,
            )
        )
    )
    slicer.app = types.SimpleNamespace(
        processEvents=lambda: None,
        layoutManager=lambda: types.SimpleNamespace(setLayout=lambda *a: None),
    )
    slicer.vtkMRMLLayoutNode = types.SimpleNamespace(SlicerLayoutFourUpView=3)
    monkeypatch.setitem(sys.modules, "slicer", slicer)


def test_generated_script_compiles(tmp_path):
    from stroke_cta_osa.qc_slicer import write_slicer_loader

    image_path = tmp_path / "cta.nii.gz"
    image_path.write_bytes(b"\x00")  # presence-only, not loaded by the test
    # Drop a couple of fake mask files matching the default roster
    for basename in ("mask_airway", "mask_fat_cervical_total", "mask_fat_parapharyngeal_left"):
        (tmp_path / f"{basename}.nii.gz").write_bytes(b"\x00")
    script = tmp_path / "demo_load_qc.py"
    out = write_slicer_loader(
        case_id="demo", image_path=image_path, case_dir=tmp_path,
        out_script=script,
    )
    text = out.read_text()
    # Compile-only check — Python syntax & no obvious indentation bugs.
    compile(text, str(out), "exec")
    # Sanity: discovered exactly the files present, in declared roster order.
    assert "mask_airway" in text
    assert "mask_fat_cervical_total" in text
    assert "mask_fat_parapharyngeal_left" in text
    # Files NOT present must not appear.
    assert "mask_fat_retropharyngeal" not in text
    assert "CASE_ID    = 'demo'" in text


def test_script_runs_against_stub_slicer(tmp_path, monkeypatch):
    """Exec the generated script with a stub `slicer` module — catches
    syntactic / attribute-access regressions that pure compile() misses."""
    from stroke_cta_osa.qc_slicer import write_slicer_loader

    _stub_slicer(monkeypatch)
    image_path = tmp_path / "cta.nii.gz"
    image_path.write_bytes(b"\x00")
    (tmp_path / "mask_airway.nii.gz").write_bytes(b"\x00")
    script = tmp_path / "demo_load_qc.py"
    write_slicer_loader(case_id="demo", image_path=image_path,
                        case_dir=tmp_path, out_script=script)
    namespace = {"__file__": str(script)}
    exec(compile(script.read_text(), str(script), "exec"), namespace)
    status = tmp_path / "demo_slicer_loader_status.txt"
    assert status.is_file()
    body = status.read_text()
    assert "Loaded QC scene for demo" in body


def test_landmark_block_emitted(tmp_path):
    from stroke_cta_osa.qc_slicer import write_slicer_loader

    image_path = tmp_path / "cta.nii.gz"
    image_path.write_bytes(b"\x00")
    script = tmp_path / "demo_load_qc.py"
    write_slicer_loader(case_id="demo", image_path=image_path,
                        case_dir=tmp_path, out_script=script,
                        min_csa_landmark_ras=(0.5, 1.5, -42.0))
    text = script.read_text()
    assert "min_csa" in text
    assert "vtkMRMLMarkupsFiducialNode" in text
    assert "-42.0" in text
    compile(text, str(script), "exec")


def test_extract_with_save_masks_writes_slicer_loader(synth_nifti_path, tmp_path):
    """End-to-end: orchestrator must drop the loader script next to masks."""
    from stroke_cta_osa.config import PipelineConfig, apply_overrides
    from stroke_cta_osa.features import extract_case

    cfg = apply_overrides(PipelineConfig(), {"output.save_masks": True})
    result = extract_case(synth_nifti_path, tmp_path, cfg, patient_id="loader_demo")
    script_path = Path(result.identifiers["slicer_loader_script"])
    assert script_path.is_file()
    assert script_path.name == "loader_demo_load_qc_in_slicer.py"
    compile(script_path.read_text(), str(script_path), "exec")
    # Auto-discovery picked up the masks the orchestrator just wrote.
    text = script_path.read_text()
    assert "mask_airway" in text
    assert "mask_fat_cervical_total" in text
