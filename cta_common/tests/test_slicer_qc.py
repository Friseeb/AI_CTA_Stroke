"""Tests for cta_common.slicer_qc using mocked `slicer`/`vtk` modules."""

import sys
import types

import pytest

from cta_common.slicer_qc import load_mask_aligned


class _Node:
    def __init__(self, name=""):
        self.name = name
        self.ijk = None
        self.default_displays = False
        self.ref_volume = None

    def GetIJKToRASMatrix(self, m):
        m.value = "cta-ijk"  # marker copied into the mask below

    def SetIJKToRASMatrix(self, m):
        self.ijk = m.value

    def CreateDefaultDisplayNodes(self):
        self.default_displays = True

    def SetReferenceImageGeometryParameterFromVolumeNode(self, vol):
        self.ref_volume = vol


class _Matrix:
    def __init__(self):
        self.value = None


def _install_fake_slicer(monkeypatch, *, lm_node):
    state = {"removed": [], "imported": []}

    seg_node = _Node("seg")

    util = types.SimpleNamespace(
        loadVolume=lambda path, properties=None: lm_node,
        loadSegmentation=lambda path: _Node("loaded-seg"),
    )
    mrmlScene = types.SimpleNamespace(
        AddNewNodeByClass=lambda cls, name: seg_node,
        RemoveNode=lambda n: state["removed"].append(n),
    )
    seg_logic = types.SimpleNamespace(
        ImportLabelmapToSegmentationNode=lambda lm, seg: state["imported"].append((lm, seg))
    )
    modules = types.SimpleNamespace(segmentations=types.SimpleNamespace(logic=lambda: seg_logic))

    fake_slicer = types.SimpleNamespace(util=util, mrmlScene=mrmlScene, modules=modules)
    fake_vtk = types.SimpleNamespace(vtkMatrix4x4=_Matrix)

    monkeypatch.setitem(sys.modules, "slicer", fake_slicer)
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)
    return seg_node, state


def test_no_cta_falls_back_to_load_segmentation(monkeypatch):
    _install_fake_slicer(monkeypatch, lm_node=_Node("lm"))
    result = load_mask_aligned("/x/mask.nii.gz", cta_node=None, name="m")
    assert result.name == "loaded-seg"


def test_aligned_path_copies_geometry_and_cleans_up(monkeypatch):
    lm = _Node("lm")
    seg_node, state = _install_fake_slicer(monkeypatch, lm_node=lm)
    cta = _Node("cta")

    result = load_mask_aligned("/x/mask.nii.gz", cta_node=cta, name="m")

    assert result is seg_node
    assert lm.ijk == "cta-ijk"            # CTA geometry copied onto the labelmap
    assert seg_node.default_displays is True
    assert seg_node.ref_volume is cta
    assert state["imported"] == [(lm, seg_node)]
    assert state["removed"] == [lm]       # temp labelmap removed


def test_failed_labelmap_returns_none(monkeypatch):
    _install_fake_slicer(monkeypatch, lm_node=None)
    assert load_mask_aligned("/x/mask.nii.gz", cta_node=_Node("cta"), name="m") is None
