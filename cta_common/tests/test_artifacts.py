"""Tests for cta_common.artifacts (metal detection + burden)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from cta_common.artifacts import (
    artifact_burden,
    artifact_masks,
    body_mask,
    classify_burden,
    detect_metal,
)

SPACING = (1.0, 1.0, 1.0)  # array order (z, y, x), 1 mm^3 voxels


def _phantom(n=40):
    """Background air (-1000) with a soft-tissue body and a central metal cube."""
    hu = np.full((n, n, n), -1000.0, dtype=np.float32)
    c = n // 2
    hu[5:n - 5, 5:n - 5, 5:n - 5] = 40.0          # soft-tissue body
    hu[c - 2:c + 2, c - 2:c + 2, c - 2:c + 2] = 4000.0  # metal cube
    return hu


def test_detect_metal_threshold():
    hu = _phantom()
    m = detect_metal(hu, metal_hu=2500)
    assert m.sum() == 4 ** 3
    assert not detect_metal(hu, metal_hu=5000).any()


def test_body_mask_excludes_background_air():
    hu = _phantom()
    body = body_mask(hu)
    assert body[20, 20, 20]          # inside body
    assert not body[0, 0, 0]         # background air excluded


def test_artifact_masks_bloom_surrounds_metal_inside_body():
    hu = _phantom()
    m = artifact_masks(hu, SPACING, bloom_mm=2.0, streak_mm=5.0)
    assert m.metal.sum() == 4 ** 3
    assert m.bloom.any()
    assert not (m.metal & m.bloom).any()          # disjoint
    assert not (m.bloom & ~m.body).any()          # bloom stays in body
    assert (m.core == (m.metal | m.bloom)).all()


def test_no_metal_returns_empty_artifact():
    hu = np.full((20, 20, 20), 40.0, dtype=np.float32)
    hu[:3] = -1000.0
    m = artifact_masks(hu, SPACING)
    assert not m.metal.any()
    assert not m.artifact.any()


def test_burden_volumes_and_roi_fraction():
    hu = _phantom()
    m = artifact_masks(hu, SPACING, bloom_mm=2.0, streak_mm=5.0)
    # ROI = a box that fully contains the metal+bloom
    roi = np.zeros_like(hu, dtype=bool)
    c = hu.shape[0] // 2
    roi[c - 6:c + 6, c - 6:c + 6, c - 6:c + 6] = True
    b = artifact_burden(m, SPACING, roi_mask=roi)
    assert b["has_metal"] is True
    assert b["roi_has_metal"] is True
    assert b["metal_ml"] == pytest.approx(64 / 1000.0)   # 4^3 voxels * 1mm^3 /1000
    assert b["artifact_ml"] >= b["core_ml"] >= b["metal_ml"]
    assert 0.0 < b["roi_artifact_fraction"] <= 1.0
    assert b["n_metal_components"] == 1


def test_classify_burden_levels():
    assert classify_burden({"has_metal": False}) == "none"
    # ROI-scoped gating: global metal but none in the ROI -> none
    assert classify_burden({"has_metal": True, "roi_has_metal": False, "roi_artifact_fraction": 0.0}) == "none"
    # metal in ROI but zero artifact spread -> none
    assert classify_burden({"roi_has_metal": True, "roi_artifact_fraction": 0.0}) == "none"
    assert classify_burden({"roi_has_metal": True, "roi_artifact_fraction": 0.005}) == "low"
    assert classify_burden({"roi_has_metal": True, "roi_artifact_fraction": 0.02}) == "moderate"
    assert classify_burden({"roi_has_metal": True, "roi_artifact_fraction": 0.20}) == "high"
    # metal present, no ROI metric available -> low
    assert classify_burden({"has_metal": True}) == "low"
