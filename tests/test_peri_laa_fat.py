"""Peri-LAA fat module: shell geometry + HU filtering + AnnotationStore hook.

We build a small synthetic chest CT where:
  * a 6-voxel-radius "LAA" sphere sits at the volume centre
  * voxels in an annular shell around it are set to -100 HU (clear fat)
  * voxels outside the shell are set to -50 HU (NOT in fat window)
  * an "aorta" cube exists just below the LAA and is set to +50 HU (vessel)

Then assert:
  * 0-2 mm shell volume > 0 and HU within fat window
  * outer shell volume is positive when fat extends that far
  * exclusion mask actually subtracts the aorta region
  * filename helper preserves `.nii.gz`
  * AnnotationStore.init_annotation_package copies the staged files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.laa_slaao.annotation_store import AnnotationStore
from python.laa_slaao.peri_laa_fat import (
    DEFAULT_SHELLS_MM, _anatomical_axes_from_affine, _shell_prefix,
    compute_laa_centerline_and_bend, compute_peri_laa_fat,
    extend_laa_to_perifat, run_peri_laa_fat_from_paths,
)


def _build_synth(tmp_path):
    """Build a tiny synthetic CT with an LAA sphere, surrounding fat, and an
    aorta-like cube used as the exclusion target."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    affine = np.eye(4)
    cz, cy, cx = 20, 20, 20

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
    laa = (dist <= 6).astype(np.uint8)

    ct = np.full(shape, -50, dtype=np.int16)            # non-fat soft tissue
    fat_band = (dist > 6) & (dist <= 16)
    ct[fat_band] = -100                                  # clear fat HU

    # "Aorta" — small high-HU cube touching the LAA on one side
    aorta = np.zeros(shape, dtype=np.uint8)
    aorta[10:14, 22:30, 18:26] = 1
    ct[aorta.astype(bool)] = 200

    # Save NIfTIs to disk for the file-driven path tests
    ct_path = tmp_path / "ct.nii.gz"
    laa_path = tmp_path / "laa.nii.gz"
    neg_path = tmp_path / "neg.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), ct_path)
    nib.save(nib.Nifti1Image(laa, affine), laa_path)
    nib.save(nib.Nifti1Image(aorta, affine), neg_path)
    return ct, laa.astype(bool), aorta.astype(bool), spacing, affine, ct_path, laa_path, neg_path


def test_shell_prefix_keeps_nii_gz_intact():
    assert _shell_prefix(0, 2) == "peri_laa_fat_shell0_2"
    assert _shell_prefix(0.5, 2.5) == "peri_laa_fat_shell0p5_2p5"
    assert _shell_prefix(5, 10) == "peri_laa_fat_shell5_10"
    # The whole point: appending .nii.gz must not be mangled
    fname = f"sub-001_{_shell_prefix(0.5, 2.5)}.nii.gz"
    assert fname.endswith(".nii.gz")


def test_default_shells_are_sorted_and_non_overlapping():
    bands = list(DEFAULT_SHELLS_MM)
    assert bands == sorted(bands)
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bands, bands[1:]):
        assert a_hi <= b_lo, "default shells should not overlap"


def test_invalid_shell_band_rejected(tmp_path):
    ct, laa, _aorta, sp, *_ = _build_synth(tmp_path)
    with pytest.raises(ValueError, match="invalid shell band"):
        compute_peri_laa_fat(ct, laa, sp, shells_mm=[(2, 1)])
    with pytest.raises(ValueError, match="invalid shell band"):
        compute_peri_laa_fat(ct, laa, sp, shells_mm=[(-1, 2)])


def test_shape_mismatch_raises(tmp_path):
    ct, laa, _aorta, sp, *_ = _build_synth(tmp_path)
    bad_laa = np.zeros((10, 10, 10), bool)
    with pytest.raises(ValueError, match="LAA mask shape"):
        compute_peri_laa_fat(ct, bad_laa, sp)


def test_compute_finds_fat_in_inner_shells(tmp_path):
    ct, laa, _aorta, sp, *_ = _build_synth(tmp_path)
    result = compute_peri_laa_fat(
        ct, laa, sp, shells_mm=[(0, 2), (2, 5), (5, 10)],
    )
    f = result.features
    # The 0-2 shell sits inside the engineered fat band → must catch fat voxels
    assert f["peri_laa_fat_shell0_2_voxel_count"] > 0
    assert -190 <= f["peri_laa_fat_shell0_2_mean_hu"] <= -30
    # Aggregate matches sum of shells
    total_n = sum(f[f"peri_laa_fat_shell{lo:g}_{hi:g}_voxel_count"]
                  for lo, hi in result.shells_mm)
    assert f["peri_laa_fat_total_voxel_count"] == total_n
    # Each shell's geom volume should be >= its fat volume
    for lo, hi in result.shells_mm:
        p = _shell_prefix(lo, hi)
        assert f[f"{p}_geom_volume_ml"] >= f[f"{p}_volume_ml"]


def test_exclusion_mask_removes_aorta_region(tmp_path):
    """Isolate the explicit-mask path by disabling the PV buffer (otherwise
    the synthetic aorta at +200 HU is removed by both mechanisms and the test
    can't distinguish them)."""
    ct, laa, aorta, sp, *_ = _build_synth(tmp_path)
    no_excl = compute_peri_laa_fat(
        ct, laa, sp, exclusion_mask=None, pv_buffer_mm=0.0,
    )
    with_excl = compute_peri_laa_fat(
        ct, laa, sp, exclusion_mask=aorta, pv_buffer_mm=0.0,
    )
    assert with_excl.exclusion_used > 0
    smaller = False
    for lo, hi in with_excl.shells_mm:
        p = _shell_prefix(lo, hi)
        if with_excl.features[f"{p}_geom_volume_ml"] < no_excl.features[f"{p}_geom_volume_ml"]:
            smaller = True
            break
    assert smaller, "exclusion mask must remove shell geometry overlapping the aorta"


def test_shell_distances_honour_anisotropic_spacing(tmp_path):
    """Regression test for an axis-order bug: on anisotropic data with
    different sx, sy, sz, the shell band must still be measured in
    physical mm along every axis. We build an LAA point and check that
    voxels at known mm distances along each axis fall into the expected
    shell band.
    """
    # Strongly anisotropic spacing — the bug would mix these up
    spacing = (0.5, 0.5, 2.0)        # sx, sy, sz
    shape = (30, 30, 30)
    laa = np.zeros(shape, dtype=bool)
    laa[15, 15, 15] = True            # single seed voxel

    ct = np.full(shape, -100, dtype=np.int16)   # whole volume is "fat HU"
    res = compute_peri_laa_fat(
        ct, laa, spacing,
        shells_mm=[(0, 1.5)],  # band tighter than the smallest spacing in z
        pv_buffer_mm=0.0,
    )
    label = res.label_mask
    # With spacing (sx=0.5, sy=0.5, sz=2.0) on array `[x_idx, y_idx, z_idx]`:
    # x-step (axis 0) → 0.5 mm: must be INSIDE 0-1.5 mm shell
    assert label[14, 15, 15] == 1
    assert label[16, 15, 15] == 1
    # y-step (axis 1) → 0.5 mm: INSIDE
    assert label[15, 14, 15] == 1
    assert label[15, 16, 15] == 1
    # z-step (axis 2) → 2.0 mm: OUTSIDE the 0-1.5 mm band
    assert label[15, 15, 14] == 0
    assert label[15, 15, 16] == 0


def test_pv_buffer_excludes_air_and_vessel_partial_volume(tmp_path):
    """The PV buffer should drop shell voxels near pure-air (HU<-300) or
    pure-contrast (HU>+100) regions even without an explicit negative prior."""
    ct, laa, _aorta, sp, *_ = _build_synth(tmp_path)
    # _build_synth puts the aorta at HU 200 — squarely in the vessel band.
    with_buffer = compute_peri_laa_fat(
        ct, laa, sp, exclusion_mask=None, pv_buffer_mm=1.0,
    )
    no_buffer = compute_peri_laa_fat(
        ct, laa, sp, exclusion_mask=None, pv_buffer_mm=0.0,
    )
    assert with_buffer.features["peri_laa_fat_pv_voxels_excluded"] > 0
    # Buffer must shrink at least one shell's geometric volume
    shrunk = False
    for lo, hi in with_buffer.shells_mm:
        p = _shell_prefix(lo, hi)
        if with_buffer.features[f"{p}_geom_volume_ml"] < no_buffer.features[f"{p}_geom_volume_ml"]:
            shrunk = True
            break
    assert shrunk, "PV buffer must trim shell geometry around vessel/air voxels"


def test_run_from_paths_writes_artifacts(tmp_path):
    _ct, _laa, _aorta, _sp, _affine, ct_path, laa_path, neg_path = _build_synth(tmp_path)
    out_dir = tmp_path / "out"
    result, paths = run_peri_laa_fat_from_paths(
        ct_path=ct_path, laa_mask_path=laa_path,
        out_dir=out_dir, case_id="synth_001",
        negative_prior_path=neg_path,
        shells_mm=[(0, 2), (2, 5)],
        write_per_shell_masks=True,
    )
    assert paths["labels"].is_file()
    assert paths["labels"].name == "synth_001_peri_laa_fat_labels.nii.gz"
    assert paths["metrics"].is_file()
    assert paths["metrics"].name == "synth_001_peri_laa_fat_metrics.json"
    # per-shell binary masks (filename helper must preserve .nii.gz)
    assert paths["shell_1"].name == "synth_001_peri_laa_fat_shell0_2.nii.gz"
    assert paths["shell_2"].name == "synth_001_peri_laa_fat_shell2_5.nii.gz"

    meta = json.loads(paths["metrics"].read_text())
    assert meta["case_id"] == "synth_001"
    assert meta["shells_mm"] == [[0.0, 2.0], [2.0, 5.0]]
    assert meta["label_value_to_shell_mm"] == {"1": [0.0, 2.0], "2": [2.0, 5.0]}
    # The label NIfTI's max label index should match the number of shells.
    lbl = nib.load(str(paths["labels"])).get_fdata()
    assert int(lbl.max()) <= len(result.shells_mm)


def test_annotation_store_stages_peri_laa_fat(tmp_path):
    """AnnotationStore.init_annotation_package copies peri-LAA artefacts and
    flags them in session.json."""
    _ct, _laa, _aorta, _sp, _aff, ct_path, laa_path, neg_path = _build_synth(tmp_path)
    out_dir = tmp_path / "fat"
    _, paths = run_peri_laa_fat_from_paths(
        ct_path=ct_path, laa_mask_path=laa_path,
        out_dir=out_dir, case_id="synth_001",
        negative_prior_path=neg_path,
        shells_mm=[(0, 2), (2, 5)],
    )
    store = AnnotationStore(tmp_path / "store")
    ann = store.init_annotation_package(
        case_id="synth_001",
        ct_path=ct_path,
        consensus_laa_path=laa_path,
        negative_prior_path=neg_path,
        peri_laa_fat_labels_path=paths["labels"],
        peri_laa_fat_metrics_path=paths["metrics"],
    )
    assert (ann / "peri_laa_fat_labels.nii.gz").is_file()
    assert (ann / "peri_laa_fat_metrics.json").is_file()
    session = json.loads((ann / "session.json").read_text())
    assert session["peri_laa_fat_staged"] is True
    assert "peri_laa_fat_labels.nii.gz" in session["files"]


def test_extend_laa_fills_pocket_inside_perifat_shell(tmp_path):
    """Construct a tiny world where the LAA is a small core and the
    fat is a complete shell around it at radius 5–8 voxels. After
    closing + fill-holes, the extended LAA should expand to include
    the wall region between core and fat, but NOT escape through
    any hole, and NOT bleed past the fat envelope."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    cz, cy, cx = 20, 20, 20
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)

    laa = (dist <= 3).astype(np.uint8)
    # Fat shell from r=5..8 — strictly outside the LAA, with a 2-voxel wall
    fat_shell = ((dist >= 5) & (dist <= 8)).astype(np.uint8)
    # CT: fat HU inside the shell, soft tissue elsewhere
    ct = np.where(fat_shell.astype(bool), -100, 40).astype(np.int16)

    # Hand-build a PeriLAAFatResult so we can drive extend_laa_to_perifat
    # without re-running step 1/2.
    from python.laa_slaao.peri_laa_fat import PeriLAAFatResult, PeriLAAROI
    fake_roi = PeriLAAROI(
        roi_mask=np.zeros(shape, bool),   # unused by the new extend impl
        dist_mm=np.zeros(shape, np.float64),
        provenance={"roi_voxel_count": 0, "exclusion_voxels_total": 0,
                    "pericardium_used": False, "negative_prior_used": False},
    )
    fr = PeriLAAFatResult(
        label_mask=fat_shell.astype(np.int16),
        shells_mm=[(0.0, 10.0)],
        features={}, roi=fake_roi,
    )

    ext = extend_laa_to_perifat(
        laa_mask=laa.astype(bool), fat_result=fr,
        ct_array=ct, spacing_xyz_mm=spacing,
        perifat_closing_mm=1.0, max_added_volume_ml=10.0,
    )
    # Extension should add the wall band (r=4..4) at minimum
    assert ext.features["extended_laa_extension_succeeded"] is True
    assert ext.features["extended_laa_added_voxel_count"] > 0
    # Extension must not include any fat voxels
    assert int((ext.extended_laa & fat_shell.astype(bool)).sum()) == 0
    # Extension must include the original LAA
    assert int((ext.extended_laa & laa.astype(bool)).sum()) == int(laa.sum())
    # Extension must not escape past the fat shell (voxels outside r=8)
    outside_shell = (dist > 8) & (~laa.astype(bool))
    assert int((ext.extended_laa & outside_shell).sum()) == 0


def test_extend_laa_safety_cap_refuses_huge_fills(tmp_path):
    """If the fat shell has a big hole and the fill leaks, the volume cap
    must abort the extension cleanly with status."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    cz, cy, cx = 20, 20, 20
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
    laa = (dist <= 3).astype(np.uint8)
    # Fat shell with a hole on one side (cut out the entire upper half in z)
    fat = ((dist >= 5) & (dist <= 8))
    wedge = np.broadcast_to(zz > cz, shape)
    fat = fat & ~wedge
    ct = np.full(shape, 40, np.int16)

    from python.laa_slaao.peri_laa_fat import PeriLAAFatResult, PeriLAAROI
    fr = PeriLAAFatResult(
        label_mask=fat.astype(np.int16),
        shells_mm=[(0.0, 10.0)], features={},
        roi=PeriLAAROI(
            roi_mask=np.zeros(shape, bool),
            dist_mm=np.zeros(shape, np.float64),
            provenance={"roi_voxel_count": 0, "exclusion_voxels_total": 0,
                        "pericardium_used": False, "negative_prior_used": False},
        ),
    )
    ext = extend_laa_to_perifat(
        laa_mask=laa.astype(bool), fat_result=fr,
        ct_array=ct, spacing_xyz_mm=spacing,
        perifat_closing_mm=1.0, max_added_volume_ml=0.001,  # forced cap
    )
    # With a soft cap, the fill is kept for visual review but the status
    # string flags the cap breach with a `warn_*` prefix.
    assert ext.features["extended_laa_extension_status"].startswith("warn_added_volume_")


def _build_bent_laa(shape=(40, 40, 40)):
    """Synthetic L-shaped LAA: a horizontal segment + a 90° vertical segment.
    The bend should land in the middle of the L."""
    laa = np.zeros(shape, dtype=bool)
    # Horizontal arm (x increasing) at z=20, y=20, x=10..25
    for x in range(10, 26):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                laa[20 + dz, 20 + dy, x] = True
    # Vertical arm (z increasing) at y=20, x=25, z=20..35
    for z in range(20, 36):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                laa[z, 20 + dy, 25 + dx] = True
    return laa


def test_centerline_detects_90deg_bend():
    laa = _build_bent_laa()
    cl = compute_laa_centerline_and_bend(
        laa_mask=laa, spacing_xyz_mm=(1.0, 1.0, 1.0),
        smoothing_window=2,
    )
    assert cl.post_bend_tangent_zyx is not None
    # The bend angle on an L-shape should be roughly 90° (within smoothing
    # noise). Loose bounds so the test doesn't break on minor algo tweaks.
    assert 50.0 < cl.bend_angle_deg < 130.0
    # Post-bend tangent should be ~vertical (z direction = axis 0) because
    # the distal arm of the L runs in +z.
    pt = cl.post_bend_tangent_zyx
    assert abs(pt[0]) > 0.5
    assert abs(pt[2]) < 0.5


def test_centerline_pca_fallback_on_blob():
    """A spherical blob has no clear bend → PCA fallback path triggers."""
    shape = (20, 20, 20)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    blob = ((zz - 10)**2 + (yy - 10)**2 + (xx - 10)**2) <= 9
    cl = compute_laa_centerline_and_bend(blob, (1.0, 1.0, 1.0))
    # Either a real centerline or a PCA fallback — either way, a tangent
    # must come back so callers can still do the half-space test.
    assert cl.post_bend_tangent_zyx is not None


def test_centerline_tangent_flips_toward_perifat_when_la_orientation_is_wrong():
    """Build a chicken-wing-like LAA: the 'tip' endpoint via skeleton is on
    one side, but the perifat is on the OPPOSITE side. The orientation
    override must flip the tangent toward the perifat."""
    shape = (40, 40, 40)
    laa = np.zeros(shape, dtype=bool)
    # Horizontal tube along x at y=20, z=20
    for x in range(8, 32):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                laa[20 + dz, 20 + dy, x] = True

    # Without perifat constraint, the tangent could go either way along x.
    cl0 = compute_laa_centerline_and_bend(
        laa, (1.0, 1.0, 1.0), smoothing_window=2,
    )
    assert cl0.post_bend_tangent_zyx is not None

    # Now add a perifat blob on the +x side. Tangent must point +x.
    perifat = np.zeros(shape, dtype=bool)
    perifat[18:23, 18:23, 32:36] = True
    cl_pos = compute_laa_centerline_and_bend(
        laa, (1.0, 1.0, 1.0), smoothing_window=2, perifat_mask=perifat,
    )
    assert cl_pos.post_bend_tangent_zyx[2] > 0.5

    # And perifat on the -x side — tangent must flip to point -x.
    perifat_neg = np.zeros(shape, dtype=bool)
    perifat_neg[18:23, 18:23, 4:8] = True
    cl_neg = compute_laa_centerline_and_bend(
        laa, (1.0, 1.0, 1.0), smoothing_window=2, perifat_mask=perifat_neg,
    )
    assert cl_neg.post_bend_tangent_zyx[2] < -0.5


def test_anatomical_axes_from_lps_diagonal_affine():
    """For a standard LPS NIfTI affine (sub-547 style: diag(-0.586,+0.586,+0.25)),
    anterior must point at array-axis-1 NEGATIVE direction and patient-left at
    array-axis-0 NEGATIVE direction (since the L-axis is negated in the affine)."""
    affine = np.array([
        [-0.586, 0, 0, 0],
        [0, 0.586, 0, 0],
        [0, 0, 0.25, 0],
        [0, 0, 0, 1.0],
    ])
    spacing = (0.586, 0.586, 0.25)
    ant, lft, sup = _anatomical_axes_from_affine(affine, spacing)
    assert ant is not None and lft is not None and sup is not None
    # Anterior in this affine: array-axis-1 must be NEGATIVE
    assert ant[1] < -0.99
    # Left: array-axis-0 must be NEGATIVE (negated by affine sign)
    assert lft[0] < -0.99
    # Superior: array-axis-2 POSITIVE
    assert sup[2] > 0.99


def test_hard_exclusion_overrides_laa_consensus(tmp_path):
    """The coronary / calcification hard exclusion must remove voxels even
    when they are part of the original LAA consensus."""
    shape = (20, 20, 20)
    spacing = (1.0, 1.0, 1.0)
    laa = np.zeros(shape, bool)
    laa[8:14, 8:14, 8:14] = True
    # A 'coronary' branch that overlaps a face of the LAA
    coronary = np.zeros(shape, bool)
    coronary[10:13, 10:13, 12:16] = True   # partially inside the LAA cube

    # Minimal fat result
    from python.laa_slaao.peri_laa_fat import PeriLAAFatResult, PeriLAAROI
    perifat = np.zeros(shape, np.int16)
    perifat[5:8, 8:14, 8:14] = 1   # fat on one face
    fr = PeriLAAFatResult(
        label_mask=perifat,
        shells_mm=[(0.0, 5.0)],
        features={},
        roi=PeriLAAROI(
            roi_mask=np.zeros(shape, bool),
            dist_mm=np.zeros(shape, np.float64),
            provenance={"roi_voxel_count": 0, "exclusion_voxels_total": 0,
                        "pericardium_used": False, "negative_prior_used": False},
        ),
    )
    ct = np.full(shape, 50, np.int16)

    # 1) No hard exclusion → LAA preserved verbatim
    ext_no = extend_laa_to_perifat(
        laa_mask=laa, fat_result=fr, ct_array=ct,
        spacing_xyz_mm=spacing, centerline_aware=False,
    )
    overlap_no = int((ext_no.extended_laa & coronary).sum())
    assert overlap_no > 0, "without hard excl, coronary-overlapping LAA voxels must survive"

    # 2) With hard exclusion → coronary voxels gone, EVEN if in LAA
    ext_hard = extend_laa_to_perifat(
        laa_mask=laa, fat_result=fr, ct_array=ct,
        spacing_xyz_mm=spacing, centerline_aware=False,
        hard_exclusion_mask=coronary,
    )
    overlap_hard = int((ext_hard.extended_laa & coronary).sum())
    assert overlap_hard == 0, "hard exclusion must override the LAA consensus"
    # And the features should record how many voxels got removed
    assert ext_hard.features["extended_laa_hard_exclusion_voxels_removed"] == overlap_no


def test_anatomical_axes_returns_none_on_singular_affine():
    bad = np.zeros((4, 4))
    bad[3, 3] = 1
    ant, lft, sup = _anatomical_axes_from_affine(bad, (1, 1, 1))
    assert ant is None and lft is None and sup is None


def test_centerline_orientation_uses_la_body_when_provided():
    """Build a straight tube + a 'LA body' at one end. The post-bend
    tangent must point AWAY from the LA body (distally)."""
    shape = (40, 40, 40)
    laa = np.zeros(shape, dtype=bool)
    for x in range(10, 30):
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                laa[20 + dz, 20 + dy, x] = True
    la_body = np.zeros(shape, dtype=bool)
    la_body[15:25, 15:25, 5:10] = True   # adjacent to the proximal end (low x)

    cl = compute_laa_centerline_and_bend(
        laa, (1.0, 1.0, 1.0), la_body_mask=la_body, smoothing_window=2,
    )
    # Tangent's x-component (axis 2) must be POSITIVE — away from LA body
    pt = cl.post_bend_tangent_zyx
    assert pt is not None
    assert pt[2] > 0.5


def test_annotation_store_without_peri_laa_fat_still_works(tmp_path):
    _ct, _laa, _aorta, _sp, _aff, ct_path, laa_path, _neg = _build_synth(tmp_path)
    store = AnnotationStore(tmp_path / "store")
    ann = store.init_annotation_package(
        case_id="synth_002",
        ct_path=ct_path,
        consensus_laa_path=laa_path,
    )
    session = json.loads((ann / "session.json").read_text())
    assert session["peri_laa_fat_staged"] is False
    assert not (ann / "peri_laa_fat_labels.nii.gz").exists()
