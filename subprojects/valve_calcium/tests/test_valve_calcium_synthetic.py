"""Synthetic smoke test: a vertical 'aorta' tube on an 'LV' sphere with an 'LA' sphere; calcium blobs placed
at the aortic cusp level and at the LA-LV interface must be scored in the right ROI."""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valve_calcium import Params, run_case  # noqa: E402


def _sphere(shape, c, r):
    g = np.mgrid[: shape[0], : shape[1], : shape[2]]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r * r


def test_roi_and_calcium(tmp_path):
    shape = (90, 90, 120)
    img = np.full(shape, 40, np.float32)
    lv = _sphere(shape, (45, 45, 35), 22)
    la = _sphere(shape, (45, 70, 35), 14) & ~lv
    myo = _sphere(shape, (45, 45, 35), 27) & ~lv & ~la
    g = np.mgrid[: shape[0], : shape[1], : shape[2]]
    tube = ((g[0] - 45) ** 2 + (g[1] - 45) ** 2 <= 10 ** 2)
    lvot = tube & (g[2] >= 52) & (g[2] < 60) & ~lv
    cusps = tube & (g[2] >= 60) & (g[2] < 72)
    rcc, lcc, ncc = cusps & (g[1] < 42), cusps & (g[1] >= 42) & (g[0] < 45), cusps & (g[1] >= 42) & (g[0] >= 45)
    aorta = tube & (g[2] >= 60)
    for m in (lv, la, lvot, aorta):
        img[m] = 350  # contrast blood pool
    img[myo] = 90
    av_cal = _sphere(shape, (45, 36, 66), 3)      # on the cusp
    mv_cal = _sphere(shape, (45, 67, 35), 3)      # LA-LV interface (LV ends at y=67)
    far_cal = _sphere(shape, (10, 10, 100), 3)    # outside everything
    for m in (av_cal, mv_cal, far_cal):
        img[m] = 900
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    cta = tmp_path / "case.nii.gz"; nib.save(nib.Nifti1Image(img, aff), cta)
    ch, si = tmp_path / "ch", tmp_path / "si"; ch.mkdir(); si.mkdir()
    for d, name, m in ((ch, "heart_atrium_left", la), (ch, "heart_ventricle_left", lv), (ch, "heart_myocardium", myo), (ch, "aorta", aorta),
                       (si, "left_ventricular_outflow_tract", lvot), (si, "right_coronary_cusp", rcc), (si, "left_coronary_cusp", lcc), (si, "non_coronary_cusp", ncc)):
        nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), d / f"{name}.nii.gz")
    # a STOPSTORM-style mitral label: a blob around the LA-LV interface
    ss = tmp_path / "ss"; ss.mkdir()
    nib.save(nib.Nifti1Image(_sphere(shape, (45, 67, 35), 8).astype(np.uint8), aff), ss / "case_Valve_Mitral.nii.gz")
    df = run_case(cta, ch, si, None, tmp_path / "out", Params(), "case", stopstorm_dir=ss)
    f = df.pivot(index="feature", columns="region", values="value")
    assert "mitral_valve_stopstorm" in f.columns and "aortic_valve_stopstorm" not in f.columns
    assert f.loc["calcium_volume_mm3", "mitral_valve_stopstorm"] > 50
    assert f.loc["calcium_volume_mm3", "aortic_valve"] > 50
    assert f.loc["calcium_volume_mm3", "mitral_annulus"] > 50
    assert f.loc["blood_reference_hu", "aortic_valve"] == 350
    assert f.loc["candidate_threshold_hu", "aortic_valve"] == 500
    assert (tmp_path / "out" / "case_valve_calcium_qc.png").is_file()
    av = np.asarray(nib.load(tmp_path / "out" / "case_av_calcium.nii.gz").dataobj) > 0
    mv = np.asarray(nib.load(tmp_path / "out" / "case_mv_calcium.nii.gz").dataobj) > 0
    assert not (av & far_cal).any() and not (mv & far_cal).any()
    assert (av & av_cal).sum() > 0.8 * av_cal.sum()
    assert (mv & mv_cal).sum() > 0.8 * mv_cal.sum()
    assert not (av & mv_cal).any()
