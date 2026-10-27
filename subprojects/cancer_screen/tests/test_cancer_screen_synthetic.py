import sys
from pathlib import Path
import numpy as np, nibabel as nib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cancer_screen import Params, run_case  # noqa: E402


def _sphere(shape, c, r):
    g = np.mgrid[: shape[0], : shape[1], : shape[2]]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r * r


def test_lung_and_thyroid_rules(tmp_path):
    shape = (100, 100, 100); aff = np.diag([1.0, 1.0, 1.0, 1.0])
    img = np.full(shape, 40, np.float32)
    lung = _sphere(shape, (30, 50, 40), 25); img[lung] = -800
    n_small = _sphere(shape, (30, 50, 40), 2); n_big = _sphere(shape, (40, 60, 50), 5)   # 4 mm and 10 mm nodules
    img[n_small] = 20; img[n_big] = 30
    thy = _sphere(shape, (75, 50, 70), 10); img[thy] = 150
    hypo = _sphere(shape, (75, 50, 70), 5); img[hypo] = 60                               # 10 mm hypodense nodule
    cta = tmp_path / "c.nii.gz"; nib.save(nib.Nifti1Image(img, aff), cta)
    ts = tmp_path / "ts"; (ts / "total").mkdir(parents=True); (ts / "lung_nodules").mkdir(); (ts / "head_glands").mkdir()
    nib.save(nib.Nifti1Image(lung.astype(np.uint8), aff), ts / "total" / "lung_upper_lobe_left.nii.gz")
    nib.save(nib.Nifti1Image(thy.astype(np.uint8), aff), ts / "total" / "thyroid_gland.nii.gz")
    nib.save(nib.Nifti1Image((n_small | n_big).astype(np.uint8), aff), ts / "lung_nodules" / "lung_nodules.nii.gz")
    df = run_case(cta, ts, tmp_path / "out", Params(), "c")
    f = df.set_index(["region", "feature"]).value
    assert f[("lung", "nodules_total")] == 2 and f[("lung", "nodules_gt_8mm")] == 1
    assert f[("thyroid", "gland_found")] == 1 and f[("thyroid", "focal_candidates_ge_8mm")] >= 1
    assert f[("lymph_nodes", "model_available")] == 0
    assert (tmp_path / "out" / "c_cancer_screen_lesions.mrk.json").is_file()
