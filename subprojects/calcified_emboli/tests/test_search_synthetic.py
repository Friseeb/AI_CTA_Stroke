"""Phantom brain: bright foci of each kind; the only distal_candidate must be the lone parenchymal sphere."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_intracranial_calcium import Params, preset, search  # noqa: E402


def _sph(shape, c, r):
    g = np.mgrid[: shape[0], : shape[1], : shape[2]]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r * r


def test_classification():
    shape = (120, 120, 100); sp = (1.0, 1.0, 1.0)
    brain = _sph(shape, (60, 60, 50), 45); skull = _sph(shape, (60, 60, 50), 50) & ~_sph(shape, (60, 60, 50), 46)
    img = np.full(shape, 35, np.float32); img[skull] = 1200
    vent = _sph(shape, (60, 60, 50), 8); third = _sph(shape, (60, 55, 50), 3)
    bg = _sph(shape, (45, 60, 50), 6) | _sph(shape, (75, 60, 50), 6)
    cow = np.zeros(shape, np.int32); g = np.mgrid[:120, :120, :100]
    tube = ((g[1] - 60) ** 2 + (g[2] - 30) ** 2) <= 3 ** 2; cow[tube & (g[0] > 60)] = 5; cow[tube & (g[0] <= 60)] = 7
    img[cow > 0] = 380
    foci = {"choroid": _sph(shape, (60, 60, 50), 2), "pineal": _sph(shape, (60, 46, 52), 2), "bg_l": _sph(shape, (45, 60, 50), 2), "bg_r": _sph(shape, (75, 60, 50), 2),
            "wall": (((g[1] - 60) ** 2 + (g[2] - 30) ** 2) <= 5 ** 2) & (((g[1] - 60) ** 2 + (g[2] - 30) ** 2) >= 3 ** 2) & (g[2] > 30) & (abs(g[0] - 85) <= 5),
            "intralum": _sph(shape, (40, 60, 30), 2), "distal": _sph(shape, (80, 80, 70), 2),
            "sym_l": _sph(shape, (35, 75, 60), 2), "sym_r": _sph(shape, (85, 75, 60), 2)}
    for m in foci.values():
        img[m] = 900
    masks = {"brain": brain, "skull": skull, "venous_sinuses": None, "ventricle": vent, "third_ventricle": third, "basal_ganglia": bg, "cow": cow, "anterior_is_plus_j": True}
    rows, out, meta, _ = search(img, masks, sp, preset(Params()))
    cats = {}
    for r in rows:
        c = r["centroid_ijk"]
        for name, m in foci.items():
            cm = np.array(np.where(m)).mean(axis=1)
            if np.linalg.norm(cm - np.array(c)) < 3:
                cats[name] = r["category"]
    assert cats["choroid"] == "choroid_plexus", cats
    assert cats["pineal"] == "pineal_habenula", cats
    assert cats["bg_l"] == "basal_ganglia" and cats["bg_r"] == "basal_ganglia"
    assert cats["wall"] == "arterial_wall"
    assert cats["intralum"] == "proximal_intraluminal"
    assert cats["sym_l"] == "symmetric_bilateral" and cats["sym_r"] == "symmetric_bilateral"
    assert cats["distal"] == "distal_candidate"
    assert meta["counts"]["distal_candidate"] == 1
