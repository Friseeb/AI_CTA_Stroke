"""Synthetic phantom: a contrast-filled tube (vessel) with (a) a bright intraluminal sphere (embolus),
(b) an eccentric crescent of wall calcium, (c) a bright blob far from the vessel, (d) skull bone adjacent.
Only (a) must survive as a high-score candidate."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detect_calcified_emboli import Params, detect  # noqa: E402


def test_embolus_vs_wall_vs_far():
    shape = (80, 80, 80); sp = (0.5, 0.5, 0.5)
    g = np.mgrid[:80, :80, :80]
    img = np.full(shape, 35, np.float32)
    tube = ((g[0] - 40) ** 2 + (g[1] - 40) ** 2) <= 5 ** 2          # 5 mm diameter vessel along z
    img[tube] = 380
    emb = ((g[0] - 40) ** 2 + (g[1] - 40) ** 2 + (g[2] - 30) ** 2) <= 3 ** 2   # 3 mm embolus
    img[emb] = 1100
    crescent = (((g[0] - 40) ** 2 + (g[1] - 40) ** 2) <= 6 ** 2) & (((g[0] - 40) ** 2 + (g[1] - 40) ** 2) >= 4 ** 2) & (g[1] > 42) & (abs(g[2] - 55) <= 6)
    img[crescent] = 900                                             # eccentric wall calcium, outside lumen mostly
    far = ((g[0] - 10) ** 2 + (g[1] - 10) ** 2 + (g[2] - 10) ** 2) <= 3 ** 2
    img[far] = 1200
    bone = (g[0] >= 70); img[bone] = 1300                           # skull slab
    vessels = tube.astype(np.int32) * 5                             # label 5 = R-MCA
    rows, cand, meta = detect(img, vessels, sp, Params(), bone=bone)
    assert meta["threshold_hu"] >= 680
    assert len(rows) >= 1
    best = max(rows, key=lambda r: r["score"])
    c = best["centroid_ijk"]
    assert abs(c[2] - 30) < 2 and abs(c[0] - 40) < 2
    assert best["vessel_labels"] == "R-MCA"
    assert best["inside_vessel_fraction"] > 0.9
    # no candidate near the far blob or the bone slab
    for r in rows:
        assert r["centroid_ijk"][0] < 60
        assert not (abs(r["centroid_ijk"][0] - 10) < 4 and abs(r["centroid_ijk"][2] - 10) < 4)
    # wall crescent, if detected at all, scores lower than the embolus
    for r in rows:
        if abs(r["centroid_ijk"][2] - 55) < 8:
            assert r["score"] < best["score"]
