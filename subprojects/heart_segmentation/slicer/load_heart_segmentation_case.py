"""Load one HolOrama heart-segmentation case into 3D Slicer: CTA + the 14-structure labelmap, named and
coloured from the case's *_holorama_heart_labels.json.

Launch (must be `open -na` on macOS, otherwise the window never accepts input):
  HEART_CASE=<case_id> open -na /Applications/Slicer.app --args --python-script <this file>
Env: HEART_CASE (required), HEART_OUT_ROOT (default <repo>/outputs/heart_segmentation),
     HEART_CTA (default <repo>/outputs/daylight_evt_extended/<case>.nii.gz). Writes slicer_load.log into the case folder.
"""
import json
import os
import re
import traceback
from pathlib import Path

import slicer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE = os.environ.get("HEART_CASE")
if not CASE:
    raise SystemExit("set HEART_CASE=<case_id> in the environment before launching")
OUT_ROOT = Path(os.environ.get("HEART_OUT_ROOT", REPO / "outputs" / "heart_segmentation"))
CASE_DIR = OUT_ROOT / CASE
CTA = Path(os.environ.get("HEART_CTA", REPO / "outputs" / "daylight_evt_extended" / f"{CASE}.nii.gz"))
LOG = CASE_DIR / "slicer_load.log"
SEG = CASE_DIR / f"{CASE}_holorama_heart_seg.nii.gz"
LABELS = CASE_DIR / f"{CASE}_holorama_heart_labels.json"
FAT_KEYS = {"pericardial_fat", "epicardial_fat"}
HIDDEN_3D = {"pericardial_fat"}  # hides the rest in 3D; toggle it on in the Segmentations module


def log(msg):
    print(msg)
    with LOG.open("a") as fh:
        fh.write(msg + "\n")


def main():
    LOG.write_text("")
    log(f"case {CASE}")
    cta = slicer.util.loadVolume(str(CTA))
    if cta is None:
        raise RuntimeError(f"CTA failed to load: {CTA}")
    cta.SetName(f"{CASE} | CTA")
    d = cta.GetDisplayNode()
    d.AutoWindowLevelOff()
    d.SetWindow(900.0)
    d.SetLevel(250.0)
    table = {int(row["id"]): row for row in json.loads(LABELS.read_text())["labels"]} if LABELS.is_file() else {}
    lm = slicer.util.loadLabelVolume(str(SEG))
    if lm is None:
        raise RuntimeError(f"labelmap failed to load: {SEG}")
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{CASE} | HolOrama heart")
    seg.CreateDefaultDisplayNodes()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(cta)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, seg)
    slicer.mrmlScene.RemoveNode(lm)
    s = seg.GetSegmentation()
    disp = seg.GetDisplayNode()
    for i in range(s.GetNumberOfSegments()):
        sid = s.GetNthSegmentID(i)
        segment = s.GetSegment(sid)
        value = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else None
        if not value:
            m = re.search(r"(\d+)\s*$", segment.GetName())
            value = int(m.group(1)) if m else None
        row = table.get(value)
        if row is None:
            log(f"segment {sid} (label {value}) not in the label table")
            continue
        segment.SetName(f"{row['id']:02d} {row['name']}")
        h = row["color"].lstrip("#")
        segment.SetColor(*(int(h[j : j + 2], 16) / 255.0 for j in (0, 2, 4)))
        is_fat = row["key"] in FAT_KEYS
        disp.SetSegmentOpacity3D(sid, 0.25 if is_fat else 0.6)
        disp.SetSegmentVisibility3D(sid, row["key"] not in HIDDEN_3D)
        disp.SetSegmentOpacity2DFill(sid, 0.3 if is_fat else 0.45)
        disp.SetSegmentVisibility2DOutline(sid, True)
        log(f"loaded: {row['name']}")
    seg.CreateClosedSurfaceRepresentation()
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=cta, fit=True)
    for target in ("03 Left atrium", "04 Left ventricle"):
        sid = s.GetSegmentIdBySegmentName(target)
        if sid:
            c = [0.0, 0.0, 0.0]
            if seg.GetSegmentCenterRAS(sid, c):
                slicer.modules.markups.logic().JumpSlicesToLocation(c[0], c[1], c[2], True)
                log(f"jumped to {target} at RAS {c}")
                break
    slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
    log("done")


try:
    main()
except Exception:
    log(traceback.format_exc())
    raise
