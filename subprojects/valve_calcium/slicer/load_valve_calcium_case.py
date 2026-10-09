"""Load one valve-calcium case into 3D Slicer: CTA + chambers, cusps, coronaries, valve ROIs, calcium.

Launch (must be `open -na` on macOS, otherwise the window never accepts input):
  VALVE_CASE=<case_id> open -na /Applications/Slicer.app --args --python-script <this file>
Env: VALVE_CASE (required), VALVE_OUT_ROOT (default <repo>/outputs/valve_calcium), VALVE_CTA (default
<repo>/outputs/daylight_evt_extended/<case>.nii.gz). Writes slicer_load.log into the case folder.
"""
import os, traceback
from pathlib import Path
import slicer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE = os.environ.get("VALVE_CASE")
if not CASE:
    raise SystemExit("set VALVE_CASE=<case_id> in the environment before launching")
OUT_ROOT = Path(os.environ.get("VALVE_OUT_ROOT", REPO / "outputs" / "valve_calcium"))
CASE_DIR = OUT_ROOT / CASE
CTA = Path(os.environ.get("VALVE_CTA", REPO / "outputs" / "daylight_evt_extended" / f"{CASE}.nii.gz"))
LOG = CASE_DIR / "slicer_load.log"

# (path, display name, rgb, opacity3d, show3d, show2d)
LAYERS = [
    ("heartchambers/heart_atrium_left.nii.gz", "LA", (0.85, 0.55, 0.75), 0.25, True, True),
    ("heartchambers/heart_ventricle_left.nii.gz", "LV", (0.75, 0.35, 0.45), 0.25, True, True),
    ("heartchambers/heart_myocardium.nii.gz", "myocardium", (0.55, 0.30, 0.30), 0.15, False, True),
    ("heartchambers/aorta.nii.gz", "aorta", (0.90, 0.25, 0.20), 0.20, True, True),
    ("aortic_sinuses/left_ventricular_outflow_tract.nii.gz", "LVOT", (0.95, 0.65, 0.30), 0.5, True, True),
    ("aortic_sinuses/right_coronary_cusp.nii.gz", "RCC", (0.95, 0.80, 0.30), 0.5, True, True),
    ("aortic_sinuses/left_coronary_cusp.nii.gz", "LCC", (0.30, 0.80, 0.40), 0.5, True, True),
    ("aortic_sinuses/non_coronary_cusp.nii.gz", "NCC", (0.30, 0.60, 0.95), 0.5, True, True),
    ("coronary/coronary_arteries.nii.gz", "coronaries", (0.95, 0.40, 0.70), 0.9, True, True),
    (f"analysis/{CASE}_av_roi.nii.gz", "aortic valve ROI", (0.00, 0.90, 0.90), 0.25, True, True),
    (f"analysis/{CASE}_mv_roi.nii.gz", "mitral annulus ROI", (0.00, 0.60, 0.90), 0.35, True, True),
    (f"stopstorm/full_grid/{CASE}_Valve_Mitral.nii.gz", "STOPSTORM mitral valve", (0.60, 0.20, 0.90), 0.6, True, True),
    (f"stopstorm/full_grid/{CASE}_Valve_Tricuspid.nii.gz", "STOPSTORM tricuspid valve", (0.20, 0.70, 0.70), 0.4, False, True),
    (f"stopstorm/full_grid/{CASE}_Valve_Pulmonic.nii.gz", "STOPSTORM pulmonic valve", (0.50, 0.50, 0.90), 0.4, False, True),
    (f"analysis/{CASE}_av_calcium.nii.gz", "AV calcium", (1.00, 0.10, 0.10), 1.0, True, True),
    (f"analysis/{CASE}_mv_calcium.nii.gz", "MV calcium", (1.00, 0.55, 0.00), 1.0, True, True),
]


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
    d = cta.GetDisplayNode(); d.AutoWindowLevelOff(); d.SetWindow(900.0); d.SetLevel(250.0)
    log("CTA loaded")
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{CASE} | valve calcium")
    seg.CreateDefaultDisplayNodes()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(cta)
    disp = seg.GetDisplayNode()
    for rel, name, rgb, op3d, show3d, show2d in LAYERS:
        path = CASE_DIR / rel
        if not path.is_file():
            log(f"missing: {path}"); continue
        lm = slicer.util.loadLabelVolume(str(path))
        if lm is None:
            log(f"failed: {path}"); continue
        tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, tmp)
        s = tmp.GetSegmentation()
        if s.GetNumberOfSegments() == 0:
            log(f"empty: {name}")
        else:
            sid = s.GetNthSegmentID(0)
            seg.GetSegmentation().CopySegmentFromSegmentation(s, sid)
            new_id = seg.GetSegmentation().GetNthSegmentID(seg.GetSegmentation().GetNumberOfSegments() - 1)
            segment = seg.GetSegmentation().GetSegment(new_id)
            segment.SetName(name); segment.SetColor(*rgb)
            disp.SetSegmentOpacity3D(new_id, op3d)
            disp.SetSegmentVisibility3D(new_id, show3d)
            disp.SetSegmentVisibility2DFill(new_id, show2d)
            disp.SetSegmentVisibility2DOutline(new_id, True)
            disp.SetSegmentOpacity2DFill(new_id, 0.35 if "calcium" not in name else 0.8)
            log(f"loaded: {name}")
        slicer.mrmlScene.RemoveNode(tmp); slicer.mrmlScene.RemoveNode(lm)
    seg.CreateClosedSurfaceRepresentation()
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=cta, fit=True)
    # jump all slices to the aortic valve calcium (or ROI) centroid
    for target in ("AV calcium", "aortic valve ROI"):
        sid = seg.GetSegmentation().GetSegmentIdBySegmentName(target)
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
