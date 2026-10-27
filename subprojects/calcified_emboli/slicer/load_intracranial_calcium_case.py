"""Load an intracranial calcium search result into 3D Slicer: CTA + class labelmap as a segmentation (one segment per
class), the TopCoW arteries, brain/ventricle context, and the embolus-candidate fiducials.

  VALVE-like env:  CCE_CASE=<case_id> CCE_IMAGE=<cta.nii.gz> [CCE_COW=<topcow_seg>] open -na /Applications/Slicer.app --args --python-script <this file>
Case folder is <repo>/outputs/calcified_emboli/<case_id> (single-case layout: total_bone/, brain_structures/,
intracranial_search/). For the cohort layout (<root>/<case>/ncct/{total,brain_structures,search}) set
CCE_CASE_DIR and CCE_MOD (ncct|cta), e.g.
  CCE_CASE=sub-cce1617 CCE_CASE_DIR=outputs/calcified_emboli_cohort/sub-cce1617 CCE_MOD=ncct \
  CCE_IMAGE=outputs/calcified_emboli_cohort/sub-cce1617/raw/sub-cce1617_acq-NCCT_ct.nii.gz open -na /Applications/Slicer.app --args --python-script <this file>
Writes slicer_load.log into the case folder.
"""
import os, traceback
from pathlib import Path
import slicer

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
CASE = os.environ.get("CCE_CASE") or (_ for _ in ()).throw(SystemExit("set CCE_CASE=<case_id>"))
CASE_DIR = Path(os.environ.get("CCE_CASE_DIR") or (REPO / "outputs" / "calcified_emboli" / CASE))
if not CASE_DIR.is_absolute():
    CASE_DIR = REPO / CASE_DIR
IMAGE = Path(os.environ.get("CCE_IMAGE") or (REPO / "outputs" / "daylight_recovered" / f"{CASE}.nii.gz"))
if not IMAGE.is_absolute():
    IMAGE = REPO / IMAGE
COW = os.environ.get("CCE_COW")
MOD = os.environ.get("CCE_MOD")  # cohort layout: ncct | cta
SUB = CASE_DIR / MOD if MOD else CASE_DIR
SEARCH = SUB / "search" if MOD else CASE_DIR / "intracranial_search"
TOTAL = SUB / "total" if MOD else CASE_DIR / "total_bone"
BRAINS = SUB / "brain_structures"
LOG = CASE_DIR / "slicer_load.log"
CLASSES = {1: ("distal_candidate", (1.0, 0.1, 0.1)), 2: ("proximal_intraluminal", (1.0, 0.5, 0.0)), 3: ("arterial_wall", (0.9, 0.8, 0.2)),
           4: ("choroid_plexus", (0.3, 0.6, 1.0)), 5: ("pineal_habenula", (0.6, 0.3, 0.9)), 6: ("basal_ganglia", (0.3, 0.8, 0.4)),
           7: ("symmetric_bilateral", (0.2, 0.7, 0.7)), 8: ("skull_dural", (0.7, 0.7, 0.7)), 9: ("dural_sinus", (0.5, 0.5, 0.8)), 10: ("parenchymal_other", (0.9, 0.9, 0.9)),
           11: ("arterial_wall_adjacent", (0.8, 0.7, 0.3)), 12: ("skull_dural_adjacent", (0.6, 0.6, 0.6)), 13: ("dural_sinus_adjacent", (0.45, 0.45, 0.7)), 14: ("subthreshold_noise", (0.35, 0.35, 0.35)), 15: ("distal_candidate_low", (0.9, 0.5, 0.5))}


def log(m):
    print(m)
    with LOG.open("a") as fh:
        fh.write(m + "\n")


def add_labelmap(path, seg, names_colors, opacity=0.5, show3d=True):
    lm = slicer.util.loadLabelVolume(str(path))
    tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, tmp)
    s = tmp.GetSegmentation(); disp = seg.GetDisplayNode()
    for i in range(s.GetNumberOfSegments()):
        sid = s.GetNthSegmentID(i); segm = s.GetSegment(sid)
        try:
            lv = int(segm.GetLabelValue())
        except Exception:
            lv = i + 1
        name, rgb = names_colors.get(lv, (f"label {lv}", (0.8, 0.8, 0.8)))
        seg.GetSegmentation().CopySegmentFromSegmentation(s, sid)
        nid = seg.GetSegmentation().GetNthSegmentID(seg.GetSegmentation().GetNumberOfSegments() - 1)
        seg.GetSegmentation().GetSegment(nid).SetName(name); seg.GetSegmentation().GetSegment(nid).SetColor(*rgb)
        disp.SetSegmentOpacity3D(nid, opacity); disp.SetSegmentVisibility3D(nid, show3d); disp.SetSegmentOpacity2DFill(nid, 0.6)
        log(f"segment {name}")
    slicer.mrmlScene.RemoveNode(tmp); slicer.mrmlScene.RemoveNode(lm)


def main():
    LOG.write_text(""); log(f"case {CASE}")
    cta = slicer.util.loadVolume(str(IMAGE)); cta.SetName(f"{CASE} | CTA")
    d = cta.GetDisplayNode(); d.AutoWindowLevelOff()
    if MOD == "ncct":
        d.SetWindow(160.0); d.SetLevel(60.0)       # brain-ish window; calcium saturates white
    else:
        d.SetWindow(700.0); d.SetLevel(200.0)
    log("image loaded")
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{CASE} | intracranial calcium classes")
    seg.CreateDefaultDisplayNodes(); seg.SetReferenceImageGeometryParameterFromVolumeNode(cta)
    add_labelmap(SEARCH / f"{CASE}_intracranial_calcium_classes.nii.gz", seg, CLASSES, opacity=1.0)
    ctx = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{CASE} | context")
    ctx.CreateDefaultDisplayNodes(); ctx.SetReferenceImageGeometryParameterFromVolumeNode(cta)
    for p, name, rgb, op in ((TOTAL / "brain.nii.gz", "brain", (0.85, 0.75, 0.7), 0.08), (BRAINS / "ventricle.nii.gz", "ventricles", (0.3, 0.6, 1.0), 0.3),
                             (BRAINS / "venous_sinuses.nii.gz", "venous sinuses", (0.4, 0.4, 0.8), 0.3)):
        if p.is_file():
            add_labelmap(p, ctx, {1: (name, rgb)}, opacity=op, show3d=name != "brain")
    if COW and Path(COW).is_file():
        cow = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{CASE} | TopCoW arteries")
        cow.CreateDefaultDisplayNodes(); cow.SetReferenceImageGeometryParameterFromVolumeNode(cta)
        names = {1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom", 11: "R-ACA", 12: "L-ACA", 15: "3rd-A2"}
        add_labelmap(Path(COW), cow, {k: (v, (0.9, 0.2, 0.2)) for k, v in names.items()}, opacity=0.5)
    for n in (seg, ctx):
        n.CreateClosedSurfaceRepresentation()
    fid = SEARCH / f"{CASE}_embolus_candidates.mrk.json"
    if fid.is_file():
        slicer.util.loadMarkups(str(fid)); log("fiducials loaded")
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=cta, fit=True)
    jump = os.environ.get("CCE_JUMP_RAS")  # "x,y,z" to centre the views on a given point
    if jump:
        x, y, z = [float(v) for v in jump.split(",")]
        slicer.modules.markups.logic().JumpSlicesToLocation(x, y, z, True); log(f"jumped to RAS {x},{y},{z}")
    sid = None if jump else (seg.GetSegmentation().GetSegmentIdBySegmentName("proximal_intraluminal") or seg.GetSegmentation().GetSegmentIdBySegmentName("distal_candidate"))
    if sid:
        c = [0.0, 0.0, 0.0]
        if seg.GetSegmentCenterRAS(sid, c):
            slicer.modules.markups.logic().JumpSlicesToLocation(c[0], c[1], c[2], True)
    log("done")


try:
    main()
except Exception:
    log(traceback.format_exc()); raise
