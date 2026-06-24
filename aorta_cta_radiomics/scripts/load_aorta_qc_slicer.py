# Aorta-only QC loader for 3D Slicer.
#
# Loads the CTA + key aorta masks for one case, forcing each mask onto the CTA's
# exact IJKToRAS so the slaobids "unexpected scales in sform" / Z-spacing issue
# does not leave masks shifted or flipped relative to the volume.
#
# Run:
#   Slicer --python-script load_aorta_qc_slicer.py            # defaults to sub-547
#   AORTA_QC_CASE=sub-255 Slicer --python-script load_aorta_qc_slicer.py
#
# RESEARCH PROTOTYPE - NOT FOR CLINICAL DIAGNOSIS.
import os
from pathlib import Path

import slicer
import vtk

# Repo root: CTA_STROKE_REPO env override, else inferred from this file's location.
_here = Path(globals().get("__file__", "")).resolve()
REPO_ROOT = Path(os.environ.get("CTA_STROKE_REPO") or
                 (_here.parents[2] if _here.name else Path.cwd()))

# Shared mask-alignment helper. Slicer's embedded Python lacks the editable
# install, so bootstrap cta_common's source dir onto sys.path.
import sys

_CTA_COMMON_SRC = REPO_ROOT / "cta_common" / "src"
if _CTA_COMMON_SRC.is_dir() and str(_CTA_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(_CTA_COMMON_SRC))
from cta_common.slicer_qc import load_mask_aligned  # noqa: E402

LOG_PATH = Path("/tmp/aorta_qc_slicer.log")
CASE_FILE = Path("/tmp/aorta_qc_case.txt")
# Case selection: sidecar file (robust through macOS `open`) > env var > default.
if CASE_FILE.exists():
    CASE = CASE_FILE.read_text().strip()
else:
    CASE = os.environ.get("AORTA_QC_CASE", "sub-547")
MASKS_DIR = REPO_ROOT / "aorta_cta_radiomics/outputs/aorta_batch_run/cases" / CASE / "masks" / CASE

# CTA: prefer a local copy; optionally fall back to a source dir from SLAOBIDS_DIR.
CTA_CANDIDATES = [REPO_ROOT / "data" / f"{CASE}_acq-CTA_ct.nii.gz"]
if os.environ.get("SLAOBIDS_DIR"):
    CTA_CANDIDATES.append(Path(os.environ["SLAOBIDS_DIR"]) / f"{CASE}_acq-CTA_ct.nii.gz")

# (display name, filename suffix after "<case>_", RGB, 2D-fill opacity)
SEG_MASKS = [
    ("Aorta cleaned", "aorta_mask_cleaned.nii.gz", (0.90, 0.10, 0.10), 0.10),
    ("Wall band", "aorta_wall_band.nii.gz", (0.0, 0.75, 0.45), 0.30),
    ("Wall from fat", "aortic_wall_candidate_from_fat_lumen.nii.gz", (0.0, 0.85, 0.55), 0.30),
    ("Lumen (contrast)", "aortic_wall_contrast_lumen_from_centerline_hu.nii.gz", (0.0, 0.85, 1.0), 0.15),
    ("Calcium dyn500", "calcification_aorta_wall_dynamic_seed500HU.nii.gz", (1.0, 0.95, 0.78), 0.75),
    ("Fat 0-2mm", "periaortic_fat_0_2mm.nii.gz", (1.0, 0.95, 0.0), 0.20),
    ("Fat 2-5mm", "periaortic_fat_2_5mm.nii.gz", (1.0, 0.70, 0.0), 0.15),
    ("Ulcer core >=2mm", "wall_lumen_protrusion_outward_ulcer_like_aorta_surface_core_depth_ge_2mm_labels_3d.nii.gz", (0.55, 0.0, 1.0), 0.75),
    ("Wall >4mm TEE", "wall_thickness_gt_4mm_TEE_analogue_labels.nii.gz", (0.95, 0.0, 0.0), 0.75),
]
# Continuous map shown as a foreground colormap, not a segmentation.
VOLUME_MAPS = [
    ("Wall thickness mm", "wall_thickness_mm.nii.gz", 6.0, 3.0),
]

# Multi-value labelmap: one integer per thickness band -> distinct color/name.
WT_BINS_SUFFIX = "wall_thickness_bins_labels.nii.gz"
WT_BINS = [
    ("WT <2mm", (0.15, 0.35, 1.0)),
    ("WT 2-3mm", (0.0, 0.75, 0.35)),
    ("WT 3-4mm", (1.0, 0.85, 0.0)),
    ("WT 4-5mm", (1.0, 0.45, 0.0)),
    ("WT >=5mm", (0.95, 0.0, 0.0)),
]


def log(*parts):
    line = "[aorta-qc] " + " ".join(str(p) for p in parts)
    print(line)
    try:
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def load_cta():
    for cand in CTA_CANDIDATES:
        if cand.exists():
            node = slicer.util.loadVolume(str(cand))
            disp = node.GetDisplayNode()
            if disp:
                disp.AutoWindowLevelOff()
                disp.SetWindow(900)
                disp.SetLevel(250)
            slicer.util.setSliceViewerLayers(background=node, fit=True)
            log("CTA:", cand)
            return node
    log("CTA NOT FOUND, tried:", *[str(c) for c in CTA_CANDIDATES])
    return None


def style_segmentation(seg, color, fill):
    disp = seg.GetDisplayNode()
    if not disp:
        return
    disp.SetVisibility2DFill(True)
    disp.SetVisibility2DOutline(True)
    disp.SetOpacity2DFill(float(fill))
    disp.SetOpacity2DOutline(1.0)
    segmentation = seg.GetSegmentation()
    n = segmentation.GetNumberOfSegments()
    for i in range(n):
        sid = segmentation.GetNthSegmentID(i)
        segmentation.GetSegment(sid).SetColor(*[float(c) for c in color])
        disp.SetSegmentVisibility(sid, i == 0 or n > 1)


def load_bins(path, cta_node, bins):
    """Load a multi-value labelmap, coloring each integer band distinctly."""
    import re

    seg = load_mask_aligned(path, cta_node, "WT Bins")
    if seg is None:
        return None
    disp = seg.GetDisplayNode()
    segmentation = seg.GetSegmentation()
    ids = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]

    def order_key(sid):
        m = re.search(r"(\d+)\s*$", segmentation.GetSegment(sid).GetName() or "")
        return int(m.group(1)) if m else 0

    ids.sort(key=order_key)
    for i, sid in enumerate(ids):
        name, color = bins[i] if i < len(bins) else (f"bin {i + 1}", (0.6, 0.6, 0.6))
        segment = segmentation.GetSegment(sid)
        segment.SetName(name)
        segment.SetColor(*[float(c) for c in color])
        if disp:
            disp.SetSegmentVisibility(sid, True)
            disp.SetSegmentOpacity2DFill(sid, 0.6)
            disp.SetSegmentOpacity2DOutline(sid, 1.0)
    if disp:
        disp.SetVisibility2DFill(True)
        disp.SetVisibility2DOutline(True)
    return seg


def load_volume_map(path, cta_node, name, window, level):
    node = slicer.util.loadVolume(str(path))
    if node is None:
        return None
    if cta_node is not None:
        ijk = vtk.vtkMatrix4x4()
        cta_node.GetIJKToRASMatrix(ijk)
        node.SetIJKToRASMatrix(ijk)
    disp = node.GetDisplayNode()
    if disp:
        disp.AutoWindowLevelOff()
        disp.SetWindow(float(window))
        disp.SetLevel(float(level))
        disp.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
    node.SetDisplayVisibility(False)  # off by default; toggle in Volumes module
    return node


def main():
    try:
        LOG_PATH.write_text("")  # fresh log per run
    except Exception:
        pass
    if not MASKS_DIR.is_dir():
        log("MASKS DIR NOT FOUND:", MASKS_DIR)
        return
    log("case", CASE, "masks", MASKS_DIR)
    cta = load_cta()

    loaded, missing = 0, 0
    for name, suffix, color, fill in SEG_MASKS:
        path = MASKS_DIR / f"{CASE}_{suffix}"
        if not path.exists():
            log("missing:", name, path.name)
            missing += 1
            continue
        seg = load_mask_aligned(path, cta, name)
        if seg:
            style_segmentation(seg, color, fill)
            loaded += 1
            log("loaded:", name)

    for name, suffix, window, level in VOLUME_MAPS:
        path = MASKS_DIR / f"{CASE}_{suffix}"
        if path.exists():
            load_volume_map(path, cta, name, window, level)
            log("loaded map:", name)

    bins_path = MASKS_DIR / f"{CASE}_{WT_BINS_SUFFIX}"
    if bins_path.exists():
        seg = load_bins(bins_path, cta, WT_BINS)
        if seg:
            log("loaded bins:", seg.GetSegmentation().GetNumberOfSegments(), "bands")
    else:
        log("missing bins:", bins_path.name)

    # Re-assert the CTA as the slice background after all overlays are added.
    if cta is not None:
        try:
            slicer.util.setSliceViewerLayers(background=cta, fit=True)
            log("background set:", cta.GetName())
        except Exception as exc:
            log("could not set background:", exc)
    else:
        log("NO CTA -> no background volume")

    log(f"done: {loaded} masks loaded, {missing} missing")


main()
