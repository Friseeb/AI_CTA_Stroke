"""
Startup script for Slicer: loads the next pending thrombus annotation case.
Usage:
    Slicer --python-script load_next_thrombus_case.py
    Slicer --python-script load_next_thrombus_case.py -- --sub 203
"""
import json, sys, subprocess, math
from pathlib import Path

WORKLIST_PATH  = Path("/tmp/thrombus_worklist.json")
THROMBUS_OUT   = Path("/home/fridmans/Desktop/CTA_intracraneal_segmentation/SLAO")
SLAODICOM_BASE = Path("/media/fridmans/Research13T/datasets/SLAODICOM")
PERF_TMP_BASE  = Path("/tmp/perf_cache")

PERF_SERIES = ["Tmax Axial Avg 5.0", "rCBF Axial Avg 5.0", "rCBV Axial Avg 5.0",
               "MTT Axial Avg 5.0", "TTP Axial Avg 5.0"]

target_sub = None
args = sys.argv
if "--sub" in args:
    idx = args.index("--sub")
    if idx + 1 < len(args):
        target_sub = args[idx + 1]

worklist = json.loads(WORKLIST_PATH.read_text())

if target_sub:
    case = next((w for w in worklist if w["sub_id"] == target_sub), None)
    if case is None:
        print(f"[ERROR] sub-{target_sub} not found in worklist")
        sys.exit(1)
else:
    pending = [w for w in worklist if not w["done"] and not w.get("skip")]
    if not pending:
        print("[DONE] All cases annotated!")
        sys.exit(0)
    case = pending[0]

sid = case["sub_id"]
print(f"\n{'='*50}")
print(f"Loading: sub-{sid}  |  NIHSS={case['nihss']}  |  {case['occ_class']}  |  {case['lvo_branch']}")
print(f"Remaining: {sum(1 for w in worklist if not w['done'] and not w.get('skip'))} / {len(worklist)}")
print(f"{'='*50}\n")


def find_and_convert_perfusion(sid):
    """Find perfusion DICOM series and convert to NIfTI with dcm2niix."""
    out_dir = PERF_TMP_BASE / f"sub-{sid}"
    if out_dir.exists() and any(out_dir.glob("*Tmax*.nii.gz")):
        print(f"[PERF] Using cached perfusion for sub-{sid}")
        return out_dir

    dicom_sid = SLAODICOM_BASE / sid
    if not dicom_sid.exists():
        print(f"[PERF] No DICOM dir for sub-{sid}")
        return None

    import pydicom
    from collections import defaultdict
    series_folders = defaultdict(set)
    try:
        for f in sorted(dicom_sid.rglob("*")):
            if not f.is_file() or f.stat().st_size < 1000:
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                desc = str(getattr(ds, "SeriesDescription", "")).strip()
                if desc in PERF_SERIES:
                    series_folders[desc].add(f.parent)
            except Exception:
                pass
    except Exception as e:
        print(f"[PERF] Scan error: {e}")
        return None

    if not series_folders:
        print(f"[PERF] No perfusion series found for sub-{sid}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for desc, folders in series_folders.items():
        folder = list(folders)[0]
        short = desc.replace(" ", "_").replace(".", "")
        result = subprocess.run(
            ["dcm2niix", "-z", "y", "-f", f"sub-{sid}_{short}", "-o", str(out_dir), str(folder)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[PERF] Converted {desc}")
        else:
            print(f"[PERF] Failed {desc}: {result.stderr[:100]}")

    return out_dir


def estimate_head_rotation(volume_node):
    """
    Estimates IS-axis rotation needed to align the brain with standard orientation.
    Uses skull PCA on mid-brain axial slices.
    Returns rotation angle in degrees (positive = CCW when viewed from above).
    """
    import numpy as np
    arr = slicer.util.arrayFromVolume(volume_node)  # (z, y, x)
    nz = arr.shape[0]
    # Use upper 70-90% of z-range to land in brain regardless of coverage
    # (DAYLIGHT: head-only ~500 slices; SLAO: head+neck+chest ~1793 slices)
    z0, z1 = int(nz * 0.70), int(nz * 0.90)
    slab = arr[z0:z1]

    angles = []
    for z in range(0, slab.shape[0], 3):
        skull = slab[z] > 200
        pts = np.argwhere(skull)
        if len(pts) < 300:
            continue
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, -1]  # principal axis (row=y, col=x)
        angle = np.degrees(np.arctan2(main[0], main[1]))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        angles.append(angle)

    if not angles:
        return 0.0

    # The principal axis is at `median_angle` from L-R.
    # Standard head: long axis (A-P) is at ~90°, so correction = median_angle - 90
    # gives how much the head is rotated away from standard.
    # We negate to get the view rotation needed.
    median_angle = float(np.median(angles))
    correction = -(median_angle - 90.0)
    print(f"[ROT] Skull PCA angle={median_angle:.1f}°  →  IS correction={correction:.1f}°")
    return correction


def apply_is_rotation(lm, angle_deg):
    """Rotates the Red (axial) slice view around the IS axis by angle_deg."""
    import vtk
    red_node = lm.sliceWidget("Red").mrmlSliceNode()
    red_node.SetOrientationToAxial()

    # Build rotation around S axis (Z in RAS)
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Standard axial T=[1,0,0], rotate by angle_deg
    tx = cos_a;  ty = -sin_a; tz = 0.0
    # Normal stays [0,0,1]
    nx = 0.0;   ny = 0.0;   nz = 1.0
    # P = N × T
    px = ny*tz - nz*ty
    py = nz*tx - nx*tz
    pz = nx*ty - ny*tx

    # Get current origin from SliceToRAS
    m = vtk.vtkMatrix4x4()
    red_node.GetSliceToRAS(m)
    ox = m.GetElement(0, 3)
    oy = m.GetElement(1, 3)
    oz = m.GetElement(2, 3)

    new_m = vtk.vtkMatrix4x4()
    new_m.SetElement(0, 0, tx); new_m.SetElement(1, 0, ty); new_m.SetElement(2, 0, tz)
    new_m.SetElement(0, 1, px); new_m.SetElement(1, 1, py); new_m.SetElement(2, 1, pz)
    new_m.SetElement(0, 2, nx); new_m.SetElement(1, 2, ny); new_m.SetElement(2, 2, nz)
    new_m.SetElement(0, 3, ox); new_m.SetElement(1, 3, oy); new_m.SetElement(2, 3, oz)
    new_m.SetElement(3, 3, 1.0)

    red_node.SetSliceToRAS(new_m)
    red_node.UpdateMatrices()


import slicer

perf_dir = None  # perfusion not needed for thrombus annotation

loaded = {}

if case["arterial"]:
    node = slicer.util.loadVolume(case["arterial"])
    node.SetName(f"sub-{sid}_CTA_arterial")
    loaded["arterial"] = node

for key, label in [("delay1", "CTA_1stdelay"), ("delay2", "CTA_2nddelay")]:
    if case.get(key):
        node = slicer.util.loadVolume(case[key])
        node.SetName(f"sub-{sid}_{label}")
        loaded[key] = node

perf_nodes = {}
if perf_dir:
    for name_key, pattern in [("Tmax", "*Tmax*"), ("rCBF", "*rCBF*"),
                               ("rCBV", "*rCBV*"), ("MTT", "*MTT*"), ("TTP", "*TTP*")]:
        matches = list(perf_dir.glob(pattern))
        nii_files = [m for m in matches if m.suffix == ".gz" and "a.nii" not in m.name]
        if not nii_files:
            nii_files = matches[:1]
        if nii_files:
            n = slicer.util.loadVolume(str(nii_files[0]))
            n.SetName(f"perf_{name_key}")
            perf_nodes[name_key] = n

lm = slicer.app.layoutManager()


def setup_views():
    art = loaded.get("arterial")
    if not art:
        return

    bounds = [0.0] * 6
    art.GetRASBounds(bounds)
    brain_z = bounds[4] + 0.70 * (bounds[5] - bounds[4])

    # All views: CTA arterial
    for vname in ["Red", "Green", "Yellow"]:
        sl = lm.sliceWidget(vname).sliceLogic()
        sl.GetSliceCompositeNode().SetBackgroundVolumeID(art.GetID())
        sl.SetSliceOffset(brain_z)
        sl.FitSliceToAll()

    # IS rotation disabled — let Slicer show default orientation


import qt
qt.QTimer.singleShot(2000, setup_views)

try:
    slicer.util.mainWindow().moduleSelector().selectModule("CTAThrombusDemarcation")
    widget = slicer.modules.CTAThrombusDemarcationWidget
    widget.outputDirText.setText(str(THROMBUS_OUT))
    if loaded.get("arterial"):
        widget.ctaSelector.setCurrentNode(loaded["arterial"])
    if loaded.get("delay1"):
        widget.delay1Selector.setCurrentNode(loaded["delay1"])
    if loaded.get("delay2"):
        widget.delay2Selector.setCurrentNode(loaded["delay2"])
except Exception as e:
    print(f"[WARN] Module not available: {e}")

try:
    pending_n = sum(1 for w in worklist if not w['done'] and not w.get('skip'))
    perf_status = f"Perf: {', '.join(perf_nodes.keys()) if perf_nodes else 'no data'}"
    slicer.util.mainWindow().statusBar().showMessage(
        f"sub-{sid} | NIHSS {case['nihss']} | {case['occ_class']} | "
        f"{case['lvo_branch'] or '—'} | {pending_n}/{len(worklist)} pending | {perf_status}",
        30000
    )
except Exception:
    pass
