import os

import slicer
import vtk

# =========================
# USER SETTINGS
# =========================
CTA_NAME = "sub-547_acq-CTA_ct"        # substring of CTA node name (fallbacks applied if not found)
LABEL_NAME = "sub-547-acq-CTA-ct_vert" # substring of vertebral mask name (fallbacks applied if not found)
OUTPUT_DIR = os.environ.get("VERTEBRAL_MANUAL_OUT_DIR", "<BIDS_ROOT>/derivatives/vertebral_manual")
SCRIPT_VERSION = "v6"
# Always resample to CTA in physical space (robust across patients)
FORCE_RESAMPLE = True
AUTO_FLIP = False
FLIP_X = False
FLIP_Y = False
FLIP_Z = False

# =========================
# HELPERS
# =========================
def findNodeByNameSubstring(substring, nodeClass):
    for n in slicer.util.getNodesByClass(nodeClass):
        if substring in n.GetName():
            return n
    return None

def findFirstNodeByKeyword(nodeClass, keywords):
    for n in slicer.util.getNodesByClass(nodeClass):
        name = n.GetName().lower()
        if any(k in name for k in keywords):
            return n
    return None

def findLabelNode(substring):
    # Try labelmap, then scalar volume, then segmentation
    node = findNodeByNameSubstring(substring, "vtkMRMLLabelMapVolumeNode")
    if node:
        return node
    node = findNodeByNameSubstring(substring, "vtkMRMLScalarVolumeNode")
    if node:
        return node
    node = findNodeByNameSubstring(substring, "vtkMRMLSegmentationNode")
    if node:
        return node
    # Fallback: any node with 'vert' in the name
    node = findFirstNodeByKeyword("vtkMRMLLabelMapVolumeNode", ["vert"])
    if node:
        return node
    node = findFirstNodeByKeyword("vtkMRMLScalarVolumeNode", ["vert"])
    if node:
        return node
    node = findFirstNodeByKeyword("vtkMRMLSegmentationNode", ["vert"])
    if node:
        return node
    raise RuntimeError(f"Node containing '{substring}' not found in labelmap/scalar/segmentation nodes")

def findCtaNode(substring):
    node = findNodeByNameSubstring(substring, "vtkMRMLScalarVolumeNode")
    if node:
        return node
    # Fallback: any scalar volume with 'cta' in the name
    node = findFirstNodeByKeyword("vtkMRMLScalarVolumeNode", ["cta"])
    if node:
        return node
    # If only one scalar volume loaded, use it
    nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if len(nodes) == 1:
        return nodes[0]
    return None

def hardenAllTransforms():
    for node in slicer.mrmlScene.GetNodes():
        if hasattr(node, "GetTransformNodeID") and node.GetTransformNodeID():
            slicer.vtkSlicerTransformLogic().hardenTransform(node)

def flip_labelmap(labelmap_node, flip_x=False, flip_y=False, flip_z=False):
    import numpy as np

    arr = slicer.util.arrayFromVolume(labelmap_node)
    # Slicer array order is [k, j, i] = [z, y, x]
    if flip_z:
        arr = np.flip(arr, axis=0)
    if flip_y:
        arr = np.flip(arr, axis=1)
    if flip_x:
        arr = np.flip(arr, axis=2)
    slicer.util.updateVolumeFromArray(labelmap_node, arr)
    labelmap_node.Modified()

def infer_flip_axes(label_node, ref_node):
    import numpy as np

    m_label = vtk.vtkMatrix4x4()
    m_ref = vtk.vtkMatrix4x4()
    label_node.GetIJKToRASMatrix(m_label)
    ref_node.GetIJKToRASMatrix(m_ref)

    flips = []
    for axis in range(3):
        col_label = np.array([m_label.GetElement(r, axis) for r in range(3)])
        col_ref = np.array([m_ref.GetElement(r, axis) for r in range(3)])
        flips.append(float(np.dot(col_label, col_ref)) < 0.0)
    return flips

def clone_labelmap(src, name_suffix="_work"):
    import numpy as np

    clone = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", src.GetName() + name_suffix
    )
    try:
        slicer.modules.volumes.logic().CopyVolumeGeometry(src, clone)
    except Exception:
        ijk_to_ras = vtk.vtkMatrix4x4()
        src.GetIJKToRASMatrix(ijk_to_ras)
        clone.SetIJKToRASMatrix(ijk_to_ras)
        clone.SetOrigin(src.GetOrigin())
        clone.SetSpacing(src.GetSpacing())
    src_arr = slicer.util.arrayFromVolume(src)
    slicer.util.updateVolumeFromArray(clone, src_arr.astype(np.uint16, copy=False))
    clone.Modified()
    return clone

def _print_geometry(node, label):
    try:
        dims = node.GetImageData().GetDimensions()
    except Exception:
        dims = None
    try:
        spacing = node.GetSpacing()
        origin = node.GetOrigin()
    except Exception:
        spacing = None
        origin = None
    print(f"{label}: dims={dims}, spacing={spacing}, origin={origin}")

def _matrix_to_list(m):
    return [[m.GetElement(r, c) for c in range(4)] for r in range(4)]

def _geom_matches(a, b, tol=1e-5):
    try:
        if a.GetImageData().GetDimensions() != b.GetImageData().GetDimensions():
            return False
    except Exception:
        return False
    if any(abs(a.GetSpacing()[i] - b.GetSpacing()[i]) > tol for i in range(3)):
        return False
    if any(abs(a.GetOrigin()[i] - b.GetOrigin()[i]) > tol for i in range(3)):
        return False
    ma = vtk.vtkMatrix4x4()
    mb = vtk.vtkMatrix4x4()
    a.GetIJKToRASMatrix(ma)
    b.GetIJKToRASMatrix(mb)
    la = _matrix_to_list(ma)
    lb = _matrix_to_list(mb)
    for r in range(4):
        for c in range(4):
            if abs(la[r][c] - lb[r][c]) > tol:
                return False
    return True

def resample_label_to_reference(labelmap_node, reference_node):
    """Resample labelmap to reference geometry (nearest neighbor)."""
    out = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", labelmap_node.GetName() + "_resampled"
    )
    volumes_logic = slicer.modules.volumes.logic()
    try:
        volumes_logic.ResampleLabelVolumeToReferenceVolume(
            labelmap_node, reference_node, out
        )
        return out
    except Exception:
        # Fallback to SimpleITK
        try:
            import SimpleITK as sitk
            import sitkUtils

            label_img = sitkUtils.PullVolumeFromSlicer(labelmap_node)
            ref_img = sitkUtils.PullVolumeFromSlicer(reference_node)
            resampled = sitk.Resample(
                label_img,
                ref_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                label_img.GetPixelID(),
            )
            sitkUtils.PushVolumeToSlicer(resampled, out)
            return out
        except Exception as exc:
            raise RuntimeError(f"Failed to resample labelmap to reference: {exc}") from exc

# =========================
# MAIN
# =========================
cta = findCtaNode(CTA_NAME)
if cta is None:
    raise RuntimeError(f"CTA node containing '{CTA_NAME}' not found")
label = findLabelNode(LABEL_NAME)

print("Found CTA:", cta.GetName())
print("Found label:", label.GetName())
print("fix_slicer_save:", SCRIPT_VERSION)
_print_geometry(cta, "CTA")
_print_geometry(label, "Label (before)")

# 1. HARDEN ALL TRANSFORMS (fixes Center issue)
print("Hardening all transforms...")
hardenAllTransforms()

# 2. PREPARE LABELMAP NODE IN CTA GEOMETRY
if label.IsA("vtkMRMLSegmentationNode"):
    # Export segmentation to labelmap (may be cropped)
    tmp_label = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", label.GetName() + "_export"
    )
    logic = slicer.modules.segmentations.logic()
    # Try common signatures across Slicer versions
    ok = False
    for call in (
        lambda: logic.ExportAllSegmentsToLabelmapNode(label, tmp_label),
        lambda: logic.ExportVisibleSegmentsToLabelmapNode(label, tmp_label),
    ):
        try:
            call()
            ok = True
            break
        except Exception:
            continue
    if not ok:
        raise RuntimeError("Failed to export segmentation to labelmap.")
    labelmap_node = clone_labelmap(tmp_label, name_suffix="_labelmap_work")
elif label.IsA("vtkMRMLLabelMapVolumeNode"):
    labelmap_node = clone_labelmap(label, name_suffix="_labelmap_work")
elif label.IsA("vtkMRMLScalarVolumeNode"):
    # Convert scalar volume to labelmap (manual, version-safe)
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", label.GetName() + "_labelmap_work"
    )
    try:
        slicer.modules.volumes.logic().CopyVolumeGeometry(cta, labelmap_node)
    except Exception:
        # Fallback: copy IJK->RAS and spacing directly
        ijk_to_ras = vtk.vtkMatrix4x4()
        cta.GetIJKToRASMatrix(ijk_to_ras)
        labelmap_node.SetIJKToRASMatrix(ijk_to_ras)
        labelmap_node.SetOrigin(cta.GetOrigin())
        labelmap_node.SetSpacing(cta.GetSpacing())
    import numpy as np

    src = slicer.util.arrayFromVolume(label)
    slicer.util.updateVolumeFromArray(labelmap_node, src.astype(np.uint16, copy=False))
else:
    raise RuntimeError("Label node is not a labelmap, scalar volume, or segmentation.")

# Optional flip before resampling/saving
if AUTO_FLIP:
    fx, fy, fz = infer_flip_axes(labelmap_node, cta)
    if fx or fy or fz:
        print(f"Auto flip from header: x={fx}, y={fy}, z={fz}")
        flip_labelmap(labelmap_node, flip_x=fx, flip_y=fy, flip_z=fz)
elif FLIP_X or FLIP_Y or FLIP_Z:
    print(f"Manual flip: x={FLIP_X}, y={FLIP_Y}, z={FLIP_Z}")
    flip_labelmap(labelmap_node, flip_x=FLIP_X, flip_y=FLIP_Y, flip_z=FLIP_Z)

# Resample to CTA geometry (force for robustness)
if FORCE_RESAMPLE:
    fixedLabel = resample_label_to_reference(labelmap_node, cta)
    _print_geometry(fixedLabel, "Label (resampled)")
else:
    if _geom_matches(labelmap_node, cta):
        fixedLabel = labelmap_node
        print("Label geometry matches CTA; skipping resample.")
        _print_geometry(fixedLabel, "Label (matched)")
    else:
        fixedLabel = resample_label_to_reference(labelmap_node, cta)
        _print_geometry(fixedLabel, "Label (resampled)")

# Force header geometry to match CTA (avoids flips in some viewers)
try:
    slicer.modules.volumes.logic().CopyVolumeGeometry(cta, fixedLabel)
except Exception:
    ijk_to_ras = vtk.vtkMatrix4x4()
    cta.GetIJKToRASMatrix(ijk_to_ras)
    fixedLabel.SetIJKToRASMatrix(ijk_to_ras)
    fixedLabel.SetOrigin(cta.GetOrigin())
    fixedLabel.SetSpacing(cta.GetSpacing())
fixedLabel.SetAndObserveTransformNodeID(None)
fixedLabel.Modified()

# 5. SAVE CLEAN FILES
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _case_prefix(cta_name: str) -> str:
    # Expect names like sub-547_acq-CTA_ct or sub-547-acq-CTA-ct
    base = cta_name.split("_")[0]
    base = base.split("-acq-CTA")[0] if "-acq-CTA" in base else base
    return base

case_id = _case_prefix(cta.GetName())
label_out = os.path.join(OUTPUT_DIR, f"{case_id}_vert_clean.nii.gz")

# Prefer SimpleITK write to avoid orientation issues in external viewers (ITK-Snap)
saved = False
try:
    import SimpleITK as sitk
    import sitkUtils

    label_img = sitkUtils.PullVolumeFromSlicer(fixedLabel)
    cta_img = sitkUtils.PullVolumeFromSlicer(cta)
    label_img = sitk.Cast(label_img, sitk.sitkUInt16)
    # Resample in physical space to CTA (robust orientation)
    label_img = sitk.Resample(
        label_img,
        cta_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    label_img.CopyInformation(cta_img)
    sitk.WriteImage(label_img, label_out, True)
    saved = True
    print("Saved label (SimpleITK):", label_out)
except Exception as exc:
    print("SimpleITK save failed, falling back to slicer.util.saveNode:", exc)

if not saved:
    slicer.util.saveNode(fixedLabel, label_out)
    print("Saved label (Slicer):", label_out)

print("DONE")
print("Saved label:", label_out)
