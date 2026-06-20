"""3D Slicer scene helpers shared by the QC loaders and the LAA module.

``slicer``/``vtk`` are imported lazily inside the functions so this module is
importable outside Slicer (for unit tests) and only touches the Slicer runtime
when actually called from within Slicer.
"""

from __future__ import annotations


def load_mask_aligned(path, cta_node, name):
    """Load a labelmap and force the CTA node's exact ``IJKToRAS`` geometry.

    Works around Slicer's "unexpected scales in sform" handling (the slaobids
    Z-flip issue): a separately loaded labelmap can land on a different transform
    than the CTA volume. Since the mask shares the CTA voxel grid, copying the
    CTA's ``IJKToRAS`` guarantees scene alignment.

    Returns a ``vtkMRMLSegmentationNode`` (via ``loadSegmentation`` when no CTA
    reference is given), or ``None`` if the labelmap failed to load.
    """
    import slicer
    import vtk

    if cta_node is None:
        return slicer.util.loadSegmentation(str(path))
    lm = slicer.util.loadVolume(str(path), properties={"labelmap": True})
    if lm is None:
        return None
    ijk = vtk.vtkMatrix4x4()
    cta_node.GetIJKToRASMatrix(ijk)
    lm.SetIJKToRASMatrix(ijk)
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    seg.CreateDefaultDisplayNodes()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(cta_node)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, seg)
    slicer.mrmlScene.RemoveNode(lm)
    return seg


__all__ = ["load_mask_aligned"]
