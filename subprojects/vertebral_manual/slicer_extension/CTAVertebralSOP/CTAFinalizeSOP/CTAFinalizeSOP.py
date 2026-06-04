import json
import os
import re
from datetime import datetime
from pathlib import Path

import slicer
import vtk
import qt

try:
    import SimpleITK as sitk
    import sitkUtils
except Exception:
    sitk = None
    sitkUtils = None

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


class CTAFinalizeSOP(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Finalize / SOP"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["Sebastian Fridman (NYU)"]
        self.parent.helpText = (
            "Finalize CTA + vertebral labelmaps for ITK-SNAP and ML training.\n"
            "One-click: harden transforms, fix cropped labels, align to CTA grid."
        )
        self.parent.acknowledgementText = "NYU Langone CTA Stroke Project"


class CTAFinalizeSOPWidget(ScriptedLoadableModuleWidget):

    def resourcePath(self, filename):
        try:
            return super().resourcePath(filename)
        except AttributeError:
            return os.path.join(os.path.dirname(__file__), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAFinalizeSOPLogic()

        self.runButton = qt.QPushButton("Finalize Current Case")
        self.runButton.toolTip = "Find CTA + vertebral label, fix and save clean labelmap"
        self.runButton.clicked.connect(self.onRun)
        self.layout.addWidget(self.runButton)

        self.logBox = qt.QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(200)
        self.layout.addWidget(self.logBox)

        self.layout.addStretch(1)

    def onRun(self):
        self.logBox.clear()
        try:
            log = self.logic.run()
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append(f"ERROR: {exc}")


class CTAFinalizeSOPLogic(ScriptedLoadableModuleLogic):
    def run(self, extra_params: dict | None = None, output_dir: str | None = None) -> dict:
        self._log("Starting CTA Finalize SOP...")
        cta = self._find_cta()
        label = self._find_label()
        self._harden_transforms()

        labelmap_node, temp_nodes = self._prepare_labelmap(label, cta)
        fixed_label = self._resample_to_cta(labelmap_node, cta)
        self._force_geometry(cta, fixed_label)

        output_dir = Path(output_dir) if output_dir else self._infer_output_dir(cta, label)
        case_id = self._infer_case_id(cta, label)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_label = output_dir / f"{case_id}_vert_clean.nii.gz"
        out_log = output_dir / f"{case_id}_vert_clean_log.json"

        self._save_label(fixed_label, cta, out_label)
        log = self._build_log(cta, label, out_label, extra_params=extra_params)
        out_log.write_text(json.dumps(log, indent=2))

        self._cleanup_nodes(temp_nodes, keep=[label, cta])
        self._log(f"Saved: {out_label}")
        self._log(f"Log: {out_log}")
        return log

    def _log(self, msg: str):
        print(f"[CTAFinalizeSOP] {msg}")

    def _find_cta(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if not nodes:
            raise RuntimeError("No scalar volumes loaded.")
        candidates = []
        for n in nodes:
            name = n.GetName().lower()
            if "vert" in name or "label" in name:
                continue
            candidates.append(n)
        preferred = [n for n in candidates if "cta" in n.GetName().lower()]
        chosen = preferred[0] if preferred else candidates[0] if candidates else nodes[0]
        self._log(f"CTA selected: {chosen.GetName()}")
        return chosen

    def _find_label(self):
        def select_by_keyword(node_class, keywords):
            for n in slicer.util.getNodesByClass(node_class):
                name = n.GetName().lower()
                if any(k in name for k in keywords):
                    return n
            return None

        # Prefer explicit "vert" labels
        label = select_by_keyword("vtkMRMLLabelMapVolumeNode", ["vert"])
        if label:
            self._log(f"Label selected (labelmap): {label.GetName()}")
            return label
        label = select_by_keyword("vtkMRMLSegmentationNode", ["vert"])
        if label:
            self._log(f"Label selected (segmentation): {label.GetName()}")
            return label
        label = select_by_keyword("vtkMRMLScalarVolumeNode", ["vert", "label", "seg"])
        if label:
            self._log(f"Label selected (scalar): {label.GetName()}")
            return label

        # Fallback: any labelmap / segmentation / scalar that is not the CTA
        labelmaps = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        if labelmaps:
            self._log(f"Label fallback (first labelmap): {labelmaps[0].GetName()}")
            return labelmaps[0]
        segs = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if segs:
            self._log(f"Label fallback (first segmentation): {segs[0].GetName()}")
            return segs[0]
        scalars = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        # exclude obvious CTA if more than one scalar is loaded
        if len(scalars) > 1:
            for n in scalars:
                name = n.GetName().lower()
                if "cta" in name:
                    continue
                self._log(f"Label fallback (scalar): {n.GetName()}")
                return n
        raise RuntimeError("No label found. Load a labelmap/segmentation or include 'vert' in its name.")

    def _harden_transforms(self):
        hardened = 0
        for node in slicer.mrmlScene.GetNodes():
            if hasattr(node, "GetTransformNodeID") and node.GetTransformNodeID():
                slicer.vtkSlicerTransformLogic().hardenTransform(node)
                hardened += 1
        self._log(f"Hardened transforms: {hardened}")

    def _prepare_labelmap(self, label, cta):
        temp_nodes = []
        if label.IsA("vtkMRMLSegmentationNode"):
            tmp = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", label.GetName() + "_export"
            )
            logic = slicer.modules.segmentations.logic()
            ok = False
            for call in (
                lambda: logic.ExportAllSegmentsToLabelmapNode(label, tmp),
                lambda: logic.ExportVisibleSegmentsToLabelmapNode(label, tmp),
            ):
                try:
                    call()
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                raise RuntimeError("Failed to export segmentation to labelmap.")
            temp_nodes.append(tmp)
            labelmap_node = tmp
        elif label.IsA("vtkMRMLLabelMapVolumeNode"):
            labelmap_node = self._clone_labelmap(label, suffix="_work")
            temp_nodes.append(labelmap_node)
        elif label.IsA("vtkMRMLScalarVolumeNode"):
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", label.GetName() + "_labelmap_work"
            )
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, labelmap_node)
            import numpy as np

            src = slicer.util.arrayFromVolume(label)
            slicer.util.updateVolumeFromArray(labelmap_node, src.astype(np.uint16, copy=False))
            temp_nodes.append(labelmap_node)
        else:
            raise RuntimeError("Label node type not supported.")
        return labelmap_node, temp_nodes

    def _clone_labelmap(self, src, suffix="_work"):
        import numpy as np

        clone = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", src.GetName() + suffix
        )
        try:
            slicer.modules.volumes.logic().CopyVolumeGeometry(src, clone)
        except Exception:
            ijk_to_ras = vtk.vtkMatrix4x4()
            src.GetIJKToRASMatrix(ijk_to_ras)
            clone.SetIJKToRASMatrix(ijk_to_ras)
            clone.SetOrigin(src.GetOrigin())
            clone.SetSpacing(src.GetSpacing())
        arr = slicer.util.arrayFromVolume(src)
        slicer.util.updateVolumeFromArray(clone, arr.astype(np.uint16, copy=False))
        clone.Modified()
        return clone

    def _resample_to_cta(self, labelmap_node, cta):
        out = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", labelmap_node.GetName() + "_resampled"
        )
        try:
            slicer.modules.volumes.logic().ResampleLabelVolumeToReferenceVolume(
                labelmap_node, cta, out
            )
            return out
        except Exception:
            if sitk is None or sitkUtils is None:
                raise RuntimeError("SimpleITK not available in Slicer.")
            label_img = sitkUtils.PullVolumeFromSlicer(labelmap_node)
            cta_img = sitkUtils.PullVolumeFromSlicer(cta)
            resampled = sitk.Resample(
                label_img,
                cta_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt16,
            )
            sitkUtils.PushVolumeToSlicer(resampled, out)
            return out

    def _force_geometry(self, cta, labelmap_node):
        try:
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, labelmap_node)
        except Exception:
            ijk_to_ras = vtk.vtkMatrix4x4()
            cta.GetIJKToRASMatrix(ijk_to_ras)
            labelmap_node.SetIJKToRASMatrix(ijk_to_ras)
            labelmap_node.SetOrigin(cta.GetOrigin())
            labelmap_node.SetSpacing(cta.GetSpacing())
        labelmap_node.SetAndObserveTransformNodeID(None)
        labelmap_node.Modified()

    def _infer_output_dir(self, cta, label):
        path = self._node_path(label) or self._node_path(cta)
        if path:
            return Path(path).parent
        return Path.home() / "vertebral_manual"

    def _node_path(self, node):
        storage = node.GetStorageNode()
        if storage and storage.GetFileName():
            return storage.GetFileName()
        return None

    def _infer_case_id(self, cta, label):
        name = cta.GetName() or label.GetName()
        match = re.search(r"sub-\d+", name)
        if match:
            return match.group(0)
        return re.sub(r"\W+", "_", name).strip("_")

    def _save_label(self, labelmap_node, cta, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sitk is None or sitkUtils is None:
            slicer.util.saveNode(labelmap_node, str(out_path))
            return
        label_img = sitkUtils.PullVolumeFromSlicer(labelmap_node)
        cta_img = sitkUtils.PullVolumeFromSlicer(cta)
        label_img = sitk.Cast(label_img, sitk.sitkUInt16)
        label_img = sitk.Resample(
            label_img,
            cta_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt16,
        )
        label_img.CopyInformation(cta_img)
        sitk.WriteImage(label_img, str(out_path), True)

    def _build_log(self, cta, label, out_label, extra_params=None):
        return {
            "timestamp": datetime.now().isoformat(),
            "cta_node": cta.GetName(),
            "label_node": label.GetName(),
            "cta_path": self._node_path(cta),
            "label_path": self._node_path(label),
            "output_label": str(out_label),
            "output_dir": str(Path(out_label).parent),
            "params": extra_params or {},
        }

    def _cleanup_nodes(self, nodes, keep=None):
        keep = set(keep or [])
        for n in nodes:
            if n in keep:
                continue
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass
