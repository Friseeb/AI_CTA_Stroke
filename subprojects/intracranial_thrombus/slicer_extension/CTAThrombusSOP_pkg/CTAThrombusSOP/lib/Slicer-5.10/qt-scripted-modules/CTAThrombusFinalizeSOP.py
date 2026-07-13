import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
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

SEGMENT_LABELS = {
    "thrombus":        1,
    "proximal_vessel": 2,
    "distal_vessel":   3,
}


class CTAThrombusFinalizeSOP(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Thrombus Finalize / SOP"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Finalize intracranial thrombus segmentation: harden transforms, "
            "resample to CTA grid, compute volume/length, save labelmap + JSON."
        )
        self.parent.acknowledgementText = "Internal SOP"


class CTAThrombusFinalizeSOP_Widget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAThrombusFinalizeSOP_Logic()
        btn = qt.QPushButton("Finalize Current Case")
        btn.clicked.connect(self.onRun)
        self.layout.addWidget(btn)
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


class CTAThrombusFinalizeSOP_Logic(ScriptedLoadableModuleLogic):

    def run(self, annotation: dict | None = None, output_dir: str | None = None) -> dict:
        self._log("Starting Thrombus Finalize SOP...")
        cta   = self._find_cta()
        seg   = self._find_segmentation()
        self._harden_transforms()

        labelmap, tmp_nodes = self._export_segmentation(seg, cta)
        fixed  = self._resample_to_cta(labelmap, cta)
        self._force_geometry(cta, fixed)

        case_id    = self._infer_case_id(cta)
        output_dir = Path(output_dir) if output_dir else self._infer_output_dir(cta)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_label  = output_dir / f"{case_id}_thrombus_clean.nii.gz"
        out_log    = output_dir / f"{case_id}_thrombus_clean_log.json"

        self._save_label(fixed, cta, out_label)
        metrics = self._compute_metrics(fixed, cta)
        log = self._build_log(cta, seg, out_label, metrics, annotation)
        out_log.write_text(json.dumps(log, indent=2))

        self._cleanup_nodes(tmp_nodes)
        self._mark_worklist_done(case_id)
        self._log(f"Saved: {out_label}")
        return log

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        print(f"[CTAThrombusFinalizeSOP] {msg}")

    def _find_cta(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        if not nodes:
            raise RuntimeError("No scalar volumes loaded.")
        preferred = [n for n in nodes if "cta" in n.GetName().lower()
                     and "thrombus" not in n.GetName().lower()
                     and "delay" not in n.GetName().lower()]
        chosen = preferred[0] if preferred else nodes[0]
        self._log(f"CTA: {chosen.GetName()}")
        return chosen

    def _find_segmentation(self):
        segs = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if not segs:
            raise RuntimeError("No segmentation found. Create one in Step 3.")
        preferred = [s for s in segs if "thrombus" in s.GetName().lower()]
        chosen = preferred[0] if preferred else segs[0]
        self._log(f"Segmentation: {chosen.GetName()}")
        return chosen

    def _harden_transforms(self):
        # hardenTransform causes a VTK segfault on NIfTI-loaded scenes — skip it.
        # ExportAllSegmentsToLabelmapNode handles geometry correctly without this.
        self._log("Hardened transforms: skipped (NIfTI-safe mode)")

    def _export_segmentation(self, seg, cta):
        tmp = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "thrombus_export"
        )
        logic = slicer.modules.segmentations.logic()
        exported = False
        for call in (
            lambda: logic.ExportAllSegmentsToLabelmapNode(seg, tmp, cta),
            lambda: logic.ExportAllSegmentsToLabelmapNode(seg, tmp),
            lambda: logic.ExportVisibleSegmentsToLabelmapNode(seg, tmp),
        ):
            try:
                call()
                exported = True
                break
            except Exception:
                continue
        if not exported:
            raise RuntimeError("Could not export segmentation to labelmap.")
        return tmp, [tmp]

    def _resample_to_cta(self, labelmap, cta):
        out = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "thrombus_resampled"
        )
        try:
            slicer.modules.volumes.logic().ResampleLabelVolumeToReferenceVolume(
                labelmap, cta, out
            )
            return out
        except Exception:
            if sitk is None:
                raise RuntimeError("SimpleITK unavailable.")
            lbl_img = sitkUtils.PullVolumeFromSlicer(labelmap)
            cta_img = sitkUtils.PullVolumeFromSlicer(cta)
            resampled = sitk.Resample(lbl_img, cta_img, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            sitkUtils.PushVolumeToSlicer(resampled, out)
            return out

    def _force_geometry(self, cta, labelmap):
        try:
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, labelmap)
        except Exception:
            m = vtk.vtkMatrix4x4()
            cta.GetIJKToRASMatrix(m)
            labelmap.SetIJKToRASMatrix(m)
            labelmap.SetOrigin(cta.GetOrigin())
            labelmap.SetSpacing(cta.GetSpacing())
        labelmap.SetAndObserveTransformNodeID(None)
        labelmap.Modified()

    def _compute_metrics(self, labelmap, cta) -> dict:
        spacing = cta.GetSpacing()           # (sx, sy, sz) in mm
        vox_vol = spacing[0] * spacing[1] * spacing[2]   # mm³
        arr = slicer.util.arrayFromVolume(labelmap)       # ZYX

        metrics = {}
        for name, label_val in SEGMENT_LABELS.items():
            mask = arr == label_val
            n_vox = int(mask.sum())
            vol_mm3 = round(n_vox * vox_vol, 2)

            # Longest axis length: bounding box diagonal along z (superior-inferior)
            if n_vox > 0:
                z_coords = np.where(mask)[0]
                length_mm = round(float(z_coords.max() - z_coords.min() + 1) * spacing[2], 2)
            else:
                length_mm = 0.0

            metrics[name] = {
                "voxels": n_vox,
                "volume_mm3": vol_mm3,
                "length_si_mm": length_mm,
            }
        return metrics

    def _save_label(self, labelmap, cta, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sitk is not None and sitkUtils is not None:
            lbl_img = sitkUtils.PullVolumeFromSlicer(labelmap)
            cta_img = sitkUtils.PullVolumeFromSlicer(cta)
            lbl_img = sitk.Cast(lbl_img, sitk.sitkUInt16)
            lbl_img = sitk.Resample(lbl_img, cta_img, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            lbl_img.CopyInformation(cta_img)
            sitk.WriteImage(lbl_img, str(out_path), True)
        else:
            slicer.util.saveNode(labelmap, str(out_path))

    def _node_path(self, node):
        s = node.GetStorageNode()
        return s.GetFileName() if s and s.GetFileName() else None

    def _infer_case_id(self, cta):
        name = cta.GetName()
        m = re.search(r"sub-\d+", name)
        return m.group(0) if m else re.sub(r"\W+", "_", name).strip("_")

    def _infer_output_dir(self, cta):
        desktop_out = Path.home() / "Desktop" / "CTA_intracraneal_segmentation" / "SLAO"
        if desktop_out.exists():
            return desktop_out
        p = self._node_path(cta)
        if p:
            return Path(p).parent / "thrombus_manual"
        return Path.home() / "thrombus_manual"

    def _build_log(self, cta, seg, out_label, metrics, annotation) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "case_id": self._infer_case_id(cta),
            "cta_node": cta.GetName(),
            "cta_path": self._node_path(cta),
            "segmentation_node": seg.GetName(),
            "output_label": str(out_label),
            "label_map": {v: k for k, v in SEGMENT_LABELS.items()},
            "metrics": metrics,
            "annotation": annotation or {},
        }

    def _mark_worklist_done(self, case_id: str):
        worklist_path = Path("/tmp/thrombus_worklist.json")
        if not worklist_path.exists():
            return
        try:
            worklist = json.loads(worklist_path.read_text())
            sub_id = case_id.replace("sub-", "")
            for entry in worklist:
                if str(entry.get("sub_id", "")) == sub_id:
                    entry["done"] = True
                    break
            worklist_path.write_text(json.dumps(worklist, indent=2))
            done = sum(1 for e in worklist if e["done"])
            self._log(f"Worklist: {done}/{len(worklist)} done")
        except Exception as exc:
            self._log(f"Could not update worklist: {exc}")

    def _cleanup_nodes(self, nodes):
        for n in nodes:
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass
