import json
from pathlib import Path

import qt
import slicer
import vtk

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

from vertebral_review_core import (
    LEFT_CURVE_NAME,
    FORAMEN_PRIOR_COLOR,
    LEFT_LABEL_VALUE,
    LEFT_SEGMENT_COLOR,
    LEFT_SEGMENT_NAME,
    REVIEW_STATUS_OPTIONS,
    RIGHT_CURVE_NAME,
    RIGHT_LABEL_VALUE,
    RIGHT_SEGMENT_COLOR,
    RIGHT_SEGMENT_NAME,
    append_review_csv,
    build_review_log,
    infer_case_id,
    normalize_reviewer_id,
    output_paths,
    validate_label_values,
    validate_review_status,
)


class CTAFinalizeSOP(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Finalize / SOP"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Finalize CTA + bilateral vertebral labels for ITK-SNAP and ML training.\n"
            "Hardens transforms, aligns output to CTA geometry, and writes review logs."
        )
        self.parent.acknowledgementText = "Internal SOP"


class CTAFinalizeSOPWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAFinalizeSOPLogic()

        form = qt.QFormLayout()
        self.ctaSelector = _node_selector(["vtkMRMLScalarVolumeNode"], none_enabled=False)
        form.addRow("CTA volume", self.ctaSelector)

        self.labelSelector = _node_selector(
            ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"],
            none_enabled=False,
        )
        form.addRow("Vertebral segmentation/label", self.labelSelector)

        self.priorSelector = _node_selector(
            ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"],
            none_enabled=True,
        )
        form.addRow("Foramen negative prior", self.priorSelector)

        self.reviewerText = qt.QLineEdit()
        self.reviewerText.setPlaceholderText("reviewer initials or ID")
        form.addRow("Reviewer ID", self.reviewerText)

        self.statusCombo = qt.QComboBox()
        for status in REVIEW_STATUS_OPTIONS:
            self.statusCombo.addItem(status)
        form.addRow("Review status", self.statusCombo)

        self.outputDirText = qt.QLineEdit()
        self.outputDirText.setPlaceholderText("Output folder")
        browse_btn = qt.QPushButton("Browse")
        browse_btn.clicked.connect(self.onBrowseOutputDir)
        out_row = qt.QHBoxLayout()
        out_row.addWidget(self.outputDirText)
        out_row.addWidget(browse_btn)
        form.addRow("Output folder", out_row)
        self.layout.addLayout(form)

        self.runButton = qt.QPushButton("Finalize Current Case")
        self.runButton.toolTip = "Save CTA-aligned bilateral vertebral labelmap and review logs"
        self.runButton.clicked.connect(self.onRun)
        self.layout.addWidget(self.runButton)

        self.logBox = qt.QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(220)
        self.layout.addWidget(self.logBox)
        self.layout.addStretch(1)

    def onBrowseOutputDir(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Output Folder", "")
        if folder:
            self.outputDirText.setText(folder)

    def onRun(self):
        self.logBox.clear()
        try:
            output_dir = _text_value(self.outputDirText).strip() or None
            log = self.logic.run(
                cta_node=self.ctaSelector.currentNode(),
                label_node=self.labelSelector.currentNode(),
                output_dir=output_dir,
                reviewer_id=_text_value(self.reviewerText),
                review_status=_combo_text(self.statusCombo),
                centerline_nodes=_find_centerline_nodes(),
                negative_prior_nodes=_selected_nodes(self.priorSelector),
                save_scene=True,
            )
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append(f"ERROR: {exc}")


class CTAFinalizeSOPLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        cta_node=None,
        label_node=None,
        extra_params: dict | None = None,
        output_dir: str | None = None,
        reviewer_id: str | None = None,
        review_status: str = "in_progress",
        centerline_nodes: list | None = None,
        negative_prior_nodes: list | None = None,
        save_scene: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Finalize one vertebral case.

        Calling with no nodes preserves the original fallback behavior: the CTA
        and label are inferred from loaded Slicer nodes.
        """
        self._log("Starting CTA Finalize SOP...")
        cta = cta_node or self._find_cta()
        label = label_node or self._find_label()
        if cta is None:
            raise RuntimeError("No CTA volume selected.")
        if label is None:
            raise RuntimeError("No vertebral segmentation or label selected.")

        reviewer = normalize_reviewer_id(reviewer_id, required=not dry_run)
        status = validate_review_status(review_status)
        output_dir_path = Path(output_dir) if output_dir else self._infer_output_dir(cta, label)
        case_id = infer_case_id(self._node_path(cta), self._node_path(label), cta.GetName(), label.GetName())
        paths = output_paths(output_dir_path, case_id)
        warnings: list[str] = []
        prior_nodes = negative_prior_nodes if negative_prior_nodes is not None else self._find_negative_prior_nodes()

        if dry_run:
            return build_review_log(
                case_id=case_id,
                reviewer_id=reviewer,
                review_status=status,
                cta_node=cta.GetName(),
                label_node=label.GetName(),
                cta_path=self._node_path(cta),
                label_path=self._node_path(label),
                output_paths=paths,
                params=extra_params,
                warnings=["dry_run: no files written"],
                negative_priors=self._negative_prior_metadata(prior_nodes),
            )

        self._harden_transforms()
        labelmap_node, temp_nodes = self._prepare_labelmap(label, cta, warnings)
        fixed_label = self._resample_to_cta(labelmap_node, cta)
        self._force_geometry(cta, fixed_label)
        warnings.extend(self._label_value_warnings(fixed_label))
        negative_priors = self._negative_prior_report(fixed_label, cta, prior_nodes, warnings, temp_nodes)

        paths.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_label(fixed_label, cta, paths.clean_label)
        centerlines_saved = self._save_centerlines(centerline_nodes or [], paths.centerlines)
        scene_saved = self._save_scene(paths.scene) if save_scene else False

        log = build_review_log(
            case_id=case_id,
            reviewer_id=reviewer,
            review_status=status,
            cta_node=cta.GetName(),
            label_node=label.GetName(),
            cta_path=self._node_path(cta),
            label_path=self._node_path(label),
            output_paths=paths,
            params=extra_params,
            warnings=warnings,
            negative_priors=negative_priors,
            centerlines_saved=centerlines_saved,
            scene_saved=scene_saved,
        )
        paths.log_json.write_text(json.dumps(log, indent=2), encoding="utf-8")
        append_review_csv(paths.review_csv, log)

        self._cleanup_nodes([*temp_nodes, fixed_label], keep=[label, cta])
        self._log(f"Saved: {paths.clean_label}")
        self._log(f"Log: {paths.log_json}")
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
            if "vert" in name or "label" in name or "seg" in name:
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

        label = select_by_keyword("vtkMRMLSegmentationNode", ["vert"])
        if label:
            self._log(f"Label selected (segmentation): {label.GetName()}")
            return label
        label = select_by_keyword("vtkMRMLLabelMapVolumeNode", ["vert"])
        if label:
            self._log(f"Label selected (labelmap): {label.GetName()}")
            return label
        label = select_by_keyword("vtkMRMLScalarVolumeNode", ["vert", "label", "seg"])
        if label:
            self._log(f"Label selected (scalar): {label.GetName()}")
            return label
        segs = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if segs:
            self._log(f"Label fallback (first segmentation): {segs[0].GetName()}")
            return segs[0]
        labelmaps = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        if labelmaps:
            self._log(f"Label fallback (first labelmap): {labelmaps[0].GetName()}")
            return labelmaps[0]
        raise RuntimeError("No label found. Load/select a vertebral segmentation or labelmap.")

    def _harden_transforms(self):
        hardened = 0
        for node in slicer.mrmlScene.GetNodes():
            if hasattr(node, "GetTransformNodeID") and node.GetTransformNodeID():
                slicer.vtkSlicerTransformLogic().hardenTransform(node)
                hardened += 1
        self._log(f"Hardened transforms: {hardened}")

    def _prepare_labelmap(self, label, cta, warnings: list[str]):
        temp_nodes = []
        if label.IsA("vtkMRMLSegmentationNode"):
            return self._segmentation_to_bilateral_labelmap(label, cta, warnings)
        if label.IsA("vtkMRMLLabelMapVolumeNode"):
            labelmap_node = self._clone_labelmap(label, suffix="_work")
            temp_nodes.append(labelmap_node)
            return labelmap_node, temp_nodes
        if label.IsA("vtkMRMLScalarVolumeNode"):
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", label.GetName() + "_labelmap_work"
            )
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, labelmap_node)
            import numpy as np

            src = slicer.util.arrayFromVolume(label)
            slicer.util.updateVolumeFromArray(labelmap_node, src.astype(np.uint16, copy=False))
            temp_nodes.append(labelmap_node)
            return labelmap_node, temp_nodes
        raise RuntimeError("Label node type not supported.")

    def _segmentation_to_bilateral_labelmap(self, segmentation_node, cta, warnings: list[str]):
        import numpy as np

        self._ensure_bilateral_segments(segmentation_node)
        temp_nodes = []
        out = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", segmentation_node.GetName() + "_bilateral_export"
        )
        slicer.modules.volumes.logic().CopyVolumeGeometry(cta, out)
        zero = np.zeros(slicer.util.arrayFromVolume(cta).shape, dtype=np.uint16)
        slicer.util.updateVolumeFromArray(out, zero)
        temp_nodes.append(out)

        for segment_name, value in ((LEFT_SEGMENT_NAME, LEFT_LABEL_VALUE), (RIGHT_SEGMENT_NAME, RIGHT_LABEL_VALUE)):
            segment_id = self._segment_id_by_name(segmentation_node, segment_name)
            if not segment_id:
                warnings.append(f"Missing segment {segment_name}; output label {value} will be empty.")
                continue
            tmp = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", f"{segmentation_node.GetName()}_{segment_name}_tmp"
            )
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, tmp)
            self._export_single_segment(segmentation_node, segment_id, tmp, cta)
            arr = slicer.util.arrayFromVolume(out)
            tmp_arr = slicer.util.arrayFromVolume(tmp)
            arr[tmp_arr > 0] = value
            slicer.util.updateVolumeFromArray(out, arr.astype(np.uint16, copy=False))
            slicer.mrmlScene.RemoveNode(tmp)
        return out, temp_nodes

    def _export_single_segment(self, segmentation_node, segment_id: str, output_node, cta):
        logic = slicer.modules.segmentations.logic()
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)
        calls = (
            lambda: logic.ExportSegmentsToLabelmapNode(segmentation_node, segment_ids, output_node, cta),
            lambda: logic.ExportSegmentsToLabelmapNode(segmentation_node, segment_ids, output_node),
        )
        for call in calls:
            try:
                call()
                return
            except Exception:
                continue
        raise RuntimeError(f"Failed to export segment id {segment_id} to labelmap.")

    def _ensure_bilateral_segments(self, segmentation_node):
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation = segmentation_node.GetSegmentation()
        for name, color in ((LEFT_SEGMENT_NAME, LEFT_SEGMENT_COLOR), (RIGHT_SEGMENT_NAME, RIGHT_SEGMENT_COLOR)):
            segment_id = self._segment_id_by_name(segmentation_node, name)
            if not segment_id:
                segment_id = segmentation.AddEmptySegment(name)
            segment = segmentation.GetSegment(segment_id)
            segment.SetName(name)
            segment.SetColor(*color)

    def _segment_id_by_name(self, segmentation_node, name: str) -> str | None:
        target = name.lower()
        segmentation = segmentation_node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            segment_name = segmentation.GetNthSegment(i).GetName().lower()
            if segment_name == target:
                return segment_id
        aliases = {
            LEFT_SEGMENT_NAME.lower(): {"left", "l", "vertebral_l", "vertebral_left", "vert_l"},
            RIGHT_SEGMENT_NAME.lower(): {"right", "r", "vertebral_r", "vertebral_right", "vert_r"},
        }
        for i in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(i)
            segment_name = segmentation.GetNthSegment(i).GetName().lower().replace(" ", "_")
            if segment_name in aliases.get(target, set()):
                return segment_id
        return None

    def _clone_labelmap(self, src, suffix="_work"):
        import numpy as np

        clone = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", src.GetName() + suffix)
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
        out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", labelmap_node.GetName() + "_resampled")
        try:
            slicer.modules.volumes.logic().ResampleLabelVolumeToReferenceVolume(labelmap_node, cta, out)
            return out
        except Exception:
            if sitk is None or sitkUtils is None:
                raise RuntimeError("SimpleITK not available in Slicer.")
            label_img = sitkUtils.PullVolumeFromSlicer(labelmap_node)
            cta_img = sitkUtils.PullVolumeFromSlicer(cta)
            resampled = sitk.Resample(label_img, cta_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
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
        try:
            storage = node.GetStorageNode()
            if storage and storage.GetFileName():
                return storage.GetFileName()
        except Exception:
            pass
        return None

    def _save_label(self, labelmap_node, cta, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sitk is None or sitkUtils is None:
            slicer.util.saveNode(labelmap_node, str(out_path))
            return
        label_img = sitkUtils.PullVolumeFromSlicer(labelmap_node)
        cta_img = sitkUtils.PullVolumeFromSlicer(cta)
        label_img = sitk.Cast(label_img, sitk.sitkUInt16)
        label_img = sitk.Resample(label_img, cta_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        label_img.CopyInformation(cta_img)
        sitk.WriteImage(label_img, str(out_path), True)

    def _label_value_warnings(self, labelmap_node) -> list[str]:
        import numpy as np

        arr = slicer.util.arrayFromVolume(labelmap_node)
        values = {int(v) for v in np.unique(arr)}
        return validate_label_values(values)

    def _negative_prior_metadata(self, nodes: list | None) -> list[dict]:
        metadata = []
        for node in self._deduplicate_nodes(nodes or []):
            metadata.append(
                {
                    "node": node.GetName(),
                    "path": self._node_path(node) or "",
                    "role": "negative_prior",
                    "overlap_voxels": 0,
                    "overlap_by_label": {str(LEFT_LABEL_VALUE): 0, str(RIGHT_LABEL_VALUE): 0},
                }
            )
        return metadata

    def _negative_prior_report(self, labelmap_node, cta, nodes: list | None, warnings: list[str], temp_nodes: list) -> list[dict]:
        import numpy as np

        reports = []
        if not nodes:
            return reports
        label_arr = slicer.util.arrayFromVolume(labelmap_node)
        label_mask = label_arr > 0
        for node in self._deduplicate_nodes(nodes):
            try:
                prior_label, prior_temp_nodes = self._prior_to_labelmap(node, cta)
                temp_nodes.extend(prior_temp_nodes)
                prior_arr = slicer.util.arrayFromVolume(prior_label)
                prior_mask = prior_arr > 0
                overlap_mask = label_mask & prior_mask
                overlap_by_label = {
                    str(LEFT_LABEL_VALUE): int(((label_arr == LEFT_LABEL_VALUE) & prior_mask).sum()),
                    str(RIGHT_LABEL_VALUE): int(((label_arr == RIGHT_LABEL_VALUE) & prior_mask).sum()),
                }
                overlap_voxels = int(overlap_mask.sum())
                report = {
                    "node": node.GetName(),
                    "path": self._node_path(node) or "",
                    "role": "negative_prior",
                    "prior_voxels": int(prior_mask.sum()),
                    "overlap_voxels": overlap_voxels,
                    "overlap_by_label": overlap_by_label,
                }
                if overlap_voxels:
                    warnings.append(
                        f"Negative prior overlap with {node.GetName()}: {overlap_voxels} voxels "
                        f"(Vert L={overlap_by_label[str(LEFT_LABEL_VALUE)]}, "
                        f"Vert R={overlap_by_label[str(RIGHT_LABEL_VALUE)]})."
                    )
                reports.append(report)
            except Exception as exc:
                warnings.append(f"Could not evaluate negative prior {node.GetName()}: {exc}")
                reports.append(
                    {
                        "node": node.GetName(),
                        "path": self._node_path(node) or "",
                        "role": "negative_prior",
                        "error": str(exc),
                        "overlap_voxels": 0,
                        "overlap_by_label": {str(LEFT_LABEL_VALUE): 0, str(RIGHT_LABEL_VALUE): 0},
                    }
                )
        return reports

    def _deduplicate_nodes(self, nodes: list) -> list:
        seen = set()
        deduped = []
        for node in nodes:
            if node is None:
                continue
            try:
                key = node.GetID() or node.GetName()
            except Exception:
                key = id(node)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped

    def _prior_to_labelmap(self, node, cta):
        import numpy as np

        temp_nodes = []
        if node.IsA("vtkMRMLSegmentationNode"):
            base = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", node.GetName() + "_prior_work")
            slicer.modules.volumes.logic().CopyVolumeGeometry(cta, base)
            temp_nodes.append(base)
            try:
                slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(node, base, cta)
            except Exception:
                zero = np.zeros(slicer.util.arrayFromVolume(cta).shape, dtype=np.uint16)
                slicer.util.updateVolumeFromArray(base, zero)
                segmentation = node.GetSegmentation()
                for i in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(i)
                    tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{node.GetName()}_prior_{i}_tmp")
                    slicer.modules.volumes.logic().CopyVolumeGeometry(cta, tmp)
                    self._export_single_segment(node, segment_id, tmp, cta)
                    arr = slicer.util.arrayFromVolume(base)
                    tmp_arr = slicer.util.arrayFromVolume(tmp)
                    arr[tmp_arr > 0] = 1
                    slicer.util.updateVolumeFromArray(base, arr.astype(np.uint16, copy=False))
                    slicer.mrmlScene.RemoveNode(tmp)
        elif node.IsA("vtkMRMLLabelMapVolumeNode"):
            base = self._clone_labelmap(node, suffix="_prior_work")
            temp_nodes.append(base)
        elif node.IsA("vtkMRMLScalarVolumeNode"):
            base = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", node.GetName() + "_prior_work")
            try:
                slicer.modules.volumes.logic().CopyVolumeGeometry(node, base)
            except Exception:
                slicer.modules.volumes.logic().CopyVolumeGeometry(cta, base)
            arr = slicer.util.arrayFromVolume(node)
            slicer.util.updateVolumeFromArray(base, arr.astype(np.uint16, copy=False))
            temp_nodes.append(base)
        else:
            raise RuntimeError("Unsupported negative-prior node type.")

        resampled = self._resample_to_cta(base, cta)
        self._force_geometry(cta, resampled)
        temp_nodes.append(resampled)
        return resampled, temp_nodes

    def _save_centerlines(self, nodes: list, out_path: Path) -> bool:
        markups = []
        for node in nodes:
            if node is None or not hasattr(node, "GetNumberOfControlPoints"):
                continue
            points = []
            for i in range(node.GetNumberOfControlPoints()):
                pos = [0.0, 0.0, 0.0]
                node.GetNthControlPointPosition(i, pos)
                points.append({"id": str(i + 1), "position": [float(v) for v in pos]})
            if points:
                markups.append(
                    {
                        "type": "Curve",
                        "coordinateSystem": "RAS",
                        "name": node.GetName(),
                        "controlPoints": points,
                    }
                )
        if not markups:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"markups": markups}, indent=2), encoding="utf-8")
        return True

    def _save_scene(self, out_path: Path) -> bool:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return bool(slicer.util.saveScene(str(out_path)))
        except Exception:
            return False

    def _cleanup_nodes(self, nodes, keep=None):
        keep = set(keep or [])
        for n in nodes:
            if n in keep:
                continue
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass

    def _find_negative_prior_nodes(self) -> list:
        nodes = []
        keywords = ("foramen", "foramina", "negative_prior")
        for node_class in ("vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"):
            for node in slicer.util.getNodesByClass(node_class):
                name = node.GetName().lower()
                if any(keyword in name for keyword in keywords):
                    nodes.append(node)
        return self._deduplicate_nodes(nodes)


def _node_selector(node_types, none_enabled=True):
    selector = slicer.qMRMLNodeComboBox()
    selector.nodeTypes = node_types
    selector.selectNodeUponCreation = True
    selector.addEnabled = False
    selector.removeEnabled = False
    selector.noneEnabled = none_enabled
    selector.showHidden = False
    selector.setMRMLScene(slicer.mrmlScene)
    return selector


def _text_value(widget) -> str:
    try:
        value = widget.text
        return value() if callable(value) else value
    except Exception:
        return ""


def _combo_text(widget) -> str:
    try:
        value = widget.currentText
        return value() if callable(value) else value
    except Exception:
        return ""


def _find_centerline_nodes() -> list:
    nodes = []
    for name in (LEFT_CURVE_NAME, RIGHT_CURVE_NAME):
        try:
            node = slicer.util.getNode(name)
            if node:
                nodes.append(node)
        except Exception:
            pass
    return nodes


def _selected_nodes(selector) -> list:
    try:
        node = selector.currentNode()
        return [node] if node is not None else []
    except Exception:
        return []
