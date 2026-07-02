import json
import os
from pathlib import Path
import traceback

import qt
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

from vertebral_review_core import (
    FORAMEN_PRIOR_COLOR,
    FORAMEN_PRIOR_NAME,
    LEFT_CURVE_NAME,
    LEFT_SEGMENT_COLOR,
    LEFT_SEGMENT_NAME,
    REVIEW_STATUS_OPTIONS,
    RIGHT_CURVE_NAME,
    RIGHT_SEGMENT_COLOR,
    RIGHT_SEGMENT_NAME,
    append_queue_status,
    first_pending_index,
    latest_queue_status_by_case,
    normalize_reviewer_id,
    queue_status_path,
    queue_status_row,
    read_manifest,
    validate_review_status,
)

_WIZARD_DIALOG = None
_WIZARD_WIDGET = None


class CTAVertebralWizard(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Vertebral Wizard"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Guided bilateral vertebral artery manual review.\n"
            "Load CTA, create left/right curves and segments, then finalize clean outputs."
        )
        self.parent.acknowledgementText = "Internal SOP"
        self.parent.dependencies = ["CTAFinalizeSOP"]


class CTAVertebralWizardWidget(ScriptedLoadableModuleWidget):
    def resourcePath(self, filename):
        """Override to handle standalone dialog usage."""
        try:
            return super().resourcePath(filename)
        except (AttributeError, NameError):
            return os.path.join(os.path.expanduser("~"), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAVertebralWizardLogic()
        self.negativePriorNodes = []
        self.caseNodes = []
        self.queueRows = []
        self.queueIndex = -1
        self.queueManifestPath = None
        self.queueStatusPath = None
        self.currentCaseRow = None

        self.layout.addWidget(self._case_group())
        self.layout.addWidget(self._queue_group())
        self.layout.addWidget(self._curves_group())
        self.layout.addWidget(self._segmentation_group())
        self.layout.addWidget(self._negative_prior_group())
        self.layout.addWidget(self._finalize_group())
        self.layout.addStretch(1)

    def _case_group(self):
        group = qt.QGroupBox("Case")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()

        self.ctaSelector = _node_selector(["vtkMRMLScalarVolumeNode"], none_enabled=False)
        form.addRow("CTA volume", self.ctaSelector)

        load_btn = qt.QPushButton("Load CTA (.nii.gz)")
        load_btn.clicked.connect(self.onLoadCTA)
        form.addRow("", load_btn)

        self.outputDirText = qt.QLineEdit()
        self.outputDirText.setPlaceholderText("Output folder")
        self.outputDirText.setText(_guess_output_dir())
        browse_btn = qt.QPushButton("Browse")
        browse_btn.clicked.connect(self.onBrowseOutputDir)
        out_row = qt.QHBoxLayout()
        out_row.addWidget(self.outputDirText)
        out_row.addWidget(browse_btn)
        form.addRow("Output folder", out_row)

        self.reviewerText = qt.QLineEdit()
        self.reviewerText.setPlaceholderText("reviewer initials or ID")
        form.addRow("Reviewer ID", self.reviewerText)

        self.statusCombo = qt.QComboBox()
        for status in REVIEW_STATUS_OPTIONS:
            self.statusCombo.addItem(status)
        form.addRow("Review status", self.statusCombo)
        layout.addLayout(form)
        return group

    def _queue_group(self):
        group = qt.QGroupBox("Queue")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()
        self.manifestPathText = qt.QLineEdit()
        self.manifestPathText.setPlaceholderText("Manifest CSV with case_id, cta_path, label_path, foramen_prior_path")
        browse_btn = qt.QPushButton("Browse")
        browse_btn.clicked.connect(self.onBrowseManifest)
        manifest_row = qt.QHBoxLayout()
        manifest_row.addWidget(self.manifestPathText)
        manifest_row.addWidget(browse_btn)
        form.addRow("Manifest", manifest_row)
        layout.addLayout(form)

        btn_row = qt.QHBoxLayout()
        load_manifest_btn = qt.QPushButton("Load Manifest")
        load_manifest_btn.clicked.connect(self.onLoadManifest)
        load_case_btn = qt.QPushButton("Load Case")
        load_case_btn.clicked.connect(self.onLoadCurrentCase)
        previous_btn = qt.QPushButton("Previous")
        previous_btn.clicked.connect(self.onQueuePrevious)
        next_btn = qt.QPushButton("Skip / Next")
        next_btn.clicked.connect(self.onQueueNext)
        for button in (load_manifest_btn, load_case_btn, previous_btn, next_btn):
            btn_row.addWidget(button)
        layout.addLayout(btn_row)

        self.queueLabel = qt.QLabel("No queue loaded")
        layout.addWidget(self.queueLabel)
        return group

    def _curves_group(self):
        group = qt.QGroupBox("Curves")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(
            qt.QLabel(
                "Create or select bilateral vertebral centerline curves. "
                "Curves are exported if they contain control points."
            )
        )
        btn = qt.QPushButton("Create/Select L/R Centerline Curves")
        btn.clicked.connect(self.onCreateCurves)
        layout.addWidget(btn)
        return group

    def _segmentation_group(self):
        group = qt.QGroupBox("Segmentation")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()
        self.vertebralSelector = _node_selector(
            ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"],
            none_enabled=True,
        )
        form.addRow("Vertebral segmentation/label", self.vertebralSelector)
        layout.addLayout(form)

        load_btn = qt.QPushButton("Load Vertebral Label (.nii.gz)")
        load_btn.clicked.connect(self.onLoadVertebralLabel)
        layout.addWidget(load_btn)

        edit_btn = qt.QPushButton("Create/Edit Bilateral Segmentation")
        edit_btn.clicked.connect(self.onOpenSegmentEditor)
        layout.addWidget(edit_btn)
        layout.addWidget(qt.QLabel("Required segment contract: label 1 = Vert L, label 2 = Vert R."))
        return group

    def _negative_prior_group(self):
        group = qt.QGroupBox("Negative Priors")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(
            qt.QLabel(
                "Optional vertebral foramen masks are loaded as context only. "
                "Finalize logs overlap with Vert L/R but does not erase labels."
            )
        )
        form = qt.QFormLayout()
        self.foramenPriorSelector = _node_selector(
            ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"],
            none_enabled=True,
        )
        form.addRow("Foramen prior", self.foramenPriorSelector)
        layout.addLayout(form)

        load_btn = qt.QPushButton("Load Foramen Prior (.nii.gz)")
        load_btn.clicked.connect(self.onLoadForamenPrior)
        layout.addWidget(load_btn)

        add_btn = qt.QPushButton("Use Selected Foramen Prior")
        add_btn.clicked.connect(self.onAddSelectedForamenPrior)
        layout.addWidget(add_btn)

        self.priorListText = qt.QTextEdit()
        self.priorListText.setReadOnly(True)
        self.priorListText.setMaximumHeight(70)
        layout.addWidget(self.priorListText)
        return group

    def _finalize_group(self):
        group = qt.QGroupBox("Finalize / QC")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(
            qt.QLabel(
                "Finalize hardens transforms, exports a CTA-aligned bilateral labelmap, "
                "and saves JSON/CSV review logs."
            )
        )
        form = qt.QFormLayout()
        self.tubeDiameterR = qt.QDoubleSpinBox()
        self.tubeDiameterR.setDecimals(2)
        self.tubeDiameterR.setRange(0.0, 20.0)
        form.addRow("Tube diameter R (mm)", self.tubeDiameterR)

        self.tubeDiameterL = qt.QDoubleSpinBox()
        self.tubeDiameterL.setDecimals(2)
        self.tubeDiameterL.setRange(0.0, 20.0)
        form.addRow("Tube diameter L (mm)", self.tubeDiameterL)

        self.intensityTol = qt.QDoubleSpinBox()
        self.intensityTol.setDecimals(1)
        self.intensityTol.setRange(0.0, 1000.0)
        form.addRow("Intensity tolerance", self.intensityTol)

        self.neighborhoodSize = qt.QDoubleSpinBox()
        self.neighborhoodSize.setDecimals(2)
        self.neighborhoodSize.setRange(0.0, 20.0)
        form.addRow("Neighbourhood size", self.neighborhoodSize)

        self.smoothingText = qt.QLineEdit()
        self.smoothingText.setPlaceholderText("method / level used")
        form.addRow("Smoothing", self.smoothingText)

        self.notesText = qt.QLineEdit()
        self.notesText.setPlaceholderText("review notes, edits, artifacts")
        form.addRow("Notes", self.notesText)
        layout.addLayout(form)

        btn = qt.QPushButton("Run Finalize SOP")
        btn.clicked.connect(self.onFinalize)
        layout.addWidget(btn)
        finalize_next_btn = qt.QPushButton("Finalize + Next Queue Case")
        finalize_next_btn.clicked.connect(self.onFinalizeAndNext)
        layout.addWidget(finalize_next_btn)
        clear_btn = qt.QPushButton("Clear Current Case Nodes")
        clear_btn.clicked.connect(self.clearCurrentCaseNodes)
        layout.addWidget(clear_btn)
        self.logBox = qt.QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(150)
        layout.addWidget(self.logBox)
        return group

    def onLoadCTA(self):
        file_path = qt.QFileDialog.getOpenFileName(None, "Select CTA NIfTI", "", "NIfTI (*.nii *.nii.gz)")
        if not file_path:
            return
        node = slicer.util.loadVolume(file_path)
        if node:
            self._track_case_node(node)
            self.ctaSelector.setCurrentNode(node)
            if not _text_value(self.outputDirText):
                self.outputDirText.setText(str(Path(file_path).parent))

    def onLoadVertebralLabel(self):
        file_path = qt.QFileDialog.getOpenFileName(
            None,
            "Select Existing Vertebral Label NIfTI",
            "",
            "NIfTI (*.nii *.nii.gz)",
        )
        if not file_path:
            return
        node = self.logic.load_labelmap(file_path)
        if node:
            self._track_case_node(node)
            self.vertebralSelector.setCurrentNode(node)
            if not _text_value(self.outputDirText):
                self.outputDirText.setText(str(Path(file_path).parent))

    def onLoadForamenPrior(self):
        file_path = qt.QFileDialog.getOpenFileName(
            None,
            "Select Vertebral Foramen Negative-Prior NIfTI",
            "",
            "NIfTI (*.nii *.nii.gz)",
        )
        if not file_path:
            return
        node = self.logic.load_labelmap(file_path)
        if node:
            self._track_case_node(node)
            if "foramen" not in node.GetName().lower() and "foramina" not in node.GetName().lower():
                node.SetName(FORAMEN_PRIOR_NAME)
            self.foramenPriorSelector.setCurrentNode(node)
            self._remember_foramen_prior(node)

    def onBrowseManifest(self):
        file_path = qt.QFileDialog.getOpenFileName(None, "Select Queue Manifest CSV", "", "CSV (*.csv)")
        if file_path:
            self.manifestPathText.setText(file_path)

    def onLoadManifest(self):
        self.logBox.clear()
        try:
            manifest_path = Path(_text_value(self.manifestPathText).strip())
            if not manifest_path.exists():
                raise RuntimeError(f"Manifest does not exist: {manifest_path}")
            self.queueManifestPath = manifest_path
            self.queueRows = read_manifest(manifest_path)
            outdir = _text_value(self.outputDirText).strip() or str(manifest_path.parent / "review_outputs")
            self.outputDirText.setText(outdir)
            self.queueStatusPath = queue_status_path(manifest_path, outdir)
            latest = latest_queue_status_by_case(self.queueStatusPath)
            self.queueIndex = first_pending_index(self.queueRows, latest)
            self._update_queue_label()
            self.loadQueueCase(self.queueIndex)
        except Exception as exc:
            self.logBox.append(f"ERROR loading manifest: {exc}")

    def onLoadCurrentCase(self):
        if not self.queueRows:
            slicer.util.errorDisplay("Load a manifest first.")
            return
        self.loadQueueCase(self.queueIndex)

    def onQueuePrevious(self):
        if not self.queueRows:
            return
        self.queueIndex = max(self.queueIndex - 1, 0)
        self.loadQueueCase(self.queueIndex)

    def onQueueNext(self):
        if not self.queueRows:
            return
        self._record_queue_status("skipped")
        self.queueIndex = min(self.queueIndex + 1, len(self.queueRows) - 1)
        self.loadQueueCase(self.queueIndex)

    def onBrowseOutputDir(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Output Folder", _text_value(self.outputDirText))
        if folder:
            self.outputDirText.setText(folder)

    def onCreateCurves(self):
        left = self.logic.get_or_create_curve(LEFT_CURVE_NAME, LEFT_SEGMENT_COLOR)
        right = self.logic.get_or_create_curve(RIGHT_CURVE_NAME, RIGHT_SEGMENT_COLOR)
        self._track_case_node(left)
        self._track_case_node(right)
        slicer.util.selectModule("Markups")
        try:
            slicer.modules.markups.logic().SetActiveListID(left)
            slicer.modules.markups.logic().StartPlaceMode(True)
        except Exception:
            pass

    def onAddSelectedForamenPrior(self):
        self._remember_foramen_prior(self.foramenPriorSelector.currentNode())

    def onOpenSegmentEditor(self):
        cta = self.ctaSelector.currentNode()
        if cta is None:
            slicer.util.errorDisplay("Please select a CTA volume.")
            return
        selected = self.vertebralSelector.currentNode()
        seg = self.logic.segmentation_from_selected_node(selected, cta)
        self._track_case_node(seg)
        self.vertebralSelector.setCurrentNode(seg)
        self.logic.ensure_bilateral_segments(seg)
        slicer.util.selectModule("SegmentEditor")
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editor.setSegmentationNode(seg)
            try:
                editor.setSourceVolumeNode(cta)
            except Exception:
                editor.setMasterVolumeNode(cta)
        except Exception:
            pass

    def onFinalize(self):
        self.logBox.clear()
        try:
            log = self._finalize_current_case()
            self._record_queue_status("completed", log=log)
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append("ERROR:")
            self.logBox.append(str(exc))
            self.logBox.append(traceback.format_exc())

    def onFinalizeAndNext(self):
        self.logBox.clear()
        try:
            log = self._finalize_current_case()
            self._record_queue_status("completed", log=log)
            self.logBox.append(json.dumps(log, indent=2))
            if self.queueRows and self.queueIndex < len(self.queueRows) - 1:
                self.queueIndex += 1
                self.loadQueueCase(self.queueIndex)
        except Exception as exc:
            self.logBox.append("ERROR:")
            self.logBox.append(str(exc))
            self.logBox.append(traceback.format_exc())

    def _finalize_current_case(self):
        from CTAFinalizeSOP import CTAFinalizeSOPLogic

        reviewer = normalize_reviewer_id(_text_value(self.reviewerText), required=True)
        status = validate_review_status(_combo_text(self.statusCombo))
        cta = self.ctaSelector.currentNode()
        vertebral_node = self.vertebralSelector.currentNode()
        if cta is None:
            raise RuntimeError("Please select a CTA volume.")
        if vertebral_node is None:
            raise RuntimeError("Please create/select a vertebral segmentation or load an existing labelmap.")
        if vertebral_node.IsA("vtkMRMLSegmentationNode"):
            self.logic.ensure_bilateral_segments(vertebral_node)
        params = {
            "tube_diameter_mm_right": _spin_value(self.tubeDiameterR),
            "tube_diameter_mm_left": _spin_value(self.tubeDiameterL),
            "intensity_tolerance": _spin_value(self.intensityTol),
            "neighbourhood_size": _spin_value(self.neighborhoodSize),
            "smoothing": _text_value(self.smoothingText).strip(),
            "notes": _text_value(self.notesText).strip(),
        }
        return CTAFinalizeSOPLogic().run(
            cta_node=cta,
            label_node=vertebral_node,
            output_dir=_text_value(self.outputDirText).strip() or None,
            reviewer_id=reviewer,
            review_status=status,
            extra_params=params,
            centerline_nodes=self.logic.centerline_nodes(),
            negative_prior_nodes=self._negative_prior_nodes(),
            save_scene=True,
        )

    def loadQueueCase(self, index: int):
        if not self.queueRows:
            return
        self.clearCurrentCaseNodes()
        self.queueIndex = max(0, min(index, len(self.queueRows) - 1))
        row = self.queueRows[self.queueIndex]
        self.currentCaseRow = row

        self.reviewerText.setText(row.get("reviewer_id", ""))
        _set_combo_text(self.statusCombo, row.get("review_status", "") or "in_progress")
        self.notesText.setText(row.get("notes", ""))

        cta_path = self._resolve_queue_path(row.get("cta_path", ""))
        if cta_path:
            cta = slicer.util.loadVolume(str(cta_path))
            if cta:
                cta.SetName(row.get("case_id") or cta.GetName())
                self._track_case_node(cta)
                self.ctaSelector.setCurrentNode(cta)

        label_path = self._resolve_queue_path(row.get("label_path", ""))
        if label_path and label_path.exists():
            label = self.logic.load_labelmap(str(label_path))
            if label:
                self._track_case_node(label)
                self.vertebralSelector.setCurrentNode(label)
        else:
            self.vertebralSelector.setCurrentNode(None)

        for prior_path_text in _split_paths(row.get("foramen_prior_path", "")):
            prior_path = self._resolve_queue_path(prior_path_text)
            if prior_path and prior_path.exists():
                prior = self.logic.load_labelmap(str(prior_path))
                if prior:
                    self._track_case_node(prior)
                    self.foramenPriorSelector.setCurrentNode(prior)
                    self._remember_foramen_prior(prior)

        self._record_queue_status("loaded")
        self._update_queue_label()

    def clearCurrentCaseNodes(self):
        for node in list(self.caseNodes):
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
        self.caseNodes = []
        self.negativePriorNodes = []
        self.currentCaseRow = None
        self._update_prior_list()
        try:
            self.ctaSelector.setCurrentNode(None)
            self.vertebralSelector.setCurrentNode(None)
            self.foramenPriorSelector.setCurrentNode(None)
        except Exception:
            pass

    def _track_case_node(self, node):
        if node is None:
            return
        existing_ids = {n.GetID() for n in self.caseNodes if n is not None}
        if node.GetID() not in existing_ids:
            self.caseNodes.append(node)

    def _record_queue_status(self, queue_status: str, log: dict | None = None):
        if not self.currentCaseRow or not self.queueStatusPath:
            return
        reviewer = _text_value(self.reviewerText).strip()
        status = _combo_text(self.statusCombo)
        notes = _text_value(self.notesText).strip()
        append_queue_status(
            self.queueStatusPath,
            queue_status_row(
                self.currentCaseRow,
                queue_status=queue_status,
                reviewer_id=reviewer,
                review_status=status,
                output_label=str((log or {}).get("output_label", "")),
                output_log=str((log or {}).get("output_log", "")),
                notes=notes,
            ),
        )

    def _update_queue_label(self):
        if not self.queueRows:
            self.queueLabel.setText("No queue loaded")
            return
        row = self.queueRows[self.queueIndex]
        status_path = str(self.queueStatusPath) if self.queueStatusPath else ""
        self.queueLabel.setText(
            f"Case {self.queueIndex + 1}/{len(self.queueRows)}: {row.get('case_id', '')} | "
            f"status log: {status_path}"
        )

    def _resolve_queue_path(self, value: str):
        if not value:
            return None
        path = Path(value)
        if path.is_absolute() or self.queueManifestPath is None:
            return path
        return self.queueManifestPath.parent / path

    def _remember_foramen_prior(self, node):
        if node is None:
            return
        self.logic.configure_negative_prior_display(node)
        existing_ids = {n.GetID() for n in self.negativePriorNodes if n is not None}
        if node.GetID() not in existing_ids:
            self.negativePriorNodes.append(node)
        self._update_prior_list()

    def _negative_prior_nodes(self):
        nodes = list(self.negativePriorNodes)
        try:
            current = self.foramenPriorSelector.currentNode()
            if current is not None:
                nodes.append(current)
        except Exception:
            pass
        return self.logic.deduplicate_nodes(nodes)

    def _update_prior_list(self):
        names = [node.GetName() for node in self.logic.deduplicate_nodes(self.negativePriorNodes)]
        self.priorListText.setPlainText("\n".join(names))


class CTAVertebralWizardLogic(ScriptedLoadableModuleLogic):
    def load_labelmap(self, file_path: str):
        """Load an existing vertebral NIfTI as a labelmap when possible."""
        try:
            node = slicer.util.loadLabelVolume(file_path)
            if node:
                return node
        except Exception:
            pass
        try:
            loaded = slicer.util.loadVolume(file_path, {"labelmap": True}, returnNode=True)
            if isinstance(loaded, tuple):
                return loaded[1] if loaded[0] else None
            return loaded
        except Exception:
            return slicer.util.loadVolume(file_path)

    def segmentation_from_selected_node(self, selected_node, cta_node):
        """Return an editable segmentation, importing an existing labelmap when needed."""
        if selected_node is not None and selected_node.IsA("vtkMRMLSegmentationNode"):
            return selected_node

        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "vertebral_seg")
        seg.CreateDefaultDisplayNodes()
        try:
            seg.SetReferenceImageGeometryParameterFromVolumeNode(cta_node)
        except Exception:
            pass

        if selected_node is not None:
            labelmap = self._as_labelmap(selected_node, cta_node)
            if labelmap is not None:
                try:
                    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, seg)
                    self._rename_imported_segments(seg)
                except Exception:
                    pass
        self.ensure_bilateral_segments(seg)
        return seg

    def _as_labelmap(self, node, cta_node):
        if node is None:
            return None
        if node.IsA("vtkMRMLLabelMapVolumeNode"):
            return node
        if not node.IsA("vtkMRMLScalarVolumeNode"):
            return None
        try:
            import numpy as np

            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", node.GetName() + "_labelmap")
            try:
                slicer.modules.volumes.logic().CopyVolumeGeometry(cta_node, labelmap)
            except Exception:
                slicer.modules.volumes.logic().CopyVolumeGeometry(node, labelmap)
            arr = slicer.util.arrayFromVolume(node)
            slicer.util.updateVolumeFromArray(labelmap, arr.astype(np.uint16, copy=False))
            return labelmap
        except Exception:
            return None

    def _rename_imported_segments(self, segmentation_node):
        segmentation = segmentation_node.GetSegmentation()
        names = (LEFT_SEGMENT_NAME, RIGHT_SEGMENT_NAME)
        colors = (LEFT_SEGMENT_COLOR, RIGHT_SEGMENT_COLOR)
        count = min(segmentation.GetNumberOfSegments(), 2)
        for i in range(count):
            segment = segmentation.GetNthSegment(i)
            segment.SetName(names[i])
            segment.SetColor(*colors[i])

    def configure_negative_prior_display(self, node):
        try:
            node.CreateDefaultDisplayNodes()
        except Exception:
            pass
        try:
            if node.IsA("vtkMRMLSegmentationNode"):
                display = node.GetDisplayNode()
                if display:
                    display.SetOpacity(0.25)
                segmentation = node.GetSegmentation()
                for i in range(segmentation.GetNumberOfSegments()):
                    segment = segmentation.GetNthSegment(i)
                    segment.SetColor(*FORAMEN_PRIOR_COLOR)
                return
        except Exception:
            pass
        try:
            display = node.GetDisplayNode()
            if display and hasattr(display, "SetOpacity"):
                display.SetOpacity(0.25)
        except Exception:
            pass

    def deduplicate_nodes(self, nodes: list) -> list:
        seen = set()
        deduped = []
        for node in nodes:
            if node is None:
                continue
            key = node.GetID() or node.GetName()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped

    def get_or_create_curve(self, name: str, color: tuple[float, float, float]):
        try:
            node = slicer.util.getNode(name)
        except Exception:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
        display = node.GetDisplayNode()
        if display is None:
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
        if display:
            display.SetSelectedColor(*color)
            display.SetColor(*color)
        return node

    def centerline_nodes(self) -> list:
        nodes = []
        for name in (LEFT_CURVE_NAME, RIGHT_CURVE_NAME):
            try:
                node = slicer.util.getNode(name)
                if node:
                    nodes.append(node)
            except Exception:
                pass
        return nodes

    def ensure_bilateral_segments(self, segmentation_node):
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation = segmentation_node.GetSegmentation()
        for name, color in ((LEFT_SEGMENT_NAME, LEFT_SEGMENT_COLOR), (RIGHT_SEGMENT_NAME, RIGHT_SEGMENT_COLOR)):
            segment_id = _segment_id_by_name(segmentation_node, name)
            if not segment_id:
                segment_id = segmentation.AddEmptySegment(name)
            segment = segmentation.GetSegment(segment_id)
            segment.SetName(name)
            segment.SetColor(*color)


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


def _segment_id_by_name(segmentation_node, name: str) -> str | None:
    target = name.lower()
    segmentation = segmentation_node.GetSegmentation()
    aliases = {
        LEFT_SEGMENT_NAME.lower(): {LEFT_SEGMENT_NAME.lower(), "left", "l", "vert_l", "vertebral_l", "vertebral_left"},
        RIGHT_SEGMENT_NAME.lower(): {RIGHT_SEGMENT_NAME.lower(), "right", "r", "vert_r", "vertebral_r", "vertebral_right"},
    }
    for i in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(i)
        segment_name = segmentation.GetNthSegment(i).GetName().lower().replace(" ", "_")
        if segment_name in aliases.get(target, {target}):
            return segment_id
    return None


def _node_path(node):
    try:
        storage = node.GetStorageNode()
        if storage and storage.GetFileName():
            return storage.GetFileName()
    except Exception:
        pass
    return None


def _guess_output_dir():
    for node_class in ("vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"):
        for node in slicer.util.getNodesByClass(node_class):
            path = _node_path(node)
            if path:
                return str(Path(path).parent)
    return ""


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


def _set_combo_text(widget, text: str):
    if not text:
        return
    try:
        index = widget.findText(text)
        if index >= 0:
            widget.setCurrentIndex(index)
    except Exception:
        pass


def _spin_value(widget):
    try:
        value = widget.value
        return value() if callable(value) else value
    except Exception:
        return None


def _split_paths(text: str) -> list[str]:
    values = []
    for part in (text or "").replace(",", ";").split(";"):
        value = part.strip()
        if value:
            values.append(value)
    return values


def show_wizard():
    """Create or reuse a single wizard dialog."""
    global _WIZARD_DIALOG, _WIZARD_WIDGET
    try:
        if _WIZARD_DIALOG is not None:
            _WIZARD_DIALOG.close()
    except Exception:
        pass
    _WIZARD_WIDGET = CTAVertebralWizardWidget()
    _WIZARD_WIDGET.setup()
    dlg = qt.QDialog()
    dlg.setWindowTitle("CTA Vertebral Wizard")
    layout = qt.QVBoxLayout(dlg)
    layout.addWidget(_WIZARD_WIDGET)
    dlg.show()
    _WIZARD_DIALOG = dlg
    return dlg
