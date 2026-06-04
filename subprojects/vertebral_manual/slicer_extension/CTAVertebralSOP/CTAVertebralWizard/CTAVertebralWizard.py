import json
import os
from datetime import datetime
from pathlib import Path

import slicer
import qt
import traceback

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


class CTAVertebralWizard(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Vertebral Wizard"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["Sebastian Fridman (NYU)"]
        self.parent.helpText = (
            "Step-by-step wizard for vertebral artery manual segmentation.\n"
            "Guides the SOP: Load CTA → Draw centerlines → Segment → Finalize."
        )
        self.parent.acknowledgementText = "NYU Langone CTA Stroke Project"
        self.parent.dependencies = ["CTAFinalizeSOP"]


class CTAVertebralWizardWidget(ScriptedLoadableModuleWidget):

    def resourcePath(self, filename):
        """Override to handle both registered-module and standalone-dialog usage."""
        try:
            return super().resourcePath(filename)
        except AttributeError:
            # Module not registered (running as standalone dialog via show_wizard)
            return os.path.join(os.path.dirname(__file__), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAVertebralWizardLogic()

        self.layout.addWidget(self._step1_group())
        self.layout.addWidget(self._step2_group())
        self.layout.addWidget(self._step3_group())
        self.layout.addWidget(self._step4_group())
        self.layout.addStretch(1)

    # ── Step 1: Select CTA ──────────────────────────────────────────

    def _step1_group(self):
        group = qt.QGroupBox("Step 1: Select CTA")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Load the CTA NIfTI. Verify axial slices are correct."
        ))

        self.ctaSelector = slicer.qMRMLNodeComboBox()
        self.ctaSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctaSelector.selectNodeUponCreation = True
        self.ctaSelector.addEnabled = False
        self.ctaSelector.removeEnabled = False
        self.ctaSelector.noneEnabled = False
        self.ctaSelector.showHidden = False
        self.ctaSelector.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(self.ctaSelector)

        loadBtn = qt.QPushButton("Load CTA (.nii.gz)")
        loadBtn.clicked.connect(self.onLoadCTA)
        layout.addWidget(loadBtn)
        return group

    # ── Step 2: Draw Centerlines ────────────────────────────────────

    def _step2_group(self):
        group = qt.QGroupBox("Step 2: Draw Vertebral Centerline")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Create TWO curve markups (Right + Left).\n"
            "Use Markups \u2192 Curve; place points along each vertebral artery."
        ))
        btn = qt.QPushButton("Create R/L Curves & Open Markups")
        btn.clicked.connect(self.onCreateCurves)
        layout.addWidget(btn)
        return group

    # ── Step 3: Segment Arteries ────────────────────────────────────

    def _step3_group(self):
        group = qt.QGroupBox("Step 3: Segment Vertebral Arteries")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Use Segment Editor to create TWO segments (Right + Left) "
            "in the SAME segmentation.\n"
            "Ensure segmentation aligns with CTA."
        ))

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = True
        self.segSelector.addEnabled = False
        self.segSelector.removeEnabled = False
        self.segSelector.noneEnabled = True
        self.segSelector.showHidden = False
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(self.segSelector)

        btn = qt.QPushButton("Create/Select Segmentation & Open Segment Editor")
        btn.clicked.connect(self.onOpenSegmentEditor)
        layout.addWidget(btn)
        return group

    # ── Step 4: Finalize / Save ─────────────────────────────────────

    def _step4_group(self):
        group = qt.QGroupBox("Step 4: Finalize / Save Clean Labelmap")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Run Finalize SOP to harden transforms, fix cropping,\n"
            "and save a clean labelmap aligned to CTA."
        ))

        layout.addWidget(qt.QLabel("Parameters (logged only; does not change segmentation):"))
        form = qt.QFormLayout()

        self.tubeDiameterR = qt.QDoubleSpinBox()
        self.tubeDiameterR.setDecimals(2)
        self.tubeDiameterR.setRange(0.0, 20.0)
        self.tubeDiameterR.setValue(0.0)
        form.addRow("Tube diameter R (mm)", self.tubeDiameterR)

        self.tubeDiameterL = qt.QDoubleSpinBox()
        self.tubeDiameterL.setDecimals(2)
        self.tubeDiameterL.setRange(0.0, 20.0)
        self.tubeDiameterL.setValue(0.0)
        form.addRow("Tube diameter L (mm)", self.tubeDiameterL)

        self.intensityTol = qt.QDoubleSpinBox()
        self.intensityTol.setDecimals(1)
        self.intensityTol.setRange(0.0, 1000.0)
        self.intensityTol.setValue(0.0)
        form.addRow("Intensity tolerance", self.intensityTol)

        self.neighborhoodSize = qt.QDoubleSpinBox()
        self.neighborhoodSize.setDecimals(2)
        self.neighborhoodSize.setRange(0.0, 20.0)
        self.neighborhoodSize.setValue(0.0)
        form.addRow("Neighbourhood size", self.neighborhoodSize)

        self.smoothingText = qt.QLineEdit()
        self.smoothingText.setPlaceholderText("e.g., 0 / 1 / 2 or method")
        form.addRow("Smoothing", self.smoothingText)

        self.notesText = qt.QLineEdit()
        self.notesText.setPlaceholderText("free-text notes on trial/error or curve edits")
        form.addRow("Notes", self.notesText)

        self.outputDirText = qt.QLineEdit()
        self.outputDirText.setPlaceholderText("Output folder (defaults to label/CTA folder)")
        self.outputDirText.setText(self._default_output_dir())
        browseBtn = qt.QPushButton("Browse")
        browseBtn.clicked.connect(self.onBrowseOutputDir)
        outRow = qt.QHBoxLayout()
        outRow.addWidget(self.outputDirText)
        outRow.addWidget(browseBtn)
        form.addRow("Output folder", outRow)

        layout.addLayout(form)
        btn = qt.QPushButton("Run Finalize SOP")
        btn.clicked.connect(self.onFinalize)
        layout.addWidget(btn)

        self.logBox = qt.QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(120)
        layout.addWidget(self.logBox)
        return group

    # ── Actions ─────────────────────────────────────────────────────

    def onLoadCTA(self):
        file_path = qt.QFileDialog.getOpenFileName(
            None, "Select CTA NIfTI", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not file_path:
            return
        node = slicer.util.loadVolume(file_path)
        if node:
            self.ctaSelector.setCurrentNode(node)

    def onCreateCurves(self):
        slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "vertebral_centerline_R"
        )
        slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "vertebral_centerline_L"
        )
        slicer.util.selectModule("Markups")

    def onOpenSegmentEditor(self):
        cta = self.ctaSelector.currentNode()
        if cta is None:
            slicer.util.errorDisplay("Please select a CTA volume.")
            return
        seg = self.segSelector.currentNode()
        if seg is None:
            seg = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "vertebral_seg"
            )
            seg.CreateDefaultDisplayNodes()
            self.segSelector.setCurrentNode(seg)

        slicer.util.selectModule("SegmentEditor")
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editor.setSegmentationNode(seg)
            editor.setSourceVolumeNode(cta)
        except Exception:
            pass

    def onFinalize(self):
        self.logBox.clear()
        try:
            from CTAFinalizeSOP import CTAFinalizeSOPLogic

            def _val(w):
                v = w.value
                return v() if callable(v) else v

            def _txt(w):
                t = w.text
                return t() if callable(t) else t

            params = {
                "tube_diameter_mm_right": _val(self.tubeDiameterR),
                "tube_diameter_mm_left": _val(self.tubeDiameterL),
                "intensity_tolerance": _val(self.intensityTol),
                "neighbourhood_size": _val(self.neighborhoodSize),
                "smoothing": _txt(self.smoothingText).strip(),
                "notes": _txt(self.notesText).strip(),
            }
            output_dir = _txt(self.outputDirText).strip() or None
            log = CTAFinalizeSOPLogic().run(extra_params=params, output_dir=output_dir)
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append("ERROR:")
            self.logBox.append(str(exc))
            self.logBox.append(traceback.format_exc())

    def onBrowseOutputDir(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Output Folder", "")
        if folder:
            self.outputDirText.setText(folder)

    # ── Helpers ─────────────────────────────────────────────────────

    def _default_output_dir(self):
        for cls in ("vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"):
            for n in slicer.util.getNodesByClass(cls):
                storage = n.GetStorageNode()
                if storage and storage.GetFileName():
                    return str(Path(storage.GetFileName()).parent)
        return ""


class CTAVertebralWizardLogic(ScriptedLoadableModuleLogic):
    pass
