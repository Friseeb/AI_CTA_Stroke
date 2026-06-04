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

_WIZARD_DIALOG = None
_WIZARD_WIDGET = None


class CTAVertebralWizard(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CTA Vertebral Wizard"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Wizard for vertebral artery manual segmentation.\n"
            "Step-by-step SOP with minimal UI."
        )
        self.parent.acknowledgementText = "Internal SOP"


_WIZARD_DIALOG = None
_WIZARD_WIDGET = None


class CTAVertebralWizardWidget(ScriptedLoadableModuleWidget):

    def resourcePath(self, filename):
        """Override to handle standalone dialog usage (module not registered)."""
        try:
            return super().resourcePath(filename)
        except (AttributeError, NameError):
            # Module not registered - return a non-existent path so
            # setupDeveloperSection skips the .ui button gracefully
            return os.path.join(os.path.expanduser("~"), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAVertebralWizardLogic()

        self.layout.addWidget(self._step1_group())
        self.layout.addWidget(self._step2_group())
        self.layout.addWidget(self._step3_group())
        self.layout.addWidget(self._step4_group())
        self.layout.addStretch(1)

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

    def _step2_group(self):
        group = qt.QGroupBox("Step 2: Draw Vertebral Centerline")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Create TWO curve markups (Right + Left).\n"
            "Use Markups → Curve; place points along each vertebral artery."
        ))
        btn = qt.QPushButton("Create R/L Curves & Open Markups")
        btn.clicked.connect(self.onCreateCurves)
        layout.addWidget(btn)
        return group

    def _step3_group(self):
        group = qt.QGroupBox("Step 3: Segment Vertebral Arteries")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Use Segment Editor to create TWO segments (Right + Left) in the SAME segmentation.\n"
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
        curve_r = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "vertebral_centerline_R"
        )
        curve_l = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "vertebral_centerline_L"
        )
        slicer.util.selectModule("Markups")
        try:
            slicer.modules.markups.logic().SetActiveListID(curve_r)
            slicer.modules.markups.logic().StartPlaceMode(True)
        except Exception:
            pass

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
            editor.setMasterVolumeNode(cta)
        except Exception:
            pass

    def onFinalize(self):
        self.logBox.clear()
        try:
            from CTAFinalizeSOP import CTAFinalizeSOPLogic

            def _spin_value(widget):
                v = widget.value
                return v() if callable(v) else v
            def _text_value(widget):
                t = widget.text
                return t() if callable(t) else t

            params = {
                "tube_diameter_mm_right": _spin_value(self.tubeDiameterR),
                "tube_diameter_mm_left": _spin_value(self.tubeDiameterL),
                "intensity_tolerance": _spin_value(self.intensityTol),
                "neighbourhood_size": _spin_value(self.neighborhoodSize),
                "smoothing": _text_value(self.smoothingText).strip(),
                "notes": _text_value(self.notesText).strip(),
            }
            output_dir = _text_value(self.outputDirText).strip() or None
            log = CTAFinalizeSOPLogic().run(extra_params=params, output_dir=output_dir)
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append("ERROR:")
            self.logBox.append(str(exc))
            self.logBox.append(traceback.format_exc())


class CTAVertebralWizardLogic(ScriptedLoadableModuleLogic):
    pass


def _node_path(node):
    try:
        storage = node.GetStorageNode()
        if storage and storage.GetFileName():
            return storage.GetFileName()
    except Exception:
        pass
    return None


def _guess_output_dir():
    # Prefer label path, fallback to CTA path
    labels = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
    if labels:
        p = _node_path(labels[0])
        if p:
            return str(Path(p).parent)
    ctas = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if ctas:
        p = _node_path(ctas[0])
        if p:
            return str(Path(p).parent)
    return ""


def _select_dir_dialog():
    return qt.QFileDialog.getExistingDirectory(None, "Select Output Folder", "")


def _safe_set_lineedit(le, value):
    try:
        le.setText(value)
    except Exception:
        pass


def _safe_get_lineedit(le):
    try:
        t = le.text
        return t() if callable(t) else t
    except Exception:
        return ""


def _safe_widget_value(widget):
    try:
        v = widget.value
        return v() if callable(v) else v
    except Exception:
        return None


def _safe_widget_text(widget):
    try:
        t = widget.text
        return t() if callable(t) else t
    except Exception:
        return ""


def _ensure_dir(path: str) -> str:
    return path


def _default_output_dir(self):
    return _guess_output_dir()


def _browse_output_dir(self):
    folder = _select_dir_dialog()
    if folder:
        _safe_set_lineedit(self.outputDirText, folder)


CTAVertebralWizardWidget._default_output_dir = _default_output_dir
CTAVertebralWizardWidget.onBrowseOutputDir = _browse_output_dir


def show_wizard():
    """Create or reuse a single wizard dialog (prevents duplicates)."""
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
