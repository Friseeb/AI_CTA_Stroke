"""CTAThrombusDemarcation — 3D Slicer scripted module.

Step-by-step wizard for manual intracranial thrombus demarcation on CTA.
Supports multiphase CTA (arterial + 1st/2nd delay) for collateral assessment.

Label map convention (saved by CTAThrombusFinalizeSOP):
  1 = thrombus
  2 = proximal vessel (optional)
  3 = distal vessel   (optional)
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import qt
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

# ── Vessel catalogue ──────────────────────────────────────────────────────────

VESSELS = [
    "-- select --",
    "ICA terminal",
    "ICA cavernous",
    "MCA M1",
    "MCA M2",
    "MCA M3",
    "ACA A1",
    "ACA A2",
    "PCA P1",
    "PCA P2",
    "Basilar",
    "Vertebrobasilar junction",
    "SCA",
    "PICA",
    "AICA",
    "Other",
]

SIDES = ["-- select --", "Left", "Right", "Bilateral", "N/A (midline)"]

TICI_OPTIONS = [
    "-- not assessed --",
    "0 – No perfusion",
    "1 – Minimal perfusion",
    "2a – <50% reperfusion",
    "2b – ≥50% reperfusion",
    "2c – Near-complete",
    "3 – Complete",
]

COLLATERAL_OPTIONS = [
    "-- not assessed --",
    "0 – Absent",
    "1 – Poor (<50% filling)",
    "2 – Moderate (≥50% filling)",
    "3 – Good (complete filling)",
]

PERVIOUSNESS_OPTIONS = [
    "-- not assessed --",
    "Pervious (contrast permeates clot)",
    "Non-pervious (no contrast in clot)",
]

_WIZARD_DIALOG = None
_WIZARD_WIDGET = None


# ── Module declaration ────────────────────────────────────────────────────────

class CTAThrombusDemarcation(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title       = "CTA Thrombus Demarcation"
        self.parent.categories  = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Wizard for manual intracranial thrombus demarcation on CTA.\n"
            "Supports multiphase CTA (arterial + delay phases).\n"
            "Saves thrombus labelmap + JSON with vessel, side, TICI, "
            "thrombus volume and SI length."
        )
        self.parent.acknowledgementText = "Internal SOP"


# ── Widget ────────────────────────────────────────────────────────────────────

class CTAThrombusDemarcationWidget(ScriptedLoadableModuleWidget):

    def resourcePath(self, filename):
        try:
            return super().resourcePath(filename)
        except (AttributeError, NameError):
            return os.path.join(os.path.expanduser("~"), "Resources", filename)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTAThrombusDemarcationLogic()

        self.layout.addWidget(self._step1_group())
        self.layout.addWidget(self._step2_group())
        self.layout.addWidget(self._step3_group())
        self.layout.addWidget(self._step4_group())
        self.layout.addStretch(1)

    # ── Step 1: Load volumes ─────────────────────────────────────────────────

    def _step1_group(self):
        group = qt.QGroupBox("Step 1: Load CTA Volumes")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Load the arterial CTA (required).\n"
            "Optionally load 1st and/or 2nd delay phases for collateral assessment."
        ))

        self.ctaSelector = self._volume_selector("CTA arterial (required)")
        layout.addWidget(qt.QLabel("Arterial CTA:"))
        layout.addWidget(self.ctaSelector)

        loadBtn = qt.QPushButton("Browse & Load Arterial CTA")
        loadBtn.clicked.connect(lambda: self._load_volume(self.ctaSelector, "arterial_cta"))
        layout.addWidget(loadBtn)

        layout.addWidget(qt.QLabel("1st Delay phase (optional):"))
        self.delay1Selector = self._volume_selector("1st Delay")
        layout.addWidget(self.delay1Selector)
        load1Btn = qt.QPushButton("Browse & Load 1st Delay")
        load1Btn.clicked.connect(lambda: self._load_volume(self.delay1Selector, "1st_delay"))
        layout.addWidget(load1Btn)

        layout.addWidget(qt.QLabel("2nd Delay phase (optional):"))
        self.delay2Selector = self._volume_selector("2nd Delay")
        layout.addWidget(self.delay2Selector)
        load2Btn = qt.QPushButton("Browse & Load 2nd Delay")
        load2Btn.clicked.connect(lambda: self._load_volume(self.delay2Selector, "2nd_delay"))
        layout.addWidget(load2Btn)

        linkBtn = qt.QPushButton("Link Volumes (sync slice views)")
        linkBtn.clicked.connect(self.onLinkVolumes)
        layout.addWidget(linkBtn)
        return group

    # ── Step 2: Annotate occlusion site ──────────────────────────────────────

    def _step2_group(self):
        group = qt.QGroupBox("Step 2: Annotate Occlusion Site")
        layout  = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Select the occluded vessel and side. Place a Fiducial markup\n"
            "at the proximal tip of the thrombus."
        ))

        form = qt.QFormLayout()

        self.vesselCombo = qt.QComboBox()
        for v in VESSELS:
            self.vesselCombo.addItem(v)
        form.addRow("Vessel:", self.vesselCombo)

        self.sideCombo = qt.QComboBox()
        for s in SIDES:
            self.sideCombo.addItem(s)
        form.addRow("Side:", self.sideCombo)

        self.ticiCombo = qt.QComboBox()
        for t in TICI_OPTIONS:
            self.ticiCombo.addItem(t)
        form.addRow("TICI (if post-EVT):", self.ticiCombo)

        self.collateralCombo = qt.QComboBox()
        for c in COLLATERAL_OPTIONS:
            self.collateralCombo.addItem(c)
        form.addRow("Collaterals:", self.collateralCombo)

        self.visibleDelay1Check = qt.QCheckBox("Thrombus visible on 1st Delay")
        self.visibleDelay2Check = qt.QCheckBox("Thrombus visible on 2nd Delay")
        form.addRow("Phase visibility:", self.visibleDelay1Check)
        form.addRow("", self.visibleDelay2Check)

        self.hyperdenseCheck = qt.QCheckBox("Hyperdense clot sign (on CTA or NCCT)")
        self.hyperdenseCheck.setToolTip(
            "Mark if the thrombus appears hyperdense relative to surrounding brain tissue "
            "on non-contrast CT or as a dense artery sign on CTA."
        )
        form.addRow("Hyperdense:", self.hyperdenseCheck)

        self.perviousnessCombo = qt.QComboBox()
        for p in PERVIOUSNESS_OPTIONS:
            self.perviousnessCombo.addItem(p)
        self.perviousnessCombo.setToolTip(
            "Clot perviousness: whether contrast material permeates the thrombus on CTA. "
            "Pervious clots typically have better lysis response."
        )
        form.addRow("Clot perviousness:", self.perviousnessCombo)

        layout.addLayout(form)

        fiducialBtn = qt.QPushButton("Place Proximal Tip Fiducial")
        fiducialBtn.setToolTip("Places a single fiducial at the proximal tip of the thrombus.")
        fiducialBtn.clicked.connect(self.onPlaceFiducial)
        layout.addWidget(fiducialBtn)
        return group

    # ── Step 3: Segment ───────────────────────────────────────────────────────

    def _step3_group(self):
        group = qt.QGroupBox("Step 3: Segment Thrombus")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Use Segment Editor to create THREE segments:\n"
            "  1 = thrombus  (required)\n"
            "  2 = proximal_vessel  (optional — vessel lumen proximal to clot)\n"
            "  3 = distal_vessel   (optional — vessel lumen distal to clot)\n\n"
            "Use Paint/Threshold brush on the arterial CTA.\n"
            "Use delay phases to judge collateral filling of distal vessel."
        ))

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = True
        self.segSelector.addEnabled   = False
        self.segSelector.removeEnabled = False
        self.segSelector.noneEnabled  = True
        self.segSelector.showHidden   = False
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(self.segSelector)

        openBtn = qt.QPushButton("Create Segmentation & Open Segment Editor")
        openBtn.clicked.connect(self.onOpenSegmentEditor)
        layout.addWidget(openBtn)

        addSegBtn = qt.QPushButton("Add Standard Segments (thrombus / proximal / distal)")
        addSegBtn.clicked.connect(self.onAddStandardSegments)
        layout.addWidget(addSegBtn)
        return group

    # ── Step 4: Finalize ──────────────────────────────────────────────────────

    def _step4_group(self):
        group = qt.QGroupBox("Step 4: Finalize & Save")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "SOP: Runs Finalize SOP — hardens transforms, resamples to CTA grid,\n"
            "computes thrombus volume and SI length, saves labelmap + JSON."
        ))

        form = qt.QFormLayout()
        self.notesText = qt.QLineEdit()
        self.notesText.setPlaceholderText("Free-text notes on segmentation quality, uncertainty…")
        form.addRow("Notes:", self.notesText)

        self.outputDirText = qt.QLineEdit()
        self.outputDirText.setPlaceholderText("Output folder (defaults to CTA folder/thrombus_manual)")
        browseBtn = qt.QPushButton("Browse")
        browseBtn.clicked.connect(self.onBrowseOutputDir)
        row = qt.QHBoxLayout()
        row.addWidget(self.outputDirText)
        row.addWidget(browseBtn)
        form.addRow("Output folder:", row)
        layout.addLayout(form)

        finalizeBtn = qt.QPushButton("Run Finalize SOP")
        finalizeBtn.clicked.connect(self.onFinalize)
        layout.addWidget(finalizeBtn)

        self.logBox = qt.QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(140)
        layout.addWidget(self.logBox)
        return group

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _volume_selector(self, name):
        sel = slicer.qMRMLNodeComboBox()
        sel.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        sel.selectNodeUponCreation = False
        sel.addEnabled   = False
        sel.removeEnabled = False
        sel.noneEnabled  = True
        sel.showHidden   = False
        sel.setMRMLScene(slicer.mrmlScene)
        return sel

    def _load_volume(self, selector, label):
        path = qt.QFileDialog.getOpenFileName(
            None, f"Select {label} NIfTI", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        node = slicer.util.loadVolume(path)
        if node:
            node.SetName(label)
            selector.setCurrentNode(node)

    def onLinkVolumes(self):
        """Set all slice views to compare volumes (linked cursor)."""
        lm = slicer.app.layoutManager()
        for name in ["Red", "Yellow", "Green"]:
            sliceLogic = lm.sliceWidget(name).sliceLogic()
            sliceLogic.GetSliceCompositeNode().SetLinkedControl(1)
        slicer.util.resetSliceViews()

    def onPlaceFiducial(self):
        fid = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "thrombus_proximal_tip"
        )
        fid.GetDisplayNode().SetSelectedColor(1, 0, 0)
        slicer.util.selectModule("Markups")
        try:
            slicer.modules.markups.logic().SetActiveListID(fid)
            slicer.modules.markups.logic().StartPlaceMode(False)
        except Exception:
            pass

    def onOpenSegmentEditor(self):
        cta = self.ctaSelector.currentNode()
        if cta is None:
            slicer.util.errorDisplay("Please select an arterial CTA volume first.")
            return
        seg = self.segSelector.currentNode()
        if seg is None:
            seg = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "thrombus_seg"
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

    def onAddStandardSegments(self):
        seg = self.segSelector.currentNode()
        if seg is None:
            slicer.util.errorDisplay("Create/select a segmentation node first (Step 3 button).")
            return
        existing = set()
        for i in range(seg.GetSegmentation().GetNumberOfSegments()):
            existing.add(seg.GetSegmentation().GetNthSegment(i).GetName())

        segments_to_add = [
            ("thrombus",        (1.0, 0.2, 0.2)),   # red
            ("proximal_vessel", (0.2, 0.6, 1.0)),   # blue
            ("distal_vessel",   (0.2, 1.0, 0.4)),   # green
        ]
        for name, color in segments_to_add:
            if name not in existing:
                seg_id = seg.GetSegmentation().AddEmptySegment(name)
                segment = seg.GetSegmentation().GetSegment(seg_id)
                segment.SetColor(*color)

    def onFinalize(self):
        self.logBox.clear()
        try:
            from CTAThrombusFinalizeSOP import CTAThrombusFinalizeSOP_Logic

            annotation = self._collect_annotation()
            output_dir = self.outputDirText.text.strip() or None
            log = CTAThrombusFinalizeSOP_Logic().run(
                annotation=annotation, output_dir=output_dir
            )
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append("ERROR:")
            self.logBox.append(str(exc))
            self.logBox.append(traceback.format_exc())

    def _collect_annotation(self) -> dict:
        def _combo_val(combo):
            t = combo.currentText
            return t() if callable(t) else t

        def _check_val(cb):
            v = cb.isChecked()
            return v() if callable(v) else v

        def _text_val(le):
            t = le.text
            return t() if callable(t) else t

        vessel = _combo_val(self.vesselCombo)
        side   = _combo_val(self.sideCombo)
        tici   = _combo_val(self.ticiCombo)
        coll   = _combo_val(self.collateralCombo)

        # Proximal tip fiducial RAS coords
        tip_ras = None
        fiducials = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        for fid in fiducials:
            if "proximal_tip" in fid.GetName().lower() or "thrombus" in fid.GetName().lower():
                if fid.GetNumberOfControlPoints() > 0:
                    ras = [0, 0, 0]
                    fid.GetNthControlPointPosition(0, ras)
                    tip_ras = [round(c, 2) for c in ras]
                    break

        perv = _combo_val(self.perviousnessCombo)

        return {
            "vessel":                 vessel if not vessel.startswith("--") else None,
            "side":                   side   if not side.startswith("--")   else None,
            "tici":                   tici   if not tici.startswith("--")   else None,
            "collaterals":            coll   if not coll.startswith("--")   else None,
            "thrombus_visible_delay1": _check_val(self.visibleDelay1Check),
            "thrombus_visible_delay2": _check_val(self.visibleDelay2Check),
            "hyperdense":             _check_val(self.hyperdenseCheck),
            "clot_perviousness":      perv if not perv.startswith("--") else None,
            "proximal_tip_ras_mm":    tip_ras,
            "notes":                  _text_val(self.notesText).strip(),
            "annotated_at":           datetime.now().isoformat(),
        }

    def onBrowseOutputDir(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Select Output Folder", "")
        if folder:
            self.outputDirText.setText(folder)


class CTAThrombusDemarcationLogic(ScriptedLoadableModuleLogic):
    pass


# ── Standalone launcher (run from Slicer Python console) ─────────────────────

def show_wizard():
    """Create or reuse a single wizard dialog."""
    global _WIZARD_DIALOG, _WIZARD_WIDGET
    try:
        if _WIZARD_DIALOG is not None:
            _WIZARD_DIALOG.close()
    except Exception:
        pass
    _WIZARD_WIDGET = CTAThrombusDemarcationWidget()
    _WIZARD_WIDGET.setup()
    dlg = qt.QDialog()
    dlg.setWindowTitle("CTA Intracranial Thrombus Demarcation")
    dlg.resize(520, 900)
    layout = qt.QVBoxLayout(dlg)
    layout.addWidget(_WIZARD_WIDGET)
    dlg.show()
    _WIZARD_DIALOG = dlg
    return dlg


def get_last_annotation():
    """Devuelve la anotacion (vessel/side/tici/...) del wizard de Demarcation
    actualmente abierto (o el ultimo creado), o None si no hay ninguno.

    Se usa desde CTAThrombusFinalizeSOP.py para que el boton standalone de
    Finalize tambien capture la anotacion cuando el usuario la eligio en el
    wizard de Demarcation pero finaliza desde el modulo Finalize por separado
    -- antes esa anotacion se perdia en silencio (ver sub-485_thrombus_clean_log.json,
    "annotation": {} vacio, agosto 2026)."""
    global _WIZARD_WIDGET
    if _WIZARD_WIDGET is None:
        return None
    try:
        return _WIZARD_WIDGET._collect_annotation()
    except Exception:
        return None
