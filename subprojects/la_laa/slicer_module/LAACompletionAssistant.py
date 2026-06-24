"""3D Slicer scripted module: LAA Completion & SLAAO Annotation Assistant.

Thin GUI over `laa_annotation_core`. The module drives the 7-step annotation
workflow (load CTA + priors -> candidate -> long-axis workspace -> prompt-assisted
correction -> optional MONAILabel update -> manual correction -> Type 1 region ->
finalize). All non-Slicer logic (label contract, prompt schema, pilot/repro
metrics, session logging, output layout) lives in the core so it can be unit
tested without Slicer.

The module is fully usable WITHOUT MONAILabel: the AI steps are optional. See
`README.md` and `../docs/SOP.md`.
"""

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
    ScriptedLoadableModuleTest,
)

from laa_annotation_core import (
    LAA_LABEL_CONTRACT,
    NEGATIVE_CATEGORIES,
    POSITIVE_CATEGORIES,
    PLUGIN_VERSION,
    TYPE1_LABEL,
    WHOLE_LAA_LABEL,
    PilotMetrics,
    PromptLog,
    CANDIDATE_SOURCE_FILES,
    append_session_csv,
    build_monai_inference_request,
    build_session_log,
    comparison_labelmap,
    comparison_metrics,
    infer_case_id,
    output_paths,
    resolve_candidate_file,
    validate_label_values,
)

# Prior sources that resolve to a file on disk, plus two manual fallbacks.
_PRIOR_SOURCES = tuple(CANDIDATE_SOURCE_FILES.keys())
CANDIDATE_SOURCES = _PRIOR_SOURCES + ("External MONAI / file…", "Empty segmentation")

PILOT_ROOT_SETTING = "LAACompletionAssistant/pilotRoot"
VISTA3D_PYTHON_SETTING = "LAACompletionAssistant/vista3dPython"

# Repo root inferred from this file: <repo>/subprojects/la_laa/slicer_module/...
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Shared utilities (cta_common). Slicer's embedded Python won't have the editable
# install on its path, so bootstrap the source dir directly.
import sys as _sys

_CTA_COMMON_SRC = _REPO_ROOT / "cta_common" / "src"
if _CTA_COMMON_SRC.is_dir() and str(_CTA_COMMON_SRC) not in _sys.path:
    _sys.path.insert(0, str(_CTA_COMMON_SRC))
from cta_common.slicer_qc import load_mask_aligned as _shared_load_mask_aligned
from cta_common.subprocess_env import make_subprocess_env as _make_subprocess_env
_ROI_VISTA3D_SCRIPT = _REPO_ROOT / "scripts" / "run_laa_vista3d_roi.py"
_ROI_SAM3D_SCRIPT = _REPO_ROOT / "scripts" / "run_laa_sam3d_roi.py"
_V3D_PROMPT_SCRIPT = _REPO_ROOT / "scripts" / "run_laa_vista3d_prompt.py"
ROI_NODE_NAME = "LAA_roi"


def _default_vista3d_python() -> str:
    """Python interpreter that has monai + transformers<5 (for VISTA3D)."""
    cand = _REPO_ROOT / ".venv_dt" / "bin" / "python"
    return str(cand) if cand.exists() else "python3"


def _clean_subprocess_env(python_exe) -> dict:
    """Environment for spawning an external Python from Slicer.

    Slicer sets PYTHONHOME/PYTHONPATH/DYLD_* that would break an external conda
    Python's imports (torch/monai). Delegates to the shared helper.
    """
    return _make_subprocess_env(python_exe)

# Lowercased contract names that are CONTEXT/hard-negative labels, i.e. not part
# of the Whole-LAA (label 1) mask. Everything else in the segmentation that is
# not the Type-1 region is treated as Whole LAA on export.
_CONTEXT_LABEL_NAMES = {
    LAA_LABEL_CONTRACT[v]["name"].lower() for v in (3, 4, 5, 6, 7)
}
_TYPE1_NAME = LAA_LABEL_CONTRACT[TYPE1_LABEL]["name"].lower()


def _default_pilot_root() -> str:
    """Best-effort default pilot root for the output folder."""
    for guess in (
        os.environ.get("LAA_PILOT_ROOT"),
        str(_REPO_ROOT / "outputs" / "laa_pilot"),
    ):
        if guess and os.path.isdir(guess):
            return guess
    return os.path.expanduser("~")


class LAACompletionAssistant(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "LAA Completion Assistant"
        self.parent.categories = ["CTA-in-AI"]
        self.parent.contributors = ["AI_CTA_Stroke"]
        self.parent.helpText = (
            "Recover the COMPLETE left atrial appendage (incl. distal "
            "hypoattenuated / SLAAO Type 1 region) from CTA, support human "
            "correction with positive/negative prompts and optional MONAILabel, "
            "and capture reproducible pilot + reproducibility metrics."
        )
        self.parent.acknowledgementText = "Internal SOP (la_laa Phase 0/1)"


class LAACompletionAssistantWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = LAACompletionAssistantLogic()
        self.promptLog = PromptLog()
        self.annotationTimer = qt.QElapsedTimer()
        self.editCount = 0
        self.candidateSourceName = ""
        self.candidateSourcePath = ""

        self.layout.addWidget(self._case_group())
        self.layout.addWidget(self._pilot_group())
        self.layout.addWidget(self._candidate_group())
        self.layout.addWidget(self._roi_group())
        self.layout.addWidget(self._workspace_group())
        self.layout.addWidget(self._prompt_group())
        self.layout.addWidget(self._monai_group())
        self.layout.addWidget(self._segmentation_group())
        self.layout.addWidget(self._finalize_group())
        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # GUI groups
    # ------------------------------------------------------------------

    def _case_group(self):
        group = qt.QGroupBox("1. Case")
        form = qt.QFormLayout(group)

        self.ctaSelector = _node_selector(["vtkMRMLScalarVolumeNode"], none_enabled=True)
        form.addRow("CTA volume", self.ctaSelector)
        load_btn = qt.QPushButton("Load CTA (.nii.gz / DICOM folder)")
        load_btn.clicked.connect(self.onLoadCTA)
        form.addRow("", load_btn)

        self.caseIdText = qt.QLineEdit()
        self.caseIdText.setPlaceholderText("inferred from CTA filename if blank")
        form.addRow("Case ID", self.caseIdText)

        self.readerText = qt.QLineEdit()
        self.readerText.setPlaceholderText("Reader A / B / C or initials")
        self.readerText.textChanged.connect(self._update_output_label)
        form.addRow("Reader ID", self.readerText)

        # Pilot root persists across reloads (QSettings); the per-case output
        # folder is derived automatically as <pilot root>/<case id>.
        self.pilotRootText = qt.QLineEdit()
        self.pilotRootText.setText(
            slicer.app.userSettings().value(PILOT_ROOT_SETTING, _default_pilot_root())
        )
        self.pilotRootText.textChanged.connect(self._on_pilot_root_changed)
        browse_btn = qt.QPushButton("Browse")
        browse_btn.clicked.connect(self.onBrowsePilotRoot)
        out_row = qt.QHBoxLayout()
        out_row.addWidget(self.pilotRootText)
        out_row.addWidget(browse_btn)
        form.addRow("Pilot root", out_row)

        self.caseOutputLabel = qt.QLabel("")
        self.caseOutputLabel.setWordWrap(True)
        self.caseOutputLabel.setStyleSheet("color: gray;")
        form.addRow("Case output", self.caseOutputLabel)
        self.caseIdText.textChanged.connect(self._update_output_label)
        self._update_output_label()
        return group

    def _pilot_group(self):
        group = qt.QGroupBox("1b. Pilot cases (load patient)")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "Pick a staged pilot case and load its CTA + candidate in one click."
        ))
        row = qt.QHBoxLayout()
        self.pilotCaseCombo = qt.QComboBox()
        row.addWidget(self.pilotCaseCombo)
        refresh_btn = qt.QPushButton("↻")
        refresh_btn.setToolTip("Rescan the pilot root for cases")
        refresh_btn.clicked.connect(self.refreshPilotCases)
        row.addWidget(refresh_btn)
        layout.addLayout(row)
        load_btn = qt.QPushButton("Load selected patient (CTA + candidate)")
        load_btn.clicked.connect(self.onLoadPilotCase)
        layout.addWidget(load_btn)
        self.refreshPilotCases()
        return group

    def refreshPilotCases(self):
        self.pilotCaseCombo.clear()
        root = Path(self.pilotRootText.text.strip()) if self.pilotRootText.text.strip() else None
        if not root or not root.exists():
            return
        cases = []
        for d in sorted(root.glob("sub-*")):
            if d.is_dir() and list(d.glob(f"laa_annotation/*/logs/{d.name}_session.json")):
                cases.append(d.name)
        self.pilotCaseCombo.addItems(cases)

    def onLoadPilotCase(self):
        try:
            self._load_pilot_case()
        except Exception as exc:  # pragma: no cover - GUI path
            slicer.util.errorDisplay(f"Load patient failed: {exc}\n{traceback.format_exc()}")

    def _load_pilot_case(self):
        case_id = self.pilotCaseCombo.currentText
        root = self.pilotRootText.text.strip()
        if not case_id or not root:
            slicer.util.errorDisplay("Set the Pilot root and pick a case first.")
            return
        reader = self.readerText.text.strip() or "readerA"
        case_dir = Path(root) / case_id
        sess_path = case_dir / "laa_annotation" / reader / "logs" / f"{case_id}_session.json"
        if not sess_path.exists():
            # fall back to whatever reader folder exists
            hits = list(case_dir.glob(f"laa_annotation/*/logs/{case_id}_session.json"))
            if not hits:
                slicer.util.errorDisplay(f"No session.json for {case_id} under {case_dir}.")
                return
            sess_path = hits[0]
            reader = sess_path.parents[1].name
        session = json.loads(sess_path.read_text())

        cta_path = session.get("cta_path", "")
        if not cta_path or not Path(cta_path).exists():
            slicer.util.errorDisplay(f"CTA not found for {case_id}: {cta_path}")
            return
        cta_node = slicer.util.loadVolume(cta_path)
        self.ctaSelector.setCurrentNode(cta_node)
        self.caseIdText.setText(case_id)
        self.readerText.setText(reader)

        cand_rel = session.get("candidate") or "candidate_masks/vista3d_laa.nii.gz"
        cand_path = case_dir / "laa_annotation" / reader / cand_rel
        if cand_path.exists():
            seg = self.logic.load_mask_aligned(str(cand_path), cta_node, f"{case_id}_candidate")
            if seg:
                self.candidateSelector.setCurrentNode(seg)
                self._set_candidate_source(
                    session.get("candidate_source", Path(cand_path).stem), str(cand_path)
                )
        self.refreshCandidateSources()
        self.logic.setup_longaxis_workspace(cta_node, None)
        self._update_output_label()
        self.statusLabel.setText(
            f"Loaded {case_id} (reader {reader}). Start timer, place points, run VISTA3D promptable, correct, Finalize."
        )

    def _case_output_dir(self) -> str:
        """Automatic per-case output dir = <pilot root>/<case id>."""
        root = self.pilotRootText.text.strip()
        case_id = self.caseIdText.text.strip() or infer_case_id(
            self.logic.node_path(self._cta_node())
        )
        if not root or not case_id:
            return ""
        return str(Path(root) / case_id)

    def _update_output_label(self, *_):
        out = self._case_output_dir()
        reader = self.readerText.text.strip() or "readerA"
        self.caseOutputLabel.setText(
            f"{out}/laa_annotation/{reader}/" if out else "(set Pilot root + Case ID)"
        )

    def _on_pilot_root_changed(self, text):
        slicer.app.userSettings().setValue(PILOT_ROOT_SETTING, text)
        self._update_output_label()

    def _candidate_group(self):
        group = qt.QGroupBox("2. LAA candidate")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "The AI candidate is NOT assumed correct. Pick a prior source and load, "
            "then correct. VISTA-3D / NUDF / TotalSegmentator (atrial_appendage_left) "
            "/ Consensus are auto-found in the case folder when present."
        ))
        src_row = qt.QHBoxLayout()
        self.candidateCombo = qt.QComboBox()
        for src in CANDIDATE_SOURCES:
            self.candidateCombo.addItem(src)
        src_row.addWidget(self.candidateCombo)
        refresh_btn = qt.QPushButton("↻")
        refresh_btn.setToolTip("Rescan the case folder for available prior sources")
        refresh_btn.clicked.connect(self.refreshCandidateSources)
        src_row.addWidget(refresh_btn)
        layout.addLayout(src_row)
        load_src_btn = qt.QPushButton("Load selected source")
        load_src_btn.clicked.connect(self.onLoadCandidateSource)
        layout.addWidget(load_src_btn)

        self.candidateSelector = _node_selector(
            ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"],
            none_enabled=True,
        )
        layout.addWidget(self.candidateSelector)
        load_btn = qt.QPushButton("Load candidate mask from file…")
        load_btn.clicked.connect(self.onLoadCandidate)
        layout.addWidget(load_btn)
        make_btn = qt.QPushButton("Create LAA segmentation (with label contract)")
        make_btn.clicked.connect(self.onCreateSegmentation)
        layout.addWidget(make_btn)
        self.candidateSourceLabel = qt.QLabel("")
        self.candidateSourceLabel.setWordWrap(True)
        self.candidateSourceLabel.setStyleSheet("color: gray;")
        layout.addWidget(self.candidateSourceLabel)
        return group

    def _roi_group(self):
        group = qt.QGroupBox("2b. VISTA3D promptable (label 108 + your points)")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel(
            "Anatomy-aware completion: place POSITIVE prompts on the missed distal "
            "tip / lobes (and optional NEGATIVE prompts) in section 4, then run. "
            "VISTA3D extends LAA (class 108) toward your points and auto-crops a "
            "window around them (fast, ~40s CPU). No ROI box needed. Output keeps "
            "the existing candidate and adds the VISTA3D-prompt result."
        ))
        run_btn = qt.QPushButton("VISTA3D promptable (candidate ∪ points)")
        run_btn.clicked.connect(self.onRunVista3dPrompt)
        layout.addWidget(run_btn)
        self.roiStatusLabel = qt.QLabel("")
        self.roiStatusLabel.setWordWrap(True)
        layout.addWidget(self.roiStatusLabel)
        return group

    def _workspace_group(self):
        group = qt.QGroupBox("3. Long-axis workspace")
        layout = qt.QVBoxLayout(group)
        layout.addWidget(qt.QLabel("Annotate in the LAA long-axis view, not pure axial slices."))
        btn = qt.QPushButton("Set up axial/sagittal/coronal + LAA long-axis")
        btn.clicked.connect(self.onSetupWorkspace)
        layout.addWidget(btn)
        return group

    def _prompt_group(self):
        group = qt.QGroupBox("4. Prompts")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()
        self.promptTypeCombo = qt.QComboBox()
        self.promptTypeCombo.addItems(["positive", "negative"])
        self.promptTypeCombo.currentTextChanged.connect(self._refresh_categories)
        form.addRow("Prompt type", self.promptTypeCombo)
        self.promptCategoryCombo = qt.QComboBox()
        form.addRow("Category", self.promptCategoryCombo)
        layout.addLayout(form)
        self._refresh_categories(self.promptTypeCombo.currentText)

        place_btn = qt.QPushButton("Place prompt point")
        place_btn.clicked.connect(self.onPlacePrompt)
        layout.addWidget(place_btn)
        self.promptCountLabel = qt.QLabel("Prompts: 0 (0+ / 0-)")
        layout.addWidget(self.promptCountLabel)
        return group

    def _monai_group(self):
        group = qt.QGroupBox("5. MONAILabel update (optional)")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()
        self.monaiServerText = qt.QLineEdit()
        self.monaiServerText.setPlaceholderText("http://localhost:8000 (leave blank to skip)")
        form.addRow("Server", self.monaiServerText)
        self.monaiModelText = qt.QLineEdit()
        self.monaiModelText.setPlaceholderText("vista3d / fine-tuned model name")
        form.addRow("Model", self.monaiModelText)
        layout.addLayout(form)
        run_btn = qt.QPushButton("Send crop + mask + prompts -> update")
        run_btn.clicked.connect(self.onRunMonai)
        layout.addWidget(run_btn)
        layout.addWidget(qt.QLabel("Module remains fully usable without MONAILabel."))
        return group

    def _segmentation_group(self):
        group = qt.QGroupBox("6/7. Manual correction + Type 1")
        layout = qt.QVBoxLayout(group)
        edit_btn = qt.QPushButton("Open Segment Editor (Paint/Erase/Scissors/Islands/Smoothing)")
        edit_btn.clicked.connect(self.onOpenSegmentEditor)
        layout.addWidget(edit_btn)
        layout.addWidget(qt.QLabel(
            "Type 1 = geometric distal hypoattenuated subregion (label 2), nested "
            "in Whole LAA. Do NOT apply HU thresholds; HU analysis is downstream."
        ))
        return group

    def _finalize_group(self):
        group = qt.QGroupBox("Finalize")
        layout = qt.QVBoxLayout(group)
        form = qt.QFormLayout()
        self.segConfidenceSpin = _confidence_spin()
        form.addRow("Segmentation confidence (0-1)", self.segConfidenceSpin)
        self.type1ConfidenceSpin = _confidence_spin()
        form.addRow("Type 1 confidence (0-1)", self.type1ConfidenceSpin)
        self.imageQualityCombo = qt.QComboBox()
        for q in (1, 2, 3, 4, 5):
            self.imageQualityCombo.addItem(str(q))
        self.imageQualityCombo.setCurrentText("3")
        form.addRow("Image quality (1-5)", self.imageQualityCombo)
        self.notesText = qt.QLineEdit()
        form.addRow("Notes", self.notesText)
        layout.addLayout(form)

        timer_row = qt.QHBoxLayout()
        start_btn = qt.QPushButton("Start timer")
        start_btn.clicked.connect(self.onStartTimer)
        timer_row.addWidget(start_btn)
        finalize_btn = qt.QPushButton("Finalize case")
        finalize_btn.clicked.connect(self.onFinalize)
        timer_row.addWidget(finalize_btn)
        layout.addLayout(timer_row)
        self.statusLabel = qt.QLabel("")
        layout.addWidget(self.statusLabel)
        return group

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _refresh_categories(self, prompt_type):
        cats = POSITIVE_CATEGORIES if prompt_type == "positive" else NEGATIVE_CATEGORIES
        self.promptCategoryCombo.clear()
        self.promptCategoryCombo.addItems(list(cats))

    def onLoadCTA(self):
        file_path = qt.QFileDialog.getOpenFileName(None, "Load CTA", "", "NIfTI (*.nii *.nii.gz)")
        if not file_path:
            return
        node = slicer.util.loadVolume(file_path)
        if node:
            self.ctaSelector.setCurrentNode(node)
            if not self.caseIdText.text:
                self.caseIdText.setText(infer_case_id(file_path))
            self._update_output_label()

    def onBrowsePilotRoot(self):
        directory = qt.QFileDialog.getExistingDirectory(None, "Pilot root folder")
        if directory:
            self.pilotRootText.setText(directory)

    def onLoadCandidate(self):
        file_path = qt.QFileDialog.getOpenFileName(None, "Load candidate", "", "NIfTI (*.nii *.nii.gz)")
        if not file_path:
            return
        node = self.logic.load_mask_aligned(file_path, self._cta_node(), "candidate")
        if node:
            self.candidateSelector.setCurrentNode(node)
            self._set_candidate_source(Path(file_path).stem, file_path)

    def _candidate_search_dirs(self):
        """Folders to scan for staged prior masks for the current case."""
        case_id = self.caseIdText.text.strip()
        root = self.pilotRootText.text.strip()
        if not case_id or not root:
            return []
        reader = self.readerText.text.strip() or "readerA"
        case_dir = Path(root) / case_id
        dirs = [case_dir / "laa_annotation" / reader / "candidate_masks"]
        # fall back to any reader's candidate_masks, plus a prior_fusion dir
        dirs += list(case_dir.glob("laa_annotation/*/candidate_masks"))
        dirs.append(case_dir / "prior_fusion")
        seen, unique = set(), []
        for d in dirs:
            if str(d) not in seen:
                seen.add(str(d))
                unique.append(d)
        return unique

    def refreshCandidateSources(self):
        """Relabel the prior-source combo with on-disk availability."""
        dirs = self._candidate_search_dirs()
        for i in range(self.candidateCombo.count):
            src = self.candidateCombo.itemText(i).split("  —")[0].split("  ✓")[0].strip()
            stems = CANDIDATE_SOURCE_FILES.get(src)
            if stems is None:
                continue  # manual fallbacks (file… / Empty)
            found = resolve_candidate_file(dirs, stems)
            label = f"{src}  ✓ {found.name}" if found else f"{src}  — not staged"
            self.candidateCombo.setItemText(i, label)

    def onLoadCandidateSource(self):
        try:
            self._load_candidate_source()
        except Exception as exc:  # pragma: no cover - GUI path
            slicer.util.errorDisplay(f"Load source failed: {exc}\n{traceback.format_exc()}")

    def _load_candidate_source(self):
        src = self.candidateCombo.currentText.split("  —")[0].split("  ✓")[0].strip()
        if src.startswith("Empty"):
            self.onCreateSegmentation()
            self._set_candidate_source("empty", "")
            return
        if src.startswith("External MONAI"):
            self.onLoadCandidate()
            return
        stems = CANDIDATE_SOURCE_FILES.get(src)
        if not stems:
            slicer.util.errorDisplay(f"Unknown source '{src}'.")
            return
        dirs = self._candidate_search_dirs()
        path = resolve_candidate_file(dirs, stems)
        if path is None:
            searched = "\n".join(f"  {d}" for d in dirs) or "  (set Case ID + Pilot root first)"
            slicer.util.errorDisplay(
                f"No {src} mask found. Looked for {stems} in:\n{searched}"
            )
            return
        node = self.logic.load_mask_aligned(str(path), self._cta_node(), f"{src}_candidate")
        if node:
            self.candidateSelector.setCurrentNode(node)
            self._set_candidate_source(src, str(path))

    def _set_candidate_source(self, name, path):
        self.candidateSourceName = name
        self.candidateSourcePath = str(path) if path else ""
        if path:
            self.candidateSourceLabel.setText(f"Candidate source: {name} ({Path(path).name})")
        else:
            self.candidateSourceLabel.setText(f"Candidate source: {name}")

    def onRunVista3dPrompt(self):
        try:
            self._run_vista3d_prompt()
        except Exception as exc:  # pragma: no cover - GUI path
            slicer.util.errorDisplay(f"VISTA3D prompt failed: {exc}\n{traceback.format_exc()}")

    def _run_vista3d_prompt(self):
        self._recover_prompts_if_empty()
        cta_node = self._cta_node()
        if cta_node is None:
            vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            cta_node = vols[0] if vols else None
        if cta_node is None:
            slicer.util.errorDisplay("No CTA volume loaded.")
            return
        ct_path = self.logic.node_path(cta_node)
        if not ct_path:
            slicer.util.errorDisplay("CTA has no file path on disk; load it from a .nii.gz.")
            return

        fg = [p.coordinate for p in self.promptLog.prompts if p.prompt_type == "positive"]
        bg = [p.coordinate for p in self.promptLog.prompts if p.prompt_type == "negative"]
        if not fg:
            slicer.util.errorDisplay(
                "Place at least one POSITIVE prompt on the missed LAA region (section 4) first."
            )
            return

        case_id = self.caseIdText.text.strip() or infer_case_id(ct_path)
        reader_id = self.readerText.text.strip() or "readerA"
        paths = output_paths(self._case_output_dir(), case_id, reader_id=reader_id).mkdirs()
        out = paths.candidate_masks / "vista3d_prompt.nii.gz"
        prior = self.logic.candidate_mask_path(self._candidate_node(), cta_node, paths)
        py = slicer.app.userSettings().value(VISTA3D_PYTHON_SETTING, _default_vista3d_python())

        self.roiStatusLabel.setText(
            f"Running VISTA3D promptable with {len(fg)} +pt / {len(bg)} -pt … (UI may freeze ~40s)"
        )
        slicer.app.processEvents()
        rc, stdout, stderr = self.logic.run_vista3d_prompt(cta_node, ct_path, out, fg, bg, prior, py)
        if rc != 0 or not Path(out).exists():
            tail = (stderr or stdout or "").strip().splitlines()[-15:]
            slicer.util.errorDisplay(
                "VISTA3D prompt failed (rc={}).\nInterpreter: {}\n\n{}".format(rc, py, "\n".join(tail) or "no output")
            )
            self.roiStatusLabel.setText("VISTA3D prompt failed — see dialog.")
            return
        seg = self.logic.load_mask_aligned(str(out), cta_node, f"{case_id}_vista3d_prompt")
        if seg:
            self.candidateSelector.setCurrentNode(seg)
        self.roiStatusLabel.setText(
            f"Loaded {out.name} (candidate ∪ VISTA3D-prompt). Add more points on still-missed regions and rerun, "
            "then hand-finish the dark tip."
        )

    def onCreateSegmentation(self):
        self.logic.create_laa_segmentation(self._cta_node())
        self.statusLabel.setText("Created LAA segmentation with label contract.")

    def onSetupWorkspace(self):
        self.logic.setup_longaxis_workspace(self._cta_node(), self._candidate_node())

    def onPlacePrompt(self):
        prompt_type = self.promptTypeCombo.currentText
        category = self.promptCategoryCombo.currentText
        model = self.monaiModelText.text or ""
        self.logic.place_prompt_point(
            prompt_type, category,
            on_placed=lambda ras: self._record_prompt(prompt_type, category, ras, model),
        )

    def _record_prompt(self, prompt_type, category, ras, model):
        self.promptLog.add(prompt_type, category, tuple(ras), model_used=model)
        self.promptCountLabel.setText(
            f"Prompts: {self.promptLog.count} "
            f"({self.promptLog.positive_count}+ / {self.promptLog.negative_count}-)"
        )

    def onRunMonai(self):
        server = self.monaiServerText.text.strip()
        if not server:
            slicer.util.infoDisplay("No MONAILabel server set; skipping (manual workflow is fine).")
            return
        request = build_monai_inference_request(
            image=self.logic.node_path(self._cta_node()) or "",
            model=self.monaiModelText.text or "vista3d",
            prompt_log=self.promptLog,
            current_label=self.logic.node_path(self._candidate_node()),
        )
        try:
            self.logic.run_monai(server, request)
            self.statusLabel.setText("MONAILabel update applied.")
        except Exception as exc:  # pragma: no cover - GUI path
            slicer.util.errorDisplay(f"MONAILabel update failed: {exc}\n{traceback.format_exc()}")

    def onOpenSegmentEditor(self):
        self.editCount += 1
        slicer.util.selectModule("SegmentEditor")

    def onStartTimer(self):
        self.annotationTimer.start()
        self.statusLabel.setText("Annotation timer started.")

    def onFinalize(self):
        try:
            self._finalize()
        except Exception as exc:  # pragma: no cover - GUI path
            slicer.util.errorDisplay(f"Finalize failed: {exc}\n{traceback.format_exc()}")

    def _recover_prompts_if_empty(self):
        """Rebuild the prompt log from scene fiducials (survives a module reload).

        Placed prompts are markups named ``prompt_<type>_<category>``. If the
        in-memory log is empty (e.g. the widget was reloaded), reconstruct it
        from those nodes so Finalize still records them.
        """
        if self.promptLog.count > 0:
            return
        log = PromptLog()
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
            name = node.GetName() or ""
            if not name.startswith("prompt_"):
                continue
            ptype, _, category = name[len("prompt_"):].partition("_")
            if ptype not in ("positive", "negative") or not category:
                continue
            for i in range(node.GetNumberOfControlPoints()):
                ras = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(i, ras)
                try:
                    log.add(ptype, category, tuple(ras), model_used=self.monaiModelText.text or "")
                except ValueError:
                    pass
        if log.count:
            self.promptLog = log

    def _finalize(self):
        if not self.pilotRootText.text.strip():
            slicer.util.errorDisplay("Set the Pilot root first.")
            return
        self._recover_prompts_if_empty()

        # Resolve CTA + segmentation, falling back to scene nodes if the
        # selectors were reset (e.g. after a module reload).
        cta_node = self._cta_node()
        if cta_node is None:
            vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            cta_node = vols[0] if vols else None
            if cta_node is not None:
                self.ctaSelector.setCurrentNode(cta_node)
        if cta_node is None:
            slicer.util.errorDisplay("No CTA volume found (load the CTA first).")
            return

        seg_node = self._candidate_node()
        if seg_node is None or not hasattr(seg_node, "GetSegmentation"):
            segs = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
            seg_node = segs[0] if segs else None
            if seg_node is not None:
                self.candidateSelector.setCurrentNode(seg_node)
        if seg_node is None:
            slicer.util.errorDisplay("No segmentation found to export (load/correct an LAA candidate first).")
            return

        case_id = self.caseIdText.text.strip() or infer_case_id(
            self.logic.node_path(cta_node)
        )
        if not case_id:
            slicer.util.errorDisplay("Set a Case ID (or load a CTA so it can be inferred).")
            return
        reader_id = self.readerText.text.strip() or "readerA"
        out_dir = self._case_output_dir()  # automatic: <pilot root>/<case id>
        paths = output_paths(out_dir, case_id, reader_id=reader_id).mkdirs()

        label_values, whole_path, type1_path = self.logic.export_masks(
            seg_node, cta_node, paths
        )
        warnings = validate_label_values(label_values)

        comparison = None
        if whole_path and self.candidateSourcePath:
            try:
                comparison = self.logic.compare_candidate_vs_final(
                    self.candidateSourcePath, seg_node, cta_node, paths
                )
            except Exception as exc:  # pragma: no cover - GUI path
                warnings.append(f"new-vs-old comparison failed: {exc}")

        elapsed_s = self.annotationTimer.elapsed() / 1000.0 if self.annotationTimer.isValid() else None
        pilot = PilotMetrics(
            case_id=case_id,
            reader_id=reader_id,
            model_used=self.monaiModelText.text or "",
            annotation_time_s=elapsed_s,
            prompt_count=self.promptLog.count,
            positive_prompt_count=self.promptLog.positive_count,
            negative_prompt_count=self.promptLog.negative_count,
            edit_count=self.editCount,
            segmentation_confidence=self.segConfidenceSpin.value,
            type1_confidence=self.type1ConfidenceSpin.value,
            image_quality=int(self.imageQualityCombo.currentText),
            type1_present=TYPE1_LABEL in {int(v) for v in label_values},
            notes=self.notesText.text or "",
        )
        pilot.save(paths.pilot_metrics())
        self.promptLog.case_id = case_id
        self.promptLog.reader_id = reader_id
        self.promptLog.save(paths.prompt_log())

        session = build_session_log(
            case_id=case_id, reader_id=reader_id, pilot=pilot,
            prompt_log=self.promptLog, output_dir=str(paths.root),
            whole_laa_mask=whole_path, type1_mask=type1_path, warnings=warnings,
            plugin_version=PLUGIN_VERSION,
        )
        session["candidate_source"] = self.candidateSourceName
        session["candidate_source_path"] = self.candidateSourcePath
        if comparison is not None:
            session["candidate_comparison"] = comparison
        paths.session_json().write_text(json.dumps(session, indent=2))
        append_session_csv(paths.session_csv(), session)

        msg = f"Finalized {case_id} -> {paths.root}"
        if comparison is not None:
            msg += (
                "\nvs candidate ({src}): Dice {d:.3f}, +{add:.2f} mL / -{rem:.2f} mL "
                "(net {net:+.2f} mL)".format(
                    src=self.candidateSourceName or "?",
                    d=comparison["dice"],
                    add=comparison["added_volume_ml"],
                    rem=comparison["removed_volume_ml"],
                    net=comparison["volume_change_ml"],
                )
            )
        if warnings:
            msg += "\nWarnings:\n- " + "\n- ".join(warnings)
        self.statusLabel.setText(msg)
        slicer.util.infoDisplay(msg)

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _cta_node(self):
        return self.ctaSelector.currentNode()

    def _candidate_node(self):
        return self.candidateSelector.currentNode()


class LAACompletionAssistantLogic(ScriptedLoadableModuleLogic):
    """Scene-interaction logic. Kept minimal; pure logic lives in the core."""

    def create_laa_segmentation(self, cta_node):
        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "LAA_annotation")
        seg_node.CreateDefaultDisplayNodes()
        if cta_node is not None:
            seg_node.SetReferenceImageGeometryParameterFromVolumeNode(cta_node)
        segmentation = seg_node.GetSegmentation()
        for value in sorted(LAA_LABEL_CONTRACT):
            spec = LAA_LABEL_CONTRACT[value]
            seg_id = segmentation.AddEmptySegment(f"label{value}", spec["name"])
            segment = segmentation.GetSegment(seg_id)
            if segment is not None:
                segment.SetColor(*spec["color"])
        return seg_node

    def setup_longaxis_workspace(self, cta_node, candidate_node):
        layout_manager = slicer.app.layoutManager()
        # Four-up gives axial/sagittal/coronal; the 3D view supports a reformatted
        # long-axis using the markups line below as the alignment reference.
        layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        if cta_node is not None:
            slicer.util.setSliceViewerLayers(background=cta_node, fit=True)
        # Encourage long-axis: drop a line markup for the reader to align the LAA.
        line = slicer.mrmlScene.GetFirstNodeByName("LAA_long_axis")
        if line is None:
            line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "LAA_long_axis")
        return line

    def create_roi_box(self, existing, cta_node):
        """Return an ROI box node, creating one centered on the view if needed."""
        roi = existing
        if roi is None:
            roi = slicer.mrmlScene.GetFirstNodeByName(ROI_NODE_NAME)
        if roi is None:
            roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", ROI_NODE_NAME)
        if cta_node is not None:
            bounds = [0.0] * 6
            cta_node.GetRASBounds(bounds)
            center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2,
                      (bounds[4] + bounds[5]) / 2]
            roi.SetCenter(center)
            roi.SetSize(60.0, 60.0, 60.0)  # mm; user drags handles to fit the LAA
        return roi

    def start_roi_drawing(self, existing):
        """Create a fresh ROI node and put the mouse into ROI place/draw mode."""
        roi = existing
        if roi is None:
            roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", ROI_NODE_NAME)
        roi.RemoveAllControlPoints()
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode")
        selection.SetActivePlaceNodeID(roi.GetID())
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        interaction.SetPlaceModePersistence(0)
        interaction.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)
        return roi

    def roi_to_ijk_box(self, roi_node, cta_node):
        """Convert an ROI box (RAS bounds) to a clamped voxel index box (lo, hi)."""
        import numpy as np
        import vtk

        bounds = [0.0] * 6
        roi_node.GetRASBounds(bounds)  # xmin,xmax,ymin,ymax,zmin,zmax (RAS)
        ras2ijk = vtk.vtkMatrix4x4()
        cta_node.GetRASToIJKMatrix(ras2ijk)
        corners = []
        for x in (bounds[0], bounds[1]):
            for y in (bounds[2], bounds[3]):
                for z in (bounds[4], bounds[5]):
                    p = ras2ijk.MultiplyPoint([x, y, z, 1.0])
                    corners.append(p[:3])
        c = np.array(corners)
        dims = cta_node.GetImageData().GetDimensions()  # (i, j, k)
        lo = np.clip(np.floor(c.min(0)).astype(int), 0, np.array(dims))
        hi = np.clip(np.ceil(c.max(0)).astype(int), 0, np.array(dims))
        return lo, hi

    def run_roi_vista3d(self, cta_node, roi_node, ct_path, out_path, python_exe):
        """Crop to the ROI and run VISTA3D (label 108) via the external script."""
        lo, hi = self.roi_to_ijk_box(roi_node, cta_node)
        cmd = [
            str(python_exe), str(_ROI_VISTA3D_SCRIPT),
            "--ct", str(ct_path), "--out", str(out_path),
            "--roi-ijk", str(lo[0]), str(lo[1]), str(lo[2]), str(hi[0]), str(hi[1]), str(hi[2]),
            "--margin", "8", "8", "4", "--device", "auto",
        ]
        return self._run_external(cmd, _clean_subprocess_env(python_exe), "VISTA3D", python_exe)

    def _ras_points_to_ijk(self, pts_ras, cta_node):
        import vtk

        m = vtk.vtkMatrix4x4()
        cta_node.GetRASToIJKMatrix(m)
        out = []
        for r in pts_ras:
            p = m.MultiplyPoint([float(r[0]), float(r[1]), float(r[2]), 1.0])
            out.append([int(round(p[0])), int(round(p[1])), int(round(p[2]))])
        return out

    def candidate_mask_path(self, candidate_node, cta_node, paths):
        """Return a path to the prior (VISTA) binary mask to union with SAM.

        Uses the candidate's source file when it was loaded from a .nii.gz;
        otherwise exports the current segmentation's union to a temp file.
        """
        if candidate_node is not None and hasattr(candidate_node, "GetSegmentation"):
            src = _node_path(candidate_node)
            if src and str(src).endswith((".nii", ".nii.gz")):
                return str(src)
            seg = candidate_node.GetSegmentation()
            union = None
            for sid in seg.GetSegmentIDs():
                arr = slicer.util.arrayFromSegmentBinaryLabelmap(candidate_node, sid, cta_node)
                if arr is None:
                    continue
                a = arr > 0
                union = a if union is None else (union | a)
            if union is not None and union.any():
                tmp = paths.candidate_masks / "_prior_tmp.nii.gz"
                self._save_binary(cta_node, union, tmp)
                return str(tmp)
        for name in ("vista3d_laa.nii.gz", "consensus_laa.nii.gz"):
            p = paths.candidate_masks / name
            if p.exists():
                return str(p)
        return None

    def run_roi_sam3d(self, cta_node, roi_node, ct_path, out_path, fg_ras, bg_ras,
                      prior_path, server, python_exe):
        """Crop to the ROI and run SAM-3D (point prompts) via the external script."""
        lo, hi = self.roi_to_ijk_box(roi_node, cta_node)
        fg = self._ras_points_to_ijk(fg_ras, cta_node)
        bg = self._ras_points_to_ijk(bg_ras, cta_node)
        fg_s = ";".join(",".join(str(v) for v in p) for p in fg)
        bg_s = ";".join(",".join(str(v) for v in p) for p in bg)
        cmd = [
            str(python_exe), str(_ROI_SAM3D_SCRIPT),
            "--ct", str(ct_path), "--out", str(out_path),
            "--roi-ijk", str(lo[0]), str(lo[1]), str(lo[2]), str(hi[0]), str(hi[1]), str(hi[2]),
            "--fg", fg_s, "--bg", bg_s, "--server", server, "--model", "sam_3d", "--device", "cpu",
        ]
        if prior_path:
            cmd += ["--prior-mask", str(prior_path)]
        env = _clean_subprocess_env(python_exe)
        return self._run_external(cmd, env, "SAM-3D", python_exe)

    def load_mask_aligned(self, path, cta_node, name):
        """Load a voxel-aligned mask and force the CTA node's exact geometry.

        Delegates to ``cta_common.slicer_qc.load_mask_aligned`` (the slaobids
        sform/Z-flip workaround), kept as a method so existing call sites work.
        """
        return _shared_load_mask_aligned(path, cta_node, name)

    def run_vista3d_prompt(self, cta_node, ct_path, out_path, fg_ras, bg_ras, prior_path, python_exe):
        """VISTA3D promptable (class 108 + points) via the bundle, points in RAS."""
        def _fmt(pts):
            return ";".join(",".join(str(c) for c in p) for p in pts)

        cmd = [
            str(python_exe), str(_V3D_PROMPT_SCRIPT),
            "--ct", str(ct_path), "--out", str(out_path),
            "--label-id", "108",
            f"--fg={_fmt(fg_ras)}", f"--bg={_fmt(bg_ras)}",
        ]
        if prior_path:
            cmd += ["--prior-mask", str(prior_path)]
        return self._run_external(cmd, _clean_subprocess_env(python_exe), "VISTA3D-prompt", python_exe)

    def _run_external(self, cmd, env, tag, python_exe):
        import subprocess

        print(f"[LAA {tag}]", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        except Exception as exc:
            return 1, "", f"Failed to launch {tag} ({python_exe}): {exc}"
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
        return proc.returncode, proc.stdout, proc.stderr

    def place_prompt_point(self, prompt_type, category, on_placed):
        node_name = f"prompt_{prompt_type}_{category}"
        fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
        color = (0.1, 0.9, 0.1) if prompt_type == "positive" else (0.9, 0.1, 0.1)
        if fid.GetDisplayNode():
            fid.GetDisplayNode().SetSelectedColor(*color)

        def _on_point_added(caller, event):
            idx = caller.GetNumberOfControlPoints() - 1
            ras = [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(idx, ras)
            on_placed(tuple(ras))
            caller.RemoveObserver(obs_id)

        obs_id = fid.AddObserver(fid.PointPositionDefinedEvent, _on_point_added)
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetActivePlaceNodeID(fid.GetID())
        slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(
            slicer.vtkMRMLInteractionNode.Place
        )

    def run_monai(self, server, request):  # pragma: no cover - network path
        """POST the request to a MONAILabel server and load the returned label.

        Imported lazily so the module loads even when monailabel/requests are
        absent. The annotation workflow does not depend on this path.
        """
        import requests

        resp = requests.post(f"{server.rstrip('/')}/infer/{request['model']}", json=request, timeout=120)
        resp.raise_for_status()
        # MONAILabel typically returns a multipart with the label file; callers
        # adapt this to their server version. Kept as a hook.
        return resp

    def export_masks(self, seg_node, cta_node, paths):
        """Export Whole-LAA (label 1) and Type-1 (label 2) masks on the CTA grid.

        Maps by SEGMENT rather than by labelmap integer value, so an imported
        candidate (whose segment carries an arbitrary value like 108) is still
        recognized: the Type-1 region is the segment named like the contract's
        "SLAAO Type 1 region"; the Whole LAA is the union of every other segment
        that is not a known context/hard-negative label (LA body, veins, etc.).
        Returns (label_values, whole_path, type1_path).
        """
        whole, type1 = self.extract_whole_and_type1(seg_node, cta_node)

        values = set()
        whole_path = paths.whole_laa_mask()
        if whole is not None and whole.any():
            values.add(WHOLE_LAA_LABEL)
            self._save_binary(cta_node, whole, whole_path)
        type1_path = None
        if type1 is not None and type1.any():
            values.add(TYPE1_LABEL)
            type1_path = paths.type1_mask()
            self._save_binary(cta_node, type1, type1_path)
        return values, (str(whole_path) if WHOLE_LAA_LABEL in values else None), (
            str(type1_path) if type1_path else None
        )

    def extract_whole_and_type1(self, seg_node, cta_node):
        """Return (whole_laa_bool, type1_bool) arrays on the CTA grid (or None).

        Maps by SEGMENT name/id so an imported candidate with an arbitrary label
        value (e.g. 108) is still recognized. Whole LAA = union of every segment
        that is neither the Type-1 region nor a context/hard-negative label.
        """
        if seg_node is None or not hasattr(seg_node, "GetSegmentation"):
            return None, None
        segmentation = seg_node.GetSegmentation()
        whole = None
        type1 = None
        for seg_id in list(segmentation.GetSegmentIDs()):
            segment = segmentation.GetSegment(seg_id)
            name = (segment.GetName() or "").strip().lower() if segment else ""
            is_type1 = (seg_id == "label2") or (_TYPE1_NAME in name) or ("type 1" in name)
            is_context = (seg_id in {f"label{v}" for v in (3, 4, 5, 6, 7)}) or (name in _CONTEXT_LABEL_NAMES)
            arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, seg_id, cta_node)
            if arr is None:
                continue
            arr = arr > 0
            if not arr.any():
                continue
            if is_type1:
                type1 = arr if type1 is None else (type1 | arr)
            elif not is_context:
                whole = arr if whole is None else (whole | arr)
        return whole, type1

    def compare_candidate_vs_final(self, old_path, seg_node, cta_node, paths):
        """Save a new-vs-old labelmap + metrics for the corrected mask vs candidate.

        Returns the metrics dict (with the saved labelmap path) or None when the
        old candidate or the final mask is unavailable.
        """
        import numpy as np

        new_whole, _ = self.extract_whole_and_type1(seg_node, cta_node)
        if new_whole is None or not new_whole.any():
            return None
        if not old_path or not Path(old_path).exists():
            return None
        old_seg = self.load_mask_aligned(str(old_path), cta_node, "_old_candidate_cmp")
        try:
            old_whole, _ = self.extract_whole_and_type1(old_seg, cta_node)
        finally:
            if old_seg is not None:
                slicer.mrmlScene.RemoveNode(old_seg)
        if old_whole is None:
            old_whole = np.zeros_like(new_whole)
        spacing = tuple(cta_node.GetSpacing()) if cta_node else (1.0, 1.0, 1.0)
        labelmap = comparison_labelmap(old_whole, new_whole)
        metrics = comparison_metrics(old_whole, new_whole, spacing=spacing)
        cmp_path = paths.comparison_mask()
        self._save_binary(cta_node, labelmap, cmp_path)
        metrics["comparison_mask"] = str(cmp_path)
        metrics["old_candidate_path"] = str(old_path)
        metrics["label_encoding"] = {"1": "unchanged", "2": "added", "3": "removed"}
        paths.comparison_metrics_path().write_text(json.dumps(metrics, indent=2))
        return metrics

    def _save_binary(self, reference_node, mask_array, path):
        import numpy as np

        # Clone the reference (CTA volume) so geometry (IJKToRAS, spacing, origin)
        # is copied exactly, then overwrite the voxels with the binary mask and
        # save as a labelmap-style NIfTI.
        binary = slicer.modules.volumes.logic().CloneVolume(
            slicer.mrmlScene, reference_node, "laa_binary_tmp"
        )
        try:
            slicer.util.updateVolumeFromArray(binary, mask_array.astype(np.uint8))
            slicer.util.saveNode(binary, str(path))
        finally:
            slicer.mrmlScene.RemoveNode(binary)

    def node_path(self, node):
        return _node_path(node)


# ----------------------------------------------------------------------------
# Module-level Slicer helpers
# ----------------------------------------------------------------------------


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


def _confidence_spin():
    spin = qt.QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.1)
    spin.setValue(0.5)
    return spin


def _node_path(node):
    if node is None:
        return None
    storage = node.GetStorageNode() if hasattr(node, "GetStorageNode") else None
    if storage and storage.GetFileName():
        return storage.GetFileName()
    return None


class LAACompletionAssistantTest(ScriptedLoadableModuleTest):
    """Self-test stub so 'Reload and Test' does not error.

    The real logic is unit-tested offline in
    subprojects/la_laa/tests/test_laa_annotation_core.py. This intentionally
    does NOT clear the MRML scene, so clicking "Reload and Test" mid-annotation
    will not discard the loaded case.
    """

    def runTest(self):
        self.delayDisplay(
            "LAA Completion Assistant: no GUI self-test (core logic tested offline). "
            "Scene preserved."
        )
