"""Generate a per-case 3D Slicer Python loader for the QC scene.

Follows the same convention as the aorta_cta_radiomics and dental subprojects:
the orchestrator writes a `<case_id>_load_qc_in_slicer.py` script next to the
saved masks; opening it from a shell or `--python-script` clears the Slicer
scene, loads the CTA with fixed W/L, and converts each labelmap into a
`vtkMRMLSegmentationNode` with per-category colors and opacities.

The script also appends a `<case_id>_slicer_loader_status.txt` next to itself
so headless runs can be audited without parsing Slicer stdout.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .logging_utils import get_logger
from .types import AirwayMaskInfo

log = get_logger("qc_slicer")


# Category → (RGB 0..1, default 3D opacity, default 2D fill opacity, default 2D outline)
_CATEGORY_DEFAULTS: dict[str, tuple[tuple[float, float, float], float, float, float]] = {
    "airway":              ((0.00, 0.85, 1.00), 0.95, 0.30, 1.00),
    "body":                ((0.70, 0.70, 0.70), 0.10, 0.05, 0.50),
    "fat_subcutaneous":    ((1.00, 0.60, 0.00), 0.50, 0.35, 1.00),
    "fat_deep":            ((1.00, 0.40, 0.00), 0.50, 0.35, 1.00),
    "fat_deep_peripharyngeal": ((0.40, 0.95, 0.65), 0.75, 0.45, 1.00),
    "fat_parapharyngeal":  ((1.00, 0.95, 0.00), 0.85, 0.55, 1.00),
    "fat_parapharyngeal_rg": ((1.00, 0.82, 0.00), 0.90, 0.60, 1.00),
    "fat_parapharyngeal_ss": ((1.00, 0.68, 0.00), 0.90, 0.60, 1.00),
    "fat_retropharyngeal": ((0.95, 0.00, 0.95), 0.85, 0.55, 1.00),
    "mandible":            ((0.95, 0.95, 0.80), 0.70, 0.45, 1.00),
    "prevertebral":        ((0.55, 0.55, 1.00), 0.45, 0.25, 1.00),
    "tongue":              ((0.80, 0.20, 0.20), 0.70, 0.40, 1.00),
    "soft_palate":         ((0.70, 0.25, 0.85), 0.70, 0.40, 1.00),
    "fat_cervical_total":  ((0.95, 0.55, 0.00), 0.15, 0.05, 0.50),
}

# Default mask label / category roster — name → (display label, category)
_DEFAULT_ROSTER: list[tuple[str, str, str]] = [
    # (mask_basename_without_extension, display_label, category)
    ("mask_airway",                   "Airway",      "airway"),
    ("mask_body",                     "Body",        "body"),
    ("mask_fat_cervical_total",       "All cerv fat", "fat_cervical_total"),
    ("mask_fat_cervical_subcutaneous", "SubQ fat",   "fat_subcutaneous"),
    ("mask_fat_cervical_deep",        "Deep fat",    "fat_deep"),
    ("mask_fat_deep_peripharyngeal",  "Periph fat",  "fat_deep_peripharyngeal"),
    ("mask_fat_retropharyngeal",      "RP fat",      "fat_retropharyngeal"),
    ("mask_prevertebral",             "C-spine",     "prevertebral"),
    ("mask_mandible",                 "Mandible",    "mandible"),
    ("mask_tongue",                   "Tongue",      "tongue"),
    ("mask_tongue_posterior",         "Tongue post", "tongue"),
    ("mask_tongue_base",              "Tongue base", "tongue"),
    ("mask_soft_palate",              "Soft palate", "soft_palate"),
    ("mask_uvula",                    "Uvula",       "soft_palate"),
    ("mask_palatine_tonsil_left",     "Tonsil L",    "soft_palate"),
    ("mask_palatine_tonsil_right",    "Tonsil R",    "soft_palate"),
]


@dataclass
class MaskSpec:
    path: str
    label: str
    category: str

    def to_dict(self) -> dict:
        rgb, op3d, op2dfill, op2doutline = _CATEGORY_DEFAULTS.get(
            self.category, ((0.5, 0.5, 0.5), 0.6, 0.35, 1.0))
        return {
            "path": self.path,
            "label": self.label,
            "category": self.category,
            "color": list(rgb),
            "opacity": op3d,
            "fill_opacity": op2dfill,
            "outline_opacity": op2doutline,
        }


def write_slicer_loader(
    case_id: str,
    image_path: Path,
    case_dir: Path,
    out_script: Path,
    masks: Optional[Iterable[MaskSpec]] = None,
    window: float = 350.0,
    level: float = 40.0,
    min_csa_landmark_ras: Optional[tuple[float, float, float]] = None,
) -> Path:
    """Write the per-case Slicer loader script.

    Args:
        case_id: human-facing identifier (becomes node prefixes + status filename).
        image_path: absolute path to the CTA NIfTI to load as the background volume.
        case_dir: directory containing the `mask_*.nii.gz` files; used to auto-
            discover the default roster when `masks` is None.
        out_script: where to write the generated `.py`.
        masks: optional explicit roster. If None, the function looks for the
            default mask basenames inside `case_dir` and includes only those
            that exist.
        window / level: soft-tissue neck window defaults
            (center 40 HU, width 350 HU — adjust if visualising bone too).
        min_csa_landmark_ras: optional (R, A, S) coords for a "min airway CSA"
            fiducial. None → no fiducial.

    Returns the path the script was written to.
    """
    out_script.parent.mkdir(parents=True, exist_ok=True)

    if masks is None:
        roster = []
        for basename, label, category in _DEFAULT_ROSTER:
            p = case_dir / f"{basename}.nii.gz"
            if p.is_file():
                roster.append(MaskSpec(str(p.resolve()), label, category))
        masks = roster

    mask_dicts = [m.to_dict() for m in masks]
    masks_json = json.dumps(mask_dicts, indent=2)

    landmark_block = ""
    if min_csa_landmark_ras is not None:
        ras = list(float(v) for v in min_csa_landmark_ras)
        landmark_block = (
            f"\n# Minimum-CSA fiducial\n"
            f"_fid = slicer.mrmlScene.AddNewNodeByClass("
            f"'vtkMRMLMarkupsFiducialNode', CASE_ID + '_min_csa')\n"
            f"_fid.CreateDefaultDisplayNodes()\n"
            f"_fid.GetDisplayNode().SetSelectedColor(0.0, 1.0, 1.0)\n"
            f"_fid.GetDisplayNode().SetGlyphScale(2.5)\n"
            f"_fid.GetDisplayNode().SetTextScale(3.0)\n"
            f"_fid.AddControlPoint({ras!r}, 'min CSA')\n"
            f"log_status('Added min-CSA fiducial at RAS', {ras!r})\n"
        )

    script = SCRIPT_TEMPLATE.format(
        case_id=repr(case_id),
        image_path=repr(str(Path(image_path).resolve())),
        masks_json=masks_json,
        window=float(window),
        level=float(level),
        landmark_block=landmark_block,
    )
    out_script.write_text(script)
    log.info("Slicer QC loader written: %s (masks=%d)", out_script, len(mask_dicts))
    return out_script


# ---------------------------------------------------------------------------
# Slicer launcher (best-effort — same convention as the dental subproject).
# ---------------------------------------------------------------------------

_SLICER_CANDIDATES = [
    "/Applications/Slicer.app/Contents/MacOS/Slicer",
    "/usr/local/bin/Slicer",
    "/opt/Slicer/Slicer",
]


def find_slicer() -> Optional[str]:
    """Locate a real 3D Slicer install.

    Checks well-known absolute paths first because `shutil.which('Slicer')`
    can return FSL's slicer binary, which cannot run Python scripts.
    """
    for p in _SLICER_CANDIDATES:
        if Path(p).exists():
            return p
    exe = shutil.which("Slicer")
    return exe if exe else None


def open_in_slicer(script_path: Path) -> bool:
    """Launch 3D Slicer with the generated loader script (non-blocking)."""
    exe = find_slicer()
    if exe is None:
        log.warning("3D Slicer not found; skipping auto-open.")
        return False
    try:
        subprocess.Popen([exe, "--python-script", str(Path(script_path).resolve())])
        log.info("Launched Slicer: %s --python-script %s", exe, script_path)
        return True
    except Exception as exc:
        log.warning("Failed to launch Slicer: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Script template — kept literal so the generated file is easy to audit.
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATE = '''\
# Auto-generated by stroke_cta_osa.qc_slicer — RESEARCH PROTOTYPE
# RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS.
# Run inside 3D Slicer:  Slicer --python-script <this_file>
from pathlib import Path

import slicer

slicer.mrmlScene.Clear(0)

CASE_ID    = {case_id}
IMAGE_PATH = {image_path}
WINDOW     = {window}
LEVEL      = {level}
MASKS      = {masks_json}

SCRIPT_PATH = Path(globals().get("__file__", ".")).resolve()
STATUS_DIR  = SCRIPT_PATH.parent if SCRIPT_PATH.name != "." else Path.cwd()
STATUS_PATH = str(STATUS_DIR / (CASE_ID + "_slicer_loader_status.txt"))


def log_status(*parts):
    text = " ".join(str(part) for part in parts)
    print(text)
    try:
        with open(STATUS_PATH, "a", encoding="utf-8") as handle:
            handle.write(text + "\\n")
    except Exception:
        pass


try:
    Path(STATUS_PATH).write_text("", encoding="utf-8")
except Exception:
    pass

# --- 1. CTA volume ---------------------------------------------------------
volume_node = slicer.util.loadVolume(IMAGE_PATH, {{"name": CASE_ID + "_CTA"}})
if volume_node:
    d = volume_node.GetDisplayNode()
    if d:
        d.AutoWindowLevelOff()
        d.SetWindow(WINDOW)
        d.SetLevel(LEVEL)
    try:
        slicer.util.setSliceViewerLayers(background=volume_node, fit=True)
    except TypeError:
        slicer.util.setSliceViewerLayers(background=volume_node)
    log_status("Loaded CTA:", IMAGE_PATH)
else:
    log_status("Could not load CTA:", IMAGE_PATH)

# --- 2. Labelmaps → Segmentations -----------------------------------------
seg_logic = slicer.modules.segmentations.logic()
loaded_segments = 0
for spec in MASKS:
    label_node = slicer.util.loadLabelVolume(
        spec["path"], {{"name": spec["label"] + "_label"}}
    )
    if not label_node:
        log_status("Could not load mask:", spec["path"])
        continue
    label_node.SetDisplayVisibility(False)

    seg_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", CASE_ID + "_" + spec["label"]
    )
    seg_node.CreateDefaultDisplayNodes()
    seg_node.SetDisplayVisibility(True)
    if volume_node:
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    seg_logic.ImportLabelmapToSegmentationNode(label_node, seg_node)

    disp = seg_node.GetDisplayNode()
    if disp:
        disp.SetVisibility(True)
        disp.SetOpacity3D(float(spec["opacity"]))
        disp.SetVisibility2DFill(True)
        disp.SetVisibility2DOutline(True)
        disp.SetOpacity2DFill(float(spec["fill_opacity"]))
        disp.SetOpacity2DOutline(float(spec["outline_opacity"]))

    segmentation = seg_node.GetSegmentation()
    n = segmentation.GetNumberOfSegments()
    for idx in range(n):
        seg = segmentation.GetNthSegment(idx)
        sid = segmentation.GetNthSegmentID(idx)
        seg.SetName(spec["label"] if n == 1 else str(idx + 1).zfill(3))
        seg.SetColor(
            float(spec["color"][0]),
            float(spec["color"][1]),
            float(spec["color"][2]),
        )
        if disp:
            disp.SetSegmentVisibility(sid, True)
            disp.SetSegmentOpacity3D(sid, float(spec["opacity"]))
            disp.SetSegmentOpacity2DFill(sid, float(spec["fill_opacity"]))
            disp.SetSegmentOpacity2DOutline(sid, float(spec["outline_opacity"]))

    # Build closed surfaces so segments render in the 3D view.
    try:
        seg_node.CreateClosedSurfaceRepresentation()
    except Exception:
        pass

    segmentation.Modified()
    seg_node.Modified()
    if disp:
        disp.Modified()
    loaded_segments += n
    log_status("Loaded mask:", spec["label"], "segments:", n, "path:", spec["path"])
    slicer.mrmlScene.RemoveNode(label_node)

# --- 3. Layout + view fitting ---------------------------------------------
try:
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
except Exception:
    pass
try:
    slicer.util.setSliceViewerLayers(background=volume_node, fit=True)
except Exception:
    pass
try:
    slicer.util.resetSliceViews()
except Exception:
    pass
try:
    slicer.util.selectModule("Segmentations")
except Exception:
    pass
{landmark_block}
slicer.app.processEvents()
log_status(
    "Loaded QC scene for", CASE_ID,
    "with", len(MASKS), "masks and", loaded_segments, "segments",
)
'''
