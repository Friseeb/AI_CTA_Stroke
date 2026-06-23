"""QC: Slicer-compatible multi-label NIfTI + color table + MRML scene generation.

Outputs (per run):
  combined_labels.nii.gz  — all segmentation labels merged into one integer label map
  combined_labels.ctbl    — Slicer / FreeSurfer-style color table
  scene.mrml              — 3D Slicer scene; double-click or File > Add Data to open
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .config import QCConfig
from .imaging_cache import label_array
from .logging_utils import get_logger

log = get_logger("qc")

_SLICER_CANDIDATES = [
    "/Applications/Slicer.app/Contents/MacOS/Slicer",
    "/usr/local/bin/Slicer",
    "/opt/Slicer/Slicer",
]


def _find_slicer() -> Optional[str]:
    # Check known 3D Slicer locations first — shutil.which("Slicer") can
    # find FSL's slicer binary which is not 3D Slicer.
    for p in _SLICER_CANDIDATES:
        if Path(p).exists():
            return p
    exe = shutil.which("Slicer")
    if exe:
        return exe
    return None


def open_slicer_scene(scene_path: Path) -> bool:
    """Launch 3D Slicer with the load_scene.py script alongside scene_path.

    Uses --python-script which is reliable across Slicer versions.
    Falls back to --python-code loadScene() if no load script exists.
    """
    exe = _find_slicer()
    if exe is None:
        log.warning("3D Slicer not found — cannot auto-open scene.")
        return False
    try:
        # Prefer the Python load script (sibling of scene.mrml)
        load_script = scene_path.parent / scene_path.name.replace("scene.mrml", "load_scene.py")
        if not load_script.exists():
            load_script = scene_path.with_suffix(".py").parent / "load_scene.py"

        if load_script.exists():
            cmd = [exe, "--python-script", str(load_script.resolve())]
        else:
            abs_path = str(scene_path.resolve()).replace("'", "\\'")
            cmd = [exe, "--python-code", f"slicer.util.loadScene(r'{abs_path}')"]

        subprocess.Popen(cmd)
        log.info("Launched Slicer: %s", " ".join(cmd))
        return True
    except Exception as exc:
        log.warning("Failed to launch 3D Slicer: %s", exc)
        return False

# ── Semantic color assignments ────────────────────────────────────────────────

_TOOTH_COLORS: dict[str, tuple[int, int, int]] = {
    # Upper right (FDI 11-18) — warm red → orange
    "11": (220, 50,  50),  "12": (222, 78,  40),  "13": (224, 108,  35),
    "14": (212, 140, 30),  "15": (200, 162,  30),  "16": (212, 132,  48),
    "17": (218, 100, 58),  "18": (218,  68,  68),
    # Upper left (FDI 21-28) — pink → magenta
    "21": (200,  50, 148), "22": (205,  48, 162), "23": (210,  42, 178),
    "24": (195,  36, 192), "25": (175,  46, 202), "26": (165,  58, 212),
    "27": (172,  68, 200), "28": (180,  82, 182),
    # Lower left (FDI 31-38) — teal → cyan
    "31": ( 38, 172, 192), "32": ( 32, 186, 186), "33": ( 28, 196, 174),
    "34": ( 34, 200, 158), "35": ( 44, 196, 142), "36": ( 54, 186, 128),
    "37": ( 64, 175, 138), "38": ( 70, 168, 155),
    # Lower right (FDI 41-48) — green → lime
    "41": ( 68, 148,  54), "42": ( 90, 164,  44), "43": (116, 176,  44),
    "44": (136, 176,  40), "45": (152, 166,  50), "46": (142, 156,  60),
    "47": (132, 150,  72), "48": (120, 154,  82),
}


def _label_color(name: str) -> tuple[int, int, int]:
    n = name.lower()
    if "upper_jawbone" in n:          return (255, 196, 140)
    if "lower_jawbone" in n:          return (240, 166, 110)
    if "pulp" in n:                   return (255, 214, 214)
    if "canal" in n:                  return (174,  98, 226)
    if "sinus" in n:                  return (144, 216, 255)
    if "pharynx" in n:                return (200, 162, 210)
    if "implant" in n:                return (196, 200, 200)
    if "crown" in n:                  return (226, 226, 184)
    if "bridge" in n:                 return (214, 214, 174)
    fdi_tag = n.split("fdi")[-1] if "fdi" in n else ""
    if fdi_tag and fdi_tag.isdigit() and len(fdi_tag) == 2:
        return _TOOTH_COLORS.get(fdi_tag, (128, 128, 128))
    return (128, 128, 128)


# ── Internal builders ─────────────────────────────────────────────────────────

def _merge_labels(
    reference: sitk.Image,
    label_files: dict[str, Path],
    label_id_map: dict[str, int],
) -> sitk.Image:
    """Merge binary label NIfTIs into a single integer label map in reference space.

    Labels that don't match the reference space are resampled using nearest-neighbor.
    """
    ref_arr = sitk.GetArrayFromImage(reference)
    arr = np.zeros(ref_arr.shape, dtype=np.int16)
    for name, lid in label_id_map.items():
        path = label_files.get(name)
        if path is None:
            continue
        try:
            label_arr = label_array(path)  # shared per-case cache (with features)
            if label_arr.shape != ref_arr.shape:
                # rare: label not in reference space — resample (read fresh)
                label_img = sitk.ReadImage(str(path))
                label_img = sitk.Resample(
                    label_img, reference,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0.0, label_img.GetPixelID(),
                )
                label_arr = sitk.GetArrayFromImage(label_img)
            arr[label_arr.astype(bool)] = lid
        except Exception as exc:
            log.warning("Skipping label %s for combined NIfTI: %s", name, exc)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference)
    return out


def _write_ctbl(path: Path, label_id_map: dict[str, int]) -> None:
    """Write a Slicer-compatible color table (FreeSurfer LUT format)."""
    lines = [
        "# Slicer color table — dental segmentation (RESEARCH PROTOTYPE)",
        "# value name R G B A",
        "0 Background 0 0 0 0",
    ]
    for name, lid in sorted(label_id_map.items(), key=lambda x: x[1]):
        r, g, b = _label_color(name)
        lines.append(f"{lid} {name} {r} {g} {b} 255")
    path.write_text("\n".join(lines) + "\n")


def _write_mrml(
    path: Path,
    cta_path: Path,
    labels_path: Path,
    ctbl_path: Path,
    n_labels: int,
    scene_name: str = "DentalSegmentation",
    window: float = 1500.0,
    level: float = 400.0,
) -> None:
    """Write a minimal 3D Slicer MRML scene (absolute paths for portability).

    Note: double-click opening works but colors load via load_scene.py which
    is more reliable. The MRML is provided as a fallback.
    """
    mrml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <MRML version="Slicer4.6.2" userTags="">
          <ColorTableStorage id="vtkMRMLColorTableStorageNode1"
            fileName="{ctbl_path.resolve()}" />
          <ColorTable id="vtkMRMLColorTableNode1"
            name="DentalColors" type="File"
            storageNodeRef="vtkMRMLColorTableStorageNode1" />

          <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode1"
            fileName="{cta_path.resolve()}" singleFile="1" />
          <ScalarVolumeDisplayNode id="vtkMRMLScalarVolumeDisplayNode1"
            window="{window:.0f}" level="{level:.0f}" autoWindowLevel="0" />
          <Volume id="vtkMRMLScalarVolumeNode1"
            name="CTA" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode1"
            displayNodeRef="vtkMRMLScalarVolumeDisplayNode1" />

          <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode2"
            fileName="{labels_path.resolve()}" singleFile="1" />
          <LabelMapVolumeDisplayNode id="vtkMRMLLabelMapVolumeDisplayNode1"
            colorNodeID="vtkMRMLColorTableNode1" visibility="1" />
          <LabelMapVolume id="vtkMRMLLabelMapVolumeNode1"
            name="{scene_name}" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode2"
            displayNodeRef="vtkMRMLLabelMapVolumeDisplayNode1" />
        </MRML>
        """)
    path.write_text(mrml)
    log.info("Slicer scene written: %s", path)


_FIDUCIAL_CATEGORIES: dict[str, dict] = {
    # Bright red — periapical lucency (low-HU shell at tooth apex; lucency on CTA)
    "periapical_lucency_candidate": {
        "display_name": "Periapical lucency (candidate)",
        "color":         (1.00, 0.20, 0.20),
        "glyph_scale":   2.5,
    },
    # Amber — periradicular bone-loss shells with low jawbone coverage
    "severe_periodontal_bone_loss_candidate": {
        "display_name": "Periodontal bone loss (candidate)",
        "color":         (1.00, 0.75, 0.10),
        "glyph_scale":   3.0,
    },
    # Bright cyan — root remnants (high-HU island without overlying crown)
    "root_remnant_candidate": {
        "display_name": "Root remnant (candidate)",
        "color":         (0.10, 0.85, 0.95),
        "glyph_scale":   2.5,
    },
}


def _extract_pathology_fiducials(
    features_path: Path,
    geometry_image: sitk.Image,
) -> dict[str, list[tuple[str, tuple[float, float, float]]]]:
    """Convert candidate pathology entries → (label, RAS_xyz) Slicer fiducials.

    - `geometry_image` must share voxel-grid geometry with the per-label NIfTIs
      that `features.py` used to compute `location_voxel` (z, y, x in numpy order).
    - LPS → RAS sign flip: Slicer's markups API uses RAS regardless of NIfTI LPS.
    """
    if not features_path.exists():
        return {}
    try:
        data = json.loads(features_path.read_text())
    except Exception as exc:
        log.warning("Could not parse %s for pathology fiducials: %s", features_path, exc)
        return {}

    markers = data.get("candidate_markers", {})

    def voxel_zyx_to_ras(loc):
        if not loc or len(loc) != 3:
            return None
        ijk_xyz = [int(loc[2]), int(loc[1]), int(loc[0])]
        lps = geometry_image.TransformIndexToPhysicalPoint(ijk_xyz)
        return (-lps[0], -lps[1], lps[2])

    out: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}
    for category in _FIDUCIAL_CATEGORIES:
        entries = markers.get(category, [])
        if not isinstance(entries, list):
            continue
        points: list[tuple[str, tuple[float, float, float]]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            ras = voxel_zyx_to_ras(entry.get("location_voxel"))
            if ras is None:
                continue
            fdi = entry.get("fdi_id") or entry.get("fdi_label", "")
            if fdi and fdi != "?":
                label = f"{idx + 1}_{fdi}"
            else:
                label = f"{idx + 1}"
            points.append((label, ras))
        if points:
            out[category] = points
    return out


def _write_load_script(
    path: Path,
    cta_path: Path,
    labels_path: Path,
    ctbl_path: Path,
    window: float = 1500.0,
    level: float = 400.0,
    segmentation_name: str = "Dental_Segmentation",
    fiducials: Optional[dict[str, list[tuple[str, tuple[float, float, float]]]]] = None,
) -> None:
    """Write a Slicer Python script that loads CTA + labels as a Segmentation,
    plus one Markups Fiducial node per pathology category.

    Run with:  Slicer --python-script /path/to/load_scene.py
    """
    script = textwrap.dedent(f"""\
        # cta-dental QC scene loader — run inside 3D Slicer
        # RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS
        import slicer

        _cta    = r'{cta_path.resolve()}'
        _labels = r'{labels_path.resolve()}'
        _ctbl   = r'{ctbl_path.resolve()}'

        # CTA: scalar volume with fixed window/level for bone
        vol = slicer.util.loadVolume(_cta)
        dn  = vol.GetDisplayNode()
        dn.SetAutoWindowLevel(0)
        dn.SetWindow({window:.0f})
        dn.SetLevel({level:.0f})

        # Color table (named + colored entries per FDI tooth and structure)
        ctbl = slicer.util.loadColorTable(_ctbl)

        # Load the multi-label volume, bind the color table, and convert it
        # into a Segmentation so each label becomes a named/colored Segment.
        lbl_vol = slicer.util.loadLabelVolume(_labels)
        lbl_vol.GetDisplayNode().SetAndObserveColorNodeID(ctbl.GetID())

        seg = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", {segmentation_name!r}
        )
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            lbl_vol, seg
        )
        seg.CreateClosedSurfaceRepresentation()

        # Drop the intermediate labelmap — the Segmentation owns the data now.
        slicer.mrmlScene.RemoveNode(lbl_vol)
        """)

    if fiducials:
        script += textwrap.dedent("""

            # ── Pathology candidate fiducials ─────────────────────────────────
            # Each category becomes one Markups Fiducial node with a fixed color
            # and a control point per candidate finding (location from features).
            def _add_fiducials(name, color, glyph_scale, points):
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", name
                )
                node.CreateDefaultDisplayNodes()
                d = node.GetDisplayNode()
                d.SetSelectedColor(*color)
                d.SetColor(*color)
                d.SetGlyphScale(glyph_scale)
                d.SetTextScale(2.5)
                for label, ras in points:
                    node.AddControlPoint(list(ras), label)
                return node

            _total_pathology = 0
            """)
        for category, points in fiducials.items():
            meta = _FIDUCIAL_CATEGORIES[category]
            script += textwrap.dedent(f"""
                _add_fiducials(
                    {meta['display_name']!r},
                    {meta['color']!r},
                    {meta['glyph_scale']!r},
                    {points!r},
                )
                _total_pathology += {len(points)}
                """)
        script += textwrap.dedent("""
            print(f"[cta-dental] Added {_total_pathology} pathology candidate fiducials.")
            """)

    script += textwrap.dedent("""

        slicer.util.resetSliceViews()
        print(f"[cta-dental] Loaded Segmentation with "
              f"{seg.GetSegmentation().GetNumberOfSegments()} segments.")
        """)
    path.write_text(script)
    log.info("Slicer load script written: %s", path)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_roi_qc(
    image: sitk.Image,
    mask: Optional[sitk.Image],
    out_dir: Path,
    cfg: QCConfig,
    spacing_info: Optional[str] = None,
    bbox_info: Optional[dict] = None,
    image_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Write ROI mask as a Slicer label map + MRML scene."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if mask is None:
        log.warning("No ROI mask for QC — skipping scene.")
        return paths

    # ROI mask as single-label NIfTI
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.int16)
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(mask)
    roi_label_path = out_dir / "roi_mask_label.nii.gz"
    sitk.WriteImage(mask_img, str(roi_label_path))
    paths["roi_mask_label"] = roi_label_path

    ctbl_path = out_dir / "roi_mask.ctbl"
    ctbl_path.write_text(
        "# Slicer color table — dentition ROI\n"
        "# value name R G B A\n"
        "0 Background 0 0 0 0\n"
        "1 Dentition_ROI 255 180 30 200\n"
    )
    paths["roi_ctbl"] = ctbl_path

    # CTA: write to qc dir only if no external path provided
    if image_path is None:
        image_path = out_dir / "cta.nii.gz"
        sitk.WriteImage(image, str(image_path))
        paths["cta"] = image_path

    mrml_path = out_dir / "roi_scene.mrml"
    _write_mrml(
        path=mrml_path,
        cta_path=image_path,
        labels_path=roi_label_path,
        ctbl_path=ctbl_path,
        n_labels=1,
        scene_name="Dentition_ROI",
        window=cfg.window_width_hu,
        level=cfg.window_center_hu,
    )
    paths["roi_scene"] = mrml_path

    load_script = out_dir / "load_roi_scene.py"
    _write_load_script(
        path=load_script,
        cta_path=image_path,
        labels_path=roi_label_path,
        ctbl_path=ctbl_path,
        window=cfg.window_width_hu,
        level=cfg.window_center_hu,
        segmentation_name="Dentition_ROI",
    )
    paths["roi_load_script"] = load_script
    return paths


def generate_segmentation_qc(
    image: sitk.Image,
    label_files: dict[str, Path],
    out_dir: Path,
    cfg: QCConfig,
    segmenter_name: str = "unknown",
    image_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Merge segmentation labels and write a Slicer scene (combined NIfTI + ctbl + MRML).

    If `features_path` points to a candidate_features.json, each pathology
    category that carries a `location_voxel` becomes a Markups Fiducial node
    in the Slicer scene.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not label_files:
        log.warning("No label files for segmentation QC.")
        return paths

    # Deterministic label ID assignment: sort by name
    label_id_map = {name: i + 1 for i, name in enumerate(sorted(label_files))}

    combined = _merge_labels(image, label_files, label_id_map)
    combined_path = out_dir / "combined_labels.nii.gz"
    sitk.WriteImage(combined, str(combined_path))
    paths["combined_labels"] = combined_path

    ctbl_path = out_dir / "combined_labels.ctbl"
    _write_ctbl(ctbl_path, label_id_map)
    paths["ctbl"] = ctbl_path

    if image_path is None:
        image_path = out_dir / "cta.nii.gz"
        sitk.WriteImage(image, str(image_path))
        paths["cta"] = image_path

    mrml_path = out_dir / "scene.mrml"
    _write_mrml(
        path=mrml_path,
        cta_path=image_path,
        labels_path=combined_path,
        ctbl_path=ctbl_path,
        n_labels=len(label_id_map),
        scene_name=f"Dental_{segmenter_name}",
        window=cfg.window_width_hu,
        level=cfg.window_center_hu,
    )
    paths["scene"] = mrml_path

    # Pathology fiducials: voxel locations from features.json are referenced to
    # the per-label NIfTI geometry, so pass the first label file as the
    # geometry source. After our origin patch all labels share that geometry.
    fiducials = None
    if features_path is not None and label_files:
        any_label = next(iter(sorted(label_files.values())))
        try:
            geom = sitk.ReadImage(str(any_label))
            fiducials = _extract_pathology_fiducials(features_path, geom) or None
        except Exception as exc:
            log.warning("Could not extract pathology fiducials: %s", exc)

    load_script = out_dir / "load_scene.py"
    _write_load_script(
        path=load_script,
        cta_path=image_path,
        labels_path=combined_path,
        ctbl_path=ctbl_path,
        window=cfg.window_width_hu,
        level=cfg.window_center_hu,
        segmentation_name=f"Dental_{segmenter_name}",
        fiducials=fiducials,
    )
    paths["load_script"] = load_script
    if fiducials:
        n_fid = sum(len(v) for v in fiducials.values())
        log.info("Pathology fiducials embedded: %d across %d categories.",
                 n_fid, len(fiducials))

    log.info("Segmentation Slicer scene: %s  (%d labels)", mrml_path, len(label_id_map))
    return paths


def generate_failure_qc(
    image: Optional[sitk.Image],
    reason: str,
    out_dir: Path,
    cfg: QCConfig,
) -> Path:
    """Write a failure JSON (no valid data to build a Slicer scene from)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "roi_failure.json"
    p.write_text(json.dumps({
        "status": "roi_detection_failed",
        "reason": reason,
        "disclaimer": "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS",
    }, indent=2))
    log.warning("ROI failure QC written: %s", p)
    return p


def generate_qc_summary_json(
    out_dir: Path,
    qc_paths: dict[str, Path],
    warnings: list[str],
    roi_quality: str,
    segmenter: str,
) -> Path:
    open_in_slicer = str(
        qc_paths.get("scene", qc_paths.get("roi_scene", ""))
    )
    summary = {
        "roi_quality": roi_quality,
        "segmenter": segmenter,
        "slicer_files": {k: str(v) for k, v in qc_paths.items()},
        "open_in_slicer": open_in_slicer,
        "warnings": warnings,
        "disclaimer": "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS",
    }
    p = out_dir / "qc_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2))
    return p
