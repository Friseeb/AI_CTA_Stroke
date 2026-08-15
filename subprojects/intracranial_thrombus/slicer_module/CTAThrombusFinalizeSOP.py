import json
import os
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

        dirLayout = qt.QHBoxLayout()
        dirLabel = qt.QLabel("Output dir:")
        self.outputDirText = qt.QLineEdit()
        self.outputDirText.setPlaceholderText("Leave blank to auto-detect")
        dirLayout.addWidget(dirLabel)
        dirLayout.addWidget(self.outputDirText)
        self.layout.addLayout(dirLayout)

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
            out_dir = self.outputDirText.text.strip() or None
            log = self.logic.run(output_dir=out_dir)
            self.logBox.append(json.dumps(log, indent=2))
        except Exception as exc:
            self.logBox.append(f"ERROR: {exc}")


class CTAThrombusFinalizeSOP_Logic(ScriptedLoadableModuleLogic):

    def run(self, annotation: dict | None = None, output_dir: str | None = None) -> dict:
        self._log("Starting Thrombus Finalize SOP...")
        cta   = self._find_cta()
        seg   = self._find_segmentation()
        self._harden_transforms(seg)

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
        self._mark_worklist_done(case_id, out_label)
        self._log(f"Saved: {out_label}")
        self._post_save_check(out_label, cta, metrics, annotation)
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

    def _harden_transforms(self, seg):
        # Only harden the segmentation node, not the CTA volume (that causes
        # a VTK segfault on NIfTI-loaded scenes). The segmentation harden is
        # safe and necessary when the user applied an IS rotation transform.
        try:
            if seg.GetParentTransformNode() is not None:
                seg.HardenTransform()
                self._log("Segmentation transform hardened")
            else:
                self._log("No transform on segmentation — nothing to harden")
        except Exception as e:
            self._log(f"[WARN] Could not harden segmentation transform: {e}")

    def _export_segmentation(self, seg, cta):
        tmp = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "thrombus_export"
        )
        logic = slicer.modules.segmentations.logic()
        # Always export with the CTA as explicit reference — this is the only
        # method that guarantees voxel positions match the arterial grid.
        # No fallbacks without a reference: a wrong-geometry export is worse
        # than a hard failure.
        logic.ExportVisibleSegmentsToLabelmapNode(seg, tmp, cta)
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
        m = vtk.vtkMatrix4x4()
        cta.GetIJKToRASMatrix(m)
        labelmap.SetIJKToRASMatrix(m)
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
        # Geometry is already guaranteed correct by _export_segmentation +
        # _force_geometry. Save directly — no re-resample that could shift voxels.
        slicer.util.saveNode(labelmap, str(out_path))

    def _node_path(self, node):
        s = node.GetStorageNode()
        return s.GetFileName() if s and s.GetFileName() else None

    def _infer_case_id(self, cta):
        name = cta.GetName()
        m = re.search(r"sub-\d+", name)
        return m.group(0) if m else re.sub(r"\W+", "_", name).strip("_")

    def _infer_output_dir(self, cta):
        p = self._node_path(cta)
        desktop_base = Path.home() / "Desktop" / "CTA_intracraneal_segmentation"
        if p and "daylightbids" in p:
            d = desktop_base / "DAYLIGHT"
            d.mkdir(parents=True, exist_ok=True)
            return d
        d = desktop_base / "SLAO"
        if d.exists():
            return d
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

    # Worklists conocidos: (ruta, nombre del campo "hecho", campo ruta de mascara)
    # El primero es el historico de DAYLIGHT (campo 'done'); el segundo es
    # SLAOEVT_Nueva (campo 'segmented'). Se actualizan los que existan y
    # contengan ese sub_id -- antes solo se tocaba el de /tmp, por lo que los
    # casos de SLAOEVT nunca se marcaban solos y habia que llamar mark_done()
    # a mano, con el riesgo de desincronizacion que eso implica.
    WORKLISTS = [
        (Path("/tmp/thrombus_worklist.json"), "done", None),
        # ruta por defecto, sobreescribible con THROMBUS_WORKLIST. Ademas se
        # buscan worklists junto al directorio de salida (_candidate_worklists).
        (Path(os.environ.get("THROMBUS_WORKLIST",
              os.path.expanduser("~/Desktop/INTRACRANEAL_THROMBUS/CTA_intracraneal_segmentation/slaoevt_worklist_v2.json"))),
         "segmented", "thrombus_seg"),
    ]

    def _candidate_worklists(self, out_label: Path | None):
        """Rutas fijas conocidas + busqueda portable junto al directorio de
        salida. Lo segundo es lo que permite que esto funcione en OTRO
        ordenador, donde la ruta del Desktop de arriba no existe: basta con
        que el worklist *.json este en la carpeta de salida o en su padre."""
        cands = list(self.WORKLISTS)
        if out_label is not None:
            for d in (out_label.parent, out_label.parent.parent):
                try:
                    for p in sorted(d.glob("*worklist*.json")):
                        cands.append((p, "segmented", "thrombus_seg"))
                except Exception:
                    pass
        seen, uniq = set(), []
        for c in cands:
            if c[0] not in seen:
                seen.add(c[0])
                uniq.append(c)
        return uniq

    def _mark_worklist_done(self, case_id: str, out_label: Path | None = None):
        sub_id = case_id.replace("sub-", "")
        for worklist_path, done_field, path_field in self._candidate_worklists(out_label):
            if not worklist_path.exists():
                continue
            try:
                worklist = json.loads(worklist_path.read_text())
                hit = False
                for entry in worklist:
                    if str(entry.get("sub_id", "")) == sub_id:
                        entry[done_field] = True
                        if path_field and out_label is not None:
                            entry[path_field] = str(out_label)
                        hit = True
                        break
                if not hit:
                    continue
                worklist_path.write_text(json.dumps(worklist, indent=2))
                done = sum(1 for e in worklist if e.get(done_field))
                total = sum(1 for e in worklist if not e.get("excluded"))
                self._log(f"Worklist [{worklist_path.name}]: {done}/{total} done")
            except Exception as exc:
                self._log(f"Could not update worklist {worklist_path.name}: {exc}")

    def _post_save_check(self, out_label: Path, cta, metrics: dict, annotation: dict | None):
        """Verificacion automatica tras guardar. Imprime en el log del modulo:
        vóxeles archivo vs log, solapamiento con el CTA, HU medio y rango, y si
        proximal_tip_ras_mm quedo relleno. Detecta al momento mascaras cruzadas
        entre pacientes (el fallo que aparecio 3 veces en DAYLIGHT)."""
        self._log("--- verificacion automatica ---")
        try:
            if sitk is None:
                self._log("  SimpleITK no disponible, verificacion omitida")
                return

            log_vox = int(metrics.get("thrombus", {}).get("voxels", -1))
            msk = sitk.ReadImage(str(out_label))
            arr = sitk.GetArrayFromImage(msk)
            file_vox = int((arr == 1).sum())
            ok_vox = (file_vox == log_vox)
            self._log(f"  voxeles archivo={file_vox} log={log_vox} -> {'OK' if ok_vox else '*** MISMATCH ***'}")

            cta_img = sitkUtils.PullVolumeFromSlicer(cta) if sitkUtils else None
            if cta_img is not None:
                res = sitk.Resample(msk, cta_img, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, 0, msk.GetPixelID())
                ra = sitk.GetArrayFromImage(res)
                n_res = int((ra > 0).sum())
                ratio = (n_res / file_vox) if file_vox else 0.0
                flag = "OK" if 0.7 <= ratio <= 1.3 else "*** NO ALINEADA CON SU CTA ***"
                self._log(f"  solapamiento con el CTA: {n_res} vox (ratio {ratio:.3f}) -> {flag}")

                idx = np.argwhere(ra > 0)
                if len(idx):
                    size = cta_img.GetSize()
                    c = idx.mean(axis=0)  # z,y,x
                    inside = (0 <= c[2] < size[0]) and (0 <= c[1] < size[1]) and (0 <= c[0] < size[2])
                    self._log(f"  centroide dentro del volumen: {'SI' if inside else '*** NO -- REVISAR ***'}")

                    cta_arr = sitk.GetArrayFromImage(cta_img).astype(float)
                    vals = cta_arr[ra > 0]
                    if vals.size:
                        mean_hu = float(vals.mean())
                        rng = "40-90 esperado"
                        flag_hu = "OK" if 40 <= mean_hu <= 90 else f"*** FUERA DE RANGO ({rng}) ***"
                        self._log(f"  HU media={mean_hu:.1f} mediana={float(np.median(vals)):.1f} "
                                  f"p5={float(np.percentile(vals,5)):.0f} p95={float(np.percentile(vals,95)):.0f} -> {flag_hu}")

            tip = (annotation or {}).get("proximal_tip_ras_mm")
            self._log(f"  proximal_tip_ras_mm: {'OK ' + str(tip) if tip else '*** VACIO -- pulsa Place Proximal Tip Fiducial ***'}")

            for f in ("vessel", "side"):
                v = (annotation or {}).get(f)
                if not v:
                    self._log(f"  {f}: *** VACIO ***")
        except Exception as exc:
            self._log(f"  verificacion fallo: {exc}")

    def _cleanup_nodes(self, nodes):
        for n in nodes:
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass
