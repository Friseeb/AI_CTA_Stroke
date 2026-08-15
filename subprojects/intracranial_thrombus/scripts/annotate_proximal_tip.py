#!/usr/bin/env python-real
"""
Anotación CIEGA del extremo proximal del trombo, para validar el criterio
automático de orientación del eje axial.

Se ejecuta DENTRO de la consola Python de 3D Slicer:

    exec(open('/home/fridmans/Documents/pwd/AI_CTA_Stroke/subprojects/intracranial_thrombus/scripts/annotate_proximal_tip.py').read())

Luego, por cada caso:

    load_next()        # carga el siguiente pendiente (CTA + máscara en rojo)
    #  ... colocar UN fiducial en el extremo PROXIMAL ...
    save_and_next()    # guarda el punto y carga el siguiente

Otros comandos:
    status()           # cuántos hechos / pendientes
    skip("motivo")     # marcar el caso actual como no anotable y pasar
    reload_current()   # recargar el caso actual sin guardar

POR QUÉ NO SE USA EL SOP COMPLETO DE CTAThrombusDemarcation
-----------------------------------------------------------
Ese módulo YA tiene el botón "Place Proximal Tip Fiducial" y guarda
`proximal_tip_ras_mm` en el log -- el mecanismo existe, simplemente no se usó.
PERO su botón de guardar reescribe `sub-XXX_thrombus_clean_log.json` Y
`sub-XXX_thrombus_clean.nii.gz` a partir del estado actual de la escena. Usarlo
solo para añadir un fiducial obligaría a re-segmentar y **sobrescribiría las
máscaras existentes**, invalidando toda la extracción axial ya hecha.

Este script NO toca ni el log ni la máscara: escribe únicamente en
`output/v2/anotacion_extremo_proximal.csv` (columnas `anotado`, `fecha`,
`notas`, y las coordenadas RAS). Es reversible y no destructivo.

CEGAMIENTO: el CSV de trabajo NO contiene la orientación que dedujo el criterio
automático. No la mires antes de anotar (está en
thrombus_axial_topography.csv), o la validación pierde su valor.
"""
import csv
import os
from datetime import datetime

# Este script se ejecuta con exec() en la consola de Slicer, donde __file__ no
# existe: la ruta se toma de THROMBUS_REPO si esta definida, y si no del valor
# por defecto del equipo donde se desarrollo.
_REPO = os.environ.get("THROMBUS_REPO", os.path.expanduser("~/Documents/pwd/AI_CTA_Stroke"))
_BASE = os.path.join(_REPO, "subprojects/intracranial_thrombus/output/v2") + os.sep
# Se recorren en orden: primero el lote de validación (30, ya hecho), luego la
# ampliación (54). load_next() salta lo que ya tiene `anotado`, así que los 30
# hechos no se vuelven a ofrecer.
CSV_PATHS = [_BASE + "anotacion_extremo_proximal.csv",
             _BASE + "anotacion_extremo_proximal_ampliacion.csv",
             # lote 3: los 36 que faltaban para completar los 121 pacientes
             # unicos (SLAO y SLAOEVT no entraron en los dos lotes anteriores,
             # mas 15 de extended/standard). Los 5 de SLAOEVT que ya tenian el
             # tip en su log de segmentacion vienen pre-rellenados.
             _BASE + "anotacion_extremo_proximal_lote3.csv"]

_current = {"row": None, "index": None, "csv": None}


def _read_rows(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd), rd.fieldnames


def _write_rows(path, rows, fields):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)      # atómico: nunca deja el CSV a medias


def status():
    tot_d = tot_p = 0
    for path in CSV_PATHS:
        rows, _ = _read_rows(path)
        if not rows:
            continue
        d = sum(1 for r in rows if (r.get("anotado") or "").strip())
        tot_d += d
        tot_p += len(rows) - d
        print(f"  {os.path.basename(path):45s} {d}/{len(rows)}")
    print(f"TOTAL anotados: {tot_d}   Pendientes: {tot_p}")
    return tot_d, tot_p


def _clear_scene():
    import slicer
    slicer.mrmlScene.Clear(0)


def load_next():
    """Carga el siguiente caso pendiente: CTA de fondo + máscara en rojo."""
    import slicer
    pend, csv_path, rows = None, None, None
    for path in CSV_PATHS:
        rr, _ = _read_rows(path)
        p = [(i, r) for i, r in enumerate(rr)
             if not (r.get("anotado") or "").strip()]
        if p:
            pend, csv_path, rows = p, path, rr
            break
    if not pend:
        print("No quedan casos pendientes en ninguna lista.")
        print("Ejecuta:  python3 comparar_orientacion.py")
        print("     y:   python3 recalcular_orientadas.py")
        return
    i, row = pend[0]
    _current["row"], _current["index"], _current["csv"] = row, i, csv_path
    n_done = len(rows) - len(pend)
    _clear_scene()

    cta = slicer.util.loadVolume(row["ruta_cta"])
    seg = slicer.util.loadLabelVolume(row["ruta_mascara"])

    # máscara en rojo, semitransparente, sobre el CTA
    slicer.util.setSliceViewerLayers(background=cta, label=seg, labelOpacity=0.45)

    # centrar la vista en el centroide de la máscara
    try:
        import numpy as np
        a = slicer.util.arrayFromVolume(seg)
        idx = np.argwhere(a > 0)
        if len(idx):
            kji = idx.mean(axis=0)          # array order z,y,x
            ijk = [kji[2], kji[1], kji[0]]
            m = vtk.vtkMatrix4x4(); seg.GetIJKToRASMatrix(m)
            ras = [m.MultiplyPoint([ijk[0], ijk[1], ijk[2], 1])[k] for k in range(3)]
            for name in ["Red", "Yellow", "Green"]:
                slicer.app.layoutManager().sliceWidget(name).sliceLogic()\
                    .SetSliceOffset(ras[{"Red": 2, "Yellow": 0, "Green": 1}[name]])
            print(f"  vista centrada en el trombo RAS={[round(c,1) for c in ras]}")
    except Exception as e:
        print(f"  (no se pudo centrar la vista: {e})")

    # nodo de fiducial listo y modo de colocación activado
    fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                              "proximal_tip")
    fid.GetDisplayNode().SetSelectedColor(0, 1, 0)
    slicer.modules.markups.logic().SetActiveListID(fid)
    slicer.modules.markups.logic().StartPlaceMode(False)

    print(f"\n=== sub-{row['sub_id']} ({row['arm']}) "
          f"[{n_done + 1}/{len(rows)}]  {os.path.basename(csv_path)} ===")
    print(f"  vaso={row['label_vessel']}  lado={row['label_side']}  "
          f"n_vox={row['n_voxels']}")
    print("  COLOCA UN FIDUCIAL en el extremo PROXIMAL del trombo")
    print("  (proximal = el que da hacia el corazón, CONTRA el flujo:")
    print("   el lado carotídeo/M1, no el lado M2/M3 distal)")
    print("  >> si ves MAS DE UN trombo pintado, usa skip('dos trombos')")
    if str(row["sub_id"]) == "284":
        print("  >> sub-284: anota TAMBIEN vaso y lado en notas, p.ej.")
        print("     save_and_next('MCA M1 izquierda')")
    print("  Luego: save_and_next()")


def _get_tip_ras():
    import slicer
    for fid in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
        if fid.GetNumberOfControlPoints() > 0:
            ras = [0.0, 0.0, 0.0]
            fid.GetNthControlPointPosition(0, ras)
            return [round(c, 2) for c in ras]
    return None


def save_and_next(notas=""):
    row, i = _current["row"], _current["index"]
    if row is None:
        print("No hay caso cargado. Ejecuta load_next()")
        return
    ras = _get_tip_ras()
    if ras is None:
        print("NO hay ningun fiducial colocado. Coloca uno antes de guardar.")
        return
    rows, fields = _read_rows(_current["csv"])
    for f in ["tip_ras_r", "tip_ras_a", "tip_ras_s"]:
        if f not in fields:
            fields.append(f)
    rows[i]["tip_ras_r"], rows[i]["tip_ras_a"], rows[i]["tip_ras_s"] = ras
    rows[i]["anotado"] = "si"
    rows[i]["fecha"] = datetime.now().isoformat(timespec="seconds")
    rows[i]["notas"] = notas
    _write_rows(_current["csv"], rows, fields)
    print(f"  guardado sub-{row['sub_id']}: RAS={ras}")
    status()
    load_next()


def skip(motivo="no anotable"):
    row, i = _current["row"], _current["index"]
    if row is None:
        print("No hay caso cargado.")
        return
    rows, fields = _read_rows(_current["csv"])
    rows[i]["anotado"] = "skip"
    rows[i]["fecha"] = datetime.now().isoformat(timespec="seconds")
    rows[i]["notas"] = motivo
    _write_rows(_current["csv"], rows, fields)
    print(f"  sub-{row['sub_id']} marcado como skip: {motivo}")
    load_next()


def reload_current():
    if _current["index"] is None:
        print("No hay caso cargado.")
        return
    _current["row"] = None
    _current["index"] = None
    _current["csv"] = None
    load_next()


try:
    import vtk  # noqa: F401  (usado en load_next)
except Exception:
    pass

print(__doc__)
status()
print("\nEmpieza con:  load_next()")
