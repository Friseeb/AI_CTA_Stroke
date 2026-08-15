# Etiquetado de trombos SLAOEVT en otro ordenador

Contenido de este disco (`SLAOEVT_etiquetado/`):

```
imagenes/                  sub-XXX/ con CTA (+ delay1 / delay2 / NCCT segun lo exportado)
extension/                 extension de Slicer CTAThrombusSOP (zip + carpeta ya descomprimida)
worklist/                  slaoevt_worklist_v2.json, load_slaoevt_case_v2.py, CSV de los 47 casos
segmentaciones_ya_hechas/  las mascaras + logs ya completadas en el PC original
README.md                  contexto de la cohorte (el nombre SLAOEVT no garantiza EVT)
NAMESPACE_WARNING.md       aviso de colisiones de sub_id entre cohortes -- LEER
```

## 1. Instalar la extension en Slicer

1. Copia la carpeta `extension/CTAThrombusSOP_pkg/CTAThrombusSOP` a un sitio fijo
   del disco duro del otro PC (p.ej. `~/CTAThrombusSOP`).
2. En Slicer: **Edit > Application Settings > Modules**, y en *Additional module
   paths* anade:
   `<ruta>/CTAThrombusSOP/lib/Slicer-5.10/qt-scripted-modules`
3. Reinicia Slicer. Deben aparecer los modulos **CTAThrombusDemarcation** y
   **CTAThrombusFinalizeSOP**.

## 2. Ajustar las rutas del loader

Edita `worklist/load_slaoevt_case_v2.py` y cambia estas dos constantes al
principio del archivo:

```python
WORKLIST_PATH = Path("<ruta al disco>/SLAOEVT_etiquetado/worklist/slaoevt_worklist_v2.json")
NUEVA_DIR     = Path("<ruta al disco>/SLAOEVT_etiquetado/imagenes")
SEG_DIR       = Path("<ruta al disco>/SLAOEVT_etiquetado/segmentaciones_nuevas")
```

`SEG_DIR` conviene que sea una carpeta NUEVA (no `segmentaciones_ya_hechas`),
asi se distingue sin ambiguedad lo etiquetado en este PC.

**Importante**: el loader comprueba que exista un backup `*_original.nii.gz`
antes de ofrecer un caso (era la senal de "reparacion de spacing terminada" en
el PC original). Como los backups NO se copian a este disco, hay que
desactivar esa comprobacion: busca la funcion `_repair_confirmed` y haz que
devuelva `True` siempre:

```python
def _repair_confirmed(w):
    return True   # imagenes ya reparadas y verificadas en el PC de origen
```

## 3. Etiquetar

En la consola de Python de Slicer:

```python
exec(open('<ruta>/SLAOEVT_etiquetado/worklist/load_slaoevt_case_v2.py').read())
status()      # cuantos llevas y cual toca
next_case()   # carga el siguiente
```

Por cada caso:
1. Segmentar el trombo (label 1) con el Segment Editor.
2. **Pulsar "Place Proximal Tip Fiducial"** y marcar el extremo proximal.
   Si ves mas de un fiducial en la escena, borra los viejos primero -- el
   modulo coge el primero que encuentra y eso ya provoco tips erroneos.
3. Rellenar vaso y lado en la anotacion.
4. Guardar con el boton del modulo Finalize. Se marca solo en el worklist y
   se imprime una verificacion automatica (voxeles, solapamiento, HU, tip).
5. `next_case()`.

Comandos utiles: `skip_case("motivo")` aparca un caso, `show_deferred()` los
lista, `unskip_case(<id>)` lo recupera.

## 4. Al volver

Copia de vuelta:
- la carpeta `segmentaciones_nuevas/` completa (mascaras + logs JSON)
- el `worklist/slaoevt_worklist_v2.json` actualizado

En el PC original, las mascaras van a
`/media/fridmans/Research13T/datasets/SLAOEVT_Nueva/thrombus_manual/`.

## Avisos

- Las imagenes de este disco YA estan reparadas (spacing Z y signo Z corregidos,
  verificados contra el DICOM original). No volver a aplicar ninguna correccion
  de geometria.
- sub-522 y sub-529 llevan ademas una rotacion IS horneada (-17.3 y +32.5
  grados). Es intencionado.
- **No mezclar con casos de DAYLIGHT**: el mismo numero de sub_id es un
  paciente DISTINTO en cada cohorte. Ver `NAMESPACE_WARNING.md`.
