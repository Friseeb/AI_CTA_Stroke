#!/usr/bin/env bash
# Exporta a un disco externo todo lo necesario para seguir etiquetando trombos
# de SLAOEVT_Nueva en OTRO ordenador.
#
# Uso:
#   ./export_slaoevt_to_disk.sh /media/fridmans/DICOM3            # 47 EVT=1, CTA+delay1  (~26 GB)
#   ./export_slaoevt_to_disk.sh /media/fridmans/DICOM3 all4       # 47 EVT=1, las 4 series (~39 GB)
#   ./export_slaoevt_to_disk.sh /media/fridmans/DICOM3 cta 86     # los 86, solo CTA       (~36 GB)
#
# Copia SOLO los .nii.gz reparados (no los backups *_original / *_prerot, que
# ocupan lo mismo y no se necesitan para etiquetar).
set -euo pipefail

DEST_ROOT="${1:?Falta el destino, p.ej. /media/fridmans/DICOM3}"
MODE="${2:-delay1}"     # cta | delay1 | all4
SCOPE="${3:-47}"        # 47 | 86

SRC=/media/fridmans/Research13T/datasets/SLAOEVT_Nueva
REPO=/home/fridmans/Documents/pwd/AI_CTA_Stroke
DESK=/home/fridmans/Desktop/INTRACRANEAL_THROMBUS/CTA_intracraneal_segmentation
DEST="$DEST_ROOT/SLAOEVT_etiquetado"

case "$MODE" in
  cta)    SERIES=(CTA) ;;
  delay1) SERIES=(CTA delay1) ;;
  all4)   SERIES=(CTA delay1 delay2 NCCT) ;;
  *) echo "MODE invalido: $MODE (usa cta|delay1|all4)"; exit 1 ;;
esac

echo "Destino : $DEST"
echo "Series  : ${SERIES[*]}"
echo "Alcance : $SCOPE casos"
echo

mkdir -p "$DEST"/{imagenes,extension,worklist,segmentaciones_ya_hechas}

# --- lista de sujetos segun alcance ---
python3 - "$SCOPE" > /tmp/_export_subs.txt <<'PY'
import json, sys
scope = sys.argv[1]
w = json.load(open('/home/fridmans/Desktop/INTRACRANEAL_THROMBUS/CTA_intracraneal_segmentation/slaoevt_worklist_v2.json'))
elig = [x for x in w if not x.get('excluded')]
if scope == '47':
    elig = [x for x in elig if x.get('evt_status') == 'EVT=1']
for e in elig:
    print(e['sub_id'])
PY
N=$(wc -l < /tmp/_export_subs.txt)
echo "Sujetos a copiar: $N"

# --- imagenes ---
COPIED=0
while read -r SID; do
  mkdir -p "$DEST/imagenes/sub-$SID"
  for S in "${SERIES[@]}"; do
    F="$SRC/sub-$SID/sub-${SID}_acq-${S}_ct.nii.gz"
    J="$SRC/sub-$SID/sub-${SID}_acq-${S}_ct.json"
    [ -f "$F" ] && rsync -a --info=progress2 "$F" "$DEST/imagenes/sub-$SID/" && COPIED=$((COPIED+1))
    [ -f "$J" ] && rsync -a "$J" "$DEST/imagenes/sub-$SID/"
  done
done < /tmp/_export_subs.txt
echo "Archivos de imagen copiados: $COPIED"

# --- extension de Slicer (ya con los parches: automarcado + verificacion) ---
rsync -a "$REPO/subprojects/intracranial_thrombus/slicer_extension/CTAThrombusSOP_install.zip" "$DEST/extension/"
rsync -a "$REPO/subprojects/intracranial_thrombus/slicer_extension/CTAThrombusSOP_pkg" "$DEST/extension/"

# --- worklist, loader y CSV de casos ---
rsync -a "$DESK/slaoevt_worklist_v2.json" "$DEST/worklist/"
rsync -a "$DESK/load_slaoevt_case_v2.py"  "$DEST/worklist/"
[ -f "$REPO/subprojects/intracranial_thrombus/output/slaoevt_evt1_47_casos.csv" ] && \
  rsync -a "$REPO/subprojects/intracranial_thrombus/output/slaoevt_evt1_47_casos.csv" "$DEST/worklist/"

# --- segmentaciones ya hechas (por si hay que consultarlas o continuar) ---
rsync -a "$SRC/thrombus_manual/" "$DEST/segmentaciones_ya_hechas/" 2>/dev/null || true

# --- documentacion ---
[ -f "$SRC/README.md" ] && rsync -a "$SRC/README.md" "$DEST/"
rsync -a "$REPO/docs/NAMESPACE_WARNING.md" "$DEST/" 2>/dev/null || true

rsync -a "$REPO/subprojects/intracranial_thrombus/scripts/LEEME_EN_EL_OTRO_PC.md" "$DEST/"
sync
echo
echo "=== HECHO ==="
du -sh "$DEST"/* 2>/dev/null
echo
echo "Lee $DEST/LEEME_EN_EL_OTRO_PC.md para las instrucciones de instalacion."
