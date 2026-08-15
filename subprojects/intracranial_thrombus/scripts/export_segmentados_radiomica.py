"""
Exporta a un disco externo TODOS los casos ya segmentados (mascara + su CTA)
para rehacer la radiomica desde cero en otro ordenador.

Estructura de salida:
    <destino>/RADIOMICA_casos_segmentados/
        manifest.csv                 sub_id, cohorte, rutas relativas, voxeles, vaso, lado...
        <cohorte>/sub-XXX/
            sub-XXX_CTA.nii.gz
            sub-XXX_thrombus_clean.nii.gz
            sub-XXX_thrombus_clean_log.json

La cohorte se conserva en la ruta a proposito: el mismo sub_id es un paciente
DISTINTO en DAYLIGHT y en SLAOEVT. Nunca aplanar esto en una sola carpeta.

Excluidos: sub-178 (dos oclusiones, ya excluido en fase0_cohort_ids.json) y
los *_WRONG.nii.gz (archivos cruzados ya detectados).

Uso:  python export_segmentados_radiomica.py /media/fridmans/DICOM3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, SLAOEVT, SLAODICOM,
                   EXPORT_ROOT, NCCT_RAW, NCCT_REG, OUTPUT, V2, DCM2NIIX)
import csv
import json
import os
import shutil
import sys

BASE = SEG_ROOT
FOLDERS = {
    'DAYLIGHT':          BASE / 'DAYLIGHT',
    'DAYLIGHT_STANDARD': BASE / 'DAYLIGHT_STANDARD',
    'SLAO':              BASE / 'SLAO',
    'SLAOEVT':           SEG_FOLDERS['SLAOEVT'],
}
EXCLUIR_SUB_ID = {178}   # dos oclusiones (coincide con fase0_cohort_ids.json)

dest_root = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPORT_ROOT
DEST = dest_root / 'RADIOMICA_casos_segmentados'
DEST.mkdir(parents=True, exist_ok=True)

rows, skipped = [], []
for cohorte, d in FOLDERS.items():
    for mask in sorted(d.glob('*_thrombus_clean.nii.gz')):
        sub = mask.name.split('_')[0]                # sub-XXX
        sid = int(sub.replace('sub-', ''))
        log = mask.with_name(mask.name.replace('.nii.gz', '_log.json'))

        if sid in EXCLUIR_SUB_ID:
            skipped.append((cohorte, sub, 'excluido: dos oclusiones')); continue
        if not log.exists():
            skipped.append((cohorte, sub, 'sin log JSON')); continue
        try:
            meta = json.load(open(log))
        except Exception as e:
            skipped.append((cohorte, sub, f'log ilegible: {e}')); continue
        cta = meta.get('cta_path')
        if not cta or not os.path.exists(cta):
            skipped.append((cohorte, sub, f'CTA no localizable: {cta}')); continue

        out = DEST / cohorte / sub
        out.mkdir(parents=True, exist_ok=True)
        cta_out = out / f'{sub}_CTA.nii.gz'
        if not cta_out.exists() or cta_out.stat().st_size != os.path.getsize(cta):
            shutil.copy2(cta, cta_out)
        shutil.copy2(mask, out / mask.name)
        shutil.copy2(log,  out / log.name)

        m = meta.get('metrics', {}).get('thrombus', {})
        a = meta.get('annotation', {})
        rows.append({
            'sub_id': sub, 'cohorte': cohorte,
            'cta': f'{cohorte}/{sub}/{cta_out.name}',
            'mascara': f'{cohorte}/{sub}/{mask.name}',
            'log': f'{cohorte}/{sub}/{log.name}',
            'voxeles': m.get('voxels'), 'volumen_mm3': m.get('volume_mm3'),
            'longitud_si_mm': m.get('length_si_mm'),
            'vaso': a.get('vessel'), 'lado': a.get('side'),
            'hiperdenso': a.get('hyperdense'), 'perviousness': a.get('clot_perviousness'),
            'tip_proximal': 'si' if a.get('proximal_tip_ras_mm') else 'no',
            'cta_origen': cta,
        })
        print(f"  {cohorte:18s} {sub:10s} {m.get('voxels','?'):>6} vox", flush=True)

with open(DEST / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\n=== {len(rows)} casos exportados a {DEST} ===")
if skipped:
    print(f"\nOmitidos ({len(skipped)}):")
    for s in skipped:
        print('  ', s)
