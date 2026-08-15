"""
Sincronizacion final a DICOM3 de todo lo trabajado:
  - mascaras nuevas / resegmentadas (sub-201) y retirada de las *_WRONG
  - base clinica reconstruida (con las exclusiones y la dedup al dia)
  - TIP_PROXIMAL.csv consolidado (122 tips: 2 lotes previos + lote3)
  - CASOS_PARA_RADIOMICA.csv con el filtro actualizado
  - NCCT recien convertidos de la cohorte SLAO
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, SLAOEVT, SLAODICOM,
                   EXPORT_ROOT, NCCT_RAW, NCCT_REG, OUTPUT, V2, DCM2NIIX)
import csv
import json
import shutil
from collections import Counter

OUT = REPO / 'subprojects/intracranial_thrombus/output'
V = OUT / 'v2'
DEST = EXPORT_ROOT / 'RADIOMICA_casos_segmentados'
SEGBASE = SEG_ROOT
SEGF = {'DAYLIGHT': SEGBASE/'DAYLIGHT', 'DAYLIGHT_STANDARD': SEGBASE/'DAYLIGHT_STANDARD',
        'SLAO': SEGBASE/'SLAO',
        'SLAOEVT': SEG_FOLDERS['SLAOEVT']}

b = list(csv.DictReader(open(OUT/'bbdd_clinica_final.csv', encoding='utf-8', errors='replace')))
print(f'pacientes en la base: {len(b)}')

# ---- 1. tips consolidados de los tres lotes -------------------------------
tip = {}
for f in ['anotacion_extremo_proximal.csv', 'anotacion_extremo_proximal_ampliacion.csv',
          'anotacion_extremo_proximal_lote3.csv']:
    p = V/f
    if not p.exists():
        continue
    for r in csv.DictReader(open(p)):
        if (r.get('anotado') or '').strip() == 'si' and str(r.get('tip_ras_r','')).strip():
            tip[str(r['sub_id']).strip()] = (r, f)

rows_tip, n_con = [], 0
for x in b:
    sid = x['sub_id']
    coh = x['carpeta_segmentacion']
    r = tip.get(sid)
    if r:
        rr, src = r
        n_con += 1
        rows_tip.append({'sub_id': f'sub-{sid}', 'cohorte': coh, 'pin': x['pin'],
                         'tiene_tip': 'SI', 'tip_R': rr['tip_ras_r'], 'tip_A': rr['tip_ras_a'],
                         'tip_S': rr['tip_ras_s'], 'origen_tip': src,
                         'vaso': rr.get('label_vessel',''), 'lado': rr.get('label_side',''),
                         'n_voxels': rr.get('n_voxels',''), 'fecha_anotacion': rr.get('fecha','')})
    else:
        rows_tip.append({'sub_id': f'sub-{sid}', 'cohorte': coh, 'pin': x['pin'],
                         'tiene_tip': 'NO', 'tip_R': '', 'tip_A': '', 'tip_S': '',
                         'origen_tip': '', 'vaso': '', 'lado': '', 'n_voxels': '',
                         'fecha_anotacion': ''})
with open(DEST/'TIP_PROXIMAL.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_tip[0].keys()))
    w.writeheader()
    w.writerows(rows_tip)
print(f'TIP_PROXIMAL.csv: {n_con}/{len(b)} con tip')

# ---- 2. mascaras y logs al dia (incluye resegmentados) --------------------
act = ret = 0
for x in b:
    sid, coh = x['sub_id'], x['carpeta_segmentacion']
    d = DEST/coh/f'sub-{sid}'
    d.mkdir(parents=True, exist_ok=True)
    for suf in ['_thrombus_clean.nii.gz', '_thrombus_clean_log.json']:
        s = SEGF[coh]/f'sub-{sid}{suf}'
        if not s.exists():
            continue
        t = d/f'sub-{sid}{suf}'
        if not t.exists() or t.stat().st_size != s.stat().st_size:
            shutil.copy2(s, t)
            act += 1
# retirar de DICOM3 cualquier mascara de casos ya excluidos/duplicados
keep = {(x['sub_id'], x['carpeta_segmentacion']) for x in b}
for coh in SEGF:
    dd = DEST/coh
    if not dd.exists():
        continue
    for sub_dir in dd.iterdir():
        if not sub_dir.is_dir():
            continue
        sid = sub_dir.name.replace('sub-', '')
        if (sid, coh) in keep:
            continue
        for m in list(sub_dir.glob('*_thrombus_clean.nii.gz')) + list(sub_dir.glob('*_thrombus_clean_log.json')):
            m.rename(m.with_name(m.name.replace('_thrombus_clean', '_thrombus_clean_NO_USAR')))
            ret += 1
print(f'mascaras actualizadas: {act} | marcadas NO_USAR (excluidos/duplicados): {ret}')

# ---- 3. NCCT recien convertidos ------------------------------------------
nc = 0
for x in b:
    sid, coh = x['sub_id'], x['carpeta_segmentacion']
    src = Path(x['cta_path']).parent/f'sub-{sid}_acq-NCCT_ct.nii.gz'
    if src.exists():
        t = DEST/coh/f'sub-{sid}'/f'sub-{sid}_NCCT.nii.gz'
        if not t.exists() or t.stat().st_size != src.stat().st_size:
            shutil.copy2(src, t)
            nc += 1
print(f'NCCT nuevos copiados: {nc}')

# ---- 4. base clinica y filtro de radiomica -------------------------------
shutil.copy2(OUT/'bbdd_clinica_final.csv', DEST/'bbdd_clinica_final.csv')
for old in DEST.glob('bbdd_clinica_1*_pacientes.csv'):
    old.unlink()
for s in ['dedup_decisiones.py', 'build_bbdd_clinica_segmentados.py']:
    p = REPO/'subprojects/intracranial_thrombus/scripts'/s
    if p.exists():
        shutil.copy2(p, DEST/s)

m = list(csv.DictReader(open(DEST/'manifest.csv')))
out = [{'sub_id': x['sub_id'], 'cohorte': x['cohorte'],
        'usar_en_radiomica': 'SI' if (x['sub_id'].replace('sub-',''), x['cohorte']) in keep else 'NO_excluido_o_duplicado',
        'cta': x['cta'], 'mascara': x['mascara'], 'log': x['log'],
        'voxeles': x['voxeles'], 'vaso': x['vaso'], 'lado': x['lado']} for x in m]
with open(DEST/'CASOS_PARA_RADIOMICA.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print('filtro radiomica:', dict(Counter(o['usar_en_radiomica'] for o in out)))
print('\nSYNC COMPLETADA')
