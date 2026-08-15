"""
Anade a la carpeta de radiomica de DICOM3 las series auxiliares (NCCT, NCCT
registrado y delays) de los 121 pacientes unicos, mas un CSV con el tip
proximal de los que lo tienen.

Se copian junto al CTA de cada caso, respetando la estructura por cohorte.
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

DEST = EXPORT_ROOT / 'RADIOMICA_casos_segmentados'
SEGBASE = SEG_ROOT
NCCT_REG = SEGBASE / 'DAYLIGHT_NCCT_REG'
NCCT_RAW = SEGBASE / 'DAYLIGHT_NCCT'
SEGF = {'DAYLIGHT': SEGBASE/'DAYLIGHT', 'DAYLIGHT_STANDARD': SEGBASE/'DAYLIGHT_STANDARD',
        'SLAO': SEGBASE/'SLAO',
        'SLAOEVT': SEG_FOLDERS['SLAOEVT']}

rows = list(csv.DictReader(open(REPO/'subprojects/intracranial_thrombus/output/bbdd_clinica_121_pacientes.csv',
                                encoding='utf-8', errors='replace')))

# tip anotado a posteriori (la fuente principal): dos CSV de batches
TIP_CSV = {}
for _f in ['anotacion_extremo_proximal.csv', 'anotacion_extremo_proximal_ampliacion.csv']:
    _p = REPO/'subprojects/intracranial_thrombus/output/v2'/_f
    if _p.exists():
        for _r in csv.DictReader(open(_p)):
            if str(_r.get('anotado','')).strip().lower() == 'si' and str(_r.get('tip_ras_r','')).strip():
                TIP_CSV[str(_r['sub_id']).strip()] = _r
print(f'tips cargados de los CSV de anotacion: {len(TIP_CSV)}')

tips, stats, faltan = [], Counter(), []
for x in rows:
    sid = x['sub_id']
    coh = x['carpeta_segmentacion']
    cta = Path(x['cta_path'])
    src_dir = cta.parent
    out = DEST / coh / f'sub-{sid}'
    out.mkdir(parents=True, exist_ok=True)

    cand = {}
    for pat, lab in [(f'sub-{sid}_acq-NCCT_ct.nii.gz', 'NCCT'),
                     (f'sub-{sid}_acq-delay1_ct.nii.gz', 'delay1'),
                     (f'sub-{sid}_acq-delay2_ct.nii.gz', 'delay2'),
                     (f'sub-{sid}_acq-CTA1stdelay_ct.nii.gz', 'delay1'),
                     (f'sub-{sid}_acq-CTA2nddelay_ct.nii.gz', 'delay2')]:
        p = src_dir / pat
        if p.exists():
            cand.setdefault(lab, p)
    p = NCCT_REG / f'sub-{sid}' / f'sub-{sid}_NCCT_reg.nii.gz'
    if p.exists():
        cand['NCCT_reg'] = p
    p = NCCT_RAW / f'sub-{sid}' / f'sub-{sid}_NCCT.nii.gz'
    if p.exists():
        cand.setdefault('NCCT', p)

    if not cand:
        faltan.append((sid, coh))
    for lab, p in cand.items():
        dst = out / f'sub-{sid}_{lab}.nii.gz'
        if not dst.exists() or dst.stat().st_size != p.stat().st_size:
            shutil.copy2(p, dst)
        stats[lab] += 1

    # tip proximal: PRIMERO los CSV de anotacion a posteriori (batches
    # validation_30 / extension_54), que es donde esta la mayoria; el log de
    # la segmentacion solo lo tienen los anotados sobre la marcha.
    tip = vaso = lado = None
    origen_tip = ''
    r = TIP_CSV.get(sid)
    if r:
        tip = [float(r['tip_ras_r']), float(r['tip_ras_a']), float(r['tip_ras_s'])]
        vaso, lado = r.get('label_vessel'), r.get('label_side')
        origen_tip = f"CSV {r.get('annotation_batch','')}"
    if tip is None:
        lg = SEGF[coh] / f'sub-{sid}_thrombus_clean_log.json'
        try:
            a = (json.load(open(lg)).get('annotation') or {})
            tip, vaso, lado = a.get('proximal_tip_ras_mm'), a.get('vessel'), a.get('side')
            if tip:
                origen_tip = 'log de segmentacion'
        except Exception:
            pass
    tips.append({'sub_id': f'sub-{sid}', 'cohorte': coh, 'pin': x['pin'],
                 'tiene_tip': 'SI' if tip else 'NO',
                 'tip_R': tip[0] if tip else '', 'tip_A': tip[1] if tip else '',
                 'tip_S': tip[2] if tip else '',
                 'origen_tip': origen_tip,
                 'vaso': vaso or '', 'lado': lado or '',
                 'series_disponibles': '|'.join(sorted(cand))})
    print(f"  sub-{sid:<6} {coh:<18} {'+'.join(sorted(cand)) or 'SIN AUXILIARES'}", flush=True)

with open(DEST/'TIP_PROXIMAL.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(tips[0].keys()))
    w.writeheader()
    w.writerows(tips)

print('\n=== series copiadas ===')
for k, v in stats.most_common():
    print(f'  {k:10s} {v}')
print(f'\ncasos sin auxiliares: {len(faltan)} -> {[s for s,_ in faltan]}')
print(f"con tip proximal: {sum(1 for t in tips if t['tiene_tip']=='SI')} de {len(tips)}")
print(f"-> {DEST/'TIP_PROXIMAL.csv'}")
