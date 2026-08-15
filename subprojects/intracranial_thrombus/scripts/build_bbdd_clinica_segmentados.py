"""
Base de datos clinica de TODOS los casos con trombo segmentado (n=134 segmentaciones).

Espina: una fila por SEGMENTACION (no por paciente), con:
  sub_id, carpeta, arm real (derivado del cta_path, no del nombre de carpeta),
  familia de namespace, pin, y marca de duplicado.

AVISOS que este script hace explicitos en la salida:
  - `arm_real` se deriva de la RUTA del CTA, no de la carpeta: 9 casos estan
    archivados en una carpeta que no corresponde a su brazo real.
  - `pin_duplicado`: 13 pines aparecen 2 veces => el mismo paciente (y en los
    casos comprobados, el MISMO estudio: shape y offset identicos) fue
    segmentado dos veces bajo cohortes distintas. Para radiomica hay que
    quedarse con una sola copia o se inflaria la N y se violaria la
    independencia de las observaciones.
  - Solo hay DOS espacios de identificadores, no cuatro: DAYLIGHT (record_id
    de REDCap) y SLAO (sub_id). SLAOEVT es un subconjunto de SLAO (126/126
    pines coinciden) y DAYLIGHT_STANDARD es el brazo Standard de DAYLIGHT.

Fuentes clinicas (todas por pin salvo donde se indica):
  EVT_Master_Dataset_LOSR_Feb25.csv   procedimiento EVT + mRS previo + mortalidad
  harmonized_clinical_dataset.csv     demografia, FRCV, etiologia, outcomes
  evt_masterlist_clinical_by_pin.csv  1 fila/pin, EVT + reconciliacion
  clinicas_n86_REVISADO_FINAL.csv     curado a mano (solo DAYLIGHT, por sub_id)
  slao_evt_priority_with_localization.csv  EVT REDCap para SLAO/SLAOEVT
  slao_lvo_pin_list.csv / daylight_lvo_pin_list.csv   localizacion de oclusion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (REPO, DATASETS, SEG_ROOT, SEG_FOLDERS, SLAOEVT, SLAODICOM,
                   EXPORT_ROOT, NCCT_RAW, NCCT_REG, OUTPUT, V2, DCM2NIIX)
import csv
import json
from collections import Counter, defaultdict

OUT = REPO / 'subprojects/intracranial_thrombus/output/bbdd_clinica_final.csv'

rows = json.load(open('/tmp/casos_134_pin.json'))

# --- pines que faltaban: se resuelven desde el REDCap de DAYLIGHT (830 filas,
# record_id == sub_id). El crosswalk de 126 filas era solo un subconjunto.
_dl = {}
for _r in csv.DictReader(open(REPO/'cohort_generation/inputs/DAYLIGHT_DATA_2026-06-18_1117.csv',
                              encoding='utf-8', errors='replace')):
    _rid = (_r.get('record_id') or '').strip()
    _pn = (_r.get('pin') or '').strip()
    if _rid and _pn:
        _dl[_rid] = _pn
_recuperados = 0
for _r in rows:
    if not _r.get('pin') and _r['familia'] == 'DAYLIGHT':
        _p = _dl.get(str(_r['sub_id']), '')
        if _p:
            _r['pin'] = _p
            _r['pin_origen'] = 'DAYLIGHT_REDCap(record_id)'
            _recuperados += 1
print(f'  pines recuperados del REDCap de DAYLIGHT: {_recuperados}')

# --- deduplicacion: 13 pines duplicados -> 121 pacientes unicos
sys.path.insert(0, str(REPO/'subprojects/intracranial_thrombus/scripts'))
from dedup_decisiones import DECISIONES
try:
    from dedup_decisiones import EXCLUIR
except ImportError:
    EXCLUIR = {}

# excluir casos rechazados por calidad de mascara
_n0 = len(rows)
rows = [r for r in rows if (r['sub_id'], r['carpeta']) not in EXCLUIR]
if _n0 != len(rows):
    print(f'  excluidos por calidad de mascara: {_n0-len(rows)}')

_pc = Counter(r['pin'] for r in rows if r['pin'])
_keep, _descartados = [], []
for r in rows:
    pin = r['pin']
    if pin and _pc[pin] > 1:
        # De un grupo de copias con el mismo pin se conserva SOLO la que tiene
        # decision explicita. Si ninguna la tiene, se conservan todas marcadas,
        # para que el problema sea visible en vez de resolverse en silencio.
        grupo = [q for q in rows if q['pin'] == pin]
        con_decision = [q for q in grupo if (q['sub_id'], q['carpeta']) in DECISIONES]
        if not con_decision:
            r['motivo_dedup'] = '*** DUPLICADO SIN DECISION ***'
            _keep.append(r)
        elif (r['sub_id'], r['carpeta']) in DECISIONES:
            r['motivo_dedup'] = DECISIONES[(r['sub_id'], r['carpeta'])]
            _keep.append(r)
        else:
            _descartados.append((r['sub_id'], r['carpeta']))
    else:
        r['motivo_dedup'] = ''
        _keep.append(r)
print(f'  deduplicacion: {len(rows)} -> {len(_keep)} filas ({len(_descartados)} copias descartadas)')
rows = _keep


def load(path, key, prefix='', filt=None):
    """Indexa un CSV por una clave; devuelve dict clave -> fila (con prefijo)."""
    out = {}
    p = Path(path)
    if not p.exists():
        print(f'  [FALTA] {path}')
        return out
    for r in csv.DictReader(open(p, encoding='utf-8', errors='replace')):
        if filt and not filt(r):
            continue
        k = (r.get(key) or '').strip()
        if not k:
            continue
        k = k.replace('sub-', '').lstrip('0') or '0'
        if k not in out:
            out[k] = {f'{prefix}{c}': v for c, v in r.items()}
    return out

print('Cargando fuentes clinicas...')
evt_master = load(REPO/'cohort_generation/inputs/EVT_Master_Dataset_LOSR_Feb25 (1).csv', 'pin', 'evtm_')
evt_bypin  = load(REPO/'cohort_generation/outputs/evt_masterlist_clinical_by_pin.csv', 'pin', 'ml_')
curado     = load(REPO/'subprojects/intracranial_thrombus/output/clinicas_n86_REVISADO_FINAL.csv', 'sub_id', 'cur_')
prio_slao  = load(REPO/'cohort_generation/inputs/slao_evt_priority_with_localization.csv', 'pin', 'slao_')
lvo_slao   = load(SEG_ROOT.parent/'slao_lvo_pin_list.csv', 'pin', 'lvo_')
lvo_dl     = load(SEG_ROOT.parent/'daylight_lvo_pin_list.csv', 'pin', 'lvo_')

# harmonized: por pin, priorizando la cohorte correspondiente
harm = defaultdict(list)
for r in csv.DictReader(open(REPO/'cohort_generation/outputs/harmonized_clinical_dataset.csv',
                             encoding='utf-8', errors='replace')):
    pin = (r.get('mrn_or_linkage_id') or '').strip()
    if pin:
        harm[pin].append(r)

COH_PREF = {'DAYLIGHT': ['DAYLIGHT', 'LOSR'], 'SLAO': ['SLAO_DAYLIGHT_BROAD', 'LOSR']}

def pick_harm(pin, familia):
    cands = harm.get(pin, [])
    if not cands:
        return {}
    for pref in COH_PREF.get(familia, []):
        sub = [c for c in cands if c.get('source_cohort') == pref]
        if sub:
            best = sub[0]
            return {f'harm_{k}': v for k, v in best.items()} | {'harm_n_filas_pin': len(cands)}
    return {f'harm_{k}': v for k, v in cands[0].items()} | {'harm_n_filas_pin': len(cands)}

pin_count = Counter(r['pin'] for r in rows if r['pin'])

out_rows = []
for r in rows:
    pin = r['pin']
    sid = str(r['sub_id'])
    d = {
        'sub_id': r['sub_id'],
        'carpeta_segmentacion': r['carpeta'],
        'arm_daylight': r['arm_real'] if r['arm_real'] in ('extended', 'standard') else 'no_aplica (SLAO no tiene brazos)',
        'familia_namespace': r['familia'],
        'pin': pin or '',
        'motivo_dedup': r.get('motivo_dedup',''),
        'pin_origen': r.get('pin_origen','crosswalk'),
        'cta_path': r['cta_path'],
    }
    if pin:
        d |= pick_harm(pin, r['familia'])
        d |= evt_master.get(pin.lstrip('0') or '0', {})
        d |= evt_bypin.get(pin.lstrip('0') or '0', {})
        d |= prio_slao.get(pin.lstrip('0') or '0', {})
        d |= (lvo_slao if r['familia'] == 'SLAO' else lvo_dl).get(pin.lstrip('0') or '0', {})
    if r['familia'] == 'DAYLIGHT':
        d |= curado.get(sid, {})
    out_rows.append(d)

cols = []
for d in out_rows:
    for c in d:
        if c not in cols:
            cols.append(c)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(out_rows)

print(f'\n{len(out_rows)} filas x {len(cols)} columnas -> {OUT}')
print('\nCobertura de variables clave:')
KEY = [('edad','harm_age'),('sexo','harm_sex'),('NIHSS basal','harm_baseline_nihss'),
       ('NIHSS (EVT reg)','evtm_nihss'),('mRS previo','evtm_baseline_mRS0_2'),
       ('mRS 90d','harm_mrs_90d'),('etiologia TOAST','harm_toast_classification'),
       ('etiologia (EVT reg)','evtm_stroke_etiology'),('EVT','harm_evt'),
       ('site_occ','evtm_site_occ'),('lvo_branch','harm_lvo_branch'),
       ('pases','evtm_passess'),('mTICI','evtm_mtici'),('tPA','harm_iv_thrombolysis'),
       ('ASPECTS','evtm_aspects'),('FA','harm_atrial_fibrillation')]
for label, c in KEY:
    n = sum(1 for d in out_rows if str(d.get(c,'')).strip() not in ('','nan','None','.'))
    print(f'  {label:22s} {n:>4}/{len(out_rows)}  ({c})')
print('\npor arm:', dict(Counter(d['arm_daylight'] for d in out_rows)))
print('sin pin:', sum(1 for d in out_rows if not d['pin']))
print('pacientes unicos (pines distintos):', len({d['pin'] for d in out_rows if d['pin']}))
