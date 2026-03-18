import pydicom
from pathlib import Path

base1 = Path(r'D:\14\Export_2026-03-07_15-51-48_1\10001944\10001945')
base2 = Path(r'D:\14\Export_2026-03-07_15-53-14_1\10002BA2\10002BA3')

def read_series_info(base):
    results = []
    for series_dir in sorted(base.iterdir()):
        files = [f for f in series_dir.iterdir() if f.is_file()]
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            desc = getattr(ds, 'SeriesDescription', 'N/A')
            thickness = getattr(ds, 'SliceThickness', 'N/A')
            modality = getattr(ds, 'Modality', 'N/A')
            series_num = getattr(ds, 'SeriesNumber', 'N/A')
            results.append({
                'dir': series_dir.name, 'n_files': len(files),
                'desc': desc, 'thickness': thickness,
                'modality': modality, 'series_num': series_num
            })
        except Exception as e:
            results.append({'dir': series_dir.name, 'n_files': len(files),
                            'desc': f'ERR:{e}', 'thickness': 'N/A',
                            'modality': 'N/A', 'series_num': 'N/A'})
    return results

print('=== Export 1 ===')
for r in read_series_info(base1):
    print(f"  {r['dir']:12s}  {r['n_files']:4d} files  [{r['modality']}]  S#{str(r['series_num']):>4}  Thick={r['thickness']}  Desc: {r['desc']}")

print()
print('=== Export 2 ===')
for r in read_series_info(base2):
    print(f"  {r['dir']:12s}  {r['n_files']:4d} files  [{r['modality']}]  S#{str(r['series_num']):>4}  Thick={r['thickness']}  Desc: {r['desc']}")
