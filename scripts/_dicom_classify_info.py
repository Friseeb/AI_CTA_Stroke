"""Read classification-relevant DICOM tags for sub-14 series."""
import pydicom
from pathlib import Path

# One representative file from each export
exports = {
    "Export1": Path(r'D:\14\Export_2026-03-07_15-51-48_1\10001944\10001945'),
    "Export2": Path(r'D:\14\Export_2026-03-07_15-53-14_1\10002BA2\10002BA3'),
}

TAGS = [
    ('StudyDescription',        '0008103e'.upper(), (0x0008, 0x1030)),
    ('ProtocolName',             '00181030',         (0x0018, 0x1030)),
    ('BodyPartExamined',         '00180015',         (0x0018, 0x0015)),
    ('SeriesDescription',        '0008103E',         (0x0008, 0x103E)),
    ('SliceThickness',           '00180050',         (0x0018, 0x0050)),
    ('ImageOrientationPatient',  '00200037',         (0x0020, 0x0037)),
]

for export_name, base in exports.items():
    print(f"\n{'='*60}")
    print(f"  {export_name}: {base}")
    print(f"{'='*60}")

    # First, get study-level tags from any one file
    first_file = None
    for series_dir in sorted(base.iterdir()):
        files = [f for f in series_dir.iterdir() if f.is_file()]
        if files:
            first_file = files[0]
            break

    if first_file:
        try:
            ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
            print(f"\n  STUDY-LEVEL TAGS (from {first_file.parent.name}/{first_file.name}):")
            print(f"    StudyDescription : {getattr(ds, 'StudyDescription', 'N/A')}")
            print(f"    ProtocolName     : {getattr(ds, 'ProtocolName', 'N/A')}")
            print(f"    BodyPartExamined : {getattr(ds, 'BodyPartExamined', 'N/A')}")
            print(f"    StudyID          : {getattr(ds, 'StudyID', 'N/A')}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n  SERIES-LEVEL TAGS:")
    for series_dir in sorted(base.iterdir()):
        files = [f for f in series_dir.iterdir() if f.is_file()]
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            desc = getattr(ds, 'SeriesDescription', 'N/A')
            proto = getattr(ds, 'ProtocolName', 'N/A')
            body = getattr(ds, 'BodyPartExamined', 'N/A')
            modality = getattr(ds, 'Modality', 'N/A')
            thick = getattr(ds, 'SliceThickness', 'N/A')
            n = len(files)
            print(f"    {series_dir.name:12s}  {n:4d}f  [{modality}]  "
                  f"SeriesDesc={desc!r}  Protocol={proto!r}  Body={body!r}  Thick={thick}")
        except Exception as e:
            print(f"    {series_dir.name:12s}  ERROR: {e}")
