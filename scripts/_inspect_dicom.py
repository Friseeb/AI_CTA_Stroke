import pydicom
from pathlib import Path
import sys

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/107/Export_2026-03-07_17-26-14_1")
seen = set()

for p in sorted(base.rglob("*")):
    if not p.is_file():
        continue
    if p.suffix.upper() in {".HTM", ".HTML", ".EXE", ".PDF", ".PNG", ".CSS", ".INF", ".LOG", ".JAR"}:
        continue
    if any(s in str(p) for s in ["IHE_PDI", "XTR_CONT", "HELP", "JRE", "PLUGINS", "REPORT", "DICOMDIR", "AUTORUN"]):
        continue
    if p.parent in seen:
        continue
    seen.add(p.parent)
    try:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        nf = len([f for f in p.parent.iterdir() if f.is_file()])
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = f"{float(ipp[2]):.1f}" if ipp and len(ipp) >= 3 else "?"
        print(
            f"DIR={p.parent.name}  files={nf}"
            f"  mod={getattr(ds,'Modality','')}"
            f"  body={getattr(ds,'BodyPartExamined','')}"
            f"  snum={getattr(ds,'SeriesNumber','')}"
            f"  z_sample={z}"
            f"  sdesc={getattr(ds,'SeriesDescription','')}"
            f"  proto={getattr(ds,'ProtocolName','')}"
            f"  study={getattr(ds,'StudyDescription','')}"
        )
    except Exception as e:
        print(f"ERROR {p.name}: {e}")
