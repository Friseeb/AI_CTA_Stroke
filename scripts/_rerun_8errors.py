"""Re-run the 8 previously-errored subjects after the two fixes:
  FIX 1 — 'injection' replaced with 'medrad injection' / 'injection images'
  FIX 2 — CardiacCT min_source_slices lowered to 50
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from convert_daylightdicom_to_bids import process_subject, SCAN_TYPE_CONFIGS

OUT_ROOT = Path(r"C:\Users\spost\Desktop\CT_image\SLAAOBIDS")
ALLOWED  = [c.name for c in SCAN_TYPE_CONFIGS]

# FIX 1 — injection keyword false-positive: ThoraxCT / CTA subjects
FIX1 = ["148", "130", "220", "250", "222"]
# FIX 2 — CardiacCT min_source_slices=50: cardiac subjects
FIX2 = ["122", "125", "240"]

for sid in FIX1 + FIX2:
    subject_dir = Path(f"D:/{sid}")
    print(f"\n[sub-{sid}]")
    results, _ = process_subject(sid, subject_dir, OUT_ROOT, ALLOWED, None, None)
    for r in results:
        tag = f"[{r.status.upper()}]"
        print(f"  {tag}  {r.nii_path.name}", end="")
        if r.error_message:
            print(f"  -- {r.error_message}", end="")
        if r.size_mb:
            print(f"  ({r.size_mb} MB)", end="")
        print()
