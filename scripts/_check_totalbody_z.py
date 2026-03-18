"""
Check all TotalBodyCT-converted subjects: compute z_coverage for the
ctbody export, then simulate whether the proposed ThoraxCT change
(z_max 500→750, add "cardiac cap" keyword) would reclassify any of them.
"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from convert_daylightdicom_to_bids import (
    collect_series, classify_export_study, _sample_z_range,
    find_export_dirs, SCAN_TYPE_CONFIGS, SCAN_TYPE_MAP,
)

MANIFEST = Path(r'C:\Users\spost\Desktop\CT_image\SLAAOBIDS\conversion_manifest_20260317_123031.csv')
BIDS_ROOT = Path(r'C:\Users\spost\Desktop\CT_image\SLAAOBIDS')

# ── Read manifest for TotalBodyCT-converted subjects ────────────────────────
totalbody_subjects = []
with open(MANIFEST, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('ctbody_status', '').strip() == 'converted':
            totalbody_subjects.append(row['subject_id'])

print(f"TotalBodyCT-converted subjects: {len(totalbody_subjects)}")
print(f"IDs: {totalbody_subjects}\n")

# ── For each subject, find the export that was classified as TotalBodyCT ────
# We re-run classification on each export to find the TotalBodyCT one,
# then record z_coverage and check ThoraxCT keyword match.

ALLOWED_ALL = [c.name for c in SCAN_TYPE_CONFIGS]

# Proposed new ThoraxCT config parameters
THORAX_Z_MIN = 200.0
THORAX_Z_MAX_NEW = 750.0   # was 500
THORAX_KEYWORDS_NEW = tuple(list(SCAN_TYPE_MAP['ThoraxCT'].study_desc_keywords) + ["cardiac cap"])

risky = []  # would be reclassified

for subj_id in totalbody_subjects:
    # DICOM source dir is on D:/
    subject_dir = Path(f'D:/{subj_id}')
    if not subject_dir.exists():
        print(f"  sub-{subj_id}: DICOM dir not found at {subject_dir}")
        continue

    export_dirs = sorted(subject_dir.glob('Export_*'))
    for export_dir in export_dirs:
        series_map = collect_series(export_dir)
        if not series_map:
            continue

        current_type = classify_export_study(series_map, ALLOWED_ALL)
        if current_type != 'TotalBodyCT':
            continue

        # This is the TotalBodyCT export — get z_coverage and study text
        cov = _sample_z_range(series_map)

        study_descs = " ".join(
            c.meta.get("study_description", "") for c in series_map.values()
        ).lower()
        proto_names = " ".join(
            c.meta.get("protocol_name", "") for c in series_map.values()
        ).lower()

        # Would this now score for ThoraxCT under new rules?
        # (1) z check with new z_max=750
        if cov is not None:
            z_pass_new = THORAX_Z_MIN <= cov <= THORAX_Z_MAX_NEW
        else:
            z_pass_new = True  # no exclusion if unknown

        # (2) keyword check with new keyword list
        kw_match = any(kw in study_descs for kw in THORAX_KEYWORDS_NEW)
        proto_match = any(kw in proto_names for kw in SCAN_TYPE_MAP['ThoraxCT'].protocol_keywords)
        thorax_score = 0
        if z_pass_new:
            if kw_match: thorax_score += 150
            if proto_match: thorax_score += 80

        # Current TotalBodyCT score (unchanged)
        totalbody_cfg = SCAN_TYPE_MAP['TotalBodyCT']
        tb_z_pass = True
        if cov is not None:
            if totalbody_cfg.z_coverage_min_mm and cov < totalbody_cfg.z_coverage_min_mm:
                tb_z_pass = False
            if totalbody_cfg.z_coverage_max_mm and cov > totalbody_cfg.z_coverage_max_mm:
                tb_z_pass = False
        tb_score = 0
        if tb_z_pass:
            if any(kw in study_descs for kw in totalbody_cfg.study_desc_keywords): tb_score += 150
            if any(kw in proto_names for kw in totalbody_cfg.protocol_keywords): tb_score += 80

        cov_str = f"{cov:.0f}" if cov is not None else "?"
        status = ""
        if thorax_score > 0 and thorax_score >= tb_score:
            status = "  *** WOULD RECLASSIFY TO ThoraxCT ***"
            risky.append((subj_id, cov_str, study_descs[:80], proto_names[:60], thorax_score, tb_score))
        elif thorax_score > 0:
            status = f"  (ThoraxCT scores {thorax_score} but TotalBodyCT scores {tb_score} — stays TotalBodyCT)"

        print(f"sub-{subj_id:>4}  z={cov_str:>4}mm  "
              f"study={study_descs[:50]!r}  "
              f"proto={proto_names[:40]!r}{status}")
        break  # found the TotalBodyCT export for this subject

print(f"\n{'='*70}")
print(f"Subjects that WOULD be reclassified as ThoraxCT: {len(risky)}")
for subj_id, cov, sd, pn, ts, tbs in risky:
    print(f"  sub-{subj_id}  z={cov}mm  study={sd!r}  proto={pn!r}")
    print(f"    ThoraxCT score={ts}  TotalBodyCT score={tbs}")
