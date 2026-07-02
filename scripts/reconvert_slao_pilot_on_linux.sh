#!/usr/bin/env bash
# Re-convert the 11 LAA pilot SLAO cases from source DICOM with dcm2niix.
# RUN THIS ON THE LINUX BOX where the SLAO DICOM lives (Research13T).
#
# Why: the slaobids NIfTIs were made with SimpleITK's ImageSeriesReader, which
# takes Z spacing from only the first slice gap -> wrong/deformed Z on series
# with non-uniform/overlapping slices (e.g. sub-142). dcm2niix derives geometry
# correctly. After running, copy the OUT folder back to the Mac and I'll repoint
# the pilot sessions + regenerate VISTA3D candidates on the corrected CTAs.
#
# Usage:
#   bash reconvert_slao_pilot_on_linux.sh [DICOM_ROOT] [OUT_DIR]
# Defaults:
#   DICOM_ROOT=/media/fridmans/Research13T/datasets/SLAODICOM
#   OUT_DIR=./slao_pilot_fixed
set -euo pipefail

DICOM_ROOT="${1:-/media/fridmans/Research13T/datasets/SLAODICOM}"
OUT_DIR="${2:-./slao_pilot_fixed}"
PILOT_IDS="138 142 255 257 260 278 284 291 294 302 547"

command -v dcm2niix >/dev/null 2>&1 || { echo "ERROR: dcm2niix not on PATH"; exit 1; }
mkdir -p "$OUT_DIR"
echo "dcm2niix: $(dcm2niix --version 2>&1 | tail -1)"
echo "DICOM_ROOT=$DICOM_ROOT  OUT_DIR=$OUT_DIR"
echo

for id in $PILOT_IDS; do
  src="$DICOM_ROOT/$id"
  if [ ! -d "$src" ]; then echo "[skip] sub-$id: no DICOM dir at $src"; continue; fi
  echo "[run ] sub-$id"
  # -z y gzip, -m y merge slices into one volume, -b n no BIDS json,
  # filename includes series# and #slices so you can spot the CTA if multiple.
  dcm2niix -z y -m y -b n -f "sub-${id}_acq-CTA_ct_s%s_n%r" -o "$OUT_DIR" "$src" \
    >/dev/null 2>"$OUT_DIR/sub-${id}.log" || echo "    dcm2niix rc=$? (see sub-${id}.log)"
done

echo
echo "=== outputs (keep the largest-slice volume per subject as the CTA) ==="
ls -la "$OUT_DIR"/*.nii.gz 2>/dev/null || echo "  (none produced)"
echo
echo "Next: copy '$OUT_DIR' back to the Mac, e.g.:"
echo "  scp -r <linuxhost>:$PWD/$OUT_DIR  <repo>/outputs/laa_pilot_fixed/"
echo "Then on the Mac I will repoint sessions + run: run_laa_pilot_candidates.py --force"
