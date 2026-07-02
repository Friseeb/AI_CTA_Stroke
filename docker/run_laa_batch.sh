#!/usr/bin/env bash
# ============================================================
# run_laa_batch.sh
#
# Run the full LAA pipeline (NUDF + VISTA3D + prior fusion)
# over all cases in the LAA volume.
#
# Expected volume layout (/mnt/LAA  →  /data/LAA inside container):
#
#   /data/LAA/
#     raw/                         ← input CTA NIfTIs
#       sub-001_acq-CTA_ct.nii.gz
#       sub-002_acq-CTA_ct.nii.gz
#       ...
#     derivatives/
#       defaced/                   ← defaced outputs (auto-created)
#       nudf_la/                   ← NUDF + TotalSegmentator outputs
#       nv_segment_ct/             ← VISTA3D LAA masks
#       prior_fusion/              ← consensus / union / intersection
#
# Usage:
#   bash docker/run_laa_batch.sh [--dry-run] [--limit N] [--workers N]
#   bash docker/run_laa_batch.sh --skip-deface      # if already defaced
#   bash docker/run_laa_batch.sh --stage nudf        # nudf only
#   bash docker/run_laa_batch.sh --stage vista3d     # vista3d only
#   bash docker/run_laa_batch.sh --stage prior       # prior fusion only
#   bash docker/run_laa_batch.sh --stage all         # all (default)
#
# Environment vars:
#   LAA_DATA_DIR   – host path to the LAA volume   (default /mnt/LAA)
#   TOTALSEG_LICENSE_KEY – academic license key for heartchambers_highres
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="ai-cta-stroke-laa:latest"
LAA_DATA_DIR="${LAA_DATA_DIR:-/mnt/LAA}"

# ---- parse args ----
DRY_RUN=""
LIMIT=""
WORKERS=4
SKIP_DEFACE=0
STAGE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN="--dry-run"; shift ;;
        --limit)     LIMIT="--limit $2"; shift 2 ;;
        --workers)   WORKERS="$2"; shift 2 ;;
        --skip-deface) SKIP_DEFACE=1; shift ;;
        --stage)     STAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- sanity checks ----
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[run_laa_batch] Image not found – run:  bash docker/run.sh build"
    exit 1
fi
if [[ ! -d "${LAA_DATA_DIR}" ]]; then
    echo "[run_laa_batch] LAA data dir not found: ${LAA_DATA_DIR}"
    echo "  Set LAA_DATA_DIR=/your/path to override."
    exit 1
fi

DOCKER_ARGS=(
    --rm
    --runtime nvidia
    --gpus all
    --shm-size 16g
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e TOTALSEG_WEIGHTS_PATH=/workspace/weights/totalseg
    -e NV_SEGMENT_CT_CACHE=/workspace/weights/nv_segment_ct
    -e HF_HOME=/workspace/weights/huggingface
    -e "TOTALSEG_LICENSE_KEY=${TOTALSEG_LICENSE_KEY:-}"
    -v "${LAA_DATA_DIR}:/data/LAA"
    -v "${REPO_ROOT}:/workspace/AI_CTA_Stroke"
    -v ai_cta_weights_totalseg:/workspace/weights/totalseg
    -v ai_cta_weights_nv_segment_ct:/workspace/weights/nv_segment_ct
    -v ai_cta_weights_huggingface:/workspace/weights/huggingface
    -w /workspace/AI_CTA_Stroke
)

# ---- stage helpers ----
run_stage() {
    echo ""
    echo "================================================================"
    echo "  STAGE: $1"
    echo "================================================================"
    docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${@:2}"
}

# ---- Stage 0: deface ----
if [[ "${SKIP_DEFACE}" -eq 0 ]] && [[ "${STAGE}" == "all" || "${STAGE}" == "deface" ]]; then
    run_stage "deface" \
        python scripts/run_cta_deface_dl_batch.py \
            --input-dir  /data/LAA/raw \
            --output-dir /data/LAA/derivatives/defaced \
            --mask-dir   /data/LAA/derivatives/deface_masks \
            ${DRY_RUN} ${LIMIT}
fi

# ---- Stage 1: NUDF + TotalSegmentator ----
if [[ "${STAGE}" == "all" || "${STAGE}" == "nudf" ]]; then
    run_stage "nudf" \
        python scripts/run_daylightbids_nudf_la_batch.py \
            --root       /data/LAA \
            --input-dir  /data/LAA/derivatives/defaced \
            --input-glob "*_defaced.nii.gz" \
            --out-dir    /data/LAA/derivatives/nudf_la \
            --device     gpu \
            --totalseg-device gpu \
            --run-highres-heart \
            --quiet-subprocess \
            --subprocess-log-dir /data/LAA/derivatives/nudf_la/_logs \
            --incremental-summary \
            ${DRY_RUN} ${LIMIT}
fi

# ---- Stage 2: VISTA3D (NV-Segment-CT) LAA ----
if [[ "${STAGE}" == "all" || "${STAGE}" == "vista3d" ]]; then
    run_stage "vista3d" \
        python - <<'PYEOF'
import subprocess, sys
from pathlib import Path

defaced_dir = Path("/data/LAA/derivatives/defaced")
out_root = Path("/data/LAA/derivatives/nv_segment_ct")
out_root.mkdir(parents=True, exist_ok=True)
cases = sorted(defaced_dir.glob("*_defaced.nii.gz"))
print(f"Found {len(cases)} defaced cases")

for case_path in cases:
    case_id = case_path.name.replace("_defaced.nii.gz", "")
    out_path = out_root / f"{case_id}_laa108.nii.gz"
    if out_path.exists():
        print(f"[SKIP] {case_id}")
        continue
    print(f"[RUN]  {case_id}")
    rc = subprocess.call([
        sys.executable, "scripts/run_nv_segment_ct_laa.py",
        "--input",     str(case_path),
        "--output",    str(out_path),
        "--model-dir", "/workspace/weights/nv_segment_ct",
        "--device",    "auto",
    ])
    if rc != 0:
        print(f"[FAIL] {case_id} exit={rc}")
PYEOF
fi

# ---- Stage 3: prior fusion ----
if [[ "${STAGE}" == "all" || "${STAGE}" == "prior" ]]; then
    run_stage "prior fusion" \
        python - <<'PYEOF'
import subprocess, sys
from pathlib import Path

defaced_dir    = Path("/data/LAA/derivatives/defaced")
nudf_root      = Path("/data/LAA/derivatives/nudf_la")
vista3d_root   = Path("/data/LAA/derivatives/nv_segment_ct")
fusion_root    = Path("/data/LAA/derivatives/prior_fusion")
fusion_root.mkdir(parents=True, exist_ok=True)

cases = sorted(defaced_dir.glob("*_defaced.nii.gz"))
print(f"Found {len(cases)} defaced cases for prior fusion")

for case_path in cases:
    case_id = case_path.name.replace("_defaced.nii.gz", "")
    out_dir = fusion_root / case_id
    summary = out_dir / f"{case_id}_prior_fusion_summary.json"
    if summary.exists():
        print(f"[SKIP] {case_id}")
        continue

    nudf_laa       = nudf_root / case_id / f"{case_id}_laa_nudf.nii.gz"
    totalseg_total = nudf_root / case_id / "totalseg_total"
    totalseg_heart = nudf_root / case_id / "totalseg_heartchambers_highres"
    totalseg_cor   = nudf_root / case_id / "totalseg_coronary_arteries"
    vista3d_out    = vista3d_root / f"{case_id}_laa108.nii.gz"

    cmd = [
        sys.executable, "scripts/run_prior_fusion.py",
        "--case-id",            case_id,
        "--input",              str(case_path),
        "--out-dir",            str(out_dir),
        "--device",             "gpu",
    ]
    if nudf_laa.exists():
        cmd += ["--nudf-laa", str(nudf_laa)]
    if totalseg_total.exists():
        cmd += ["--totalseg-total-dir", str(totalseg_total)]
    if totalseg_heart.exists():
        cmd += ["--totalseg-heart-dir", str(totalseg_heart)]
    if totalseg_cor.exists():
        cmd += ["--totalseg-coronary-dir", str(totalseg_cor)]
    if vista3d_out.exists():
        cmd += ["--vista3d-combined", str(vista3d_out)]
    if not totalseg_heart.exists():
        cmd += ["--run-missing-tasks"]

    print(f"[RUN]  {case_id}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[FAIL] {case_id} exit={rc}")
PYEOF
fi

echo ""
echo "=== LAA batch complete ==="
echo "Outputs in: ${LAA_DATA_DIR}/derivatives/"
