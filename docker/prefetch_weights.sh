#!/usr/bin/env bash
# ============================================================
# prefetch_weights.sh
#
# Download all model weights into the named Docker volumes
# BEFORE running the pipeline, so the container never needs
# internet access during batch processing.
#
# Run once on the DGX Spark after pulling the image:
#   bash docker/prefetch_weights.sh
#
# Override the LAA data path if needed:
#   LAA_DATA_DIR=/nvme/LAA bash docker/prefetch_weights.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="ai-cta-stroke-laa:latest"
LAA_DATA_DIR="${LAA_DATA_DIR:-/mnt/LAA}"

cd "${REPO_ROOT}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[prefetch] Image not found – building first ..."
    docker build -f docker/Dockerfile -t "${IMAGE}" .
fi

DOCKER_ARGS=(
    --rm
    --runtime nvidia
    --gpus all
    --shm-size 4g
    -e NVIDIA_VISIBLE_DEVICES=all
    -e TOTALSEG_WEIGHTS_PATH=/workspace/weights/totalseg
    -e NV_SEGMENT_CT_CACHE=/workspace/weights/nv_segment_ct
    -e HF_HOME=/workspace/weights/huggingface
    -v "${LAA_DATA_DIR}:/data/LAA"
    -v "${REPO_ROOT}:/workspace/AI_CTA_Stroke"
    -v ai_cta_weights_totalseg:/workspace/weights/totalseg
    -v ai_cta_weights_nv_segment_ct:/workspace/weights/nv_segment_ct
    -v ai_cta_weights_huggingface:/workspace/weights/huggingface
    -w /workspace/AI_CTA_Stroke
)

echo "=== [1/3] TotalSegmentator: total task ==="
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
    python -c "
import os, pathlib
os.environ['TOTALSEG_WEIGHTS_PATH'] = '/workspace/weights/totalseg'
from totalsegmentator.python_api import totalsegmentator
# Trigger weight download without actual data by checking the model dir
import totalsegmentator.weights as w
print('Downloading TotalSegmentator total weights ...')
w.get_weights_dir()
" || echo "[WARN] TotalSegmentator weight pre-fetch skipped (may download on first run)"

echo ""
echo "=== [2/3] TotalSegmentator: heartchambers_highres task ==="
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
    python -c "
import os
os.environ['TOTALSEG_WEIGHTS_PATH'] = '/workspace/weights/totalseg'
# heartchambers_highres requires an academic license key.
# Set TOTALSEG_LICENSE_KEY env var or it will prompt interactively.
license_key = os.environ.get('TOTALSEG_LICENSE_KEY', '')
if not license_key:
    print('[SKIP] TOTALSEG_LICENSE_KEY not set – run with:')
    print('  TOTALSEG_LICENSE_KEY=<your_key> bash docker/prefetch_weights.sh')
else:
    from totalsegmentator.python_api import totalsegmentator
    print('License key found – heartchambers_highres weights will download on first use.')
" || echo "[WARN] heartchambers_highres weight check skipped"

echo ""
echo "=== [3/3] NV-Segment-CT (VISTA3D) weights from HuggingFace ==="
docker run "${DOCKER_ARGS[@]}" "${IMAGE}" \
    python -c "
import os
os.environ['HF_HOME'] = '/workspace/weights/huggingface'
os.environ['NV_SEGMENT_CT_CACHE'] = '/workspace/weights/nv_segment_ct'
from huggingface_hub import snapshot_download
print('Downloading nvidia/NV-Segment-CT ...')
snapshot_download(
    repo_id='nvidia/NV-Segment-CT',
    local_dir='/workspace/weights/nv_segment_ct',
    ignore_patterns=['*.bin'],   # prefer .safetensors
)
print('Done.')
"

echo ""
echo "=== Prefetch complete ==="
echo "Named volumes populated:"
echo "  ai_cta_weights_totalseg"
echo "  ai_cta_weights_nv_segment_ct"
echo "  ai_cta_weights_huggingface"
