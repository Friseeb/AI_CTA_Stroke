#!/usr/bin/env bash
# ============================================================
# run.sh – build and/or launch the LAA container on DGX Spark
#
# Usage:
#   bash docker/run.sh              # interactive shell
#   bash docker/run.sh build        # (re)build image only
#   bash docker/run.sh <cmd...>     # run a specific command
#
# The LAA data volume is expected at /mnt/LAA on the host.
# Override with:  LAA_DATA_DIR=/nvme/my_laa bash docker/run.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="ai-cta-stroke-laa:latest"
LAA_DATA_DIR="${LAA_DATA_DIR:-/mnt/LAA}"

cd "${REPO_ROOT}"

if [[ "${1:-}" == "build" ]]; then
    echo "[run.sh] Building ${IMAGE} ..."
    docker build -f docker/Dockerfile -t "${IMAGE}" .
    echo "[run.sh] Build complete."
    exit 0
fi

# Build if image is missing
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[run.sh] Image not found – building first ..."
    docker build -f docker/Dockerfile -t "${IMAGE}" .
fi

# Verify the LAA data directory exists on the host
if [[ ! -d "${LAA_DATA_DIR}" ]]; then
    echo "[run.sh] WARNING: LAA data directory not found at ${LAA_DATA_DIR}"
    echo "         Set LAA_DATA_DIR=<path> to override, or create ${LAA_DATA_DIR}"
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
    # LAA data volume
    -v "${LAA_DATA_DIR}:/data/LAA"
    # Live code mount – remove for production
    -v "${REPO_ROOT}:/workspace/AI_CTA_Stroke"
    # Shared weight caches
    -v ai_cta_weights_totalseg:/workspace/weights/totalseg
    -v ai_cta_weights_nv_segment_ct:/workspace/weights/nv_segment_ct
    -v ai_cta_weights_huggingface:/workspace/weights/huggingface
    -w /workspace/AI_CTA_Stroke
)

if [[ $# -eq 0 ]]; then
    # Interactive shell
    docker run -it "${DOCKER_ARGS[@]}" "${IMAGE}" bash
else
    docker run -it "${DOCKER_ARGS[@]}" "${IMAGE}" "$@"
fi
