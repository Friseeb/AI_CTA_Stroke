#!/usr/bin/env bash
set -euo pipefail

# Create a CUDA env for TotalSegmentator / MONAI on NVIDIA (e.g., 3090 x2).
# Usage: bash scripts/setup_totalseg_env_cuda.sh [env_name] [cuda_tag]
# cuda_tag examples: cu121, cu118 (match your driver/toolkit)

ENV_NAME="${1:-totalseg-gpu}"
CUDA_TAG="${2:-cu121}"
PY_VER="3.10"
TORCH_VER="2.2.2"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python="${PY_VER}"
conda activate "${ENV_NAME}"

# Install torch/torchvision matching your CUDA.
pip install "torch==${TORCH_VER}" "torchvision==0.17.2" --index-url "${TORCH_INDEX}"

# Core packages
pip install totalsegmentator monai nibabel SimpleITK

cat <<EOF

Env ready: ${ENV_NAME}
Check GPU visibility:
  conda activate ${ENV_NAME}
  python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
PY

If torch cannot see GPUs, align CUDA_TAG with your driver (cu118/cu121), and ensure drivers are current.
Set TOTALSEG_FORCE_GPU=1 to avoid CPU fallback when GPUs exist.
EOF
