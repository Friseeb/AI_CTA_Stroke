#!/usr/bin/env bash
set -euo pipefail

# Create a separate env for NV-Segment-CT (VISTA3D) inference.
# Usage: bash scripts/setup_nv_segment_ct_env.sh [env_name]

ENV_NAME="${1:-nv-segment-ct}"
PY_VER="3.10"
TORCH_VER="2.6.0"
TORCHVISION_VER="0.21.0"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"  # macOS wheels include MPS support

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python="${PY_VER}"
conda activate "${ENV_NAME}"

pip install "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" --extra-index-url "${TORCH_INDEX}"
pip install monai==1.5.0 transformers huggingface_hub nibabel numpy scipy einops safetensors

cat <<'EOF'

Env ready: ${ENV_NAME}
Tips:
  export PYTORCH_ENABLE_MPS_FALLBACK=1   # fallback to CPU if MPS op missing
EOF
