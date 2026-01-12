#!/usr/bin/env bash
set -euo pipefail

# Create an Apple Silicon (MPS/CPU) env for TotalSegmentator/ MONAI.
# Usage: bash scripts/setup_totalseg_env_mac.sh [env_name]

ENV_NAME="${1:-totalseg-mac}"
PY_VER="3.10"
TORCH_VER="2.2.2"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"  # MPS uses the CPU wheel entrypoint

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python="${PY_VER}"
conda activate "${ENV_NAME}"

# Install torch/torchvision with MPS support (macOS arm64, metal backend).
pip install "torch==${TORCH_VER}" "torchvision==0.17.2" --extra-index-url "${TORCH_INDEX}"

# Core packages
pip install totalsegmentator monai nibabel SimpleITK

# Helpful env flags for MPS
cat <<'EOF'

Env ready: ${ENV_NAME}
Before running, consider:
  export PYTORCH_ENABLE_MPS_FALLBACK=1   # fallback CPU ops instead of crashing
EOF
