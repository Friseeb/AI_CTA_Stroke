#!/usr/bin/env bash
set -euo pipefail

# Create a dedicated env for CardiacCTExplorer.
# Usage: bash scripts/setup_cardiac_ct_explorer_env.sh [env_name]

ENV_NAME="${1:-cardiac-ct-explorer}"
PY_VER="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Env ${ENV_NAME} already exists. Reusing."
else
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

conda activate "${ENV_NAME}"

# Torch (CPU wheel for macOS; GPU users can install CUDA build manually)
pip install torch torchvision --extra-index-url "${TORCH_INDEX}"

# CardiacCTExplorer and deps (prefer local clone if present)
if [ -d "AI_CTA_Stroke/external/cardiac_ct_explorer" ]; then
  pip install -e "AI_CTA_Stroke/external/cardiac_ct_explorer"
else
  pip install CardiacCTExplorer
fi

# Extra deps for IO
pip install nibabel numpy scipy

cat <<'EOF'

Env ready: ${ENV_NAME}
Reminder: TotalSegmentator requires an academic license key for heartchambers_highres.
See: https://backend.totalsegmentator.com/license-academic/
EOF
