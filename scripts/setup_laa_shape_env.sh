#!/usr/bin/env bash
set -euo pipefail

# Create a dedicated env for LAA shape descriptors (STACOM2025 scripts).
# Usage: bash scripts/setup_laa_shape_env.sh [env_name]

ENV_NAME="${1:-laa-shape}"
PY_VER="3.10"

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

# Core deps used by the STACOM2025 scripts
pip install numpy SimpleITK vtk scikit-learn matplotlib

cat <<'EOF'

Env ready: ${ENV_NAME}
EOF
