#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="topcow_claim"
CUDA=0
CLONE=1
FORCE=0

usage() {
  cat <<'EOF'
Usage: setup_topcow_env.sh [options]

Options:
  --env-name NAME   Conda environment name (default: topcow_claim)
  --cuda            Install CUDA-enabled PyTorch (uses cu121 index)
  --cpu             Install CPU-only PyTorch (default)
  --no-clone        Skip cloning TopCoW CLAIM repo
  --force           Recreate env and re-pull repo if present
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --cuda)
      CUDA=1
      shift
      ;;
    --cpu)
      CUDA=0
      shift
      ;;
    --no-clone)
      CLONE=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE}" -eq 1 ]]; then
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "Env ${ENV_NAME} already exists."
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.topcow_claim.yml"
fi

conda activate "${ENV_NAME}"

OS="$(uname -s)"
if [[ "${CUDA}" -eq 1 ]]; then
  if [[ "${OS}" == "Darwin" ]]; then
    echo "CUDA wheels not available on macOS; installing CPU PyTorch."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  fi
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# nnUNet v2 expects these paths to exist
export nnUNet_raw="${HOME}/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
export nnUNet_results="${HOME}/nnUNet_results"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

CLAIM_DIR="${ROOT_DIR}/external/topcow_claim"
if [[ "${CLONE}" -eq 1 ]]; then
  if [[ -d "${CLAIM_DIR}" ]]; then
    if [[ "${FORCE}" -eq 1 ]]; then
      git -C "${CLAIM_DIR}" pull
    fi
  else
    git clone https://github.com/claim-berlin/TopCoW_2024_MRA_winning_solution.git "${CLAIM_DIR}"
  fi

  if [[ -d "${CLAIM_DIR}/topcow-2024-nnunet" ]]; then
    pip install -e "${CLAIM_DIR}/topcow-2024-nnunet"
  else
    echo "Missing topcow-2024-nnunet directory in ${CLAIM_DIR}."
    exit 1
  fi
else
  echo "Skipped cloning TopCoW CLAIM repo (--no-clone)."
fi

echo ""
echo "TopCoW environment ready."
echo "Next:"
echo "  1) Download weights (Zenodo 14191592)"
echo "  2) Run: python ${ROOT_DIR}/scripts/run_topcow_claim.py --help"
