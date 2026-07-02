#!/usr/bin/env bash
# ============================================================
# setup_env.sh — create the laa-pipeline conda environment
#
# Auto-detects: OS, CPU architecture, and CUDA version, then
# installs the correct PyTorch variant and all pipeline deps.
#
# Usage:
#   bash scripts/setup_env.sh              # create laa-pipeline
#   bash scripts/setup_env.sh my-env       # custom name
#   bash scripts/setup_env.sh laa-pipeline --force  # recreate
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-laa-pipeline}"
FORCE="${2:-}"

# ---- locate conda -----------------------------------------
if command -v conda >/dev/null 2>&1; then
    CONDA="$(command -v conda)"
elif [[ -f "$HOME/miniforge3/bin/conda" ]]; then
    CONDA="$HOME/miniforge3/bin/conda"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
else
    echo "conda not found. Install Miniforge first:"
    echo "  # Linux aarch64:"
    echo "  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh | bash"
    echo "  # Linux x86_64 / macOS:"
    echo "  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh | bash"
    exit 1
fi

CONDA_BASE="$("${CONDA}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- detect platform + GPU --------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

detect_cuda_version() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        # e.g. "CUDA Version: 13.0" → "13.0"
        nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1
    fi
}

CUDA_VER="$(detect_cuda_version || true)"
CUDA_MAJOR="${CUDA_VER%%.*}"

# Determine torch index URL
if [[ "$OS" == "Darwin" ]]; then
    PLATFORM="macos"
    TORCH_INDEX=""          # macOS: standard PyPI includes MPS support
    ENV_YML="envs/macos.yml"
elif [[ "$ARCH" == "aarch64" ]] && [[ -n "$CUDA_MAJOR" ]]; then
    PLATFORM="linux_cuda_arm64"
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    ENV_YML="envs/linux_cuda_arm64.yml"
elif [[ "$ARCH" == "x86_64" ]] && [[ -n "$CUDA_MAJOR" ]] && [[ "$CUDA_MAJOR" -ge 12 ]]; then
    PLATFORM="linux_cuda_x86"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    ENV_YML="envs/linux_cuda_x86.yml"
else
    PLATFORM="cpu"
    TORCH_INDEX=""
    ENV_YML="envs/cpu.yml"
fi

echo "========================================"
echo "  Platform:    $PLATFORM"
echo "  OS/Arch:     $OS / $ARCH"
echo "  CUDA version: ${CUDA_VER:-none}"
echo "  Torch index:  ${TORCH_INDEX:-default PyPI}"
echo "  Env name:     $ENV_NAME"
echo "========================================"

# ---- remove existing env if --force -----------------------
if [[ "$FORCE" == "--force" ]]; then
    echo "[setup] Removing existing env '$ENV_NAME' ..."
    conda env remove -y -n "$ENV_NAME" 2>/dev/null || true
fi

# ---- 1. Base conda env (no torch yet) ----------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] Env '$ENV_NAME' already exists. Skipping conda create."
    echo "        Run with --force to recreate from scratch."
else
    echo "[setup] Creating conda env '$ENV_NAME' (Python 3.10) ..."
    conda create -y -n "$ENV_NAME" -c conda-forge \
        python=3.10 \
        nibabel SimpleITK pydicom \
        numpy scipy networkx pandas tqdm \
        scikit-image scikit-learn matplotlib pyyaml \
        vtk trimesh rtree
fi

PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# ---- 2. PyTorch -------------------------------------------
echo "[setup] Installing PyTorch (${PLATFORM}) ..."
if [[ -n "$TORCH_INDEX" ]]; then
    "${PIP}" install --index-url "$TORCH_INDEX" torch torchvision
elif [[ "$OS" == "Darwin" ]]; then
    # macOS: plain install; includes MPS kernels
    "${PIP}" install torch torchvision
    echo "  → On Apple Silicon set: export PYTORCH_ENABLE_MPS_FALLBACK=1"
else
    "${PIP}" install torch torchvision
fi

# ---- 3. Medical imaging stack -----------------------------
echo "[setup] Installing MONAI, VISTA3D deps, TotalSegmentator ..."
"${PIP}" install \
    "monai>=1.6.0" \
    "transformers>=4.40,<5" \
    huggingface_hub einops safetensors tokenizers \
    TotalSegmentator

if [[ "$OS" != "Darwin" ]] || [[ "$PLATFORM" != "windows" ]]; then
    "${PIP}" install CardiacCTExplorer 2>/dev/null \
        || echo "[warn] CardiacCTExplorer install failed — skipping (optional)"
fi

# ---- 4. Dental + app deps ---------------------------------
echo "[setup] Installing dental pipeline and Streamlit UI deps ..."
"${PIP}" install \
    typer rich "pydantic>=2.6" \
    pyradiomics \
    "streamlit>=1.35" plotly

# ---- 5. Local packages ------------------------------------
echo "[setup] Installing local packages ..."
"${PIP}" install -e "${REPO_ROOT}/cta_common" 2>/dev/null || true
"${PIP}" install -e "${REPO_ROOT}/subprojects/cta-dental-opportunistic-screening" 2>/dev/null || true
"${PIP}" install -e "${REPO_ROOT}"

# ---- 6. Verify GPU ----------------------------------------
echo ""
echo "[setup] Verifying installation ..."
"${CONDA_BASE}/envs/${ENV_NAME}/bin/python" -c "
import torch
cuda = torch.cuda.is_available()
mps  = getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {cuda}' + (f'  [{torch.cuda.get_device_name(0)}]' if cuda else ''))
print(f'  MPS  available: {mps}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate:  conda activate ${ENV_NAME}"
echo "Run UI:    bash scripts/run_app.sh"
echo ""
echo "Run dental on one case:"
echo "  PATH=\"\${CONDA_BASE}/envs/${ENV_NAME}/bin:\$PATH\" \\"
echo "  cta-dental run <input.nii.gz> --out <outdir> --case-id <id>"
