#!/usr/bin/env bash
# ============================================================
# setup_dgx_spark_env.sh
#
# Create the unified LAA + dental pipeline conda environment
# for the DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13.0).
#
# Usage:
#   bash scripts/setup_dgx_spark_env.sh          # creates laa-pipeline
#   bash scripts/setup_dgx_spark_env.sh my-env   # custom name
#
# Why a script instead of just `conda env create -f environment.dgx_spark.yml`:
#   conda-forge resolves torch to a CPU-only wheel on aarch64.
#   We must pip-install torch from the cu130 index AFTER conda creates the base.
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-laa-pipeline}"
CONDA="${CONDA_EXE:-$(command -v conda)}"

if [[ -z "${CONDA}" ]]; then
    echo "conda not found. Install Miniforge first:" >&2
    echo "  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh | bash" >&2
    exit 1
fi

CONDA_BASE="$("${CONDA}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- 1. Create base env ----------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] Env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[setup] Creating conda env '${ENV_NAME}' (Python 3.10) ..."
    conda create -y -n "${ENV_NAME}" -c conda-forge \
        python=3.10 \
        nibabel SimpleITK pydicom \
        numpy scipy networkx pandas tqdm \
        scikit-image scikit-learn matplotlib pyyaml \
        vtk trimesh rtree
fi

conda activate "${ENV_NAME}"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# ---- 2. PyTorch with CUDA 13.0 (Blackwell / sm_100) -----------
echo "[setup] Installing PyTorch cu130 for aarch64 Blackwell ..."
"${PIP}" install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision

# ---- 3. MONAI (>=1.6.0 required; 1.5.0 pins torch<2.7) --------
echo "[setup] Installing MONAI and VISTA3D deps ..."
"${PIP}" install \
    "monai>=1.6.0" \
    "transformers>=4.40,<5" \
    huggingface_hub einops safetensors tokenizers

# ---- 4. Segmentation tools ------------------------------------
echo "[setup] Installing TotalSegmentator and CardiacCTExplorer ..."
"${PIP}" install TotalSegmentator CardiacCTExplorer

# ---- 5. Dental pipeline CLI deps ------------------------------
echo "[setup] Installing dental pipeline deps ..."
"${PIP}" install typer rich "pydantic>=2.6" pyradiomics

# ---- 6. Local packages (cta_common + dental) ------------------
echo "[setup] Installing local packages ..."
"${PIP}" install -e "${REPO_ROOT}/cta_common"
"${PIP}" install -e "${REPO_ROOT}/subprojects/cta-dental-opportunistic-screening"
"${PIP}" install -e "${REPO_ROOT}"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo ""
echo "Verify GPU:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
echo ""
echo "Run dental on one case:"
echo "  PATH=\"\${CONDA_BASE}/envs/${ENV_NAME}/bin:\$PATH\" \\"
echo "  cta-dental run <input.nii.gz> --out <outdir> --case-id <id>"
