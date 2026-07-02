# ============================================================
# AI_CTA_Stroke – LAA + dental pipeline image for DGX Spark
# Base: NVIDIA PyTorch 25.04 (CUDA 12.8 → driver-compat with
#       CUDA 13.0 on GB10 Grace Blackwell, sm_100, aarch64).
#
# The NGC base image already ships torch with CUDA. We upgrade
# to the cu130 wheel so sm_100 kernels are available, then add
# the medical-imaging stack on top.
# ============================================================
FROM nvcr.io/nvidia/pytorch:25.04-py3

# ---------- OS deps ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        dcm2niix \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python tooling ------------------------------------
RUN pip install --no-cache-dir --upgrade pip

# ---------- PyTorch: upgrade to cu130 for GB10 (sm_100) ------
# The 25.04 base ships torch with CUDA 12.8; cu130 wheels add
# full Blackwell (sm_100) support and match the DGX Spark driver.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch torchvision

# ---------- Core I/O + numerics ------------------------------
RUN pip install --no-cache-dir \
        "nibabel>=5.2" \
        "numpy>=1.25" \
        "scipy>=1.10" \
        "networkx>=3.2" \
        pandas \
        tqdm \
        pydicom \
        SimpleITK \
        scikit-image \
        scikit-learn \
        matplotlib \
        pyyaml

# ---------- MONAI + VISTA3D (NV-Segment-CT) ------------------
# monai>=1.6.0: 1.5.0 pins torch<2.7 which conflicts with cu130 builds.
# transformers<5: 5.x renames model attributes that VISTA3D relies on.
RUN pip install --no-cache-dir \
        "monai>=1.6.0" \
        "transformers>=4.40,<5" \
        huggingface_hub \
        einops \
        safetensors \
        tokenizers

# ---------- Segmentation models ------------------------------
RUN pip install --no-cache-dir TotalSegmentator
RUN pip install --no-cache-dir CardiacCTExplorer

# ---------- Shape / mesh + radiomics -------------------------
RUN pip install --no-cache-dir \
        vtk \
        trimesh \
        rtree \
        pyradiomics

# ---------- Dental pipeline CLI deps -------------------------
RUN pip install --no-cache-dir \
        typer \
        rich \
        "pydantic>=2.6"

# ---------- Project code -------------------------------------
WORKDIR /workspace/AI_CTA_Stroke
COPY . .

RUN pip install --no-cache-dir -e ./cta_common && \
    pip install --no-cache-dir -e ./subprojects/cta-dental-opportunistic-screening && \
    pip install --no-cache-dir -e .

# ---------- Convenience env vars -----------------------------
ENV PYTHONPATH="/workspace/AI_CTA_Stroke:${PYTHONPATH}"
ENV TOTALSEG_WEIGHTS_PATH="/workspace/weights/totalseg"
ENV NV_SEGMENT_CT_CACHE="/workspace/weights/nv_segment_ct"
ENV HF_HOME="/workspace/weights/huggingface"

RUN mkdir -p \
        /workspace/weights/totalseg \
        /workspace/weights/nv_segment_ct \
        /workspace/weights/huggingface

CMD ["bash"]
