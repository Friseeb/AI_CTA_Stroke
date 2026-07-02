# AI_CTA_Stroke — Developer Guide

## Quick Start (any platform)

```bash
# 1. Clone and enter the repo
git clone <repo-url> AI_CTA_Stroke && cd AI_CTA_Stroke

# 2. Install Miniforge (skip if conda already available)
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh | bash

# 3. Create the environment (auto-detects OS / GPU)
bash scripts/setup_env.sh

# 4. Launch the UI
bash scripts/run_app.sh
# → open http://localhost:8501
```

Windows:

```bat
scripts\setup_env.bat
scripts\run_app.bat
```

---

## Environment Files

All platform environments live in `envs/`. **One environment covers all pipelines** (LAA + dental + segmentation).

| File | Target |
| ---- | ------ |
| `envs/linux_cuda_arm64.yml` | DGX Spark — GB10 Grace Blackwell, aarch64, CUDA 13.0 |
| `envs/linux_cuda_x86.yml` | Linux x86_64 — Ampere/Ada/Hopper, CUDA 12.8 |
| `envs/macos.yml` | macOS — Apple Silicon (MPS) or Intel (CPU) |
| `envs/windows.yml` | Windows 10/11 — NVIDIA GPU, CUDA 12.8 |
| `envs/cpu.yml` | Any platform, CPU-only fallback |

`scripts/setup_env.sh` auto-detects which file to use. You rarely need to pick manually.

### Why not just `conda env create -f envs/...`?

On aarch64 Linux, conda resolves torch to a CPU-only wheel. `setup_env.sh` installs torch from
the correct index URL (`cu130` on DGX Spark, `cu128` on x86 Linux/Windows) **after** conda creates
the base env. The YAML files document the target but the script is the authoritative installer.

### Deprecated environment files (repo root)

The old per-tool YAMLs at the repo root are kept for historical reference but **should not be used**:
`environment.cardiac_ct_explorer.yml`, `environment.nv_segment_ct.yml` — both install CPU-only torch.

---

## Key Package Constraints

| Package | Constraint | Why |
| ------- | ---------- | --- |
| `torch` | `cu130` index on aarch64 | Only source for GB10 Blackwell (sm_100) CUDA support |
| `monai` | `>=1.6.0` | 1.5.0 pins `torch<2.7`, conflicts with cu130 builds |
| `transformers` | `>=4.40,<5` | 5.x renames model attributes — breaks VISTA3D loading |

---

## Running Scripts (without activating the env)

```bash
# Prepend conda bin so cta-dental, TotalSegmentator, dcm2niix are findable
PATH="$HOME/miniforge3/envs/laa-pipeline/bin:$PATH" cta-dental run ...

# Or call Python directly
$HOME/miniforge3/envs/laa-pipeline/bin/python scripts/run_prior_fusion.py ...
```

---

## Streamlit Dashboard

```bash
bash scripts/run_app.sh          # Linux / macOS
scripts\run_app.bat              # Windows
```

Opens at `http://localhost:8501`. Provides:

- **Single Patient** — run Dental / LAA / Both pipelines with live log output
- **Batch** — folder + glob, progress bar, resumable
- **Results Viewer** — metrics, periapical findings, axial/coronal/sagittal slice preview

---

## Data (this installation)

- Raw DICOMs: `/media/friseb/LAAforLAAs/<case_number>/Export_*/...`
- DICOM files have no `.dcm` extension (DICOMDIR-style export)
- Target series per case: description `"CTA * 0.5 CE"` (~1455 slices, highest count)

### DICOM → NIfTI (dcm2niix is in the env)

```bash
dcm2niix -z y -f sub-<N>_acq-CTA_ct \
  -o <out>/raw \
  "/media/friseb/LAAforLAAs/<N>/Export_*/<study>/<CTA_series>"
```

---

## Docker (alternative)

```bash
bash docker/run.sh build          # build image
bash docker/run.sh                # interactive shell
LAA_DATA_DIR=/media/friseb/LAAforLAAs bash docker/run.sh
```

The Dockerfile mirrors `envs/linux_cuda_arm64.yml` exactly.
