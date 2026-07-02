@echo off
REM ============================================================
REM setup_env.bat — create the laa-pipeline conda environment
REM Windows 10/11 + CUDA 12.8 (NVIDIA GPU)
REM
REM Usage (Anaconda Prompt or PowerShell with conda):
REM   scripts\setup_env.bat
REM   scripts\setup_env.bat my-env
REM ============================================================

SET ENV_NAME=%1
IF "%ENV_NAME%"=="" SET ENV_NAME=laa-pipeline

SET REPO_ROOT=%~dp0..

echo ========================================
echo   Platform:   Windows x86_64
echo   Env name:   %ENV_NAME%
echo   Torch:      cu128 (CUDA 12.8)
echo ========================================

REM 1. Base conda env
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [setup] Env '%ENV_NAME%' already exists.
) ELSE (
    echo [setup] Creating conda env '%ENV_NAME%' ...
    conda create -y -n %ENV_NAME% -c conda-forge ^
        python=3.10 ^
        nibabel pydicom ^
        numpy scipy networkx pandas tqdm ^
        scikit-image scikit-learn matplotlib pyyaml ^
        trimesh rtree
)

REM 2. PyTorch cu128
echo [setup] Installing PyTorch CUDA 12.8 ...
conda run -n %ENV_NAME% pip install ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    torch torchvision

REM 3. Medical imaging stack
echo [setup] Installing MONAI, VISTA3D, TotalSegmentator ...
conda run -n %ENV_NAME% pip install ^
    "monai>=1.6.0" ^
    "transformers>=4.40,<5" ^
    huggingface_hub einops safetensors tokenizers ^
    TotalSegmentator SimpleITK

REM 4. Dental + UI
echo [setup] Installing dental pipeline and Streamlit UI ...
conda run -n %ENV_NAME% pip install ^
    typer rich "pydantic>=2.6" ^
    pyradiomics ^
    "streamlit>=1.35" plotly

REM 5. Local packages
echo [setup] Installing local packages ...
conda run -n %ENV_NAME% pip install -e "%REPO_ROOT%\cta_common"
conda run -n %ENV_NAME% pip install -e "%REPO_ROOT%\subprojects\cta-dental-opportunistic-screening"
conda run -n %ENV_NAME% pip install -e "%REPO_ROOT%"

REM 6. Verify
echo.
echo [setup] Verifying ...
conda run -n %ENV_NAME% python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

echo.
echo === Setup complete ===
echo Activate:   conda activate %ENV_NAME%
echo Run UI:     scripts\run_app.bat
echo.
