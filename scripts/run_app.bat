@echo off
REM Launch the AI CTA Stroke Streamlit dashboard on Windows.
REM Usage: scripts\run_app.bat [port]
SET PORT=%1
IF "%PORT%"=="" SET PORT=8501

SET REPO_ROOT=%~dp0..
SET CONDA_BIN=%USERPROFILE%\miniconda3\envs\laa-pipeline\Scripts

IF NOT EXIST "%CONDA_BIN%\streamlit.exe" (
    echo ERROR: laa-pipeline env not found or streamlit not installed.
    echo Run: scripts\setup_env.bat
    exit /b 1
)

echo Starting UI at http://localhost:%PORT%

REM Open default browser after 3 seconds
start /b cmd /c "timeout /t 3 >nul && start http://localhost:%PORT%"

"%CONDA_BIN%\streamlit.exe" run "%REPO_ROOT%\app\streamlit_app.py" ^
    --server.port %PORT% ^
    --server.headless true ^
    --browser.gatherUsageStats false
