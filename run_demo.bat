@echo off
cd /d "%~dp0"

echo ======================================================================
echo             AFM-RL Interactive Matching Demo Launcher
echo ======================================================================

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if "%~1"=="" (
    echo Launching file picker dialog...
    "%PYTHON_EXE%" scripts\demo_pair.py --gui
) else (
    "%PYTHON_EXE%" scripts\demo_pair.py %*
)

echo.
pause
