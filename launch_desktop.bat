@echo off
title 3D Reconstruction Studio
cd /d "%~dp0"

:: Use venv if it exists, otherwise use system Python
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python desktop_app.py %*
pause
