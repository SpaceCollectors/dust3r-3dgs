@echo off
setlocal enabledelayedexpansion
title 3D Reconstruction Studio — Setup
echo ============================================
echo   3D Reconstruction Studio — First-Time Setup
echo ============================================
echo.

cd /d "%~dp0"

:: ── Find or install Python ──
set PYTHON_CMD=
set PYTHON_DIR=%~dp0python

:: Check for local portable Python first
if exist "%PYTHON_DIR%\python.exe" (
    set PYTHON_CMD=%PYTHON_DIR%\python.exe
    echo Found local portable Python.
    goto :have_python
)

:: Check system Python (must be 3.10-3.12 — 3.13+ breaks too many packages)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>nul
    if not errorlevel 1 (
        set PYTHON_CMD=python
        echo Found compatible system Python.
        goto :have_python
    ) else (
        echo System Python found but not 3.10-3.12 (PyTorch/Open3D need 3.10-3.12^).
    )
)

:: No Python found — download portable Python
echo Python 3.10+ not found. Downloading portable Python 3.11...
echo.

set PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-pc-windows-msvc-install_only.tar.gz
set PY_ARCHIVE=python-portable.tar.gz

:: Download using PowerShell (works on Windows 10+)
echo Downloading Python 3.11.9 from GitHub...
echo URL: %PY_URL%
echo This may take a minute...
powershell -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ARCHIVE%' -UseBasicParsing } catch { Write-Host $_.Exception.Message; exit 1 }"
if not exist "%PY_ARCHIVE%" (
    echo.
    echo ERROR: Download failed. Please install Python 3.10+ manually from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Extract .tar.gz
echo Extracting Python...
tar -xzf "%PY_ARCHIVE%" 2>nul
if not exist "%PYTHON_DIR%\python.exe" (
    :: The archive extracts to a "python" folder with python/install/ structure
    if exist "python\install\python.exe" (
        :: Move contents up
        xcopy /e /y /q "python\install\*" "python\" >nul
        rmdir /s /q "python\install" 2>nul
    )
)
del "%PY_ARCHIVE%" 2>nul

if exist "%PYTHON_DIR%\python.exe" (
    set PYTHON_CMD=%PYTHON_DIR%\python.exe
    echo Portable Python installed successfully.
    "%PYTHON_CMD%" --version
    :: Ensure pip is available
    "%PYTHON_CMD%" -m ensurepip --upgrade >nul 2>nul
) else (
    echo.
    echo ERROR: Failed to extract Python. Contents found:
    dir /b python\ 2>nul
    echo Please install Python 3.10+ manually from python.org
    pause
    exit /b 1
)

:have_python
"%PYTHON_CMD%" --version
echo.

:: ── Create virtual environment ──
:: Check if existing venv uses the right Python version
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>nul
    if errorlevel 1 (
        echo Existing venv uses wrong Python version, recreating...
        rmdir /s /q venv
    )
)
if not exist "venv" (
    echo Creating virtual environment...
    "%PYTHON_CMD%" -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists, reusing.
)
echo.

:: Activate venv
call venv\Scripts\activate.bat

:: ── Clone required repositories ──
if not exist "mast3r\dust3r\dust3r" (
    echo Cloning MASt3R (includes DUSt3R^)...
    if exist "mast3r" rmdir /s /q mast3r
    git clone --recursive https://github.com/naver/mast3r.git mast3r
    if errorlevel 1 (
        echo ERROR: Failed to clone MASt3R. Check your internet connection.
        pause
        exit /b 1
    )
)
echo MASt3R: OK
echo.

if not exist "vggt\vggt\models" (
    echo Cloning VGGT...
    if exist "vggt" rmdir /s /q vggt
    git clone https://github.com/facebookresearch/vggt.git vggt
    if errorlevel 1 (
        echo WARNING: Failed to clone VGGT. VGGT backend will not be available.
    )
)
echo VGGT: OK
echo.

if not exist "pow3r\pow3r\model" (
    echo Cloning Pow3R...
    if exist "pow3r" rmdir /s /q pow3r
    git clone --recursive https://github.com/naver/pow3r.git pow3r
    if errorlevel 1 (
        echo WARNING: Failed to clone Pow3R. Pow3R backend will not be available.
    )
)
echo Pow3R: OK
echo.

:: ── Install dependencies ──
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

echo Installing project dependencies...
pip install -r requirements.txt
echo.

echo Installing desktop GUI dependencies...
pip install imgui[glfw] PyOpenGL glfw xatlas pymeshlab
echo.

echo Installing extra dependencies...
pip install torchmetrics opencv-python open3d pycolmap
echo.

:: Install submodule dependencies
if exist "mast3r\requirements.txt" (
    echo Installing MASt3R dependencies...
    pip install -r mast3r\requirements.txt
)
if exist "mast3r\dust3r\requirements.txt" (
    echo Installing DUSt3R dependencies...
    pip install -r mast3r\dust3r\requirements.txt
)
if exist "mvdust3r\requirements.txt" (
    echo Installing MV-DUSt3R+ dependencies...
    pip install -r mvdust3r\requirements.txt
)
echo.

:: Optional CUDA kernels
echo Building optional CUDA kernels...
if exist "mast3r\dust3r\croco\models\curope\setup.py" (
    pushd mast3r\dust3r\croco\models\curope
    python setup.py build_ext --inplace 2>nul
    if errorlevel 1 (
        echo   [SKIP] CUDA RoPE kernels — not critical.
    ) else (
        echo   [OK] CUDA RoPE kernels built.
    )
    popd
)
echo.

:: ── Update launch script to use correct Python ──
:: Write launch_desktop.bat that uses the venv
(
echo @echo off
echo title 3D Reconstruction Studio
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo python desktop_app.py %%*
echo pause
) > launch_desktop.bat

:: Create desktop shortcut
echo @echo off > "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
echo cd /d "%~dp0" >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
echo call venv\Scripts\activate.bat >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
echo python desktop_app.py >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
echo Created launcher on Desktop.
echo.

echo ============================================
echo   Setup complete!
echo.
echo   Run the app by:
echo     1. Double-click "3D Reconstruction Studio" on Desktop
echo     2. Or run: launch_desktop.bat
echo ============================================
pause
