@echo off
title 3D Reconstruction Studio — Setup
echo ============================================
echo   3D Reconstruction Studio — First-Time Setup
echo ============================================
echo.

:: Check for Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Show Python version
python --version
echo.

:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
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

:: Install PyTorch with CUDA
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

:: Install main requirements
echo Installing project dependencies...
pip install -r requirements.txt
echo.

:: Install desktop app dependencies (ImGui + OpenGL)
echo Installing desktop GUI dependencies...
pip install imgui[glfw] PyOpenGL glfw xatlas
echo.

:: Install extra dependencies (MV-DUSt3R+, depth estimation, etc.)
echo Installing extra dependencies...
pip install torchmetrics opencv-python open3d
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

:: Try to build CUDA RoPE kernels (optional, speeds up inference)
echo Building optional CUDA kernels...
if exist "mast3r\dust3r\croco\models\curope\setup.py" (
    pushd mast3r\dust3r\croco\models\curope
    python setup.py build_ext --inplace 2>nul
    if errorlevel 1 (
        echo   [SKIP] CUDA RoPE kernels — not critical, inference still works.
    ) else (
        echo   [OK] CUDA RoPE kernels built.
    )
    popd
)
echo.

:: Create desktop shortcut
echo Creating desktop shortcut...
python -c "import os, sys; exec(open(os.path.join(os.path.dirname(sys.argv[0]) if sys.argv[0] else '.', 'create_shortcut.py')).read())" 2>nul
if errorlevel 1 (
    :: Fallback: create a simple launcher .bat on the desktop
    echo @echo off > "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
    echo cd /d "%~dp0" >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
    echo call venv\Scripts\activate.bat >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
    echo python desktop_app.py >> "%USERPROFILE%\Desktop\3D Reconstruction Studio.bat"
    echo Created launcher on Desktop.
)
echo.

echo ============================================
echo   Setup complete!
echo.
echo   You can now run the app by:
echo     1. Double-click "3D Reconstruction Studio" on your Desktop
echo     2. Or run: launch_desktop.bat
echo ============================================
pause
