@echo off
title MASt3R → COLMAP → 3DGS Pipeline

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting MASt3R app...
echo Open http://127.0.0.1:7860 in your browser
echo.

python app.py %*

pause
