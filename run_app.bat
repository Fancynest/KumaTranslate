@echo off
title Manga Translator AI Web UI
echo ============================================================
echo   Manga Translator AI (Local Edition)
echo   GPU: NVIDIA GeForce RTX 2050 (CUDA)
echo ============================================================
echo.
echo Membuka browser di: http://127.0.0.1:7860 ...
set PYTHONIOENCODING=utf-8
start http://127.0.0.1:7860
.\.venv\Scripts\python.exe app.py
pause
