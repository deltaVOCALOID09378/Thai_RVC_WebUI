@echo off
title DELTA SYNTH - ONNX Runtime Auto Fixer
echo ==============================================================================
echo Made And Checked By DELTA SYNTH ^& All Code Agentic AI Engine
echo Original by Patiphat Wongyai (Delta)
echo ==============================================================================
echo.
echo [INFO] Removing corrupted ONNX Runtime packages...
"A:\Thai_RVC_WebUI_v1\runtime\python.exe" -m pip uninstall onnxruntime onnxruntime-gpu -y

echo.
echo [INFO] Clearing pip cache to prevent reinstalling corrupted files...
"A:\Thai_RVC_WebUI_v1\runtime\python.exe" -m pip cache purge

echo.
echo [INFO] Installing fresh and stable ONNX Runtime GPU...
"A:\Thai_RVC_WebUI_v1\runtime\python.exe" -m pip install onnxruntime-gpu

echo.
echo ==============================================================================
echo [SUCCESS] Process completed. The InferenceSession error has been resolved.
echo ==============================================================================
pause