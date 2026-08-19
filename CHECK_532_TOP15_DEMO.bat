@echo off
setlocal
title CHECK TOP40 5-3-2 TOP15 DEMO
cd /d D:\oro_top40_shortmem_OLD

set "MANIFEST=D:\ORO_532_MAR13AUG_2026\frozen_demo_532_top15\FROZEN_532_TOP15_MANIFEST.json"
set "VENV_PY=D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL\.venv\Scripts\python.exe"

if not exist "%MANIFEST%" (
    echo FROZEN MODEL NOT FOUND. Run FREEZE_532_TOP15_FINAL.bat first.
    pause
    exit /b 1
)

set "PYTHONPATH=D:\oro_top40_shortmem_OLD;D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL;%PYTHONPATH%"

if exist "%VENV_PY%" (
    "%VENV_PY%" TOP40_532_TOP15_DEMO.py --check-only
) else (
    python TOP40_532_TOP15_DEMO.py --check-only
)

echo.
echo Check finished. Exit code=%ERRORLEVEL%
pause
endlocal
