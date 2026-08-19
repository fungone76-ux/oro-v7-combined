@echo off
setlocal
title TOP40 5-3-2 TOP15 DEMO
cd /d D:\oro_top40_shortmem_OLD

set "MANIFEST=D:\ORO_532_MAR13AUG_2026\frozen_demo_532_top15\FROZEN_532_TOP15_MANIFEST.json"
set "VENV_PY=D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL\.venv\Scripts\python.exe"

if not exist "%MANIFEST%" (
    echo ============================================================================
    echo FROZEN MODEL NOT FOUND
    echo %MANIFEST%
    echo.
    echo Run FREEZE_532_TOP15_FINAL.bat first.
    echo ============================================================================
    pause
    exit /b 1
)

set "PYTHONPATH=D:\oro_top40_shortmem_OLD;D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL;%PYTHONPATH%"

echo ============================================================================
echo START TOP40 5/3/2 TOP15 - DEMO ORDERS ENABLED
echo Training frozen through 13-Aug-2026. Aug-14 onward is not used for training.
echo ============================================================================

if exist "%VENV_PY%" (
    "%VENV_PY%" TOP40_532_TOP15_DEMO.py --enable-demo-orders
) else (
    python TOP40_532_TOP15_DEMO.py --enable-demo-orders
)

echo.
echo Runtime stopped. Exit code=%ERRORLEVEL%
pause
endlocal
