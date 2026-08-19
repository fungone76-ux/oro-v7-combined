@echo off
setlocal
title FREEZE TOP40 5-3-2 TOP15 FINAL
cd /d D:\oro_top40_shortmem_OLD

echo ============================================================================
echo FREEZE TOP40 5/3/2 TOP15 - TRAIN 01 MAR -> 13 AUG 2026
echo ============================================================================
python run_532_freeze_final_mar13aug_2026.py
if errorlevel 1 (
    echo.
    echo FREEZE FAILED
    pause
    exit /b 1
)

echo.
echo FREEZE COMPLETED
echo Artifact folder: D:\ORO_532_MAR13AUG_2026\frozen_demo_532_top15
pause
endlocal
