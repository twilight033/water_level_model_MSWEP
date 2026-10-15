@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 顺序运行物理归一化和观测归一化，各自完成两个阶段。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_cross_section86.ps1" -Config "%~dp0configs\cross_section86_strong_physical.json" -OutDir "%~dp0results\cross_section86_strong_physical_snapshot_v2" %*
set "RUN_RESULT=%ERRORLEVEL%"
if not "%RUN_RESULT%"=="0" goto finish
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_cross_section86.ps1" -Config "%~dp0configs\cross_section86_strong_observed.json" -OutDir "%~dp0results\cross_section86_strong_observed_snapshot_v2" %*
set "RUN_RESULT=%ERRORLEVEL%"
:finish
echo.
if not "%RUN_RESULT%"=="0" echo 运行失败，请查看上面的错误。已有同配置结果可续跑。
pause
exit /b %RUN_RESULT%
