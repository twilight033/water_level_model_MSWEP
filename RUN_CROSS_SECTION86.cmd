@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_cross_section86.ps1" %*
set "RUN_RESULT=%ERRORLEVEL%"
echo.
if not "%RUN_RESULT%"=="0" echo 运行失败，请查看上面的错误。
pause
exit /b %RUN_RESULT%
