@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_cross_section86.ps1" -Smoke -Device cpu %*
set "RUN_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RUN_RESULT%
