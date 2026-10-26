@echo off
setlocal
set "DEMO_PY=%~1"
if defined DEMO_PY goto run
if exist "%~dp0.venv\Scripts\python.exe" (
  set "DEMO_PY=%~dp0.venv\Scripts\python.exe"
  goto run
)
set "DEMO_PY=python"
:run
set "PYTHONDONTWRITEBYTECODE=1"
"%DEMO_PY%" -u "%~dp0run_demo.py"
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" echo Demo failed. See README.md for dependency installation.
if "%DEMO_EXIT%"=="0" echo Done. Outputs are in demo\figures.
pause
exit /b %DEMO_EXIT%
