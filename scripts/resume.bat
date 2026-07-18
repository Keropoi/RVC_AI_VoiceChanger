@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
    echo Usage: scripts\resume.bat RUN_ID
    exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found. Run scripts\setup_venv.bat first.
    exit /b 1
)

".venv\Scripts\python.exe" -m rvc_auto_trainer resume --run-id "%~1"
endlocal
