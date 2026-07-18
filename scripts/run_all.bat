@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found. Run scripts\setup_venv.bat first.
    exit /b 1
)

".venv\Scripts\python.exe" -m rvc_auto_trainer run --config config\example_windows_3090.yaml
if errorlevel 1 (
    echo Pipeline failed. Check the run logs directory.
    exit /b 1
)

echo Pipeline completed successfully.
endlocal
