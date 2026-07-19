@echo off
setlocal
cd /d "%~dp0.."

if not exist ".speaker_venv\Scripts\python.exe" (
  py -3.11 -m venv .speaker_venv
  if errorlevel 1 exit /b 1
)

call ".speaker_venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call ".speaker_venv\Scripts\python.exe" -m pip install -e ".[speaker]"
if errorlevel 1 exit /b 1

echo.
echo Speaker environment ready.
echo Accept the model conditions at:
echo https://huggingface.co/pyannote/speaker-diarization-community-1
echo Then set HF_TOKEN for the current PowerShell session and run sort-speakers.
endlocal
