@echo off
setlocal
cd /d "%~dp0.."

if not exist ".speaker_venv\Scripts\python.exe" (
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo pyannote.audio 4.x requires Python 3.10 or newer.
    where winget.exe >nul 2>nul
    if errorlevel 1 (
      echo Python 3.11 was not found and winget is unavailable.
      echo Install Python 3.11, then rerun this script.
      exit /b 1
    )
    choice /C YN /M "Install Python 3.11 for the independent speaker environment"
    if errorlevel 2 exit /b 2
    winget install --id Python.Python.3.11 --exact --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 exit /b 1
  )
  py -3.11 -m venv .speaker_venv
  if errorlevel 1 exit /b 1
)

call ".speaker_venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Existing .speaker_venv uses Python older than 3.10. Remove only that generated
  echo environment and rerun this script.
  exit /b 1
)

call ".speaker_venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call ".speaker_venv\Scripts\python.exe" -m pip install -e ".[speaker]"
if errorlevel 1 exit /b 1

call ".speaker_venv\Scripts\python.exe" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo WARNING: PyTorch in .speaker_venv cannot currently use CUDA.
  echo Install the Windows CUDA wheel recommended by https://pytorch.org/get-started/locally/
)

echo.
echo Speaker environment ready.
echo Accept the model conditions at:
echo https://huggingface.co/pyannote/speaker-diarization-community-1
echo Then set HF_TOKEN for the current PowerShell session and run sort-speakers.
endlocal
