@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0"

REM ============================================================
REM Hugging Face: keep token + model cache inside this app folder
REM (so it won't affect other Hugging Face tools/apps)
REM ============================================================
set "HF_HOME=%~dp0hf_home"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

set "PYTHON=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [error] venv Python not found:
    echo !PYTHON!
    popd
    endlocal
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0gazou_kiritori.py"

popd
endlocal
pause