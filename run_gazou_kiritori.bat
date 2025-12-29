@echo off
setlocal
pushd "%~dp0"

REM ============================================================
REM Hugging Face: keep token + model cache inside this app folder
REM (so it won't affect other Hugging Face tools/apps)
REM ============================================================
set "HF_HOME=%~dp0hf_home"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

REM ★ venv を必ず有効化
call "%~dp0venv\Scripts\activate.bat"

REM ★ venv の python でアプリを起動
python "%~dp0gazou_kiritori.py"

popd
endlocal
pause