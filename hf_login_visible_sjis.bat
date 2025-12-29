@echo off
setlocal
pushd "%~dp0"

REM ============================================================
REM Hugging Face トークンログイン（表示入力版 / このアプリ専用）
REM - 保存先（トークン/キャッシュ）: .\hf_home\
REM - 注意: この版は入力したトークンが画面に見えます（肩越し覗き見に注意）
REM ============================================================

set "HF_HOME=%~dp0hf_home"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

REM venv check
if not exist "%~dp0venv\Scripts\activate.bat" (
  echo [エラー] venv が見つかりません:
  echo   "%~dp0venv\Scripts\activate.bat"
  echo.
  echo 先に venv を作成して、必要なライブラリをインストールしてください。
  popd
  endlocal
  pause
  exit /b 1
)

call "%~dp0venv\Scripts\activate.bat"

REM ============================================================
REM huggingface_hub が無い（または壊れている/1.0以上）ときだけインストール
REM ※ ついでに 1.0 未満に固定（huggingface_hub<1.0）
REM ============================================================
python -c "import sys,importlib,importlib.util; s=importlib.util.find_spec('huggingface_hub'); m=importlib.import_module('huggingface_hub') if s else None; v=getattr(m,'__version__','0') if m else '0'; major=int(str(v).split('.')[0] or 0) if m else 999; sys.exit(0 if (m and major < 1) else 1)" >nul 2>nul

if errorlevel 1 (
  echo.
  echo [INFO] huggingface_hub が見つからないか、バージョンが不適切なためインストールします: huggingface_hub^<1.0
  python -m pip install -U "huggingface_hub<1.0"
) else (
  echo.
  echo [INFO] huggingface_hub はインストール済みです（pip の実行をスキップ）
)

echo.
echo ============================================================
echo Hugging Face トークンログイン（このアプリ専用 / 表示入力版）
echo.
echo 1) 「hf_」から始まる Access Token を入力してください（画面に表示されます）
echo 2) Enter を押してください
echo ============================================================
echo.

set "HF_TOKEN="
set /p HF_TOKEN=Access Token (hf_...):

if "%HF_TOKEN%"=="" (
  echo.
  echo [エラー] トークンが空です。もう一度実行してください。
  popd
  endlocal
  pause
  exit /b 1
)

REM ★ トークンは「環境変数」経由で Python に渡す（コマンドラインに直書きしない）
python -c "import os; from huggingface_hub import login; t=os.environ.get('HF_TOKEN',''); assert t; login(token=t, add_to_git_credential=False); print('ログイン完了（トークンはアプリ配下のHF_HOMEに保存されました）')"

REM 使い終わったら環境変数を消す（このバッチ内だけ）
set "HF_TOKEN="

echo.
echo 完了しました。
echo トークン/キャッシュ保存先:
echo   "%HF_HOME%"
echo.


popd
endlocal
pause
