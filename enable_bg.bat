@echo off
chcp 932 >nul
setlocal EnableExtensions
pushd "%~dp0"

echo ============================================================
echo 背景切り抜き機能を有効化します（CPU / GPU 選択式）
echo ============================================================
echo.

REM --- 必要ファイルチェック ---
if not exist "%~dp0requirements_bg.txt" (
  echo [ERROR] requirements_bg.txt が見つかりません。
  echo   "%~dp0requirements_bg.txt"
  goto :err
)

REM --- venv check ---
if not exist "%~dp0venv\Scripts\activate.bat" (
  echo [ERROR] venv が見つかりません。
  echo 先に install_app.bat を実行してください。
  goto :err
)

REM --- Hugging Face 保存先をアプリ配下に固定 ---
set "HF_HOME=%~dp0hf_home"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 goto :err

python -m pip install -U pip
if errorlevel 1 goto :err

echo.
echo [選択] 実行モードを選んでください:
echo   1) CPU
echo   2) GPU - cu126
echo   3) GPU - cu128
echo   4) GPU - cu118
echo.
set /p MODE=番号を入力 (1-4): 
if "%MODE%"=="" set "MODE=1"

set "TORCH_INDEX="
set "MODE_NAME="

if "%MODE%"=="1" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
  set "MODE_NAME=CPU"
) else if "%MODE%"=="2" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
  set "MODE_NAME=GPU (cu126)"
) else if "%MODE%"=="3" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
  set "MODE_NAME=GPU (cu128)"
) else if "%MODE%"=="4" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
  set "MODE_NAME=GPU (cu118)"
) else (
  echo [ERROR] 不正な入力です。1?4 を入力してください。
  goto :err
)

echo.
echo ============================================================
echo モード: %MODE_NAME%
echo PyTorch index-url: %TORCH_INDEX%
echo ============================================================
echo.

REM --- GPU選択時の目安：nvidia-smi があるか ---
if not "%MODE%"=="1" (
  where nvidia-smi >nul 2>&1
  if errorlevel 1 (
    echo [WARN] nvidia-smi が見つかりません。
    echo        NVIDIA ドライバ未導入、または PATH に無い可能性があります。
    echo        （インストールは続行します）
    echo.
  ) else (
    echo [INFO] nvidia-smi:
    nvidia-smi
    echo.
  )
)

echo [STEP] 既存の torch / torchvision / torchaudio を削除します（入っていれば）...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo [STEP] PyTorch をインストールします（時間がかかります）...
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
  echo.
  echo [ERROR] PyTorch のインストールに失敗しました。
  echo - ネット接続を確認してください
  echo - だめなら別のGPUタグ（cu126/128/118）を試してください
  goto :err
)

echo.
echo [STEP] 背景切り抜き用ライブラリをインストールします...
pip install -r "%~dp0requirements_bg.txt"
if errorlevel 1 goto :err

echo.
echo [STEP] 動作確認...
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('device:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
if errorlevel 1 goto :err

echo.
echo [OK] 背景切り抜き機能の有効化が完了しました。
echo.
echo ▼次（モデルを入れる）
echo   - RMBG-1.4: install_rmbg_1_4.bat
echo   - RMBG-2.0: 先に hf_login_visible_sjis.bat → install_rmbg_2_0.bat
echo.
goto :end

:err
echo.
echo [ERROR] 背景切り抜き機能の有効化に失敗しました。
popd
endlocal
pause
exit /b 1

:end
popd
endlocal
pause
exit /b 0
