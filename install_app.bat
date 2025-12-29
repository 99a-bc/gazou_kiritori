@echo off
setlocal
pushd "%~dp0"

echo ============================================================
echo gazou_kiritori セットアップ（アプリ本体）
echo ============================================================
echo.

if not exist "%~dp0requirements_app.txt" (
  echo [ERROR] requirements_app.txt が見つかりません。
  echo   "%~dp0requirements_app.txt"
  goto :err
)

REM venv 作成（既にあればそのまま）
if not exist "%~dp0venv\Scripts\activate.bat" (
  echo [INFO] venv を作成します...
  python -m venv venv
  if errorlevel 1 goto :err
) else (
  echo [INFO] venv は既に存在します: venv\
)

call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 goto :err

python -m pip install -U pip setuptools wheel
if errorlevel 1 goto :err

echo.
echo [INFO] アプリ本体の依存関係をインストールします...
pip install -r "%~dp0requirements_app.txt"
if errorlevel 1 goto :err

echo.
echo [OK] アプリ本体のセットアップが完了しました。
echo.
echo ▼背景切り抜き機能を使う場合は、次を実行してください
echo   - CPUで使う: enable_bg_cpu.bat
echo   - GPUで使う: enable_bg_gpu.bat
echo.
goto :end

:err
echo.
echo [ERROR] セットアップに失敗しました。
echo - Python が入っているか
echo - ネット接続
echo - 権限（会社PC等でpipがブロックされる場合）
echo を確認してください。
echo.
popd
endlocal
pause
exit /b 1

:end
popd
endlocal
pause
exit /b 0
