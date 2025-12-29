@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 932 >nul

rem ============================================================
rem RMBG-1.4 installer (non-commercial) - with license consent
rem Cache under .\hf_home\hub
rem ============================================================

pushd "%~dp0"

set "APP_DIR=%CD%"
set "HF_HOME=%APP_DIR%\hf_home"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%HF_HUB_CACHE%" (
  mkdir "%HF_HUB_CACHE%" >nul 2>&1
)

set "PY=%APP_DIR%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo ============================================================
echo 【重要】RMBG-1.4 のライセンスについて
echo ------------------------------------------------------------
echo - 本モデルは「非商用」用途向けのライセンスで提供されています。
echo - 「商用利用」する場合は、BRIA と別途 商用ライセンス契約が必要です。
echo.
echo 参考:
echo   https://huggingface.co/briaai/RMBG-1.4
echo ------------------------------------------------------------
echo 上記を理解し、ライセンス条件に同意した場合のみ続行してください。
echo ============================================================
echo.

set "ANS="
set /p "ANS=同意してインストールを続行しますか？ [y/N]: "
if /I not "!ANS!"=="y" (
  echo.
  echo [中止] 同意が確認できなかったため中止しました。
  echo.
  popd
  endlocal
  pause
  exit /b 1
)

echo.
echo Enter を押すとインストールを開始します（中止は Ctrl+C）:
set /p "DUMMY="

rem ---- ensure huggingface_hub (<1.0) ----
"%PY%" -c "import sys, huggingface_hub as h; v=str(getattr(h,'__version__','0')); sys.exit(0 if int(v.split('.')[0])<1 else 2)" >nul 2>&1
set "HFCHECK=%ERRORLEVEL%"

if not "%HFCHECK%"=="0" (
  echo.
  echo [INFO] huggingface_hub が見つからないか、バージョンが不適切なためインストールします: huggingface_hub^<1.0
  "%PY%" -m pip install "huggingface_hub<1.0"
  if errorlevel 1 goto :err
)

echo.
echo ============================================================
echo RMBG-1.4 をダウンロードします
echo   リポジトリ: briaai/RMBG-1.4
echo   キャッシュ  : "%HF_HUB_CACHE%"
echo ============================================================
echo.

set "TMPPY=%TEMP%\dl_rmbg14_%RANDOM%.py"

> "%TMPPY%"  echo import os,sys,traceback,shutil
>>"%TMPPY%" echo from huggingface_hub import snapshot_download
>>"%TMPPY%" echo import huggingface_hub.file_download as fd
>>"%TMPPY%" echo # Patch: symlink fallback to copy on WinError 1314 (no symlink privilege)
>>"%TMPPY%" echo _orig = fd._create_symlink
>>"%TMPPY%" echo def _patched(src, dst, new_blob=False):
>>"%TMPPY%" echo ^    try:
>>"%TMPPY%" echo ^        return _orig(src, dst, new_blob=new_blob)
>>"%TMPPY%" echo ^    except OSError as e:
>>"%TMPPY%" echo ^        if getattr(e, 'winerror', None) == 1314:
>>"%TMPPY%" echo ^            try:
>>"%TMPPY%" echo ^                os.makedirs(os.path.dirname(dst), exist_ok=True)
>>"%TMPPY%" echo ^            except Exception:
>>"%TMPPY%" echo ^                pass
>>"%TMPPY%" echo ^            shutil.copy2(src, dst)
>>"%TMPPY%" echo ^            return
>>"%TMPPY%" echo ^        raise
>>"%TMPPY%" echo fd._create_symlink = _patched
>>"%TMPPY%" echo repo = 'briaai/RMBG-1.4'
>>"%TMPPY%" echo print('[PY] start:', repo)
>>"%TMPPY%" echo try:
>>"%TMPPY%" echo ^    p = snapshot_download(repo_id=repo)
>>"%TMPPY%" echo ^    print('[PY] done:', p)
>>"%TMPPY%" echo except Exception as e:
>>"%TMPPY%" echo ^    print('[PY] ERROR:', type(e).__name__, e)
>>"%TMPPY%" echo ^    traceback.print_exc()
>>"%TMPPY%" echo ^    sys.exit(1)

"%PY%" "%TMPPY%"
set "RC=%ERRORLEVEL%"
del "%TMPPY%" >nul 2>&1

if not "%RC%"=="0" goto :err

echo.
echo [OK] RMBG-1.4 のインストールが完了しました。
echo 保存先:
echo   "%HF_HUB_CACHE%\models--briaai--RMBG-1.4"
echo.
popd
endlocal
pause
exit /b 0

:err
echo.
echo [ERROR] RMBG-1.4 のインストールに失敗しました。
echo - ネット接続がある状態で再実行してください
echo - 企業ネットワーク等で遮断される場合はプロキシ設定が必要なことがあります
echo - WinError 1314 が出る環境では「開発者モードON」または「管理者として実行」で改善することがあります
echo.
popd
endlocal
pause
exit /b 1
