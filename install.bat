@echo off
cd /d "%~dp0"
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo セットアップ完了！
pause