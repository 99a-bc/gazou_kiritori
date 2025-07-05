@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python gazou_kiritori.py
pause