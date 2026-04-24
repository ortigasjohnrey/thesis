@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
    echo Virtual environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate
python app.py
pause
