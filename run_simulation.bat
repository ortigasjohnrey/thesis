@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate" (
    echo Virtual environment not found. Run setup_env.bat first.
    exit /b 1
)
call .venv\Scripts\activate
python scripts\run_simulation.py
endlocal
