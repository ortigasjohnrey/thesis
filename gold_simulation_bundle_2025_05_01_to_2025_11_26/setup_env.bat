@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Environment ready.
echo Activate later with: .venv\Scripts\activate
endlocal
