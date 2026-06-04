@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Virtual environment not found: "%PYTHON%"
  exit /b 1
)

cd /d "%ROOT%"
"%PYTHON%" app.py
