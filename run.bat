@echo off
setlocal

set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    exit /b 1
)

set PYTHONPATH=src
%VENV_DIR%\Scripts\python.exe src\main.py %*
